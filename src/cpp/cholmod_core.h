// CHOLMOD's shared state: patterns, numeric factors, per-thread workspaces, and the
// operations the FFI handlers, the GHK simulator and the numpy core all use.
#ifndef SPARSAX_CHOLMOD_CORE_H_
#define SPARSAX_CHOLMOD_CORE_H_

#include <cholmod.h>

#include "common.h"

// ---------------------------------------------------------------------------
// CHOLMOD state
//
// Concurrency.  There is no global lock.  Each sparsity pattern's symbolic
// analysis (cholmod_analyze) and upper-triangular CSC structure are shared and
// read-only once built.  Numeric factors live in a per-pattern cache keyed on
// the COO values; each is factorized into its own copy of the symbolic factor
// and afterwards only read (by cholmod_solve2, the logdet and the selected
// inverse), so threads share them freely.  CHOLMOD's workspace (cholmod_common),
// the value buffer and the solve workspaces are per thread.  (A global mutex
// used to guard a single factor per pattern that every call refactored in
// place, so concurrent per-chain calls ran one at a time.)
// ---------------------------------------------------------------------------

inline void error_handler(int status, const char* file, int line,
                          const char* message) {
  // Errors are surfaced as XLA errors; keep CHOLMOD off stderr.
  (void)status;
  (void)file;
  (void)line;
  (void)message;
}

// Settings and counters shared by every thread.
inline std::atomic<int> g_chol_supernodal{CHOLMOD_AUTO};  // set_supernodal
// set_options(ordering=...): a CHOLMOD_* ordering, or one of these two.
constexpr int kCholOrderingAuto = -1;  // CHOLMOD's default: AMD, METIS if that is poor
constexpr int kCholOrderingBest = -2;  // try AMD, METIS and NESDIS, keep the best
inline std::atomic<int> g_chol_ordering{kCholOrderingAuto};
inline std::atomic<size_t> g_chol_cache_cap{32};          // set_num_cache_size
inline std::atomic<int64_t> g_num_factorizations{0};      // real refactors
inline std::atomic<int64_t> g_num_ldl_conversions{0};     // Lldl copies built
inline std::atomic<uint64_t> g_chol_generation{1};        // bumped by clear_cache

// Free a factor from whichever thread drops the last reference.  CHOLMOD needs
// *a* cholmod_common to free an object, not the one that allocated it.
inline void chol_free_factor(cholmod_factor** L) {
  if (!*L) return;
  cholmod_common c;
  cholmod_start(&c);
  cholmod_free_factor(L, &c);
  cholmod_finish(&c);
}

// A numeric LL' factor with refcounted lifetime, read-only once built, so the
// cache, outstanding tokens and in-flight calls can all hold it.  Lldl, the
// simplicial LDL' copy that cholmod_updown, cholmod_rowdel and the simplicial
// selected inverse need, is built on first use.
struct CholNumeric {
  cholmod_factor* L = nullptr;  // null: the factorization failed (on_failure="nan")
  double logdet = std::numeric_limits<double>::quiet_NaN();
  std::mutex ldl_mtx;  // guards the one-time construction of Lldl
  cholmod_factor* Lldl = nullptr;
  bool factored() const { return L != nullptr; }
  ~CholNumeric() {
    chol_free_factor(&L);
    chol_free_factor(&Lldl);
  }
};
using CholNumPtr = std::shared_ptr<CholNumeric>;

using CholSlot = NumericSlot<CholNumeric>;

// The shared, immutable-after-create pattern: the upper-triangular CSC
// structure and cholmod_analyze's symbolic factor, plus the value-keyed numeric
// cache, whose mutex is held only across a lookup or an insert.
struct CholPattern {
  int64_t n = 0;
  std::vector<int32_t> Ai, Aj;  // full COO copy, for exact hit verification
  std::vector<int64_t> pos;     // COO k -> upper-CSC slot, -1 if i > j
  std::vector<int32_t> Ap, Ar;  // upper-triangular CSC column pointers / rows
  int64_t nnz_csc = 0;
  cholmod_factor* Lsym = nullptr;  // symbolic analysis, copied per factorization
  double lnz = 0.0, flops = 0.0;   // fill and factorization flops of that analysis
  std::mutex cache_mtx;
  std::vector<CholSlot> cache;
  FactorsInFlight in_flight;
  size_t next = 0;  // round-robin eviction cursor
  ~CholPattern() { chol_free_factor(&Lsym); }
};

inline std::mutex g_chol_reg_mtx;  // guards the pattern registry only
inline PatternRegistry<CholPattern> g_chol_registry;

// Per-thread CHOLMOD workspace, value buffer and solve workspaces.  XLA's
// thread pool may run one chain's calls on different threads, so none of this
// belongs to a chain; it is scratch.
struct CholThreadLocal {
  cholmod_common c;
  bool started = false;
  std::vector<double> values;  // upper-CSC values being factorized
  cholmod_dense* X = nullptr;  // cholmod_solve2 workspaces, reused across solves
  cholmod_dense* Y = nullptr;
  cholmod_dense* E = nullptr;
  std::vector<double> scratch, tmpA, tmpB;
  // The last pattern this thread used: solver loops hit the same one every call.
  std::shared_ptr<CholPattern> last_pat;
  uint64_t last_gen = 0;

  cholmod_common* get() {
    if (!started) {
      cholmod_start(&c);
      c.print = 0;
      c.error_handler = error_handler;
      // Always produce an LL' factor: simplicial LDL' would silently accept
      // indefinite matrices, and an actual L is needed for MODE_L / MODE_LT
      // solves (e.g. sampling from N(0, A^{-1})).
      c.final_ll = 1;
      started = true;
    }
    c.supernodal = g_chol_supernodal.load(std::memory_order_relaxed);
    return &c;
  }
  ~CholThreadLocal() {
    if (!started) return;
    if (X) cholmod_free_dense(&X, &c);
    if (Y) cholmod_free_dense(&Y, &c);
    if (E) cholmod_free_dense(&E, &c);
    last_pat.reset();
    cholmod_finish(&c);
  }
};
extern thread_local CholThreadLocal t_chol;

using CholHeldFactor = HeldFactor<CholPattern, CholNumeric>;
inline TokenTable<CholHeldFactor> g_chol_tokens(kTokenCholmod, "sparsax");

std::shared_ptr<CholPattern> chol_get_pattern(const int32_t* Ai, const int32_t* Aj,
                                              int64_t nnz, int64_t n, std::string* err);
CholNumPtr chol_get_factor(const std::shared_ptr<CholPattern>& pat, const double* Ax,
                           int64_t nnz, std::string* err);
double factor_logdet(const cholmod_factor* L);
ffi::Error chol_solve_one(cholmod_factor* L, int mode, const double* bdata, int64_t n,
                          int64_t nrhs, double* xdata);
ffi::Error chol_ensure_ldl(CholNumeric* num);
ffi::Error chol_selinv(const CholPattern* pat, CholNumeric* num, int64_t nnz, double* z);

// What the Python module needs.
void chol_set_cache_size(size_t cap);
void chol_clear_cache();
size_t chol_pattern_count();
int64_t chol_factorizations();
bool chol_has_supernodal_selinv();

#endif  // SPARSAX_CHOLMOD_CORE_H_
