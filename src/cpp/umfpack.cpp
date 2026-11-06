// UMFPACK: multifrontal sparse LU, the second LU backend.
#include <umfpack.h>

#include "lu_backends.h"
#include "lu_handlers.h"

// ===========================================================================
// UMFPACK: multifrontal sparse LU for NON-symmetric matrices, the third FFI
// target beside CHOLMOD and KLU, with the same COO-in / pattern-keyed-cache
// shape as the KLU path above and the same API surface (umf_solve, umf_logdet,
// umf_factor / umf_solve_factor / umf_logdet_factor).
//
// Why a second LU backend. It is NOT to recover symbolic reuse -- KLU already
// has that, and UMFPACK has no klu_refactor equivalent to bind: SuiteSparse
// gives KLU a third stage (klu_analyze -> klu_factor -> klu_refactor) whose
// refactor reuses the symbolic analysis AND the pivot ordering, while UMFPACK
// has only umfpack_*_symbolic -> umfpack_*_numeric and re-pivots inside
// numeric() on every call, being multifrontal with partial pivoting. What
// UMFPACK buys is a better algorithm once the graph densifies: KLU is a scalar
// circuit-simulation code with no supernodal BLAS3 path, so its pivot-ordering
// reuse stops paying as fronts grow wide. Measured on distance-band graphs at
// n = 3,000, KLU wins ~4-5x at mean degree <= 8 and loses 3.5-8.6x above ~25.
//
// The crossover is not a constant -- it falls with n (mean degree ~30 at
// n = 1,500 down to ~15 at n = 12,000) -- so sparsax.lu_backend routes on KLU's
// flops per factor entry instead (klu_pattern_work).
//
// Structure mirrors KLU exactly: a shared pattern (the umfpack_di_symbolic
// analysis + CSC structure, immutable after create) owning a content-addressed
// LRU of numeric factors keyed on the COO values, with the expensive numeric
// work done outside the registry lock; plus a token side-table holding strong
// references so lu-style factor tokens survive LRU eviction.
// ===========================================================================

// Numeric factors retained per UMFPACK pattern. Same role and default as the
// KLU cache (see set_umf_cache_size): under vmap it must be at least the batch
// size or a factor-once-solve-many sweep silently refactorizes.
static std::atomic<size_t> g_umf_cache_cap{32};

// Fill a Control array with the settings every UMFPACK call in this file uses.
// Print level 0 keeps UMFPACK off stderr (errors are surfaced as XLA errors,
// as in the CHOLMOD path). Iterative refinement is off: KLU does none, so
// leaving UMFPACK's default of 2 refinement steps on would make a solve ~3x
// dearer than its KLU counterpart and quietly bias any timing-based routing
// between the two. With IRSTEP == 0 and sys in {A, A'}, UMFPACK does not access
// the matrix during a solve at all, so umf_solve_one passes NULL for Ap/Ai/Ax
// and no per-factor copy of the values has to be kept alive.
static void umf_control(double* Control) {
  umfpack_di_defaults(Control);
  Control[UMFPACK_PRL] = 0;
  Control[UMFPACK_IRSTEP] = 0;
}

// A numeric factor with refcounted lifetime, exactly as for KLU: the LRU and
// any thread currently solving with it both hold a shared_ptr, so eviction
// never frees a factor an in-flight solve is still using.
struct UmfNumericHolder {
  void* num = nullptr;
  double rcond = std::numeric_limits<double>::quiet_NaN();  // Info[UMFPACK_RCOND]
  bool factored() const { return num != nullptr; }
  ~UmfNumericHolder() {
    if (num) umfpack_di_free_numeric(&num);
  }
};
using UmfNumPtr = std::shared_ptr<UmfNumericHolder>;

using UmfNumericSlot = NumericSlot<UmfNumericHolder>;

// The shared, immutable-after-create pattern: umfpack_di_symbolic's analysis
// (column pre-ordering + supernodal column elimination tree) plus the CSC
// structure. UMFPACK treats the Symbolic object as read-only during numeric,
// so all threads share one without locking; the numeric LRU has its own short
// mutex, held only across the lookup/insert.
struct UmfPattern {
  int64_t n = 0;
  std::vector<int32_t> Ai, Aj;   // full COO copy, for exact hit verification
  std::vector<int32_t> Ap;       // CSC column pointers, size n+1
  std::vector<int32_t> Ci;       // CSC row indices, size nnz_csc
  std::vector<int64_t> pos;      // COO k -> CSC slot
  int64_t nnz_csc = 0;
  void* symbolic = nullptr;
  std::mutex cache_mtx;             // guards `cache` / `next` only
  std::vector<UmfNumericSlot> cache;
  FactorsInFlight in_flight;
  size_t next = 0;                  // round-robin eviction cursor
  ~UmfPattern() {
    if (symbolic) umfpack_di_free_symbolic(&symbolic);
  }
};

static std::mutex g_umf_reg_mtx;  // guards the shared pattern registry only
static PatternRegistry<UmfPattern> g_umf_registry;
static std::atomic<uint64_t> g_umf_generation{1};  // bumped by clear_cache
static std::atomic<int64_t> g_umf_num_factorizations{0};

// UMFPACK factor tokens: a slot holds a strong reference to the numeric factor, so
// the per-pattern LRU cannot evict a factor a live token references.
using UmfHeldFactor = HeldFactor<UmfPattern, UmfNumericHolder>;
static TokenTable<UmfHeldFactor> g_umf_tokens(kTokenUmfpack, "sparsax(umfpack)");

// No lock around the native calls. UMFPACK documents the Symbolic object as
// unmodified by umfpack_di_numeric, and a Numeric object as unmodified by
// umfpack_di_wsolve and umfpack_di_get_determinant; each thread brings its own
// Control and solve workspace (t_umf). So, unlike KLU, whose solve writes into
// the factor, threads may solve against one shared factor concurrently.

// Build the CSC structure and run umfpack_di_symbolic.  Caller holds g_umf_reg_mtx.
// Ax (the values of the call that first meets the pattern; may be null) goes to
// umfpack_di_symbolic: its strategy test reads the diagonal, and without values
// it assumes the unsymmetric strategy.  For I - rho W the symmetric strategy it
// then picks makes every later numeric factorization 1.35-1.95x faster.  The
// analysis stays valid for any values on the pattern.
static std::shared_ptr<UmfPattern> umf_create_pattern_locked(
    const int32_t* Ai, const int32_t* Aj, const double* Ax, int64_t nnz,
    int64_t n, std::string* err) {
  auto e = std::make_shared<UmfPattern>();
  e->n = n;
  e->Ai.assign(Ai, Ai + nnz);
  e->Aj.assign(Aj, Aj + nnz);
  if (!build_csc_from_coo(Ai, Aj, nnz, n, "umfpack", e->Ap, e->Ci, e->pos,
                          e->nnz_csc, err))
    return nullptr;

  std::vector<double> Ax_csc;
  if (Ax) {
    Ax_csc.assign(e->nnz_csc > 0 ? e->nnz_csc : 1, 0.0);
    for (int64_t k = 0; k < nnz; ++k)
      if (e->pos[k] >= 0) Ax_csc[e->pos[k]] += Ax[k];
  }
  double Control[UMFPACK_CONTROL];
  umf_control(Control);
  int status = umfpack_di_symbolic(
      static_cast<int32_t>(n), static_cast<int32_t>(n), e->Ap.data(),
      e->Ci.data(), Ax ? Ax_csc.data() : nullptr, &e->symbolic, Control, nullptr);
  if (status != UMFPACK_OK || !e->symbolic) {
    *err = "sparsax(umfpack): umfpack_di_symbolic failed (status " +
           std::to_string(status) + ")";
    return nullptr;
  }

  g_umf_registry[pattern_hash(Ai, Aj, nnz, n)].push_back(e);
  return e;
}

// Per-thread Control + scratch. umfpack_di_wsolve takes its workspace from the
// caller (Wi of n int32s, W of n doubles with refinement off) and allocates
// nothing, so a solve never touches malloc.
struct UmfThreadLocal {
  double control[UMFPACK_CONTROL];
  bool ready = false;
  std::vector<double> csc, bcol, xcol, w;
  std::vector<int32_t> wi;
  std::shared_ptr<UmfPattern> last_pat;  // see get_pattern
  uint64_t last_gen = 0;
  double* get() {
    if (!ready) {
      umf_control(control);
      ready = true;
    }
    return control;
  }
};
static thread_local UmfThreadLocal t_umf;

static std::shared_ptr<UmfPattern> umf_get_pattern(const int32_t* Ai,
                                                   const int32_t* Aj,
                                                   const double* Ax,
                                                   int64_t nnz, int64_t n,
                                                   std::string* err) {
  return get_pattern(g_umf_reg_mtx, g_umf_registry, g_umf_generation, t_umf, Ai, Aj,
                     nnz, n, [&] {
                       return umf_create_pattern_locked(Ai, Aj, Ax, nnz, n, err);
                     });
}

// Return a refcounted numeric factor for these COO values: hit the pattern's
// shared LRU (short lock) or build a fresh umfpack_di_numeric outside the lock
// with this thread's own Control + scratch.
static UmfNumPtr umf_get_factor(const std::shared_ptr<UmfPattern>& pat,
                                const double* Ax, int64_t nnz,
                                std::string* err) {
  uint64_t h = fnv1a(Ax, nnz * sizeof(double), 14695981039346656037ULL);
  UmfNumPtr hit = find_or_claim_factor(*pat, Ax, nnz, h, [] {});
  if (hit) return hit;
  InFlightClaim claim(pat->cache_mtx, pat->in_flight, h);

  if (!all_finite(Ax, nnz))
    return factorization_failed<UmfNumericHolder>(
        "sparsax(umfpack): matrix has a non-finite value", err);

  double* Control = t_umf.get();
  t_umf.csc.assign(pat->nnz_csc > 0 ? pat->nnz_csc : 1, 0.0);
  for (int64_t k = 0; k < nnz; ++k)
    if (pat->pos[k] >= 0) t_umf.csc[pat->pos[k]] += Ax[k];

  void* raw = nullptr;
  double Info[UMFPACK_INFO];
  int status = umfpack_di_numeric(pat->Ap.data(), pat->Ci.data(),
                                  t_umf.csc.data(), pat->symbolic, &raw, Control,
                                  Info);
  // A singular matrix comes back as UMFPACK_WARNING_singular_matrix with a
  // usable Numeric object; treat it as failure so umf_solve / umf_logdet raise
  // rather than returning silent infinities, matching the KLU path.
  if (status != UMFPACK_OK || !raw) {
    if (raw) umfpack_di_free_numeric(&raw);
    std::string message = "sparsax(umfpack): umfpack_di_numeric failed (status " +
                          std::to_string(status) + "; matrix may be singular)";
    if (status == UMFPACK_WARNING_singular_matrix)
      return factorization_failed<UmfNumericHolder>(std::move(message), err);
    *err = std::move(message);
    return nullptr;
  }
  g_umf_num_factorizations.fetch_add(1, std::memory_order_relaxed);
  auto holder = std::make_shared<UmfNumericHolder>();
  holder->num = raw;
  holder->rcond = Info[UMFPACK_RCOND];
  cache_factor(*pat, g_umf_cache_cap, h, Ax, nnz, holder);
  return holder;
}

// log|det(A)| straight from UMFPACK, the one quantity this backend exposes as a
// library call where KLU does not: umfpack_di_get_determinant reads the
// determinant off the LU factors and the permutations and returns it in
// mantissa/exponent form (det == Mx * 10^Ex), so nothing overflows however
// large n gets. klu_logdet_one above has to reconstruct the same number as a
// sum of n logarithms off Udiag and Rs; both are overflow-safe, but this one
// accumulates no rounding over n terms. That is the SAR Jacobian
// log|I - rho W| for a directed W. Returns NaN on a singular determinant.
static double umf_logdet_one(void* num) {
  double Mx = 0.0, Ex = 0.0;
  int status = umfpack_di_get_determinant(&Mx, &Ex, num, nullptr);
  // The over/underflow warnings say only that Mx * 10^Ex would not fit in a
  // double if the caller multiplied it out. We never do -- log|det| is
  // log|Mx| + Ex*log(10), which is the point of asking for the split form --
  // so those two are successes here. Anything else (a singular matrix, an
  // invalid Numeric object) is not.
  if (status != UMFPACK_OK &&
      status != UMFPACK_WARNING_determinant_underflow &&
      status != UMFPACK_WARNING_determinant_overflow)
    return std::numeric_limits<double>::quiet_NaN();
  if (Mx == 0.0 || !std::isfinite(Mx) || !std::isfinite(Ex))
    return std::numeric_limits<double>::quiet_NaN();
  return std::log(std::fabs(Mx)) + Ex * std::log(10.0);
}

// The determinant's mantissa carries its sign.
static double umf_sign_one(void* num) {
  double Mx = 0.0, Ex = 0.0;
  umfpack_di_get_determinant(&Mx, &Ex, num, nullptr);
  return Mx < 0.0 ? -1.0 : 1.0;
}

// Solve A x = b (trans == false) or A^T x = b (trans == true, used by the VJP)
// for one right-hand-side block. UMFPACK solves a single RHS per call, so a
// multi-RHS block loops over columns; b/x are row-major (JAX layout), so each
// column is gathered into scratch and scattered back. With refinement off the
// matrix itself is not accessed, hence the NULL Ap/Ai/Ax.  umfpack_di_wsolve
// only reads the factor, and each thread brings its own Control and workspace,
// so a block of two or more panels is split across threads (for_each_panel).
static std::string umf_solve_panel(void* num, int sys, const double* bdata,
                                   int64_t n, int64_t ld, int64_t j0, int64_t w,
                                   double* xdata) {
  UmfThreadLocal& tl = t_umf;
  double* Control = tl.get();
  tl.wi.resize(n);
  tl.w.resize(n);
  auto call = [&](double* x, const double* b) -> std::string {
    int status = umfpack_di_wsolve(sys, nullptr, nullptr, nullptr, x, b, num,
                                   Control, nullptr, tl.wi.data(), tl.w.data());
    if (status != UMFPACK_OK)
      return "sparsax(umfpack): umfpack_di_solve failed (status " +
             std::to_string(status) + ")";
    return "";
  };
  if (ld == 1) return call(xdata, bdata);

  tl.bcol.resize(n);
  tl.xcol.resize(n);
  for (int64_t j = j0; j < j0 + w; ++j) {
    for (int64_t i = 0; i < n; ++i) tl.bcol[i] = bdata[i * ld + j];
    std::string e = call(tl.xcol.data(), tl.bcol.data());
    if (!e.empty()) return e;
    for (int64_t i = 0; i < n; ++i) xdata[i * ld + j] = tl.xcol[i];
  }
  return "";
}

static ffi::Error umf_solve_one(UmfNumericHolder* holder, bool trans,
                                const double* bdata, int64_t n, int64_t nrhs,
                                double* xdata) {
  void* num = holder->num;
  int sys = trans ? UMFPACK_At : UMFPACK_A;
  const unsigned T = panel_threads(nrhs);
  std::string err =
      T > 1 ? for_each_panel(nrhs, T,
                             [&] {
                               return [&](int64_t j0, int64_t w) {
                                 return umf_solve_panel(num, sys, bdata, n, nrhs,
                                                        j0, w, xdata);
                               };
                             })
            : umf_solve_panel(num, sys, bdata, n, nrhs, 0, nrhs, xdata);
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

struct UmfBackend {
  using Pattern = UmfPattern;
  using Holder = UmfNumericHolder;
  using Held = UmfHeldFactor;
  static constexpr const char* kName = "sparsax(umfpack)";
  static constexpr TokenIssuer kIssuer = kTokenUmfpack;
  static TokenTable<Held>& tokens() { return g_umf_tokens; }
  // The analysis reads the values of the call that first meets the pattern; a batch
  // offers its first element's.
  static std::shared_ptr<Pattern> pattern(const int32_t* Ai, const int32_t* Aj,
                                          const double* Ax, int64_t nnz, int64_t n,
                                          std::string* err) {
    return umf_get_pattern(Ai, Aj, Ax, nnz, n, err);
  }
  static std::shared_ptr<Holder> factor(const std::shared_ptr<Pattern>& pat,
                                        const double* Ax, int64_t nnz,
                                        std::string* err) {
    return umf_get_factor(pat, Ax, nnz, err);
  }
  static ffi::Error solve(const Pattern&, Holder& num, bool trans, const double* b,
                          int64_t n, int64_t nrhs, double* x) {
    return umf_solve_one(&num, trans, b, n, nrhs, x);
  }
  static double logdet(const Pattern&, const Holder& num) {
    return umf_logdet_one(num.num);
  }
  static double sign(const Pattern&, const Holder& num) {
    return umf_sign_one(num.num);
  }
  static double rcond(const Pattern&, const Holder& num) { return num.rcond; }
};

SPARSAX_DEFINE_LU_HANDLERS(SparsaxUmf, UmfBackend);

void umf_set_cache_size(size_t cap) {
  g_umf_cache_cap = cap;
  trim_numeric_caches(g_umf_reg_mtx, g_umf_registry, cap);
}

// Each pattern owns its numeric cache, so its factors are freed once no in-flight
// solve still holds them.  Outstanding tokens go too.
void umf_clear_cache() {
  std::lock_guard<std::mutex> lk(g_umf_reg_mtx);
  clear_patterns_locked(g_umf_registry);
  g_umf_generation.fetch_add(1, std::memory_order_relaxed);
  g_umf_tokens.clear();
}

size_t umf_pattern_count() {
  std::lock_guard<std::mutex> lk(g_umf_reg_mtx);
  size_t count = 0;
  for (auto& [h, chain] : g_umf_registry) count += chain.size();
  return count;
}

int64_t umf_factorizations() {
  return g_umf_num_factorizations.load(std::memory_order_relaxed);
}
