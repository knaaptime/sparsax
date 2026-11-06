// KLU: sparse LU for general matrices, the first LU backend.
#include <klu.h>

#include "lu_backends.h"
#include "lu_handlers.h"

// ===========================================================================
// KLU: sparse LU for NON-symmetric matrices (e.g. A = I - rho W, where W is a
// row-standardised / asymmetric spatial weights matrix that cannot be
// d-symmetrised) as XLA FFI custom calls, mirroring the CHOLMOD path above.
//
// Same design as CHOLMOD — per-thread klu_common workspaces, a shared
// per-pattern symbolic analysis (klu_analyze reused across value changes) — plus a small
// content-addressed LRU of *numeric* factors per pattern. The LRU is what makes
// factor-reuse survive jax.vmap: a vmapped shift-invert Krylov basis issues the
// same B distinct value-vectors (one per chain) across several sequential solve
// calls at the same rho; a single-slot value cache would thrash (each batched
// call cycles all B chains through the one slot, so every solve refactors), but
// an LRU of capacity >= B keeps every chain's factor live, so the m+1 basis
// solves cost one klu_factor per chain and m cheap klu_solves — matching numpy's
// factorisation reuse while running the whole batch in one native FFI call.
// ===========================================================================

// Numeric factors retained per pattern in the shared cache (klu_get_factor).
// Small: within one chain a sweep touches only a couple of distinct ρ.
// Tunable from Python via set_lu_cache_size; atomic because concurrent solves
// read it.
static std::atomic<size_t> g_lu_cache_cap{32};

// A numeric factor with refcounted lifetime: the shared cache and any thread
// currently solving with it both hold a shared_ptr, so evicting it from the
// cache never frees a factor a concurrent solve is still using.
struct KluNumericHolder {
  klu_numeric* num = nullptr;
  bool factored() const { return num != nullptr; }
  ~KluNumericHolder() {
    if (num) {
      klu_common c;
      klu_defaults(&c);
      klu_free_numeric(&num, &c);
    }
  }
};
using KluNumPtr = std::shared_ptr<KluNumericHolder>;

using KluNumericSlot = NumericSlot<KluNumericHolder>;

// ---------------------------------------------------------------------------
// Shared, immutable-after-create pattern: the fill-reducing analysis (klu_analyze
// = AMD/COLAMD ordering + BTF) plus the CSC structure — read-only, so all threads
// share it without locking (klu_factor / klu_solve treat the symbolic as const).
// It also owns a content-addressed numeric-factor cache (keyed on Ax) guarded by
// its own short mutex: threads hold the lock only for the lookup/insert, and run
// the expensive klu_factor / klu_solve OUTSIDE the lock with a per-thread common,
// so concurrent per-device (pmap) solves run in parallel.  The cache is shared
// (not thread-local) so a factor is reused regardless of which XLA thread-pool
// thread happens to run a given solve.
// ---------------------------------------------------------------------------
struct KluPattern {
  int64_t n = 0;
  std::vector<int32_t> Ai, Aj;   // full COO copy, for exact hit verification
  std::vector<int32_t> Ap;       // CSC column pointers, size n+1
  std::vector<int32_t> Ci;       // CSC row indices, size nnz_csc
  std::vector<int64_t> pos;      // COO k -> CSC slot
  int64_t nnz_csc = 0;
  klu_symbolic* symbolic = nullptr;
  std::mutex cache_mtx;             // guards `cache` / `next` only
  std::vector<KluNumericSlot> cache;
  FactorsInFlight in_flight;
  size_t next = 0;                  // round-robin eviction cursor
  ~KluPattern() {
    if (symbolic) {
      klu_common c;
      klu_defaults(&c);
      klu_free_symbolic(&symbolic, &c);
    }
  }
};

static std::mutex g_klu_reg_mtx;  // guards the shared pattern registry only
static PatternRegistry<KluPattern> g_klu_registry;
static std::atomic<uint64_t> g_klu_generation{1};  // bumped by clear_cache
// Total numeric (re)factorizations, for tests/introspection (atomic: threaded).
static std::atomic<int64_t> g_klu_num_factorizations{0};

// ---------------------------------------------------------------------------
// KLU factor tokens (lu_factor / lu_solve_factor): the non-symmetric analogue
// of the CHOLMOD factor/solve_factor primitives. A lu_factor(A) call computes
// (or fetches from the per-pattern LRU) a klu_numeric for Ax and returns an
// opaque int64[2] token; lu_solve_factor consumes it for an unbounded sequence
// of solves (A x = b or A^T x = b) without re-hashing Ax or refactoring. This
// expresses the recurrence V_{j+1} = A^{-1}(G V_j) that lu_solve cannot.
//
// The token holds a *strong* shared_ptr<KluNumericHolder> (KluNumPtr) in a
// side table, so the factor cannot be evicted by the per-pattern LRU while a
// token references it — the KLU LRU eviction simply drops its shared_ptr, but
// the side table's copy keeps the factor alive until the token is discarded by
// XLA. This differs from the CHOLMOD path (which copies the factor) because
// KLU numeric factors are already refcounted shared objects.
// ---------------------------------------------------------------------------

using KluHeldFactor = HeldFactor<KluPattern, KluNumericHolder>;
static TokenTable<KluHeldFactor> g_klu_tokens(kTokenKlu, "sparsax(klu)");

// Concurrency.  There is no global KLU lock.  A pattern's symbolic analysis is
// read-only once built; klu_factor allocates a fresh numeric factor using this
// thread's own klu_common; a logdet only reads a finished factor's U diagonal.
// The one piece of a numeric factor that a solve writes is its workspace, and each
// thread brings its own (see klu_solve_one).  (A global mutex used to serialize all
// KLU work, so concurrent per-chain solves ran one at a time, and then a per-factor
// one made threads sharing a factor take turns.  The "wrong draws" that motivated
// them came from two threads solving into one shared workspace at once.)

// Build full (non-symmetric) CSC — every entry, sorted by (col, row), duplicates
// merged — and run klu_analyze. Caller holds g_klu_reg_mtx.
static std::shared_ptr<KluPattern> klu_create_pattern_locked(
    const int32_t* Ai, const int32_t* Aj, int64_t nnz, int64_t n,
    std::string* err) {
  auto e = std::make_shared<KluPattern>();
  e->n = n;
  e->Ai.assign(Ai, Ai + nnz);
  e->Aj.assign(Aj, Aj + nnz);
  if (!build_csc_from_coo(Ai, Aj, nnz, n, "klu", e->Ap, e->Ci, e->pos,
                          e->nnz_csc, err))
    return nullptr;

  klu_common c;
  klu_defaults(&c);
  e->symbolic = klu_analyze(static_cast<int32_t>(n), e->Ap.data(), e->Ci.data(),
                            &c);
  if (!e->symbolic || c.status != KLU_OK) {
    *err = "sparsax(klu): klu_analyze failed (status " +
           std::to_string(c.status) + ")";
    return nullptr;
  }

  g_klu_registry[pattern_hash(Ai, Aj, nnz, n)].push_back(e);
  return e;
}

// Per-thread KLU common + scratch buffers.  klu_common is a small POD of control
// params + stats (no heap workspace), so one per thread removes the shared-common
// serialization; XLA's CPU thread pool can run a chain's solves on different pool
// threads, so this is keyed on the OS thread, not on the chain (the numeric cache
// below is shared to make reuse thread-agnostic).
struct KluThreadLocal {
  klu_common common;
  bool ready = false;
  std::vector<double> csc, work, xwork;
  std::shared_ptr<KluPattern> last_pat;  // see get_pattern
  uint64_t last_gen = 0;
  klu_common* get() {
    if (!ready) {
      klu_defaults(&common);
      ready = true;
    }
    return &common;
  }
};
static thread_local KluThreadLocal t_klu;

static std::shared_ptr<KluPattern> klu_get_pattern(const int32_t* Ai,
                                                   const int32_t* Aj,
                                                   int64_t nnz, int64_t n,
                                                   std::string* err) {
  return get_pattern(g_klu_reg_mtx, g_klu_registry, g_klu_generation, t_klu, Ai, Aj,
                     nnz, n, [&] {
                       return klu_create_pattern_locked(Ai, Aj, nnz, n, err);
                     });
}

// Return a refcounted numeric factor for these COO values: hit the pattern's
// shared cache (short lock) or build a fresh klu_factor OUTSIDE the lock with
// this thread's own common + scratch, so concurrent solves run in parallel.
// A refactor reuses the recycled factor's pivot sequence; if the new values
// make that sequence unstable -- reciprocal pivot growth below this -- the
// factor is redone with fresh partial pivoting.  For I - rho W inside the
// stability region the growth stays near 1.
constexpr double kKluMinRgrowth = 1e-8;

static KluNumPtr klu_get_factor(const std::shared_ptr<KluPattern>& pat,
                                const double* Ax, int64_t nnz,
                                std::string* err) {
  uint64_t h = fnv1a(Ax, nnz * sizeof(double), 14695981039346656037ULL);
  klu_numeric* recycled = nullptr;
  KluNumPtr hit = find_or_claim_factor(*pat, Ax, nnz, h, [&] {
    if (KluNumericSlot* v =
            evictable_slot_locked(pat->cache, pat->next, g_lu_cache_cap)) {
      recycled = v->num->num;
      v->num->num = nullptr;
      v->num.reset();
      v->hash = 0;
      v->Ax.clear();
    }
  });
  if (hit) return hit;
  InFlightClaim claim(pat->cache_mtx, pat->in_flight, h);

  // Miss: scatter + factor outside the lock (per-thread common + scratch).
  klu_common* c = t_klu.get();
  // Partial pivoting does not see a NaN: the factorization "succeeds", and only
  // the solves show it.  One off U's diagonal does not even reach the logdet.
  if (!all_finite(Ax, nnz)) {
    if (recycled) klu_free_numeric(&recycled, c);
    return factorization_failed<KluNumericHolder>(
        "sparsax(klu): matrix has a non-finite value", err);
  }
  t_klu.csc.assign(pat->nnz_csc > 0 ? pat->nnz_csc : 1, 0.0);
  for (int64_t k = 0; k < nnz; ++k)
    if (pat->pos[k] >= 0) t_klu.csc[pat->pos[k]] += Ax[k];

  // klu_refactor into the recycled factor skips the pivot search and the
  // allocation (1.2-1.8x faster than klu_factor on I - rho W).
  klu_numeric* raw = nullptr;
  if (recycled) {
    c->status = KLU_OK;
    if (klu_refactor(pat->Ap.data(), pat->Ci.data(), t_klu.csc.data(),
                     pat->symbolic, recycled, c) &&
        c->status == KLU_OK &&
        klu_rgrowth(pat->Ap.data(), pat->Ci.data(), t_klu.csc.data(),
                    pat->symbolic, recycled, c) &&
        std::isfinite(c->rgrowth) && c->rgrowth >= kKluMinRgrowth)
      raw = recycled;
    else
      klu_free_numeric(&recycled, c);
  }

  c->status = KLU_OK;
  if (!raw)
    raw = klu_factor(pat->Ap.data(), pat->Ci.data(), t_klu.csc.data(),
                     pat->symbolic, c);
  if (!raw || c->status != KLU_OK) {
    const int status = c->status;
    if (raw) klu_free_numeric(&raw, c);
    std::string message = "sparsax(klu): klu_factor failed (status " +
                          std::to_string(status) + "; matrix may be singular)";
    if (status == KLU_SINGULAR)
      return factorization_failed<KluNumericHolder>(std::move(message), err);
    *err = std::move(message);
    return nullptr;
  }
  g_klu_num_factorizations.fetch_add(1, std::memory_order_relaxed);
  auto holder = std::make_shared<KluNumericHolder>();
  holder->num = raw;
  cache_factor(*pat, g_lu_cache_cap, h, Ax, nnz, holder);
  return holder;
}

// log|det(A)| from a klu_numeric factor. KLU factors A (possibly row-scaled by
// Rs and BTF-permuted) as L U Pnum; the diagonal of U is held in num->Udiag,
// size n. For a non-singular matrix, |det(A)| = prod_i |U_ii| * prod_i |Rs_i|
// (row i is scaled by 1/Rs_i before factoring, so the un-scaled determinant is
// the scaled one times the product of the scale factors). Rs is NULL when no
// scaling is applied. The BTF off-diagonal blocks do not affect the diagonal
// product. Returns the logdet in *out; sets it non-finite on a singular U
// (any zero or NaN U_ii).
static double klu_logdet_one(const klu_numeric* num, int64_t n) {
  const double* Udiag = static_cast<const double*>(num->Udiag);
  const double* Rs = num->Rs;
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    double u = Udiag[i];
    if (u == 0.0 || !std::isfinite(u)) return std::numeric_limits<double>::quiet_NaN();
    acc += std::log(std::fabs(u));
  }
  if (Rs) {
    for (int64_t i = 0; i < n; ++i) {
      double r = Rs[i];
      if (r == 0.0 || !std::isfinite(r))
        return std::numeric_limits<double>::quiet_NaN();
      acc += std::log(std::fabs(r));
    }
  }
  return acc;
}

// A permutation's sign, from the parity of its cycles.
static double permutation_sign(const int32_t* perm, int64_t n) {
  std::vector<char> seen(n, 0);
  double sign = 1.0;
  for (int64_t i = 0; i < n; ++i) {
    int64_t length = 0;
    for (int64_t j = i; !seen[j]; j = perm[j]) {
      seen[j] = 1;
      ++length;
    }
    if (length > 0 && length % 2 == 0) sign = -sign;
  }
  return sign;
}

// The sign of det(A).  KLU factors P (R^-1 A) Q = L U with L unit lower triangular,
// so det(A) is the product of the permutations' signs, the row scales and U's
// diagonal.  The off-diagonal blocks of the block triangular form do not enter.
static double klu_sign_one(const klu_symbolic* symbolic, const klu_numeric* num,
                           int64_t n) {
  double sign = permutation_sign(num->Pnum, n) * permutation_sign(symbolic->Q, n);
  const double* Udiag = static_cast<const double*>(num->Udiag);
  for (int64_t i = 0; i < n; ++i)
    if (Udiag[i] < 0.0) sign = -sign;
  if (num->Rs)
    for (int64_t i = 0; i < n; ++i)
      if (num->Rs[i] < 0.0) sign = -sign;
  return sign;
}

// Solve A x = b (trans == false) or A^T x = b (trans == true, used by the VJP)
// for one right-hand-side block. b/x are row-major (JAX layout); KLU is
// column-major with leading dimension n, so multi-RHS blocks transpose through
// a work buffer and KLU solves in place.
//
// Concurrency: klu_solve writes into the factor's workspace (Xwork) and only reads
// every other field, so each thread solves against a shallow copy of the factor
// whose Xwork (4n entries) is its own.  Threads can then share a factor (chains at
// bit-identical values, say, all started at one rho) without a lock.  A block of
// two or more panels is split across threads (see for_each_panel).
static ffi::Error klu_solve_one(const KluPattern* pat, KluNumericHolder* holder,
                                bool trans, const double* bdata, int64_t n,
                                int64_t nrhs, double* xdata) {
  auto klu_call = [&](klu_numeric* num, int64_t w, double* work,
                      klu_common* c) -> std::string {
    c->status = KLU_OK;
    int ok = trans ? klu_tsolve(pat->symbolic, num, static_cast<int32_t>(n),
                                static_cast<int32_t>(w), work, c)
                   : klu_solve(pat->symbolic, num, static_cast<int32_t>(n),
                               static_cast<int32_t>(w), work, c);
    if (!ok || c->status != KLU_OK)
      return "sparsax(klu): klu_solve failed (status " +
             std::to_string(c->status) + ")";
    return "";
  };

  const unsigned T = panel_threads(nrhs);
  if (T > 1) {
    std::string err = for_each_panel(nrhs, T, [&] {
      klu_numeric local = *holder->num;
      std::vector<double> xwork(4 * n), work;
      local.Xwork = xwork.data();
      klu_common c;
      klu_defaults(&c);
      return [&, local, xwork = std::move(xwork), work = std::move(work),
              c](int64_t j0, int64_t w) mutable {
        local.Xwork = xwork.data();
        work.resize(n * w);
        gather_cols(bdata, n, nrhs, j0, w, work.data());
        std::string e = klu_call(&local, w, work.data(), &c);
        if (e.empty()) scatter_cols(work.data(), n, nrhs, j0, w, xdata);
        return e;
      };
    });
    if (!err.empty()) return ffi::Error::Internal(err);
    return ffi::Error::Success();
  }

  klu_common* c = t_klu.get();
  std::vector<double>& work = t_klu.work;
  work.resize(n * nrhs);
  if (nrhs == 1)
    std::memcpy(work.data(), bdata, n * sizeof(double));
  else
    gather_cols(bdata, n, nrhs, 0, nrhs, work.data());
  t_klu.xwork.resize(4 * n);
  klu_numeric local = *holder->num;
  local.Xwork = t_klu.xwork.data();
  const std::string err = klu_call(&local, nrhs, work.data(), c);
  if (!err.empty()) return ffi::Error::Internal(err);
  if (nrhs == 1)
    std::memcpy(xdata, work.data(), n * sizeof(double));
  else
    scatter_cols(work.data(), n, nrhs, 0, nrhs, xdata);
  return ffi::Error::Success();
}

struct KluBackend {
  using Pattern = KluPattern;
  using Holder = KluNumericHolder;
  using Held = KluHeldFactor;
  static constexpr const char* kName = "sparsax(klu)";
  static constexpr TokenIssuer kIssuer = kTokenKlu;
  static TokenTable<Held>& tokens() { return g_klu_tokens; }
  static std::shared_ptr<Pattern> pattern(const int32_t* Ai, const int32_t* Aj,
                                          const double*, int64_t nnz, int64_t n,
                                          std::string* err) {
    return klu_get_pattern(Ai, Aj, nnz, n, err);
  }
  static std::shared_ptr<Holder> factor(const std::shared_ptr<Pattern>& pat,
                                        const double* Ax, int64_t nnz,
                                        std::string* err) {
    return klu_get_factor(pat, Ax, nnz, err);
  }
  static ffi::Error solve(const Pattern& pat, Holder& num, bool trans,
                          const double* b, int64_t n, int64_t nrhs, double* x) {
    return klu_solve_one(&pat, &num, trans, b, n, nrhs, x);
  }
  static double logdet(const Pattern& pat, const Holder& num) {
    return klu_logdet_one(num.num, pat.n);
  }
  static double sign(const Pattern& pat, const Holder& num) {
    return klu_sign_one(pat.symbolic, num.num, pat.n);
  }
  static double rcond(const Pattern& pat, const Holder& num) {
    klu_common* c = t_klu.get();
    klu_rcond(pat.symbolic, num.num, c);
    return c->rcond;
  }
};

SPARSAX_DEFINE_LU_HANDLERS(SparsaxLu, KluBackend);

// See lu_work in the Python module.  The canonical values keep KLU's threshold
// pivoting on the diagonal, so the factor's structure is the symbolic one and depends
// on the pattern alone: diagonal 1, each off-diagonal -0.5 / (its row's degree).
LuWork klu_pattern_work(const int32_t* Ai, const int32_t* Aj, int64_t nnz, int64_t n,
                        std::string* err) {
  std::vector<int32_t> rows(Ai, Ai + nnz), cols(Aj, Aj + nnz);
  for (int32_t i = 0; i < n; ++i) {
    rows.push_back(i);
    cols.push_back(i);
  }
  std::vector<int32_t> Ap, Ci;
  std::vector<int64_t> pos;
  int64_t nnz_csc = 0;
  if (!build_csc_from_coo(rows.data(), cols.data(), static_cast<int64_t>(rows.size()), n,
                          "klu", Ap, Ci, pos, nnz_csc, err))
    return {};
  std::vector<int32_t> degree(n, 0);
  for (int64_t j = 0; j < n; ++j)
    for (int32_t p = Ap[j]; p < Ap[j + 1]; ++p)
      if (Ci[p] != j) ++degree[Ci[p]];
  std::vector<double> values(nnz_csc);
  for (int64_t j = 0; j < n; ++j)
    for (int32_t p = Ap[j]; p < Ap[j + 1]; ++p)
      values[p] = Ci[p] == j ? 1.0 : -0.5 / degree[Ci[p]];

  klu_common c;
  klu_defaults(&c);
  klu_symbolic* symbolic = klu_analyze(static_cast<int32_t>(n), Ap.data(), Ci.data(), &c);
  klu_numeric* numeric =
      symbolic ? klu_factor(Ap.data(), Ci.data(), values.data(), symbolic, &c) : nullptr;
  LuWork work;
  if (numeric && klu_flops(symbolic, numeric, &c)) {
    work.flops = c.flops;
    work.lnz = numeric->lnz;
    work.unz = numeric->unz;
  } else {
    *err = "sparsax(klu): could not measure the pattern (status " +
           std::to_string(c.status) + ")";
  }
  if (numeric) klu_free_numeric(&numeric, &c);
  if (symbolic) klu_free_symbolic(&symbolic, &c);
  return work;
}

void klu_set_cache_size(size_t cap) {
  g_lu_cache_cap = cap;
  trim_numeric_caches(g_klu_reg_mtx, g_klu_registry, cap);
}

// Each pattern owns its numeric cache, so its factors are freed once no in-flight
// solve still holds them.  Outstanding tokens go too.
void klu_clear_cache() {
  std::lock_guard<std::mutex> lk(g_klu_reg_mtx);
  clear_patterns_locked(g_klu_registry);
  g_klu_generation.fetch_add(1, std::memory_order_relaxed);
  g_klu_tokens.clear();
}

size_t klu_pattern_count() {
  std::lock_guard<std::mutex> lk(g_klu_reg_mtx);
  size_t count = 0;
  for (auto& [h, chain] : g_klu_registry) count += chain.size();
  return count;
}

int64_t klu_factorizations() {
  return g_klu_num_factorizations.load(std::memory_order_relaxed);
}
