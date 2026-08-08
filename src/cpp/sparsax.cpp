// sparsax: CHOLMOD sparse Cholesky as XLA FFI custom calls.
//
// Design:
//   - A single global cholmod_common guarded by a mutex (CHOLMOD's workspace
//     is not thread-safe; BLAS parallelism inside CHOLMOD is unaffected).
//   - Symbolic analyses are cached in a registry keyed by a hash of the
//     sparsity pattern (n, Ai, Aj), with full memcmp verification on hit.
//     Repeated solves with the same pattern reuse cholmod_analyze() output;
//     repeated solves with identical values also skip cholmod_factorize().
//   - Input is COO of a symmetric matrix. Entries with i <= j (upper triangle
//     plus diagonal) are used; entries with i > j are ignored, so both
//     "full symmetric" and "upper-only" input work. Duplicates are summed.
//
// The FFI handlers never touch Python and hold no GIL, so they run at full
// native speed inside XLA-compiled programs.

#include <cholmod.h>
#include <klu.h>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;
namespace nb = nanobind;

// ---------------------------------------------------------------------------
// Global CHOLMOD state
// ---------------------------------------------------------------------------

static std::mutex g_mutex;
static cholmod_common g_common;
static bool g_started = false;

static void error_handler(int status, const char* file, int line,
                          const char* message) {
  // Errors are surfaced as XLA errors; keep CHOLMOD off stderr.
  (void)status;
  (void)file;
  (void)line;
  (void)message;
}

static void ensure_started_locked() {
  if (!g_started) {
    cholmod_start(&g_common);
    g_common.print = 0;
    g_common.error_handler = error_handler;
    // Always produce an LL' factor: simplicial LDL' would silently accept
    // indefinite matrices, and an actual L is needed for MODE_L / MODE_LT
    // solves (e.g. sampling from N(0, A^{-1})).
    g_common.final_ll = 1;
    g_started = true;
  }
}

// ---------------------------------------------------------------------------
// Pattern registry (symbolic analysis cache)
// ---------------------------------------------------------------------------

struct PatternEntry {
  int64_t n = 0;
  std::vector<int32_t> Ai, Aj;   // full COO copy, for exact hit verification
  std::vector<int64_t> pos;      // COO k -> offset into A->x, -1 if i > j
  cholmod_sparse* A = nullptr;   // upper-triangular CSC, stype = 1
  cholmod_factor* L = nullptr;
  std::vector<double> last_Ax;   // values of the last successful factorization
  double logdet = 0.0;           // log|A| of the last successful factorization
  bool factored = false;
  // Persistent cholmod_solve2 workspaces, reused across solves.
  cholmod_dense* Xwork = nullptr;
  cholmod_dense* Ywork = nullptr;
  cholmod_dense* Ework = nullptr;
  // updown scratch factors. Lldl is a persistent simplicial LDL' copy of the
  // base factor, rebuilt only when the base is refactored (tracked by epoch);
  // Lupd is a per-call working copy of Lldl that cholmod_updown mutates, so the
  // base factor is never touched.
  cholmod_factor* Lldl = nullptr;
  cholmod_factor* Lupd = nullptr;
  int64_t factor_epoch = 0;   // bumped on each real (re)factorization of L
  int64_t ldl_epoch = -1;     // epoch Lldl was built from; -1 == not built

  // Value-keyed numeric-factor cache for the token primitives
  // (factor/solve_factor/logdet_factor). Each NumSlot holds a *copy* of the
  // base factor L for a given set of COO values; a Factor token references a
  // slot by index, and solve_factor/logdet_factor reuse that copy without
  // re-hashing or refactoring. The slot's factor is a CHOLMOD-owned copy, so
  // the base factor L is never mutated.
  struct NumSlot {
    uint64_t hash = 0;
    std::vector<double> Ax;  // values that produced `Lf`
    cholmod_factor* Lf = nullptr;
    double logdet = 0.0;
  };
  std::vector<NumSlot> num_cache;
  size_t num_next = 0;  // round-robin cursor (set alongside set_num_cache_size)
};

// A Factor token: an opaque reference to a NumSlot in a pattern's num_cache.
// Stored as an int64 pair (pattern_key, slot) in the XLA buffer; resolved back
// to a (PatternEntry*, NumSlot*) under the global lock.
struct FactorRef {
  uint64_t pattern_key = 0;
  int64_t slot = -1;
};

static uint64_t g_factor_next_key = 1;  // 0 reserved for "no pattern"
static std::mutex g_factor_ref_mtx;
// Maps a Factor token's pattern_key -> the owning PatternEntry* (raw pointer;
// safe because the pattern registry never frees entries except in clear_cache,
// which also clears this map under the same lock).
static std::unordered_map<uint64_t, PatternEntry*> g_factor_pattern_keys;

static std::unordered_map<uint64_t, std::vector<std::unique_ptr<PatternEntry>>>
    g_registry;
// Fast path: solver loops hit the same pattern every call.
static PatternEntry* g_last_entry = nullptr;
// Count of actual numeric (re)factorizations, for tests/introspection. Bumped
// only when factorize_locked truly refactors (not on the value-cache skip).
static int64_t g_num_factorizations = 0;

static uint64_t fnv1a(const void* data, size_t nbytes, uint64_t h) {
  const unsigned char* p = static_cast<const unsigned char*>(data);
  size_t nwords = nbytes / 8;
  for (size_t i = 0; i < nwords; ++i) {
    uint64_t v;
    std::memcpy(&v, p + i * 8, 8);
    h ^= v;
    h *= 1099511628211ULL;
  }
  for (size_t i = nwords * 8; i < nbytes; ++i) {
    h ^= p[i];
    h *= 1099511628211ULL;
  }
  return h;
}

static uint64_t pattern_hash(const int32_t* Ai, const int32_t* Aj, int64_t nnz,
                             int64_t n) {
  uint64_t h = 14695981039346656037ULL;
  h = fnv1a(&n, sizeof(n), h);
  h = fnv1a(Ai, nnz * sizeof(int32_t), h);
  h = fnv1a(Aj, nnz * sizeof(int32_t), h);
  return h;
}

static void free_entry_locked(PatternEntry* e) {
  if (e->L) cholmod_free_factor(&e->L, &g_common);
  if (e->A) cholmod_free_sparse(&e->A, &g_common);
  if (e->Xwork) cholmod_free_dense(&e->Xwork, &g_common);
  if (e->Ywork) cholmod_free_dense(&e->Ywork, &g_common);
  if (e->Ework) cholmod_free_dense(&e->Ework, &g_common);
  if (e->Lldl) cholmod_free_factor(&e->Lldl, &g_common);
  if (e->Lupd) cholmod_free_factor(&e->Lupd, &g_common);
  for (auto& s : e->num_cache)
    if (s.Lf) cholmod_free_factor(&s.Lf, &g_common);
  e->num_cache.clear();
}

// Build CSC (upper triangle, sorted, duplicates merged) and run
// cholmod_analyze. Returns nullptr and sets `err` on failure.
static PatternEntry* create_entry_locked(const int32_t* Ai, const int32_t* Aj,
                                         int64_t nnz, int64_t n,
                                         std::string* err) {
  for (int64_t k = 0; k < nnz; ++k) {
    if (Ai[k] < 0 || Ai[k] >= n || Aj[k] < 0 || Aj[k] >= n) {
      *err = "sparsax: COO index out of range for matrix dimension " +
             std::to_string(n);
      return nullptr;
    }
  }

  auto entry = std::make_unique<PatternEntry>();
  entry->n = n;
  entry->Ai.assign(Ai, Ai + nnz);
  entry->Aj.assign(Aj, Aj + nnz);
  entry->pos.assign(nnz, -1);

  // Upper-triangle entries sorted by (col, row); keep original k for mapping.
  struct Trip {
    int32_t i, j;
    int64_t k;
  };
  std::vector<Trip> upper;
  upper.reserve(nnz);
  for (int64_t k = 0; k < nnz; ++k)
    if (Ai[k] <= Aj[k]) upper.push_back({Ai[k], Aj[k], k});
  std::sort(upper.begin(), upper.end(), [](const Trip& a, const Trip& b) {
    return a.j != b.j ? a.j < b.j : a.i < b.i;
  });

  // Count unique (i, j) slots.
  int64_t nnz_csc = 0;
  for (size_t t = 0; t < upper.size(); ++t)
    if (t == 0 || upper[t].i != upper[t - 1].i || upper[t].j != upper[t - 1].j)
      ++nnz_csc;

  cholmod_sparse* A = cholmod_allocate_sparse(
      n, n, nnz_csc, /*sorted=*/1, /*packed=*/1, /*stype=*/1, CHOLMOD_REAL,
      &g_common);
  if (!A) {
    *err = "sparsax: cholmod_allocate_sparse failed";
    return nullptr;
  }

  int32_t* Ap = static_cast<int32_t*>(A->p);
  int32_t* Ar = static_cast<int32_t*>(A->i);
  double* Axv = static_cast<double*>(A->x);
  std::memset(Ap, 0, (n + 1) * sizeof(int32_t));

  int64_t slot = -1;
  for (size_t t = 0; t < upper.size(); ++t) {
    if (t == 0 || upper[t].i != upper[t - 1].i ||
        upper[t].j != upper[t - 1].j) {
      ++slot;
      Ar[slot] = upper[t].i;
      Axv[slot] = 0.0;
      Ap[upper[t].j + 1] += 1;  // per-column counts, prefix-summed below
    }
    entry->pos[upper[t].k] = slot;
  }
  for (int64_t j = 0; j < n; ++j) Ap[j + 1] += Ap[j];

  entry->A = A;
  entry->L = cholmod_analyze(A, &g_common);
  if (!entry->L || g_common.status < CHOLMOD_OK) {
    free_entry_locked(entry.get());
    *err = "sparsax: cholmod_analyze failed (status " +
           std::to_string(g_common.status) + ")";
    return nullptr;
  }

  uint64_t h = pattern_hash(Ai, Aj, nnz, n);
  auto& chain = g_registry[h];
  chain.push_back(std::move(entry));
  return chain.back().get();
}

static bool entry_matches(const PatternEntry* e, const int32_t* Ai,
                          const int32_t* Aj, int64_t nnz, int64_t n) {
  return e->n == n && e->Ai.size() == static_cast<size_t>(nnz) &&
         std::memcmp(e->Ai.data(), Ai, nnz * sizeof(int32_t)) == 0 &&
         std::memcmp(e->Aj.data(), Aj, nnz * sizeof(int32_t)) == 0;
}

static PatternEntry* get_or_create_entry_locked(const int32_t* Ai,
                                                const int32_t* Aj, int64_t nnz,
                                                int64_t n, std::string* err) {
  if (g_last_entry && entry_matches(g_last_entry, Ai, Aj, nnz, n))
    return g_last_entry;
  PatternEntry* found = nullptr;
  auto it = g_registry.find(pattern_hash(Ai, Aj, nnz, n));
  if (it != g_registry.end()) {
    for (auto& e : it->second) {
      if (entry_matches(e.get(), Ai, Aj, nnz, n)) {
        found = e.get();
        break;
      }
    }
  }
  if (!found) found = create_entry_locked(Ai, Aj, nnz, n, err);
  if (found) g_last_entry = found;
  return found;
}

// log|A| from the LL' factor diagonal (final_ll guarantees is_ll). Doubles
// as the positive-definiteness check: any nonpositive or NaN diagonal entry
// makes the result non-finite.
static double factor_logdet(const cholmod_factor* L) {
  double acc = 0.0;
  int64_t n = static_cast<int64_t>(L->n);
  if (L->is_super) {
    // Supernodal: diagonal of column j in supernode s sits at
    // x[px[s] + c*ld + c] with c = j - super[s], ld = pi[s+1] - pi[s].
    const int32_t* super = static_cast<const int32_t*>(L->super);
    const int32_t* pi = static_cast<const int32_t*>(L->pi);
    const int32_t* px = static_cast<const int32_t*>(L->px);
    const double* xv = static_cast<const double*>(L->x);
    int64_t nsuper = static_cast<int64_t>(L->nsuper);
    for (int64_t s = 0; s < nsuper; ++s) {
      int64_t ld = pi[s + 1] - pi[s];
      int64_t ncols = super[s + 1] - super[s];
      for (int64_t c = 0; c < ncols; ++c)
        acc += 2.0 * std::log(xv[px[s] + c * ld + c]);
    }
  } else {
    // Simplicial: the first stored entry of each column is the diagonal.
    const int32_t* p = static_cast<const int32_t*>(L->p);
    const double* xv = static_cast<const double*>(L->x);
    for (int64_t j = 0; j < n; ++j)
      acc += (L->is_ll ? 2.0 : 1.0) * std::log(xv[p[j]]);
  }
  return acc;
}

// Scatter COO values into A->x and (re)factorize, skipping the numeric
// factorization entirely when the values are identical to the previous call.
static bool factorize_locked(PatternEntry* e, const double* Ax, int64_t nnz,
                             std::string* err) {
  if (e->factored && e->last_Ax.size() == static_cast<size_t>(nnz) &&
      std::memcmp(e->last_Ax.data(), Ax, nnz * sizeof(double)) == 0)
    return true;

  double* Axv = static_cast<double*>(e->A->x);
  int64_t nnz_csc = static_cast<int64_t>(e->A->nzmax);
  std::memset(Axv, 0, nnz_csc * sizeof(double));
  for (int64_t k = 0; k < nnz; ++k)
    if (e->pos[k] >= 0) Axv[e->pos[k]] += Ax[k];

  g_common.status = CHOLMOD_OK;
  cholmod_factorize(e->A, e->L, &g_common);
  if (g_common.status == CHOLMOD_NOT_POSDEF) {
    e->factored = false;
    e->last_Ax.clear();
    *err = "sparsax: matrix is not positive definite (failure at column " +
           std::to_string(e->L->minor) + ")";
    return false;
  }
  if (g_common.status < CHOLMOD_OK) {
    e->factored = false;
    e->last_Ax.clear();
    *err = "sparsax: cholmod_factorize failed (status " +
           std::to_string(g_common.status) + ")";
    return false;
  }
  double ld = factor_logdet(e->L);
  if (!std::isfinite(ld)) {
    // Simplicial LDL' accepts indefinite matrices; the final_ll conversion
    // turns negative pivots into NaNs, which land here.
    e->factored = false;
    e->last_Ax.clear();
    *err = "sparsax: matrix is not positive definite";
    return false;
  }
  e->last_Ax.assign(Ax, Ax + nnz);
  e->logdet = ld;
  e->factored = true;
  e->factor_epoch++;  // invalidates any cached simplicial copy (Lldl)
  g_num_factorizations++;
  return true;
}

// Populate (or fetch) a NumSlot for the given Ax values: returns a copy of
// the base factor for these values. On a cache hit, the existing slot is
// reused (no refactoring); on a miss, the base is factored and its factor is
// copied into a new slot. Caller must hold g_mutex.
static PatternEntry::NumSlot* get_or_make_num_slot_locked(
    PatternEntry* e, const double* Ax, int64_t nnz, std::string* err) {
  uint64_t h = fnv1a(Ax, nnz * sizeof(double), 14695981039346656037ULL);
  for (auto& s : e->num_cache)
    if (s.hash == h && s.Ax.size() == static_cast<size_t>(nnz) &&
        std::memcmp(s.Ax.data(), Ax, nnz * sizeof(double)) == 0)
      return &s;
  if (!factorize_locked(e, Ax, nnz, err)) return nullptr;
  cholmod_factor* Lf = cholmod_copy_factor(e->L, &g_common);
  if (!Lf) {
    *err = "sparsax: cholmod_copy_factor failed (status " +
           std::to_string(g_common.status) + ")";
    return nullptr;
  }
  PatternEntry::NumSlot slot;
  slot.hash = h;
  slot.Ax.assign(Ax, Ax + nnz);
  slot.Lf = Lf;
  slot.logdet = e->logdet;
  e->num_cache.push_back(std::move(slot));
  return &e->num_cache.back();
}

// Decode a Factor token buffer (int64[2] per token, or a batch thereof) into
// a list of FactorRef. The buffer layout is [key0, slot0, key1, slot1, ...].
static std::vector<FactorRef> decode_factor_refs(const int64_t* data,
                                                 int64_t count) {
  std::vector<FactorRef> refs(count);
  for (int64_t i = 0; i < count; ++i) {
    refs[i].pattern_key = static_cast<uint64_t>(data[i * 2]);
    refs[i].slot = data[i * 2 + 1];
  }
  return refs;
}

// Resolve a FactorRef to its (PatternEntry*, NumSlot*). Caller must hold g_mutex
// (the pattern registry lock); the factor-ref map is guarded by
// g_factor_ref_mtx. Returns nullptr on a stale/invalid token and sets `err`.
static PatternEntry::NumSlot* resolve_factor_ref_locked(
    const FactorRef& ref, PatternEntry** entry_out, std::string* err) {
  if (ref.pattern_key == 0 || ref.slot < 0) {
    *err = "sparsax: invalid factor token (empty)";
    return nullptr;
  }
  std::lock_guard<std::mutex> lk(g_factor_ref_mtx);
  auto it = g_factor_pattern_keys.find(ref.pattern_key);
  if (it == g_factor_pattern_keys.end()) {
    *err = "sparsax: stale factor token (pattern cleared)";
    return nullptr;
  }
  PatternEntry* e = it->second;
  if (ref.slot >= static_cast<int64_t>(e->num_cache.size())) {
    *err = "sparsax: stale factor token (slot out of range)";
    return nullptr;
  }
  if (entry_out) *entry_out = e;
  return &e->num_cache[ref.slot];
}

// Look up (or build) the pattern entry and (re)factorize it, throwing a Python
// exception on failure. For the numpy-callable core functions (used by non-JAX
// frontends such as the PyTensor Ops), which report errors via exceptions
// rather than ffi::Error. Caller must hold g_mutex.
static PatternEntry* prepare_entry_locked(const int32_t* Ai, const int32_t* Aj,
                                          const double* Ax, int64_t nnz,
                                          int64_t n) {
  std::string err;
  PatternEntry* e = get_or_create_entry_locked(Ai, Aj, nnz, n, &err);
  if (!e) throw std::runtime_error(err);
  if (!factorize_locked(e, Ax, nnz, &err)) throw std::runtime_error(err);
  return e;
}

// Solve L L' x = b for one right-hand side block using factor L and the
// entry's persistent solve2 workspaces. bdata/xdata are row-major (JAX
// layout); CHOLMOD is column-major, so multi-RHS blocks are transposed
// through `scratch` (reused across calls).
static ffi::Error solve_one_locked(PatternEntry* e, cholmod_factor* L, int mode,
                                   const double* bdata, int64_t n, int64_t nrhs,
                                   double* xdata, std::vector<double>* scratch) {
  const double* bcol = bdata;
  if (nrhs > 1) {
    scratch->resize(n * nrhs);
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < nrhs; ++j)
        (*scratch)[i + j * n] = bdata[i * nrhs + j];
    bcol = scratch->data();
  }

  cholmod_dense B;
  std::memset(&B, 0, sizeof(B));
  B.nrow = n;
  B.ncol = nrhs;
  B.nzmax = n * nrhs;
  B.d = n;
  B.x = const_cast<double*>(bcol);
  B.xtype = CHOLMOD_REAL;
  B.dtype = CHOLMOD_DOUBLE;

  // cholmod_solve2 reuses the X/Y/E workspaces held in the entry, avoiding
  // cholmod_solve's per-call allocations.
  int ok = cholmod_solve2(mode, L, &B, nullptr, &e->Xwork, nullptr, &e->Ywork,
                          &e->Ework, &g_common);
  if (!ok || !e->Xwork || g_common.status < CHOLMOD_OK)
    return ffi::Error::Internal("sparsax: cholmod_solve failed (status " +
                                std::to_string(g_common.status) + ")");

  const double* Xx = static_cast<const double*>(e->Xwork->x);
  if (nrhs == 1) {
    std::memcpy(xdata, Xx, n * sizeof(double));
  } else {
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < nrhs; ++j) xdata[i * nrhs + j] = Xx[i + j * n];
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// solve handler:  (Ai, Aj, Ax, b; mode) -> x with x.shape == b.shape
// b may be (n,) or (n, nrhs). `mode` maps directly to CHOLMOD_A etc.
// ---------------------------------------------------------------------------

static ffi::Error SolveF64Impl(ffi::Buffer<ffi::S32> Ai,
                               ffi::Buffer<ffi::S32> Aj,
                               ffi::Buffer<ffi::F64> Ax,
                               ffi::Buffer<ffi::F64> b,
                               ffi::ResultBuffer<ffi::F64> x, int64_t mode) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("sparsax: b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("sparsax: invalid solve mode");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  std::vector<double> scratch;
  return solve_one_locked(e, e->L, static_cast<int>(mode), b.typed_data(), n,
                          nrhs, x->typed_data(), &scratch);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSolveF64, SolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Attr<int64_t>("mode"));

// ---------------------------------------------------------------------------
// batched solve handler:  (Ai, Aj, Ax[B,nnz], b[B,n(,nrhs)]; mode) -> x[B,...]
// One FFI call solves a whole batch that shares a sparsity pattern, refactoring
// per element and reusing the cached analysis + solve workspaces. This is what
// jax.vmap(solve) lowers to (see the custom_vmap rule in Python), so the batch
// loop runs in C++ rather than as XLA per-iteration dispatch. Ai/Aj stay
// unbatched — the pattern is identical across the batch.
// ---------------------------------------------------------------------------

static ffi::Error SolveBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                      ffi::Buffer<ffi::S32> Aj,
                                      ffi::Buffer<ffi::F64> Ax,
                                      ffi::Buffer<ffi::F64> b,
                                      ffi::ResultBuffer<ffi::F64> x,
                                      int64_t mode) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return ffi::Error::InvalidArgument(
        "sparsax: batched b must be 2D or 3D");
  int64_t batch = bdims[0];
  int64_t n = bdims[1];
  int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[0] != batch || axdims[1] != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: batched Ax must have shape (batch, nnz) matching b");
  if (static_cast<int64_t>(Aj.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj must have the same length");
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("sparsax: invalid solve mode");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  std::vector<double> scratch;
  for (int64_t s = 0; s < batch; ++s) {
    if (!factorize_locked(e, Axd + s * nnz, nnz, &err))
      return ffi::Error::Internal(err);
    ffi::Error r =
        solve_one_locked(e, e->L, static_cast<int>(mode), bd + s * bstride, n,
                         nrhs, xd + s * bstride, &scratch);
    if (r.failure()) return r;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSolveBatchedF64, SolveBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b  [B,...]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x  [B,...]
                                  .Attr<int64_t>("mode"));

// ---------------------------------------------------------------------------
// logdet handler:  (Ai, Aj, Ax; n) -> scalar log|A|
// Shares the factorization cache with solve, so a solve followed by a logdet
// with identical values factorizes only once.
// ---------------------------------------------------------------------------

static ffi::Error LogdetF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  out->typed_data()[0] = e->logdet;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetF64, LogdetF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// factor-solve handler: factor A once, then apply K independent solve *chains*
// from that single factor, plus an optional logdet.
//
//   args:  Ai, Aj, Ax, b_0, ..., b_{K-1}   (each b_k is (n) or (n, nrhs_k))
//   rets:  x_0, ..., x_{K-1}[, logdet]      (x_k has the same shape as b_k)
//   attrs: mode_chain  — all K mode sequences flattened end to end
//          chain_lens  — length K; chain_lens[k] modes belong to b_k
//          want_logdet — 0/1; when 1 the final ret is the scalar log|A|
//          n           — matrix dimension (needed when K == 0)
//
// Each chain is applied left to right: x_k = m_last(... m_1(m_0(b_k))), where
// each m is a CHOLMOD solve system (MODE_*). This expresses e.g. the Gibbs
// Gaussian step from one factorization: mean = A^{-1} b via chain [MODE_A], and
// a draw's factor part P' L^{-T} z via chain [MODE_LT, MODE_PT].
// ---------------------------------------------------------------------------

// Apply all K solve chains for one already-factored system. bptr/xptr hold the
// K input/output row-major blocks; tmpA/tmpB are reused ping-pong scratch.
static ffi::Error apply_solves_locked(
    PatternEntry* e, cholmod_factor* L, int64_t n,
    const std::vector<const double*>& bptr, const std::vector<int64_t>& nrhs,
    const int64_t* mode_chain, const int64_t* chain_lens, int64_t K,
    const std::vector<double*>& xptr, std::vector<double>* scratch,
    std::vector<double>* tmpA, std::vector<double>* tmpB) {
  int64_t coff = 0;
  for (int64_t k = 0; k < K; ++k) {
    int64_t len = chain_lens[k];
    int64_t w = nrhs[k];
    if (len == 0) {
      // Empty chain: identity copy b_k -> x_k.
      std::memcpy(xptr[k], bptr[k], n * w * sizeof(double));
      continue;
    }
    const double* in = bptr[k];
    for (int64_t s = 0; s < len; ++s) {
      int64_t mode = mode_chain[coff + s];
      if (mode < 0 || mode > 8)
        return ffi::Error::InvalidArgument("sparsax: invalid solve mode");
      // Final step writes into the output buffer; intermediates ping-pong.
      double* out;
      if (s == len - 1)
        out = xptr[k];
      else {
        std::vector<double>* t = (s % 2 == 0) ? tmpA : tmpB;
        t->resize(n * w);
        out = t->data();
      }
      ffi::Error r = solve_one_locked(e, L, static_cast<int>(mode), in, n, w,
                                      out, scratch);
      if (r.failure()) return r;
      in = out;
    }
    coff += len;
  }
  return ffi::Error::Success();
}

static ffi::Error FactorSolveF64Impl(
    ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj, ffi::Buffer<ffi::F64> Ax,
    ffi::RemainingArgs rhs, ffi::RemainingRets rets,
    ffi::Span<const int64_t> mode_chain, ffi::Span<const int64_t> chain_lens,
    int64_t want_logdet, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");
  int64_t K = static_cast<int64_t>(chain_lens.size());
  if (static_cast<int64_t>(rhs.size()) != K)
    return ffi::Error::InvalidArgument(
        "sparsax: number of right-hand sides must match chain_lens");
  if (static_cast<int64_t>(rets.size()) != K + (want_logdet ? 1 : 0))
    return ffi::Error::InvalidArgument(
        "sparsax: number of results must match rhs count (+ logdet)");

  std::vector<const double*> bptr(K);
  std::vector<double*> xptr(K);
  std::vector<int64_t> nrhs(K);
  for (int64_t k = 0; k < K; ++k) {
    auto b = rhs.get<ffi::Buffer<ffi::F64>>(k);
    auto x = rets.get<ffi::Buffer<ffi::F64>>(k);
    if (b.has_error()) return b.error();
    if (x.has_error()) return x.error();
    auto bdims = b->dimensions();
    if (bdims.size() < 1 || bdims.size() > 2 || bdims[0] != n)
      return ffi::Error::InvalidArgument(
          "sparsax: each rhs must be (n) or (n, nrhs) with matching n");
    bptr[k] = b->typed_data();
    xptr[k] = (*x)->typed_data();
    nrhs[k] = bdims.size() == 2 ? bdims[1] : 1;
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  std::vector<double> scratch, tmpA, tmpB;
  ffi::Error r = apply_solves_locked(e, e->L, n, bptr, nrhs, mode_chain.begin(),
                                     chain_lens.begin(), K, xptr, &scratch, &tmpA,
                                     &tmpB);
  if (r.failure()) return r;

  if (want_logdet) {
    auto ld = rets.get<ffi::Buffer<ffi::F64>>(K);
    if (ld.has_error()) return ld.error();
    (*ld)->typed_data()[0] = e->logdet;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorSolveF64, FactorSolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .RemainingArgs()                // b_0..b_{K-1}
                                  .RemainingRets()                // x_0..[,logdet]
                                  .Attr<ffi::Span<const int64_t>>("mode_chain")
                                  .Attr<ffi::Span<const int64_t>>("chain_lens")
                                  .Attr<int64_t>("want_logdet")
                                  .Attr<int64_t>("n"));

// Batched factor-solve: leading axis is the batch. Ax is (B, nnz); each rhs is
// (B, n[, nrhs_k]); results mirror that. The system is factored exactly ONCE
// per batch element, then every chain for that element is solved from it — so a
// vmapped multi-solve costs one factorization per element, not one per chain.
static ffi::Error FactorSolveBatchedF64Impl(
    ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj, ffi::Buffer<ffi::F64> Ax,
    ffi::RemainingArgs rhs, ffi::RemainingRets rets,
    ffi::Span<const int64_t> mode_chain, ffi::Span<const int64_t> chain_lens,
    int64_t want_logdet, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[1] != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: batched Ax must have shape (batch, nnz)");
  int64_t batch = axdims[0];
  int64_t K = static_cast<int64_t>(chain_lens.size());
  if (static_cast<int64_t>(rhs.size()) != K)
    return ffi::Error::InvalidArgument(
        "sparsax: number of right-hand sides must match chain_lens");
  if (static_cast<int64_t>(rets.size()) != K + (want_logdet ? 1 : 0))
    return ffi::Error::InvalidArgument(
        "sparsax: number of results must match rhs count (+ logdet)");

  // Per-rhs element strides and per-batch base pointers.
  std::vector<const double*> bbase(K);
  std::vector<double*> xbase(K);
  std::vector<int64_t> nrhs(K), bstride(K);
  for (int64_t k = 0; k < K; ++k) {
    auto b = rhs.get<ffi::Buffer<ffi::F64>>(k);
    auto x = rets.get<ffi::Buffer<ffi::F64>>(k);
    if (b.has_error()) return b.error();
    if (x.has_error()) return x.error();
    auto bdims = b->dimensions();
    if (bdims.size() < 2 || bdims.size() > 3 || bdims[0] != batch ||
        bdims[1] != n)
      return ffi::Error::InvalidArgument(
          "sparsax: each batched rhs must be (batch, n[, nrhs])");
    nrhs[k] = bdims.size() == 3 ? bdims[2] : 1;
    bstride[k] = n * nrhs[k];
    bbase[k] = b->typed_data();
    xbase[k] = (*x)->typed_data();
  }
  double* ldbase = nullptr;
  if (want_logdet) {
    auto ld = rets.get<ffi::Buffer<ffi::F64>>(K);
    if (ld.has_error()) return ld.error();
    ldbase = (*ld)->typed_data();
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  std::vector<double> scratch, tmpA, tmpB;
  std::vector<const double*> bptr(K);
  std::vector<double*> xptr(K);
  for (int64_t s = 0; s < batch; ++s) {
    if (!factorize_locked(e, Axd + s * nnz, nnz, &err))
      return ffi::Error::Internal(err);
    for (int64_t k = 0; k < K; ++k) {
      bptr[k] = bbase[k] + s * bstride[k];
      xptr[k] = xbase[k] + s * bstride[k];
    }
    ffi::Error r =
        apply_solves_locked(e, e->L, n, bptr, nrhs, mode_chain.begin(),
                            chain_lens.begin(), K, xptr, &scratch, &tmpA, &tmpB);
    if (r.failure()) return r;
    if (want_logdet) ldbase[s] = e->logdet;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorSolveBatchedF64,
                              FactorSolveBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .RemainingArgs()                // b_0..b_{K-1}
                                  .RemainingRets()                // x_0..[,logdet]
                                  .Attr<ffi::Span<const int64_t>>("mode_chain")
                                  .Attr<ffi::Span<const int64_t>>("chain_lens")
                                  .Attr<int64_t>("want_logdet")
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// factor / solve_factor / logdet_factor: the "hold a numeric factor open"
// primitives. A factor(A) call computes (or fetches from a value-keyed cache)
// a numeric factorization of A and returns an opaque token; solve_factor and
// logdet_factor consume that token for an unbounded sequence of solves + a
// logdet, without re-hashing Ax or refactoring. This is the structure
// factor_solve cannot express: a recurrence V_{j+1} = A^{-1}(G V_j) that needs
// the previous solve's output as the next solve's input.
//
// The token is an int64[2] pair (pattern_key, slot) carried as XLA state.
//   pattern_key — an integer assigned to the PatternEntry on first factor()
//                 call, registered in g_factor_pattern_keys so the token can
//                 be resolved back to its pattern after the XLA buffers are
//                 dropped (e.g. across fori_loop iterations).
//   slot        — index into PatternEntry::num_cache, the value-keyed numeric
//                 factor cache. A hit reuses an existing factor copy; a miss
//                 factors A, copies L into a new slot, and returns its index.
//
// Under vmap, factor produces one token per batch element (one factorization
// per element, matching factor_solve's batch semantics). solve_factor's vmap
// over b reuses a single token — the key win for the Krylov recurrence, where
// one factor serves m+1 solves.
// ---------------------------------------------------------------------------

static ffi::Error FactorF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::S64> token, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  PatternEntry::NumSlot* slot =
      get_or_make_num_slot_locked(e, Ax.typed_data(), nnz, &err);
  if (!slot) return ffi::Error::Internal(err);

  // Assign a pattern key (the pattern pointer itself, unique while the
  // pattern is alive) and register it so tokens can be resolved later. The
  // map is cleared together with the pattern registry in clear_cache.
  uint64_t key = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(e));
  {
    std::lock_guard<std::mutex> lk(g_factor_ref_mtx);
    g_factor_pattern_keys.emplace(key, e);
  }
  int64_t* td = token->typed_data();
  td[0] = static_cast<int64_t>(key);
  td[1] = static_cast<int64_t>(slot - &e->num_cache[0]);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorF64, FactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Attr<int64_t>("n"));

static ffi::Error SolveFactorF64Impl(ffi::Buffer<ffi::S64> token,
                                     ffi::Buffer<ffi::F64> b,
                                     ffi::ResultBuffer<ffi::F64> x,
                                     int64_t mode) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("sparsax: b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  if (static_cast<int64_t>(token.element_count()) != 2)
    return ffi::Error::InvalidArgument("sparsax: factor token must be int64[2]");
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("sparsax: invalid solve mode");

  std::vector<FactorRef> refs = decode_factor_refs(token.typed_data(), 1);

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e = nullptr;
  PatternEntry::NumSlot* slot = resolve_factor_ref_locked(refs[0], &e, &err);
  if (!slot) return ffi::Error::InvalidArgument(err);

  std::vector<double> scratch;
  return solve_one_locked(e, slot->Lf, static_cast<int>(mode), b.typed_data(), n,
                          nrhs, x->typed_data(), &scratch);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSolveFactorF64, SolveFactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Attr<int64_t>("mode"));

// Batched solve_factor: the leading axis is the batch. Two vmap shapes are
// supported:
//   (a) vmap over b against ONE factor: token is unbatched int64[2], b is
//       (B, n[, nrhs]) — the key Krylov-recurrence case (m+1 solves, one
//       factor).
//   (b) vmap over (token, b) — one factor per batch element: token is
//       (B, 2), b is (B, n[, nrhs]).
static ffi::Error SolveFactorBatchedF64Impl(ffi::Buffer<ffi::S64> token,
                                            ffi::Buffer<ffi::F64> b,
                                            ffi::ResultBuffer<ffi::F64> x,
                                            int64_t mode) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return ffi::Error::InvalidArgument(
        "sparsax: batched b must be 2D or 3D");
  int64_t batch = bdims[0];
  int64_t n = bdims[1];
  int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  auto tdims = token.dimensions();
  bool token_batched = (tdims.size() == 2 && tdims[0] == batch && tdims[1] == 2);
  bool token_scalar = (tdims.size() == 1 && tdims[0] == 2);
  if (!token_batched && !token_scalar)
    return ffi::Error::InvalidArgument(
        "sparsax: token must be int64[2] or (batch, 2) matching b");
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("sparsax: invalid solve mode");

  std::vector<FactorRef> refs;
  if (token_scalar) {
    refs = decode_factor_refs(token.typed_data(), 1);
  } else {
    refs = decode_factor_refs(token.typed_data(), batch);
  }

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  std::vector<double> scratch;
  for (int64_t s = 0; s < batch; ++s) {
    const FactorRef& ref = token_scalar ? refs[0] : refs[s];
    std::string err;
    PatternEntry* e = nullptr;
    PatternEntry::NumSlot* slot = resolve_factor_ref_locked(ref, &e, &err);
    if (!slot) return ffi::Error::InvalidArgument(err);
    ffi::Error r = solve_one_locked(e, slot->Lf, static_cast<int>(mode),
                                    bd + s * bstride, n, nrhs,
                                    xd + s * bstride, &scratch);
    if (r.failure()) return r;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSolveFactorBatchedF64,
                              SolveFactorBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [2] or [B,2]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b [B,...]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x [B,...]
                                  .Attr<int64_t>("mode"));

static ffi::Error LogdetFactorF64Impl(ffi::Buffer<ffi::S64> token,
                                      ffi::ResultBuffer<ffi::F64> out) {
  if (static_cast<int64_t>(token.element_count()) != 2)
    return ffi::Error::InvalidArgument("sparsax: factor token must be int64[2]");

  std::vector<FactorRef> refs = decode_factor_refs(token.typed_data(), 1);

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry::NumSlot* slot = resolve_factor_ref_locked(refs[0], nullptr, &err);
  if (!slot) return ffi::Error::InvalidArgument(err);
  out->typed_data()[0] = slot->logdet;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetFactorF64, LogdetFactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  );

// Batched logdet_factor: vmap over a batch of tokens (one logdet per chain).
static ffi::Error LogdetFactorBatchedF64Impl(
    ffi::Buffer<ffi::S64> token, ffi::ResultBuffer<ffi::F64> out) {
  auto tdims = token.dimensions();
  if (tdims.size() != 2 || tdims[1] != 2)
    return ffi::Error::InvalidArgument(
        "sparsax: batched token must be (batch, 2)");
  int64_t batch = tdims[0];
  std::vector<FactorRef> refs = decode_factor_refs(token.typed_data(), batch);

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  double* od = out->typed_data();
  for (int64_t s = 0; s < batch; ++s) {
    std::string err;
    PatternEntry::NumSlot* slot = resolve_factor_ref_locked(refs[s], nullptr, &err);
    if (!slot) return ffi::Error::InvalidArgument(err);
    od[s] = slot->logdet;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetFactorBatchedF64,
                              LogdetFactorBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [B,2]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet [B]
                                  );

// ---------------------------------------------------------------------------
// Simplicial LDL' base factor (Lldl). cholmod_updown and the selected-inverse
// recurrence both need an LDL' (not LL', not supernodal) factor. Lldl is a
// persistent simplicial LDL' copy of the base factor L, rebuilt only when the
// base is refactored (tracked by epoch), so the LL'->LDL' conversion is paid
// once per base change rather than once per call.
// ---------------------------------------------------------------------------

static ffi::Error ensure_ldl_locked(PatternEntry* e) {
  if (e->Lldl && e->ldl_epoch == e->factor_epoch) return ffi::Error::Success();
  if (e->Lldl) cholmod_free_factor(&e->Lldl, &g_common);
  e->Lldl = cholmod_copy_factor(e->L, &g_common);
  if (!e->Lldl)
    return ffi::Error::Internal("sparsax: cholmod_copy_factor failed");
  if (!cholmod_change_factor(CHOLMOD_REAL, /*to_ll=*/0, /*to_super=*/0,
                             /*to_packed=*/1, /*to_monotonic=*/1, e->Lldl,
                             &g_common))
    return ffi::Error::Internal("sparsax: cholmod_change_factor failed");
  e->ldl_epoch = e->factor_epoch;
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// selected-inverse handler:  (Ai, Aj, Ax; n) -> z[nnz]
//     z[k] = (A^{-1})[Ai[k], Aj[k]]
//
// Computes the entries of A^{-1} that lie in the pattern of A (a subset of the
// factor's fill pattern) via Takahashi's recurrence on the LDL' factor, without
// forming the dense inverse. This is exactly what logdet's reverse-mode rule
// needs: d log|A| / d A = A^{-1}, so d log|A| / d Ax[k] is (A^{-1}) at the COO
// position (Ai[k], Aj[k]) (see the custom_vjp in Python for the symmetric
// doubling of off-diagonal entries).
//
// Recurrence (A = P' L D L' P, L unit lower-triangular, D diagonal; work in the
// permuted space, then map back through Perm). Processing columns j = n-1..0,
// with below-diagonal pattern rows r_0<...<r_{p-1} of column j and l_b=L[r_b,j]:
//     Z[r_a, j] = - sum_b l_b * Z[r_a, r_b]           (selected inverse Z)
//     Z[j, j]   = 1/D[j] - sum_a l_a * Z[r_a, j]
// The pairs (r_a, r_b) are all in the factor's pattern (fill closure), so the
// recurrence stays within the stored structure.
// ---------------------------------------------------------------------------

static ffi::Error selinv_locked(PatternEntry* e, int64_t nnz, double* z) {
  ffi::Error r = ensure_ldl_locked(e);
  if (r.failure()) return r;
  cholmod_factor* L = e->Lldl;
  int64_t n = static_cast<int64_t>(L->n);
  const int32_t* Lp = static_cast<const int32_t*>(L->p);
  const int32_t* Li = static_cast<const int32_t*>(L->i);
  const double* Lx = static_cast<const double*>(L->x);
  const int32_t* Perm = static_cast<const int32_t*>(L->Perm);

  std::vector<double> Z(Lp[n], 0.0);  // selected inverse, same structure as L
  std::vector<double> work(n, 0.0);   // dense scatter workspace (kept zeroed)
  std::vector<double> acc;

  for (int64_t j = n - 1; j >= 0; --j) {
    int64_t ps = Lp[j], pe = Lp[j + 1];
    int64_t cnt = pe - ps - 1;  // below-diagonal entries of column j
    // acc[a] = sum_b L[r_b, j] * Z[r_a, r_b], accumulated symmetrically so that
    // each column of Z is scattered into `work` exactly once.
    acc.assign(cnt, 0.0);
    for (int64_t b = 0; b < cnt; ++b) {
      int64_t c = Li[ps + 1 + b];  // r_b: column of Z to scatter
      double lb = Lx[ps + 1 + b];
      int64_t cs = Lp[c], ce = Lp[c + 1];
      work[c] = Z[cs];  // Z[c, c]
      for (int64_t t = cs + 1; t < ce; ++t) work[Li[t]] = Z[t];
      for (int64_t a = b; a < cnt; ++a) {
        double za = work[Li[ps + 1 + a]];  // Z[r_a, r_b], r_a >= r_b (stored)
        acc[a] += lb * za;
        if (a != b) acc[b] += Lx[ps + 1 + a] * za;  // symmetric partner
      }
      work[c] = 0.0;  // restore workspace to all-zero
      for (int64_t t = cs + 1; t < ce; ++t) work[Li[t]] = 0.0;
    }
    double diag = 1.0 / Lx[ps];  // 1/D[j]
    for (int64_t a = 0; a < cnt; ++a) {
      double zaj = -acc[a];
      Z[ps + 1 + a] = zaj;
      diag -= Lx[ps + 1 + a] * zaj;
    }
    Z[ps] = diag;
  }

  // Z is A^{-1} in the permuted space: Z[u,v] = (A^{-1})[Perm[u], Perm[v]].
  // Map each COO position back through the inverse permutation.
  std::vector<int32_t> iperm(n);
  for (int64_t k = 0; k < n; ++k) iperm[Perm[k]] = static_cast<int32_t>(k);
  const int32_t* Ai = e->Ai.data();
  const int32_t* Aj = e->Aj.data();
  for (int64_t k = 0; k < nnz; ++k) {
    int64_t u = iperm[Ai[k]], v = iperm[Aj[k]];
    if (u < v) std::swap(u, v);  // stored entry lives in column min, row max
    double val = 0.0;
    if (u == v) {
      val = Z[Lp[v]];
    } else {
      for (int64_t t = Lp[v] + 1, te = Lp[v + 1]; t < te; ++t)
        if (Li[t] == u) {
          val = Z[t];
          break;
        }
    }
    z[k] = val;
  }
  return ffi::Error::Success();
}

static ffi::Error SelinvF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::F64> z, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  return selinv_locked(e, nnz, z->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSelinvF64, SelinvF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // z [nnz]
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// updown-solve handler:  (Ai, Aj, Ax, C[n,k], b[n(,nrhs)]; mode, downdate)
//     -> x solving (A ± C C') x = b, and out_logdet = log|A ± C C'|.
//
// The base matrix A is factored once (cached). Each call rebuilds a simplicial
// LDL' copy of that factor, applies cholmod_updown for the rank-k modification
// C C', and solves — much cheaper than refactoring A ± C C' from scratch when A
// is held fixed and only the low-rank term C varies. `update=1` adds C C',
// `downdate=1` (downdate flag) subtracts it. The base cached factor is never
// mutated, so the op is a pure function of its inputs.
// C is dense (n, k), row-major (JAX layout).
// ---------------------------------------------------------------------------

static ffi::Error UpdownSolveF64Impl(ffi::Buffer<ffi::S32> Ai,
                                     ffi::Buffer<ffi::S32> Aj,
                                     ffi::Buffer<ffi::F64> Ax,
                                     ffi::Buffer<ffi::F64> C,
                                     ffi::Buffer<ffi::F64> b,
                                     ffi::ResultBuffer<ffi::F64> x,
                                     ffi::ResultBuffer<ffi::F64> out_logdet,
                                     int64_t mode, int64_t downdate) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("sparsax: b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");
  auto cdims = C.dimensions();
  if (cdims.size() != 2 || cdims[0] != n)
    return ffi::Error::InvalidArgument(
        "sparsax: C must have shape (n, k) matching b's dimension n");
  int64_t k = cdims[1];
  if (mode < 0 || mode > 8)
    return ffi::Error::InvalidArgument("sparsax: invalid solve mode");

  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();

  std::string err;
  PatternEntry* e =
      get_or_create_entry_locked(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!e) return ffi::Error::InvalidArgument(err);
  if (!factorize_locked(e, Ax.typed_data(), nnz, &err))
    return ffi::Error::Internal(err);

  // cholmod_updown requires a simplicial LDL' factor (it cannot update a
  // supernodal or LL' factor). Lldl is a persistent simplicial LDL' copy of the
  // base factor (rebuilt only on refactor); the per-call working copy Lupd
  // (which updown mutates) is a plain simplicial->simplicial copy of Lldl,
  // leaving the base factor pristine.
  ffi::Error ldl_err = ensure_ldl_locked(e);
  if (ldl_err.failure()) return ldl_err;

  if (e->Lupd) cholmod_free_factor(&e->Lupd, &g_common);
  e->Lupd = cholmod_copy_factor(e->Lldl, &g_common);
  if (!e->Lupd)
    return ffi::Error::Internal("sparsax: cholmod_copy_factor failed");

  // Build sparse C (n x k) from the dense, row-major input via a column-major
  // temporary. cholmod_updown works in the factor's permuted space
  // (L D L' = P A P'), so C's rows must be permuted by L->Perm — permuted row
  // kk takes original row Perm[kk]. cholmod_solve then applies P/P' itself, so
  // the returned solution is in the original ordering. dense_to_sparse drops
  // explicit zeros, so a sparse update column yields a sparse C and cheap updown.
  cholmod_dense Cd;
  std::memset(&Cd, 0, sizeof(Cd));
  Cd.nrow = n;
  Cd.ncol = k;
  Cd.nzmax = n * k;
  Cd.d = n;
  Cd.xtype = CHOLMOD_REAL;
  Cd.dtype = CHOLMOD_DOUBLE;
  std::vector<double> Ccol(n * k);
  const double* Cin = C.typed_data();
  const int32_t* Perm = static_cast<const int32_t*>(e->Lupd->Perm);
  for (int64_t kk = 0; kk < n; ++kk) {
    int64_t orig = Perm[kk];
    for (int64_t j = 0; j < k; ++j) Ccol[kk + j * n] = Cin[orig * k + j];
  }
  Cd.x = Ccol.data();

  cholmod_sparse* Cs = cholmod_dense_to_sparse(&Cd, /*values=*/1, &g_common);
  if (!Cs)
    return ffi::Error::Internal("sparsax: cholmod_dense_to_sparse failed");

  int ok = cholmod_updown(downdate ? 0 : 1, Cs, e->Lupd, &g_common);
  cholmod_free_sparse(&Cs, &g_common);
  if (!ok || g_common.status < CHOLMOD_OK)
    return ffi::Error::Internal("sparsax: cholmod_updown failed (status " +
                                std::to_string(g_common.status) + ")");

  double ld = factor_logdet(e->Lupd);
  if (!std::isfinite(ld))
    return ffi::Error::Internal(
        "sparsax: updated matrix A ± C C' is not positive definite");
  out_logdet->typed_data()[0] = ld;

  std::vector<double> scratch;
  return solve_one_locked(e, e->Lupd, static_cast<int>(mode), b.typed_data(), n,
                          nrhs, x->typed_data(), &scratch);
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodUpdownSolveF64, UpdownSolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()   // C [n,k]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  .Attr<int64_t>("mode")
                                  .Attr<int64_t>("downdate"));

// ---------------------------------------------------------------------------
// Numpy-callable core (framework-agnostic, no XLA). These call the same cached
// CHOLMOD core as the FFI handlers but take and return plain numpy arrays, so
// non-JAX frontends — the PyTensor Ops in sparsax.pytensor — can reach the
// solver and its selected-inverse gradient without going through XLA. Errors
// are raised as Python exceptions.
// ---------------------------------------------------------------------------

using ArrI32 =
    nb::ndarray<const int32_t, nb::ndim<1>, nb::c_contig, nb::device::cpu>;
using ArrF64 = nb::ndarray<const double, nb::c_contig, nb::device::cpu>;

static void check_coo_np(const ArrI32& Ai, const ArrI32& Aj, const ArrF64& Ax,
                         int64_t nnz) {
  if (static_cast<int64_t>(Aj.shape(0)) != nnz || Ax.ndim() != 1 ||
      static_cast<int64_t>(Ax.shape(0)) != nnz)
    throw std::runtime_error(
        "sparsax: Ai, Aj, Ax must be 1D with the same length");
}

static nb::ndarray<nb::numpy, double> solve_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax,
                                               ArrF64 b, int64_t mode) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  if (b.ndim() < 1 || b.ndim() > 2)
    throw std::runtime_error("sparsax: b must be 1D or 2D");
  if (mode < 0 || mode > 8)
    throw std::runtime_error("sparsax: invalid solve mode");
  int64_t n = static_cast<int64_t>(b.shape(0));
  int64_t nrhs = b.ndim() == 2 ? static_cast<int64_t>(b.shape(1)) : 1;

  double* out = new double[static_cast<size_t>(n) * nrhs];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_started_locked();
    PatternEntry* e =
        prepare_entry_locked(Ai.data(), Aj.data(), Ax.data(), nnz, n);
    std::vector<double> scratch;
    ffi::Error r = solve_one_locked(e, e->L, static_cast<int>(mode), b.data(), n,
                                    nrhs, out, &scratch);
    if (r.failure()) throw std::runtime_error("sparsax: solve failed");
  }
  if (b.ndim() == 2)
    return nb::ndarray<nb::numpy, double>(
        out, {static_cast<size_t>(n), static_cast<size_t>(nrhs)}, owner);
  return nb::ndarray<nb::numpy, double>(out, {static_cast<size_t>(n)}, owner);
}

static double logdet_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  std::lock_guard<std::mutex> lock(g_mutex);
  ensure_started_locked();
  PatternEntry* e =
      prepare_entry_locked(Ai.data(), Aj.data(), Ax.data(), nnz, n);
  return e->logdet;
}

static nb::ndarray<nb::numpy, double> selinv_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax,
                                                int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  double* out = new double[nnz > 0 ? static_cast<size_t>(nnz) : 1];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_started_locked();
    PatternEntry* e =
        prepare_entry_locked(Ai.data(), Aj.data(), Ax.data(), nnz, n);
    ffi::Error r = selinv_locked(e, nnz, out);
    if (r.failure()) throw std::runtime_error("sparsax: selinv failed");
  }
  return nb::ndarray<nb::numpy, double>(out, {static_cast<size_t>(nnz)}, owner);
}

// ===========================================================================
// KLU: sparse LU for NON-symmetric matrices (e.g. A = I - rho W, where W is a
// row-standardised / asymmetric spatial weights matrix that cannot be
// d-symmetrised) as XLA FFI custom calls, mirroring the CHOLMOD path above.
//
// Same design — a global klu_common under g_mutex, a per-pattern symbolic
// cache (klu_analyze reused across value changes) — plus a small
// content-addressed LRU of *numeric* factors per pattern. The LRU is what makes
// factor-reuse survive jax.vmap: a vmapped shift-invert Krylov basis issues the
// same B distinct value-vectors (one per chain) across several sequential solve
// calls at the same rho; a single-slot value cache would thrash (each batched
// call cycles all B chains through the one slot, so every solve refactors), but
// an LRU of capacity >= B keeps every chain's factor live, so the m+1 basis
// solves cost one klu_factor per chain and m cheap klu_solves — matching numpy's
// factorisation reuse while running the whole batch in one native FFI call.
// ===========================================================================

// Numeric factors retained per pattern *per thread* (see the thread-local cache
// below). Small: within one chain a sweep touches only a couple of distinct ρ.
// Tunable from Python via set_lu_cache_size.
static size_t g_lu_cache_cap = 32;

// A numeric factor with refcounted lifetime: the shared cache and any thread
// currently solving with it both hold a shared_ptr, so evicting it from the
// cache never frees a factor a concurrent solve is still using.
struct KluNumericHolder {
  klu_numeric* num = nullptr;
  ~KluNumericHolder() {
    if (num) {
      klu_common c;
      klu_defaults(&c);
      klu_free_numeric(&num, &c);
    }
  }
};
using KluNumPtr = std::shared_ptr<KluNumericHolder>;

struct KluNumericSlot {
  uint64_t hash = 0;
  std::vector<double> Ax;   // COO values that produced `num`, for memcmp verify
  KluNumPtr num;
};

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
static std::unordered_map<uint64_t, std::vector<std::shared_ptr<KluPattern>>>
    g_klu_registry;
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
//
// There is no lu_logdet_factor: KLU factors non-symmetric matrices, and det
// is not needed for the spatial-sampler use case (the log-density's logdet
// comes from the CHOLMOD path on the symmetric precision matrix).
// ---------------------------------------------------------------------------

// A KLU factor token: an opaque reference to a slot in g_klu_factor_slots.
struct KluFactorRef {
  uint64_t slot_key = 0;  // key into g_klu_factor_slots (0 == invalid)
};

// A held factor: the pattern (for the symbolic + structure) plus the numeric
// factor. Owned by a shared_ptr so the token table and any in-flight solve
// share ownership. Cleared by clear_cache (which clears g_klu_registry first,
// then g_klu_factor_slots under g_klu_reg_mtx — same lock as the registry).
struct KluHeldFactor {
  std::shared_ptr<KluPattern> pat;
  KluNumPtr num;
};
static std::mutex g_klu_factor_mtx;  // guards g_klu_factor_slots only
static std::unordered_map<uint64_t, KluHeldFactor> g_klu_factor_slots;
static uint64_t g_klu_factor_next_key = 1;  // 0 reserved for "invalid"

// Resolve a KluFactorRef to its held factor. Caller must hold g_klu_solve_mtx
// (the numeric work is serialized; the lookup takes g_klu_factor_mtx briefly).
// Returns nullptr and sets `err` on a stale/invalid token.
static std::shared_ptr<KluHeldFactor> resolve_klu_factor_ref_locked(
    const KluFactorRef& ref, std::string* err) {
  if (ref.slot_key == 0) {
    *err = "sparsax(klu): invalid factor token (empty)";
    return nullptr;
  }
  std::lock_guard<std::mutex> lk(g_klu_factor_mtx);
  auto it = g_klu_factor_slots.find(ref.slot_key);
  if (it == g_klu_factor_slots.end()) {
    *err = "sparsax(klu): stale factor token (slot cleared)";
    return nullptr;
  }
  // Return a shared_ptr copy so the caller's solve is safe even if the token
  // is cleared concurrently (clear_cache takes g_klu_solve_mtx too, so this
  // cannot happen mid-solve, but the copy is cheap and defensive).
  return std::make_shared<KluHeldFactor>(it->second);
}

// Serializes the KLU factor/solve calls.  SuiteSparse KLU factor/solve are not
// safely *concurrent* across threads sharing a Symbolic (they gave wrong draws
// and no speedup under jax.pmap — KLU parallelises across processes, as the
// NumPy/joblib path does, not threads), so we serialize the numeric work while
// keeping the *shared* factor cache (so reuse is thread-agnostic under XLA's
// thread pool).  Arithmetic-heavy samplers (e.g. cross-section reduced-NB) still
// win under pmap because their per-device XLA work runs in parallel; this only
// serializes the (comparatively small) solve.
static std::mutex g_klu_solve_mtx;

static bool klu_pattern_matches(const KluPattern* e, const int32_t* Ai,
                                const int32_t* Aj, int64_t nnz, int64_t n) {
  return e->n == n && e->Ai.size() == static_cast<size_t>(nnz) &&
         std::memcmp(e->Ai.data(), Ai, nnz * sizeof(int32_t)) == 0 &&
         std::memcmp(e->Aj.data(), Aj, nnz * sizeof(int32_t)) == 0;
}

// Build full (non-symmetric) CSC — every entry, sorted by (col, row), duplicates
// merged — and run klu_analyze. Caller holds g_klu_reg_mtx.
static std::shared_ptr<KluPattern> klu_create_pattern_locked(
    const int32_t* Ai, const int32_t* Aj, int64_t nnz, int64_t n,
    std::string* err) {
  for (int64_t k = 0; k < nnz; ++k) {
    if (Ai[k] < 0 || Ai[k] >= n || Aj[k] < 0 || Aj[k] >= n) {
      *err = "sparsax(klu): COO index out of range for matrix dimension " +
             std::to_string(n);
      return nullptr;
    }
  }

  auto e = std::make_shared<KluPattern>();
  e->n = n;
  e->Ai.assign(Ai, Ai + nnz);
  e->Aj.assign(Aj, Aj + nnz);
  e->pos.assign(nnz, -1);

  struct Trip {
    int32_t i, j;
    int64_t k;
  };
  std::vector<Trip> trips;
  trips.reserve(nnz);
  for (int64_t k = 0; k < nnz; ++k) trips.push_back({Ai[k], Aj[k], k});
  std::sort(trips.begin(), trips.end(), [](const Trip& a, const Trip& b) {
    return a.j != b.j ? a.j < b.j : a.i < b.i;
  });

  int64_t nnz_csc = 0;
  for (size_t t = 0; t < trips.size(); ++t)
    if (t == 0 || trips[t].i != trips[t - 1].i || trips[t].j != trips[t - 1].j)
      ++nnz_csc;

  e->nnz_csc = nnz_csc;
  e->Ap.assign(n + 1, 0);
  e->Ci.assign(nnz_csc > 0 ? nnz_csc : 1, 0);

  int64_t slot = -1;
  for (size_t t = 0; t < trips.size(); ++t) {
    if (t == 0 || trips[t].i != trips[t - 1].i ||
        trips[t].j != trips[t - 1].j) {
      ++slot;
      e->Ci[slot] = trips[t].i;
      e->Ap[trips[t].j + 1] += 1;  // per-column counts, prefix-summed below
    }
    e->pos[trips[t].k] = slot;
  }
  for (int64_t j = 0; j < n; ++j) e->Ap[j + 1] += e->Ap[j];

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

static std::shared_ptr<KluPattern> klu_get_pattern(const int32_t* Ai,
                                                   const int32_t* Aj,
                                                   int64_t nnz, int64_t n,
                                                   std::string* err) {
  std::lock_guard<std::mutex> lk(g_klu_reg_mtx);
  auto it = g_klu_registry.find(pattern_hash(Ai, Aj, nnz, n));
  if (it != g_klu_registry.end())
    for (auto& e : it->second)
      if (klu_pattern_matches(e.get(), Ai, Aj, nnz, n)) return e;
  return klu_create_pattern_locked(Ai, Aj, nnz, n, err);
}

// Per-thread KLU common + scratch buffers.  klu_common is a small POD of control
// params + stats (no heap workspace), so one per thread removes the shared-common
// serialization; XLA's CPU thread pool can run a chain's solves on different pool
// threads, so this is keyed on the OS thread, not on the chain (the numeric cache
// below is shared to make reuse thread-agnostic).
struct KluThreadLocal {
  klu_common common;
  bool ready = false;
  std::vector<double> csc, work;
  klu_common* get() {
    if (!ready) {
      klu_defaults(&common);
      ready = true;
    }
    return &common;
  }
};
static thread_local KluThreadLocal t_klu;

// Return a refcounted numeric factor for these COO values: hit the pattern's
// shared cache (short lock) or build a fresh klu_factor OUTSIDE the lock with
// this thread's own common + scratch, so concurrent solves run in parallel.
static KluNumPtr klu_get_factor(const std::shared_ptr<KluPattern>& pat,
                                const double* Ax, int64_t nnz,
                                std::string* err) {
  uint64_t h = fnv1a(Ax, nnz * sizeof(double), 14695981039346656037ULL);
  {
    std::lock_guard<std::mutex> lk(pat->cache_mtx);
    for (auto& s : pat->cache) {
      if (s.num && s.hash == h && s.Ax.size() == static_cast<size_t>(nnz) &&
          std::memcmp(s.Ax.data(), Ax, nnz * sizeof(double)) == 0)
        return s.num;  // shared_ptr copy — safe to use after unlock
    }
  }

  // Miss: scatter + factor outside the lock (per-thread common + scratch).
  klu_common* c = t_klu.get();
  t_klu.csc.assign(pat->nnz_csc > 0 ? pat->nnz_csc : 1, 0.0);
  for (int64_t k = 0; k < nnz; ++k)
    if (pat->pos[k] >= 0) t_klu.csc[pat->pos[k]] += Ax[k];

  c->status = KLU_OK;
  klu_numeric* raw = klu_factor(pat->Ap.data(), pat->Ci.data(), t_klu.csc.data(),
                                pat->symbolic, c);
  if (!raw || c->status != KLU_OK) {
    if (raw) klu_free_numeric(&raw, c);
    *err = "sparsax(klu): klu_factor failed (status " +
           std::to_string(c->status) + "; matrix may be singular)";
    return nullptr;
  }
  g_klu_num_factorizations.fetch_add(1, std::memory_order_relaxed);
  auto holder = std::make_shared<KluNumericHolder>();
  holder->num = raw;

  {
    std::lock_guard<std::mutex> lk(pat->cache_mtx);
    if (pat->cache.size() < g_lu_cache_cap) {
      pat->cache.push_back({h, std::vector<double>(Ax, Ax + nnz), holder});
    } else {
      KluNumericSlot& s = pat->cache[pat->next];
      s.hash = h;
      s.Ax.assign(Ax, Ax + nnz);
      s.num = holder;  // drops the old shared_ptr; freed once no solve holds it
      pat->next = (pat->next + 1) % pat->cache.size();
    }
  }
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

// Solve A x = b (trans == false) or A^T x = b (trans == true, used by the VJP)
// for one right-hand-side block, using this thread's common + scratch. b/x are
// row-major (JAX layout); KLU is column-major with leading dimension n, so
// multi-RHS blocks transpose through `work` and KLU solves in place.
//
// Concurrency: distinct chains carry distinct ρ, hence distinct numeric factors,
// so concurrent solves touch distinct KLU objects (their per-Numeric work
// buffers).  The per-chain init jitter makes bit-identical ρ across chains — the
// only way two threads would share one Numeric in a concurrent solve —
// vanishingly unlikely.
static ffi::Error klu_solve_one(const KluPattern* pat, klu_numeric* num,
                                bool trans, const double* bdata, int64_t n,
                                int64_t nrhs, double* xdata) {
  klu_common* c = t_klu.get();
  std::vector<double>& work = t_klu.work;
  work.resize(n * nrhs);
  if (nrhs == 1) {
    std::memcpy(work.data(), bdata, n * sizeof(double));
  } else {
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < nrhs; ++j)
        work[i + j * n] = bdata[i * nrhs + j];
  }

  c->status = KLU_OK;
  int ok = trans ? klu_tsolve(pat->symbolic, num, static_cast<int32_t>(n),
                              static_cast<int32_t>(nrhs), work.data(), c)
                 : klu_solve(pat->symbolic, num, static_cast<int32_t>(n),
                             static_cast<int32_t>(nrhs), work.data(), c);
  if (!ok || c->status != KLU_OK)
    return ffi::Error::Internal("sparsax(klu): klu_solve failed (status " +
                                std::to_string(c->status) + ")");

  if (nrhs == 1) {
    std::memcpy(xdata, work.data(), n * sizeof(double));
  } else {
    for (int64_t i = 0; i < n; ++i)
      for (int64_t j = 0; j < nrhs; ++j)
        xdata[i * nrhs + j] = work[i + j * n];
  }
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// lu_solve handler:  (Ai, Aj, Ax, b; trans) -> x with x.shape == b.shape
// b may be (n,) or (n, nrhs). trans != 0 solves A^T x = b (for the adjoint).
// ---------------------------------------------------------------------------

static ffi::Error LuSolveF64Impl(ffi::Buffer<ffi::S32> Ai,
                                 ffi::Buffer<ffi::S32> Aj,
                                 ffi::Buffer<ffi::F64> Ax,
                                 ffi::Buffer<ffi::F64> b,
                                 ffi::ResultBuffer<ffi::F64> x, int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("sparsax(klu): b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): Ai, Aj, Ax must have the same length");

  // No global lock: the shared pattern (klu_analyze) is fetched under a short
  // registry lock; the numeric factor comes from the pattern's shared cache
  // (short lock), and the klu_factor / klu_solve run with this thread's own
  // common, so concurrent per-device (pmap) solves run in parallel.  `pat` and
  // `num` are held (shared_ptr) for the duration, so a concurrent clear_cache /
  // eviction cannot free them mid-solve.
  std::string err;
  auto pat = klu_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  KluNumPtr num = klu_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);

  return klu_solve_one(pat.get(), num->num, trans != 0, b.typed_data(), n, nrhs,
                       x->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuSolveF64, LuSolveF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Attr<int64_t>("trans"));

// ---------------------------------------------------------------------------
// batched lu_solve:  (Ai, Aj, Ax[B,nnz], b[B,n(,nrhs)]; trans) -> x[B,...]
// One FFI call for a whole batch that shares a sparsity pattern. The LRU keeps
// each element's factor across successive batched calls, so a vmapped
// factor-then-many-solves pattern refactors once per element, not once per call.
// ---------------------------------------------------------------------------

static ffi::Error LuSolveBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                        ffi::Buffer<ffi::S32> Aj,
                                        ffi::Buffer<ffi::F64> Ax,
                                        ffi::Buffer<ffi::F64> b,
                                        ffi::ResultBuffer<ffi::F64> x,
                                        int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): batched b must be 2D or 3D");
  int64_t batch = bdims[0];
  int64_t n = bdims[1];
  int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[0] != batch || axdims[1] != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): batched Ax must have shape (batch, nnz) matching b");
  if (static_cast<int64_t>(Aj.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): Ai, Aj must have the same length");

  // Batched path: one FFI call solves a whole vmap batch on a single thread
  // (this is what jax.vmap lowers to). Uses the shared factor cache, so the m+1
  // solves of a Krylov basis at a fixed ρ reuse one factor per batch element.
  // (Multi-core parallelism comes from pmap over the unbatched handler, not this
  // loop.)
  std::string err;
  auto pat = klu_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);

  const double* Axd = Ax.typed_data();
  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  for (int64_t s = 0; s < batch; ++s) {
    KluNumPtr num = klu_get_factor(pat, Axd + s * nnz, nnz, &err);
    if (!num) return ffi::Error::Internal(err);
    ffi::Error r = klu_solve_one(pat.get(), num->num, trans != 0,
                                 bd + s * bstride, n, nrhs, xd + s * bstride);
    if (r.failure()) return r;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuSolveBatchedF64, LuSolveBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b  [B,...]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x  [B,...]
                                  .Attr<int64_t>("trans"));

// ---------------------------------------------------------------------------
// lu_factor / lu_solve_factor: the non-symmetric "hold a numeric factor open"
// primitives, mirroring the CHOLMOD factor/solve_factor tokens. A lu_factor(A)
// call computes (or fetches from the per-pattern LRU) a klu_numeric for Ax and
// returns an opaque int64[2] token; lu_solve_factor consumes it for an
// unbounded sequence of A x = b / A^T x = b solves without re-hashing Ax or
// refactoring. This is the recurrence structure lu_solve cannot express
// (V_{j+1} = A^{-1}(G V_j) needs the previous solve's output as the next RHS).
//
// The token holds a strong shared_ptr<KluNumericHolder> in g_klu_factor_slots,
// so the per-pattern LRU cannot evict the factor while a token references it.
// Under vmap, lu_factor produces one token per batch element (one factorization
// each); vmap(lu_solve_factor, in_axes=(None, 0)) reuses one token across a
// batch of RHS — the key Krylov-recurrence shape (one factor, m+1 solves).
//
// No lu_logdet_factor: KLU factors non-symmetric matrices, and det is not
// needed for the spatial-sampler use case (the log-density logdet comes from
// the CHOLMOD path on the symmetric precision matrix).
// ---------------------------------------------------------------------------

static ffi::Error LuFactorF64Impl(ffi::Buffer<ffi::S32> Ai,
                                  ffi::Buffer<ffi::S32> Aj,
                                  ffi::Buffer<ffi::F64> Ax,
                                  ffi::ResultBuffer<ffi::S64> token,
                                  int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): Ai, Aj, Ax must have the same length");

  std::string err;
  auto pat = klu_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  KluNumPtr num = klu_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);

  // Register a held factor in the token table. The shared_ptr copy keeps the
  // factor alive independently of the per-pattern LRU.
  uint64_t key;
  {
    std::lock_guard<std::mutex> fk(g_klu_factor_mtx);
    key = g_klu_factor_next_key++;
    g_klu_factor_slots.emplace(key, KluHeldFactor{pat, num});
  }
  int64_t* td = token->typed_data();
  td[0] = static_cast<int64_t>(key);
  td[1] = 0;  // reserved (single-slot token; KLU uses one key, not key+slot)
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuFactorF64, LuFactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Attr<int64_t>("n"));

static ffi::Error LuSolveFactorF64Impl(ffi::Buffer<ffi::S64> token,
                                       ffi::Buffer<ffi::F64> b,
                                       ffi::ResultBuffer<ffi::F64> x,
                                       int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return ffi::Error::InvalidArgument("sparsax(klu): b must be 1D or 2D");
  int64_t n = bdims[0];
  int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  if (static_cast<int64_t>(token.element_count()) != 2)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): factor token must be int64[2]");

  KluFactorRef ref;
  ref.slot_key = static_cast<uint64_t>(token.typed_data()[0]);

  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  std::string err;
  auto held = resolve_klu_factor_ref_locked(ref, &err);
  if (!held) return ffi::Error::InvalidArgument(err);
  return klu_solve_one(held->pat.get(), held->num->num, trans != 0,
                       b.typed_data(), n, nrhs, x->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuSolveFactorF64, LuSolveFactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x
                                  .Attr<int64_t>("trans"));

// Batched lu_solve_factor: the leading axis is the batch. Two vmap shapes:
//   (a) vmap over b against ONE factor: token unbatched int64[2], b is
//       (B, n[, nrhs]) — the Krylov-recurrence case (m+1 solves, one factor).
//   (b) vmap over (token, b) — one factor per batch element: token (B, 2),
//       b (B, n[, nrhs]).
static ffi::Error LuSolveFactorBatchedF64Impl(ffi::Buffer<ffi::S64> token,
                                              ffi::Buffer<ffi::F64> b,
                                              ffi::ResultBuffer<ffi::F64> x,
                                              int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): batched b must be 2D or 3D");
  int64_t batch = bdims[0];
  int64_t n = bdims[1];
  int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  auto tdims = token.dimensions();
  bool token_batched = (tdims.size() == 2 && tdims[0] == batch && tdims[1] == 2);
  bool token_scalar = (tdims.size() == 1 && tdims[0] == 2);
  if (!token_batched && !token_scalar)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): token must be int64[2] or (batch, 2) matching b");

  std::vector<KluFactorRef> refs;
  if (token_scalar) {
    refs.push_back(
        {static_cast<uint64_t>(token.typed_data()[0])});
  } else {
    refs.reserve(batch);
    for (int64_t s = 0; s < batch; ++s)
      refs.push_back(
          {static_cast<uint64_t>(token.typed_data()[s * 2])});
  }

  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  for (int64_t s = 0; s < batch; ++s) {
    const KluFactorRef& ref = token_scalar ? refs[0] : refs[s];
    std::string err;
    auto held = resolve_klu_factor_ref_locked(ref, &err);
    if (!held) return ffi::Error::InvalidArgument(err);
    ffi::Error r =
        klu_solve_one(held->pat.get(), held->num->num, trans != 0,
                      bd + s * bstride, n, nrhs, xd + s * bstride);
    if (r.failure()) return r;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuSolveFactorBatchedF64,
                              LuSolveFactorBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [2] or [B,2]
                                  .Arg<ffi::Buffer<ffi::F64>>()   // b [B,...]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // x [B,...]
                                  .Attr<int64_t>("trans"));

// ---------------------------------------------------------------------------
// lu_logdet: log|det(A)| for a general (non-symmetric) sparse matrix via KLU.
//
// Computed from the LU factor's U diagonal (plus the row-scale correction): an
// O(n) sum of log|U_ii|, far cheaper than anything Python-side, and genuinely
// needed for non-symmetric spatial models where log|I - rho W| appears in the
// log-likelihood Jacobian and W is row-standardised (cannot go through CHOLMOD).
// Shares the per-pattern LRU factor cache with lu_solve, so a lu_solve and a
// lu_logdet with identical values factorize only once. Differentiable is not
// provided (the non-symmetric logdet gradient is the full selected inverse,
// which KLU does not expose); use the CHOLMOD path for symmetric A when a
// gradient is needed.
//
// lu_logdet_factor: the same quantity read from a held token (no refactor).
// ---------------------------------------------------------------------------

static ffi::Error LuLogdetF64Impl(ffi::Buffer<ffi::S32> Ai,
                                 ffi::Buffer<ffi::S32> Aj,
                                 ffi::Buffer<ffi::F64> Ax,
                                 ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): Ai, Aj, Ax must have the same length");

  std::string err;
  auto pat = klu_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  KluNumPtr num = klu_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);

  double ld = klu_logdet_one(num->num, n);
  if (!std::isfinite(ld))
    return ffi::Error::Internal("sparsax(klu): matrix is singular");
  out->typed_data()[0] = ld;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuLogdetF64, LuLogdetF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  .Attr<int64_t>("n"));

// Batched lu_logdet: (Ai, Aj, Ax[B,nnz]; n) -> logdet[B]. One FFI call factors
// the whole batch (reusing the LRU per element) and reads each U diagonal.
static ffi::Error LuLogdetBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                         ffi::Buffer<ffi::S32> Aj,
                                         ffi::Buffer<ffi::F64> Ax,
                                         ffi::ResultBuffer<ffi::F64> out,
                                         int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[1] != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): batched Ax must have shape (batch, nnz)");
  if (static_cast<int64_t>(Aj.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): Ai, Aj must have the same length");
  int64_t batch = axdims[0];

  std::string err;
  auto pat = klu_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  const double* Axd = Ax.typed_data();
  double* od = out->typed_data();
  for (int64_t s = 0; s < batch; ++s) {
    KluNumPtr num = klu_get_factor(pat, Axd + s * nnz, nnz, &err);
    if (!num) return ffi::Error::Internal(err);
    double ld = klu_logdet_one(num->num, n);
    if (!std::isfinite(ld))
      return ffi::Error::Internal("sparsax(klu): matrix is singular");
    od[s] = ld;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuLogdetBatchedF64,
                              LuLogdetBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet [B]
                                  .Attr<int64_t>("n"));

static ffi::Error LuLogdetFactorF64Impl(ffi::Buffer<ffi::S64> token,
                                       ffi::ResultBuffer<ffi::F64> out) {
  if (static_cast<int64_t>(token.element_count()) != 2)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): factor token must be int64[2]");
  KluFactorRef ref;
  ref.slot_key = static_cast<uint64_t>(token.typed_data()[0]);

  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  std::string err;
  auto held = resolve_klu_factor_ref_locked(ref, &err);
  if (!held) return ffi::Error::InvalidArgument(err);
  int64_t n = held->pat->n;
  double ld = klu_logdet_one(held->num->num, n);
  if (!std::isfinite(ld))
    return ffi::Error::Internal("sparsax(klu): matrix is singular");
  out->typed_data()[0] = ld;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuLogdetFactorF64, LuLogdetFactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  );

// Batched lu_logdet_factor: vmap over a batch of tokens (one logdet per chain).
static ffi::Error LuLogdetFactorBatchedF64Impl(ffi::Buffer<ffi::S64> token,
                                               ffi::ResultBuffer<ffi::F64> out) {
  auto tdims = token.dimensions();
  if (tdims.size() != 2 || tdims[1] != 2)
    return ffi::Error::InvalidArgument(
        "sparsax(klu): batched token must be (batch, 2)");
  int64_t batch = tdims[0];
  std::vector<KluFactorRef> refs;
  refs.reserve(batch);
  for (int64_t s = 0; s < batch; ++s)
    refs.push_back({static_cast<uint64_t>(token.typed_data()[s * 2])});

  std::lock_guard<std::mutex> lk(g_klu_solve_mtx);
  double* od = out->typed_data();
  for (int64_t s = 0; s < batch; ++s) {
    std::string err;
    auto held = resolve_klu_factor_ref_locked(refs[s], &err);
    if (!held) return ffi::Error::InvalidArgument(err);
    int64_t n = held->pat->n;
    double ld = klu_logdet_one(held->num->num, n);
    if (!std::isfinite(ld))
      return ffi::Error::Internal("sparsax(klu): matrix is singular");
    od[s] = ld;
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholgraphLuLogdetFactorBatchedF64,
                              LuLogdetFactorBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [B,2]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet [B]
                                  );

// ---------------------------------------------------------------------------
// nanobind module
// ---------------------------------------------------------------------------

NB_MODULE(sparsax_cpp, m) {
  m.doc() = "CHOLMOD sparse Cholesky as XLA FFI custom calls";

  m.def("solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSolveF64));
  });
  m.def("solve_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSolveBatchedF64));
  });
  m.def("logdet_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodLogdetF64));
  });
  m.def("selinv_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSelinvF64));
  });
  m.def("updown_solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodUpdownSolveF64));
  });
  m.def("factor_solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodFactorSolveF64));
  });
  m.def("factor_solve_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodFactorSolveBatchedF64));
  });

  // factor / solve_factor / logdet_factor — hold-a-numeric-factor-open tokens.
  m.def("factor_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodFactorF64));
  });
  m.def("solve_factor_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSolveFactorF64));
  });
  m.def("solve_factor_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodSolveFactorBatchedF64));
  });
  m.def("logdet_factor_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodLogdetFactorF64));
  });
  m.def("logdet_factor_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholmodLogdetFactorBatchedF64));
  });

  // KLU sparse-LU handlers for non-symmetric matrices.
  m.def("lu_solve_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuSolveF64));
  });
  m.def("lu_solve_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuSolveBatchedF64));
  });
  // KLU factor-token + logdet handlers.
  m.def("lu_factor_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuFactorF64));
  });
  m.def("lu_solve_factor_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuSolveFactorF64));
  });
  m.def("lu_solve_factor_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuSolveFactorBatchedF64));
  });
  m.def("lu_logdet_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuLogdetF64));
  });
  m.def("lu_logdet_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuLogdetBatchedF64));
  });
  m.def("lu_logdet_factor_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuLogdetFactorF64));
  });
  m.def("lu_logdet_factor_batched_f64_capsule", []() {
    return nb::capsule(reinterpret_cast<void*>(CholgraphLuLogdetFactorBatchedF64));
  });

  // Numeric factors retained per KLU pattern *per thread* (>= the distinct ρ
  // touched in a chain's sweep for the Krylov-basis reuse to land).
  m.def("set_lu_cache_size", [](size_t n) { g_lu_cache_cap = n < 1 ? 1 : n; });

  // Cap on the value-keyed numeric-factor cache per CHOLMOD pattern for the
  // factor/solve_factor/logdet_factor token primitives. When a pattern's
  // num_cache would exceed this cap, the oldest slot (by insertion order) is
  // evicted and its factor copy freed; any outstanding token referencing it
  // becomes stale and resolves to an error. The token-based API is the
  // recommended path when guaranteed reuse is needed (hold the token), so this
  // cache mostly matters for repeated factor() calls with the same Ax.
  m.def("set_num_cache_size", [](size_t n) {
    std::lock_guard<std::mutex> lock(g_mutex);
    // Truncate existing caches immediately if shrinking.
    for (auto& [h, chain] : g_registry)
      for (auto& e : chain) {
        if (e->num_cache.size() <= n) continue;
        for (size_t i = n; i < e->num_cache.size(); ++i)
          if (e->num_cache[i].Lf) cholmod_free_factor(&e->num_cache[i].Lf, &g_common);
        e->num_cache.resize(n);
        if (e->num_next > n) e->num_next = 0;
      }
  });

  // Numpy-callable core, for non-JAX frontends (sparsax.pytensor).
  m.def("solve_np", &solve_np);
  m.def("logdet_np", &logdet_np);
  m.def("selinv_np", &selinv_np);

  // Total numeric (re)factorizations performed, for tests/introspection
  // (CHOLMOD + KLU across all threads).
  m.def("factorization_count", []() {
    std::lock_guard<std::mutex> lock(g_mutex);
    return g_num_factorizations +
           g_klu_num_factorizations.load(std::memory_order_relaxed);
  });

  m.def("clear_cache", []() {
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      g_last_entry = nullptr;
      if (g_started) {
        for (auto& [h, chain] : g_registry)
          for (auto& e : chain) free_entry_locked(e.get());
        g_registry.clear();
      }
      // Invalidate all outstanding factor tokens: their pattern pointers are
      // dangling once the patterns are freed, so drop the key map under the
      // same lock that owns the pattern lifetime.
      std::lock_guard<std::mutex> lk(g_factor_ref_mtx);
      g_factor_pattern_keys.clear();
    }
    // Drop shared KLU patterns; each pattern owns its numeric-factor cache, so
    // those factors are freed once no in-flight solve still holds them (they are
    // refcounted).  (Invoke when the sampler is idle, not concurrently with
    // active solves.) Also invalidate outstanding KLU factor tokens.
    {
      std::lock_guard<std::mutex> lk(g_klu_reg_mtx);
      g_klu_registry.clear();
      std::lock_guard<std::mutex> fk(g_klu_factor_mtx);
      g_klu_factor_slots.clear();
    }
  });

  m.def("cache_size", []() {
    size_t count = 0;
    {
      std::lock_guard<std::mutex> lock(g_mutex);
      for (auto& [h, chain] : g_registry) count += chain.size();
    }
    {
      std::lock_guard<std::mutex> lk(g_klu_reg_mtx);
      for (auto& [h, chain] : g_klu_registry) count += chain.size();
    }
    return count;
  });

  // 0 = CHOLMOD_SIMPLICIAL, 1 = CHOLMOD_AUTO, 2 = CHOLMOD_SUPERNODAL.
  // Only affects patterns analyzed after the call; callers clear the cache.
  m.def("set_supernodal", [](int v) {
    std::lock_guard<std::mutex> lock(g_mutex);
    ensure_started_locked();
    g_common.supernodal = v;
  });
}
