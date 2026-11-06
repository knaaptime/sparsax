// What every backend shares: threads, the on_failure policy, pattern registries, the
// value-keyed numeric caches, and factor tokens.
//
// sparsax: SuiteSparse sparse direct solvers as XLA FFI custom calls --
// CHOLMOD (Cholesky, symmetric positive definite), KLU (LU, general
// non-symmetric), and UMFPACK (multifrontal LU, general non-symmetric).
//
// Design:
//   - Symbolic analyses are cached in a registry keyed by a hash of the
//     sparsity pattern (n, Ai, Aj), with full memcmp verification on hit, and
//     numeric factors in a per-pattern cache keyed on the values: repeated
//     calls with the same pattern reuse the analysis, and calls with identical
//     values skip the factorization.
//   - No global lock. Shared state is read-only or behind short lookup locks,
//     and each thread has its own CHOLMOD / KLU / UMFPACK workspace, so calls
//     from concurrent threads (one per MCMC chain, say) run in parallel.
//   - Input is COO of a symmetric matrix. Entries with i <= j (upper triangle
//     plus diagonal) are used; entries with i > j are ignored, so both
//     "full symmetric" and "upper-only" input work. Duplicates are summed.
//
// The FFI handlers never touch Python and hold no GIL, so they run at full
// native speed inside XLA-compiled programs.
#ifndef SPARSAX_COMMON_H_
#define SPARSAX_COMMON_H_

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_map>
#include <vector>

#include "xla/ffi/api/ffi.h"

namespace ffi = xla::ffi;

// ---------------------------------------------------------------------------
// Many-column solves on several threads
//
// A solve with many right-hand sides is split into column panels solved on
// several threads, for every backend.  The factor is only read by a solve
// (cholmod_solve2, umfpack_di_wsolve, and klu_solve apart from its Xwork, which
// KLU workers replace with their own), and every worker brings its own
// workspaces, so the panels need no lock.  Each backend's per-thread state is
// thread_local, so a worker thread builds its own on first use.
// ---------------------------------------------------------------------------

// Worker threads: SPARSAX_NUM_THREADS if set, else every hardware thread.
inline unsigned sparsax_num_threads() {
  static const unsigned n = [] {
    if (const char* env = std::getenv("SPARSAX_NUM_THREADS")) {
      long v = std::strtol(env, nullptr, 10);
      if (v > 0) return static_cast<unsigned>(v);
    }
    unsigned hw = std::thread::hardware_concurrency();
    return hw > 0 ? hw : 1u;
  }();
  return n;
}

// Columns per panel.  A block of fewer than two panels is solved serially.
constexpr int64_t kPanel = 64;

// The threads this call may use.  A worker on a batch holds a share of them, so a
// batch of many-column solves divides the machine rather than multiplying it.
inline thread_local unsigned t_thread_share = 0;  // 0: no share set, all of them

inline unsigned thread_budget() {
  return t_thread_share ? t_thread_share : sparsax_num_threads();
}

// Threads for an nrhs-column solve; 1 means solve serially.
inline unsigned panel_threads(int64_t nrhs) {
  return static_cast<unsigned>(
      std::max<int64_t>(1, std::min<int64_t>(thread_budget(), nrhs / kPanel)));
}

// Run body(failed) -> error string ("" on success) on T threads, the caller
// among them, and return the first error.  Bodies must take their work from a
// shared cursor and stop once `failed` is set: however many threads start, the
// ones running finish the work.
//
// An exception must not leave a spawned thread: std::terminate would take the
// whole process with it.  XLA turns one on the calling thread into a Python
// error, so a failed allocation has to reach the caller the same way from any
// thread.
template <class Body>
inline std::string run_on_threads(unsigned T, Body body) {
  std::atomic<bool> failed{false};
  std::mutex err_mtx;
  std::string err;
  auto record = [&](std::string e) {
    std::lock_guard<std::mutex> lk(err_mtx);
    if (err.empty()) err = std::move(e);
    failed = true;
  };
  auto guarded = [&] {
    try {
      std::string e = body(failed);
      if (!e.empty()) record(std::move(e));
    } catch (const std::exception& e) {
      record(std::string("sparsax: ") + e.what());
    } catch (...) {
      record("sparsax: unknown exception in a worker thread");
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(T - 1);
  for (unsigned t = 1; t < T; ++t) {
    try {
      pool.emplace_back(guarded);
    } catch (const std::system_error&) {
      break;  // out of threads; those already running take all the work
    }
  }
  guarded();
  for (auto& th : pool) th.join();
  return err;
}

// Run the panels of an nrhs-column block on T threads (the caller is one).
// make_worker() runs once per thread and returns that thread's solver,
// solve(j0, w) -> error string ("" on success), for columns [j0, j0 + w).
// Returns the first error.
template <class MakeWorker>
inline std::string for_each_panel(int64_t nrhs, unsigned T,
                                  MakeWorker make_worker) {
  const int64_t panels = (nrhs + kPanel - 1) / kPanel;
  std::atomic<int64_t> next{0};
  return run_on_threads(T, [&](std::atomic<bool>& failed) -> std::string {
    auto solve = make_worker();
    for (int64_t p; !failed.load() && (p = next.fetch_add(1)) < panels;) {
      const int64_t j0 = p * kPanel;
      std::string e = solve(j0, std::min(kPanel, nrhs - j0));
      if (!e.empty()) return e;
    }
    return "";
  });
}

// set_options(on_failure=...).  A factorization can fail on the numbers: the
// matrix is not positive definite, or it is singular, or a value is not finite.
// A sampler proposing such a matrix wants to reject the proposal, and under vmap
// it needs the other chains to survive, which an error cannot give it.  With
// "nan", the *_get_factor functions return a holder with no numeric factor in it,
// and each handler answers for that element with NaN.  Anything else that goes
// wrong (bad arguments, memory, a stale token) still raises.
inline std::atomic<bool> g_nan_on_failure{false};

inline bool nan_on_failure() {
  return g_nan_on_failure.load(std::memory_order_relaxed);
}

inline void fill_nan(double* p, int64_t count) {
  std::fill(p, p + count, std::numeric_limits<double>::quiet_NaN());
}

inline bool all_finite(const double* x, int64_t count) {
  for (int64_t i = 0; i < count; ++i)
    if (!std::isfinite(x[i])) return false;
  return true;
}

// What a *_get_factor returns when the numbers defeat the factorization.
template <class Holder>
inline std::shared_ptr<Holder> factorization_failed(std::string message,
                                                    std::string* err) {
  if (nan_on_failure()) return std::make_shared<Holder>();
  *err = std::move(message);
  return nullptr;
}

// Run body(s) -> error string ("" on success) for s in [0, batch), on as many
// threads as the batch is worth.  Elements are independent: each has its own factor
// and output.  A thread costs tens of microseconds to start, so each must have work
// to repay that.  element_work is a stand-in for one element's cost, nnz + n; on an
// M3 Max a thread pays its way at about 4000 units, a batch of two on n = 900, say.
inline constexpr int64_t kBatchWorkPerThread = 4000;
inline std::atomic<int64_t> g_batch_work_per_thread{kBatchWorkPerThread};

template <class Body>
inline std::string for_each_batch_element(int64_t batch, int64_t element_work,
                                          Body body) {
  const unsigned budget = thread_budget();
  const int64_t per_thread = g_batch_work_per_thread.load(std::memory_order_relaxed);
  const int64_t worth =
      per_thread > 0 ? element_work * batch / per_thread : batch;
  const unsigned T = static_cast<unsigned>(std::max<int64_t>(
      1, std::min({worth, batch, static_cast<int64_t>(budget)})));
  const unsigned share = std::max(1u, budget / T);
  std::atomic<int64_t> next{0};
  return run_on_threads(T, [&](std::atomic<bool>& failed) -> std::string {
    const unsigned saved = t_thread_share;
    t_thread_share = share;
    std::string err;
    for (int64_t s; err.empty() && !failed.load() && (s = next.fetch_add(1)) < batch;)
      err = body(s);
    t_thread_share = saved;
    return err;
  });
}

inline std::string error_text(const ffi::Error& error) {
  return error.failure() ? error.message() : std::string();
}

// Columns [j0, j0 + w) of a row-major (n, ld) block, to and from column-major.
inline void gather_cols(const double* b, int64_t n, int64_t ld, int64_t j0,
                        int64_t w, double* out) {
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < w; ++j) out[i + j * n] = b[i * ld + j0 + j];
}
inline void scatter_cols(const double* in, int64_t n, int64_t ld, int64_t j0,
                         int64_t w, double* x) {
  for (int64_t i = 0; i < n; ++i)
    for (int64_t j = 0; j < w; ++j) x[i * ld + j0 + j] = in[i + j * n];
}

inline uint64_t fnv1a(const void* data, size_t nbytes, uint64_t h) {
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

inline uint64_t pattern_hash(const int32_t* Ai, const int32_t* Aj, int64_t nnz,
                             int64_t n) {
  uint64_t h = 14695981039346656037ULL;
  h = fnv1a(&n, sizeof(n), h);
  h = fnv1a(Ai, nnz * sizeof(int32_t), h);
  h = fnv1a(Aj, nnz * sizeof(int32_t), h);
  return h;
}

template <class Pattern>
inline bool pattern_matches(const Pattern& e, const int32_t* Ai, const int32_t* Aj,
                            int64_t nnz, int64_t n) {
  return e.n == n && e.Ai.size() == static_cast<size_t>(nnz) &&
         std::memcmp(e.Ai.data(), Ai, nnz * sizeof(int32_t)) == 0 &&
         std::memcmp(e.Aj.data(), Aj, nnz * sizeof(int32_t)) == 0;
}

template <class Pattern>
using PatternRegistry =
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<Pattern>>>;

// The shared pattern for this COO structure, found or created (`create` runs under
// the registry's lock and adds it).  Every call comes through here, and hashing and
// comparing a large pattern take a while, so they run outside the lock.  A solver
// loop asks for one pattern over and over, so each thread keeps the last it used;
// `generation` is bumped by clear_cache, which makes threads forget theirs.
template <class Pattern, class ThreadLocal, class Create>
inline std::shared_ptr<Pattern> get_pattern(std::mutex& registry_mtx,
                                            PatternRegistry<Pattern>& registry,
                                            const std::atomic<uint64_t>& generation,
                                            ThreadLocal& tl, const int32_t* Ai,
                                            const int32_t* Aj, int64_t nnz, int64_t n,
                                            Create create) {
  const uint64_t gen = generation.load(std::memory_order_relaxed);
  if (tl.last_pat && tl.last_gen == gen &&
      pattern_matches(*tl.last_pat, Ai, Aj, nnz, n))
    return tl.last_pat;

  const uint64_t h = pattern_hash(Ai, Aj, nnz, n);
  std::vector<std::shared_ptr<Pattern>> candidates;
  {
    std::lock_guard<std::mutex> lk(registry_mtx);
    auto it = registry.find(h);
    if (it != registry.end()) candidates = it->second;
  }
  std::shared_ptr<Pattern> found;
  for (auto& e : candidates)
    if (pattern_matches(*e, Ai, Aj, nnz, n)) {
      found = e;
      break;
    }
  if (!found) {
    std::lock_guard<std::mutex> lk(registry_mtx);
    // Another thread may have created it since the lock was last free.
    auto it = registry.find(h);
    if (it != registry.end())
      for (auto& e : it->second)
        if (pattern_matches(*e, Ai, Aj, nnz, n)) {
          found = e;
          break;
        }
    if (!found) found = create();
  }
  if (found) {
    tl.last_pat = found;
    tl.last_gen = gen;
  }
  return found;
}

// Factorizations under way on a pattern, by the hash of their values.  Threads
// asking for values already being factored (chains started at one point, a batch
// that broadcast one matrix) wait for that factor rather than repeat the work.  The
// waiter then looks in the cache again; a hash collision or a failed factorization
// just sends it on to factor for itself.  Guarded by the pattern's cache_mtx.
struct FactorsInFlight {
  std::condition_variable done;
  std::vector<uint64_t> hashes;

  bool contains(uint64_t h) const {
    return std::find(hashes.begin(), hashes.end(), h) != hashes.end();
  }
};

// Ends a claim on the way out of a factorization, however it ends.
class InFlightClaim {
 public:
  InFlightClaim(std::mutex& mtx, FactorsInFlight& in_flight, uint64_t h)
      : mtx_(mtx), in_flight_(in_flight), h_(h) {}
  ~InFlightClaim() {
    {
      std::lock_guard<std::mutex> lk(mtx_);
      auto& hashes = in_flight_.hashes;
      hashes.erase(std::find(hashes.begin(), hashes.end(), h_));
    }
    in_flight_.done.notify_all();
  }
  InFlightClaim(const InFlightClaim&) = delete;
  InFlightClaim& operator=(const InFlightClaim&) = delete;

 private:
  std::mutex& mtx_;
  FactorsInFlight& in_flight_;
  uint64_t h_;
};

// A cached numeric factor and the values that produced it, compared on a hit.
template <class Holder>
struct NumericSlot {
  uint64_t hash = 0;
  std::vector<double> Ax;
  std::shared_ptr<Holder> num;
};

// The factor cached for these values, or nullptr once this thread has claimed the
// work of building it.  It waits while another thread builds the same values, and
// on_miss runs under the pattern's lock, where a backend can detach a factor to
// recycle.  The caller builds outside the lock, under an InFlightClaim.
template <class Pattern, class OnMiss>
inline auto find_or_claim_factor(Pattern& pat, const double* Ax, int64_t nnz,
                                 uint64_t h, OnMiss on_miss)
    -> decltype(pat.cache[0].num) {
  std::unique_lock<std::mutex> lk(pat.cache_mtx);
  for (;;) {
    for (auto& s : pat.cache) {
      if (s.num && s.hash == h && s.Ax.size() == static_cast<size_t>(nnz) &&
          std::memcmp(s.Ax.data(), Ax, nnz * sizeof(double)) == 0)
        return s.num;  // shared_ptr copy: safe to use after unlock
    }
    if (!pat.in_flight.contains(h)) break;
    pat.in_flight.done.wait(lk);
  }
  pat.in_flight.hashes.push_back(h);
  on_miss();
  return nullptr;
}

// Put a finished factor in the cache: a free slot, or else the next in rotation,
// which drops the old factor once nothing else holds it.
template <class Pattern, class Holder>
inline void cache_factor(Pattern& pat, size_t cap, uint64_t h, const double* Ax,
                         int64_t nnz, const std::shared_ptr<Holder>& num) {
  std::lock_guard<std::mutex> lk(pat.cache_mtx);
  if (pat.cache.size() < cap) {
    pat.cache.push_back({h, std::vector<double>(Ax, Ax + nnz), num});
  } else if (!pat.cache.empty()) {
    if (pat.next >= pat.cache.size()) pat.next = 0;
    auto& slot = pat.cache[pat.next];
    slot.hash = h;
    slot.Ax.assign(Ax, Ax + nnz);
    slot.num = num;
    pat.next = (pat.next + 1) % pat.cache.size();
  }
}

// Lower a value cache's cap at once: the cap is a bound on memory.  Tokens hold their
// own references, so this never invalidates one.
template <class Pattern>
inline void trim_numeric_caches(std::mutex& registry_mtx,
                                PatternRegistry<Pattern>& registry, size_t cap) {
  std::lock_guard<std::mutex> lk(registry_mtx);
  for (auto& [h, chain] : registry)
    for (auto& p : chain) {
      std::lock_guard<std::mutex> ck(p->cache_mtx);
      if (p->cache.size() > cap) p->cache.resize(cap);
      if (p->next >= p->cache.size()) p->next = 0;
    }
}

// Drop a registry's patterns.  A thread's last_pat can outlive this until that
// thread's next call, so empty the numeric caches here rather than leave them
// pinning factors.  Caller holds the registry's lock.
template <class Pattern>
inline void clear_patterns_locked(PatternRegistry<Pattern>& registry) {
  for (auto& [h, chain] : registry)
    for (auto& p : chain) {
      std::lock_guard<std::mutex> ck(p->cache_mtx);
      p->cache.clear();
      p->next = 0;
    }
  registry.clear();
}

// Recycling evicted factors.  Once a pattern's cache is full, every new set of
// values evicts a slot.  If nothing but the slot holds that slot's factor (no
// in-flight solve, no token -- both take their own reference), its numeric
// storage is detached here, under cache_mtx, and the new factorization is
// written into it instead of into a fresh allocation.  The slot is left empty
// (a lookup skips it) until the insert below refills a slot.
template <class Slot>
inline Slot* evictable_slot_locked(std::vector<Slot>& cache, size_t next,
                                   size_t cap) {
  if (cache.empty() || cache.size() < cap) return nullptr;
  Slot& s = cache[next % cache.size()];
  if (!s.num || s.num.use_count() != 1) return nullptr;
  return &s;
}

// The shape checks every handler on a batch of values needs: Ax is (batch, nnz).
inline std::string check_batched_coo(const ffi::Buffer<ffi::S32>& Ai,
                                     const ffi::Buffer<ffi::S32>& Aj,
                                     const ffi::Buffer<ffi::F64>& Ax, int64_t* batch,
                                     int64_t* nnz) {
  *nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[1] != *nnz)
    return "sparsax: batched Ax must have shape (batch, nnz)";
  if (static_cast<int64_t>(Aj.element_count()) != *nnz)
    return "sparsax: Ai, Aj must have the same length";
  *batch = axdims[0];
  return "";
}

// Token tables are bounded.  A token pins its numeric factor, and tokens are
// plain integer arrays with no destructor, so a table that only grew kept every
// factor a sampler ever built through factor / lu_factor / umf_factor (24 MB
// per factor at n = 40,000).  Each table keeps its newest g_token_cap tokens
// and drops the oldest (keys only increase, so the oldest is begin()).  A token
// used after it is dropped fails as stale; callers that hold a token across
// many later factorizations must refactor, or raise the cap.
inline std::atomic<size_t> g_token_cap{64};

template <class Map>
inline void trim_token_table_locked(Map& slots) {
  size_t cap = std::max<size_t>(1, g_token_cap.load(std::memory_order_relaxed));
  while (slots.size() > cap) slots.erase(slots.begin());
}

inline const char* kStaleTokenHint =
    " -- more than set_token_cache_size() newer tokens were created since this "
    "one (or clear_cache ran); refactor, or raise the token cache size";

// token[0] of a factorization that failed under on_failure="nan".  It names no
// slot: a consumer resolves it to an empty held factor and answers with NaN,
// whatever the option says by then.
constexpr int64_t kFailedTokenKey = -1;

// token[1].  Each backend numbers its own slots from 1, so a slot key alone can
// name a live, unrelated factor in another backend's table.
enum TokenIssuer : int64_t {
  kTokenCholmod = 1,
  kTokenKlu = 2,
  kTokenUmfpack = 3,
};

inline const char* token_issuer_name(int64_t issuer) {
  switch (issuer) {
    case kTokenCholmod:
      return "factor (CHOLMOD)";
    case kTokenKlu:
      return "lu_factor (KLU)";
    case kTokenUmfpack:
      return "umf_factor (UMFPACK)";
    default:
      return nullptr;
  }
}

// Check `count` int64[2] tokens against the backend consuming them.  Returns ""
// or an error message.
inline std::string check_token_issuer(const int64_t* tokens, int64_t count,
                                      TokenIssuer expected) {
  for (int64_t i = 0; i < count; ++i) {
    const int64_t issuer = tokens[i * 2 + 1];
    if (issuer == expected) continue;
    const char* from = token_issuer_name(issuer);
    if (!from)
      return "sparsax: invalid factor token (not produced by factor, lu_factor "
             "or umf_factor)";
    return std::string("sparsax: this token came from ") + from +
           " but was passed to a function for " + token_issuer_name(expected) +
           "; tokens are not interchangeable between backends";
  }
  return "";
}

// A token solve takes n from b, and only the held factor knows its real size:
// UMFPACK and KLU trust the caller and run off the end of a short b.
inline std::string check_token_rows(int64_t factor_n, int64_t b_rows) {
  if (factor_n == b_rows) return "";
  return "sparsax: the factor is " + std::to_string(factor_n) + " x " +
         std::to_string(factor_n) + " but b has " + std::to_string(b_rows) +
         " rows";
}

// What a token keeps alive: the factor, and the pattern for its structure.
template <class Pattern, class Holder>
struct HeldFactor {
  std::shared_ptr<Pattern> pat;
  std::shared_ptr<Holder> num;
};

// One backend's tokens.  Keys only increase, 0 means "invalid", and the table keeps
// the newest g_token_cap.  A copy of the references leaves `resolve`, so the factor
// stays alive for the caller's solve even if clear_cache runs meanwhile.
template <class Held>
class TokenTable {
 public:
  TokenTable(TokenIssuer issuer, const char* who) : issuer_(issuer), who_(who) {}

  void issue(const Held& held, int64_t* token) {
    token[1] = issuer_;
    if (!held.num->factored()) {
      token[0] = kFailedTokenKey;
      return;
    }
    std::lock_guard<std::mutex> lk(mtx_);
    const uint64_t key = next_key_++;
    slots_.emplace(key, held);
    trim_token_table_locked(slots_);
    token[0] = static_cast<int64_t>(key);
  }

  // Sets err and returns nullptr on a stale or invalid token.  A failed token
  // resolves to an empty held factor, which its consumers answer with NaN.
  std::shared_ptr<Held> resolve(uint64_t key, std::string* err) {
    if (key == 0) {
      *err = who_ + ": invalid factor token (empty)";
      return nullptr;
    }
    if (key == static_cast<uint64_t>(kFailedTokenKey))
      return std::make_shared<Held>();
    std::lock_guard<std::mutex> lk(mtx_);
    auto it = slots_.find(key);
    if (it == slots_.end()) {
      *err = who_ + ": stale factor token" + kStaleTokenHint;
      return nullptr;
    }
    return std::make_shared<Held>(it->second);
  }

  void clear() {
    std::lock_guard<std::mutex> lk(mtx_);
    slots_.clear();
  }

 private:
  TokenIssuer issuer_;
  std::string who_;
  std::mutex mtx_;
  std::map<uint64_t, Held> slots_;
  uint64_t next_key_ = 1;
};

#endif  // SPARSAX_COMMON_H_
