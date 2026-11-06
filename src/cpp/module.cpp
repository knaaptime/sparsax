// The Python module: the FFI handlers, settings, and the numpy core.
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "cholmod_core.h"
#include "ghk.h"
#include "handlers.h"
#include "lu_backends.h"

namespace nb = nanobind;

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

// Look up the pattern and its factor for these values, raising on failure.
static std::pair<std::shared_ptr<CholPattern>, CholNumPtr> chol_prepare_np(
    const int32_t* Ai, const int32_t* Aj, const double* Ax, int64_t nnz,
    int64_t n) {
  std::string err;
  auto pat = chol_get_pattern(Ai, Aj, nnz, n, &err);
  if (!pat) throw std::runtime_error(err);
  CholNumPtr num = chol_get_factor(pat, Ax, nnz, &err);
  if (!num) throw std::runtime_error(err);
  return {pat, num};
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
    nb::gil_scoped_release release;
    auto [pat, num] = chol_prepare_np(Ai.data(), Aj.data(), Ax.data(), nnz, n);
    if (num->L) {
      ffi::Error r = chol_solve_one(num->L, static_cast<int>(mode), b.data(), n,
                                    nrhs, out);
      if (r.failure()) throw std::runtime_error("sparsax: solve failed");
    } else {
      fill_nan(out, n * nrhs);
    }
  }
  if (b.ndim() == 2)
    return nb::ndarray<nb::numpy, double>(
        out, {static_cast<size_t>(n), static_cast<size_t>(nrhs)}, owner);
  return nb::ndarray<nb::numpy, double>(out, {static_cast<size_t>(n)}, owner);
}

// GHK's uniforms and the raw Sobol points, (n_draws, n_steps) and (n_points, dims),
// for the tests.  Both run the code the simulator runs.
static nb::ndarray<nb::numpy, double> ghk_uniforms_np(int64_t n_draws, int64_t n_steps,
                                                      int64_t sobol_dims, int64_t seed,
                                                      int64_t key) {
  if (sobol_dims < 0 || sobol_dims > sobol_max_dims())
    throw std::runtime_error("sparsax: invalid number of Sobol dimensions");
  double* out = new double[static_cast<size_t>(n_draws) * n_steps];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  ghk_uniforms(n_draws, n_steps, sobol_dims, seed, key, out);
  return nb::ndarray<nb::numpy, double>(
      out, {static_cast<size_t>(n_draws), static_cast<size_t>(n_steps)}, owner);
}

static nb::ndarray<nb::numpy, double> sobol_points_np(int64_t n_points, int64_t dims) {
  if (dims < 1 || dims > sobol_max_dims())
    throw std::runtime_error("sparsax: invalid number of Sobol dimensions");
  double* out = new double[static_cast<size_t>(n_points) * dims];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  sobol_points(n_points, dims, out);
  return nb::ndarray<nb::numpy, double>(
      out, {static_cast<size_t>(n_points), static_cast<size_t>(dims)}, owner);
}

// What the analysis of this pattern chose and what it will cost.
static nb::dict analysis_info(ArrI32 Ai, ArrI32 Aj, int64_t n) {
  std::string err;
  auto pat = chol_get_pattern(Ai.data(), Aj.data(),
                              static_cast<int64_t>(Ai.shape(0)), n, &err);
  if (!pat) throw std::runtime_error(err);
  const char* name = "other";
  switch (pat->Lsym->ordering) {
    case CHOLMOD_AMD:
      name = "amd";
      break;
    case CHOLMOD_METIS:
      name = "metis";
      break;
    case CHOLMOD_NESDIS:
      name = "nesdis";
      break;
    default:
      break;
  }
  nb::dict info;
  info["ordering"] = name;
  info["lnz"] = pat->lnz;
  info["flops"] = pat->flops;
  return info;
}

// KLU's work on this pattern, at values that depend on the pattern alone.
static nb::dict lu_work(ArrI32 Ai, ArrI32 Aj, int64_t n) {
  if (Ai.shape(0) != Aj.shape(0))
    throw std::runtime_error("sparsax: Ai, Aj must have the same length");
  std::string err;
  const LuWork work = klu_pattern_work(Ai.data(), Aj.data(),
                                       static_cast<int64_t>(Ai.shape(0)), n, &err);
  if (!err.empty()) throw std::runtime_error(err);
  nb::dict info;
  info["flops"] = work.flops;
  info["lnz"] = work.lnz;
  info["unz"] = work.unz;
  return info;
}

static double logdet_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  nb::gil_scoped_release release;
  auto [pat, num] = chol_prepare_np(Ai.data(), Aj.data(), Ax.data(), nnz, n);
  return num->logdet;
}

static nb::ndarray<nb::numpy, double> selinv_np(ArrI32 Ai, ArrI32 Aj, ArrF64 Ax,
                                                int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.shape(0));
  check_coo_np(Ai, Aj, Ax, nnz);
  double* out = new double[nnz > 0 ? static_cast<size_t>(nnz) : 1];
  nb::capsule owner(out,
                    [](void* p) noexcept { delete[] static_cast<double*>(p); });
  {
    nb::gil_scoped_release release;
    auto [pat, num] = chol_prepare_np(Ai.data(), Aj.data(), Ax.data(), nnz, n);
    if (num->L) {
      ffi::Error r = chol_selinv(pat.get(), num.get(), nnz, out);
      if (r.failure()) throw std::runtime_error("sparsax: selinv failed");
    } else {
      fill_nan(out, nnz);
    }
  }
  return nb::ndarray<nb::numpy, double>(out, {static_cast<size_t>(nnz)}, owner);
}

NB_MODULE(sparsax_cpp, m) {
  m.doc() = "SuiteSparse sparse direct solvers (CHOLMOD, KLU, UMFPACK) as XLA FFI custom calls";

  // Every FFI handler, by the name JAX will call it.
  m.def("ffi_handlers", []() {
    nb::dict handlers;
#define SPARSAX_ADD_HANDLER(name, symbol) \
  handlers[name] = nb::capsule(reinterpret_cast<void*>(symbol));
    SPARSAX_FFI_HANDLERS(SPARSAX_ADD_HANDLER)
#undef SPARSAX_ADD_HANDLER
    return handlers;
  });

  // For the tests and benchmarks: 0 uses a thread per element up to the budget, a
  // huge value stays serial, and a negative one restores the default.
  m.def("set_batch_work_per_thread", [](int64_t work) {
    g_batch_work_per_thread = work < 0 ? kBatchWorkPerThread : work;
  });




  // Numeric factors retained per KLU pattern in the shared cache (>= the
  // distinct ρ touched per sweep across chains for Krylov-basis reuse to land).
  m.def("set_lu_cache_size", [](size_t n) { klu_set_cache_size(n < 1 ? 1 : n); });

  // Same, for the UMFPACK numeric-factor cache.
  m.def("set_umf_cache_size", [](size_t n) { umf_set_cache_size(n < 1 ? 1 : n); });

  // Newest factor tokens kept per backend (factor / lu_factor / umf_factor).
  m.def("set_token_cache_size", [](size_t n) { g_token_cap = n < 1 ? 1 : n; });

  // Numeric factors retained per CHOLMOD pattern (default 32).
  m.def("set_num_cache_size", [](size_t n) { chol_set_cache_size(n < 1 ? 1 : n); });

  // Numpy-callable core, for non-JAX frontends (sparsax.pytensor).
  m.def("analysis_info", &analysis_info);
  m.def("lu_work", &lu_work);
  m.def("set_ordering", [](int ordering) { g_chol_ordering = ordering; });
  m.def("ghk_uniforms_np", &ghk_uniforms_np);
  m.def("sobol_points_np", &sobol_points_np);
  m.def("solve_np", &solve_np);
  m.def("logdet_np", &logdet_np);
  m.def("selinv_np", &selinv_np);

  // Total numeric (re)factorizations performed, for tests/introspection
  // (CHOLMOD + KLU + UMFPACK across all threads).
  m.def("factorization_count", []() {
    return chol_factorizations() + klu_factorizations() + umf_factorizations();
  });

  m.def("set_nan_on_failure", [](bool nan) { g_nan_on_failure = nan; });

  // Simplicial LDL' copies built so far, and whether a supernodal factor's
  // selected inverse avoids them; for the tests and the wheel checks.
  m.def("ldl_conversion_count", []() {
    return g_num_ldl_conversions.load(std::memory_order_relaxed);
  });
  m.def("has_supernodal_selinv", &chol_has_supernodal_selinv);

  m.def("clear_cache", []() {
    chol_clear_cache();
    klu_clear_cache();
    umf_clear_cache();
  });

  m.def("cache_size", []() {
    return chol_pattern_count() + klu_pattern_count() + umf_pattern_count();
  });

  // 0 = CHOLMOD_SIMPLICIAL, 1 = CHOLMOD_AUTO, 2 = CHOLMOD_SUPERNODAL.
  // Only affects patterns analyzed after the call; callers clear the cache.
  m.def("set_supernodal", [](int v) { g_chol_supernodal = v; });
}
