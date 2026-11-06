// CHOLMOD's FFI handlers.
#include "cholmod_core.h"

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

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  if (!num->L) {
    fill_nan(x->typed_data(), n * nrhs);
    return ffi::Error::Success();
  }
  return chol_solve_one(num->L, static_cast<int>(mode), b.typed_data(), n, nrhs,
                        x->typed_data());
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
// One FFI call solves a whole batch that shares a sparsity pattern, factoring
// each element through the cache. This is what jax.vmap(solve) lowers to (see
// the custom_vmap rule in Python), so the batch loop runs in C++ rather than as
// XLA per-iteration dispatch. Ai/Aj stay unbatched.
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

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    CholNumPtr num = chol_get_factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    if (!num->L) {
      fill_nan(xd + s * bstride, bstride);
      return "";
    }
    return error_text(chol_solve_one(num->L, static_cast<int>(mode),
                                     bd + s * bstride, n, nrhs, xd + s * bstride));
  });
  if (!err.empty()) return ffi::Error::Internal(err);
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
// Shares the factor cache with solve, so a solve followed by a logdet with
// identical values factorizes only once.
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

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  out->typed_data()[0] = num->logdet;
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetF64, LogdetF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet
                                  .Attr<int64_t>("n"));

static ffi::Error LogdetBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                       ffi::Buffer<ffi::S32> Aj,
                                       ffi::Buffer<ffi::F64> Ax,
                                       ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  int64_t batch = 0, nnz = 0;
  std::string err = check_batched_coo(Ai, Aj, Ax, &batch, &nnz);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  const double* Axd = Ax.typed_data();
  double* od = out->typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    CholNumPtr num = chol_get_factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    od[s] = num->logdet;
    return "";
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetBatchedF64, LogdetBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet [B]
                                  .Attr<int64_t>("n"));

// rcond handlers: CHOLMOD's reciprocal condition estimate, from the factor's diagonal.
static double chol_rcond_of(const CholNumeric& num) {
  return num.L ? cholmod_rcond(num.L, t_chol.get())
               : std::numeric_limits<double>::quiet_NaN();
}

static ffi::Error RcondF64Impl(ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
                               ffi::Buffer<ffi::F64> Ax,
                               ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  const int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");
  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  out->typed_data()[0] = chol_rcond_of(*num);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodRcondF64, RcondF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // rcond
                                  .Attr<int64_t>("n"));

static ffi::Error RcondBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                      ffi::Buffer<ffi::S32> Aj,
                                      ffi::Buffer<ffi::F64> Ax,
                                      ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  int64_t batch = 0, nnz = 0;
  std::string err = check_batched_coo(Ai, Aj, Ax, &batch, &nnz);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  const double* Axd = Ax.typed_data();
  double* od = out->typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    CholNumPtr num = chol_get_factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    od[s] = chol_rcond_of(*num);
    return "";
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodRcondBatchedF64, RcondBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // rcond [B]
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

// Apply all K solve chains for one factor. bptr/xptr hold the K input/output
// row-major blocks; intermediates ping-pong through the thread's tmpA/tmpB.
static ffi::Error chol_apply_solves(cholmod_factor* L, int64_t n,
                                    const std::vector<const double*>& bptr,
                                    const std::vector<int64_t>& nrhs,
                                    const int64_t* mode_chain,
                                    const int64_t* chain_lens, int64_t K,
                                    const std::vector<double*>& xptr) {
  CholThreadLocal& tl = t_chol;
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
      if (s == len - 1) {
        out = xptr[k];
      } else {
        std::vector<double>& t = (s % 2 == 0) ? tl.tmpA : tl.tmpB;
        t.resize(n * w);
        out = t.data();
      }
      ffi::Error r = chol_solve_one(L, static_cast<int>(mode), in, n, w, out);
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

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);

  if (num->L) {
    ffi::Error r = chol_apply_solves(num->L, n, bptr, nrhs, mode_chain.begin(),
                                     chain_lens.begin(), K, xptr);
    if (r.failure()) return r;
  } else {
    for (int64_t k = 0; k < K; ++k) fill_nan(xptr[k], n * nrhs[k]);
  }

  if (want_logdet) {
    auto ld = rets.get<ffi::Buffer<ffi::F64>>(K);
    if (ld.has_error()) return ld.error();
    (*ld)->typed_data()[0] = num->logdet;
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

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    CholNumPtr num = chol_get_factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    std::vector<const double*> bptr(K);
    std::vector<double*> xptr(K);
    for (int64_t k = 0; k < K; ++k) {
      bptr[k] = bbase[k] + s * bstride[k];
      xptr[k] = xbase[k] + s * bstride[k];
    }
    if (num->L) {
      e = error_text(chol_apply_solves(num->L, n, bptr, nrhs, mode_chain.begin(),
                                       chain_lens.begin(), K, xptr));
      if (!e.empty()) return e;
    } else {
      for (int64_t k = 0; k < K; ++k) fill_nan(xptr[k], bstride[k]);
    }
    if (want_logdet) ldbase[s] = num->logdet;
    return "";
  });
  if (!err.empty()) return ffi::Error::Internal(err);
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
// primitives. A factor(A) call fetches (or computes) the numeric factor for A's
// values and returns an opaque token; solve_factor and logdet_factor consume
// that token for an unbounded sequence of solves + a logdet, without re-hashing
// Ax or refactoring. This is the structure factor_solve cannot express: a
// recurrence V_{j+1} = A^{-1}(G V_j) that needs the previous solve's output as
// the next solve's input.
//
// The token is an int64[2]: [0] names a slot in a side table that holds strong
// references to the factor and its pattern, so the cache can evict the factor
// without invalidating the token; only clear_cache and the token cap do that.
// [1] names the backend that issued it (see check_token_issuer).
//
// Under vmap, factor produces one token per batch element (one factorization
// per element, matching factor_solve's batch semantics). solve_factor's vmap
// over b reuses a single token — the key win for the Krylov recurrence, where
// one factor serves m+1 solves.
// ---------------------------------------------------------------------------

// Decode the slot keys of a token buffer (int64[2] per token, or a batch).
static std::vector<uint64_t> decode_factor_keys(const int64_t* data,
                                                int64_t count) {
  std::vector<uint64_t> keys(count);
  for (int64_t i = 0; i < count; ++i)
    keys[i] = static_cast<uint64_t>(data[i * 2]);
  return keys;
}

static ffi::Error FactorF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::S64> token, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  g_chol_tokens.issue({pat, num}, token->typed_data());
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorF64, FactorF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::S64>>()   // token [2]
                                  .Attr<int64_t>("n"));

static ffi::Error FactorBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                       ffi::Buffer<ffi::S32> Aj,
                                       ffi::Buffer<ffi::F64> Ax,
                                       ffi::ResultBuffer<ffi::S64> token, int64_t n) {
  int64_t batch = 0, nnz = 0;
  std::string err = check_batched_coo(Ai, Aj, Ax, &batch, &nnz);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  const double* Axd = Ax.typed_data();
  int64_t* td = token->typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    CholNumPtr num = chol_get_factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    g_chol_tokens.issue({pat, num}, td + s * 2);
    return "";
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodFactorBatchedF64, FactorBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Ret<ffi::Buffer<ffi::S64>>()   // token [B,2]
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

  std::string err = check_token_issuer(token.typed_data(), 1, kTokenCholmod);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto held = g_chol_tokens.resolve(decode_factor_keys(token.typed_data(), 1)[0],
                                 &err);
  if (!held) return ffi::Error::InvalidArgument(err);
  if (!held->num) {
    fill_nan(x->typed_data(), n * nrhs);
    return ffi::Error::Success();
  }
  err = check_token_rows(held->pat->n, n);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  return chol_solve_one(held->num->L, static_cast<int>(mode), b.typed_data(), n,
                        nrhs, x->typed_data());
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

  std::string err =
      check_token_issuer(token.typed_data(), token_scalar ? 1 : batch, kTokenCholmod);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  std::vector<uint64_t> keys =
      decode_factor_keys(token.typed_data(), token_scalar ? 1 : batch);
  std::vector<std::shared_ptr<CholHeldFactor>> held(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) {
    held[i] = g_chol_tokens.resolve(keys[i], &err);
    if (!held[i]) return ffi::Error::InvalidArgument(err);
    if (held[i]->num) {
      err = check_token_rows(held[i]->pat->n, n);
      if (!err.empty()) return ffi::Error::InvalidArgument(err);
    }
  }

  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  int64_t bstride = n * nrhs;
  err = for_each_batch_element(batch, n, [&](int64_t s) -> std::string {
    const CholHeldFactor& h = *held[token_scalar ? 0 : s];
    if (!h.num) {
      fill_nan(xd + s * bstride, bstride);
      return "";
    }
    return error_text(chol_solve_one(h.num->L, static_cast<int>(mode),
                                     bd + s * bstride, n, nrhs, xd + s * bstride));
  });
  if (!err.empty()) return ffi::Error::Internal(err);
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

  std::string err = check_token_issuer(token.typed_data(), 1, kTokenCholmod);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto held = g_chol_tokens.resolve(decode_factor_keys(token.typed_data(), 1)[0],
                                 &err);
  if (!held) return ffi::Error::InvalidArgument(err);
  out->typed_data()[0] = held->num ? held->num->logdet
                                   : std::numeric_limits<double>::quiet_NaN();
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
  std::string err = check_token_issuer(token.typed_data(), batch, kTokenCholmod);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  std::vector<uint64_t> keys = decode_factor_keys(token.typed_data(), batch);

  double* od = out->typed_data();
  for (int64_t s = 0; s < batch; ++s) {
    auto held = g_chol_tokens.resolve(keys[s], &err);
    if (!held) return ffi::Error::InvalidArgument(err);
    od[s] = held->num ? held->num->logdet
                      : std::numeric_limits<double>::quiet_NaN();
  }
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodLogdetFactorBatchedF64,
                              LogdetFactorBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S64>>()   // token [B,2]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logdet [B]
                                  );

static ffi::Error SelinvF64Impl(ffi::Buffer<ffi::S32> Ai,
                                ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::F64> z, int64_t n) {
  int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  if (!num->L) {
    fill_nan(z->typed_data(), nnz);
    return ffi::Error::Success();
  }
  return chol_selinv(pat.get(), num.get(), nnz, z->typed_data());
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSelinvF64, SelinvF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Ret<ffi::Buffer<ffi::F64>>()   // z [nnz]
                                  .Attr<int64_t>("n"));

static ffi::Error SelinvBatchedF64Impl(ffi::Buffer<ffi::S32> Ai,
                                       ffi::Buffer<ffi::S32> Aj,
                                       ffi::Buffer<ffi::F64> Ax,
                                       ffi::ResultBuffer<ffi::F64> z, int64_t n) {
  int64_t batch = 0, nnz = 0;
  std::string err = check_batched_coo(Ai, Aj, Ax, &batch, &nnz);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  const double* Axd = Ax.typed_data();
  double* zd = z->typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    CholNumPtr num = chol_get_factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    if (!num->L) {
      fill_nan(zd + s * nnz, nnz);
      return "";
    }
    return error_text(chol_selinv(pat.get(), num.get(), nnz, zd + s * nnz));
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodSelinvBatchedF64, SelinvBatchedF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax [B,nnz]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // z [B,nnz]
                                  .Attr<int64_t>("n"));

// ---------------------------------------------------------------------------
// updown-solve handler:  (Ai, Aj, Ax, C[n,k], b[n(,nrhs)]; mode, downdate)
//     -> x solving (A ± C C') x = b, and out_logdet = log|A ± C C'|.
//
// The base matrix A's factor comes from the cache. Each call copies that
// factor's simplicial LDL' form, applies cholmod_updown for the rank-k
// modification C C', and solves — much cheaper than refactoring A ± C C' from
// scratch when A is held fixed and only the low-rank term C varies. `downdate=0`
// adds C C', `downdate=1` subtracts it. The cached factor is never mutated, so
// the op is a pure function of its inputs. C is dense (n, k), row-major.
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

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  auto answer_nan = [&] {
    fill_nan(x->typed_data(), n * nrhs);
    fill_nan(out_logdet->typed_data(), 1);
    return ffi::Error::Success();
  };
  if (!num->L) return answer_nan();

  // cholmod_updown requires a simplicial LDL' factor (it cannot update a
  // supernodal or LL' factor), and it mutates the factor, so it works on a
  // per-call copy of the cached factor's LDL' form.
  ffi::Error ldl_err = chol_ensure_ldl(num.get());
  if (ldl_err.failure()) return ldl_err;
  cholmod_common* c = t_chol.get();
  cholmod_factor* Lupd = cholmod_copy_factor(num->Lldl, c);
  if (!Lupd) return ffi::Error::Internal("sparsax: cholmod_copy_factor failed");
  auto free_upd = [c](cholmod_factor* f) { cholmod_free_factor(&f, c); };
  std::unique_ptr<cholmod_factor, decltype(free_upd)> upd_guard(Lupd, free_upd);

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
  const int32_t* Perm = static_cast<const int32_t*>(Lupd->Perm);
  for (int64_t kk = 0; kk < n; ++kk) {
    int64_t orig = Perm[kk];
    for (int64_t j = 0; j < k; ++j) Ccol[kk + j * n] = Cin[orig * k + j];
  }
  Cd.x = Ccol.data();

  cholmod_sparse* Cs = cholmod_dense_to_sparse(&Cd, /*values=*/1, c);
  if (!Cs)
    return ffi::Error::Internal("sparsax: cholmod_dense_to_sparse failed");

  int ok = cholmod_updown(downdate ? 0 : 1, Cs, Lupd, c);
  cholmod_free_sparse(&Cs, c);
  if (nan_on_failure() && c->status == CHOLMOD_NOT_POSDEF) return answer_nan();
  if (!ok || c->status < CHOLMOD_OK)
    return ffi::Error::Internal("sparsax: cholmod_updown failed (status " +
                                std::to_string(c->status) + ")");

  double ld = factor_logdet(Lupd);
  if (!std::isfinite(ld)) {
    if (nan_on_failure()) return answer_nan();
    return ffi::Error::Internal(
        "sparsax: updated matrix A ± C C' is not positive definite");
  }
  out_logdet->typed_data()[0] = ld;

  return chol_solve_one(Lupd, static_cast<int>(mode), b.typed_data(), n, nrhs,
                        x->typed_data());
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

