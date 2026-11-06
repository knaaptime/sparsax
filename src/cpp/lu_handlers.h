#ifndef SPARSAX_LU_HANDLERS_H_
#define SPARSAX_LU_HANDLERS_H_

#include "common.h"

// Build full (non-symmetric) CSC from COO: every entry, sorted by (col, row)
// with duplicates merged into one slot, plus `pos`, the COO-slot -> CSC-slot
// map used to scatter values on every refactorization. Shared by the KLU and
// UMFPACK paths below, which both take the whole matrix (no upper-triangle
// folding, unlike CHOLMOD). Returns false and sets *err on an out-of-range
// index; `who` names the backend in that message.
inline bool build_csc_from_coo(const int32_t* Ai, const int32_t* Aj, int64_t nnz,
                               int64_t n, const char* who,
                               std::vector<int32_t>& Ap,
                               std::vector<int32_t>& Ci,
                               std::vector<int64_t>& pos, int64_t& nnz_csc,
                               std::string* err) {
  for (int64_t k = 0; k < nnz; ++k) {
    if (Ai[k] < 0 || Ai[k] >= n || Aj[k] < 0 || Aj[k] >= n) {
      *err = std::string("sparsax(") + who +
             "): COO index out of range for matrix dimension " +
             std::to_string(n);
      return false;
    }
  }

  pos.assign(nnz, -1);

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

  nnz_csc = 0;
  for (size_t t = 0; t < trips.size(); ++t)
    if (t == 0 || trips[t].i != trips[t - 1].i || trips[t].j != trips[t - 1].j)
      ++nnz_csc;

  Ap.assign(n + 1, 0);
  Ci.assign(nnz_csc > 0 ? nnz_csc : 1, 0);

  int64_t slot = -1;
  for (size_t t = 0; t < trips.size(); ++t) {
    if (t == 0 || trips[t].i != trips[t - 1].i ||
        trips[t].j != trips[t - 1].j) {
      ++slot;
      Ci[slot] = trips[t].i;
      Ap[trips[t].j + 1] += 1;  // per-column counts, prefix-summed below
    }
    pos[trips[t].k] = slot;
  }
  for (int64_t j = 0; j < n; ++j) Ap[j + 1] += Ap[j];
  return true;
}

// ===========================================================================
// The handlers of an LU backend.  KLU and UMFPACK take the same arguments and give
// the same results, so one set of handlers serves both, through a small description
// of each backend.  `b` may be (n,) or (n, nrhs); trans != 0 solves A^T x = b, for
// the adjoint.  A batch has a leading axis, one factor per element, as many threads
// as it is worth (for_each_batch_element), and one shared analysis.
// ===========================================================================

template <class B>
static ffi::Error lu_invalid(const char* text) {
  return ffi::Error::InvalidArgument(std::string(B::kName) + ": " + text);
}

// A logdet from a factor, or NaN from a failed one.  "" or an error message.
template <class B>
static std::string lu_logdet_into(const typename B::Pattern* pat,
                                  const typename B::Holder* num, double* out) {
  const double ld = num && num->factored()
                        ? B::logdet(*pat, *num)
                        : std::numeric_limits<double>::quiet_NaN();
  // A failed factorization only gets here if on_failure="nan" let it through.
  if (num && num->factored() && !std::isfinite(ld) && !nan_on_failure())
    return std::string(B::kName) + ": matrix is singular";
  *out = ld;
  return "";
}

template <class B>
static ffi::Error LuSolveImpl(ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
                              ffi::Buffer<ffi::F64> Ax, ffi::Buffer<ffi::F64> b,
                              ffi::ResultBuffer<ffi::F64> x, int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return lu_invalid<B>("b must be 1D or 2D");
  const int64_t n = bdims[0];
  const int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  const int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return lu_invalid<B>("Ai, Aj, Ax must have the same length");

  // No global lock: `pat` and `num` are held (shared_ptr) for the duration, so a
  // concurrent clear_cache or eviction cannot free them mid-solve.
  std::string err;
  auto pat = B::pattern(Ai.typed_data(), Aj.typed_data(), Ax.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  auto num = B::factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  if (!num->factored()) {
    fill_nan(x->typed_data(), n * nrhs);
    return ffi::Error::Success();
  }
  return B::solve(*pat, *num, trans != 0, b.typed_data(), n, nrhs, x->typed_data());
}

template <class B>
static ffi::Error LuSolveBatchedImpl(ffi::Buffer<ffi::S32> Ai,
                                     ffi::Buffer<ffi::S32> Aj,
                                     ffi::Buffer<ffi::F64> Ax,
                                     ffi::Buffer<ffi::F64> b,
                                     ffi::ResultBuffer<ffi::F64> x, int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return lu_invalid<B>("batched b must be 2D or 3D");
  const int64_t batch = bdims[0];
  const int64_t n = bdims[1];
  const int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  const int64_t nnz = static_cast<int64_t>(Ai.element_count());
  auto axdims = Ax.dimensions();
  if (axdims.size() != 2 || axdims[0] != batch || axdims[1] != nnz)
    return lu_invalid<B>("batched Ax must have shape (batch, nnz) matching b");
  if (static_cast<int64_t>(Aj.element_count()) != nnz)
    return lu_invalid<B>("Ai, Aj must have the same length");

  std::string err;
  auto pat = B::pattern(Ai.typed_data(), Aj.typed_data(),
                        batch > 0 ? Ax.typed_data() : nullptr, nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);

  const double* Axd = Ax.typed_data();
  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  const int64_t bstride = n * nrhs;
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    auto num = B::factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    if (!num->factored()) {
      fill_nan(xd + s * bstride, bstride);
      return "";
    }
    return error_text(B::solve(*pat, *num, trans != 0, bd + s * bstride, n, nrhs,
                               xd + s * bstride));
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

template <class B>
static ffi::Error LuFactorImpl(ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
                               ffi::Buffer<ffi::F64> Ax,
                               ffi::ResultBuffer<ffi::S64> token, int64_t n) {
  const int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return lu_invalid<B>("Ai, Aj, Ax must have the same length");
  std::string err;
  auto pat = B::pattern(Ai.typed_data(), Aj.typed_data(), Ax.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  auto num = B::factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  B::tokens().issue({pat, num}, token->typed_data());
  return ffi::Error::Success();
}

template <class B>
static ffi::Error LuFactorBatchedImpl(ffi::Buffer<ffi::S32> Ai,
                                      ffi::Buffer<ffi::S32> Aj,
                                      ffi::Buffer<ffi::F64> Ax,
                                      ffi::ResultBuffer<ffi::S64> token, int64_t n) {
  int64_t batch = 0, nnz = 0;
  std::string err = check_batched_coo(Ai, Aj, Ax, &batch, &nnz);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto pat = B::pattern(Ai.typed_data(), Aj.typed_data(),
                        batch > 0 ? Ax.typed_data() : nullptr, nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  const double* Axd = Ax.typed_data();
  int64_t* td = token->typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    auto num = B::factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    B::tokens().issue({pat, num}, td + s * 2);
    return "";
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

template <class B>
static ffi::Error LuSolveFactorImpl(ffi::Buffer<ffi::S64> token,
                                    ffi::Buffer<ffi::F64> b,
                                    ffi::ResultBuffer<ffi::F64> x, int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 1 || bdims.size() > 2)
    return lu_invalid<B>("b must be 1D or 2D");
  const int64_t n = bdims[0];
  const int64_t nrhs = bdims.size() == 2 ? bdims[1] : 1;
  if (static_cast<int64_t>(token.element_count()) != 2)
    return lu_invalid<B>("factor token must be int64[2]");
  std::string err = check_token_issuer(token.typed_data(), 1, B::kIssuer);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto held = B::tokens().resolve(static_cast<uint64_t>(token.typed_data()[0]), &err);
  if (!held) return ffi::Error::InvalidArgument(err);
  if (!held->num) {
    fill_nan(x->typed_data(), n * nrhs);
    return ffi::Error::Success();
  }
  err = check_token_rows(held->pat->n, n);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  return B::solve(*held->pat, *held->num, trans != 0, b.typed_data(), n, nrhs,
                  x->typed_data());
}

// Two shapes: one token and a batch of b, or a token per element.
template <class B>
static ffi::Error LuSolveFactorBatchedImpl(ffi::Buffer<ffi::S64> token,
                                           ffi::Buffer<ffi::F64> b,
                                           ffi::ResultBuffer<ffi::F64> x,
                                           int64_t trans) {
  auto bdims = b.dimensions();
  if (bdims.size() < 2 || bdims.size() > 3)
    return lu_invalid<B>("batched b must be 2D or 3D");
  const int64_t batch = bdims[0];
  const int64_t n = bdims[1];
  const int64_t nrhs = bdims.size() == 3 ? bdims[2] : 1;
  auto tdims = token.dimensions();
  const bool token_batched =
      tdims.size() == 2 && tdims[0] == batch && tdims[1] == 2;
  const bool token_scalar = tdims.size() == 1 && tdims[0] == 2;
  if (!token_batched && !token_scalar)
    return lu_invalid<B>("token must be int64[2] or (batch, 2) matching b");
  std::string err =
      check_token_issuer(token.typed_data(), token_scalar ? 1 : batch, B::kIssuer);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);

  // Tokens resolve here, so that a bad one is an argument error before any work.
  std::vector<std::shared_ptr<typename B::Held>> held(token_scalar ? 1 : batch);
  for (size_t i = 0; i < held.size(); ++i) {
    held[i] = B::tokens().resolve(static_cast<uint64_t>(token.typed_data()[i * 2]), &err);
    if (!held[i]) return ffi::Error::InvalidArgument(err);
    if (held[i]->num) {
      err = check_token_rows(held[i]->pat->n, n);
      if (!err.empty()) return ffi::Error::InvalidArgument(err);
    }
  }

  const double* bd = b.typed_data();
  double* xd = x->typed_data();
  const int64_t bstride = n * nrhs;
  err = for_each_batch_element(batch, n, [&](int64_t s) -> std::string {
    const typename B::Held& h = *held[token_scalar ? 0 : s];
    if (!h.num) {
      fill_nan(xd + s * bstride, bstride);
      return "";
    }
    return error_text(B::solve(*h.pat, *h.num, trans != 0, bd + s * bstride, n, nrhs,
                               xd + s * bstride));
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

// A handler that reads something off the factor of these values: a logdet, say.
// read(pat, num, out) writes `width` doubles and returns "" or an error message.
template <class B, class Read>
static ffi::Error lu_read_factor(ffi::Buffer<ffi::S32>& Ai, ffi::Buffer<ffi::S32>& Aj,
                                 ffi::Buffer<ffi::F64>& Ax, int64_t n, double* out,
                                 Read read) {
  const int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return lu_invalid<B>("Ai, Aj, Ax must have the same length");
  std::string err;
  auto pat = B::pattern(Ai.typed_data(), Aj.typed_data(), Ax.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  auto num = B::factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  err = read(pat.get(), num.get(), out);
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

// The same over a batch of values, on as many threads as the batch is worth.
template <class B, class Read>
static ffi::Error lu_read_factors_batched(ffi::Buffer<ffi::S32>& Ai,
                                          ffi::Buffer<ffi::S32>& Aj,
                                          ffi::Buffer<ffi::F64>& Ax, int64_t n,
                                          double* out, int64_t width, Read read) {
  int64_t batch = 0, nnz = 0;
  std::string err = check_batched_coo(Ai, Aj, Ax, &batch, &nnz);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto pat = B::pattern(Ai.typed_data(), Aj.typed_data(),
                        batch > 0 ? Ax.typed_data() : nullptr, nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  const double* Axd = Ax.typed_data();
  err = for_each_batch_element(batch, nnz + n, [&](int64_t s) -> std::string {
    std::string e;
    auto num = B::factor(pat, Axd + s * nnz, nnz, &e);
    if (!num) return e;
    return read(pat.get(), num.get(), out + s * width);
  });
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

template <class B>
static ffi::Error LuLogdetImpl(ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
                               ffi::Buffer<ffi::F64> Ax,
                               ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  return lu_read_factor<B>(Ai, Aj, Ax, n, out->typed_data(), lu_logdet_into<B>);
}

template <class B>
static ffi::Error LuLogdetBatchedImpl(ffi::Buffer<ffi::S32> Ai,
                                      ffi::Buffer<ffi::S32> Aj,
                                      ffi::Buffer<ffi::F64> Ax,
                                      ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  return lu_read_factors_batched<B>(Ai, Aj, Ax, n, out->typed_data(), 1,
                                    lu_logdet_into<B>);
}

// [sign, log|det|], as numpy.linalg.slogdet gives them.
template <class B>
static std::string lu_slogdet_into(const typename B::Pattern* pat,
                                   const typename B::Holder* num, double* out) {
  std::string err = lu_logdet_into<B>(pat, num, out + 1);
  if (!err.empty()) return err;
  out[0] = num->factored() && std::isfinite(out[1])
               ? B::sign(*pat, *num)
               : std::numeric_limits<double>::quiet_NaN();
  return "";
}

template <class B>
static ffi::Error LuSlogdetImpl(ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
                                ffi::Buffer<ffi::F64> Ax,
                                ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  return lu_read_factor<B>(Ai, Aj, Ax, n, out->typed_data(), lu_slogdet_into<B>);
}

template <class B>
static ffi::Error LuSlogdetBatchedImpl(ffi::Buffer<ffi::S32> Ai,
                                       ffi::Buffer<ffi::S32> Aj,
                                       ffi::Buffer<ffi::F64> Ax,
                                       ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  return lu_read_factors_batched<B>(Ai, Aj, Ax, n, out->typed_data(), 2,
                                    lu_slogdet_into<B>);
}

// The library's reciprocal condition estimate: smallest over largest pivot.
template <class B>
static std::string lu_rcond_into(const typename B::Pattern* pat,
                                 const typename B::Holder* num, double* out) {
  *out = num->factored() ? B::rcond(*pat, *num)
                         : std::numeric_limits<double>::quiet_NaN();
  return "";
}

template <class B>
static ffi::Error LuRcondImpl(ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
                              ffi::Buffer<ffi::F64> Ax,
                              ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  return lu_read_factor<B>(Ai, Aj, Ax, n, out->typed_data(), lu_rcond_into<B>);
}

template <class B>
static ffi::Error LuRcondBatchedImpl(ffi::Buffer<ffi::S32> Ai,
                                     ffi::Buffer<ffi::S32> Aj,
                                     ffi::Buffer<ffi::F64> Ax,
                                     ffi::ResultBuffer<ffi::F64> out, int64_t n) {
  return lu_read_factors_batched<B>(Ai, Aj, Ax, n, out->typed_data(), 1,
                                    lu_rcond_into<B>);
}

template <class B>
static ffi::Error LuLogdetFactorImpl(ffi::Buffer<ffi::S64> token,
                                     ffi::ResultBuffer<ffi::F64> out) {
  if (static_cast<int64_t>(token.element_count()) != 2)
    return lu_invalid<B>("factor token must be int64[2]");
  std::string err = check_token_issuer(token.typed_data(), 1, B::kIssuer);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  auto held = B::tokens().resolve(static_cast<uint64_t>(token.typed_data()[0]), &err);
  if (!held) return ffi::Error::InvalidArgument(err);
  err = lu_logdet_into<B>(held->pat.get(), held->num.get(), out->typed_data());
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

template <class B>
static ffi::Error LuLogdetFactorBatchedImpl(ffi::Buffer<ffi::S64> token,
                                            ffi::ResultBuffer<ffi::F64> out) {
  auto tdims = token.dimensions();
  if (tdims.size() != 2 || tdims[1] != 2)
    return lu_invalid<B>("batched token must be (batch, 2)");
  const int64_t batch = tdims[0];
  std::string err = check_token_issuer(token.typed_data(), batch, B::kIssuer);
  if (!err.empty()) return ffi::Error::InvalidArgument(err);
  double* od = out->typed_data();
  for (int64_t s = 0; s < batch; ++s) {
    auto held =
        B::tokens().resolve(static_cast<uint64_t>(token.typed_data()[s * 2]), &err);
    if (!held) return ffi::Error::InvalidArgument(err);
    err = lu_logdet_into<B>(held->pat.get(), held->num.get(), od + s);
    if (!err.empty()) return ffi::Error::Internal(err);
  }
  return ffi::Error::Success();
}

// The bindings, the same for both backends, under each one's symbol names.
#define SPARSAX_COO_ARGS                \
  .Arg<ffi::Buffer<ffi::S32>>() /* Ai */ \
      .Arg<ffi::Buffer<ffi::S32>>() /* Aj */ \
      .Arg<ffi::Buffer<ffi::F64>>() /* Ax */

#define SPARSAX_DEFINE_LU_HANDLERS(Prefix, Backend)                                 \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##SolveF64, LuSolveImpl<Backend>,                                       \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Arg<ffi::Buffer<ffi::F64>>()                \
          .Ret<ffi::Buffer<ffi::F64>>()                                             \
          .Attr<int64_t>("trans"));                                                 \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##SolveBatchedF64, LuSolveBatchedImpl<Backend>,                         \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Arg<ffi::Buffer<ffi::F64>>()                \
          .Ret<ffi::Buffer<ffi::F64>>()                                             \
          .Attr<int64_t>("trans"));                                                 \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##FactorF64, LuFactorImpl<Backend>,                                     \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::S64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##FactorBatchedF64, LuFactorBatchedImpl<Backend>,                       \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::S64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Prefix##SolveFactorF64, LuSolveFactorImpl<Backend>, \
                                ffi::Ffi::Bind()                                    \
                                    .Arg<ffi::Buffer<ffi::S64>>()                   \
                                    .Arg<ffi::Buffer<ffi::F64>>()                   \
                                    .Ret<ffi::Buffer<ffi::F64>>()                   \
                                    .Attr<int64_t>("trans"));                       \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Prefix##SolveFactorBatchedF64,                      \
                                LuSolveFactorBatchedImpl<Backend>,                  \
                                ffi::Ffi::Bind()                                    \
                                    .Arg<ffi::Buffer<ffi::S64>>()                   \
                                    .Arg<ffi::Buffer<ffi::F64>>()                   \
                                    .Ret<ffi::Buffer<ffi::F64>>()                   \
                                    .Attr<int64_t>("trans"));                       \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##LogdetF64, LuLogdetImpl<Backend>,                                     \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::F64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##LogdetBatchedF64, LuLogdetBatchedImpl<Backend>,                       \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::F64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##SlogdetF64, LuSlogdetImpl<Backend>,                         \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::F64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##SlogdetBatchedF64, LuSlogdetBatchedImpl<Backend>,           \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::F64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##RcondF64, LuRcondImpl<Backend>,                             \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::F64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(                                                    \
      Prefix##RcondBatchedF64, LuRcondBatchedImpl<Backend>,               \
      ffi::Ffi::Bind() SPARSAX_COO_ARGS.Ret<ffi::Buffer<ffi::F64>>()                \
          .Attr<int64_t>("n"));                                                     \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Prefix##LogdetFactorF64, LuLogdetFactorImpl<Backend>, \
                                ffi::Ffi::Bind()                                    \
                                    .Arg<ffi::Buffer<ffi::S64>>()                   \
                                    .Ret<ffi::Buffer<ffi::F64>>());                 \
  XLA_FFI_DEFINE_HANDLER_SYMBOL(Prefix##LogdetFactorBatchedF64,                     \
                                LuLogdetFactorBatchedImpl<Backend>,                 \
                                ffi::Ffi::Bind()                                    \
                                    .Arg<ffi::Buffer<ffi::S64>>()                   \
                                    .Ret<ffi::Buffer<ffi::F64>>())

#endif  // SPARSAX_LU_HANDLERS_H_
