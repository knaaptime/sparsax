// Every FFI handler: the name JAX calls it by, and its symbol.
#ifndef SPARSAX_HANDLERS_H_
#define SPARSAX_HANDLERS_H_

#include "xla/ffi/api/ffi.h"

#define SPARSAX_FFI_HANDLERS(X) \
  X("sparsax_argmax_logprob_f64", CholmodArgmaxLogprobF64) \
  X("sparsax_factor_batched_f64", CholmodFactorBatchedF64) \
  X("sparsax_factor_f64", CholmodFactorF64) \
  X("sparsax_factor_solve_batched_f64", CholmodFactorSolveBatchedF64) \
  X("sparsax_factor_solve_f64", CholmodFactorSolveF64) \
  X("sparsax_logdet_batched_f64", CholmodLogdetBatchedF64) \
  X("sparsax_logdet_f64", CholmodLogdetF64) \
  X("sparsax_logdet_factor_batched_f64", CholmodLogdetFactorBatchedF64) \
  X("sparsax_logdet_factor_f64", CholmodLogdetFactorF64) \
  X("sparsax_lu_factor_batched_f64", SparsaxLuFactorBatchedF64) \
  X("sparsax_lu_factor_f64", SparsaxLuFactorF64) \
  X("sparsax_lu_logdet_batched_f64", SparsaxLuLogdetBatchedF64) \
  X("sparsax_lu_logdet_f64", SparsaxLuLogdetF64) \
  X("sparsax_lu_logdet_factor_batched_f64", SparsaxLuLogdetFactorBatchedF64) \
  X("sparsax_lu_logdet_factor_f64", SparsaxLuLogdetFactorF64) \
  X("sparsax_lu_rcond_batched_f64", SparsaxLuRcondBatchedF64) \
  X("sparsax_lu_rcond_f64", SparsaxLuRcondF64) \
  X("sparsax_lu_slogdet_batched_f64", SparsaxLuSlogdetBatchedF64) \
  X("sparsax_lu_slogdet_f64", SparsaxLuSlogdetF64) \
  X("sparsax_lu_solve_batched_f64", SparsaxLuSolveBatchedF64) \
  X("sparsax_lu_solve_f64", SparsaxLuSolveF64) \
  X("sparsax_lu_solve_factor_batched_f64", SparsaxLuSolveFactorBatchedF64) \
  X("sparsax_lu_solve_factor_f64", SparsaxLuSolveFactorF64) \
  X("sparsax_rcond_batched_f64", CholmodRcondBatchedF64) \
  X("sparsax_rcond_f64", CholmodRcondF64) \
  X("sparsax_selinv_batched_f64", CholmodSelinvBatchedF64) \
  X("sparsax_selinv_f64", CholmodSelinvF64) \
  X("sparsax_solve_batched_f64", CholmodSolveBatchedF64) \
  X("sparsax_solve_f64", CholmodSolveF64) \
  X("sparsax_solve_factor_batched_f64", CholmodSolveFactorBatchedF64) \
  X("sparsax_solve_factor_f64", CholmodSolveFactorF64) \
  X("sparsax_umf_factor_batched_f64", SparsaxUmfFactorBatchedF64) \
  X("sparsax_umf_factor_f64", SparsaxUmfFactorF64) \
  X("sparsax_umf_logdet_batched_f64", SparsaxUmfLogdetBatchedF64) \
  X("sparsax_umf_logdet_f64", SparsaxUmfLogdetF64) \
  X("sparsax_umf_logdet_factor_batched_f64", SparsaxUmfLogdetFactorBatchedF64) \
  X("sparsax_umf_logdet_factor_f64", SparsaxUmfLogdetFactorF64) \
  X("sparsax_umf_rcond_batched_f64", SparsaxUmfRcondBatchedF64) \
  X("sparsax_umf_rcond_f64", SparsaxUmfRcondF64) \
  X("sparsax_umf_slogdet_batched_f64", SparsaxUmfSlogdetBatchedF64) \
  X("sparsax_umf_slogdet_f64", SparsaxUmfSlogdetF64) \
  X("sparsax_umf_solve_batched_f64", SparsaxUmfSolveBatchedF64) \
  X("sparsax_umf_solve_f64", SparsaxUmfSolveF64) \
  X("sparsax_umf_solve_factor_batched_f64", SparsaxUmfSolveFactorBatchedF64) \
  X("sparsax_umf_solve_factor_f64", SparsaxUmfSolveFactorF64) \
  X("sparsax_updown_solve_f64", CholmodUpdownSolveF64)

extern "C" XLA_FFI_Error* CholmodArgmaxLogprobF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodFactorSolveBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodFactorSolveF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodLogdetBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodLogdetF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodLogdetFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodLogdetFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuLogdetBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuLogdetF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuLogdetFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuLogdetFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuRcondBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuRcondF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuSlogdetBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuSlogdetF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuSolveBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuSolveF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuSolveFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxLuSolveFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodRcondBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodRcondF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodSelinvBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodSelinvF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodSolveBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodSolveF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodSolveFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodSolveFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfLogdetBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfLogdetF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfLogdetFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfLogdetFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfRcondBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfRcondF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfSlogdetBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfSlogdetF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfSolveBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfSolveF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfSolveFactorBatchedF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* SparsaxUmfSolveFactorF64(XLA_FFI_CallFrame* call_frame);
extern "C" XLA_FFI_Error* CholmodUpdownSolveF64(XLA_FFI_CallFrame* call_frame);

#endif  // SPARSAX_HANDLERS_H_
