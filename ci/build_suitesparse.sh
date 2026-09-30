#!/usr/bin/env bash
# Build the SuiteSparse components sparsax links (CHOLMOD, KLU, UMFPACK and
# their ordering libraries) for the macOS and Linux wheels.
#
# Why from source rather than Homebrew / yum:
#   - No OpenMP.  Homebrew's SuiteSparse links libomp, which delocate then
#     bundles into the wheel.  A second OpenMP runtime in the process (conda's,
#     or the one scikit-learn's wheels bundle) aborts it with
#     "OMP: Error #15: ... libomp.dylib already initialized".
#   - No Fortran.  Homebrew's AMD links libgfortran/libquadmath/libgcc_s,
#     which were bundled too; SuiteSparse needs no Fortran.
#   - A current SuiteSparse on Linux.  manylinux_2_28's (EL8) suitesparse-devel
#     is SuiteSparse 4.4 (2014), with ATLAS as BLAS.
#
# BLAS: Accelerate on macOS; the pthreads build of OpenBLAS on Linux.  Both
# are multithreaded without an OpenMP runtime (an OpenMP build of OpenBLAS
# would bring libgomp back into the wheel).  Threads matter for
# UMFPACK and for CHOLMOD's supernodal kernels on large supernodes (measured on
# Accelerate: ~28% faster UMFPACK on a denser graph, ~10% CHOLMOD on a 3-D
# grid, nothing on planar kNN graphs).  The pthreads build is also safe under
# sparsax's concurrent callers.  With many concurrent callers, cap BLAS
# threads (OPENBLAS_NUM_THREADS) to avoid oversubscription.
set -euo pipefail

VERSION="${SUITESPARSE_VERSION:-7.14.1}"
PREFIX="${SUITESPARSE_PREFIX:-/tmp/suitesparse}"
WORKDIR="${SUITESPARSE_WORKDIR:-/tmp}"
SRC="${WORKDIR}/SuiteSparse-${VERSION}"

case "$(uname -s)" in
  Darwin)
    BLAS_ARGS=(-DBLA_VENDOR=Apple)
    ;;
  Linux)
    # OpenBLAS from source too: EL8's openblas links libgfortran/libquadmath
    # (its LAPACK is compiled Fortran), which auditwheel would bundle.
    # NOFORTRAN + C_LAPACK builds LAPACK from its C translation; USE_THREAD
    # without USE_OPENMP is the pthreads build.  DYNAMIC_ARCH selects kernels
    # at runtime, so the wheel is not tied to the build machine's CPU.
    OPENBLAS_VERSION="${OPENBLAS_VERSION:-0.3.34}"
    OPENBLAS_SRC="${WORKDIR}/OpenBLAS-${OPENBLAS_VERSION}"
    if [ ! -d "$OPENBLAS_SRC" ]; then
      curl -fsSL "https://github.com/OpenMathLib/OpenBLAS/releases/download/v${OPENBLAS_VERSION}/OpenBLAS-${OPENBLAS_VERSION}.tar.gz" \
        | tar xz -C "$WORKDIR"
    fi
    case "$(uname -m)" in
      x86_64) OPENBLAS_TARGET=PRESCOTT ;;
      aarch64) OPENBLAS_TARGET=ARMV8 ;;
      *) echo "build_suitesparse.sh: unsupported arch $(uname -m)" >&2; exit 1 ;;
    esac
    OPENBLAS_FLAGS=(NOFORTRAN=1 C_LAPACK=1 USE_THREAD=1 USE_OPENMP=0
      NUM_THREADS=64 DYNAMIC_ARCH=1 TARGET="$OPENBLAS_TARGET" NO_STATIC=1)
    make -C "$OPENBLAS_SRC" -j"$(getconf _NPROCESSORS_ONLN)" "${OPENBLAS_FLAGS[@]}" libs netlib shared
    make -C "$OPENBLAS_SRC" "${OPENBLAS_FLAGS[@]}" PREFIX="$PREFIX" install
    OPENBLAS_LIB="${PREFIX}/lib/libopenblas.so"
    BLAS_ARGS=(-DBLA_VENDOR=OpenBLAS -DBLAS_LIBRARIES="$OPENBLAS_LIB" -DLAPACK_LIBRARIES="$OPENBLAS_LIB")
    ;;
  *)
    echo "build_suitesparse.sh: unsupported platform $(uname -s)" >&2
    exit 1
    ;;
esac

if [ ! -d "$SRC" ]; then
  curl -fsSL "https://github.com/DrTimothyAldenDavis/SuiteSparse/archive/refs/tags/v${VERSION}.tar.gz" \
    | tar xz -C "$WORKDIR"
fi

cmake -S "$SRC" -B "${WORKDIR}/suitesparse-build" \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_INSTALL_PREFIX="$PREFIX" \
  -DSUITESPARSE_ENABLE_PROJECTS="suitesparse_config;amd;camd;colamd;ccolamd;cholmod;btf;klu;umfpack" \
  -DSUITESPARSE_USE_OPENMP=OFF \
  -DSUITESPARSE_USE_FORTRAN=OFF \
  -DSUITESPARSE_USE_CUDA=OFF \
  -DSUITESPARSE_DEMOS=OFF \
  -DBUILD_TESTING=OFF \
  "${BLAS_ARGS[@]}"
cmake --build "${WORKDIR}/suitesparse-build" --parallel "$(getconf _NPROCESSORS_ONLN)"
cmake --install "${WORKDIR}/suitesparse-build"
