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
# are multithreaded without an OpenMP runtime (the OpenMP build of OpenBLAS,
# libopenblaso, would bring libgomp back into the wheel).  Threads matter for
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
    (dnf install -y openblas-devel || yum install -y openblas-devel)
    # FindBLAS's OpenBLAS vendor picks the serial libopenblas, so name the
    # pthreads library explicitly; SuiteSparse then uses it as given, with
    # BLA_VENDOR telling it which vendor it is.  Fail rather than fall back.
    OPENBLAS_P="$(ls /usr/lib64/libopenblasp.so /usr/lib/libopenblasp.so 2>/dev/null | head -n 1)"
    if [ -z "$OPENBLAS_P" ]; then
      echo "build_suitesparse.sh: pthreads OpenBLAS (libopenblasp.so) not found" >&2
      exit 1
    fi
    BLAS_ARGS=(-DBLA_VENDOR=OpenBLAS -DBLAS_LIBRARIES="$OPENBLAS_P" -DLAPACK_LIBRARIES="$OPENBLAS_P")
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
