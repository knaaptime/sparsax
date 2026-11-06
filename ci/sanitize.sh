#!/usr/bin/env bash
# Run the test suite under a sanitizer, with every batch on its own threads.
#
#   ci/sanitize.sh address|thread [pytest arguments]
#
# sparsax must be built with the same sanitizer and compiler first:
#
#   CXXFLAGS="${CXXFLAGS:-} -fsanitize=address -fno-omit-frame-pointer" \
#   LDFLAGS="${LDFLAGS:-} -fsanitize=address" \
#     pip install -e . --no-build-isolation --no-deps
#
# Python itself is not instrumented, so the sanitizer's runtime has to be injected
# before it starts.  This sees sparsax's own code only: SuiteSparse and BLAS are
# not instrumented.
set -euo pipefail

sanitizer="${1:?usage: ci/sanitize.sh address|thread [pytest arguments]}"
shift

case "$(uname -s):$sanitizer" in
  Darwin:address) runtime_name=libclang_rt.asan_osx_dynamic.dylib ;;
  Darwin:thread) runtime_name=libclang_rt.tsan_osx_dynamic.dylib ;;
  Linux:address) runtime_name=libasan.so ;;
  Linux:thread) runtime_name=libtsan.so ;;
  *)
    echo "ci/sanitize.sh: no $sanitizer sanitizer on $(uname -s)" >&2
    exit 1
    ;;
esac
runtime="$("${CXX:-c++}" -print-file-name="$runtime_name")"
if [ ! -f "$runtime" ]; then
  echo "ci/sanitize.sh: ${CXX:-c++} has no $runtime_name" >&2
  exit 1
fi

# Keep the injection for the tests' subprocesses (strip_env).  CPython and jaxlib
# never free some of their memory (detect_leaks), and jaxlib bundles its own copies
# of some libraries (detect_odr_violation).
export ASAN_OPTIONS=detect_leaks=0:detect_odr_violation=0:halt_on_error=1:strip_env=0
export TSAN_OPTIONS=halt_on_error=1:report_signal_unsafe=0:strip_env=0
export SPARSAX_TEST_PARALLEL_BATCHES=1
case "$(uname -s)" in
  Darwin) export DYLD_INSERT_LIBRARIES="$runtime" ;;
  Linux) export LD_PRELOAD="$runtime" ;;
esac
exec python -m pytest tests -q "$@"
