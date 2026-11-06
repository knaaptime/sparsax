// What the Python module needs from the two LU backends.
#ifndef SPARSAX_LU_BACKENDS_H_
#define SPARSAX_LU_BACKENDS_H_

#include <cstddef>
#include <cstdint>
#include <string>

// KLU's work on a pattern, from which the Python module chooses a backend.
struct LuWork {
  double flops = 0.0, lnz = 0.0, unz = 0.0;
};
LuWork klu_pattern_work(const int32_t* Ai, const int32_t* Aj, int64_t nnz, int64_t n,
                        std::string* err);

void klu_set_cache_size(size_t cap);
void klu_clear_cache();
size_t klu_pattern_count();
int64_t klu_factorizations();

void umf_set_cache_size(size_t cap);
void umf_clear_cache();
size_t umf_pattern_count();
int64_t umf_factorizations();

#endif  // SPARSAX_LU_BACKENDS_H_
