// The draws of argmax_logprob's GHK simulator, for the tests.
#ifndef SPARSAX_GHK_H_
#define SPARSAX_GHK_H_

#include <cstdint>

// One step's uniforms for each draw, row-major (n_draws, n_steps).
void ghk_uniforms(int64_t n_draws, int64_t n_steps, int64_t sobol_dims, int64_t seed,
                  int64_t key, double* out);
// The raw Sobol points, row-major (n_points, dims).
void sobol_points(int64_t n_points, int64_t dims, double* out);
int sobol_max_dims();

#endif  // SPARSAX_GHK_H_
