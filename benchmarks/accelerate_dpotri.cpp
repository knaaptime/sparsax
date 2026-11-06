// Why chol_block_inverse does not call dpotri.  Threaded Accelerate's dpotri and dtrtri
// return wrong values, intermittently and with no sign of failure, at n near 145 to
// 500.  On macOS 27.2 with an M3 Max: 166 of 3000 calls at n = 145, none at n = 100,
// and none with VECLIB_MAXIMUM_THREADS=1.
//
//   clang++ -O2 -std=c++17 accelerate_dpotri.cpp -framework Accelerate -o dpotri && ./dpotri
//
// No CHOLMOD and no sparsax here: factor a fixed SPD matrix once, then invert from that
// factor again and again, and compare each answer with a plain triple loop.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
extern "C" {
void dpotrf_(const char*, const int*, double*, const int*, int*);
void dpotri_(const char*, const int*, double*, const int*, int*);
void dtrtri_(const char*, const char*, const int*, double*, const int*, int*);
void dlauum_(const char*, const int*, double*, const int*, int*);
}
static void trial(int n, int lda, int reps) {
  std::vector<double> A((size_t)lda * n, 0.0);
  srand48(1);
  for (int c = 0; c < n; ++c) for (int r = c; r < n; ++r) A[r + (size_t)c * lda] = (r == c) ? n + 1.0 : drand48() - 0.5;
  int info = 0; dpotrf_("L", &n, A.data(), &lda, &info);
  // reference: naive inverse of L, then L^-T L^-1
  std::vector<double> Li((size_t)n * n, 0.0), Ref((size_t)n * n, 0.0);
  for (int c = 0; c < n; ++c) { Li[c + (size_t)c * n] = 1.0 / A[c + (size_t)c * lda];
    for (int r = c + 1; r < n; ++r) { double acc = 0; for (int k = c; k < r; ++k) acc += A[r + (size_t)k * lda] * Li[k + (size_t)c * n]; Li[r + (size_t)c * n] = -acc / A[r + (size_t)r * lda]; } }
  for (int c = 0; c < n; ++c) for (int r = c; r < n; ++r) { double acc = 0; for (int k = r; k < n; ++k) acc += Li[k + (size_t)r * n] * Li[k + (size_t)c * n]; Ref[r + (size_t)c * n] = acc; }
  auto err = [&](const std::vector<double>& B) { double e = 0; for (int c = 0; c < n; ++c) for (int r = c; r < n; ++r) e = std::max(e, std::fabs(B[r + (size_t)c * lda] - Ref[r + (size_t)c * n])); return e; };
  long bad_potri = 0, bad_pair = 0; double w1 = 0, w2 = 0;
  std::vector<double> B;
  for (int rep = 0; rep < reps; ++rep) {
    B = A; dpotri_("L", &n, B.data(), &lda, &info); double e = err(B); if (e > 1e-9) { bad_potri++; w1 = std::max(w1, e); }
    B = A; dtrtri_("L", "N", &n, B.data(), &lda, &info); dlauum_("L", &n, B.data(), &lda, &info); e = err(B); if (e > 1e-9) { bad_pair++; w2 = std::max(w2, e); }
  }
  printf("n=%4d lda=%4d: dpotri wrong %5ld/%d (worst %.1e)   dtrtri+dlauum wrong %5ld/%d (worst %.1e)\n", n, lda, bad_potri, reps, w1, bad_pair, reps, w2);
  fflush(stdout);
}
int main() {
  for (int n : {32, 64, 100, 145, 200, 300, 500, 1000}) { int reps = n <= 300 ? 3000 : (n <= 500 ? 500 : 100); trial(n, n, reps); trial(n, n + 37, reps); }
}
