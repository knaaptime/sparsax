// argmax_logprob: the GHK simulator on a sparse-precision Gaussian.
#include <array>

#include "cholmod_core.h"
#include "ghk.h"
#include "sobol_directions.h"

// ---------------------------------------------------------------------------
// argmax log-probability (GHK) handler:
//   (Ai, Aj, Ax, mean[P,n], chosen[P], competitors[P,K], keys[P];
//    n_draws, seed, want_grad, sobol_dims) -> logp[P], grad[P,n] (or [0])
//
// For each path p, log P(argmax_k x_k = chosen[p]) where x ~ N(mean[p], A^{-1})
// and A, given in COO, is the SPD *precision*: the choice probability of a
// multinomial probit whose utilities form a Gaussian Markov random field.  It
// is estimated by the GHK simulator with n_draws draws per path.
//
// The probability is that the differences d_k = x_k - x_j (k != j, j chosen)
// are all negative.  Writing x_{-j} = d + x_j 1, the precision of d is
//     P_d = A_{-j,-j} - v v'/c,   v = (A 1)_{-j},   c = 1'A1,
// sparse minus rank one.  With A_{-j,-j} = L L' and w = L^{-1} v,
// P_d = L G G' L', where G G' = I - w w'/c has the closed-form semiseparable
// factor G_kt = w_k beta_t (k > t).  GHK draws z ~ N(0, I) and sets
// d = mu + L^{-T} G^{-T} z backward, last variable first: each step reads one
// column of L and a running sum s = sum_{k>t} w_k y_k, so a draw costs
// O(nnz(L)) rather than the O(n^2) of the dense differenced GHK.
//
// GHK's variance depends on the order in which the constraints are imposed:
// the alternatives most likely to beat the chosen one should be drawn first.
// competitors[p] lists them, most competitive first (-1 marks an unused slot).
// They are moved to the end of the elimination order by deleting their rows
// from the factor (cholmod_rowdel) and appending a dense K x K Schur block,
// which keeps the fill-reducing order for everything else.  The chosen
// alternative's row is deleted the same way, which is how A_{-j,-j} comes from
// A's single cached factor.  Each path works on its own copy of the factor's
// simplicial LDL' form; the cached factor is never mutated.
//
// Draws come from a counter-based hash of (seed, keys[p], draw, step), so a
// path's draws are the same on every call and the simulated log-likelihood is
// a smooth function of the means, as simulated maximum likelihood needs.  The
// competitor lists fix the variable order, so hold them fixed while optimizing.
// Paths run in parallel (SPARSAX_NUM_THREADS).
//
// With want_grad, grad[p] = d logp[p] / d mean[p], by a reverse sweep through
// the recursion.  Per draw and step it needs only dz/db, the slope of the
// truncated draw z = Phi^{-1}(u Phi(b)) in its bound, and the inverse Mills
// ratio phi(b)/Phi(b); everything else is linear with fixed coefficients, and
// the adjoint of each step's "below" sum scatters back through the same factor
// columns, so the sweep costs about as much as the forward pass.
// ---------------------------------------------------------------------------

namespace ghk {

constexpr double kSqrt2 = 1.41421356237309504880;
constexpr double kLogSqrt2Pi = 0.91893853320467274178;

// log Phi(b), accurate in both tails.
inline double log_ndtr(double b) {
  if (b > 0.0) return std::log1p(-0.5 * std::erfc(b / kSqrt2));
  if (b > -30.0) return std::log(0.5 * std::erfc(-b / kSqrt2));
  const double b2 = b * b;
  const double s = 1.0 - 1.0 / b2 + 3.0 / (b2 * b2) - 15.0 / (b2 * b2 * b2) +
                   105.0 / (b2 * b2 * b2 * b2);
  return -0.5 * b2 - std::log(-b) - kLogSqrt2Pi + std::log(s);
}

// Phi^{-1}(p) from log p (p <= 1): Wichura's AS241 (PPND16), taking the tail
// branch's r = sqrt(-log p) straight from log p so tiny p never underflows.
inline double ndtri_log(double logp) {
  const double p = std::exp(logp);
  const double q = p - 0.5;
  double r, num, den;
  if (std::fabs(q) <= 0.425) {
    r = 0.180625 - q * q;
    num = (((((((2509.0809287301226727 * r + 33430.575583588128105) * r +
                67265.770927008700853) * r + 45921.953931549871457) * r +
              13731.693765509461125) * r + 1971.5909503065514427) * r +
            133.14166789178437745) * r + 3.387132872796366608);
    den = (((((((5226.495278852545925 * r + 28729.085735721942674) * r +
                39307.89580009271061) * r + 21213.794301586595867) * r +
              5394.1960214247511077) * r + 687.1870074920579083) * r +
            42.313330701600911252) * r + 1.0);
    return q * num / den;
  }
  r = q < 0.0 ? std::sqrt(-logp) : std::sqrt(-std::log1p(-p));
  if (r <= 5.0) {
    r -= 1.6;
    num = (((((((7.7454501427834140764e-4 * r + 0.0227238449892691845833) * r +
                0.24178072517745061177) * r + 1.27045825245236838258) * r +
              3.64784832476320460504) * r + 5.7694972214606914055) * r +
            4.6303378461565452959) * r + 1.42343711074968357734);
    den = (((((((1.05075007164441684324e-9 * r + 5.475938084995344946e-4) * r +
                0.0151986665636164571966) * r + 0.14810397642748007459) * r +
              0.68976733498510000455) * r + 1.6763848301838038494) * r +
            2.05319162663775882187) * r + 1.0);
  } else {
    r -= 5.0;
    num = (((((((2.01033439929228813265e-7 * r + 2.71155556874348757815e-5) * r +
                0.0012426609473880784386) * r + 0.026532189526576123093) * r +
              0.29656057182850489123) * r + 1.7848265399172913358) * r +
            5.4637849111641143699) * r + 6.6579046435011037772);
    den = (((((((2.04426310338993978564e-15 * r + 1.4215117583164458887e-7) * r +
                1.8463183175100546818e-5) * r + 7.868691311456132591e-4) * r +
              0.0148753612908506148525) * r + 0.13692988092273580531) * r +
            0.59983220655588793769) * r + 1.0);
  }
  const double x = num / den;
  return q < 0.0 ? -x : x;
}

inline double log_phi(double x) { return -0.5 * x * x - kLogSqrt2Pi; }

// splitmix64's finalizer.
inline uint64_t mix64(uint64_t x) {
  x ^= x >> 30;
  x *= 0xBF58476D1CE4E5B9ULL;
  x ^= x >> 27;
  x *= 0x94D049BB133111EBULL;
  x ^= x >> 31;
  return x;
}
constexpr uint64_t kGolden = 0x9E3779B97F4A7C15ULL;

// Hash state for (seed, key); one more mix gives a draw's base, and a step's
// uniform is one more again.
inline uint64_t path_state(uint64_t seed, uint64_t key) {
  return mix64(mix64(seed + kGolden) ^ (key + kGolden));
}
inline uint64_t draw_base(uint64_t state, uint64_t r) {
  return mix64(state ^ (r + kGolden));
}
// U(0, 1), never exactly 0 or 1.  52 bits, not 53: (2^53 - 1) + 0.5 is not a
// double and rounds up to 2^53, which would give exactly 1.
inline double uniform(uint64_t base, uint64_t step) {
  const uint64_t h = mix64(base ^ (step + kGolden));
  return (static_cast<double>(h >> 12) + 0.5) * 0x1.0p-52;
}

// Scrambled Sobol draws.  GHK averages a product over draws, so its variance
// responds to how evenly the draws fill the cube, above all in the first steps,
// where the rivals that matter are drawn.  Nested uniform (Owen) scrambling flips
// each digit by a hash of the digits above it.  That keeps the points' balance,
// makes each point uniform on the cube, so the estimator stays unbiased, and gives
// every (seed, key) its own independent points.

using SobolDirections = std::array<std::array<uint32_t, 32>, sobol::kDims>;

// Joe and Kuo's recurrence, with the direction numbers scaled to 32 bits.
inline const SobolDirections& sobol_directions() {
  static const SobolDirections directions = [] {
    SobolDirections V{};
    for (int i = 0; i < 32; ++i) V[0][i] = uint32_t{1} << (31 - i);
    for (int d = 1; d < sobol::kDims; ++d) {
      const sobol::Row& row = sobol::kRows[d - 1];
      const int s = row.s;
      for (int i = 0; i < s; ++i)
        V[d][i] = static_cast<uint32_t>(row.m[i]) << (31 - i);
      for (int i = s; i < 32; ++i) {
        uint32_t v = V[d][i - s] ^ (V[d][i - s] >> s);
        for (int k = 1; k < s; ++k)
          if ((row.a >> (s - 1 - k)) & 1) v ^= V[d][i - k];
        V[d][i] = v;
      }
    }
    return V;
  }();
  return directions;
}

// Dimension `dim` of the first R points, by Gray code; R <= 2^32.
template <class Visit>
inline void sobol_column(int64_t R, int dim, Visit visit) {
  const std::array<uint32_t, 32>& V = sobol_directions()[dim];
  uint32_t x = 0;
  for (int64_t r = 0; r < R; ++r) {
    if (r > 0) {
      int lowest_set = 0;
      while (!((r >> lowest_set) & 1)) ++lowest_set;
      x ^= V[lowest_set];
    }
    visit(r, x);
  }
}

// The prefix's leading 1 makes each node of the digit tree a distinct input.  In
// the first 2^16 points each dimension gives every point a node of its own at depth
// 16, so one hash there scrambles the low 16 digits exactly as sixteen would; past
// 2^16 points it still preserves the balance, as a shift within the node.
inline uint32_t owen_scramble(uint32_t x, uint64_t dim_seed) {
  uint32_t out = 0;
  uint64_t prefix = 1;
  for (int i = 31; i >= 16; --i) {
    const uint32_t digit = (x >> i) & 1u;
    const uint32_t flip =
        static_cast<uint32_t>(mix64(dim_seed + prefix * kGolden) >> 63);
    out = (out << 1) | (digit ^ flip);
    prefix = (prefix << 1) | digit;
  }
  const uint32_t low = static_cast<uint32_t>(mix64(dim_seed + prefix * kGolden) >> 48);
  return (out << 16) | ((x & 0xFFFFu) ^ low);
}

// One step's uniforms for all R draws: scrambled Sobol in the first sobol_dims
// steps, independent after them.  Depends on neither R nor the batch, so more
// draws extend the same points.  Below its 32 Sobol digits a point takes its bits
// from the draw's own hash, which is what the digit tree would give them.
inline void step_uniforms(int64_t R, uint64_t step, int64_t sobol_dims,
                          uint64_t state, const uint64_t* base, double* u) {
  if (static_cast<int64_t>(step) >= sobol_dims) {
    for (int64_t r = 0; r < R; ++r) u[r] = uniform(base[r], step);
    return;
  }
  const uint64_t dim_seed = mix64(mix64(~state) ^ (step + kGolden));
  sobol_column(R, static_cast<int>(step), [&](int64_t r, uint32_t x) {
    const uint64_t high = static_cast<uint64_t>(owen_scramble(x, dim_seed)) << 20;
    const uint64_t low = mix64(base[r] ^ (step + kGolden)) >> 44;
    u[r] = (static_cast<double>(high | low) + 0.5) * 0x1.0p-52;
  });
}

// In-place lower Cholesky of a dense row-major K x K matrix; false if not PD.
inline bool dense_chol(double* M, int64_t K) {
  for (int64_t j = 0; j < K; ++j) {
    double d = M[j * K + j];
    for (int64_t k = 0; k < j; ++k) d -= M[j * K + k] * M[j * K + k];
    if (!(d > 0.0)) return false;
    d = std::sqrt(d);
    M[j * K + j] = d;
    for (int64_t i = j + 1; i < K; ++i) {
      double v = M[i * K + j];
      for (int64_t k = 0; k < j; ++k) v -= M[i * K + k] * M[j * K + k];
      M[i * K + j] = v / d;
    }
    for (int64_t k = j + 1; k < K; ++k) M[j * K + k] = 0.0;
  }
  return true;
}

// Full symmetric CSC of the COO matrix (entries with i <= j, mirrored;
// duplicates summed), in the original ordering.
struct SymCSC {
  std::vector<int64_t> p;
  std::vector<int32_t> i;
  std::vector<double> x;
};

inline SymCSC build_sym_csc(const int32_t* Ai, const int32_t* Aj,
                            const double* Ax, int64_t nnz, int64_t n) {
  SymCSC S;
  std::vector<int64_t> cnt(n + 1, 0);
  for (int64_t k = 0; k < nnz; ++k) {
    if (Ai[k] > Aj[k]) continue;
    ++cnt[Aj[k] + 1];
    if (Ai[k] != Aj[k]) ++cnt[Ai[k] + 1];
  }
  for (int64_t c = 0; c < n; ++c) cnt[c + 1] += cnt[c];
  std::vector<int32_t> ri(cnt[n]);
  std::vector<double> rx(cnt[n]);
  std::vector<int64_t> fill(cnt.begin(), cnt.end() - 1);
  for (int64_t k = 0; k < nnz; ++k) {
    const int32_t a = Ai[k], b = Aj[k];
    if (a > b) continue;
    ri[fill[b]] = a;
    rx[fill[b]++] = Ax[k];
    if (a != b) {
      ri[fill[a]] = b;
      rx[fill[a]++] = Ax[k];
    }
  }
  // Sort each column by row and merge duplicates.
  S.p.assign(n + 1, 0);
  std::vector<std::pair<int32_t, double>> col;
  for (int64_t c = 0; c < n; ++c) {
    col.clear();
    for (int64_t t = cnt[c]; t < cnt[c + 1]; ++t) col.emplace_back(ri[t], rx[t]);
    std::sort(col.begin(), col.end(),
              [](const auto& u, const auto& v) { return u.first < v.first; });
    for (size_t t = 0; t < col.size(); ++t) {
      if (t > 0 && col[t].first == col[t - 1].first) {
        S.x.back() += col[t].second;
      } else {
        S.i.push_back(col[t].first);
        S.x.push_back(col[t].second);
      }
    }
    S.p[c + 1] = static_cast<int64_t>(S.i.size());
  }
  return S;
}

// Read-only inputs shared by every path of one call.
struct Shared {
  int64_t n, K, R;
  int64_t sobol_dims;        // steps drawn from the scrambled Sobol sequence
  uint64_t seed;
  const double* mean;        // [P, n]
  const int32_t* chosen;     // [P]
  const int32_t* comp;       // [P, K]
  const int64_t* keys;       // [P]
  const cholmod_factor* Lldl;  // A's simplicial LDL' factor (never mutated)
  const int32_t* Perm;       // permuted position -> original index
  std::vector<int32_t> iperm;  // original index -> permuted position
  SymCSC A;                  // A, full symmetric, original ordering
  std::vector<double> rowsum;  // A 1
  double total;              // 1'A1
};

// Per-thread scratch, sized once and reused across paths.
struct Work {
  std::vector<char> del, isC;
  std::vector<int32_t> Celim;  // competitors in elimination order (most competitive last)
  std::vector<double> b, X, Scc, Dsqrt, w, Gd, beta, mu, vC, wC, GdC, betaC, muC;
  std::vector<double> eN, eC, s, logw, below, u;
  std::vector<uint64_t> base;
  // reverse sweep: per step, the variable it drew and per-draw dz/db, phi/Phi
  std::vector<int64_t> step_var;  // >= 0: kept position t; < 0: competitor -1-a
  std::vector<double> dzdb, mills, eNbar, eCbar, sbar, ybar, bbar, belowbar, omega;
};

// One path: log P(argmax = chosen[p]).  Returns "" or an error message.
inline std::string path_logprob(const Shared& sh, int64_t p, Work& wk,
                                cholmod_common* c, double* out, double* grad) {
  const int64_t n = sh.n, R = sh.R;
  // Positive definiteness lost on this path: its answer, or an error.
  auto numeric_failure = [&](const char* message) -> std::string {
    if (!nan_on_failure()) return message;
    fill_nan(out, 1);
    if (grad) fill_nan(grad + p * n, n);
    return "";
  };
  const int32_t j = sh.chosen[p];
  if (j < 0 || j >= n) return "sparsax: chosen index out of range";
  const double* m = sh.mean + p * n;

  // Competitors: valid, distinct, not the chosen one.  Elimination order is
  // the reverse of the listed (draw) order.
  wk.del.assign(n, 0);
  wk.isC.assign(n, 0);
  std::vector<int32_t> listed;
  for (int64_t a = 0; a < sh.K; ++a) {
    const int32_t q = sh.comp[p * sh.K + a];
    if (q < 0) continue;
    if (q >= n) return "sparsax: competitor index out of range";
    if (q == j) return "sparsax: a path's competitors must not include its chosen index";
    if (wk.isC[q]) return "sparsax: duplicate competitor index";
    wk.isC[q] = 1;
    listed.push_back(q);
  }
  const int64_t Kp = static_cast<int64_t>(listed.size());
  if (Kp >= n) return "sparsax: competitors must leave at least one other alternative";
  wk.Celim.assign(listed.rbegin(), listed.rend());

  // Factor of A_{-j,-j} with the competitors decoupled: delete their rows.
  cholmod_factor* F = cholmod_copy_factor(const_cast<cholmod_factor*>(sh.Lldl), c);
  if (!F) return "sparsax: cholmod_copy_factor failed";
  auto free_f = [c](cholmod_factor* f) { cholmod_free_factor(&f, c); };
  std::unique_ptr<cholmod_factor, decltype(free_f)> guard(F, free_f);
  auto drop = [&](int32_t q) -> bool {
    const int32_t k = sh.iperm[q];
    wk.del[k] = 1;
    return cholmod_rowdel(static_cast<size_t>(k), nullptr, F, c) &&
           c->status >= CHOLMOD_OK;
  };
  if (!drop(j)) return "sparsax: cholmod_rowdel failed";
  for (int32_t q : wk.Celim)
    if (!drop(q)) return "sparsax: cholmod_rowdel failed";
  if (F->is_super || F->is_ll)
    return "sparsax: expected a simplicial LDL' factor after cholmod_rowdel";
  const int32_t* Fp = static_cast<const int32_t*>(F->p);
  const int32_t* Fi = static_cast<const int32_t*>(F->i);
  const int32_t* Fnz = static_cast<const int32_t*>(F->nz);
  const double* Fx = static_cast<const double*>(F->x);

  wk.Dsqrt.assign(n, 0.0);
  for (int64_t t = 0; t < n; ++t) {
    if (wk.del[t]) continue;
    if (Fi[Fp[t]] != t) return "sparsax: factor column does not start at its diagonal";
    const double d = Fx[Fp[t]];
    if (!(d > 0.0))
      return numeric_failure("sparsax: precision is not positive definite");
    wk.Dsqrt[t] = std::sqrt(d);
  }
  // b <- D^{-1/2} L_u^{-1} b over the kept positions (L_u unit lower).
  auto forward = [&](double* bb) {
    for (int64_t t = 0; t < n; ++t) {
      if (wk.del[t]) {
        bb[t] = 0.0;
        continue;
      }
      const double yt = bb[t];
      if (yt != 0.0)
        for (int64_t q = Fp[t] + 1, qe = Fp[t] + Fnz[t]; q < qe; ++q)
          if (!wk.del[Fi[q]]) bb[Fi[q]] -= Fx[q] * yt;
    }
    for (int64_t t = 0; t < n; ++t) bb[t] = wk.del[t] ? 0.0 : bb[t] / wk.Dsqrt[t];
  };

  // Schur block for the competitors: X' = D^{-1/2} L_u^{-1} A_{N,C},
  // Lcc Lcc' = A_CC - X X'.
  wk.X.assign(Kp * n, 0.0);
  wk.b.resize(n);
  std::vector<int32_t> cslot(Kp > 0 ? n : 0, -1);
  for (int64_t a = 0; a < Kp; ++a) cslot[wk.Celim[a]] = static_cast<int32_t>(a);
  wk.Scc.assign(Kp * Kp, 0.0);
  for (int64_t a = 0; a < Kp; ++a) {
    const int32_t q = wk.Celim[a];
    std::fill(wk.b.begin(), wk.b.end(), 0.0);
    for (int64_t t = sh.A.p[q]; t < sh.A.p[q + 1]; ++t) {
      const int32_t r = sh.A.i[t];
      if (r == j) continue;
      if (wk.isC[r]) {
        wk.Scc[cslot[r] * Kp + a] = sh.A.x[t];
        continue;
      }
      wk.b[sh.iperm[r]] += sh.A.x[t];
    }
    forward(wk.b.data());
    std::copy(wk.b.begin(), wk.b.end(), wk.X.begin() + a * n);
  }
  for (int64_t a = 0; a < Kp; ++a)
    for (int64_t b2 = 0; b2 <= a; ++b2) {
      double acc = 0.0;
      const double* xa = wk.X.data() + a * n;
      const double* xb = wk.X.data() + b2 * n;
      for (int64_t t = 0; t < n; ++t) acc += xa[t] * xb[t];
      wk.Scc[a * Kp + b2] -= acc;
    }
  if (Kp > 0 && !dense_chol(wk.Scc.data(), Kp))
    return numeric_failure("sparsax: competitor Schur block is not positive definite");
  const double* Lcc = wk.Scc.data();

  // Rank-one term: v = (A 1)_{-j}; w = L_full^{-1} v.
  wk.w.assign(n, 0.0);
  for (int64_t t = 0; t < n; ++t)
    if (!wk.del[t]) wk.w[t] = sh.rowsum[sh.Perm[t]];
  forward(wk.w.data());
  wk.wC.resize(Kp);
  for (int64_t a = 0; a < Kp; ++a) {
    double v = sh.rowsum[wk.Celim[a]];
    const double* xa = wk.X.data() + a * n;
    for (int64_t t = 0; t < n; ++t) v -= xa[t] * wk.w[t];
    for (int64_t b2 = 0; b2 < a; ++b2) v -= Lcc[a * Kp + b2] * wk.wC[b2];
    wk.wC[a] = v / Lcc[a * Kp + a];
  }
  // Semiseparable factor of I - w w'/c, forward over [kept positions, competitors].
  wk.Gd.assign(n, 1.0);
  wk.beta.assign(n, 0.0);
  wk.GdC.resize(Kp);
  wk.betaC.resize(Kp);
  double sigma = -1.0 / sh.total;
  auto semisep = [&](double wt, double* gd, double* bt) -> bool {
    const double g = 1.0 + sigma * wt * wt;
    if (!(g > 0.0)) return false;
    *gd = std::sqrt(g);
    *bt = sigma * wt / *gd;
    sigma /= g;
    return true;
  };
  for (int64_t t = 0; t < n; ++t)
    if (!wk.del[t] && !semisep(wk.w[t], &wk.Gd[t], &wk.beta[t]))
      return numeric_failure("sparsax: rank-one correction lost positive definiteness");
  for (int64_t a = 0; a < Kp; ++a)
    if (!semisep(wk.wC[a], &wk.GdC[a], &wk.betaC[a]))
      return numeric_failure("sparsax: rank-one correction lost positive definiteness");

  // Means of the differences.
  wk.mu.assign(n, 0.0);
  for (int64_t t = 0; t < n; ++t)
    if (!wk.del[t]) wk.mu[t] = m[sh.Perm[t]] - m[j];
  wk.muC.resize(Kp);
  for (int64_t a = 0; a < Kp; ++a) wk.muC[a] = m[wk.Celim[a]] - m[j];

  // GHK, backward over [kept positions, competitors]: competitors first.
  wk.eN.resize(n * R);
  wk.eC.resize(Kp * R);
  wk.s.assign(R, 0.0);
  wk.logw.assign(R, 0.0);
  wk.below.resize(R);
  wk.u.resize(R);
  wk.base.resize(R);
  const uint64_t state =
      path_state(sh.seed, static_cast<uint64_t>(sh.keys[p]));
  for (int64_t r = 0; r < R; ++r) wk.base[r] = draw_base(state, r);
  double* s = wk.s.data();
  double* logw = wk.logw.data();
  double* below = wk.below.data();
  const int64_t n_steps = (n - 1 - Kp) + Kp;  // every alternative but the chosen
  if (grad) {
    wk.step_var.resize(n_steps);
    wk.dzdb.resize(n_steps * R);
    wk.mills.resize(n_steps * R);
  }
  uint64_t step = 0;
  auto ghk_step = [&](double ltt, double mut, double gd, double bt, double wt,
                      double* e, int64_t var) {
    double* dz = grad ? wk.dzdb.data() + step * R : nullptr;
    double* ml = grad ? wk.mills.data() + step * R : nullptr;
    step_uniforms(R, step, sh.sobol_dims, state, wk.base.data(), wk.u.data());
    for (int64_t r = 0; r < R; ++r) {
      const double bnd = gd * (-ltt * mut + below[r]) + bt * s[r];
      const double u = wk.u[r];
      double z;
      if (bnd > 8.0) {  // Phi(bnd) rounds to 1: an untruncated draw
        z = std::min(ndtri_log(std::log(u)), bnd);
        if (grad) dz[r] = ml[r] = 0.0;
      } else {
        const double lp = log_ndtr(bnd);
        logw[r] += lp;
        z = std::min(ndtri_log(std::log(u) + lp), bnd);
        if (grad) {
          // u Phi(b) = Phi(z), so dz/db = Phi(z) phi(b) / (Phi(b) phi(z)).
          ml[r] = std::exp(log_phi(bnd) - lp);
          dz[r] = std::exp(log_ndtr(z) - lp + log_phi(bnd) - log_phi(z));
        }
      }
      const double y = (z - bt * s[r]) / gd;
      e[r] = (y - below[r]) / ltt;
      s[r] += wt * y;
    }
    if (grad) wk.step_var[step] = var;
    ++step;
  };
  for (int64_t a = Kp - 1; a >= 0; --a) {
    std::fill(below, below + R, 0.0);
    for (int64_t b2 = a + 1; b2 < Kp; ++b2) {
      const double l = Lcc[b2 * Kp + a];
      const double* eb = wk.eC.data() + b2 * R;
      for (int64_t r = 0; r < R; ++r) below[r] += l * eb[r];
    }
    ghk_step(Lcc[a * Kp + a], wk.muC[a], wk.GdC[a], wk.betaC[a], wk.wC[a],
             wk.eC.data() + a * R, -1 - a);
  }
  for (int64_t t = n - 1; t >= 0; --t) {
    if (wk.del[t]) continue;
    std::fill(below, below + R, 0.0);
    const double dsq = wk.Dsqrt[t];
    for (int64_t q = Fp[t] + 1, qe = Fp[t] + Fnz[t]; q < qe; ++q) {
      const int32_t row = Fi[q];
      if (wk.del[row]) continue;
      const double l = Fx[q] * dsq;
      const double* er = wk.eN.data() + static_cast<int64_t>(row) * R;
      for (int64_t r = 0; r < R; ++r) below[r] += l * er[r];
    }
    for (int64_t a = 0; a < Kp; ++a) {
      const double l = wk.X[a * n + t];
      if (l == 0.0) continue;
      const double* ea = wk.eC.data() + a * R;
      for (int64_t r = 0; r < R; ++r) below[r] += l * ea[r];
    }
    ghk_step(dsq, wk.mu[t], wk.Gd[t], wk.beta[t], wk.w[t],
             wk.eN.data() + t * R, t);
  }
  double mx = -std::numeric_limits<double>::infinity();
  for (int64_t r = 0; r < R; ++r) mx = std::max(mx, logw[r]);
  double acc = 0.0;
  for (int64_t r = 0; r < R; ++r) acc += std::exp(logw[r] - mx);
  *out = mx + std::log(acc / static_cast<double>(R));
  if (!grad) return "";

  // Reverse sweep.  d logp / d logw_r = omega_r, the normalized weights; each
  // step's log Phi(b) enters logw directly.  Adjoints: eNbar / eCbar collect
  // d/de from the later steps whose "below" sums read e; sbar is d/ds of the
  // running sum after the step being undone.
  wk.omega.resize(R);
  for (int64_t r = 0; r < R; ++r) wk.omega[r] = std::exp(logw[r] - mx) / acc;
  wk.eNbar.assign(n * R, 0.0);
  wk.eCbar.assign(Kp * R, 0.0);
  wk.sbar.assign(R, 0.0);
  wk.ybar.resize(R);
  wk.bbar.resize(R);
  wk.belowbar.resize(R);
  double* g = grad + p * n;
  std::fill(g, g + n, 0.0);
  double gsum = 0.0;
  for (int64_t st = n_steps - 1; st >= 0; --st) {
    const int64_t var = wk.step_var[st];
    const bool comp_step = var < 0;
    const int64_t a = comp_step ? -1 - var : -1;
    const int64_t t = comp_step ? -1 : var;
    const double ltt = comp_step ? Lcc[a * Kp + a] : wk.Dsqrt[t];
    const double gd = comp_step ? wk.GdC[a] : wk.Gd[t];
    const double bt = comp_step ? wk.betaC[a] : wk.beta[t];
    const double wt = comp_step ? wk.wC[a] : wk.w[t];
    const double* ebar = comp_step ? wk.eCbar.data() + a * R : wk.eNbar.data() + t * R;
    const double* dz = wk.dzdb.data() + st * R;
    const double* ml = wk.mills.data() + st * R;
    double mubar = 0.0;
    for (int64_t r = 0; r < R; ++r) {
      // e = (y - below)/ltt and s' = s + wt y
      const double yb = ebar[r] / ltt + wk.sbar[r] * wt;
      double belb = -ebar[r] / ltt;
      // y = (z - bt s)/gd
      const double zb = yb / gd;
      double sb = wk.sbar[r] - yb * bt / gd;
      // z = Phi^{-1}(u Phi(b)), and log Phi(b) adds to logw
      const double bb = zb * dz[r] + wk.omega[r] * ml[r];
      // b = gd (-ltt mu + below) + bt s
      mubar += -gd * ltt * bb;
      belb += gd * bb;
      sb += bt * bb;
      wk.sbar[r] = sb;
      wk.belowbar[r] = belb;
    }
    // Scatter d/d(below) into the e's that the forward step read.
    const double* bel = wk.belowbar.data();
    if (comp_step) {
      for (int64_t b2 = a + 1; b2 < Kp; ++b2) {
        const double l = Lcc[b2 * Kp + a];
        double* eb = wk.eCbar.data() + b2 * R;
        for (int64_t r = 0; r < R; ++r) eb[r] += l * bel[r];
      }
      g[wk.Celim[a]] += mubar;
    } else {
      const double dsq = wk.Dsqrt[t];
      for (int64_t q = Fp[t] + 1, qe = Fp[t] + Fnz[t]; q < qe; ++q) {
        const int32_t row = Fi[q];
        if (wk.del[row]) continue;
        const double l = Fx[q] * dsq;
        double* eb = wk.eNbar.data() + static_cast<int64_t>(row) * R;
        for (int64_t r = 0; r < R; ++r) eb[r] += l * bel[r];
      }
      for (int64_t a2 = 0; a2 < Kp; ++a2) {
        const double l = wk.X[a2 * n + t];
        if (l == 0.0) continue;
        double* eb = wk.eCbar.data() + a2 * R;
        for (int64_t r = 0; r < R; ++r) eb[r] += l * bel[r];
      }
      g[sh.Perm[t]] += mubar;
    }
    gsum += mubar;
  }
  g[j] -= gsum;  // mu_k = mean_k - mean_j
  return "";
}

}  // namespace ghk

static ffi::Error ArgmaxLogprobF64Impl(
    ffi::Buffer<ffi::S32> Ai, ffi::Buffer<ffi::S32> Aj,
    ffi::Buffer<ffi::F64> Ax, ffi::Buffer<ffi::F64> mean,
    ffi::Buffer<ffi::S32> chosen, ffi::Buffer<ffi::S32> competitors,
    ffi::Buffer<ffi::S64> keys, ffi::ResultBuffer<ffi::F64> logp,
    ffi::ResultBuffer<ffi::F64> grad, int64_t n_draws, int64_t seed,
    int64_t want_grad, int64_t sobol_dims) {
  auto mdims = mean.dimensions();
  if (mdims.size() != 2)
    return ffi::Error::InvalidArgument("sparsax: mean must be 2D (paths, n)");
  const int64_t P = mdims[0], n = mdims[1];
  const int64_t nnz = static_cast<int64_t>(Ai.element_count());
  if (static_cast<int64_t>(Aj.element_count()) != nnz ||
      static_cast<int64_t>(Ax.element_count()) != nnz)
    return ffi::Error::InvalidArgument(
        "sparsax: Ai, Aj, Ax must have the same length");
  auto cdims = competitors.dimensions();
  if (static_cast<int64_t>(chosen.element_count()) != P ||
      static_cast<int64_t>(keys.element_count()) != P || cdims.size() != 2 ||
      cdims[0] != P)
    return ffi::Error::InvalidArgument(
        "sparsax: chosen, competitors and keys must have one row per path");
  if (n_draws < 1) return ffi::Error::InvalidArgument("sparsax: n_draws must be >= 1");
  if (sobol_dims < 0 || sobol_dims > sobol::kDims)
    return ffi::Error::InvalidArgument("sparsax: invalid number of Sobol dimensions");
  if (sobol_dims > 0 && n_draws > (int64_t{1} << 32))
    return ffi::Error::InvalidArgument(
        "sparsax: the Sobol sequence has at most 2**32 points");
  if (want_grad && static_cast<int64_t>(grad->element_count()) != P * n)
    return ffi::Error::InvalidArgument("sparsax: grad must have shape (paths, n)");
  if (P == 0) return ffi::Error::Success();

  std::string err;
  auto pat = chol_get_pattern(Ai.typed_data(), Aj.typed_data(), nnz, n, &err);
  if (!pat) return ffi::Error::InvalidArgument(err);
  CholNumPtr num = chol_get_factor(pat, Ax.typed_data(), nnz, &err);
  if (!num) return ffi::Error::Internal(err);
  if (!num->L) {
    fill_nan(logp->typed_data(), P);
    if (want_grad) fill_nan(grad->typed_data(), P * n);
    return ffi::Error::Success();
  }
  ffi::Error ldl_err = chol_ensure_ldl(num.get());
  if (ldl_err.failure()) return ldl_err;

  ghk::Shared sh;
  sh.n = n;
  sh.K = cdims[1];
  sh.R = n_draws;
  sh.sobol_dims = sobol_dims;
  sh.seed = static_cast<uint64_t>(seed);
  sh.mean = mean.typed_data();
  sh.chosen = chosen.typed_data();
  sh.comp = competitors.typed_data();
  sh.keys = keys.typed_data();
  sh.Lldl = num->Lldl;
  sh.Perm = static_cast<const int32_t*>(num->Lldl->Perm);
  sh.iperm.assign(n, 0);
  for (int64_t k = 0; k < n; ++k) sh.iperm[sh.Perm[k]] = static_cast<int32_t>(k);
  sh.A = ghk::build_sym_csc(Ai.typed_data(), Aj.typed_data(), Ax.typed_data(),
                            nnz, n);
  sh.rowsum.assign(n, 0.0);
  sh.total = 0.0;
  for (int64_t q = 0; q < n; ++q) {
    for (int64_t t = sh.A.p[q]; t < sh.A.p[q + 1]; ++t) sh.rowsum[q] += sh.A.x[t];
    sh.total += sh.rowsum[q];
  }
  if (!(sh.total > 0.0))
    return ffi::Error::InvalidArgument("sparsax: 1'A1 must be positive");

  double* out = logp->typed_data();
  double* gout = want_grad ? grad->typed_data() : nullptr;
  const unsigned T = static_cast<unsigned>(
      std::max<int64_t>(1, std::min<int64_t>(sparsax_num_threads(), P)));
  std::atomic<int64_t> next{0};
  const std::string run_err =
      run_on_threads(T, [&](std::atomic<bool>& failed) -> std::string {
        ghk::Work wk;
        cholmod_common* c = t_chol.get();
        for (int64_t p; !failed.load() && (p = next.fetch_add(1)) < P;) {
          std::string e = ghk::path_logprob(sh, p, wk, c, out + p, gout);
          if (!e.empty()) return e;
        }
        return "";
      });
  if (!run_err.empty()) return ffi::Error::Internal(run_err);
  return ffi::Error::Success();
}

XLA_FFI_DEFINE_HANDLER_SYMBOL(CholmodArgmaxLogprobF64, ArgmaxLogprobF64Impl,
                              ffi::Ffi::Bind()
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Ai
                                  .Arg<ffi::Buffer<ffi::S32>>()   // Aj
                                  .Arg<ffi::Buffer<ffi::F64>>()   // Ax
                                  .Arg<ffi::Buffer<ffi::F64>>()   // mean [P,n]
                                  .Arg<ffi::Buffer<ffi::S32>>()   // chosen [P]
                                  .Arg<ffi::Buffer<ffi::S32>>()   // competitors [P,K]
                                  .Arg<ffi::Buffer<ffi::S64>>()   // keys [P]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // logp [P]
                                  .Ret<ffi::Buffer<ffi::F64>>()   // grad [P,n] or [0]
                                  .Attr<int64_t>("n_draws")
                                  .Attr<int64_t>("seed")
                                  .Attr<int64_t>("want_grad")
                                  .Attr<int64_t>("sobol_dims"));

void ghk_uniforms(int64_t n_draws, int64_t n_steps, int64_t sobol_dims, int64_t seed,
                  int64_t key, double* out) {
  const uint64_t state =
      ghk::path_state(static_cast<uint64_t>(seed), static_cast<uint64_t>(key));
  std::vector<uint64_t> base(n_draws);
  for (int64_t r = 0; r < n_draws; ++r) base[r] = ghk::draw_base(state, r);
  std::vector<double> u(n_draws);
  for (int64_t step = 0; step < n_steps; ++step) {
    ghk::step_uniforms(n_draws, static_cast<uint64_t>(step), sobol_dims, state,
                       base.data(), u.data());
    for (int64_t r = 0; r < n_draws; ++r) out[r * n_steps + step] = u[r];
  }
}

void sobol_points(int64_t n_points, int64_t dims, double* out) {
  for (int64_t d = 0; d < dims; ++d)
    ghk::sobol_column(n_points, static_cast<int>(d), [&](int64_t r, uint32_t x) {
      out[r * dims + d] = static_cast<double>(x) * 0x1.0p-32;
    });
}

int sobol_max_dims() { return sobol::kDims; }
