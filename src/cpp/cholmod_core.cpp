#include "cholmod_core.h"

thread_local CholThreadLocal t_chol;

// BLAS for the supernodal selected inverse, under the names and the integer
// width SuiteSparse was configured with.  CMake defines
// SPARSAX_SUPERNODAL_SELINV on the platforms where these symbols resolve; see
// CMakeLists.txt.  Elsewhere the selected inverse keeps its simplicial form.
#if defined(SPARSAX_SUPERNODAL_SELINV) && defined(SUITESPARSE_BLAS) && \
    defined(SUITESPARSE_BLAS_INT)
#define SPARSAX_HAVE_SUPERNODAL_SELINV 1
using blas_int = SUITESPARSE_BLAS_INT;
extern "C" {
void SUITESPARSE_BLAS(dtrsm, DTRSM)(const char* side, const char* uplo,
                                    const char* trans, const char* diag,
                                    const blas_int* m, const blas_int* n,
                                    const double* alpha, const double* a,
                                    const blas_int* lda, double* b,
                                    const blas_int* ldb);
void SUITESPARSE_BLAS(dsymm, DSYMM)(const char* side, const char* uplo,
                                    const blas_int* m, const blas_int* n,
                                    const double* alpha, const double* a,
                                    const blas_int* lda, const double* b,
                                    const blas_int* ldb, const double* beta,
                                    double* c, const blas_int* ldc);
void SUITESPARSE_BLAS(dgemm, DGEMM)(const char* transa, const char* transb,
                                    const blas_int* m, const blas_int* n,
                                    const blas_int* k, const double* alpha,
                                    const double* a, const blas_int* lda,
                                    const double* b, const blas_int* ldb,
                                    const double* beta, double* c,
                                    const blas_int* ldc);
}
#endif

// A cholmod_sparse header over the pattern's shared upper-CSC structure and a
// caller-owned value array.  CHOLMOD reads the matrix; it never writes it.
static cholmod_sparse chol_upper_view(CholPattern* p, double* x) {
  cholmod_sparse A;
  std::memset(&A, 0, sizeof(A));
  A.nrow = static_cast<size_t>(p->n);
  A.ncol = static_cast<size_t>(p->n);
  A.nzmax = static_cast<size_t>(p->nnz_csc);
  A.p = p->Ap.data();
  A.i = p->Ar.data();
  A.x = x;
  A.stype = 1;
  A.itype = CHOLMOD_INT;
  A.xtype = CHOLMOD_REAL;
  A.dtype = CHOLMOD_DOUBLE;
  A.sorted = 1;
  A.packed = 1;
  return A;
}

// Build the upper-triangular CSC structure (sorted, duplicates merged) and run
// cholmod_analyze.  Caller holds g_chol_reg_mtx.  Returns nullptr and sets
// `err` on failure.
static std::shared_ptr<CholPattern> chol_create_pattern_locked(
    const int32_t* Ai, const int32_t* Aj, int64_t nnz, int64_t n,
    std::string* err) {
  for (int64_t k = 0; k < nnz; ++k) {
    if (Ai[k] < 0 || Ai[k] >= n || Aj[k] < 0 || Aj[k] >= n) {
      *err = "sparsax: COO index out of range for matrix dimension " +
             std::to_string(n);
      return nullptr;
    }
  }

  auto e = std::make_shared<CholPattern>();
  e->n = n;
  e->Ai.assign(Ai, Ai + nnz);
  e->Aj.assign(Aj, Aj + nnz);
  e->pos.assign(nnz, -1);

  // Upper-triangle entries sorted by (col, row); keep original k for mapping.
  struct Trip {
    int32_t i, j;
    int64_t k;
  };
  std::vector<Trip> upper;
  upper.reserve(nnz);
  for (int64_t k = 0; k < nnz; ++k)
    if (Ai[k] <= Aj[k]) upper.push_back({Ai[k], Aj[k], k});
  std::sort(upper.begin(), upper.end(), [](const Trip& a, const Trip& b) {
    return a.j != b.j ? a.j < b.j : a.i < b.i;
  });

  int64_t nnz_csc = 0;
  for (size_t t = 0; t < upper.size(); ++t)
    if (t == 0 || upper[t].i != upper[t - 1].i || upper[t].j != upper[t - 1].j)
      ++nnz_csc;

  e->Ap.assign(n + 1, 0);
  e->Ar.assign(nnz_csc > 0 ? nnz_csc : 1, 0);
  int64_t slot = -1;
  for (size_t t = 0; t < upper.size(); ++t) {
    if (t == 0 || upper[t].i != upper[t - 1].i ||
        upper[t].j != upper[t - 1].j) {
      ++slot;
      e->Ar[slot] = upper[t].i;
      e->Ap[upper[t].j + 1] += 1;  // per-column counts, prefix-summed below
    }
    e->pos[upper[t].k] = slot;
  }
  for (int64_t j = 0; j < n; ++j) e->Ap[j + 1] += e->Ap[j];
  e->nnz_csc = nnz_csc;

  cholmod_common* c = t_chol.get();
  const int ordering = g_chol_ordering.load(std::memory_order_relaxed);
  if (ordering == kCholOrderingAuto) {
    c->nmethods = 0;
  } else if (ordering == kCholOrderingBest) {
    c->nmethods = 3;
    c->method[0].ordering = CHOLMOD_AMD;
    c->method[1].ordering = CHOLMOD_METIS;
    c->method[2].ordering = CHOLMOD_NESDIS;
  } else {
    c->nmethods = 1;
    c->method[0].ordering = ordering;
  }
  std::vector<double> zeros(e->Ar.size(), 0.0);
  cholmod_sparse A = chol_upper_view(e.get(), zeros.data());
  c->status = CHOLMOD_OK;
  e->Lsym = cholmod_analyze(&A, c);
  if (!e->Lsym || c->status < CHOLMOD_OK) {
    *err = c->status == CHOLMOD_NOT_INSTALLED
               ? std::string(
                     "sparsax: this CHOLMOD was built without its Partition "
                     "module, so the METIS and NESDIS orderings are unavailable")
               : "sparsax: cholmod_analyze failed (status " +
                     std::to_string(c->status) + ")";
    return nullptr;
  }
  e->lnz = c->lnz;
  e->flops = c->fl;

  g_chol_registry[pattern_hash(Ai, Aj, nnz, n)].push_back(e);
  return e;
}

std::shared_ptr<CholPattern> chol_get_pattern(const int32_t* Ai,
                                                     const int32_t* Aj,
                                                     int64_t nnz, int64_t n,
                                                     std::string* err) {
  return get_pattern(g_chol_reg_mtx, g_chol_registry, g_chol_generation, t_chol, Ai,
                     Aj, nnz, n, [&] {
                       return chol_create_pattern_locked(Ai, Aj, nnz, n, err);
                     });
}

// log|A| from the LL' factor diagonal (final_ll guarantees is_ll). Doubles
// as the positive-definiteness check: any nonpositive or NaN diagonal entry
// makes the result non-finite.
double factor_logdet(const cholmod_factor* L) {
  double acc = 0.0;
  int64_t n = static_cast<int64_t>(L->n);
  if (L->is_super) {
    // Supernodal: diagonal of column j in supernode s sits at
    // x[px[s] + c*ld + c] with c = j - super[s], ld = pi[s+1] - pi[s].
    const int32_t* super = static_cast<const int32_t*>(L->super);
    const int32_t* pi = static_cast<const int32_t*>(L->pi);
    const int32_t* px = static_cast<const int32_t*>(L->px);
    const double* xv = static_cast<const double*>(L->x);
    int64_t nsuper = static_cast<int64_t>(L->nsuper);
    for (int64_t s = 0; s < nsuper; ++s) {
      int64_t ld = pi[s + 1] - pi[s];
      int64_t ncols = super[s + 1] - super[s];
      for (int64_t c = 0; c < ncols; ++c)
        acc += 2.0 * std::log(xv[px[s] + c * ld + c]);
    }
  } else {
    // Simplicial: the first stored entry of each column is the diagonal.
    const int32_t* p = static_cast<const int32_t*>(L->p);
    const double* xv = static_cast<const double*>(L->x);
    for (int64_t j = 0; j < n; ++j)
      acc += (L->is_ll ? 2.0 : 1.0) * std::log(xv[p[j]]);
  }
  return acc;
}

// Return a numeric factor for these COO values: a hit on the pattern's cache,
// or a fresh factorization into a copy of the symbolic factor, done outside
// any lock with this thread's workspace and then cached.
CholNumPtr chol_get_factor(const std::shared_ptr<CholPattern>& pat,
                                  const double* Ax, int64_t nnz,
                                  std::string* err) {
  uint64_t h = fnv1a(Ax, nnz * sizeof(double), 14695981039346656037ULL);
  cholmod_factor* recycled = nullptr;
  // A numeric factor refactors in place (cholmod_factorize keeps its
  // supernodal/simplicial form), skipping the copy of the symbolic factor.
  CholNumPtr hit = find_or_claim_factor(*pat, Ax, nnz, h, [&] {
    if (CholSlot* v = evictable_slot_locked(
            pat->cache, pat->next,
            g_chol_cache_cap.load(std::memory_order_relaxed))) {
      recycled = v->num->L;
      v->num->L = nullptr;
      v->num.reset();
      v->hash = 0;
      v->Ax.clear();
    }
  });
  if (hit) return hit;
  InFlightClaim claim(pat->cache_mtx, pat->in_flight, h);

  cholmod_common* c = t_chol.get();
  std::vector<double>& vals = t_chol.values;
  vals.assign(pat->Ar.size(), 0.0);
  for (int64_t k = 0; k < nnz; ++k)
    if (pat->pos[k] >= 0) vals[pat->pos[k]] += Ax[k];

  auto num = std::make_shared<CholNumeric>();
  num->L = recycled ? recycled : cholmod_copy_factor(pat->Lsym, c);
  if (!num->L) {
    *err = "sparsax: cholmod_copy_factor failed (status " +
           std::to_string(c->status) + ")";
    return nullptr;
  }
  cholmod_sparse A = chol_upper_view(pat.get(), vals.data());
  c->status = CHOLMOD_OK;
  cholmod_factorize(&A, num->L, c);
  if (c->status == CHOLMOD_NOT_POSDEF)
    return factorization_failed<CholNumeric>(
        "sparsax: matrix is not positive definite (failure at column " +
            std::to_string(num->L->minor) + ")",
        err);
  if (c->status < CHOLMOD_OK) {
    *err = "sparsax: cholmod_factorize failed (status " +
           std::to_string(c->status) + ")";
    return nullptr;
  }
  num->logdet = factor_logdet(num->L);
  if (!std::isfinite(num->logdet)) {
    // Simplicial LDL' accepts indefinite matrices; the final_ll conversion
    // turns negative pivots into NaNs, which land here.
    return factorization_failed<CholNumeric>(
        "sparsax: matrix is not positive definite", err);
  }
  g_num_factorizations.fetch_add(1, std::memory_order_relaxed);
  cache_factor(*pat, g_chol_cache_cap.load(std::memory_order_relaxed), h, Ax, nnz, num);
  return num;
}

// Solve L L' x = b for one right-hand-side block with this thread's solve2
// workspaces.  bdata/xdata are row-major (JAX layout); CHOLMOD is column-major,
// so multi-RHS blocks are transposed through scratch.
// Columns [j0, j0 + w) of an (n, ld) row-major block, with this thread's
// common and workspaces.  cholmod_solve2 reuses the thread's X/Y/E workspaces,
// avoiding cholmod_solve's per-call allocations, and does not modify L.
static std::string chol_solve_panel(cholmod_factor* L, int mode,
                                    const double* bdata, int64_t n, int64_t ld,
                                    int64_t j0, int64_t w, double* xdata) {
  CholThreadLocal& tl = t_chol;
  cholmod_common* c = tl.get();
  const double* bcol = bdata;
  if (ld > 1) {
    tl.scratch.resize(n * w);
    gather_cols(bdata, n, ld, j0, w, tl.scratch.data());
    bcol = tl.scratch.data();
  }

  cholmod_dense B;
  std::memset(&B, 0, sizeof(B));
  B.nrow = n;
  B.ncol = w;
  B.nzmax = n * w;
  B.d = n;
  B.x = const_cast<double*>(bcol);
  B.xtype = CHOLMOD_REAL;
  B.dtype = CHOLMOD_DOUBLE;

  int ok = cholmod_solve2(mode, L, &B, nullptr, &tl.X, nullptr, &tl.Y, &tl.E, c);
  if (!ok || !tl.X || c->status < CHOLMOD_OK)
    return "sparsax: cholmod_solve failed (status " + std::to_string(c->status) +
           ")";

  const double* Xx = static_cast<const double*>(tl.X->x);
  if (ld == 1)
    std::memcpy(xdata, Xx, n * sizeof(double));
  else
    scatter_cols(Xx, n, ld, j0, w, xdata);
  return "";
}

// Solve with one (n,) or (n, nrhs) row-major right-hand-side block; a block of
// two or more panels is split across threads (see for_each_panel).
ffi::Error chol_solve_one(cholmod_factor* L, int mode,
                                 const double* bdata, int64_t n, int64_t nrhs,
                                 double* xdata) {
  const unsigned T = panel_threads(nrhs);
  std::string err =
      T > 1 ? for_each_panel(nrhs, T,
                             [&] {
                               return [&](int64_t j0, int64_t w) {
                                 return chol_solve_panel(L, mode, bdata, n, nrhs,
                                                         j0, w, xdata);
                               };
                             })
            : chol_solve_panel(L, mode, bdata, n, nrhs, 0, nrhs, xdata);
  if (!err.empty()) return ffi::Error::Internal(err);
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// Simplicial LDL' copy (Lldl). cholmod_updown, cholmod_rowdel and the simplicial
// selected-inverse recurrence need an LDL' (not LL', not supernodal) factor.
// Each numeric factor builds its copy once, on first use, under its own mutex.
// ---------------------------------------------------------------------------

ffi::Error chol_ensure_ldl(CholNumeric* num) {
  std::lock_guard<std::mutex> lk(num->ldl_mtx);
  if (num->Lldl) return ffi::Error::Success();
  cholmod_common* c = t_chol.get();
  cholmod_factor* F = cholmod_copy_factor(num->L, c);
  if (!F) return ffi::Error::Internal("sparsax: cholmod_copy_factor failed");
  if (!cholmod_change_factor(CHOLMOD_REAL, /*to_ll=*/0, /*to_super=*/0,
                             /*to_packed=*/1, /*to_monotonic=*/1, F, c)) {
    cholmod_free_factor(&F, c);
    return ffi::Error::Internal("sparsax: cholmod_change_factor failed");
  }
  num->Lldl = F;
  g_num_ldl_conversions.fetch_add(1, std::memory_order_relaxed);
  return ffi::Error::Success();
}

// ---------------------------------------------------------------------------
// selected-inverse handler:  (Ai, Aj, Ax; n) -> z[nnz]
//     z[k] = (A^{-1})[Ai[k], Aj[k]]
//
// Computes the entries of A^{-1} that lie in the pattern of A (a subset of the
// factor's fill pattern) via Takahashi's recurrence on the factor, without
// forming the dense inverse. This is exactly what logdet's reverse-mode rule
// needs: d log|A| / d A = A^{-1}, so d log|A| / d Ax[k] is (A^{-1}) at the COO
// position (Ai[k], Aj[k]) (see the custom_vjp in Python for the symmetric
// doubling of off-diagonal entries).
//
// Recurrence (A = P' L D L' P, L unit lower-triangular, D diagonal; work in the
// permuted space, then map back through Perm). Processing columns j = n-1..0,
// with below-diagonal pattern rows r_0<...<r_{p-1} of column j and l_b=L[r_b,j]:
//     Z[r_a, j] = - sum_b l_b * Z[r_a, r_b]           (selected inverse Z)
//     Z[j, j]   = 1/D[j] - sum_a l_a * Z[r_a, j]
// The pairs (r_a, r_b) are all in the factor's pattern (fill closure), so the
// recurrence stays within the stored structure.
//
// A supernodal factor runs the same recurrence one dense block at a time, with
// BLAS.  A simplicial factor, which AUTO chooses when L is very sparse, keeps the
// scalar form above.
// ---------------------------------------------------------------------------

#ifdef SPARSAX_HAVE_SUPERNODAL_SELINV
// (L L')^{-1} in place, on the lower triangle of an n x n block.  This is the
// supernode recurrence below once more, on the two halves of the block, and costs
// what dpotri does.  It is not dpotri because Accelerate's threaded dpotri and
// dtrtri return wrong values on about one call in twenty at n near 145, tapering
// off by n = 500, with no sign of failure (macOS 27.2, M3 Max; never with one BLAS
// thread).  dtrsm, dsymm and dgemm passed the same stress test.  `work` is scratch.
constexpr blas_int kBlockInverseBase = 32;

static void chol_block_inverse(double* A, blas_int n, blas_int lda,
                               std::vector<double>& work) {
  if (n <= kBlockInverseBase) {
    work.resize(static_cast<size_t>(n) * n);
    double* Linv = work.data();
    for (blas_int c = 0; c < n; ++c) {
      Linv[c + c * n] = 1.0 / A[c + c * lda];
      for (blas_int r = c + 1; r < n; ++r) {
        double acc = 0.0;
        for (blas_int k = c; k < r; ++k) acc += A[r + k * lda] * Linv[k + c * n];
        Linv[r + c * n] = -acc / A[r + r * lda];
      }
    }
    for (blas_int c = 0; c < n; ++c)
      for (blas_int r = c; r < n; ++r) {
        double acc = 0.0;
        for (blas_int k = r; k < n; ++k) acc += Linv[k + r * n] * Linv[k + c * n];
        A[r + c * lda] = acc;
      }
    return;
  }
  const double one = 1.0, zero = 0.0, minus_one = -1.0;
  const blas_int n1 = n / 2, n2 = n - n1;
  double* A21 = A + n1;
  double* A22 = A + n1 + n1 * lda;
  // A21 <- Y = L21 L11^{-1}, while L11 is still there.
  SUITESPARSE_BLAS(dtrsm, DTRSM)("R", "L", "N", "N", &n2, &n1, &one, A, &lda, A21,
                                 &lda);
  chol_block_inverse(A, n1, lda, work);
  chol_block_inverse(A22, n2, lda, work);
  // Z21 = -Z22 Y and Z11 -= Y' Z21, through scratch: Y is still needed for the second.
  work.resize(static_cast<size_t>(n2) * n1 + static_cast<size_t>(n1) * n1);
  double* Z21 = work.data();
  double* YtZ21 = Z21 + static_cast<size_t>(n2) * n1;
  SUITESPARSE_BLAS(dsymm, DSYMM)("L", "L", &n2, &n1, &minus_one, A22, &lda, A21,
                                 &lda, &zero, Z21, &n2);
  SUITESPARSE_BLAS(dgemm, DGEMM)("T", "N", &n1, &n1, &n2, &one, A21, &lda, Z21, &n2,
                                 &zero, YtZ21, &n1);
  for (blas_int c = 0; c < n1; ++c) {
    for (blas_int r = c; r < n1; ++r) A[r + c * lda] -= YtZ21[r + c * n1];
    std::copy(Z21 + c * n2, Z21 + (c + 1) * n2, A21 + c * lda);
  }
}

// For a supernode with L = [L11; L21] and Y = L21 L11^{-1}, in LL' form,
//     Z21 = -Z22 Y,    Z11 = (L11 L11')^{-1} - Y' Z21,
// where Z22 is the block of Z on the rows below the supernode, assembled from
// its finished ancestors.  CHOLMOD appends each supernode's rows in ascending
// order, so two sorted lists merge into relative indices.  Z is laid out as
// L->x and only its lower part is touched, so it needs no initialization.
// Reads L only, so threads may share the factor.
static void chol_selinv_supernodal(const CholPattern* pat,
                                   const cholmod_factor* L, int64_t nnz,
                                   double* z) {
  const int64_t n = static_cast<int64_t>(L->n);
  const int64_t nsuper = static_cast<int64_t>(L->nsuper);
  const int32_t* Super = static_cast<const int32_t*>(L->super);
  const int32_t* Lpi = static_cast<const int32_t*>(L->pi);
  const int32_t* Lpx = static_cast<const int32_t*>(L->px);
  const int32_t* Ls = static_cast<const int32_t*>(L->s);
  const double* Lx = static_cast<const double*>(L->x);
  const int32_t* Perm = static_cast<const int32_t*>(L->Perm);

  std::unique_ptr<double[]> Z(new double[L->xsize]);
  std::vector<int32_t> col2super(n);
  for (int64_t s = 0; s < nsuper; ++s)
    for (int32_t j = Super[s]; j < Super[s + 1]; ++j)
      col2super[j] = static_cast<int32_t>(s);

  std::vector<double> Y, Z22, YtZ21, inverse_work;
  std::vector<int32_t> pos;
  const double one = 1.0, zero = 0.0, minus_one = -1.0;
  for (int64_t s = nsuper - 1; s >= 0; --s) {
    const blas_int nsrow = Lpi[s + 1] - Lpi[s];
    const blas_int nscol = Super[s + 1] - Super[s];
    const blas_int nr = nsrow - nscol;
    const double* Lsx = Lx + Lpx[s];
    double* Zs = Z.get() + Lpx[s];

    // Z11 <- (L11 L11')^{-1}
    for (blas_int c = 0; c < nscol; ++c)
      std::copy(Lsx + c + c * nsrow, Lsx + nscol + c * nsrow,
                Zs + c + c * nsrow);
    chol_block_inverse(Zs, nscol, nsrow, inverse_work);
    if (nr == 0) continue;

    // Y <- L21 L11^{-1}
    Y.resize(static_cast<size_t>(nr) * nscol);
    for (blas_int c = 0; c < nscol; ++c)
      std::copy(Lsx + nscol + c * nsrow, Lsx + nsrow + c * nsrow, &Y[c * nr]);
    SUITESPARSE_BLAS(dtrsm, DTRSM)("R", "L", "N", "N", &nr, &nscol, &one, Lsx,
                                   &nsrow, Y.data(), &nr);

    // Z22, lower part: walk the rows below s, one ancestor supernode at a time.
    const int32_t* rows = Ls + Lpi[s] + nscol;
    Z22.resize(static_cast<size_t>(nr) * nr);
    pos.resize(nr);
    for (blas_int q = 0; q < nr;) {
      const int32_t t = col2super[rows[q]];
      const int64_t nsrow_t = Lpi[t + 1] - Lpi[t];
      const int32_t* rows_t = Ls + Lpi[t];
      int32_t p = rows[q] - Super[t];
      for (blas_int a = q; a < nr; ++a) {
        while (rows_t[p] < rows[a]) ++p;
        pos[a] = p;
      }
      for (; q < nr && rows[q] < Super[t + 1]; ++q) {
        const double* Zc = Z.get() + Lpx[t] + (rows[q] - Super[t]) * nsrow_t;
        double* out = &Z22[static_cast<size_t>(q) * nr];
        for (blas_int a = q; a < nr; ++a) out[a] = Zc[pos[a]];
      }
    }

    // Z21 <- -Z22 Y, then Z11 <- Z11 - Y' Z21.
    SUITESPARSE_BLAS(dsymm, DSYMM)("L", "L", &nr, &nscol, &minus_one, Z22.data(),
                                   &nr, Y.data(), &nr, &zero, Zs + nscol, &nsrow);
    YtZ21.resize(static_cast<size_t>(nscol) * nscol);
    SUITESPARSE_BLAS(dgemm, DGEMM)("T", "N", &nscol, &nscol, &nr, &one, Y.data(),
                                   &nr, Zs + nscol, &nsrow, &zero, YtZ21.data(),
                                   &nscol);
    for (blas_int c = 0; c < nscol; ++c)
      for (blas_int r = c; r < nscol; ++r)
        Zs[r + c * nsrow] -= YtZ21[r + static_cast<size_t>(c) * nscol];
  }

  // Z[u, v] = (A^{-1})[Perm[u], Perm[v]]; the entry with u >= v is stored.
  std::vector<int32_t> iperm(n);
  for (int64_t k = 0; k < n; ++k) iperm[Perm[k]] = static_cast<int32_t>(k);
  const int32_t* Ai = pat->Ai.data();
  const int32_t* Aj = pat->Aj.data();
  for (int64_t k = 0; k < nnz; ++k) {
    int32_t u = iperm[Ai[k]], v = iperm[Aj[k]];
    if (u < v) std::swap(u, v);
    const int32_t s = col2super[v];
    const int64_t nsrow = Lpi[s + 1] - Lpi[s];
    const int32_t* rows = Ls + Lpi[s];
    const int64_t row = std::lower_bound(rows, rows + nsrow, u) - rows;
    z[k] = Z[Lpx[s] + (v - Super[s]) * nsrow + row];
  }
}
#endif  // SPARSAX_HAVE_SUPERNODAL_SELINV

ffi::Error chol_selinv(const CholPattern* pat, CholNumeric* num,
                              int64_t nnz, double* z) {
#ifdef SPARSAX_HAVE_SUPERNODAL_SELINV
  if (num->L->is_super) {
    chol_selinv_supernodal(pat, num->L, nnz, z);
    return ffi::Error::Success();
  }
#endif
  ffi::Error r = chol_ensure_ldl(num);
  if (r.failure()) return r;
  const cholmod_factor* L = num->Lldl;
  int64_t n = static_cast<int64_t>(L->n);
  const int32_t* Lp = static_cast<const int32_t*>(L->p);
  const int32_t* Li = static_cast<const int32_t*>(L->i);
  const double* Lx = static_cast<const double*>(L->x);
  const int32_t* Perm = static_cast<const int32_t*>(L->Perm);

  std::vector<double> Z(Lp[n], 0.0);  // selected inverse, same structure as L
  std::vector<double> work(n, 0.0);   // dense scatter workspace (kept zeroed)
  std::vector<double> acc;

  for (int64_t j = n - 1; j >= 0; --j) {
    int64_t ps = Lp[j], pe = Lp[j + 1];
    int64_t cnt = pe - ps - 1;  // below-diagonal entries of column j
    // acc[a] = sum_b L[r_b, j] * Z[r_a, r_b], accumulated symmetrically so that
    // each column of Z is scattered into `work` exactly once.
    acc.assign(cnt, 0.0);
    for (int64_t b = 0; b < cnt; ++b) {
      int64_t c = Li[ps + 1 + b];  // r_b: column of Z to scatter
      double lb = Lx[ps + 1 + b];
      int64_t cs = Lp[c], ce = Lp[c + 1];
      work[c] = Z[cs];  // Z[c, c]
      for (int64_t t = cs + 1; t < ce; ++t) work[Li[t]] = Z[t];
      for (int64_t a = b; a < cnt; ++a) {
        double za = work[Li[ps + 1 + a]];  // Z[r_a, r_b], r_a >= r_b (stored)
        acc[a] += lb * za;
        if (a != b) acc[b] += Lx[ps + 1 + a] * za;  // symmetric partner
      }
      work[c] = 0.0;  // restore workspace to all-zero
      for (int64_t t = cs + 1; t < ce; ++t) work[Li[t]] = 0.0;
    }
    double diag = 1.0 / Lx[ps];  // 1/D[j]
    for (int64_t a = 0; a < cnt; ++a) {
      double zaj = -acc[a];
      Z[ps + 1 + a] = zaj;
      diag -= Lx[ps + 1 + a] * zaj;
    }
    Z[ps] = diag;
  }

  // Z is A^{-1} in the permuted space: Z[u,v] = (A^{-1})[Perm[u], Perm[v]].
  // Map each COO position back through the inverse permutation.
  std::vector<int32_t> iperm(n);
  for (int64_t k = 0; k < n; ++k) iperm[Perm[k]] = static_cast<int32_t>(k);
  const int32_t* Ai = pat->Ai.data();
  const int32_t* Aj = pat->Aj.data();
  for (int64_t k = 0; k < nnz; ++k) {
    int64_t u = iperm[Ai[k]], v = iperm[Aj[k]];
    if (u < v) std::swap(u, v);  // stored entry lives in column min, row max
    double val = 0.0;
    if (u == v) {
      val = Z[Lp[v]];
    } else {
      for (int64_t t = Lp[v] + 1, te = Lp[v + 1]; t < te; ++t)
        if (Li[t] == u) {
          val = Z[t];
          break;
        }
    }
    z[k] = val;
  }
  return ffi::Error::Success();
}

void chol_set_cache_size(size_t cap) {
  g_chol_cache_cap = cap;
  trim_numeric_caches(g_chol_reg_mtx, g_chol_registry, cap);
}

// Drop the patterns and invalidate outstanding factor tokens.  The factors are
// refcounted, so any in-flight call still holding one keeps it alive; threads notice
// the generation bump and forget their last pattern.
void chol_clear_cache() {
  std::lock_guard<std::mutex> lk(g_chol_reg_mtx);
  clear_patterns_locked(g_chol_registry);
  g_chol_generation.fetch_add(1, std::memory_order_relaxed);
  g_chol_tokens.clear();
}

size_t chol_pattern_count() {
  std::lock_guard<std::mutex> lk(g_chol_reg_mtx);
  size_t count = 0;
  for (auto& [h, chain] : g_chol_registry) count += chain.size();
  return count;
}

int64_t chol_factorizations() {
  return g_num_factorizations.load(std::memory_order_relaxed);
}

bool chol_has_supernodal_selinv() {
#ifdef SPARSAX_HAVE_SUPERNODAL_SELINV
  return true;
#else
  return false;
#endif
}
