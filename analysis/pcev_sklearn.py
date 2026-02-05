# pcev_sklearn.py
# sklearn-compatible PCEV (no blocks) with a fast PyTorch permutation test.
# - Fit/transform API for seamless Pipeline/CV use.
# - Exact analytic p-value for q=1.
# - Permutation p-value computed in large Torch batches on CPU.
#
# Dependencies:
#   numpy, scipy, scikit-learn, torch (CPU)
#
# MIT License

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from numpy.linalg import LinAlgError
from scipy import linalg, stats
from sklearn.base import BaseEstimator, TransformerMixin
from typing import Optional, Tuple, Dict, Any

try:
    import torch
    _TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    _TORCH_AVAILABLE = False


# ---------------------------------------------------------------------
#                    Internal linear algebra utilities
# ---------------------------------------------------------------------

def _proj_out_with_Q(A: np.ndarray, Q: Optional[np.ndarray]) -> np.ndarray:
    """Residualize columns of A on the subspace spanned by Q: (I - Q Q^T) A.

    Parameters
    ----------
    A : (n, k) array
    Q : (n, r_eff) array or None
        Orthonormal basis of the confounder space (economy-QR). If None, returns A.

    Returns
    -------
    (n, k) array of residuals.
    """
    if Q is None:
        return A
    return A - Q @ (Q.T @ A)


def _fit_Q(C: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Return an orthonormal basis Q of col(C) via economy QR on centered C.

    Parameters
    ----------
    C : (n, r) or None

    Returns
    -------
    Q : (n, r_eff) or None
    """
    if C is None or C.size == 0:
        return None
    C = np.asarray(C, float)
    Cc = C - C.mean(axis=0, keepdims=True)
    Q, _ = linalg.qr(Cc, mode="economic")
    return Q


def _cov_colwise(X: np.ndarray) -> np.ndarray:
    """Unbiased sample covariance of columns of X: X^T X / (n-1)."""
    n = X.shape[0]
    return (X.T @ X) / (n - 1)


def _pinv_psd(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """Moore–Penrose pseudoinverse for symmetric PSD matrix via eigen-decomposition."""
    w, V = linalg.eigh(A, overwrite_a=False, check_finite=True)
    if w.size == 0:
        return A
    cutoff = rcond * float(np.max(w))
    inv = np.zeros_like(w)
    np.divide(1.0, w, out=inv, where=w > cutoff)
    return (V * inv) @ V.T


def _chol_whiten_gen_eig(SB: np.ndarray, SW: np.ndarray, use_torch: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Generalized eigen solve SB v = λ SW v using direct scipy.linalg.eigh or torch.

    Parameters
    ----------
    SB : (p, p) array
        Between covariance matrix
    SW : (p, p) array  
        Within covariance matrix
    use_torch : bool
        If True and torch is available, use torch for potentially faster computation

    Returns
    -------
    V : (p, p) SW-orthonormal eigenvectors, columns
    lam : (p,) ascending eigenvalues
    
    Notes
    -----
    Fixed to use direct generalized eigenvalue solver instead of Cholesky whitening
    which was causing incorrect eigenvalue computation (334x error).
    """
    if use_torch and _TORCH_AVAILABLE:
        return _chol_whiten_gen_eig_torch(SB, SW)
    
    # Use scipy's direct generalized eigenvalue solver
    lam, V = linalg.eigh(SB, SW, check_finite=True)  # ascending
    
    # Normalize eigenvectors to be SW-orthonormal: V^T SW V = I
    norms = np.sqrt(np.einsum("ij,ij->j", V, SW @ V))
    norms[norms == 0] = 1.0
    V = V / norms
    return V, lam


def _chol_whiten_gen_eig_torch(SB: np.ndarray, SW: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Torch-accelerated generalized eigenvalue solve SB v = λ SW v.
    
    This can be faster for larger matrices due to optimized BLAS operations.
    """
    import torch
    
    device = torch.device('cpu')  # GPU gives little benefit for typical p~1000
    dtype = torch.float64  # Match numpy precision
    
    # Convert to torch tensors
    SB_t = torch.as_tensor(SB, dtype=dtype, device=device)
    SW_t = torch.as_tensor(SW, dtype=dtype, device=device)
    
    # Solve generalized eigenvalue problem using torch
    # torch.linalg.eigh doesn't support generalized form directly,
    # so we use the scipy approach but with torch tensors where beneficial
    
    # For now, fall back to numpy since torch doesn't have generalized eigh
    # But we can still benefit from torch's optimized matrix operations in batched scenarios
    lam, V = linalg.eigh(SB, SW, check_finite=True)
    
    # Use torch for the normalization (can be faster for large matrices)
    V_t = torch.as_tensor(V, dtype=dtype, device=device)
    SW_t = torch.as_tensor(SW, dtype=dtype, device=device)
    
    norms_t = torch.sqrt(torch.sum(V_t * (SW_t @ V_t), dim=0))
    norms_t[norms_t == 0] = 1.0
    V_normalized = (V_t / norms_t).cpu().numpy()
    
    return V_normalized, lam


# ---------------------------------------------------------------------
#                      Residualizer helper (train/test)
# ---------------------------------------------------------------------


@dataclass
class _Residualizer:
    y_mean: np.ndarray
    x_mean: np.ndarray
    c_mean: Optional[np.ndarray]
    beta_y: np.ndarray
    beta_x: np.ndarray


def _fit_residualizer(
    Y: np.ndarray,
    X: np.ndarray,
    C: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, _Residualizer]:
    """Residualize Y and X on confounders C using training data only."""

    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]

    y_mean = Y.mean(axis=0, keepdims=True)
    x_mean = X.mean(axis=0, keepdims=True)

    Y_centered = Y - y_mean
    X_centered = X - x_mean

    if C is None or getattr(C, "size", 0) == 0:
        beta_y = np.zeros((0, Y.shape[1]))
        beta_x = np.zeros((0, X.shape[1]))
        resid = _Residualizer(y_mean=y_mean, x_mean=x_mean, c_mean=None, beta_y=beta_y, beta_x=beta_x)
        return Y_centered, X_centered, resid

    C = np.asarray(C, float)
    c_mean = C.mean(axis=0, keepdims=True)
    C_centered = C - c_mean

    beta_y, *_ = np.linalg.lstsq(C_centered, Y_centered, rcond=None)
    beta_x, *_ = np.linalg.lstsq(C_centered, X_centered, rcond=None)

    Yr = Y_centered - C_centered @ beta_y
    Xr = X_centered - C_centered @ beta_x

    resid = _Residualizer(y_mean=y_mean, x_mean=x_mean, c_mean=c_mean, beta_y=beta_y, beta_x=beta_x)
    return Yr, Xr, resid


def _apply_residualizer(
    residualizer: _Residualizer,
    *,
    Y: Optional[np.ndarray] = None,
    X: Optional[np.ndarray] = None,
    C: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Apply a previously fitted residualizer to new data."""

    outputs: Dict[str, np.ndarray] = {}

    if Y is not None:
        Y_arr = np.asarray(Y, float)
        Y_centered = Y_arr - residualizer.y_mean
        if residualizer.beta_y.size:
            if C is None:
                raise ValueError("Confounders C must be provided to residualize Y.")
            C_arr = np.asarray(C, float)
            C_centered = C_arr - residualizer.c_mean
            Yr = Y_centered - C_centered @ residualizer.beta_y
        else:
            Yr = Y_centered
        outputs["Y"] = Yr

    if X is not None:
        X_arr = np.asarray(X, float)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        X_centered = X_arr - residualizer.x_mean
        if residualizer.beta_x.size:
            if C is None:
                raise ValueError("Confounders C must be provided to residualize X.")
            C_arr = np.asarray(C, float)
            C_centered = C_arr - residualizer.c_mean
            Xr = X_centered - C_centered @ residualizer.beta_x
        else:
            Xr = X_centered
        outputs["X"] = Xr

    return outputs


# ---------------------------------------------------------------------
#                          sklearn Estimator
# ---------------------------------------------------------------------

class PCEV(BaseEstimator, TransformerMixin):
    """Principal Component of Explained Variance (no blocks), sklearn-compatible.

    The estimator takes multivariate responses **Y** as `X` in fit/transform and the
    covariate(s) of interest **x** as `y` in `fit`. Confounders **C** are passed via
    `fit(..., C=C)` and are projected out internally using an economy-QR projector.

    After `fit`, call `transform(Y)` to obtain PCEV scores for the first
    `n_components` directions. Use `exact_pvalue_(y)` to get the exact analytic
    p-value for q=1. Use `permutation_test(...)` to get a fast PyTorch-based
    permutation p-value on the data used in `fit`.

    Parameters
    ----------
    n_components : int, default=1
        Number of PCEV components to keep (ordered by decreasing generalized root).
    ridge : float, default=1e-6
        Ridge added to the within-covariance SW for numerical stability.
    rcond : float, default=1e-12
        Relative threshold when building PSD pseudoinverses.

    Attributes
    ----------
    components_ : (n_components, p) array
        SW-normalized generalized eigenvectors, rows are components.
    lambdas_ : (k,) array
        Descending positive generalized eigenvalues (k ≤ min(p, q)).
    lambda_max_ : float
        Largest generalized eigenvalue.
    explained_variance_ratio_ : (k,) array
        For each positive root: h² = λ / (1 + λ).
    ridge_ : float
        Ridge used.
    Q_ : (n, r_eff) or None
        Orthonormal confounder basis from training C.
    n_features_in_ : int
        Number of columns in Y at fit time.

    Stored for fast permutation (computed at fit)
    ---------------------------------------------
    Yr_ : (n, p)
        Centered and residualized training Y on C.
    Xr_ : (n, q)
        Centered and residualized training x on C.
    Yt_ : (p, n)
        Yr_.T
    SW_inv_ : (p, p)
        Inverse of SW via Cholesky-solve; used in permutation.
    Sxx_ : (q, q)
        Training Sxx = cov(Xr_). Constant across permutations.
    Sxx_inv_sqrt_ : (q, q)
        Symmetric inverse square root of Sxx_ (for q ≥ 1).
    sxx_scalar_ : float
        Scalar Sxx when q=1 (optimization).
    """

    def __init__(self, n_components: int = 1, ridge: float = 1e-6, rcond: float = 1e-12, use_torch_backend: bool = False):
        self.n_components = int(n_components)
        self.ridge = float(ridge)
        self.rcond = float(rcond)
        self.use_torch_backend = bool(use_torch_backend)

    # ----------------------------- fit -----------------------------

    def fit(self, X: np.ndarray, y: np.ndarray | None = None, *, C: Optional[np.ndarray] = None):
        """Fit the PCEV on Y=X with respect to covariate(s) y, optionally controlling for C.

        Parameters
        ----------
        X : (n, p) array
            Multivariate responses Y.
        y : (n,) or (n, q) array
            Covariate(s) of interest.
        C : (n, r) array or None
            Confounders. Only the training C must be passed here (no leakage).

        Returns
        -------
        self
        """
        Y = np.asarray(X, float)
        z = np.asarray(y, float) if y is not None else None
        if z is None:
            raise ValueError("y must be provided as the covariate(s) of interest.")
        if z.ndim == 1:
            z = z[:, None]

        n, p = Y.shape
        q = z.shape[1]

        Yr, Xr, residualizer = _fit_residualizer(Y, z, C)
        self.residualizer_ = residualizer
        if C is None or getattr(C, "size", 0) == 0:
            rank_C = 0
        else:
            rank_C = int(np.linalg.matrix_rank(np.asarray(C, float)))
        self.rank_C_ = rank_C
        # Backwards compatibility attribute (no longer used for projections)
        self.Q_ = None

        # Covariances
        Syy = _cov_colwise(Yr)                                 # p×p
        Sxx = _cov_colwise(Xr)                                 # q×q
        Sxx_inv = _pinv_psd(Sxx, rcond=self.rcond)
        Sxy = (Yr.T @ Xr) / (n - 1)                            # p×q

        # Explained and within covariance
        SB = Sxy @ (Sxx_inv @ Sxy.T)                           # p×p
        SW = Syy - SB
        if self.ridge > 0:
            SW = SW + self.ridge * np.eye(p)

        # Generalized eigen
        try:
            V, lam_all = _chol_whiten_gen_eig(SB, SW, use_torch=self.use_torch_backend)
        except LinAlgError:
            SW_pinv = _pinv_psd(SW, rcond=self.rcond)
            M = SW_pinv @ SB
            lam_all, V = linalg.eigh(M, check_finite=True)
            norms = np.sqrt(np.einsum("ij,ij->j", V, SW @ V))
            norms[norms == 0] = 1.0
            V = V / norms

        # Keep positive roots, sort descending
        pos = lam_all > 1e-15
        lam = lam_all[pos]
        if lam.size == 0:
            self.components_ = np.zeros((min(self.n_components, p), p))
            self.lambdas_ = np.array([], dtype=float)
            self.lambda_max_ = 0.0
            self.explained_variance_ratio_ = np.array([], dtype=float)
        else:
            idx = np.argsort(lam)[::-1]
            lam = lam[idx]
            V = V[:, pos][:, idx]
            k = min(self.n_components, V.shape[1])
            components = V[:, :k].T
            # Orient components for reproducible sign (align with Cov(t, X))
            for j in range(components.shape[0]):
                cov_vec = components[j] @ Sxy  # shape (q,)
                if cov_vec.ndim == 0:
                    cov_val = float(cov_vec)
                else:
                    if cov_vec.size == 0:
                        cov_val = 0.0
                    else:
                        idx_max = int(np.argmax(np.abs(cov_vec)))
                        cov_val = float(cov_vec[idx_max])
                if cov_val < 0:
                    components[j] *= -1.0
            self.components_ = components
            self.lambdas_ = lam
            self.lambda_max_ = float(lam[0])
            self.explained_variance_ratio_ = lam / (1.0 + lam)

        # Cache for fast permutation on the training dataset
        # SW_inv via Cholesky-solve for numerical stability
        try:
            L = linalg.cholesky(SW, lower=True, check_finite=True)
            I = np.eye(p)
            SW_inv = linalg.cho_solve((L, True), I, check_finite=True)
        except LinAlgError:
            SW_inv = _pinv_psd(SW, rcond=self.rcond)

        # Sxx inverse sqrt for general q; scalar for q=1
        if q == 1:
            self.sxx_scalar_ = float(Sxx.ravel()[0])
            self.Sxx_inv_sqrt_ = None
        else:
            # symmetric inverse square root of Sxx
            w, U = linalg.eigh(Sxx, check_finite=True)
            w[w < 0] = 0.0
            w_inv_sqrt = np.zeros_like(w)
            threshold = 1e-15 * np.max(w)
            sqrt_w = np.zeros_like(w)
            np.sqrt(w, out=sqrt_w, where=w > 0)
            np.divide(1.0, sqrt_w, out=w_inv_sqrt, where=w > threshold)
            self.Sxx_inv_sqrt_ = (U * w_inv_sqrt) @ U.T
            self.sxx_scalar_ = None

        self.Yr_ = Yr
        self.Xr_ = Xr
        self.Yt_ = Yr.T
        self.SW_inv_ = SW_inv
        self.Sxx_ = Sxx

        self.ridge_ = float(self.ridge)
        self.n_features_in_ = p
        self.n_samples_in_ = n
        return self

    # -------------------------- transform --------------------------

    def transform(
        self,
        X: np.ndarray,
        *,
        C: Optional[np.ndarray] = None,
        residualize: bool = True,
    ) -> np.ndarray:
        """Project Y onto learned components.

        Parameters
        ----------
        X : (n, p) array
            Multivariate responses (train or test).
        C : (n, r) array or None, optional
            Confounders for the provided data. Required when the model was
            trained with confounders and `residualize=True`.
        residualize : bool, default=True
            If True, use the training residualizer to project out confounders
            and center using training means. If False, only center using the
            provided sample mean (legacy behaviour).

        Returns
        -------
        T : (n, n_components) array
            PCEV scores.
        """
        return self.project_scores(X, C=C, residualize=residualize)

    def project_scores(
        self,
        Y: np.ndarray,
        *,
        C: Optional[np.ndarray] = None,
        residualize: bool = True,
    ) -> np.ndarray:
        """Return PCEV component scores for provided data."""
        if not hasattr(self, "components_"):
            raise RuntimeError("Call fit before project_scores.")

        if residualize:
            if not hasattr(self, "residualizer_"):
                raise RuntimeError("Residualizer not available; re-fit the estimator.")
            needs_C = self.residualizer_.beta_y.size > 0
            res = _apply_residualizer(self.residualizer_, Y=Y, C=C if needs_C else None)
            Yr = res["Y"]
        else:
            Y_arr = np.asarray(Y, float)
            Yr = Y_arr - Y_arr.mean(axis=0, keepdims=True)

        return Yr @ self.components_.T

    def residualize(
        self,
        *,
        Y: Optional[np.ndarray] = None,
        x: Optional[np.ndarray] = None,
        C: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """Residualize Y and/or x using the training residualizer."""
        if not hasattr(self, "residualizer_"):
            raise RuntimeError("Residualizer not available; re-fit the estimator.")
        need_C = self.residualizer_.beta_y.size > 0 or self.residualizer_.beta_x.size > 0
        kwargs: Dict[str, Any] = {}
        if Y is not None:
            kwargs["Y"] = Y
        if x is not None:
            kwargs["X"] = x
        if not kwargs:
            raise ValueError("Provide at least one of Y or x to residualize().")
        return _apply_residualizer(self.residualizer_, C=C if need_C else None, **kwargs)

    # ------------------------- exact p-value -----------------------

    def exact_pvalue_(self, y: np.ndarray) -> Tuple[float, Dict[str, int | float]]:
        """Exact analytic p-value for q=1.

        Parameters
        ----------
        y : (n,) or (n,1) array
            Covariate used in `fit`.

        Returns
        -------
        pval : float
        info : dict with keys {'F','df1','df2'}

        Notes
        -----
        Uses h² = λ/(1+λ) from training and df2 = n_eff - p - 1 with n_eff = n - rank(C).
        """
        z = np.asarray(y, float)
        if z.ndim == 2 and z.shape[1] != 1:
            raise ValueError("Exact p-value is implemented here for q=1 only.")
        if self.explained_variance_ratio_.size == 0:
            return 1.0, {"F": 0.0, "df1": self.n_features_in_, "df2": 1}

        # Effective n from training C
        rC = getattr(self, "rank_C_", 0)
        n = z.shape[0]
        p = self.n_features_in_
        df1 = p
        df2 = n - rC - p - 1
        if df2 <= 0:
            raise ValueError("Nonpositive df2; increase n or reduce p or confounders.")
        h2 = float(self.explained_variance_ratio_[0])
        F = ((n - rC - p - 1) / p) * (h2 / max(1.0 - h2, 1e-15))
        pval = stats.f.sf(F, df1, df2)
        return float(pval), {"F": float(F), "df1": int(df1), "df2": int(df2)}

    # ----------------------- permutation p-value -------------------

    def permutation_test(
        self,
        *,
        n_perm: int = 1000,
        strat: Optional[np.ndarray] = None,
        batch_size: int = 2048,
        seed: int | None = 0,
        dtype: str = "float32",
        device: str = "cpu",
    ) -> Dict[str, np.ndarray | float]:
        """Fast PyTorch permutation p-value on the **training dataset**.

        This method reuses cached training matrices (Yr_, Xr_, SW_inv_, Sxx_) and
        permutes the **rows of Xr_** within strata. It computes the null of the
        largest generalized root λ (equivalently h²) in large Torch batches.

        Parameters
        ----------
        n_perm : int, default=1000
            Number of permutations.
        strat : (n,) array of ints or None
            Stratification labels. If None, one stratum is used.
        batch_size : int, default=2048
            Number of permutations per Torch batch. Increase to utilize BLAS better.
        seed : int or None, default=0
            Torch RNG seed for reproducibility.
        dtype : {'float32','float64'}, default='float32'
            Torch dtype used in permutation calculations.
        device : str, default='cpu'
            Torch device. CPU is recommended; GPU yields limited benefit at p≈10^3.

        Returns
        -------
        dict
            {
              'p_perm': float,
              'h2_obs': float,
              'null_h2': (n_perm,) array
            }

        Notes
        -----
        - Requires PyTorch. Raises if Torch is unavailable.
        - Operates on the data used in `fit`. Do not pass test data here.
        - For q=1 it reduces to a quadratic form, avoiding any eigen-decomposition.
        - For q≥1 it eigen-decomposes q×q matrices in batch.
        """
        if not _TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for permutation_test.")

        if not hasattr(self, "Yr_"):
            raise RuntimeError("Call fit before permutation_test.")

        Yr = self.Yr_              # n×p
        Xr = self.Xr_              # n×q
        SWi = self.SW_inv_         # p×p
        Sxx = self.Sxx_            # q×q
        n, p = Yr.shape
        q = Xr.shape[1]

        # Stratification
        if strat is None:
            strat = np.zeros(n, dtype=int)
        else:
            strat = np.asarray(strat)
            if strat.shape[0] != n:
                raise ValueError("strat length must equal n used in fit.")
        groups = [np.flatnonzero(strat == s) for s in np.unique(strat)]

        # Torch setup
        torch.set_grad_enabled(False)
        _dtype = torch.float32 if dtype == "float32" else torch.float64
        dev = torch.device(device)

        Yt = torch.as_tensor(Yr.T.copy(), dtype=_dtype, device=dev)      # p×n
        Xr_base = torch.as_tensor(Xr.copy(), dtype=_dtype, device=dev)   # n×q
        SWi_t = torch.as_tensor(SWi.copy(), dtype=_dtype, device=dev)    # p×p

        if q == 1:
            sxx_scalar = float(Sxx.ravel()[0])
        else:
            # Sxx^{-1/2} for whitening q×q
            w, U = np.linalg.eigh(Sxx)
            w[w < 0] = 0.0
            w_inv_sqrt = np.zeros_like(w)
            threshold = 1e-15 * np.max(w)
            sqrt_w = np.zeros_like(w)
            np.sqrt(w, out=sqrt_w, where=w > 0)
            np.divide(1.0, sqrt_w, out=w_inv_sqrt, where=w > threshold)
            Sxx_inv_sqrt = (U * w_inv_sqrt) @ U.T
            Sxx_inv_sqrt_t = torch.as_tensor(Sxx_inv_sqrt, dtype=_dtype, device=dev)

        # Observed h2 from training fit
        h2_obs = float(self.explained_variance_ratio_[0]) if self.explained_variance_ratio_.size else 0.0

        # Build Torch index lists for each stratum
        idx_groups = [torch.as_tensor(g, dtype=torch.long, device=dev) for g in groups]
        g = torch.Generator(device=dev)
        if seed is not None:
            g.manual_seed(int(seed))

        # Main loop in batches
        null_vals = []
        remain = int(n_perm)

        while remain > 0:
            b = min(batch_size, remain)

            # Allocate permuted Xr batch: shape (n, b, q)
            Xperm = torch.empty((n, b, q), dtype=_dtype, device=dev)

            # Fill by strata using manual indexing for each batch
            for ig in idx_groups:
                m = ig.numel()
                # Get local data for this group
                Yg = Xr_base.index_select(0, ig)                           # (m, q)
                
                # For each batch, create permutation and fill
                for batch_idx in range(b):
                    perm = torch.randperm(m, generator=g, device=dev)      # (m,)
                    Yg_perm = Yg.index_select(0, perm)                     # (m, q)
                    # Write permuted data to the batch tensor
                    Xperm[ig, batch_idx, :] = Yg_perm

            # Compute Sxy for the batch: each column is one permutation
            # Flatten to (n, b*q) then matmul with Yt (p×n)
            Xflat = Xperm.reshape(n, b * q)
            Sxy_flat = (Yt @ Xflat) / float(n - 1)                         # p×(b*q)
            Sxy_b = Sxy_flat.reshape(p, b, q).permute(1, 0, 2)             # b×p×q

            if q == 1:
                # Quadratic form: λ = (sxy^T SW^{-1} sxy) / sxx
                # Sxy_b: b×p×1 -> squeeze to b×p
                sxy = Sxy_b.squeeze(-1)                                    # b×p
                tmp = torch.matmul(sxy, SWi_t.T)                           # b×p
                num = (sxy * tmp).sum(dim=1)                               # b
                lam = num / float(sxx_scalar)                               # b
                h2 = lam / (1.0 + lam)                                     # b
            else:
                # Batched M_b = Sxy_b^T SW^{-1} Sxy_b (q×q per perm)
                tmp = torch.einsum('ij,bjp->bip', SWi_t, Sxy_b)            # b×p×q
                M = torch.einsum('bpi,bpj->bij', Sxy_b, tmp)               # b×q×q
                # Whiten: B_b = Sxx^{-1/2} M_b Sxx^{-1/2}
                B = torch.einsum('ij,bjk,kl->bil', Sxx_inv_sqrt_t, M, Sxx_inv_sqrt_t)  # b×q×q
                w = torch.linalg.eigvalsh(B)                                # b×q
                lam = w[..., -1]                                            # b
                h2 = lam / (1.0 + lam)

            null_vals.append(h2.detach().cpu().numpy())
            remain -= b

        null_h2 = np.concatenate(null_vals) if len(null_vals) else np.zeros(0, dtype=float)

        # Mid-p with +1 correction
        p_perm = (np.sum(null_h2 > h2_obs) + 0.5 * np.sum(null_h2 == h2_obs) + 1.0) / (n_perm + 1.0)

        return {"p_perm": float(p_perm), "h2_obs": float(h2_obs), "null_h2": null_h2}

    # ----------------------- variable importance --------------------

    def vip(
        self,
        X,
        *,
        C: Optional[np.ndarray] = None,
        signed: bool = True,
        use_residualized: bool = True,
    ):
        """
        Variable Importance on Projection (VIP) for each outcome variable.

        VIP_j = Cor(Y_j, t), where t = Y @ w_PCEV on the provided data.
        By default both Y and t are residualized using the training projector Q_.

        Parameters
        ----------
        X : array (n, p)
            Outcomes to evaluate VIP on (train or test).
        signed : bool, default=True
            If True return signed correlations. If False return absolute values.
        use_residualized : bool, default=True
            If True residualize Y using training Q_ before computing VIP to
            reflect association conditional on confounders (recommended).

        Returns
        -------
        vip : (p,) array
            Correlations per outcome with the first PCEV score.
        """
        if not hasattr(self, "components_") or self.components_.shape[0] == 0:
            raise RuntimeError("Call fit before vip.")
        scores = self.project_scores(X, C=C, residualize=use_residualized)
        if scores.shape[1] == 0:
            return np.zeros(np.asarray(X, float).shape[1], dtype=float)
        t = scores[:, 0]
        if use_residualized:
            res = self.residualize(Y=X, C=C)
            Ycorr = res["Y"]
        else:
            Y_arr = np.asarray(X, float)
            Ycorr = Y_arr - Y_arr.mean(axis=0, keepdims=True)
        t = t - t.mean()
        # Std devs
        yt_sd = np.sqrt(np.sum(t * t))
        y_sd = np.sqrt(np.sum(Ycorr * Ycorr, axis=0))
        # Avoid divide-by-zero
        y_sd[y_sd == 0] = np.inf
        if yt_sd == 0:
            return np.zeros(Y.shape[1], dtype=float)
        # Correlations
        corr = (Ycorr.T @ t) / (y_sd * yt_sd)
        if not signed:
            corr = np.abs(corr)
        return np.asarray(corr).ravel()

    def vip_table(self, X, feature_names=None, **kwargs):
        """
        Convenience: return a 2-col table [feature, VIP] sorted by |VIP| desc.

        Parameters
        ----------
        X : (n, p) array
        feature_names : list[str] or None

        Returns
        -------
        list[tuple]
            [(name, vip_value), ...] sorted by absolute VIP.
        """
        v = self.vip(X, **kwargs)
        names = feature_names if feature_names is not None else [f"Y{j}" for j in range(len(v))]
        order = np.argsort(np.abs(v))[::-1]
        return [(names[i], float(v[i])) for i in order]
