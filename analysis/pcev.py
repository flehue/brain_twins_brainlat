# pcev.py
# Principal Component of Explained Variance (PCEV) — Python, no blocks.
# CPU-only, vectorized. Stable generalized eigen solve.
#
# Dependencies: numpy, scipy, joblib (optional for permutations)
#
# MIT License

from __future__ import annotations
import numpy as np
from numpy.linalg import LinAlgError
from scipy import linalg, stats
from joblib import Parallel, delayed
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any

# ---------------------------------------------------------------------
#                      Linear Algebra Utilities
# ---------------------------------------------------------------------

def _proj_out(A: np.ndarray, B: Optional[np.ndarray]) -> np.ndarray:
    """
    Residualize A on B: return (I - P_B) A using economy QR.

    Parameters
    ----------
    A : array, shape (n, k)
        Matrix to be residualized.
    B : array or None, shape (n, r)
        Nuisance design. If None or empty, A is returned unchanged.

    Returns
    -------
    A_res : array, shape (n, k)
        Residuals of A after projecting out col(B).
    """
    if B is None or B.size == 0:
        return A
    Q, _ = linalg.qr(B, mode="economic")
    # A - QQ^T A without forming P explicitly
    return A - Q @ (Q.T @ A)


def _cov_colwise(X: np.ndarray) -> np.ndarray:
    """
    Unbiased sample covariance matrix of columns of X.

    Parameters
    ----------
    X : array, shape (n, p)

    Returns
    -------
    S : array, shape (p, p)
        Covariance with denominator (n-1).
    """
    n = X.shape[0]
    return (X.T @ X) / (n - 1)


def _pinv_psd(A: np.ndarray, rcond: float = 1e-12) -> np.ndarray:
    """
    Moore–Penrose pseudoinverse for a symmetric PSD matrix via eigen-decomposition.

    Parameters
    ----------
    A : array, shape (p, p)
        Symmetric PSD matrix.
    rcond : float
        Relative threshold on eigenvalues to invert.

    Returns
    -------
    A_pinv : array, shape (p, p)
    """
    w, V = linalg.eigh(A, overwrite_a=False, check_finite=True)
    if w.size == 0:
        return A
    cutoff = rcond * float(np.max(w))
    inv = np.where(w > cutoff, 1.0 / w, 0.0)
    return (V * inv) @ V.T


def _chol_whiten_gen_eig(SB: np.ndarray, SW: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solve the generalized eigenproblem SB v = λ SW v using direct scipy.linalg.eigh.

    Parameters
    ----------
    SB : array, shape (p, p)
        Symmetric explained covariance.
    SW : array, shape (p, p)
        Symmetric within/residual covariance, positive definite or ridge-stabilized.

    Returns
    -------
    V : array, shape (p, p)
        SW-orthonormal eigenvectors (columns). V^T SW V = I.
    lam : array, shape (p,)
        Eigenvalues sorted ascending (scipy.linalg.eigh convention).

    Notes
    -----
    Fixed to use direct generalized eigenvalue solver instead of Cholesky whitening
    which was causing incorrect eigenvalue computation (334x error).
    """
    # Use scipy's direct generalized eigenvalue solver
    lam, V = linalg.eigh(SB, SW, check_finite=True)  # ascending
    
    # Normalize eigenvectors to be SW-orthonormal: V^T SW V = I
    norms = np.sqrt(np.einsum("ij,ij->j", V, SW @ V))
    norms[norms == 0] = 1.0
    V = V / norms
    return V, lam


def _effective_n(n: int, C: Optional[np.ndarray]) -> int:
    """
    Effective sample size after partialing out C.

    n_eff = n - rank(C)
    """
    if C is None or C.size == 0:
        return n
    r = np.linalg.matrix_rank(np.asarray(C, float))
    return n - int(r)

# ---------------------------------------------------------------------
#                              Core PCEV
# ---------------------------------------------------------------------

@dataclass
class PCEVFit:
    """
    Container for PCEV fit results.

    Attributes
    ----------
    weights : array (p,)
        First PCEV loading vector (largest root). SW-normalized.
    scores : array (n,)
        PCEV scores t = Y_res @ weights.
    h2 : float
        Proportion of variance explained by X: λ_max / (1 + λ_max).
    lambda_max : float
        Largest generalized eigenvalue.
    lambdas : array (k,)
        All non-zero generalized eigenvalues (descending). k ≤ min(p, q).
    n, p, q : int
        Sample size, number of responses, number of covariates of interest.
    ridge : float
        Ridge value applied to SW.
    """
    weights: np.ndarray
    scores: np.ndarray
    h2: float
    lambda_max: float
    lambdas: np.ndarray
    n: int
    p: int
    q: int
    ridge: float


def pcev_fit(
    Y: np.ndarray,
    X: np.ndarray,
    C: Optional[np.ndarray] = None,
    *,
    ridge: float = 1e-6,
) -> PCEVFit:
    """
    Fit PCEV without blocks by solving SB v = λ SW v.

    Maximizes h^2 = Var(E[w^T Y | X, C]) / Var(w^T Y | C), after residualizing Y and X on C.

    Parameters
    ----------
    Y : array, shape (n, p)
        Multivariate responses.
    X : array, shape (n, q) or (n,)
        Covariates of interest.
    C : array or None, shape (n, r)
        Nuisance/confounders to partial out from Y and X.
    ridge : float
        Nonnegative ridge added to SW for numerical stability.

    Returns
    -------
    PCEVFit
        Fit object with weights, scores, λ_max, h2, and all roots.

    Notes
    -----
    - Uses Cholesky whitening for a stable generalized eigen solve.
    - If SW is not PD, add a small `ridge` (default 1e-6).
    """
    Y = np.asarray(Y, dtype=float)
    X = np.asarray(X, dtype=float)
    n, p = Y.shape
    if X.ndim == 1:
        X = X[:, None]
    q = X.shape[1]

    # Residualize on C and center
    Yr = _proj_out(Y - Y.mean(axis=0), C)
    Xr = _proj_out(X - X.mean(axis=0), C)

    # Covariances
    Syy = _cov_colwise(Yr)                    # p×p
    Sxx = _cov_colwise(Xr)                    # q×q
    Sxx_inv = _pinv_psd(Sxx)
    Sxy = (Yr.T @ Xr) / (n - 1)               # p×q

    # Explained and residual covariance
    SB = Sxy @ (Sxx_inv @ Sxy.T)              # p×p, rank ≤ q
    SW = Syy - SB
    if ridge > 0:
        SW = SW + ridge * np.eye(p)

    # Generalized eigen solve
    try:
        V, lam_all = _chol_whiten_gen_eig(SB, SW)
    except LinAlgError:
        # Fallback through pseudoinverse if SW is not PD even after ridge
        SW_pinv = _pinv_psd(SW)
        M = SW_pinv @ SB                      # symmetric PSD
        lam_all, V = linalg.eigh(M)           # ascending
        # SW-normalize eigenvectors
        norms = np.sqrt(np.einsum("ij,ij->j", V, SW @ V))
        norms[norms == 0] = 1.0
        V = V / norms

    # Keep only positive roots (numerical guard)
    pos = lam_all > 1e-15
    lam_pos = lam_all[pos]
    # Sort descending for convenience
    idx = np.argsort(lam_pos)[::-1]
    lam_desc = lam_pos[idx]
    V_desc = V[:, pos][:, idx]

    # First component - handle case with no positive eigenvalues
    if V_desc.shape[1] > 0:
        v1 = V_desc[:, 0]
        lam1 = float(lam_desc[0])
        h2 = lam1 / (1.0 + lam1)
        t = Yr @ v1
    else:
        # No positive eigenvalues - return zeros
        v1 = np.zeros(p)
        lam1 = 0.0
        h2 = 0.0
        t = np.zeros(n)

    return PCEVFit(
        weights=v1,
        scores=t,
        h2=float(h2),
        lambda_max=lam1,
        lambdas=lam_desc.copy(),
        n=n, p=p, q=q,
        ridge=float(ridge),
    )

# ---------------------------------------------------------------------
#                           P-value Calculators
# ---------------------------------------------------------------------

def pcev_pvalue_exact(fit: PCEVFit, X: np.ndarray, C: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Exact analytic p-value for q = 1 using the classical F-test on R^2 = h^2.

    Test
    ----
    H0: h^2 = 0  vs  H1: h^2 > 0

    Statistic
    ---------
    F = ((n_eff - p - 1) / p) * (h^2 / (1 - h^2))  ~  F_{p, n_eff - p - 1}

    Parameters
    ----------
    fit : PCEVFit
        Result from pcev_fit.
    X : array, shape (n,) or (n,1)
        Covariate of interest used in the fit.
    C : array or None
        Nuisance as in the fit.

    Returns
    -------
    pval : float
    info : dict
        Contains 'F', 'df1', 'df2'.

    Raises
    ------
    ValueError
        If q != 1 or df2 ≤ 0.
    """
    X = np.asarray(X)
    q = X.shape[1] if X.ndim == 2 else 1
    if q != 1:
        raise ValueError("Exact p-value is defined here only for q == 1.")
    n_eff = _effective_n(fit.n, C)
    df1 = fit.p
    df2 = n_eff - fit.p - 1
    if df2 <= 0:
        raise ValueError(f"Nonpositive df2={df2}. Need n_eff > p + 1.")
    h2 = float(fit.h2)
    F = ((n_eff - fit.p - 1) / fit.p) * (h2 / max(1.0 - h2, 1e-15))
    pval = stats.f.sf(F, df1, df2)
    return float(pval), {"F": float(F), "df1": int(df1), "df2": int(df2)}


def pcev_pvalue_wilks_rao(fit: PCEVFit, C: Optional[np.ndarray] = None) -> Tuple[float, Dict[str, Any]]:
    """
    Wilks' lambda with Rao's F-approximation for q ≥ 1.

    Definition
    ----------
    Λ = ∏_{i=1}^{k} 1/(1 + λ_i),   k = min(p, q_eff)
    where λ_i are the positive generalized roots (descending).

    Rao's approximation (MANOVA-style) maps Λ to an F-statistic:
      m = (|p - q| - 1)/2
      s = sqrt((p^2 q^2 - 4) / (p^2 + q^2 - 5))  if denominator > 0 else 1
      nR = (n_eff - p - 1 - q)/2
      F ≈ ((1 - Λ^{1/s}) / (Λ^{1/s})) * ((2 nR + s + 1)/(2 m + s + 1))
      df1 = (2 m + s + 1) * p
      df2 = (2 nR + s + 1) * q

    Parameters
    ----------
    fit : PCEVFit
        Result from pcev_fit. Needs all positive roots in `fit.lambdas`.
    C : array or None
        Nuisance as in the fit (only its rank affects n_eff).

    Returns
    -------
    pval : float
    info : dict
        Contains 'Lambda', 'F', 'df1', 'df2'.

    Notes
    -----
    - This is an approximation. For q=1 prefer `pcev_pvalue_exact`.
    - Requires n_eff > p + q.
    """
    n_eff = _effective_n(fit.n, C)
    p, q = fit.p, fit.q
    if n_eff <= p + q:
        raise ValueError("Need n_eff > p + q for Wilks/Rao approximation.")

    # Wilks' lambda using all positive generalized roots
    lam = np.asarray(fit.lambdas, float)
    if lam.size == 0:
        return 1.0, {"Lambda": 1.0, "F": 0.0, "df1": p, "df2": n_eff - p - 1}

    Lambda = float(np.prod(1.0 / (1.0 + lam)))
    # Guards
    Lambda = min(max(Lambda, 1e-15), 1.0 - 1e-15)

    # Rao F-approximation
    s_den = p * p + q * q - 5.0
    s_num = p * p * q * q - 4.0
    s = np.sqrt(s_num / s_den) if s_den > 0 and s_num > 0 else 1.0
    m = 0.5 * (abs(p - q) - 1.0)
    nR = 0.5 * (n_eff - p - 1.0 - q)

    L_pow = Lambda ** (1.0 / s)
    F = ((1.0 - L_pow) / L_pow) * ((2.0 * nR + s + 1.0) / (2.0 * m + s + 1.0))
    df1 = int((2.0 * m + s + 1.0) * p)
    df2 = int((2.0 * nR + s + 1.0) * q)
    df1 = max(df1, 1)
    df2 = max(df2, 1)

    pval = stats.f.sf(F, df1, df2)
    return float(pval), {"Lambda": float(Lambda), "F": float(F), "df1": df1, "df2": df2}


def pcev_permutation_test(
    Y: np.ndarray,
    X: np.ndarray,
    C: Optional[np.ndarray] = None,
    *,
    n_perm: int = 1000,
    strat: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator | int] = None,
    ridge: float = 1e-6,
    n_jobs: int = -1,
    prefer: str = "threads",
) -> Dict[str, Any]:
    """
    Stratified permutation test for the largest root (h^2).

    Procedure
    ---------
    - Residualize Y and X on C.
    - Compute observed h^2.
    - Permute rows of X within each stratum of `strat` and recompute h^2.
    - Mid-p value with +1 correction.

    Parameters
    ----------
    Y, X, C : arrays
        As in `pcev_fit`. X may be 1d or 2d.
    n_perm : int
        Number of permutations.
    strat : array-like or None, shape (n,)
        Integer labels for strata. If None, all samples share one stratum.
    rng : int or np.random.Generator or None
        Seed or Generator for reproducibility.
    ridge : float
        Ridge passed to the PCEV fit.
    n_jobs : int
        Parallel workers for joblib. Default uses all cores.
    prefer : {'threads','processes'}
        Joblib backend hint.

    Returns
    -------
    dict
        {
          'p_perm': float,
          'h2_obs': float,
          'lambda_obs': float,
          'null_h2': array(n_perm,),
          'weights': array(p,),
          'scores': array(n,)
        }
    """
    Y = np.asarray(Y, float)
    X = np.asarray(X, float)
    n = Y.shape[0]
    if X.ndim == 1:
        X = X[:, None]

    if strat is None:
        strat = np.zeros(n, dtype=int)
    else:
        strat = np.asarray(strat)
        if strat.shape[0] != n:
            raise ValueError("strat length must equal n.")

    # Observed
    fit = pcev_fit(Y, X, C, ridge=ridge)
    h2_obs = float(fit.h2)

    # Pre-index strata
    groups = [np.flatnonzero(strat == s) for s in np.unique(strat)]

    rng = np.random.default_rng(rng)

    def _one_perm(seed: int) -> float:
        rg = np.random.default_rng(seed)
        Xp = X.copy()
        for g in groups:
            Xp[g] = Xp[rg.permutation(g), :]
        return pcev_fit(Y, Xp, C, ridge=ridge).h2

    seeds = rng.integers(0, 2**32 - 1, size=n_perm, dtype=np.uint32)
    null_h2 = Parallel(n_jobs=n_jobs, prefer=prefer, require="sharedmem")(
        delayed(_one_perm)(int(s)) for s in seeds
    )
    null_h2 = np.asarray(null_h2, float)

    # Mid-p with +1 correction
    p_perm = (np.sum(null_h2 > h2_obs) + 0.5 * np.sum(null_h2 == h2_obs) + 1) / (n_perm + 1)

    return {
        "p_perm": float(p_perm),
        "h2_obs": h2_obs,
        "lambda_obs": float(fit.lambda_max),
        "null_h2": null_h2,
        "weights": fit.weights.copy(),
        "scores": fit.scores.copy(),
    }

# ---------------------------------------------------------------------
#                       One-shot Fit + Inference
# ---------------------------------------------------------------------

def pcev_infer(
    Y: np.ndarray,
    X: np.ndarray,
    C: Optional[np.ndarray] = None,
    *,
    inference: str = "exact",
    n_perm: int = 0,
    strat: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator | int] = None,
    ridge: float = 1e-6,
    n_jobs: int = -1,
    prefer: str = "threads",
) -> Dict[str, Any]:
    """
    Convenience wrapper: fit PCEV then compute p-values per chosen inference.

    Parameters
    ----------
    Y, X, C : arrays
        As in `pcev_fit`.
    inference : {'exact','approx','permutation','both'}
        - 'exact': exact analytic p for q=1; raises if q>1.
        - 'approx': Wilks/Rao approximation using all positive roots.
        - 'permutation': stratified permutation p only.
        - 'both': analytic (exact if q=1 else approx) and permutation (requires n_perm>0).
    n_perm, strat, rng, ridge, n_jobs, prefer
        Passed to the relevant functions.

    Returns
    -------
    dict
        Includes 'fit' (PCEVFit), analytic p (if requested), and permutation outputs (if requested).
    """
    fit = pcev_fit(Y, X, C, ridge=ridge)
    q = X.shape[1] if np.asarray(X).ndim == 2 else 1

    out: Dict[str, Any] = {"fit": fit}

    if inference in ("exact", "both"):
        if q != 1:
            raise ValueError("Exact inference requested but q != 1. Use 'approx' or 'permutation'.")
        p_a, info = pcev_pvalue_exact(fit, X, C)
        out["p_analytic"] = p_a
        out["analytic_info"] = info

    if inference == "approx":
        p_a, info = pcev_pvalue_wilks_rao(fit, C)
        out["p_analytic"] = p_a
        out["analytic_info"] = info

    if inference in ("permutation", "both"):
        if n_perm <= 0:
            raise ValueError("Permutation inference requested but n_perm <= 0.")
        perm = pcev_permutation_test(
            Y, X, C, n_perm=n_perm, strat=strat, rng=rng, ridge=ridge, n_jobs=n_jobs, prefer=prefer
        )
        out["p_perm"] = perm["p_perm"]
        out["perm"] = perm

    return out

# ---------------------------------------------------------------------
#                                   CLI
# ---------------------------------------------------------------------

def _parse_cli():
    import argparse, json, sys, os
    ap = argparse.ArgumentParser(description="PCEV (no blocks): fit + inference.")
    ap.add_argument("--Y", required=True, help="Path to .npy for Y (n×p).")
    ap.add_argument("--X", required=True, help="Path to .npy for X (n×q or n,).")
    ap.add_argument("--C", default=None, help="Path to .npy for C (n×r).")
    ap.add_argument("--inference", choices=["exact","approx","permutation","both"], default="exact")
    ap.add_argument("--n_perm", type=int, default=0, help="Permutations if permutation/both.")
    ap.add_argument("--strat", default=None, help="Path to .npy int labels of length n.")
    ap.add_argument("--seed", type=int, default=None, help="Seed for permutations.")
    ap.add_argument("--ridge", type=float, default=1e-6, help="Ridge for SW.")
    ap.add_argument("--n_jobs", type=int, default=-1, help="joblib workers.")
    ap.add_argument("--prefer", choices=["threads","processes"], default="threads")
    ap.add_argument("--save_prefix", default="pcev", help="Prefix for saved arrays.")
    args = ap.parse_args()

    Y = np.load(args.Y)
    X = np.load(args.X)
    C = None if args.C in (None, "None") else np.load(args.C)
    strat = None if args.strat in (None, "None") else np.load(args.strat)

    res = pcev_infer(
        Y, X, C,
        inference=args.inference,
        n_perm=args.n_perm,
        strat=strat,
        rng=args.seed,
        ridge=args.ridge,
        n_jobs=args.n_jobs,
        prefer=args.prefer,
    )

    # Print a compact JSON summary
    summary = {
        "n": res["fit"].n,
        "p": res["fit"].p,
        "q": res["fit"].q,
        "h2": float(res["fit"].h2),
        "lambda_max": float(res["fit"].lambda_max),
        "ridge": float(res["fit"].ridge),
    }
    if "p_analytic" in res:
        summary["p_analytic"] = float(res["p_analytic"])
        summary["analytic_info"] = res["analytic_info"]
    if "p_perm" in res:
        summary["p_perm"] = float(res["p_perm"])
    print(json.dumps(summary, indent=2))

    # Save heavy arrays
    np.save(f"{args.save_prefix}_weights.npy", res["fit"].weights)
    np.save(f"{args.save_prefix}_scores.npy", res["fit"].scores)
    if "perm" in res:
        np.save(f"{args.save_prefix}_null_h2.npy", res["perm"]["null_h2"])


if __name__ == "__main__":
    _parse_cli()
