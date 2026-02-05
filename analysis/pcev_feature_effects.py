"""Utilities for repeated CV feature comparison with PCEV."""

from __future__ import annotations

import io
import itertools
import threading
import warnings
from contextlib import redirect_stdout
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

try:
    from .pcev_sklearn import PCEV, _apply_residualizer, _cov_colwise, _pinv_psd
except ImportError:  # pragma: no cover
    from pcev_sklearn import PCEV, _apply_residualizer, _cov_colwise, _pinv_psd
import time


FEATURE_PREFIXES: Tuple[str, ...] = (
    "ent_E",
    "ent_I",
    "EI_ent",
    "rate_E",
    "rate_I",
    "EI_rate",
)

CONFOUNDERS_CATEGORICAL: Tuple[str, ...] = ("Sex", "Diagnosis", "Country")
CONFOUNDERS_NUMERIC: Tuple[str, ...] = ("gof_corr",)


@dataclass(frozen=True)
class FeatureCombo:
    key: str
    label: str
    groups: Tuple[str, ...]
    columns: Tuple[str, ...]


@dataclass
class CvContext:
    Y: np.ndarray
    X: np.ndarray
    subject_ids: np.ndarray
    strata: np.ndarray
    conf_matrix: np.ndarray
    splits: List[List[Tuple[np.ndarray, np.ndarray]]]
    diag_labels: Optional[np.ndarray]
    sex_labels: Optional[np.ndarray]

    def with_exposure(self, X: np.ndarray) -> "CvContext":
        return CvContext(
            Y=self.Y,
            X=X,
            subject_ids=self.subject_ids,
            strata=self.strata,
            conf_matrix=self.conf_matrix,
            splits=self.splits,
            diag_labels=self.diag_labels,
            sex_labels=self.sex_labels,
        )


def make_feature_map(df: pd.DataFrame, prefixes: Sequence[str] = FEATURE_PREFIXES) -> Dict[str, List[str]]:
    feature_map: Dict[str, List[str]] = {}
    for pref in prefixes:
        cols = df.filter(regex=fr"^{pref}_").columns.tolist()
        if not cols:
            raise KeyError(f"No columns match prefix '{pref}'")
        feature_map[pref] = cols
    return feature_map


def build_combinations(feature_map: Dict[str, Sequence[str]]) -> List[FeatureCombo]:
    combos: List[FeatureCombo] = []
    groups = sorted(feature_map.keys())
    # singletons
    for g in groups:
        cols = tuple(feature_map[g])
        combos.append(
            FeatureCombo(
                key=g,
                label=g,
                groups=(g,),
                columns=cols,
            )
        )
    # pairs (unordered)
    for g1, g2 in itertools.combinations(groups, 2):
        cols = tuple(feature_map[g1]) + tuple(feature_map[g2])
        key = f"{g1}__{g2}"
        label = f"{g1}+{g2}"
        combos.append(
            FeatureCombo(
                key=key,
                label=label,
                groups=(g1, g2),
                columns=cols,
            )
        )
    # all features together
    all_cols: List[str] = []
    for g in groups:
        all_cols.extend(feature_map[g])
    combos.append(
        FeatureCombo(
            key="__".join(groups),
            label="all_features",
            groups=tuple(groups),
            columns=tuple(all_cols),
        )
    )
    return combos


def make_confounder_transformer(
    categorical: Sequence[str] | None = None,
    numeric: Sequence[str] | None = None,
) -> ColumnTransformer:
    if categorical is None:
        categorical = CONFOUNDERS_CATEGORICAL
    if numeric is None:
        numeric = CONFOUNDERS_NUMERIC
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(categorical),
            ),
            ("numeric", "passthrough", list(numeric)),
        ],
        remainder="drop",
    )
    return encoder


def make_strata(df: pd.DataFrame, diag_col: str = "Diagnosis", sex_col: str = "Sex") -> np.ndarray:
    combined = df[[diag_col, sex_col]].astype(str).agg("__".join, axis=1)
    uniques, inverse = np.unique(combined.to_numpy(), return_inverse=True)
    return inverse


@dataclass
class FoldResult:
    combo_key: str
    combo_label: str
    repeat: int
    fold: int
    n_train: int
    n_test: int
    h2_train: float
    h2_test: float
    r2_test: float
    covariate_group: Optional[str] = None


@dataclass
class ScoreRecord:
    combo_key: str
    combo_label: str
    repeat: int
    fold: int
    subject_id: object
    score: float
    x_value_vec: Optional[Tuple[float, ...]] = None
    x_residualized_vec: Optional[Tuple[float, ...]] = None
    y_residualized_vec: Optional[Tuple[float, ...]] = None
    diagnosis_label: Optional[object] = None
    sex_label: Optional[object] = None
    covariate_group: Optional[str] = None

    def to_dict(self, *, legacy_scalar_alias: bool = False) -> Dict[str, object]:
        def _scalar_from_vec(vec: Optional[Tuple[float, ...]]) -> Optional[float]:
            if vec is None or len(vec) != 1:
                return None
            value = vec[0]
            return float(value) if np.isfinite(value) else None

        data: Dict[str, object] = {
            "combo_key": self.combo_key,
            "combo_label": self.combo_label,
            "repeat": self.repeat,
            "fold": self.fold,
            "subject_id": self.subject_id,
            "score": self.score,
            "x_value_vec": self.x_value_vec,
            "x_residualized_vec": self.x_residualized_vec,
            "y_residualized_vec": self.y_residualized_vec,
            "diagnosis_label": self.diagnosis_label,
            "sex_label": self.sex_label,
            "covariate_group": self.covariate_group,
        }

        scalar_value = _scalar_from_vec(self.x_value_vec)
        scalar_residualized = _scalar_from_vec(self.x_residualized_vec)
        data["x_value"] = scalar_value
        data["x_residualized"] = scalar_residualized

        if legacy_scalar_alias:
            data["age"] = scalar_value
            data["age_residualized"] = scalar_residualized

        return data


@dataclass
class HoldoutMetrics:
    combo_key: str
    combo_label: str
    heldout_country: object
    n_train: int
    n_test: int
    h2_train_full: float
    r2_out_country: float


@dataclass
class HoldoutScoreRecord:
    combo_key: str
    combo_label: str
    heldout_country: object
    subject_id: object
    score: float
    x_value_vec: Optional[Tuple[float, ...]] = None
    x_residualized_vec: Optional[Tuple[float, ...]] = None

    def to_dict(self, *, legacy_scalar_alias: bool = False) -> Dict[str, object]:
        def _scalar_from_vec(vec: Optional[Tuple[float, ...]]) -> Optional[float]:
            if vec is None or len(vec) != 1:
                return None
            value = vec[0]
            return float(value) if np.isfinite(value) else None

        data: Dict[str, object] = {
            "combo_key": self.combo_key,
            "combo_label": self.combo_label,
            "heldout_country": self.heldout_country,
            "subject_id": self.subject_id,
            "score": self.score,
            "x_value_vec": self.x_value_vec,
            "x_residualized_vec": self.x_residualized_vec,
        }
        scalar_value = _scalar_from_vec(self.x_value_vec)
        scalar_residualized = _scalar_from_vec(self.x_residualized_vec)
        data["x_value"] = scalar_value
        data["x_residualized"] = scalar_residualized
        if legacy_scalar_alias:
            data["age"] = scalar_value
            data["age_residualized"] = scalar_residualized
        return data


def _evaluate_fold(
    *,
    combo: FeatureCombo,
    covariate_group: Optional[str],
    repeat_idx: int,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray,
    conf_matrix: Optional[np.ndarray] = None,
    confounders_df: Optional[pd.DataFrame] = None,
    subject_ids: np.ndarray,
    store_scalar_x: bool,
    diag_labels: Optional[np.ndarray],
    sex_labels: Optional[np.ndarray],
    pcev_kwargs: Optional[Dict[str, object]] = None,
    train_strata: Optional[np.ndarray] = None,
) -> Tuple[FoldResult, List[ScoreRecord]]:
    if pcev_kwargs is None:
        pcev_kwargs = {}
    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    X_train = X[train_idx]
    X_test = X[test_idx]

    if conf_matrix is not None:
        C_train = conf_matrix[train_idx]
        C_test = conf_matrix[test_idx]
    elif confounders_df is not None:
        transformer = make_confounder_transformer()
        C_train = transformer.fit_transform(confounders_df.iloc[train_idx])
        C_test = transformer.transform(confounders_df.iloc[test_idx])
    else:
        raise ValueError("Either conf_matrix or confounders_df must be provided")

    estimator = PCEV(**pcev_kwargs)
    estimator.fit(Y_train, X_train, C=C_train)

    if getattr(estimator, "explained_variance_ratio_", None) is not None and estimator.explained_variance_ratio_.size:
        h2_train = float(estimator.explained_variance_ratio_[0])
    else:
        h2_train = float("nan")

    scores_test = estimator.project_scores(Y_test, C=C_test)[:, 0]
    res_test = estimator.residualize(Y=Y_test, x=X_test, C=C_test)
    Y_test_res = np.asarray(res_test.get("Y", np.empty((0, 0))), dtype=float)
    X_test_res = np.asarray(res_test.get("X", np.empty((0, 0))), dtype=float)
    if X_test_res.ndim == 1:
        X_test_res = X_test_res[:, None]
    if Y_test_res.ndim == 1:
        Y_test_res = Y_test_res[:, None]

    w = estimator.components_[0] if getattr(estimator, "components_", None) is not None else np.empty(0, dtype=float)
    h2_test = _compute_out_of_sample_h2(Y_test_res, X_test_res, w, rcond=getattr(estimator, "rcond", 1e-12))

    if X_test_res.shape[1] == 1:
        r2 = h2_test if np.isfinite(h2_test) else float("nan")
    else:
        r2 = float("nan")

    fold_result = FoldResult(
        combo_key=combo.key,
        combo_label=combo.label,
        repeat=repeat_idx,
        fold=fold_idx,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        h2_train=h2_train,
        h2_test=h2_test,
        r2_test=r2,
        covariate_group=covariate_group,
    )

    records: List[ScoreRecord] = []
    diag_subset = diag_labels[test_idx] if diag_labels is not None else [None] * test_idx.size
    sex_subset = sex_labels[test_idx] if sex_labels is not None else [None] * test_idx.size

    for row_idx, (sid, score, diag_val, sex_val) in enumerate(
        zip(subject_ids[test_idx], scores_test, diag_subset, sex_subset)
    ):
        raw_vec = X_test[row_idx]
        res_vec = X_test_res[row_idx] if X_test_res.shape[0] > row_idx else np.full((X_test.shape[1],), np.nan)
        
        # Store residualized Y for this subject
        y_res_vec = Y_test_res[row_idx] if Y_test_res.shape[0] > row_idx else np.full(Y_test_res.shape[1], np.nan)
        y_residualized_vec = tuple(float(v) for v in y_res_vec.tolist()) if np.all(np.isfinite(y_res_vec)) else None

        if store_scalar_x:
            raw_val = raw_vec[0] if raw_vec.size else np.nan
            res_val = res_vec[0] if res_vec.size else np.nan
            x_value_vec = (float(raw_val),) if np.isfinite(raw_val) else None
            x_residualized_vec = (float(res_val),) if np.isfinite(res_val) else None
        else:
            x_value_vec = tuple(float(v) for v in raw_vec.tolist()) if np.all(np.isfinite(raw_vec)) else None
            x_residualized_vec = tuple(float(v) for v in res_vec.tolist()) if np.all(np.isfinite(res_vec)) else None

        records.append(
            ScoreRecord(
                combo_key=combo.key,
                combo_label=combo.label,
                repeat=repeat_idx,
                fold=fold_idx,
                subject_id=sid,
                score=float(score),
                x_value_vec=x_value_vec,
                x_residualized_vec=x_residualized_vec,
                y_residualized_vec=y_residualized_vec,
                diagnosis_label=diag_val,
                sex_label=sex_val,
                covariate_group=covariate_group,
            )
        )

    return fold_result, records



def _compute_out_of_sample_h2(
    Y_res: np.ndarray,
    X_res: np.ndarray,
    component: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> float:
    if Y_res.size == 0 or X_res.size == 0:
        return float("nan")
    n = Y_res.shape[0]
    if n <= 1:
        return float("nan")
    if X_res.ndim == 1:
        X_res = X_res[:, None]
    if Y_res.ndim == 1:
        Y_res = Y_res[:, None]

    Syy = _cov_colwise(Y_res)
    Sxx = _cov_colwise(X_res)
    try:
        Sxx_inv = _pinv_psd(Sxx, rcond=rcond)
    except Exception:
        return float("nan")
    Sxy = (Y_res.T @ X_res) / (n - 1)
    Sb = Sxy @ (Sxx_inv @ Sxy.T)

    w = np.asarray(component, dtype=float).ravel()
    if w.size == 0 or Syy.size == 0:
        return float("nan")

    numerator = float(w @ (Sb @ w))
    denominator = float(w @ (Syy @ w))
    if denominator <= 1e-12 or numerator < 0:
        return float("nan")
    h2 = numerator / denominator
    return float(np.clip(h2, 0.0, 1.0))


def _permute_columns_within_strata(
    df: pd.DataFrame,
    *,
    columns: Sequence[str],
    strata: np.ndarray,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Permute specified columns within each stratum, returning a new DataFrame."""
    if df.empty or not columns:
        return df.copy()
    out = df.copy()
    strata = np.asarray(strata)
    unique = np.unique(strata)
    for col in columns:
        values = out[col].to_numpy(copy=True)
        for s in unique:
            idx = np.flatnonzero(strata == s)
            if idx.size <= 1:
                continue
            shuffled = idx[rng.permutation(idx.size)]
            values[idx] = values[shuffled]
        out[col] = values
    return out


def _permute_matrix_rows(
    matrix: np.ndarray,
    groups: Tuple[np.ndarray, ...],
    rng: np.random.Generator,
) -> np.ndarray:
    """Permute matrix rows within each group, returning a new array."""
    if not groups:
        return np.asarray(matrix, dtype=float).copy()
    base = np.asarray(matrix, dtype=float)
    permuted = base.copy()
    for idx in groups:
        if idx.size <= 1:
            continue
        order = idx[rng.permutation(idx.size)]
    permuted[idx] = base[order]
    return permuted


# All permutation logic now uses unified numpy implementation 
# that follows same statistical logic as numpy backend


def _prepare_score_exposure_data(group: pd.DataFrame) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    vec_key = "x_residualized_vec"
    scalar_key = "x_residualized"

    if "score" not in group.columns:
        return None

    exposures: List[np.ndarray] = []
    scores: List[float] = []

    if vec_key in group.columns:
        vec_frame = group[["score", vec_key]].dropna(subset=["score", vec_key])
        for score, vec in zip(vec_frame["score"], vec_frame[vec_key]):
            arr = np.asarray(vec, dtype=float)
            if arr.ndim != 1 or arr.size == 0 or not np.all(np.isfinite(arr)):
                continue
            exposures.append(arr)
            scores.append(float(score))

    if not exposures and scalar_key in group.columns:
        scalar_frame = group[["score", scalar_key]].dropna(subset=["score", scalar_key])
        for score, scalar_val in zip(scalar_frame["score"], scalar_frame[scalar_key]):
            if scalar_val is None or not np.isfinite(scalar_val):
                continue
            exposures.append(np.asarray([float(scalar_val)], dtype=float))
            scores.append(float(score))

    if not exposures:
        return None

    lengths = {arr.shape[0] for arr in exposures}
    if len(lengths) != 1:
        return None

    try:
        X = np.vstack(exposures)
    except ValueError:
        return None
    s = np.asarray(scores, dtype=float)
    return s, X



def _compute_vip_from_scores(group: pd.DataFrame, n_features: int) -> np.ndarray:
    """Compute VIP scores using out-of-sample data per repetition."""
    if group.empty or "score" not in group.columns or "y_residualized_vec" not in group.columns:
        return np.full(n_features, np.nan)
    
    scores = group["score"].values
    scores_centered = scores - scores.mean()
    
    # Extract residualized neural features
    Y_list = []
    for y_vec in group["y_residualized_vec"]:
        if y_vec is not None:
            Y_list.append(np.array(y_vec))
    
    if not Y_list:
        return np.full(n_features, np.nan)
    
    Y_residualized = np.vstack(Y_list)  # Shape: (n_subjects, n_features)
    
    if Y_residualized.shape[1] != n_features:
        return np.full(n_features, np.nan)
    
    # Compute VIP for each neural feature
    vip_scores = np.zeros(n_features)
    for j in range(n_features):
        y_feature_j = Y_residualized[:, j]
        y_feature_centered = y_feature_j - y_feature_j.mean()
        
        # VIP = correlation between neural feature j and PCEV scores
        if np.std(scores_centered) > 1e-12 and np.std(y_feature_centered) > 1e-12:
            correlation = np.corrcoef(scores_centered, y_feature_centered)[0, 1]
            vip_scores[j] = correlation if np.isfinite(correlation) else 0.0
        else:
            vip_scores[j] = 0.0
    
    return vip_scores


def _compute_repeat_vip_from_scores(score_df_local: pd.DataFrame, feature_combo: FeatureCombo, covariate_group: Optional[str] = None, n_repeats: int = 5) -> np.ndarray:
    """Compute VIP scores per repetition."""
    n_features = len(feature_combo.columns)
    vip_values = np.full((n_repeats, n_features), np.nan)
    
    if score_df_local.empty:
        return vip_values
        
    subset = score_df_local.loc[score_df_local["combo_key"] == feature_combo.key]
    if covariate_group is not None and "covariate_group" in subset.columns:
        subset = subset.loc[subset["covariate_group"] == covariate_group]
    
    if subset.empty:
        return vip_values
    
    for repeat_idx, group in subset.groupby("repeat"):
        try:
            vip_rep = _compute_vip_from_scores(group, n_features)
            if 0 <= repeat_idx < vip_values.shape[0]:
                vip_values[int(repeat_idx), :] = vip_rep
        except Exception:
            pass  # Keep NaN values
    
    return vip_values


def _compute_h2_from_scores(group: pd.DataFrame) -> float:
    prepared = _prepare_score_exposure_data(group)
    if prepared is None:
        return float("nan")
    scores, X = prepared
    if scores.ndim != 1 or X.ndim != 2:
        return float("nan")

    mask = np.isfinite(scores) & np.all(np.isfinite(X), axis=1)
    scores = scores[mask]
    X = X[mask]

    n_samples = scores.shape[0]
    n_features = X.shape[1] if X.ndim == 2 else 1
    if n_samples <= 1 or n_features == 0 or n_samples <= n_features:
        return float("nan")

    scores_centered = scores - scores.mean()
    X_centered = X - X.mean(axis=0, keepdims=True)

    Syy = float(np.dot(scores_centered, scores_centered) / (n_samples - 1))
    if not np.isfinite(Syy) or Syy <= 1e-12:
        return float("nan")

    Sxx = (X_centered.T @ X_centered) / (n_samples - 1)
    try:
        Sxx_inv = _pinv_psd(Sxx, rcond=1e-12)
    except Exception:
        return float("nan")

    Sxy = (scores_centered @ X_centered) / (n_samples - 1)  # shape (q,)
    Sb = Sxy @ (Sxx_inv @ Sxy.T)
    if Sb < 0:
        return float("nan")
    h2 = Sb / Syy
    return float(np.clip(h2, 0.0, 1.0))


def _make_repeated_splits(
    strata: np.ndarray,
    *,
    n_repeats: int,
    n_splits: int,
    seed: int,
) -> List[List[Tuple[np.ndarray, np.ndarray]]]:
    splits: List[List[Tuple[np.ndarray, np.ndarray]]] = []
    dummy = np.zeros_like(strata)
    for repeat_idx in range(n_repeats):
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed + repeat_idx)
        fold_pairs = [(np.asarray(train_idx), np.asarray(test_idx)) for train_idx, test_idx in cv.split(dummy, strata)]
        splits.append(fold_pairs)
    return splits


def _fit_confounders_matrix(
    df: pd.DataFrame,
    *,
    categorical: Sequence[str] | None = None,
    numeric: Sequence[str] | None = None,
) -> Tuple[np.ndarray, ColumnTransformer]:
    categorical = CONFOUNDERS_CATEGORICAL if categorical is None else tuple(categorical)
    numeric = CONFOUNDERS_NUMERIC if numeric is None else tuple(numeric)
    confounder_cols = list(categorical) + list(numeric)
    transformer = make_confounder_transformer(categorical=categorical, numeric=numeric)
    matrix = transformer.fit_transform(df[confounder_cols])
    return np.asarray(matrix, dtype=float), transformer


def _build_cv_context(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    x_cols: Sequence[str],
    id_col: str,
    diag_col: Optional[str],
    sex_col: Optional[str],
    strat_labels: Optional[np.ndarray],
    n_repeats: int,
    n_splits: int,
    seed: int,
    confounder_categorical: Sequence[str],
    confounder_numeric: Sequence[str],
    conf_matrix: Optional[np.ndarray],
) -> CvContext:
    feature_cols = list(feature_combo.columns)
    x_cols = list(x_cols)
    Y = df[feature_cols].to_numpy(dtype=float, copy=False)
    X = df[x_cols].to_numpy(dtype=float, copy=False)
    subject_ids = df[id_col].to_numpy()
    diag_labels = df[diag_col].astype(str).to_numpy() if diag_col else None
    sex_labels = df[sex_col].astype(str).to_numpy() if sex_col else None

    if strat_labels is not None:
        strata = np.asarray(strat_labels)
        if strata.shape[0] != df.shape[0]:
            raise ValueError("strat_labels must match number of rows in df.")
    else:
        fallback_diag = diag_col or ("Diagnosis" if "Diagnosis" in df.columns else None)
        fallback_sex = sex_col or ("Sex" if "Sex" in df.columns else None)
        if fallback_diag is None or fallback_sex is None:
            strata = np.zeros(df.shape[0], dtype=int)
        else:
            strata = make_strata(df, diag_col=fallback_diag, sex_col=fallback_sex)

    if conf_matrix is None:
        conf_matrix, _ = _fit_confounders_matrix(
            df,
            categorical=confounder_categorical,
            numeric=confounder_numeric,
        )
    else:
        conf_matrix = np.asarray(conf_matrix, dtype=float)
        if conf_matrix.shape[0] != df.shape[0]:
            raise ValueError("conf_matrix must have the same number of rows as df.")

    splits = _make_repeated_splits(
        strata,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
    )

    return CvContext(
        Y=np.asarray(Y, dtype=float),
        X=np.asarray(X, dtype=float),
        subject_ids=subject_ids,
        strata=np.asarray(strata, dtype=int),
        conf_matrix=np.asarray(conf_matrix, dtype=float),
        splits=splits,
        diag_labels=diag_labels,
        sex_labels=sex_labels,
    )


def _permute_design_matrix_within_strata(
    X: np.ndarray,
    strata: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    base = np.asarray(X, dtype=float)
    strata = np.asarray(strata)
    permuted = base.copy()
    unique = np.unique(strata)
    for s in unique:
        idx = np.flatnonzero(strata == s)
        if idx.size <= 1:
            continue
        shuffled = rng.permutation(idx.size)
        permuted[idx] = base[idx[shuffled]]
    return permuted


def _run_repeated_cv_generic(
    df: Optional[pd.DataFrame],
    *,
    feature_combo: FeatureCombo,
    x_cols: Sequence[str],
    id_col: str,
    covariate_group: Optional[str],
    diag_col: Optional[str],
    sex_col: Optional[str],
    n_repeats: int,
    n_splits: int,
    seed: int,
    n_jobs: int,
    pcev_kwargs: Optional[Dict[str, object]],
    confounder_categorical: Sequence[str],
    confounder_numeric: Sequence[str],
    splits: Optional[List[List[Tuple[np.ndarray, np.ndarray]]]],
    conf_matrix: Optional[np.ndarray],
    legacy_scalar_alias: bool,
    context: Optional[CvContext] = None,
    joblib_verbose: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_cols = list(x_cols)
    store_scalar_x = len(x_cols) == 1

    if context is None:
        if df is None:
            raise ValueError("df must be provided when context is None.")

        feature_cols = list(feature_combo.columns)
        Y = df[feature_cols].to_numpy(dtype=float)
        X = df[x_cols].to_numpy(dtype=float)
        subject_ids = df[id_col].to_numpy()
        diag_labels = df[diag_col].astype(str).to_numpy() if diag_col else None
        sex_labels = df[sex_col].astype(str).to_numpy() if sex_col else None

        strata = make_strata(
            df,
            diag_col=diag_col or "Diagnosis",
            sex_col=sex_col or "Sex",
        )
        if splits is None:
            splits = _make_repeated_splits(
                strata,
                n_repeats=n_repeats,
                n_splits=n_splits,
                seed=seed,
            )
        if conf_matrix is None:
            conf_matrix, _ = _fit_confounders_matrix(
                df,
                categorical=confounder_categorical,
                numeric=confounder_numeric,
            )
        else:
            conf_matrix = np.asarray(conf_matrix, dtype=float)
            if conf_matrix.shape[0] != df.shape[0]:
                raise ValueError("conf_matrix must have the same number of rows as df.")
    else:
        if splits is not None:
            warnings.warn("Provided splits are ignored when context is used.", RuntimeWarning)
        if conf_matrix is not None:
            warnings.warn("Provided conf_matrix is ignored when context is used.", RuntimeWarning)

        Y = context.Y
        X = context.X
        subject_ids = context.subject_ids
        diag_labels = context.diag_labels
        sex_labels = context.sex_labels
        strata = context.strata
        conf_matrix = context.conf_matrix
        splits = context.splits

    start_time = time.time()
    def _worker(args: Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[List[FoldResult], List[ScoreRecord]]:
        repeat_idx, fold_pairs = args
        repeat_folds: List[FoldResult] = []
        repeat_scores: List[ScoreRecord] = []
        for fold_idx, (train_idx, test_idx) in enumerate(fold_pairs):
            fold_res, records = _evaluate_fold(
                combo=feature_combo,
                covariate_group=covariate_group,
                repeat_idx=repeat_idx,
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                Y=Y,
                X=X,
                conf_matrix=conf_matrix,
                confounders_df=None,
                subject_ids=subject_ids,
                store_scalar_x=store_scalar_x,
                diag_labels=diag_labels,
                sex_labels=sex_labels,
                pcev_kwargs=pcev_kwargs,
                train_strata=strata[train_idx],
            )
            repeat_folds.append(fold_res)
            repeat_scores.extend(records)
        return repeat_folds, repeat_scores

    tasks = [(repeat_idx, fold_pairs) for repeat_idx, fold_pairs in enumerate(splits)]
    if n_jobs == 1:
        outputs = [_worker(task) for task in tasks]
    else:
        outputs = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
            require="sharedmem",
            verbose=joblib_verbose,
        )(delayed(_worker)(task) for task in tasks)

    end_time = time.time()
    print(f"Completed {n_repeats}x{n_splits}-fold CV for combo '{feature_combo.label}' in {end_time - start_time:.2f} seconds.")
    fold_records: List[FoldResult] = []
    score_records: List[ScoreRecord] = []
    for folds, scores in outputs:
        fold_records.extend(folds)
        score_records.extend(scores)

    fold_df = pd.DataFrame([r.__dict__ for r in fold_records])
    score_dicts = [
        record.to_dict(legacy_scalar_alias=legacy_scalar_alias and store_scalar_x)
        for record in score_records
    ]
    score_df = pd.DataFrame(score_dicts)
    return fold_df, score_df


def run_repeated_cv(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    age_col: Optional[str] = None,
    x_cols: Optional[Sequence[str] | str] = None,
    id_col: str,
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
    splits: Optional[List[List[Tuple[np.ndarray, np.ndarray]]]] = None,
    conf_matrix: Optional[np.ndarray] = None,
    confounder_categorical: Sequence[str] = CONFOUNDERS_CATEGORICAL,
    confounder_numeric: Sequence[str] = CONFOUNDERS_NUMERIC,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_repeats <= 0 or n_splits <= 1:
        raise ValueError("n_repeats must be > 0 and n_splits must be > 1")
    if x_cols is None:
        if age_col is None:
            raise ValueError("Either x_cols or age_col must be provided.")
        x_cols = [age_col]
    elif isinstance(x_cols, str):
        x_cols = [x_cols]

    x_cols = list(x_cols)

    fold_df, score_df = _run_repeated_cv_generic(
        df,
        feature_combo=feature_combo,
        x_cols=x_cols,
        id_col=id_col,
        covariate_group=None,
        diag_col="Diagnosis" if "Diagnosis" in df.columns else None,
        sex_col="Sex" if "Sex" in df.columns else None,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
        n_jobs=n_jobs,
        pcev_kwargs=pcev_kwargs,
        confounder_categorical=confounder_categorical,
        confounder_numeric=confounder_numeric,
        splits=splits,
        conf_matrix=conf_matrix,
        legacy_scalar_alias=len(x_cols) == 1,
    )
    return fold_df, score_df


def run_repeated_cv_with_permutations(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    id_col: str,
    x_cols: Optional[Sequence[str] | str] = None,
    age_col: Optional[str] = None,
    covariate_group: Optional[str] = None,
    n_repeats: int = 5,
    n_splits: int = 5,
    n_perm: int = 1000,
    seed: int = 1234,
    strat_labels: Optional[np.ndarray] = None,
    cv_n_jobs: int = 1,
    perm_n_jobs: int = 1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
    conf_matrix: Optional[np.ndarray] = None,
    confounder_categorical: Sequence[str] = CONFOUNDERS_CATEGORICAL,
    confounder_numeric: Sequence[str] = CONFOUNDERS_NUMERIC,
    progress: bool = True,
) -> Dict[str, object]:
    """
    Run repeated CV together with a permutation-based null for the average held-out h².
    
    Uses numpy-only implementation for all computations. Performs permutation testing
    by shuffling columns within diagnosis+sex strata and running the full CV pipeline
    to ensure statistical correctness.
    """

    if n_repeats <= 0 or n_splits <= 1:
        raise ValueError("n_repeats must be > 0 and n_splits must be > 1")
    if n_perm <= 0:
        raise ValueError("n_perm must be > 0")

    if x_cols is None:
        if age_col is None:
            raise ValueError("Either x_cols or age_col must be provided.")
        x_cols = [age_col]
    elif isinstance(x_cols, str):
        x_cols = [x_cols]
    else:
        x_cols = list(x_cols)

    backend_used = "numpy"
    joblib_verbose = 10 if progress else 0

    pcev_kwargs = {} if pcev_kwargs is None else dict(pcev_kwargs)
    diag_col = "Diagnosis" if "Diagnosis" in df.columns else None
    sex_col = "Sex" if "Sex" in df.columns else None

    base_context = _build_cv_context(
        df,
        feature_combo=feature_combo,
        x_cols=x_cols,
        id_col=id_col,
        diag_col=diag_col,
        sex_col=sex_col,
        strat_labels=strat_labels,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
        confounder_categorical=confounder_categorical,
        confounder_numeric=confounder_numeric,
        conf_matrix=conf_matrix,
    )

    def _run_cv_with_context(cv_context: CvContext) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if joblib_verbose > 0:
            fold_df_local, score_df_local = _run_repeated_cv_generic(
                None,
                feature_combo=feature_combo,
                x_cols=x_cols,
                id_col=id_col,
                covariate_group=covariate_group,
                diag_col=diag_col,
                sex_col=sex_col,
                n_repeats=n_repeats,
                n_splits=n_splits,
                seed=seed,
                n_jobs=cv_n_jobs,
                pcev_kwargs=pcev_kwargs,
                confounder_categorical=confounder_categorical,
                confounder_numeric=confounder_numeric,
                splits=None,
                conf_matrix=None,
                legacy_scalar_alias=len(x_cols) == 1,
                context=cv_context,
                joblib_verbose=joblib_verbose,
            )
        else:
            buffer = io.StringIO()
            with redirect_stdout(buffer):
                fold_df_local, score_df_local = _run_repeated_cv_generic(
                    None,
                    feature_combo=feature_combo,
                    x_cols=x_cols,
                    id_col=id_col,
                    covariate_group=covariate_group,
                    diag_col=diag_col,
                    sex_col=sex_col,
                    n_repeats=n_repeats,
                    n_splits=n_splits,
                    seed=seed,
                    n_jobs=cv_n_jobs,
                    pcev_kwargs=pcev_kwargs,
                    confounder_categorical=confounder_categorical,
                    confounder_numeric=confounder_numeric,
                    splits=None,
                    conf_matrix=None,
                    legacy_scalar_alias=len(x_cols) == 1,
                    context=cv_context,
                    joblib_verbose=joblib_verbose,
                )
        return fold_df_local, score_df_local

    def _compute_repeat_h2_from_scores(score_df_local: pd.DataFrame) -> np.ndarray:
        values = np.full(n_repeats, np.nan, dtype=float)
        if score_df_local.empty:
            return values
        subset = score_df_local.loc[score_df_local["combo_key"] == feature_combo.key]
        if covariate_group is not None and "covariate_group" in subset.columns:
            subset = subset.loc[subset["covariate_group"] == covariate_group]
        if subset.empty:
            return values
        for repeat_idx, group in subset.groupby("repeat"):
            try:
                val = _compute_h2_from_scores(group)
            except Exception:
                val = float("nan")
            if 0 <= repeat_idx < values.size:
                values[int(repeat_idx)] = float(val)
        return values

    _, observed_score_df = _run_cv_with_context(base_context)
    observed_repeats = _compute_repeat_h2_from_scores(observed_score_df)
    observed_mean = float(np.nanmean(observed_repeats)) if observed_repeats.size else float("nan")

    observed_vip_repeats = _compute_repeat_vip_from_scores(observed_score_df, feature_combo, covariate_group, n_repeats)
    observed_vip_mean = np.nanmean(observed_vip_repeats, axis=0) if observed_vip_repeats.size else np.full(len(feature_combo.columns), np.nan)
    observed_vip_std = np.nanstd(observed_vip_repeats, axis=0, ddof=1) if observed_vip_repeats.size else np.full(len(feature_combo.columns), np.nan)

    rows: List[Dict[str, object]] = []
    for repeat_idx, value in enumerate(observed_repeats):
        rows.append(
            {
                "type": "observed",
                "stat": "repeat",
                "repeat": int(repeat_idx),
                "fold": None,
                "perm_id": -1,
                "h2": float(value),
            }
        )
    rows.append(
        {
            "type": "observed",
            "stat": "mean",
            "repeat": None,
            "fold": None,
            "perm_id": -1,
            "h2": observed_mean,
        }
    )

    seed_sequence = np.random.SeedSequence(seed + 50_000)
    perm_seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(n_perm)]

    def _run_single_permutation(perm_id: int, seed_val: int) -> Tuple[int, np.ndarray, float, np.ndarray]:
        rng = np.random.default_rng(seed_val)
        X_perm = _permute_design_matrix_within_strata(base_context.X, base_context.strata, rng)
        perm_context = base_context.with_exposure(X_perm)
        _, perm_score_df = _run_cv_with_context(perm_context)
        rep_vals = _compute_repeat_h2_from_scores(perm_score_df)
        perm_mean = float(np.nanmean(rep_vals)) if rep_vals.size else float("nan")
        perm_vip_repeats = _compute_repeat_vip_from_scores(perm_score_df, feature_combo, covariate_group, n_repeats)
        perm_vip_mean = np.nanmean(perm_vip_repeats, axis=0) if perm_vip_repeats.size else np.full(len(feature_combo.columns), np.nan)
        return perm_id, rep_vals, perm_mean, perm_vip_mean

    iterator = list(zip(range(n_perm), perm_seeds))
    perm_results: List[Tuple[int, np.ndarray, float, np.ndarray]]
    if perm_n_jobs == 1:
        if progress:
            try:
                from tqdm.auto import tqdm  # type: ignore  # pragma: no cover
                iterator_iter = tqdm(iterator, total=n_perm, desc="Permutations")
            except Exception:  # pragma: no cover
                iterator_iter = iterator
            perm_results = [_run_single_permutation(pid, seed_val) for pid, seed_val in iterator_iter]
        else:
            perm_results = [_run_single_permutation(pid, seed_val) for pid, seed_val in iterator]
    else:
        tracked_runner = _run_single_permutation
        if progress:
            total = len(iterator)
            counter = {"done": 0}
            counter_lock = threading.Lock()
            eff_workers = max(effective_n_jobs(perm_n_jobs), 1)
            target_updates = max(eff_workers * 4, 20)
            progress_every = max(1, total // target_updates) if total else 1

            def _tracked_run(perm_id: int, seed_val: int) -> Tuple[int, np.ndarray, float, np.ndarray]:
                result = _run_single_permutation(perm_id, seed_val)
                with counter_lock:
                    counter["done"] += 1
                    done = counter["done"]
                    if done == total or done % progress_every == 0:
                        timestamp = time.strftime("%H:%M:%S")
                        print(f"[{timestamp}] Completed {done:,}/{total:,} permutations")
                return result

            tracked_runner = _tracked_run

        parallel_kwargs = {
            "n_jobs": perm_n_jobs,
            "prefer": "threads",
            "require": "sharedmem",
            "verbose": joblib_verbose,
        }
        if progress:
            parallel_kwargs["batch_size"] = 1
        perm_results = Parallel(**parallel_kwargs)(
            delayed(tracked_runner)(pid, seed_val) for pid, seed_val in iterator
        )

    n_features = len(feature_combo.columns)
    observed_vip_abs = np.abs(observed_vip_mean)
    observed_vip_valid = np.isfinite(observed_vip_abs)

    perm_means: List[float] = []
    vip_total = np.zeros(n_features, dtype=np.int64)
    vip_exceed = np.zeros(n_features, dtype=np.int64)
    dropped = 0
    for perm_id, rep_vals, perm_mean, perm_vip_mean in perm_results:
        if not np.isfinite(perm_mean):
            dropped += 1
            continue
        perm_means.append(float(perm_mean))
        perm_vip_mean_arr = np.asarray(perm_vip_mean, dtype=float)
        if perm_vip_mean_arr.size == 0:
            continue
        perm_vip_abs = np.abs(perm_vip_mean_arr)
        finite_mask = np.isfinite(perm_vip_abs) & observed_vip_valid
        if np.any(finite_mask):
            vip_total[finite_mask] += 1
            compare = perm_vip_abs[finite_mask] >= observed_vip_abs[finite_mask]
            vip_exceed[finite_mask] += compare.astype(np.int64)

    if dropped and progress:
        warnings.warn(f"Dropped {dropped} permutations due to numerical issues.", RuntimeWarning)

    perm_means_arr = np.asarray(perm_means, dtype=float)
    if perm_means_arr.size:
        greater = np.sum(perm_means_arr > observed_mean)
        equal = np.sum(perm_means_arr == observed_mean)
        total = float(perm_means_arr.size)
        p_value_mid = (greater + 0.5 * equal) / total
        p_value_addone = (greater + equal + 1.0) / (total + 1.0)
    else:
        p_value_mid = float("nan")
        p_value_addone = float("nan")

    perm_vip_matrix = np.empty((0, n_features))
    vip_p_values = np.full(n_features, np.nan, dtype=float)
    valid_counts = vip_total > 0
    if np.any(valid_counts):
        vip_p_values[valid_counts] = (vip_exceed[valid_counts] + 1.0) / (vip_total[valid_counts] + 1.0)

    results_df = pd.DataFrame(rows, columns=["type", "stat", "repeat", "fold", "perm_id", "h2"])

    return {
        "observed_repeats": observed_repeats,
        "observed_mean": observed_mean,
        "observed_vip_repeats": observed_vip_repeats,
        "observed_vip_mean": observed_vip_mean,
        "observed_vip_std": observed_vip_std,
        "perm_means": perm_means_arr,
        "perm_vip_means": perm_vip_matrix,
        "vip_p_values": vip_p_values,
        "feature_names": feature_combo.columns,
        "p_value_addone": p_value_addone,
        "p_value_mid": p_value_mid,
        "results": results_df,
        "backend_used": backend_used,
        "dropped_permutations": int(dropped),
    }

def run_repeated_cv_covariates(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    covariate_group: str,
    covariate_cols: Sequence[str],
    id_col: str,
    diag_col: str = "Diagnosis",
    sex_col: str = "Sex",
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
    confounder_categorical: Sequence[str] = CONFOUNDERS_CATEGORICAL,
    confounder_numeric: Sequence[str] = ("Age", "gof_corr"),
    splits: Optional[List[List[Tuple[np.ndarray, np.ndarray]]]] = None,
    conf_matrix: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_repeats <= 0 or n_splits <= 1:
        raise ValueError("n_repeats must be > 0 and n_splits must be > 1")

    covariate_cols = list(covariate_cols)

    fold_df, score_df = _run_repeated_cv_generic(
        df,
        feature_combo=feature_combo,
        x_cols=covariate_cols,
        id_col=id_col,
        covariate_group=covariate_group,
        diag_col=diag_col,
        sex_col=sex_col,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
        n_jobs=n_jobs,
        pcev_kwargs=pcev_kwargs,
        confounder_categorical=confounder_categorical,
        confounder_numeric=confounder_numeric,
        splits=splits,
        conf_matrix=conf_matrix,
        legacy_scalar_alias=False,
    )
    return fold_df, score_df


def summarize_covariate_folds(fold_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame(
            columns=[
                "covariate_group",
                "combo_key",
                "combo_label",
                "repeat",
                "h2_train_mean",
                "h2_test_mean",
                "r2_test_mean",
                "h2_out",
            ]
        )

    summary = (
        fold_df.groupby(["covariate_group", "combo_key", "combo_label", "repeat"], as_index=False)
        .agg(
            h2_train_mean=("h2_train", "mean"),
            h2_test_mean=("h2_test", "mean"),
            r2_test_mean=("r2_test", "mean"),
        )
    )

    if not score_df.empty:
        h2_vals = score_df.groupby(
            ["covariate_group", "combo_key", "combo_label", "repeat"],
            group_keys=False,
        ).apply(_compute_h2_from_scores, include_groups=False)

        if isinstance(h2_vals, pd.Series):
            h2_vals = h2_vals.to_frame(name="h2_out").reset_index()
        else:
            h2_vals = h2_vals.reset_index().rename(columns={0: "h2_out"})

        summary = summary.merge(
            h2_vals,
            on=["covariate_group", "combo_key", "combo_label", "repeat"],
            how="left",
        )
    else:
        summary["h2_out"] = np.nan

    return summary


def summarize_repeats(fold_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame(columns=[
            "combo_key",
            "combo_label",
            "repeat",
            "h2_train_mean",
            "h2_test_mean",
            "r2_test_mean",
            "r2_out_mean",
        ])

    fold_summary = (
        fold_df.groupby(["combo_key", "combo_label", "repeat"], as_index=False)
        .agg(
            h2_train_mean=("h2_train", "mean"),
            h2_test_mean=("h2_test", "mean"),
            r2_test_mean=("r2_test", "mean"),
        )
    )

    if score_df.empty:
        fold_summary["r2_out_mean"] = np.nan
        fold_summary["h2_out"] = np.nan
        return fold_summary

    residual_col = "x_residualized" if "x_residualized" in score_df.columns else None
    if residual_col is None and "age_residualized" in score_df.columns:
        residual_col = "age_residualized"
    if residual_col is not None:
        if score_df[residual_col].dropna().empty:
            residual_col = None

    if residual_col is None:
        fold_summary["r2_out_mean"] = np.nan
        fold_summary["h2_out"] = np.nan
        return fold_summary

    def _compute_r2(group: pd.DataFrame) -> float:
        if group.shape[0] <= 2:
            return float("nan")
        scores = group["score"].to_numpy(float)
        x_res = group[residual_col].to_numpy(float)
        if np.allclose(scores, scores[0]) or np.allclose(x_res, x_res[0]):
            return float("nan")
        corr = np.corrcoef(scores, x_res)[0, 1]
        corr = float(np.clip(corr, -1.0, 1.0))
        if not np.isfinite(corr):
            return float("nan")
        return corr ** 2

    r2_vals = (
        score_df.groupby(["combo_key", "combo_label", "repeat"], group_keys=False)[["score", residual_col]]
        .apply(_compute_r2, include_groups=False)
        .reset_index(name="r2_out_mean")
    )

    merged = fold_summary.merge(r2_vals, on=["combo_key", "combo_label", "repeat"], how="left")

    h2_vals = (
        score_df.groupby(["combo_key", "combo_label", "repeat"], group_keys=False)
        .apply(_compute_h2_from_scores, include_groups=False)
        .reset_index(name="h2_out")
    )

    merged = merged.merge(h2_vals, on=["combo_key", "combo_label", "repeat"], how="left")
    return merged


def _evaluate_holdout_country(
    *,
    combo: FeatureCombo,
    Y_train: np.ndarray,
    X_train: np.ndarray,
    C_train: np.ndarray,
    Y_test: np.ndarray,
    X_test: np.ndarray,
    C_test: np.ndarray,
    subject_ids_test: np.ndarray,
    heldout_country: object,
    store_scalar_x: bool,
    pcev_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[HoldoutMetrics, List[HoldoutScoreRecord]]:
    """Fit on the training set and score the held-out country."""
    if pcev_kwargs is None:
        pcev_kwargs = {}

    estimator = PCEV(**pcev_kwargs)
    estimator.fit(Y_train, X_train, C=C_train)

    if getattr(estimator, "explained_variance_ratio_", None) is not None and estimator.explained_variance_ratio_.size:
        h2_train_full = float(estimator.explained_variance_ratio_[0])
    else:
        h2_train_full = float("nan")

    if Y_test.size == 0:
        metrics = HoldoutMetrics(
            combo_key=combo.key,
            combo_label=combo.label,
            heldout_country=heldout_country,
            n_train=int(Y_train.shape[0]),
            n_test=0,
            h2_train_full=h2_train_full,
            r2_out_country=float("nan"),
        )
        return metrics, []

    scores_test = estimator.project_scores(Y_test, C=C_test)[:, 0]
    res_test = estimator.residualize(Y=Y_test, x=X_test, C=C_test)
    Y_test_res = np.asarray(res_test.get("Y", np.empty((0, 0))), dtype=float)
    X_test_res = np.asarray(res_test.get("X", np.empty((0, 0))), dtype=float)
    if X_test_res.ndim == 1:
        X_test_res = X_test_res[:, None]
    if Y_test_res.ndim == 1:
        Y_test_res = Y_test_res[:, None]

    w = estimator.components_[0] if getattr(estimator, "components_", None) is not None else np.empty(0, dtype=float)
    h2_country = _compute_out_of_sample_h2(Y_test_res, X_test_res, w, rcond=getattr(estimator, "rcond", 1e-12))
    r2_country = h2_country if (np.isfinite(h2_country) and X_test_res.shape[1] == 1) else float("nan")

    score_records: List[HoldoutScoreRecord] = []
    for row_idx, (sid, score) in enumerate(zip(subject_ids_test, scores_test)):
        raw_vec = X_test[row_idx]
        res_vec = X_test_res[row_idx] if X_test_res.shape[0] > row_idx else np.full((X_test.shape[1],), np.nan)

        if store_scalar_x:
            raw_val = raw_vec[0] if raw_vec.size else np.nan
            res_val = res_vec[0] if res_vec.size else np.nan
            x_value_vec = (float(raw_val),) if np.isfinite(raw_val) else None
            x_residualized_vec = (float(res_val),) if np.isfinite(res_val) else None
        else:
            x_value_vec = tuple(float(v) for v in raw_vec.tolist()) if np.all(np.isfinite(raw_vec)) else None
            x_residualized_vec = tuple(float(v) for v in res_vec.tolist()) if np.all(np.isfinite(res_vec)) else None

        score_records.append(
            HoldoutScoreRecord(
                combo_key=combo.key,
                combo_label=combo.label,
                heldout_country=heldout_country,
                subject_id=sid,
                score=float(score),
                x_value_vec=x_value_vec,
                x_residualized_vec=x_residualized_vec,
            )
        )

    metrics = HoldoutMetrics(
        combo_key=combo.key,
        combo_label=combo.label,
        heldout_country=heldout_country,
        n_train=int(Y_train.shape[0]),
        n_test=int(Y_test.shape[0]),
        h2_train_full=h2_train_full,
        r2_out_country=r2_country,
    )

    return metrics, score_records


def evaluate_combinations(
    df: pd.DataFrame,
    combos: Sequence[FeatureCombo],
    *,
    age_col: Optional[str] = None,
    x_cols: Optional[Sequence[str] | str] = None,
    id_col: str,
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
    confounder_categorical: Sequence[str] = CONFOUNDERS_CATEGORICAL,
    confounder_numeric: Sequence[str] = CONFOUNDERS_NUMERIC,
) -> Dict[str, Dict[str, object]]:
    if x_cols is None:
        if age_col is None:
            raise ValueError("Either x_cols or age_col must be provided.")
        x_cols = [age_col]
    elif isinstance(x_cols, str):
        x_cols = [x_cols]

    results: Dict[str, Dict[str, object]] = {}
    diag_col = "Diagnosis" if "Diagnosis" in df.columns else None
    sex_col = "Sex" if "Sex" in df.columns else None
    strata = make_strata(
        df,
        diag_col=diag_col or "Diagnosis",
        sex_col=sex_col or "Sex",
    )
    splits = _make_repeated_splits(
        strata,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
    )
    conf_matrix, _ = _fit_confounders_matrix(
        df,
        categorical=confounder_categorical,
        numeric=confounder_numeric,
    )
    for combo in combos:
        fold_df, score_df = _run_repeated_cv_generic(
            df,
            feature_combo=combo,
            x_cols=x_cols,
            id_col=id_col,
            covariate_group=None,
            diag_col=diag_col,
            sex_col=sex_col,
            n_repeats=n_repeats,
            n_splits=n_splits,
            seed=seed,
            n_jobs=n_jobs,
            pcev_kwargs=pcev_kwargs,
            splits=splits,
            conf_matrix=conf_matrix,
            confounder_categorical=confounder_categorical,
            confounder_numeric=confounder_numeric,
            legacy_scalar_alias=len(x_cols) == 1,
        )
        summary_df = summarize_repeats(fold_df, score_df)
        results[combo.key] = {
            "combo": combo,
            "fold_metrics": fold_df,
            "scores": score_df,
            "summary": summary_df,
        }
    return results


def run_leave_one_country_out(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    age_col: Optional[str] = None,
    x_cols: Optional[Sequence[str] | str] = None,
    id_col: str,
    country_col: str = "Country",
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, pd.DataFrame]:
    """Run inner repeated CV while leaving each country out as an outer test set."""
    if x_cols is None:
        if age_col is None:
            raise ValueError("Either x_cols or age_col must be provided.")
        x_cols = [age_col]
    elif isinstance(x_cols, str):
        x_cols = [x_cols]
    x_cols = list(x_cols)
    store_scalar_x = len(x_cols) == 1

    countries = (
        df[country_col]
        .dropna()
        .astype(str)
        .sort_values()
        .unique()
    )

    inner_fold_frames: List[pd.DataFrame] = []
    inner_score_frames: List[pd.DataFrame] = []
    inner_summary_frames: List[pd.DataFrame] = []
    holdout_metric_records: List[HoldoutMetrics] = []
    holdout_score_frames: List[pd.DataFrame] = []

    for country in countries:
        start_time = time.time()
        test_mask = df[country_col].astype(str) == country
        train_df = df.loc[~test_mask].reset_index(drop=True)
        test_df = df.loc[test_mask].reset_index(drop=True)

        if train_df.empty or test_df.empty:
            continue

        train_conf_matrix, train_transformer = _fit_confounders_matrix(train_df)
        train_splits = _make_repeated_splits(
            make_strata(train_df),
            n_repeats=n_repeats,
            n_splits=n_splits,
            seed=seed,
        )

        fold_df, score_df = run_repeated_cv(
            train_df,
            feature_combo=feature_combo,
            x_cols=x_cols,
            id_col=id_col,
            n_repeats=n_repeats,
            n_splits=n_splits,
            seed=seed,
            n_jobs=n_jobs,
            pcev_kwargs=pcev_kwargs,
            splits=train_splits,
            conf_matrix=train_conf_matrix,
        )

        summary_df = summarize_repeats(fold_df, score_df)

        fold_df_country = fold_df.copy()
        fold_df_country["heldout_country"] = country
        inner_fold_frames.append(fold_df_country)

        score_df_country = score_df.copy()
        score_df_country["heldout_country"] = country
        inner_score_frames.append(score_df_country)

        summary_df_country = summary_df.copy()
        summary_df_country["heldout_country"] = country
        inner_summary_frames.append(summary_df_country)

        feature_cols = list(feature_combo.columns)
        Y_train = train_df[feature_cols].to_numpy(float)
        X_train = train_df[x_cols].to_numpy(float)
        Y_test = test_df[feature_cols].to_numpy(float)
        X_test = test_df[x_cols].to_numpy(float)
        confounder_cols = list(CONFOUNDERS_CATEGORICAL) + list(CONFOUNDERS_NUMERIC)
        C_test = train_transformer.transform(test_df[confounder_cols]) if test_df.shape[0] else np.empty((0, train_conf_matrix.shape[1]))

        holdout_metrics, holdout_scores = _evaluate_holdout_country(
            combo=feature_combo,
            Y_train=Y_train,
            X_train=X_train,
            C_train=train_conf_matrix,
            Y_test=Y_test,
            X_test=X_test,
            C_test=np.asarray(C_test, dtype=float),
            subject_ids_test=test_df[id_col].to_numpy(),
            heldout_country=country,
            store_scalar_x=store_scalar_x,
            pcev_kwargs=pcev_kwargs,
        )
        holdout_metric_records.append(holdout_metrics)

        elapsed_time = time.time() - start_time
        print(f"Country {country} done in {elapsed_time:.2f} seconds")

        if holdout_scores:
            holdout_score_frames.append(
                pd.DataFrame([s.to_dict(legacy_scalar_alias=store_scalar_x) for s in holdout_scores])
            )

    inner_fold_df = (
        pd.concat(inner_fold_frames, ignore_index=True)
        if inner_fold_frames
        else pd.DataFrame()
    )
    inner_score_df = (
        pd.concat(inner_score_frames, ignore_index=True)
        if inner_score_frames
        else pd.DataFrame()
    )
    inner_summary_df = (
        pd.concat(inner_summary_frames, ignore_index=True)
        if inner_summary_frames
        else pd.DataFrame()
    )
    holdout_metrics_df = (
        pd.DataFrame([m.__dict__ for m in holdout_metric_records])
        if holdout_metric_records
        else pd.DataFrame(
            columns=[
                "combo_key",
                "combo_label",
                "heldout_country",
                "n_train",
                "n_test",
                "h2_train_full",
                "r2_out_country",
            ]
        )
    )
    holdout_scores_df = (
        pd.concat(holdout_score_frames, ignore_index=True)
        if holdout_score_frames
        else pd.DataFrame()
    )

    return {
        "inner_fold_metrics": inner_fold_df,
        "inner_scores": inner_score_df,
        "inner_summary": inner_summary_df,
        "holdout_metrics": holdout_metrics_df,
        "holdout_scores": holdout_scores_df,
    }


def evaluate_leave_one_country_out(
    df: pd.DataFrame,
    combos: Sequence[FeatureCombo],
    *,
    age_col: str,
    id_col: str,
    country_col: str = "Country",
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Evaluate every feature combo with leave-one-country-out nested CV."""
    results: Dict[str, Dict[str, pd.DataFrame]] = {}
    for combo in combos:
        start_time = time.time()

        results_combo = run_leave_one_country_out(
            df,
            feature_combo=combo,
            age_col=age_col,
            id_col=id_col,
            country_col=country_col,
            n_repeats=n_repeats,
            n_splits=n_splits,
            seed=seed,
            n_jobs=n_jobs,
            pcev_kwargs=pcev_kwargs,
        )
        results[combo.key] = {"combo": combo, **results_combo}
        elapsed_time = time.time() - start_time
        print(f"Combo {combo.key} done in {elapsed_time:.2f} seconds")
    return results


def build_metric_matrices(
    summary_df: pd.DataFrame,
    metric: str,
    *,
    feature_order: Sequence[str] = FEATURE_PREFIXES,
) -> pd.DataFrame:
    size = len(feature_order)
    matrix = pd.DataFrame(np.nan, index=feature_order, columns=feature_order)
    if summary_df.empty:
        return matrix

    combo_means = (
        summary_df.groupby(["combo_key", "combo_label"], as_index=False)[metric].mean()
    )
    lookup = {
        tuple(sorted(key.split("__"))): val
        for key, val in zip(combo_means["combo_key"], combo_means[metric])
    }

    for i, g1 in enumerate(feature_order):
        for j, g2 in enumerate(feature_order):
            if i == j:
                key = (g1,)
            else:
                key = tuple(sorted((g1, g2)))
            val = lookup.get(key)
            matrix.iloc[i, j] = val
    return matrix


# =============================================================================
