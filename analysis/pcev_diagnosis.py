"""Diagnosis-focused PCEV cross-validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple
import threading
import time
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed, effective_n_jobs
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder

try:
    from .pcev_feature_effects import (
        FeatureCombo,
        build_combinations,
        build_metric_matrices,
        make_feature_map,
    )
    from .pcev_sklearn import PCEV, _cov_colwise, _pinv_psd
except ImportError:  # pragma: no cover
    from pcev_feature_effects import (
        FeatureCombo,
        build_combinations,
        build_metric_matrices,
        make_feature_map,
    )
    from pcev_sklearn import PCEV, _cov_colwise, _pinv_psd

__all__ = [
    "CONFOUNDERS_CATEGORICAL",
    "CONFOUNDERS_NUMERIC",
    "DiagnosisFoldResult",
    "DiagnosisScoreRecord",
    "build_combinations",
    "build_metric_matrices",
    "evaluate_combinations",
    "make_confounder_transformer",
    "make_feature_map",
    "make_strata",
    "run_repeated_cv",
    "run_repeated_cv_with_permutations",
    "summarize_repeats",
]


CONFOUNDERS_CATEGORICAL: Tuple[str, ...] = ("Sex", "Country")
CONFOUNDERS_NUMERIC: Tuple[str, ...] = ("Age", "gof_corr")

# Numerical stability helpers for hypothesis-test summaries
_KW_PVALUE_FLOOR: float = 1e-300
_KW_LOGP_CLIP: float = 50.0


@dataclass
class DiagnosisFoldResult:
    combo_key: str
    combo_label: str
    repeat: int
    fold: int
    n_train: int
    n_test: int
    h2_train: float
    h2_test: float
    kw_H: float
    kw_pvalue: float
    kw_logp: float
    epsilon_sq: float
    eta_sq: float


@dataclass
class DiagnosisScoreRecord:
    combo_key: str
    combo_label: str
    repeat: int
    fold: int
    subject_id: object
    diagnosis_label: object
    diagnosis_code: int
    score: float


@dataclass
class DiagnosisCvContext:
    Y: np.ndarray
    X: np.ndarray
    diag_codes: np.ndarray
    diag_labels: np.ndarray
    subject_ids: np.ndarray
    conf_matrix: np.ndarray
    splits: List[List[Tuple[np.ndarray, np.ndarray]]]
    perm_strata: np.ndarray
    n_repeats: int

    def permute_within_strata(
        self,
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Permute diagnosis labels within pre-defined strata (sex-only)."""
        codes = self.diag_codes.copy()
        labels = self.diag_labels.copy()
        order = np.arange(self.diag_codes.size)
        strata = self.perm_strata
        for value in np.unique(strata):
            idx = np.flatnonzero(strata == value)
            if idx.size <= 1:
                continue
            shuffled = idx[rng.permutation(idx.size)]
            codes[idx] = self.diag_codes[shuffled]
            labels[idx] = self.diag_labels[shuffled]
            order[idx] = shuffled
        X_perm = self.X[order]
        return codes, labels, X_perm


def _compute_out_of_sample_h2(
    Y_res: np.ndarray,
    X_res: np.ndarray,
    component: np.ndarray,
    *,
    rcond: float = 1e-12,
) -> float:
    """Compute held-out h² using residualised data."""
    if Y_res.size == 0 or X_res.size == 0:
        return float("nan")
    Y_res = np.asarray(Y_res, dtype=float)
    X_res = np.asarray(X_res, dtype=float)
    if Y_res.ndim == 1:
        Y_res = Y_res[:, None]
    if X_res.ndim == 1:
        X_res = X_res[:, None]
    if Y_res.shape[0] <= 1 or X_res.shape[0] != Y_res.shape[0]:
        return float("nan")

    try:
        Syy = _cov_colwise(Y_res)
        Sxx = _cov_colwise(X_res)
        Sxx_inv = _pinv_psd(Sxx, rcond=rcond)
    except Exception:
        return float("nan")

    Sxy = (Y_res.T @ X_res) / (Y_res.shape[0] - 1)
    Sb = Sxy @ (Sxx_inv @ Sxy.T)

    w = np.asarray(component, dtype=float).ravel()
    if w.size == 0 or Syy.size == 0:
        return float("nan")

    numerator = float(w @ (Sb @ w))
    denominator = float(w @ (Syy @ w))
    if denominator <= 1e-12 or numerator < 0:
        return float("nan")
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def make_confounder_transformer() -> ColumnTransformer:
    encoder = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                list(CONFOUNDERS_CATEGORICAL),
            ),
            ("numeric", "passthrough", list(CONFOUNDERS_NUMERIC)),
        ],
        remainder="drop",
    )
    return encoder


def make_strata(df: pd.DataFrame, diag_col: str = "Diagnosis", sex_col: str = "Sex") -> np.ndarray:
    combined = df[[diag_col, sex_col]].astype(str).agg("__".join, axis=1)
    _, inverse = np.unique(combined.to_numpy(), return_inverse=True)
    return inverse


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
        fold_pairs = [
            (np.asarray(train_idx), np.asarray(test_idx))
            for train_idx, test_idx in cv.split(dummy, strata)
        ]
        splits.append(fold_pairs)
    return splits


def _fit_confounders_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, ColumnTransformer]:
    confounder_cols = list(CONFOUNDERS_CATEGORICAL) + list(CONFOUNDERS_NUMERIC)
    transformer = make_confounder_transformer()
    matrix = transformer.fit_transform(df[confounder_cols])
    return np.asarray(matrix, dtype=float), transformer


def _prep_target(df: pd.DataFrame, diag_col: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    diag_series = df[diag_col].astype("category")
    diag_codes = diag_series.cat.codes.to_numpy()
    diag_labels = diag_series.astype(str).to_numpy()
    one_hot = pd.get_dummies(diag_series, drop_first=False, dtype=float)
    return np.asarray(one_hot, dtype=float), diag_codes, diag_labels, list(one_hot.columns)


def _effect_size_metrics(scores: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    mask = np.isfinite(scores)
    scores = scores[mask]
    labels = labels[mask]
    if scores.size < 2:
        return {
            "kw_H": float("nan"),
            "kw_pvalue": float("nan"),
            "kw_logp": float("nan"),
            "epsilon_sq": float("nan"),
            "eta_sq": float("nan"),
        }

    unique_labels = pd.unique(labels)
    groups = [scores[labels == lab] for lab in unique_labels]
    groups = [g for g in groups if g.size > 0]
    if len(groups) < 2:
        return {
            "kw_H": float("nan"),
            "kw_pvalue": float("nan"),
            "kw_logp": float("nan"),
            "epsilon_sq": float("nan"),
            "eta_sq": float("nan"),
        }

    try:
        kw_stat, kw_pvalue = stats.kruskal(*groups, nan_policy="omit")
    except Exception:
        return {
            "kw_H": float("nan"),
            "kw_pvalue": float("nan"),
            "kw_logp": float("nan"),
            "epsilon_sq": float("nan"),
            "eta_sq": float("nan"),
        }

    if np.isfinite(kw_pvalue) and kw_pvalue > 0:
        kw_logp_raw = -np.log10(max(kw_pvalue, _KW_PVALUE_FLOOR))
        kw_logp = float(np.clip(kw_logp_raw, 0.0, _KW_LOGP_CLIP))
    elif kw_pvalue == 0:
        kw_logp = float(_KW_LOGP_CLIP)
        kw_pvalue = float(_KW_PVALUE_FLOOR)
    else:
        kw_logp = float("nan")

    n = float(sum(g.size for g in groups))
    g = float(len(groups))
    if n > g:
        epsilon_sq = float((kw_stat - g + 1.0) / (n - g))
        epsilon_sq = float(np.clip(epsilon_sq, 0.0, 1.0))
    else:
        epsilon_sq = float("nan")

    overall_mean = float(np.mean(scores))
    ss_total = float(np.sum((scores - overall_mean) ** 2))
    if ss_total <= 1e-12:
        eta_sq = float("nan")
    else:
        ss_between = 0.0
        for grp in groups:
            mean_g = float(np.mean(grp))
            ss_between += grp.size * (mean_g - overall_mean) ** 2
        eta_sq = float(np.clip(ss_between / ss_total, 0.0, 1.0))

    return {
        "kw_H": float(kw_stat),
        "kw_pvalue": float(kw_pvalue),
        "kw_logp": kw_logp,
        "epsilon_sq": epsilon_sq,
        "eta_sq": eta_sq,
    }


def _evaluate_fold(
    *,
    combo: FeatureCombo,
    repeat_idx: int,
    fold_idx: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    Y: np.ndarray,
    X: np.ndarray,
    diag_labels: np.ndarray,
    diag_codes: np.ndarray,
    conf_matrix: np.ndarray,
    subject_ids: np.ndarray,
    pcev_kwargs: Optional[Dict[str, object]] = None,
) -> Tuple[DiagnosisFoldResult, List[DiagnosisScoreRecord]]:
    if pcev_kwargs is None:
        pcev_kwargs = {}

    Y_train = Y[train_idx]
    Y_test = Y[test_idx]
    X_train = X[train_idx]
    X_test = X[test_idx]
    C_train = conf_matrix[train_idx]
    C_test = conf_matrix[test_idx]

    estimator = PCEV(**pcev_kwargs)
    estimator.fit(Y_train, X_train, C=C_train)

    if getattr(estimator, "explained_variance_ratio_", None) is not None and estimator.explained_variance_ratio_.size:
        h2_train = float(estimator.explained_variance_ratio_[0])
    else:
        h2_train = float("nan")

    res_test = estimator.residualize(Y=Y_test, x=X_test, C=C_test)
    Y_test_res = np.asarray(res_test.get("Y", np.empty((0, 0))), dtype=float)
    X_test_res = np.asarray(res_test.get("X", np.empty((0, 0))), dtype=float)
    if Y_test_res.ndim == 1:
        Y_test_res = Y_test_res[:, None]
    if X_test_res.ndim == 1:
        X_test_res = X_test_res[:, None]

    component = getattr(estimator, "components_", None)
    if component is not None and len(component):
        primary_component = component[0]
    else:
        primary_component = np.empty(0, dtype=float)

    h2_test = _compute_out_of_sample_h2(
        Y_test_res,
        X_test_res,
        primary_component,
        rcond=getattr(estimator, "rcond", 1e-12),
    )

    scores_test = estimator.project_scores(Y_test, C=C_test)[:, 0]
    diag_labels_test = diag_labels[test_idx]
    diag_codes_test = diag_codes[test_idx]

    effect_metrics = _effect_size_metrics(scores_test, diag_labels_test)

    fold_result = DiagnosisFoldResult(
        combo_key=combo.key,
        combo_label=combo.label,
        repeat=repeat_idx,
        fold=fold_idx,
        n_train=int(train_idx.size),
        n_test=int(test_idx.size),
        h2_train=h2_train,
        h2_test=h2_test,
        kw_H=effect_metrics["kw_H"],
        kw_pvalue=effect_metrics["kw_pvalue"],
        kw_logp=effect_metrics["kw_logp"],
        epsilon_sq=effect_metrics["epsilon_sq"],
        eta_sq=effect_metrics["eta_sq"],
    )

    records: List[DiagnosisScoreRecord] = []
    scores_iter = zip(
        subject_ids[test_idx],
        diag_labels_test,
        diag_codes_test,
        scores_test,
    )
    for sid, label, code, score in scores_iter:
        records.append(
            DiagnosisScoreRecord(
                combo_key=combo.key,
                combo_label=combo.label,
                repeat=repeat_idx,
                fold=fold_idx,
                subject_id=sid,
                diagnosis_label=label,
                diagnosis_code=int(code),
                score=float(score),
            )
        )

    return fold_result, records


def _build_cv_context(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    diag_col: str,
    id_col: str,
    n_repeats: int,
    n_splits: int,
    seed: int,
    splits: Optional[List[List[Tuple[np.ndarray, np.ndarray]]]] = None,
    conf_matrix: Optional[np.ndarray] = None,
) -> DiagnosisCvContext:
    feature_cols = list(feature_combo.columns)
    Y = df[feature_cols].to_numpy(float)
    subject_ids = df[id_col].to_numpy()

    X, diag_codes, diag_labels, _ = _prep_target(df, diag_col)

    if conf_matrix is None:
        conf_matrix, _ = _fit_confounders_matrix(df)
    else:
        conf_matrix = np.asarray(conf_matrix, dtype=float)
        if conf_matrix.shape[0] != df.shape[0]:
            raise ValueError("conf_matrix must have the same number of rows as df.")

    if splits is None:
        strata = make_strata(df, diag_col=diag_col)
        splits = _make_repeated_splits(
            strata,
            n_repeats=n_repeats,
            n_splits=n_splits,
            seed=seed,
        )
    effective_repeats = len(splits)

    if "Sex" in df.columns:
        sex_series = df["Sex"].astype("category")
        perm_strata = sex_series.cat.codes.to_numpy()
    else:
        perm_strata = np.zeros(df.shape[0], dtype=int)

    return DiagnosisCvContext(
        Y=Y,
        X=X,
        diag_codes=diag_codes,
        diag_labels=diag_labels,
        subject_ids=subject_ids,
        conf_matrix=conf_matrix,
        splits=splits,
        perm_strata=perm_strata,
        n_repeats=effective_repeats,
    )


def _run_repeated_cv_from_context(
    context: DiagnosisCvContext,
    *,
    combo: FeatureCombo,
    n_jobs: int,
    pcev_kwargs: Optional[Dict[str, object]],
    X_override: Optional[np.ndarray] = None,
    diag_codes_override: Optional[np.ndarray] = None,
    diag_labels_override: Optional[np.ndarray] = None,
    joblib_verbose: int = 0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if pcev_kwargs is None:
        pcev_kwargs = {}

    Y = context.Y
    X = np.asarray(X_override if X_override is not None else context.X, dtype=float)
    diag_codes = np.asarray(
        diag_codes_override if diag_codes_override is not None else context.diag_codes,
        dtype=int,
    )
    diag_labels = np.asarray(
        diag_labels_override if diag_labels_override is not None else context.diag_labels,
        dtype=object,
    )
    conf_matrix = context.conf_matrix
    subject_ids = context.subject_ids

    def _worker(args: Tuple[int, List[Tuple[np.ndarray, np.ndarray]]]) -> Tuple[List[DiagnosisFoldResult], List[DiagnosisScoreRecord]]:
        repeat_idx, fold_pairs = args
        repeat_folds: List[DiagnosisFoldResult] = []
        repeat_scores: List[DiagnosisScoreRecord] = []
        for fold_idx, (train_idx, test_idx) in enumerate(fold_pairs):
            fold_res, records = _evaluate_fold(
                combo=combo,
                repeat_idx=repeat_idx,
                fold_idx=fold_idx,
                train_idx=train_idx,
                test_idx=test_idx,
                Y=Y,
                X=X,
                diag_labels=diag_labels,
                diag_codes=diag_codes,
                conf_matrix=conf_matrix,
                subject_ids=subject_ids,
                pcev_kwargs=pcev_kwargs,
            )
            repeat_folds.append(fold_res)
            repeat_scores.extend(records)
        return repeat_folds, repeat_scores

    tasks = [(repeat_idx, fold_pairs) for repeat_idx, fold_pairs in enumerate(context.splits)]

    if n_jobs == 1:
        outputs = [_worker(task) for task in tasks]
    else:
        outputs = Parallel(
            n_jobs=n_jobs,
            prefer="threads",
            require="sharedmem",
            verbose=joblib_verbose,
        )(delayed(_worker)(task) for task in tasks)

    fold_records = [fold for fold_list, _ in outputs for fold in fold_list]
    score_records_nested = [records for _, records in outputs]

    fold_df = pd.DataFrame([r.__dict__ for r in fold_records])
    score_df = pd.DataFrame([s.__dict__ for records in score_records_nested for s in records])
    return fold_df, score_df


def run_repeated_cv(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    diag_col: str,
    id_col: str,
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
    splits: Optional[List[List[Tuple[np.ndarray, np.ndarray]]]] = None,
    conf_matrix: Optional[np.ndarray] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if n_repeats <= 0 or n_splits <= 1:
        raise ValueError("n_repeats must be > 0 and n_splits must be > 1")

    context = _build_cv_context(
        df,
        feature_combo=feature_combo,
        diag_col=diag_col,
        id_col=id_col,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
        splits=splits,
        conf_matrix=conf_matrix,
    )

    fold_df, score_df = _run_repeated_cv_from_context(
        context,
        combo=feature_combo,
        n_jobs=n_jobs,
        pcev_kwargs=pcev_kwargs,
    )
    return fold_df, score_df


def run_repeated_cv_with_permutations(
    df: pd.DataFrame,
    *,
    feature_combo: FeatureCombo,
    diag_col: str,
    id_col: str,
    n_repeats: int = 5,
    n_splits: int = 5,
    n_perm: int = 1000,
    seed: int = 1234,
    n_jobs: int = 1,
    perm_n_jobs: int = 1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
    progress: bool = True,
) -> Dict[str, object]:
    if n_repeats <= 0 or n_splits <= 1:
        raise ValueError("n_repeats must be > 0 and n_splits must be > 1")
    if n_perm <= 0:
        raise ValueError("n_perm must be > 0")
    
    joblib_verbose = 10 if progress else 0

    context = _build_cv_context(
        df,
        feature_combo=feature_combo,
        diag_col=diag_col,
        id_col=id_col,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
    )

    fold_df_obs, score_df_obs = _run_repeated_cv_from_context(
        context,
        combo=feature_combo,
        n_jobs=n_jobs,
        pcev_kwargs=pcev_kwargs,
    )
    summary_obs = summarize_repeats(fold_df_obs, score_df_obs)
    summary_obs = summary_obs.sort_values("repeat")

    def _extract_metric(df_metric: pd.DataFrame, column: str, length: int) -> np.ndarray:
        values = np.full(length, np.nan, dtype=float)
        if column not in df_metric.columns or "repeat" not in df_metric.columns:
            return values
        subset = df_metric.loc[:, ["repeat", column]].dropna(subset=[column])
        for repeat, value in subset.itertuples(index=False):
            if 0 <= repeat < length:
                values[int(repeat)] = float(value)
        return values

    n_repeats_effective = context.n_repeats
    observed_h2_repeats = _extract_metric(summary_obs, "h2_test_mean", n_repeats_effective)
    observed_epsilon_repeats = _extract_metric(summary_obs, "epsilon_sq_out", n_repeats_effective)
    observed_eta_repeats = _extract_metric(summary_obs, "eta_sq_out", n_repeats_effective)

    observed_h2_mean = float(np.nanmean(observed_h2_repeats)) if observed_h2_repeats.size else float("nan")
    observed_epsilon_mean = float(np.nanmean(observed_epsilon_repeats)) if observed_epsilon_repeats.size else float("nan")
    observed_eta_mean = float(np.nanmean(observed_eta_repeats)) if observed_eta_repeats.size else float("nan")

    seed_sequence = np.random.SeedSequence(seed + 71_000)
    perm_seeds = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in seed_sequence.spawn(n_perm)]

    def _run_single_permutation(perm_id: int, seed_val: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
        rng = np.random.default_rng(seed_val)
        diag_codes_perm, diag_labels_perm, X_perm = context.permute_within_strata(rng)
        fold_df_perm, score_df_perm = _run_repeated_cv_from_context(
            context,
            combo=feature_combo,
            n_jobs=n_jobs,
            pcev_kwargs=pcev_kwargs,
            X_override=X_perm,
            diag_codes_override=diag_codes_perm,
            diag_labels_override=diag_labels_perm,
        )
        summary_perm = summarize_repeats(fold_df_perm, score_df_perm).sort_values("repeat")
        h2_vals = _extract_metric(summary_perm, "h2_test_mean", n_repeats_effective)
        epsilon_vals = _extract_metric(summary_perm, "epsilon_sq_out", n_repeats_effective)
        eta_vals = _extract_metric(summary_perm, "eta_sq_out", n_repeats_effective)
        return perm_id, h2_vals, epsilon_vals, eta_vals

    iterator = list(zip(range(n_perm), perm_seeds))
    perm_results: List[Tuple[int, np.ndarray, np.ndarray, np.ndarray]]
    if perm_n_jobs == 1:
        if progress:
            try:
                from tqdm.auto import tqdm  # type: ignore  # pragma: no cover
                iterator_iter = tqdm(iterator, total=n_perm, desc="Permutations")
            except Exception:  # pragma: no cover
                iterator_iter = iterator
            perm_outputs = [_run_single_permutation(pid, seed_val) for pid, seed_val in iterator_iter]
        else:
            perm_outputs = [_run_single_permutation(pid, seed_val) for pid, seed_val in iterator]
    else:
        tracked_runner = _run_single_permutation
        if progress:
            total = len(iterator)
            counter = {"done": 0}
            counter_lock = threading.Lock()
            eff_workers = max(effective_n_jobs(perm_n_jobs), 1)
            target_updates = max(eff_workers * 4, 20)
            progress_every = max(1, total // target_updates) if total else 1

            def _tracked_run(perm_id: int, seed_val: int) -> Tuple[int, np.ndarray, np.ndarray, np.ndarray]:
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
        perm_outputs = Parallel(**parallel_kwargs)(
            delayed(tracked_runner)(pid, seed_val) for pid, seed_val in iterator
        )

    perm_h2_matrix: List[np.ndarray] = []
    perm_epsilon_matrix: List[np.ndarray] = []
    perm_eta_matrix: List[np.ndarray] = []
    dropped = 0

    for _, h2_vals, epsilon_vals, eta_vals in perm_outputs:
        if np.all(np.isnan(h2_vals)) and np.all(np.isnan(epsilon_vals)) and np.all(np.isnan(eta_vals)):
            dropped += 1
            continue
        perm_h2_matrix.append(h2_vals)
        perm_epsilon_matrix.append(epsilon_vals)
        perm_eta_matrix.append(eta_vals)

    if dropped and progress:
        warnings.warn(f"Dropped {dropped} permutations due to numerical issues.", RuntimeWarning)

    perm_h2_array = np.vstack(perm_h2_matrix) if perm_h2_matrix else np.empty((0, n_repeats_effective))
    perm_epsilon_array = np.vstack(perm_epsilon_matrix) if perm_epsilon_matrix else np.empty((0, n_repeats_effective))
    perm_eta_array = np.vstack(perm_eta_matrix) if perm_eta_matrix else np.empty((0, n_repeats_effective))

    def _compute_p_value(observed: float, perm_matrix: np.ndarray) -> float:
        if perm_matrix.size == 0 or not np.isfinite(observed):
            return float("nan")
        perm_means = np.nanmean(perm_matrix, axis=1)
        finite = perm_means[np.isfinite(perm_means)]
        if finite.size == 0:
            return float("nan")
        greater = np.sum(finite >= observed)
        equal = np.sum(finite == observed)
        return (greater + equal + 1.0) / (finite.size + 1.0)

    p_value_h2 = _compute_p_value(observed_h2_mean, perm_h2_array)
    p_value_epsilon = _compute_p_value(observed_epsilon_mean, perm_epsilon_array)
    p_value_eta = _compute_p_value(observed_eta_mean, perm_eta_array)

    return {
        "observed_summary": summary_obs,
        "observed_h2_repeats": observed_h2_repeats,
        "observed_h2_mean": observed_h2_mean,
        "observed_epsilon_repeats": observed_epsilon_repeats,
        "observed_epsilon_mean": observed_epsilon_mean,
        "observed_eta_repeats": observed_eta_repeats,
        "observed_eta_mean": observed_eta_mean,
        "perm_h2_repeats": perm_h2_array,
        "perm_epsilon_repeats": perm_epsilon_array,
        "perm_eta_repeats": perm_eta_array,
        "p_value_h2": p_value_h2,
        "p_value_epsilon": p_value_epsilon,
        "p_value_eta": p_value_eta,
        "dropped_permutations": int(dropped),
        "n_perm": n_perm,
        "backend_used": "numpy",
    }


def _aggregate_scores(score_df: pd.DataFrame) -> pd.DataFrame:
    if score_df.empty:
        return pd.DataFrame(columns=[
            "combo_key",
            "combo_label",
            "repeat",
            "kw_H_out",
            "kw_p_out",
            "kw_logp_out",
            "epsilon_sq_out",
            "eta_sq_out",
            "n_subjects_out",
        ])

    def _summarize(group: pd.DataFrame) -> pd.Series:
        metrics = _effect_size_metrics(group["score"].to_numpy(float), group["diagnosis_label"].to_numpy())
        return pd.Series({
            "kw_H_out": metrics["kw_H"],
            "kw_p_out": metrics["kw_pvalue"],
            "kw_logp_out": metrics["kw_logp"],
            "epsilon_sq_out": metrics["epsilon_sq"],
            "eta_sq_out": metrics["eta_sq"],
            "n_subjects_out": group.shape[0],
        })

    aggregated = (
        score_df.groupby(["combo_key", "combo_label", "repeat"], as_index=False)
        .apply(_summarize, include_groups=False)
        .reset_index(drop=True)
    )
    return aggregated


def summarize_repeats(fold_df: pd.DataFrame, score_df: pd.DataFrame) -> pd.DataFrame:
    if fold_df.empty:
        return pd.DataFrame(columns=[
            "combo_key",
            "combo_label",
            "repeat",
            "h2_train_mean",
            "h2_test_mean",
            "kw_H_mean",
            "kw_p_mean",
            "kw_logp_mean",
            "epsilon_sq_mean",
            "eta_sq_mean",
            "kw_H_out",
            "kw_p_out",
            "kw_logp_out",
            "epsilon_sq_out",
            "eta_sq_out",
            "h2_out",
        ])

    fold_summary = (
        fold_df.groupby(["combo_key", "combo_label", "repeat"], as_index=False)
        .agg(
            h2_train_mean=("h2_train", "mean"),
            h2_test_mean=("h2_test", "mean"),
            kw_H_mean=("kw_H", "mean"),
            kw_p_mean=("kw_pvalue", "mean"),
            kw_logp_mean=("kw_logp", "mean"),
            epsilon_sq_mean=("epsilon_sq", "mean"),
            eta_sq_mean=("eta_sq", "mean"),
        )
    )

    score_summary = _aggregate_scores(score_df)

    merged = fold_summary.merge(
        score_summary,
        on=["combo_key", "combo_label", "repeat"],
        how="left",
    )
    merged["h2_out"] = merged["h2_test_mean"]
    return merged


def evaluate_combinations(
    df: pd.DataFrame,
    combos: Sequence[FeatureCombo],
    *,
    diag_col: str,
    id_col: str,
    n_repeats: int = 5,
    n_splits: int = 5,
    seed: int = 1234,
    n_jobs: int = -1,
    pcev_kwargs: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    results: Dict[str, Dict[str, object]] = {}
    strata = make_strata(df, diag_col=diag_col)
    splits = _make_repeated_splits(
        strata,
        n_repeats=n_repeats,
        n_splits=n_splits,
        seed=seed,
    )
    conf_matrix, _ = _fit_confounders_matrix(df)
    for combo in combos:
        fold_df, score_df = run_repeated_cv(
            df,
            feature_combo=combo,
            diag_col=diag_col,
            id_col=id_col,
            n_repeats=n_repeats,
            n_splits=n_splits,
            seed=seed,
            n_jobs=n_jobs,
            pcev_kwargs=pcev_kwargs,
            splits=splits,
            conf_matrix=conf_matrix,
        )
        summary_df = summarize_repeats(fold_df, score_df)
        results[combo.key] = {
            "combo": combo,
            "fold_metrics": fold_df,
            "scores": score_df,
            "summary": summary_df,
        }
    return results
