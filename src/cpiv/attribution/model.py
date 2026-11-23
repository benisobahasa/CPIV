"""Multi-label causal attribution model with open-set (unknown cause) scoring."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler

from ..config import CAUSES


def _make_base(backend: str, random_state: int, n_estimators: int):
    if backend == "rf":
        return RandomForestClassifier(n_estimators=n_estimators, min_samples_leaf=2, n_jobs=-1,
                                      class_weight="balanced_subsample", random_state=random_state)
    if backend == "hgb":
        return HistGradientBoostingClassifier(max_iter=n_estimators, learning_rate=0.08,
                                              random_state=random_state)
    if backend == "logreg":
        return LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced")
    raise ValueError(f"unknown backend '{backend}' (rf | hgb | logreg)")


class CausalAttributor:
    """Predicts which known causes are present in a (reference, test) pair.

    Outputs one probability per cause (multi-label, so combinations are allowed) and a
    novelty score for open-set detection of manipulations outside the known cause list.
    """

    def __init__(self, backend: str = "rf", causes: list[str] | None = None,
                 n_estimators: int = 300, threshold: float = 0.5, random_state: int = 0):
        self.backend, self.causes = backend, list(causes or CAUSES)
        self.n_estimators, self.threshold, self.random_state = n_estimators, threshold, random_state
        self.feature_names: list[str] = []
        self.scaler = StandardScaler()
        self.clf: OneVsRestClassifier | None = None
        self.cov: LedoitWolf | None = None
        self.novelty_scale = 1.0
        self.baseline_: np.ndarray | None = None     # per-feature median of "no change" rows
        self.baseline_mad_: np.ndarray | None = None

    # ------------------------------------------------------------------ fitting
    def fit(self, X: np.ndarray, Y: np.ndarray, feature_names: list[str],
            X_val: np.ndarray | None = None) -> CausalAttributor:
        X, Y = np.asarray(X, float), np.asarray(Y, int)
        assert Y.shape[1] == len(self.causes), "Y columns must match causes"
        self.feature_names = list(feature_names)
        Xs = self.scaler.fit_transform(X)
        self.clf = OneVsRestClassifier(_make_base(self.backend, self.random_state, self.n_estimators))
        self.clf.fit(Xs, Y)

        identity = Y.sum(1) == 0
        ref = X[identity] if identity.sum() >= 5 else X
        self.baseline_ = np.median(ref, axis=0)
        # floor the robust spread with a fraction of the global std so z-scores stay interpretable
        self.baseline_mad_ = np.maximum(1.4826 * np.median(np.abs(ref - self.baseline_), axis=0),
                                        0.1 * np.std(X, axis=0)) + 1e-9

        self.cov = LedoitWolf().fit(Xs)
        self.calibrate_open_set(X_val if X_val is not None else X)
        return self

    def calibrate_open_set(self, X_known: np.ndarray, quantile: float = 0.99) -> None:
        """Set the novelty scale so that ``quantile`` of known data scores 1.0."""
        d = self._mahalanobis(np.asarray(X_known, float))
        self.novelty_scale = float(np.quantile(d, quantile)) + 1e-9

    # --------------------------------------------------------------- inference
    def _mahalanobis(self, X: np.ndarray) -> np.ndarray:
        Xs = self.scaler.transform(X)
        return np.sqrt(np.maximum(self.cov.mahalanobis(Xs), 0))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(np.asarray(X, float))
        return self.clf.predict_proba(self.scaler.transform(X))

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= self.threshold).astype(int)

    def novelty(self, X: np.ndarray) -> np.ndarray:
        """>1 means the sample lies outside what 99% of known data looks like."""
        return self._mahalanobis(np.atleast_2d(np.asarray(X, float))) / self.novelty_scale

    def uncertainty(self, X: np.ndarray) -> np.ndarray:
        """Mean binary entropy over causes (high = the model is unsure)."""
        p = np.clip(self.predict_proba(X), 1e-6, 1 - 1e-6)
        return np.mean(-(p * np.log2(p) + (1 - p) * np.log2(1 - p)), axis=1)

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: str | Path) -> CausalAttributor:
        return joblib.load(path)
