"""Out-of-distribution (OOD) detection for electrofacies predictions.

Identifies well-log samples that fall outside the training data distribution,
signalling that model predictions may be unreliable.  Two complementary
strategies are supported:

- **Mahalanobis distance** -- parametric; measures how far a sample lies from
  the training centroid, accounting for feature correlations via the
  covariance matrix.  Fast at inference time and easy to interpret.
- **Isolation Forest** -- non-parametric; detects anomalies through random
  recursive partitioning.  More robust to non-Gaussian feature distributions.

Both methods expose a common ``fit`` / ``score`` / ``predict`` interface so
they can be swapped transparently in the pipeline configuration.
"""

from __future__ import annotations

import logging
from typing import Optional

import joblib
import numpy as np
from scipy.spatial.distance import mahalanobis as _scipy_mahalanobis
from sklearn.ensemble import IsolationForest

logger = logging.getLogger(__name__)


class OODDetector:
    """Out-of-distribution detector with a unified interface.

    Parameters
    ----------
    method : str, optional
        Detection method: ``'mahalanobis'`` (default) or
        ``'isolation_forest'``.
    percentile_threshold : float, optional
        Percentile of the training OOD scores used to set the decision
        threshold.  Samples scoring above this percentile are flagged as
        OOD.  Defaults to 95.

    Attributes
    ----------
    threshold_ : float or None
        Learned decision threshold after :meth:`fit`.
    """

    _VALID_METHODS = ("mahalanobis", "isolation_forest")

    def __init__(
        self,
        method: str = "mahalanobis",
        percentile_threshold: float = 95,
    ) -> None:
        if method not in self._VALID_METHODS:
            raise ValueError(
                f"method must be one of {self._VALID_METHODS}, got '{method}'."
            )
        self.method = method
        self.percentile_threshold = percentile_threshold

        # State set by fit()
        self.threshold_: Optional[float] = None
        self._mean: Optional[np.ndarray] = None
        self._cov_inv: Optional[np.ndarray] = None
        self._iso_forest: Optional[IsolationForest] = None
        self._fitted: bool = False

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, X_train: np.ndarray) -> "OODDetector":
        """Fit the detector on the training feature matrix.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training features.  NaN values are **not** permitted; the
            caller must impute or drop them before calling ``fit``.

        Returns
        -------
        OODDetector
            ``self``, for method chaining.

        Raises
        ------
        ValueError
            If *X_train* contains NaN values or has fewer than 2 samples.
        """
        X = np.asarray(X_train, dtype=np.float64)
        self._validate_input(X, context="fit")

        if self.method == "mahalanobis":
            self._fit_mahalanobis(X)
        else:
            self._fit_isolation_forest(X)

        # Compute threshold from training scores.
        train_scores = self.score(X)
        self.threshold_ = float(np.percentile(train_scores, self.percentile_threshold))

        self._fitted = True
        logger.info(
            "OODDetector fitted (method='%s', threshold=%.4f at %.0fth "
            "percentile, n_train=%d, n_features=%d).",
            self.method,
            self.threshold_,
            self.percentile_threshold,
            X.shape[0],
            X.shape[1],
        )
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def score(self, X: np.ndarray) -> np.ndarray:
        """Compute OOD score for each sample.

        Higher scores indicate greater deviation from the training
        distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of shape (n_samples,)
            Per-sample OOD score.

        Raises
        ------
        RuntimeError
            If the detector has not been fitted.
        """
        X = np.asarray(X, dtype=np.float64)
        self._validate_input(X, context="score")

        if self.method == "mahalanobis":
            return self._score_mahalanobis(X)
        else:
            return self._score_isolation_forest(X)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict whether each sample is out-of-distribution.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Feature matrix.

        Returns
        -------
        np.ndarray of bool, shape (n_samples,)
            ``True`` for samples classified as OOD (score > threshold).

        Raises
        ------
        RuntimeError
            If the detector has not been fitted.
        """
        if self.threshold_ is None:
            raise RuntimeError(
                "OODDetector has not been fitted; call fit() first."
            )
        scores = self.score(X)
        ood_flags = scores > self.threshold_

        n_ood = ood_flags.sum()
        pct = 100.0 * n_ood / len(ood_flags) if len(ood_flags) > 0 else 0.0
        logger.info(
            "OOD prediction: %d / %d flagged (%.1f%%).",
            n_ood,
            len(ood_flags),
            pct,
        )
        return ood_flags

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Serialize the fitted detector to disk using joblib.

        Parameters
        ----------
        path : str
            Destination file path.
        """
        joblib.dump(self, path)
        logger.info("OODDetector saved to %s.", path)

    @classmethod
    def load(cls, path: str) -> "OODDetector":
        """Deserialize a detector from disk.

        Parameters
        ----------
        path : str
            Path to a previously saved detector.

        Returns
        -------
        OODDetector
        """
        detector = joblib.load(path)
        if not isinstance(detector, cls):
            raise TypeError(
                f"Loaded object is {type(detector).__name__}, "
                f"expected OODDetector."
            )
        logger.info("OODDetector loaded from %s.", path)
        return detector

    # ------------------------------------------------------------------
    # Internal: Mahalanobis
    # ------------------------------------------------------------------

    def _fit_mahalanobis(self, X: np.ndarray) -> None:
        """Compute mean, regularised inverse covariance."""
        self._mean = X.mean(axis=0)

        n_features = X.shape[1]

        if n_features == 1:
            # Single feature: covariance is just variance.
            var = np.var(X, ddof=1) if X.shape[0] > 1 else 1.0
            var = max(var, 1e-6)
            self._cov_inv = np.array([[1.0 / var]])
            logger.debug(
                "Single-feature Mahalanobis: var=%.6f.", var
            )
            return

        # Full covariance matrix with regularisation for numerical stability.
        cov = np.cov(X, rowvar=False)

        # First regularisation pass: small diagonal perturbation.
        reg = 1e-6 * np.eye(n_features)
        cov_reg = cov + reg

        try:
            self._cov_inv = np.linalg.inv(cov_reg)
        except np.linalg.LinAlgError:
            # Stronger regularisation if still singular.
            logger.warning(
                "Covariance matrix near-singular; applying stronger "
                "regularisation (1e-3 * I)."
            )
            cov_reg = cov + 1e-3 * np.eye(n_features)
            self._cov_inv = np.linalg.inv(cov_reg)

        logger.debug(
            "Mahalanobis fit: mean shape %s, cov_inv shape %s.",
            self._mean.shape,
            self._cov_inv.shape,
        )

    def _score_mahalanobis(self, X: np.ndarray) -> np.ndarray:
        """Mahalanobis distance from training centroid for each sample."""
        if self._mean is None or self._cov_inv is None:
            raise RuntimeError(
                "Mahalanobis parameters not set; call fit() first."
            )

        scores = np.empty(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            scores[i] = _scipy_mahalanobis(X[i], self._mean, self._cov_inv)
        return scores

    # ------------------------------------------------------------------
    # Internal: Isolation Forest
    # ------------------------------------------------------------------

    def _fit_isolation_forest(self, X: np.ndarray) -> None:
        """Fit an sklearn IsolationForest."""
        self._iso_forest = IsolationForest(
            contamination=0.05,
            random_state=42,
        )
        self._iso_forest.fit(X)
        logger.debug(
            "IsolationForest fitted on %d samples, %d features.",
            X.shape[0],
            X.shape[1],
        )

    def _score_isolation_forest(self, X: np.ndarray) -> np.ndarray:
        """Anomaly score from Isolation Forest.

        sklearn's ``decision_function`` returns values where more negative
        means more anomalous.  We negate so that *higher = more OOD*,
        consistent with the Mahalanobis convention.
        """
        if self._iso_forest is None:
            raise RuntimeError(
                "IsolationForest not fitted; call fit() first."
            )
        # decision_function: lower (more negative) = more anomalous.
        raw_scores = self._iso_forest.decision_function(X)
        return -raw_scores

    # ------------------------------------------------------------------
    # Validation helper
    # ------------------------------------------------------------------

    def _validate_input(self, X: np.ndarray, context: str = "") -> None:
        """Shared input validation for fit / score / predict."""
        if X.ndim == 1:
            raise ValueError(
                f"Input must be 2-D (n_samples, n_features), got 1-D array "
                f"with shape {X.shape}.  Reshape single-feature data with "
                f"X.reshape(-1, 1)."
            )
        if X.ndim != 2:
            raise ValueError(
                f"Input must be 2-D, got shape {X.shape}."
            )
        if np.isnan(X).any():
            raise ValueError(
                f"Input contains NaN values ({context}).  "
                "Please impute or drop NaN rows before calling the "
                "OODDetector."
            )
        if context == "fit" and X.shape[0] < 2:
            raise ValueError(
                f"Need at least 2 samples to fit the detector, got "
                f"{X.shape[0]}."
            )
