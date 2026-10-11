"""Shallow heads fitted on B_j: concatenation LR, FBCSP-only LR and class-conditional GMM."""
from __future__ import annotations

import common  # noqa: F401  (adds legacy package to sys.path)

import numpy as np
from scipy.special import logsumexp
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.mixture import GaussianMixture
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def lr_head(F: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(F, y)


class GMMHead:
    """Scaler -> PCA(d) -> one diagonal Gaussian per class; posterior with class priors from B."""

    def __init__(self, d_max: int = 8, n_components: int = 1, reg_covar: float = 1e-3, seed: int = 0):
        self.d_max, self.M, self.reg, self.seed = d_max, n_components, reg_covar, seed

    def fit(self, Z: np.ndarray, y: np.ndarray) -> "GMMHead":
        d = int(min(self.d_max, Z.shape[1], len(y) - 1))
        self.sc = StandardScaler().fit(Z)
        self.pca = PCA(n_components=d, random_state=self.seed).fit(self.sc.transform(Z))
        U = self.pca.transform(self.sc.transform(Z))
        self.classes = np.unique(y)
        self.log_prior = np.log(np.array([(y == c).mean() for c in self.classes]))
        self.gmms = [GaussianMixture(self.M, covariance_type="diag", reg_covar=self.reg,
                                     random_state=self.seed).fit(U[y == c]) for c in self.classes]
        self.dim = d
        return self

    def predict_proba(self, Z: np.ndarray) -> np.ndarray:
        U = self.pca.transform(self.sc.transform(Z))
        ll = np.stack([g.score_samples(U) for g in self.gmms], 1) + self.log_prior
        return np.exp(ll - logsumexp(ll, axis=1, keepdims=True))
