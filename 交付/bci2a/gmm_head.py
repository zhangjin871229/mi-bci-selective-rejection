"""在 ViT 特征上拟合每类 GaussianMixture，用于概率分类。"""

from __future__ import annotations

import pickle

from pathlib import Path

import numpy as np

import torch

from sklearn.decomposition import PCA

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from sklearn.mixture import GaussianMixture

from torch.utils.data import DataLoader

from .config import N_CLASSES

from .dataset import BCIC2aTFDataset, DatasetManifest, collate_batch

from .models import ChannelViT

class FeatureReducer:

    """ViT 高维特征 → 低维，再送入 GMM。"""

    METHODS = ("none", "pca", "lda")

    def __init__(self, method: str = "pca", n_components: int = 32, random_state: int = 42):

        if method not in self.METHODS:

            raise ValueError(f"gmm_reduce 应为 {self.METHODS}，收到 {method!r}")

        self.method = method

        self.n_components = n_components

        self.random_state = random_state

        self._model: PCA | LinearDiscriminantAnalysis | None = None

        self.n_components_: int | None = None

    def _effective_dim(self, n_features: int, n_samples: int) -> int:

        if self.method == "none":

            return n_features

        cap = n_features

        if self.method == "lda":

            cap = min(cap, N_CLASSES - 1)

        cap = min(cap, n_samples - 1)

        return max(1, min(self.n_components, cap))

    def fit(self, X: np.ndarray, y: np.ndarray) -> "FeatureReducer":

        if self.method == "none":

            self.n_components_ = X.shape[1]

            return self

        n_comp = self._effective_dim(X.shape[1], X.shape[0])

        self.n_components_ = n_comp

        if self.method == "pca":

            self._model = PCA(n_components=n_comp, random_state=self.random_state)

            self._model.fit(X)

        else:

            self._model = LinearDiscriminantAnalysis(n_components=n_comp)

            self._model.fit(X, y)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:

        if self.method == "none" or self._model is None:

            return X

        return self._model.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:

        self.fit(X, y)

        return self.transform(X)

class PerClassGMM:

    def __init__(

        self,

        n_components: int = 4,

        random_state: int = 42,

        reducer: FeatureReducer | None = None,

        covariance_type: str = "auto",

        reg_covar: float = 1e-4,

    ):

        self.n_components = n_components

        self.random_state = random_state

        self.reducer = reducer

        self.covariance_type = covariance_type

        self.reg_covar = reg_covar

        self.models: list[GaussianMixture | None] = [None] * N_CLASSES

    def _prepare_features(self, features: np.ndarray, labels: np.ndarray | None, *, fit: bool) -> np.ndarray:

        if self.reducer is None:

            return features

        if fit:

            assert labels is not None

            return self.reducer.fit_transform(features, labels)

        return self.reducer.transform(features)

    def _covariance_type_for(self, n_features: int, n_samples: int) -> str:

        if self.covariance_type != "auto":

            return self.covariance_type

        # 小样本高维时用对角协方差更稳

        if n_features > 16 or n_samples < n_features * 3:

            return "diag"

        return "full"

    def fit(self, features: np.ndarray, labels: np.ndarray) -> "PerClassGMM":

        features = self._prepare_features(features, labels, fit=True)

        for c in range(N_CLASSES):

            mask = labels == c

            Xc = features[mask]

            if len(Xc) == 0:

                self.models[c] = None

                continue

            n_comp = min(self.n_components, max(1, len(Xc) // 3))
            cov_type = self._covariance_type_for(features.shape[1], len(Xc))
            fitted = None
            for try_comp in range(n_comp, 0, -1):
                gmm = GaussianMixture(
                    n_components=try_comp,
                    covariance_type=cov_type,
                    random_state=self.random_state,
                    reg_covar=self.reg_covar,
                    max_iter=200,
                )
                try:
                    gmm.fit(Xc)
                    fitted = gmm
                    break
                except ValueError:
                    continue
            self.models[c] = fitted

        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:

        features = self._prepare_features(features, labels=None, fit=False)

        n = len(features)

        log_probs = np.full((n, N_CLASSES), -np.inf, dtype=np.float64)

        for c, gmm in enumerate(self.models):

            if gmm is None:

                continue

            log_probs[:, c] = gmm.score_samples(features)

        log_probs = log_probs - log_probs.max(axis=1, keepdims=True)

        probs = np.exp(log_probs)

        probs /= probs.sum(axis=1, keepdims=True) + 1e-12

        return probs.astype(np.float32)

    def predict(self, features: np.ndarray) -> np.ndarray:

        return self.predict_proba(features).argmax(axis=1)

    def save(self, path: Path) -> None:

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("wb") as f:

            pickle.dump(self, f)

    @classmethod

    def load(cls, path: Path) -> "PerClassGMM":

        with path.open("rb") as f:

            return pickle.load(f)

def make_gmm_head(cfg) -> PerClassGMM:

    reducer = None

    if getattr(cfg, "gmm_reduce", "none") != "none":

        reducer = FeatureReducer(

            method=cfg.gmm_reduce,

            n_components=cfg.gmm_reduce_dim,

            random_state=cfg.seed,

        )

    return PerClassGMM(

        n_components=cfg.gmm_components_per_class,

        random_state=cfg.seed,

        reducer=reducer,

        covariance_type=getattr(cfg, "gmm_covariance", "auto"),

    )

@torch.no_grad()

def extract_features(

    model: ChannelViT,

    manifest: DatasetManifest,

    indices: list[int],

    device: torch.device,

    batch_size: int = 64,

    norm_mu: np.ndarray | None = None,

    norm_std: np.ndarray | None = None,

) -> tuple[np.ndarray, np.ndarray]:

    model.eval()

    ds = BCIC2aTFDataset(manifest, indices, norm_mu=norm_mu, norm_std=norm_std)

    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    feats, labels = [], []

    for x, y, _ in loader:

        x = x.to(device)

        z = model.forward_features(x).cpu().numpy()

        feats.append(z)

        labels.append(y.numpy())

    return np.concatenate(feats, 0), np.concatenate(labels, 0)
