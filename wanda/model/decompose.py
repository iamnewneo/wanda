from umap import UMAP
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA, TruncatedSVD, PCA
from wanda import config


class DecomposeData(BaseEstimator, TransformerMixin):
    model_path = f"{config.BASE_PATH}/models/DecomposeModel.pkl"

    def __init__(self) -> None:
        self.clf = UMAP(
            random_state=42,
            n_components=100,
            n_neighbors=15,
            min_dist=0.15,
            metric="correlation",
            verbose=False,
            n_jobs=config.N_JOBS,
        )
        # self.clf = TruncatedSVD(n_components=100, random_state=42)
        # self.clf = KernelPCA(n_components=100, random_state=42, n_jobs=config.N_JOBS)
        self.clf = PCA(n_components=100, random_state=42)
        self.scaler = StandardScaler()

    def fit(self, X, y=None):
        self.scaler.fit(X)
        X_trans = self.scaler.transform(X)
        self.clf.fit(X_trans)

    def transform(self, X):
        X_trans = self.scaler.transform(X)
        return self.clf.transform(X)

    def fit_transform(self, X, y=None):
        self.scaler.fit(X)
        X_trans = self.scaler.transform(X)
        return self.clf.fit_transform(X_trans)
