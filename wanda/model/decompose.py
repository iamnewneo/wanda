from umap import UMAP
from sklearn.manifold import Isomap
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import KernelPCA
from wanda import config


class DecomposeData(BaseEstimator, TransformerMixin):
    model_path = f"{config.BASE_PATH}/models/DecomposeModel.pkl"

    def __init__(self) -> None:
        # self.clf = UMAP(
        #     random_state=42,
        #     n_components=100,
        #     n_neighbors=15,
        #     min_dist=0.15,
        #     metric="correlation",
        #     verbose=False,
        #     n_jobs=config.N_JOBS,
        # )
        # self.clf = TruncatedSVD(n_components=100, n_iter=20, random_state=42)
        self.clf = KernelPCA(n_components=100, random_state=42, n_jobs=config.N_JOBS)

    def fit(self, X, y=None):
        self.clf.fit(X)

    def transform(self, X):
        return self.clf.transform(X)

    def fit_transform(self, X, y=None):
        return self.clf.fit_transform(X)
