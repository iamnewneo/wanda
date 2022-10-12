import torch
import joblib
import numpy as np
from tqdm import tqdm
from wanda import config
from pyod.models.ecod import ECOD
from pyod.models.deep_svdd import DeepSVDD


class ECODModel:
    def __init__(self, contamination) -> None:
        self.model_name = "ECOD"
        self.ecod_clf = None
        self.ecod_model_path = f"{config.BASE_PATH}/models/ECOD.pkl"
        self.contamination = contamination

    def fit(self, preprocessed_data):
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        n_jobs = 16
        if config.ENV == "dev":
            n_jobs = -1
        self.ecod_clf = ECOD(contamination=self.contamination, n_jobs=n_jobs).fit(
            preprocessed_data
        )

    def predict(self, X):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if self.ecod_clf is None:
            self.load_model()
        y_preds = []
        n = X.shape[0]
        chunk_size = 1000
        if n > 1000:
            for i in tqdm(range(0, n, chunk_size)):
                chunk = X[i : i + chunk_size]
                y_preds.extend(self.ecod_clf.predict(chunk))
        else:
            y_preds = self.ecod_clf.predict(X)
        y_preds = np.array(y_preds).flatten()
        return y_preds

    def save_model(self):
        if self.ecod_clf is not None:
            joblib.dump(self.ecod_clf, self.ecod_model_path)
            print(f"{self.model_name} Model saved at: {self.ecod_model_path}")

    def load_model(self):
        self.ecod_clf = joblib.load(self.ecod_model_path)


class DeepSVDDModel:
    def __init__(self, contamination) -> None:
        self.model_name = "Deep SVDD"
        self.deep_svd_clf = None
        self.deep_svd_model_path = f"{config.BASE_PATH}/models/DeepSVDD.pkl"
        self.contamination = contamination

    def fit(self, preprocessed_data):
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        epochs = 10
        if config.ENV == "DEV":
            epochs = 1
        self.deep_svd_clf = DeepSVDD(
            use_ae=True, contamination=self.contamination, epochs=epochs
        ).fit(preprocessed_data)

    def predict(self, X):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if self.deep_svd_clf is None:
            self.load_model()
        y_preds = []
        n = X.shape[0]
        chunk_size = 1000
        if n > 1000:
            for i in tqdm(range(0, n, chunk_size)):
                chunk = X[i : i + chunk_size]
                y_preds.extend(self.deep_svd_clf.predict(chunk))
        else:
            y_preds = self.deep_svd_clf.predict(X)
        y_preds = np.array(y_preds).flatten()
        return y_preds

    def save_model(self):
        if self.deep_svd_clf is not None:
            joblib.dump(self.deep_svd_clf, self.deep_svd_model_path)
            print(f"{self.model_name} Model saved at: {self.deep_svd_model_path}")

    def load_model(self):
        self.deep_svd_clf = joblib.load(self.deep_svd_model_path)
