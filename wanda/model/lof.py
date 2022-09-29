import torch
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.neighbors import LocalOutlierFactor
from wanda import config


class LOFModel:
    def __init__(self) -> None:
        self.model_name = "Local Outlier Factor (LOF)"
        self.lof_clf = None
        self.lof_model_path = f"{config.BASE_PATH}/models/LOF.pkl"

    def fit(self, preprocessed_data):
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        n_jobs = -1
        if config.ENV == "prod":
            n_jobs = 8
        self.lof_clf = LocalOutlierFactor(
            n_neighbors=50, novelty=True, n_jobs=n_jobs
        ).fit(preprocessed_data)

    def predict(self, X):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if self.lof_clf is None:
            self.load_model()
        y_preds = []
        n = X.shape[0]
        chunk_size = 1000
        if n > 1000:
            for i in tqdm(range(0, n, chunk_size)):
                chunk = X[i : i + chunk_size]
                y_preds.extend(self.lof_clf.predict(chunk))
        else:
            y_preds = self.lof_clf.predict(X)
        y_preds = np.array(y_preds).flatten()
        return y_preds

    def save_model(self):
        if self.lof_clf is not None:
            joblib.dump(self.lof_clf, self.lof_model_path)
            print(f"LOF Model saved at: {self.lof_model_path}")

    def load_model(self):
        self.lof_clf = joblib.load(self.lof_model_path)
