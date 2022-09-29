import torch
import joblib
import numpy as np
from tqdm import tqdm
from sklearn.ensemble import IsolationForest
from wanda import config
from wanda.model.cnn_hscore import HSCnnDataPreprocessor


class IsoForestModel:
    def __init__(self) -> None:
        self.model_name = "Isolation Forest"
        self.hs_cnn_preprocessor = HSCnnDataPreprocessor()
        self.iso_forest_clf = None
        self.iso_forest_model_path = f"{config.BASE_PATH}/models/IsoForest.pkl"

    def fit(self, data_loader):
        preprocessed_data, _ = self.hs_cnn_preprocessor.get_preprocess_data(data_loader)
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        self.iso_forest_clf = IsolationForest(
            random_state=42, n_estimators=200, max_features=0.5, n_jobs=-1
        ).fit(preprocessed_data)

    def predict(self, X):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if self.iso_forest_clf is None:
            self.load_model()
        y_preds = []
        n = X.shape[0]
        chunk_size = 1000
        if n > 1000:
            for i in tqdm(range(0, n, chunk_size)):
                chunk = X[i : i + chunk_size]
                y_preds.extend(self.iso_forest_clf.predict(chunk))
        else:
            y_preds = self.iso_forest_clf.predict(X)
        y_preds = np.array(y_preds).flatten()
        return y_preds

    def save_model(self):
        if self.iso_forest_clf is not None:
            joblib.dump(self.iso_forest_clf, self.iso_forest_model_path)
            print(f"Isolation Forest Model saved at: {self.iso_forest_model_path}")

    def load_model(self):
        self.iso_forest_clf = joblib.load(self.iso_forest_model_path)
