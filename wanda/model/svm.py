import torch
import joblib
import numpy as np
from sklearn import linear_model
from sklearn.base import BaseEstimator, TransformerMixin
from tqdm import tqdm
from wanda import config


class SVMModel(BaseEstimator, TransformerMixin):
    model_path = f"{config.BASE_PATH}/models/WandaSVM.pkl"

    def __init__(self, nu, power_t) -> None:
        self.nu = nu
        self.power_t = power_t
        self.model_name = "SVM"
        self.svm_clf = None
        self.model_path = self.model_path

    def fit(self, preprocessed_data, y=None):
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        self.svm_clf = linear_model.SGDOneClassSVM(
            nu=self.nu,
            power_t=self.power_t,
            max_iter=50000,
            verbose=False,
            random_state=42,
            learning_rate="optimal",
        ).fit(preprocessed_data)

    def predict(self, X):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if self.svm_clf is None:
            self.load_model()
        y_preds = []
        n = X.shape[0]
        chunk_size = 1000
        if n > 1000:
            for i in tqdm(range(0, n, chunk_size)):
                chunk = X[i : i + chunk_size]
                y_preds.extend(self.svm_clf.predict(chunk))
        else:
            y_preds = self.svm_clf.predict(X)
        y_preds = np.array(y_preds).flatten()
        return y_preds

    def predict_proba(self, X):
        y = self.predict(X)
        return y

    def decision_function(self, X):
        return self.predict_proba(X)

    def save_model(self):
        if self.svm_clf is not None:
            joblib.dump(self.svm_clf, self.model_path)
            print(f"SVM Model saved at: {self.model_path}")

    def load_model(self):
        self.svm_clf = joblib.load(self.model_path)
