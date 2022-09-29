import torch
import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn import linear_model
from tqdm import tqdm
from wanda import config


class SVMModel:
    def __init__(self) -> None:
        self.model_name = "SVM"
        self.svm_clf = None
        self.svm_model_path = f"{config.BASE_PATH}/models/WandaSVM.pkl"

    def fit(self, preprocessed_data):
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        try:
            self.svm_clf = linear_model.SGDOneClassSVM(
                nu=0.1,
                max_iter=50000,
                verbose=True,
                random_state=42,
                learning_rate="optimal",
            ).fit(preprocessed_data)
        except:
            self.svm_clf = OneClassSVM(
                nu=0.1,
                gamma="auto",
                degree=5,
                max_iter=10000,
                cache_size=2000,
                verbose=True,
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

    def save_model(self):
        if self.svm_clf is not None:
            joblib.dump(self.svm_clf, self.svm_model_path)
            print(f"SVM Model saved at: {self.svm_model_path}")

    def load_model(self):
        self.svm_clf = joblib.load(self.svm_model_path)
