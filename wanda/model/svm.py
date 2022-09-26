import torch
import joblib
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.linear_model import SGDOneClassSVM
from wanda import config
from wanda.model.cnn_hscore import WandaHSCNN
from wanda.data_loader.data_loader import create_hs_data_loader


class SVMModel:
    def __init__(self) -> None:
        self.cnn_hs = WandaHSCNN()
        self.cnn_hs.load_state_dict(
            torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt")
        )
        self.cnn_hs.eval()

        self.svm_clf = None
        self.svm_model_path = f"{config.BASE_PATH}/models/WandaSVM.pkl"

    def get_preprocess_data(self, data_loader):
        tranformed_images = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                X_1_images = batch["X_1"]
                activation_maps, _ = self.cnn_hs(X_1_images)
                activation_maps = torch.flatten(activation_maps, start_dim=1)
                tranformed_images.append(activation_maps)
                labels.append(batch["label"].numpy())

        flattened_tranformed_images = torch.cat(tranformed_images)
        labels = np.concatenate(labels, axis=0)
        labels = labels.ravel()
        return flattened_tranformed_images, labels

    def fit(self, data_loader):
        preprocessed_data, _ = self.get_preprocess_data(data_loader)
        if torch.is_tensor(preprocessed_data):
            preprocessed_data = preprocessed_data.detach().numpy()
        # self.svm_clf = OneClassSVM(
        #     gamma="auto", degree=5, max_iter=10000, cache_size=2000, verbose=True
        # ).fit(preprocessed_data)
        self.svm_clf = SGDOneClassSVM(
            max_iter=10000, verbose=True, random_state=42, learning_rate="optimal",
        ).fit(preprocessed_data)

    def predict(self, X):
        if torch.is_tensor(X):
            X = X.detach().numpy()
        if self.svm_clf is None:
            self.load_model()
        return self.svm_clf.predict(X)

    def save_model(self):
        if self.svm_clf is not None:
            joblib.dump(self.svm_clf, self.svm_model_path)
            print(f"SVM Model saved at: {self.svm_model_path}")

    def load_model(self):
        self.svm_clf = joblib.load(self.svm_model_path)
