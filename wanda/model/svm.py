from distutils.command.config import config
import torch
from sklearn.externals import joblib
from sklearn.svm import OneClassSVM
from wanda import config
from wanda.model.cnn_hscore import WandaHSCNN
from wanda.data_loader.data_loader import create_hs_data_loader


class SVMModel:
    def __init__(self) -> None:
        self.cnn_hs = WandaHSCNN()
        self.cnn_hs.eval()
        self.cnn_hs.load_state_dict(
            torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt")
        )

        self.svm_clf = None
        self.svm_model_path = f"{config.BASE_PATH}/models/WandaSVM.pkl"

    def get_preprocess_data(self):
        hs_train_loader = create_hs_data_loader(batch_size=128)
        tranformed_images = []
        for idx, batch in enumerate(hs_train_loader):
            X_1_images = batch["X_1"]
            activation_maps, _ = self.cnn_hs.predict(X_1_images, batch_idx=idx)
            activation_maps = torch.flatten(activation_maps, start_dim=1)
            tranformed_images.append(activation_maps)

        flattened_tranformed_images = torch.cat(tranformed_images)
        return flattened_tranformed_images

    def fit(self):
        preprocessed_data = self.get_preprocess_data()
        preprocessed_data = preprocessed_data.numpy()
        self.svm_clf = OneClassSVM(gamma="auto").fit(preprocessed_data)

    def predict(self, X):
        if isinstance(X, torch.tensor):
            X = X.numpy()
        if self.svm_clf is None:
            self.svm_clf = joblib.load(self.svm_model_path)
        return self.svm_clf.predict(X)

    def save_model(self):
        if self.svm_clf is not None:
            joblib.dump(self.svm_model_path)
            print(f"SVM Model saved to: {self.svm_model_path}")
