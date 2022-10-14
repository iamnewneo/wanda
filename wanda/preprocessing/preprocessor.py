import torch
import numpy as np
from wanda import config
from sklearn.decomposition import KernelPCA
from wanda.model.cnn_hscore import WandaHSCNN


class HSCnnDataPreprocessor:
    def __init__(self) -> None:
        self.cnn_hs = WandaHSCNN()
        self.cnn_hs.load_state_dict(
            torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt")
        )
        self.cnn_hs.eval()
        self.svd = None

    def svd_transform(self, X):
        if self.svd is None:
            self.svd = KernelPCA(
                n_components=200,
                kernel="rbf",
                max_iter=100,
                random_state=42,
                n_jobs=config.N_JOBS,
            )
            self.svd.fit(X)
        return self.svd.transform(X)

    def get_preprocess_data(self, data_loader, ids=False):
        tranformed_images = []
        labels = []
        ids_list = []
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                X_1_images = batch["X_1"]
                activation_maps, _ = self.cnn_hs(X_1_images)
                activation_maps = torch.flatten(activation_maps, start_dim=1)
                tranformed_images.append(activation_maps)
                labels.append(batch["label"].numpy())
                ids_list.append(batch["id"])

        flattened_tranformed_images = torch.cat(tranformed_images)
        labels = np.concatenate(labels, axis=0)
        labels = labels.ravel()
        ids_list = [item for sublist in ids_list for item in sublist]
        flattened_tranformed_images = self.svd_transform(flattened_tranformed_images)
        if ids:
            return flattened_tranformed_images, labels, ids_list
        return flattened_tranformed_images, labels


class SkDataPreprocessor:
    def __init__(self) -> None:
        self.svd = None

    def svd_transform(self, X):
        if self.svd is None:
            self.svd = KernelPCA(
                n_components=200, max_iter=100, random_state=42, n_jobs=config.N_JOBS
            )
            self.svd.fit(X)
        return self.svd.transform(X)

    def get_preprocess_data(self, data_loader, ids=False):
        tranformed_images = []
        labels = []
        ids_list = []
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                X_1_images = batch["X_1"]
                image_np_array = X_1_images.detach().numpy()
                image_np_array = np.squeeze(image_np_array, axis=1)
                image_np_array = image_np_array.reshape((image_np_array.shape[0], -1))
                tranformed_images.append(image_np_array)
                labels.append(batch["label"].numpy())
                ids_list.append(batch["id"])

        flattened_tranformed_images = np.concatenate(tranformed_images)
        labels = np.concatenate(labels, axis=0)
        labels = labels.ravel()
        ids_list = [item for sublist in ids_list for item in sublist]
        flattened_tranformed_images = self.svd_transform(flattened_tranformed_images)
        if ids:
            return flattened_tranformed_images, labels, ids_list
        return flattened_tranformed_images, labels
