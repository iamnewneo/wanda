import torch
import numpy as np
from tqdm import tqdm
from wanda import config
from sklearn.decomposition import TruncatedSVD
from umap import UMAP
from wanda.model.cnn_hscore import WandaHSCNN


class HSCnnDataPreprocessor:
    def __init__(self, svd_tranformation=True) -> None:
        self.cnn_hs = WandaHSCNN()
        self.cnn_hs.load_state_dict(
            torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt")
        )
        self.cnn_hs.eval()
        self.batch_size = 1000
        self.svd = UMAP(
            n_components=100,
            n_neighbors=15,
            min_dist=0.15,
            metric="correlation",
            verbose=False,
            n_jobs=config.N_JOBS,
        )
        self.svd_tranformation = svd_tranformation

    def svd_transform(self, X):
        # length = len(X)
        # total = (length // self.batch_size) + 1
        # for i in tqdm(range(0, length, self.batch_size), total=total):
        #     self.svd.partial_fit(X[i : i + self.batch_size])
        return self.svd.fit_transform(X)

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
        if self.svd_tranformation:
            flattened_tranformed_images = self.svd_transform(
                flattened_tranformed_images
            )
        if ids:
            return flattened_tranformed_images, labels, ids_list
        return flattened_tranformed_images, labels


class SkDataPreprocessor:
    def __init__(self, svd_tranformation=True) -> None:
        self.batch_size = 1000
        self.svd = UMAP(
            n_components=100,
            n_neighbors=15,
            min_dist=0.15,
            metric="correlation",
            verbose=False,
            n_jobs=config.N_JOBS,
        )
        self.svd_tranformation = svd_tranformation

    def svd_transform(self, X):
        # length = len(X)
        # total = (length // self.batch_size) + 1
        # for i in tqdm(range(0, length, self.batch_size), total=total):
        #     self.svd.partial_fit(X[i : i + self.batch_size])
        return self.svd.fit_transform(X)

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
        if self.svd_tranformation:
            flattened_tranformed_images = self.svd_transform(
                flattened_tranformed_images
            )
        if ids:
            return flattened_tranformed_images, labels, ids_list
        return flattened_tranformed_images, labels
