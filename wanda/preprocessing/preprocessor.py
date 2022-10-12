import torch
from torch import nn
import numpy as np
from wanda import config
import pytorch_lightning as pl
from wanda.model.cnn_hscore import WandaHSCNN


class HSCnnDataPreprocessor:
    def __init__(self) -> None:
        self.cnn_hs = WandaHSCNN()
        self.cnn_hs.load_state_dict(
            torch.load(f"{config.BASE_PATH}/models/WandaHSCNN.pt")
        )
        self.cnn_hs.eval()

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


class SkDataPreprocessor:
    def __init__(self) -> None:
        pass

    def get_preprocess_data(self, data_loader):
        tranformed_images = []
        labels = []
        with torch.no_grad():
            for idx, batch in enumerate(data_loader):
                X_1_images = batch["X_1"]
                image_np_array = X_1_images.detach().numpy()
                image_np_array = np.squeeze(image_np_array, axis=1)
                image_np_array = image_np_array.reshape((image_np_array.shape[0], -1))
                tranformed_images.append(image_np_array)
                labels.append(batch["label"].numpy())

        flattened_tranformed_images = np.concatenate(tranformed_images)
        labels = np.concatenate(labels, axis=0)
        labels = labels.ravel()
        return flattened_tranformed_images, labels
