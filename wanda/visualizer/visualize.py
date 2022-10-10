import math
import torch
import random
import numpy as np
import seaborn as sns
from collections import defaultdict
import matplotlib.pyplot as plt
from wanda.utils.util import un_norm_image
from wanda.preprocessing.data_reader import DATA_DIR
from wanda.data_loader.dataset import HSWifiTrainDataset
from wanda import config


class Visualize:
    def __init__(self, hs_cnn, train=False, n_samples=4) -> None:
        self.train = train
        self.processed_data_path = f"{DATA_DIR}/processed"
        self.hs_cnn = hs_cnn
        self.hs_cnn.eval()
        self.n_samples = n_samples

    def visualize(self, samples_dict):
        for label, images in samples_dict.items():
            for image_idx, image in enumerate(images):
                activation_maps = image["activation_maps"]
                act_map_len = len(activation_maps)
                n_rows = math.ceil((act_map_len + 1) / 3)
                fig, axes = plt.subplots(nrows=n_rows, ncols=3, figsize=(15, 15))
                for i, ax in enumerate(axes.flat, start=0):
                    if i == 0:
                        ax.set_title(f"Input Image Label: {label}")
                        permuted_image = un_norm_image(torch.from_numpy(image["image"]))
                        permuted_image = permuted_image.detach().numpy()
                        permuted_image = np.moveaxis(permuted_image, 0, -1)
                        ax.imshow(permuted_image)
                    else:
                        ax.set_title(f"Channel: {i}")
                        sns.heatmap(activation_maps[i - 1], square=True, ax=ax)
                fig.tight_layout()
                plt.savefig(
                    f"{config.BASE_PATH}/plots/label_{label}_image_idx_{image_idx}_heatmap.png"
                )

    def get_plotting_data(self, samples):
        with torch.no_grad():
            data_dict = {}
            for label, v in samples.items():
                image_list = []
                for image in v:
                    image = image.unsqueeze(dim=0)
                    activation_maps, _ = self.hs_cnn(image)
                    image_list.append(
                        {
                            "activation_maps": activation_maps.squeeze()
                            .detach()
                            .numpy(),
                            "image": image.squeeze().squeeze().detach().numpy(),
                            "label": label,
                        }
                    )
                data_dict[label] = image_list
        return data_dict

    def random_visualize(self, labels):
        data = HSWifiTrainDataset(train=self.train)
        #### samples = {label: [images]}
        samples = defaultdict(list)
        for label in labels:
            n_samples = self.n_samples
            random_indexes = list(range(len(data)))
            random.shuffle(random_indexes)
            for i in random_indexes:
                if label == data[i]["label"]:
                    samples[label].append(data[i]["X_1"])
                    n_samples -= 1
                if n_samples == 0:
                    break

        plotting_data = self.get_plotting_data(samples)
        self.visualize(plotting_data)
