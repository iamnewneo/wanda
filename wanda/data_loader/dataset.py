import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from wanda import config
from wanda.utils.util import get_tranforms


class HSWifiTrainDataset(Dataset):
    def __init__(self, train):
        if train:
            self.df = pd.read_csv(f"{config.BASE_PATH}/data/processed/train.csv")
        else:
            self.df = pd.read_csv(f"{config.BASE_PATH}/data/processed/test.csv")
        if config.ENV == "dev":
            self.df = self.df.sample(300).reset_index(drop=True)
        self.transform = get_tranforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        X_1_path = row["path"]
        X_2_path = row["next_image"]
        label = row["label"]

        X_1 = Image.open(X_1_path).convert("RGB")
        X_2 = Image.open(X_2_path).convert("RGB")
        X_1 = self.transform(X_1)
        X_2 = self.transform(X_2)
        return {
            "X_1": X_1.type(torch.float32) / 255,
            "X_2": X_2.type(torch.float32) / 255,
            "label": label,
        }
