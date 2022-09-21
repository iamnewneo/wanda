import os
import glob
import pandas as pd
from scipy.io import loadmat
from wanda import config

DATA_DIR = f"{config.BASE_PATH}/data"
TRAIN_DIR = f"{DATA_DIR}/Training_Data_Set_Without_Interference"
TEST_DIR = f"{DATA_DIR}/Test_Data_Set_With_Interference"


class HSDataReader:
    def __init__(self, train=True) -> None:
        if train:
            _data_dir = TRAIN_DIR
        else:
            _data_dir = TEST_DIR
        self.train = train
        self.sig_snr_neg_10 = f"{_data_dir}/WiFi2_-10"
        self.sig_snr_0 = f"{_data_dir}/WiFi2_0"
        self.sig_snr_10 = f"{_data_dir}/WiFi2_10"
        self.sig_snr_20 = f"{_data_dir}/WiFi2_20"
        self.processed_data_path = f"{DATA_DIR}/processed"
        self.headers = ["path", "snr", "image_index", "next_image", "label"]

    def get_snr_path(self, snr):
        if snr == -10:
            return f"{self.sig_snr_neg_10}"
        elif snr == 0:
            return f"{self.sig_snr_0}"
        elif snr == 10:
            return f"{self.sig_snr_10}"
        elif snr == 20:
            return f"{self.sig_snr_20}"
        else:
            raise (f"Invlalid snr: {snr}")

    def get_snr_df(self, snr):
        df_list = []
        snr_path = self.get_snr_path(snr)
        folders = glob.glob(f"{snr_path}/*")
        for folder in folders:
            temp_df = self.get_df_from_folder(folder, snr)
            df_list.append(temp_df)

        df = pd.concat(df_list, ignore_index=True)
        return df

    def get_df_from_folder(self, folder_path, snr):
        snr_files = glob.glob(f"{folder_path}/*.png", recursive=True)
        # For Train label = 0
        labels = 0
        if not self.train:
            labels = loadmat(f"{folder_path}/groundTruth.mat")["groundTruth"][0]
            labels = labels[: len(snr_files)]

        df = pd.DataFrame()
        df["path"] = snr_files
        df["snr"] = snr
        df["image_index"] = (
            df["path"].apply(lambda x: x.split("_")[-1].replace(".png", "")).astype(int)
        )
        df = df.sort_values(by=["snr", "image_index"], ascending=True)
        df["label"] = labels
        df["next_image"] = df.groupby("snr")["path"].shift(-1)
        df = df.dropna()

        return df

    def get_train_df(self):
        sig_snr_neg_10_df = self.get_snr_df(-10)
        sig_snr_0_df = self.get_snr_df(0)
        sig_snr_10_df = self.get_snr_df(10)
        sig_snr_20_df = self.get_snr_df(20)
        df = pd.concat(
            [sig_snr_neg_10_df, sig_snr_0_df, sig_snr_10_df, sig_snr_20_df],
            ignore_index=True,
        )
        df = df[self.headers]
        if self.train:
            df.to_csv(f"{self.processed_data_path}/train.csv", index=False)
        else:
            df.to_csv(f"{self.processed_data_path}/test.csv", index=False)
        return df

    def process_dataset(self):
        self.get_train_df()


if __name__ == "__main__":
    data_reader = HSDataReader()
    df = data_reader.get_train_df()

    print(df.head())
