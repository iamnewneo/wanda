import glob
import pandas as pd
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
        self.headers = ["id", "path", "snr", "image_index", "next_image", "label"]

    def get_empty_df(self):
        df = pd.DataFrame(columns=self.headers)
        return df

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
        if df_list:
            df = pd.concat(df_list, ignore_index=True)
        else:
            df = self.get_empty_df()
        return df

    def clean_df(self, df):
        df["label_shifted"] = df["label"].shift(-3)
        df["label_shifted"] = df["label_shifted"].fillna(0)
        df["label_shifted"] = df["label_shifted"] + df["label"]
        df["label_shifted"] = df["label_shifted"].astype(int)
        df = df[df.label_shifted != 1].reset_index(drop=True)
        return df

    def get_df_from_folder(self, folder_path, snr):
        snr_files = glob.glob(f"{folder_path}/*.png", recursive=True)

        df = pd.DataFrame()
        df["path"] = snr_files
        df["snr"] = snr
        if self.train:
            df["image_index"] = (
                df["path"]
                .apply(lambda x: x.split("_")[-1].replace(".png", ""))
                .astype(int)
            )
        else:
            df["image_index"] = (
                df["path"]
                .apply(lambda x: x.split("_")[-2].replace(".png", ""))
                .astype(int)
            )
        df = df.sort_values(by=["snr", "image_index"], ascending=True)

        # For Train label = 0
        df["label"] = 0
        if not self.train:
            df["label_STR"] = df["path"].apply(
                lambda x: x.split("_")[-1].replace(".png", "")
            )
            df["label"] = (
                df["label_STR"].apply(lambda x: 1 if x == "ON" else 0).astype(int)
            )
        df["next_image"] = df.groupby("snr")["path"].shift(-50)
        df = df.dropna()

        df = self.clean_df(df)
        df["id"] = df["image_index"].astype(str) + "_" + df["path"]
        df = df[self.headers]
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
