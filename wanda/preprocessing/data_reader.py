import pandas as pd
import glob
from wanda import config

DATA_DIR = f"{config.BASE_PATH}/data"
TRAIN_DIR = f"{DATA_DIR}/Training_Data_Set_Without_Interference"


class HSDataReader:
    def __init__(self) -> None:
        self.sig_snr_neg_10 = f"{TRAIN_DIR}/Wifi2_-10"
        self.sig_snr_0 = f"{TRAIN_DIR}/Wifi2_0"
        self.sig_snr_10 = f"{TRAIN_DIR}/Wifi2_10"
        self.sig_snr_20 = f"{TRAIN_DIR}/Wifi2_20"
        self.processed_data_path = f"{DATA_DIR}/processed"

    def get_train_df(self):
        sig_snr_neg_10_files = glob.glob(f"{self.sig_snr_neg_10}/**/*.png", recursive=True)
        sig_snr_0_files = glob.glob(f"{self.sig_snr_0}/**/*.png", recursive=True)
        sig_snr_10_files = glob.glob(f"{self.sig_snr_10}/**/*.png", recursive=True)
        sig_snr_20_files = glob.glob(f"{self.sig_snr_20}/**/*.png", recursive=True)

        sig_snr_neg_10_df = pd.DataFrame()
        sig_snr_neg_10_df["path"] = sig_snr_neg_10_files
        sig_snr_neg_10_df["snr"] = -10
        sig_snr_neg_10_df["label"] = 0

        sig_snr_0_df = pd.DataFrame()
        sig_snr_0_df["path"] = sig_snr_0_files
        sig_snr_0_df["snr"] = -0
        sig_snr_0_df["label"] = 0

        sig_snr_10_df = pd.DataFrame()
        sig_snr_10_df["path"] = sig_snr_10_files
        sig_snr_10_df["snr"] = 10
        sig_snr_10_df["label"] = 0

        sig_snr_20_df = pd.DataFrame()
        sig_snr_20_df["path"] = sig_snr_20_files
        sig_snr_20_df["snr"] = 20
        sig_snr_20_df["label"] = 0

        df = pd.concat(
            [sig_snr_neg_10_df, sig_snr_0_df, sig_snr_10_df, sig_snr_20_df],
            ignore_index=True,
        )
        df = df.reset_index(drop=True)
        df["image_index"] = df["path"].apply(lambda x: x.split("_")[-1].replace(".png", "")).astype(int)
        df = df.sort_values(by=["snr", "image_index"], ascending=True)
        df['next_image'] = df.groupby('snr')['path'].shift(-1)
        df = df.dropna()
        df.to_csv(f"{self.processed_data_path}/train.csv", index=False)
        return df

    def process_dataset(self):
        self.get_train_df()


if __name__ == "__main__":
    data_reader = HSDataReader()
    df = data_reader.get_train_df()

    print(df.head())