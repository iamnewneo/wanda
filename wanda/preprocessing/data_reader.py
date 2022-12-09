import glob
import pandas as pd
from wanda import config
from sklearn.model_selection import train_test_split

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


class PrednetDataReader:
    def __init__(self, train=True) -> None:
        _data_dir = f"{DATA_DIR}/PredNet_Spectrogram_Data_Set"
        self.train = train
        self.sig_snr_neg_10 = f"{_data_dir}/Error_-10db.csv"
        self.sig_snr_0 = f"{_data_dir}/Error_0db.csv"
        self.sig_snr_10 = f"{_data_dir}/Error_10db.csv"
        self.sig_snr_20 = f"{_data_dir}/Error_20db.csv"
        self.processed_data_path = f"{DATA_DIR}/processed"

        self.sig_snr_neg_10_labels = f"{_data_dir}/GT_-10db.csv"
        self.sig_snr_0_labels = f"{_data_dir}/GT_0db.csv"
        self.sig_snr_10_labels = f"{_data_dir}/GT_10db.csv"
        self.sig_snr_20_labels = f"{_data_dir}/GT_20db.csv"

    def preprocess_data(self):
        df_snr_neg_10 = pd.read_csv(self.sig_snr_neg_10, header=None)
        df_snr_0 = pd.read_csv(self.sig_snr_0, header=None)
        df_snr_10 = pd.read_csv(self.sig_snr_10, header=None)
        df_snr_20 = pd.read_csv(self.sig_snr_20, header=None)

        df_snr_neg_10_labels = pd.read_csv(self.sig_snr_neg_10_labels, names=["label"])
        df_snr_0_labels = pd.read_csv(self.sig_snr_0_labels, names=["label"])
        df_snr_10_labels = pd.read_csv(self.sig_snr_10_labels, names=["label"])
        df_snr_20_labels = pd.read_csv(self.sig_snr_20_labels, names=["label"])

        df_snr_neg_10["label"] = df_snr_neg_10_labels["label"]
        df_snr_0["label"] = df_snr_0_labels["label"]
        df_snr_10["label"] = df_snr_10_labels["label"]
        df_snr_20["label"] = df_snr_20_labels["label"]

        df_final = pd.concat(
            [df_snr_neg_10, df_snr_0, df_snr_10, df_snr_20], ignore_index=True
        ).reset_index(drop=True)

        y = df_final["label"]
        X = df_final.drop(["label"], axis=1).reset_index(drop=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, stratify=y, test_size=0.25, random_state=42
        )

        df_train = X_train
        df_train["label"] = y_train
        df_test = X_test
        df_test["label"] = y_test

        temp = df_train[df_train.label == 1].reset_index(drop=True)
        df_train = df_train[df_train.label == 0].reset_index(drop=True)
        df_test = pd.concat([df_test, temp], ignore_index=True).reset_index(drop=True)

        df_train.to_csv(f"{self.processed_data_path}/prednet_train.csv", index=False)
        df_test.to_csv(f"{self.processed_data_path}/prednet_test.csv", index=False)


if __name__ == "__main__":
    data_reader = HSDataReader()
    df = data_reader.get_train_df()

    print(df.head())
