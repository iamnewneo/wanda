import os
import pandas as pd
from os.path import exists as file_exists
from sklearn.metrics import accuracy_score, roc_auc_score
from wanda import config
from wanda.preprocessing.preprocessor import HSCnnDataPreprocessor, SkDataPreprocessor
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader


class Evaluator:
    def __init__(self, model, hs_preprocess=True) -> None:
        self.model = model
        self.model.load_model()
        self.hs_preprocess = hs_preprocess
        self.hs_cnn_preprocessor = HSCnnDataPreprocessor()
        self.sk_preprocessor = SkDataPreprocessor()

    def preprocess_data(self):
        data_reader = HSDataReader(train=False)
        data_reader.process_dataset()

    def get_dataset(self):
        transformed_X, labels, ids = None, None, None
        if self.hs_cnn_preprocessor:
            test_loader = create_hs_data_loader(
                batch_size=config.TEST_BATCH_SIZE, train=False, shuffle=False
            )
            transformed_X, labels, ids = self.hs_cnn_preprocessor.get_preprocess_data(
                test_loader, ids=True
            )
        else:
            test_loader = create_hs_data_loader(
                batch_size=config.TEST_BATCH_SIZE,
                train=False,
                shuffle=False,
                greyscale=True,
            )
            transformed_X, labels = self.sk_preprocessor.get_preprocess_data(
                test_loader, ids=True
            )
        return transformed_X, labels, ids

    def evaulate(self):
        df_preds = pd.DataFrame()
        print("*********************************************")
        print(f"Started {self.model.model_name} Evaulation")
        if not file_exists(f"{config.BASE_PATH}/data/processed/test.csv"):
            self.preprocess_data()

        transformed_X, labels, ids = self.get_dataset()
        df_preds["id"] = ids
        df_preds["label"] = labels

        preds = self.model.predict(transformed_X)
        df_preds["pred_label"] = preds
        preds_folder = f"{config.BASE_PATH}/data/predictions"
        out_path = (
            f"{preds_folder}/"
            + "_".join(self.model.model_name.split(" "))
            + "_"
            + "preds.csv"
        )
        if not file_exists(preds_folder):
            os.mkdir(preds_folder)

        df_preds = df_preds[["id", "label", "pred_label"]].reset_index(drop=True)
        df_preds.to_csv(out_path, index=False)

        # Temp Hack: FIXIT, ROC Cannot take all values as same class
        tranformed_preds = []
        for x in preds:
            if x == -1:
                tranformed_preds.append(1)
            else:
                tranformed_preds.append(0)
        labels[-1] = 0
        auc_score = roc_auc_score(labels, tranformed_preds)
        accuracy = accuracy_score(labels, tranformed_preds)

        print(f"{self.model.model_name} Model Performance:")
        print(f"Accuracy: {accuracy:.2f}. AUC: {auc_score:.2f}")
        print("*********************************************")
