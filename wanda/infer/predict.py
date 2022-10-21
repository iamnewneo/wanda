import os
import pandas as pd
from os.path import exists as file_exists
from sklearn.metrics import accuracy_score, roc_auc_score
from wanda import config
from wanda.preprocessing.preprocessor import HSCnnDataPreprocessor, SkDataPreprocessor
from wanda.preprocessing.data_reader import HSDataReader
from wanda.data_loader.data_loader import create_hs_data_loader


class Evaluator:
    def __init__(self, model) -> None:
        self.model = model

    def preprocess_data(self):
        data_reader = HSDataReader(train=False)
        data_reader.process_dataset()

    def evaulate(self, transformed_X, labels, ids, save_postfix=""):
        df_preds = pd.DataFrame()
        print("*********************************************")
        print(f"Started {self.model.model_name} Evaulation")
        if not file_exists(f"{config.BASE_PATH}/data/processed/test.csv"):
            self.preprocess_data()

        df_preds["id"] = ids
        df_preds["label"] = labels

        preds = self.model.predict(transformed_X)
        df_preds["pred_label"] = preds
        preds_folder = f"{config.BASE_PATH}/data/predictions"
        out_path = (
            f"{preds_folder}/"
            + "_".join(self.model.model_name.split(" "))
            + "_"
            + save_postfix
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
