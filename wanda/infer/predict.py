import os
import pandas as pd
from os.path import exists as file_exists
from sklearn.metrics import roc_auc_score
from wanda import config
from wanda.preprocessing.data_reader import HSDataReader
from wanda.utils.util import switch_labels


class Evaluator:
    def __init__(self, model) -> None:
        self.model = model

    def preprocess_data(self):
        data_reader = HSDataReader(train=False)
        data_reader.process_dataset()

    def evaulate(self, transformed_X, labels, ids, save_postfix=""):
        if save_postfix != "":
            save_postfix = save_postfix + "_"
        df_preds = pd.DataFrame()
        print("*********************************************")
        print(f"Started {self.model.model_name} Evaulation")
        if not file_exists(f"{config.BASE_PATH}/data/processed/test.csv"):
            self.preprocess_data()

        df_preds["id"] = ids
        # df_preds["label"] = labels

        preds = self.model.decision_function(transformed_X)
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

        df_preds = df_preds[["id", "pred_label"]].reset_index(drop=True)
        df_preds.to_csv(out_path, index=False)

        # # Temp Hack: FIXIT, ROC Cannot take all values as same class
        # tranformed_preds = []
        # for x in preds:
        #     if x == -1:
        #         tranformed_preds.append(1)
        #     else:
        #         tranformed_preds.append(0)
        # labels[-1] = 0
        auc_score = get_auc_score(labels, preds)
        # accuracy = accuracy_score(labels, tranformed_preds)

        print(f"{self.model.model_name} Model Performance:")
        print(f" AUC: {auc_score:.2f}")
        print("*********************************************")
