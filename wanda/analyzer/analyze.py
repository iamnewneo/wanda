from sklearn.metrics import roc_auc_score
import pandas as pd
from wanda.config import BASE_PATH
import matplotlib.pyplot as plt
import numpy as np


def auc_group(df):
    y_hat = df.label
    y = df.pred_label
    return roc_auc_score(y_hat, y)


def detail_analyze_model():
    SNR = [-10, 0, 10, 20]
    root_path = f"{BASE_PATH}/data/predictions"
    df_test = pd.read_csv(f"{BASE_PATH}/data/processed/test.csv")

    df_svm = pd.read_csv(f"{root_path}/SVM_plain_preds.csv")
    df_svm = df_svm.merge(df_test, how="left", on="id")
    df_svm["model"] = "SVM"

    df_isof = pd.read_csv(f"{root_path}/Isolation_Forest_plain_preds.csv")
    df_isof = df_isof.merge(df_test, how="left", on="id")
    df_isof["model"] = "ISOF"

    df_ecod = pd.read_csv(f"{root_path}/ECOD_plain_preds.csv")
    df_ecod = df_ecod.merge(df_test, how="left", on="id")
    df_ecod["model"] = "ECOD"

    # df_svdd = pd.read_csv(f"{root_path}/SVDD_preds.csv")
    # df_svdd = df_svdd.merge(df_test, how="left", on="id")
    # df_svdd["model"] = "SVDD"

    df_hs = pd.concat([df_svm, df_isof, df_ecod], ignore_index=True).reset_index(
        drop=True
    )
    df_hs["snr"] = df_hs["snr"].astype(int)

    df_svm = pd.read_csv(f"{root_path}/SVM_plain_preds.csv")
    df_svm = df_svm.merge(df_test, how="left", on="id")
    df_svm["model"] = "SVM"

    df_isof = pd.read_csv(f"{root_path}/Isolation_Forest_plain_preds.csv")
    df_isof = df_isof.merge(df_test, how="left", on="id")
    df_isof["model"] = "ISOF"

    df_ecod = pd.read_csv(f"{root_path}/ECOD_plain_preds.csv")
    df_ecod = df_ecod.merge(df_test, how="left", on="id")
    df_ecod["model"] = "ECOD"

    # df_svdd = pd.read_csv(f"{root_path}/SVDD_preds.csv")
    # df_svdd = df_svdd.merge(df_test, how="left", on="id")
    # df_svdd["model"] = "SVDD"

    df_plain = pd.concat([df_svm, df_isof, df_ecod], ignore_index=True).reset_index(
        drop=True
    )
    df_hs["snr"] = df_hs["snr"].astype(int)

    df_hs_auc = (
        df_hs.groupby(["model", "snr"])
        .apply(auc_group)
        .reset_index(drop=False)
        .rename(columns={0: "auc"})
    )
    df_plain_auc = (
        df_plain.groupby(["model", "snr"]).apply(auc_group).reset_index(drop=False)
    ).rename(columns={0: "auc"})

    for snr in SNR:
        plot_snr_comparisons(df_hs_auc, df_plain_auc, snr)


def plot_snr_comparisons(df_hs, df_plain, snr):
    df_hs = df_hs[df_hs.snr == snr].reset_index(drop=True)
    df_plain = df_plain[df_plain.snr == snr].reset_index(drop=True)
    labels = list(df_hs["model"].unique())

    df_plot = pd.DataFrame()
    df_plot["Algorithm"] = df_hs["model"]
    df_plot["H_Score"] = df_hs["auc"]
    df_plot["Without_H_Score"] = df_plain["auc"]

    ax = df_plot.plot(
        x="Algorithm",
        kind="bar",
        stacked=False,
        title=f"SNR: {snr} Wise AUC Comparison",
    )
    ax.figure.savefig(f"{BASE_PATH}/plots/snr_{snr}_comparison.jpeg")
