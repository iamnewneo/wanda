from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from wanda.config import BASE_PATH
import matplotlib.pyplot as plt
import numpy as np
from wanda.utils.util import switch_labels, get_auc_score

# MODELS = ["SVM", "ISOF", "ECOD"]
MODELS = ["SVM", "ISOF", "ECOD", "SVDD"]
MODEL_NAME_MAP = {
    "ISOF": "Isolation_Forest",
    "SVDD": "Deep_SVDD",
    "ECOD": "ECOD",
    "SVM": "SVM",
}


def auc_group(df):
    cols = df.columns.tolist()
    if "model" not in cols:
        raise ValueError(f"Model Not Found in Dataframe")
    models_in_df = df["model"].unique()
    if len(models_in_df) == 1:
        model_name = models_in_df[0]
        model_name = MODEL_NAME_MAP[model_name]
        y_hat = df.label
        y = df.pred_label
        return get_auc_score(y_hat, y, model_name=model_name)
    else:
        raise ValueError(f"More Models in Dataframe: {len(models_in_df)}. Required 1")


def detail_analyze_model():
    SNR = [-10, 0, 10, 20]
    root_path = f"{BASE_PATH}/data/predictions"
    df_test = pd.read_csv(f"{BASE_PATH}/data/processed/test.csv")

    df_svm = pd.read_csv(f"{root_path}/SVM_preds.csv")
    df_svm = df_svm.merge(df_test, how="left", on="id")
    df_svm["model"] = "SVM"

    df_isof = pd.read_csv(f"{root_path}/Isolation_Forest_preds.csv")
    df_isof = df_isof.merge(df_test, how="left", on="id")
    df_isof["model"] = "ISOF"

    df_ecod = pd.read_csv(f"{root_path}/ECOD_preds.csv")
    df_ecod = df_ecod.merge(df_test, how="left", on="id")
    df_ecod["model"] = "ECOD"

    df_svdd = pd.read_csv(f"{root_path}/Deep_SVDD_preds.csv")
    df_svdd = df_svdd.merge(df_test, how="left", on="id")
    df_svdd["model"] = "SVDD"

    df_hs = pd.concat(
        [df_svm, df_isof, df_ecod, df_svdd], ignore_index=True
    ).reset_index(drop=True)
    df_hs["snr"] = df_hs["snr"].astype(int)

    df_hs_auc = (
        df_hs.groupby(["model", "snr"])
        .apply(auc_group)
        .reset_index(drop=False)
        .rename(columns={0: "auc"})
    )

    df_svm = pd.read_csv(f"{root_path}/SVM_plain_preds.csv")
    df_svm = df_svm.merge(df_test, how="left", on="id")
    df_svm["model"] = "SVM"

    df_isof = pd.read_csv(f"{root_path}/Isolation_Forest_plain_preds.csv")
    df_isof = df_isof.merge(df_test, how="left", on="id")
    df_isof["model"] = "ISOF"

    df_ecod = pd.read_csv(f"{root_path}/ECOD_plain_preds.csv")
    df_ecod = df_ecod.merge(df_test, how="left", on="id")
    df_ecod["model"] = "ECOD"

    df_svdd = pd.read_csv(f"{root_path}/Deep_SVDD_plain_preds.csv")
    df_svdd = df_svdd.merge(df_test, how="left", on="id")
    df_svdd["model"] = "SVDD"

    df_plain = pd.concat(
        [df_svm, df_isof, df_ecod, df_svdd], ignore_index=True
    ).reset_index(drop=True)
    df_plain["snr"] = df_plain["snr"].astype(int)

    df_plain_auc = (
        df_plain.groupby(["model", "snr"]).apply(auc_group).reset_index(drop=False)
    ).rename(columns={0: "auc"})

    for snr in SNR:
        plot_snr_comparisons(df_hs_auc, df_plain_auc, snr)
        plot_snr_curves(df_hs, df_plain, snr)


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
    plt.tight_layout()
    ax.figure.savefig(f"{BASE_PATH}/plots/snr_{snr}_comparison.jpeg")
    plt.close()


def plot_snr_curves(df_hs, df_plain, snr):
    df_hs = df_hs[df_hs.snr == snr].reset_index(drop=True)
    df_plain = df_plain[df_plain.snr == snr].reset_index(drop=True)
    for model in MODELS:
        temp_hs = df_hs[df_hs.model == model].reset_index(drop=True)
        fpr, tpr, threshold = None, None, None
        if model in ["SVDD", "ECOD"]:
            fpr, tpr, threshold = roc_curve(temp_hs["label"], temp_hs["pred_label"])
        elif model in ["ISOF", "SVM"]:
            fpr, tpr, threshold = roc_curve(
                switch_labels(temp_hs["label"]), temp_hs["pred_label"]
            )
        plt.plot(fpr, tpr, label=f"{model}")
    plt.title(f"ROC Curve for SNR: {snr} With H-Score")
    plt.legend(loc="lower right")
    # plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/plots/ROC_curve_snr_{snr}_With_Score.jpeg")
    plt.close()

    for model in MODELS:
        temp_plain = df_plain[df_plain.model == model].reset_index(drop=True)
        fpr, tpr, threshold = None, None, None
        if model in ["SVDD", "ECOD"]:
            fpr, tpr, threshold = roc_curve(
                temp_plain["label"], temp_plain["pred_label"]
            )
        elif model in ["ISOF", "SVM"]:
            fpr, tpr, threshold = roc_curve(
                switch_labels(temp_plain["label"]), temp_plain["pred_label"]
            )
        plt.plot(fpr, tpr, label=f"{model}")
    plt.title(f"ROC Curve for SNR: {snr} Without H-Score")
    plt.legend(loc="lower right")
    # plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/plots/ROC_curve_snr_{snr}_Without_Score.jpeg")
    plt.close()
