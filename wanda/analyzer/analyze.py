from sklearn.metrics import roc_auc_score, roc_curve
import pandas as pd
from wanda.config import BASE_PATH
import matplotlib.pyplot as plt
import numpy as np
from wanda.utils.util import switch_labels, get_auc_score

# MODELS = ["SVM", "ISOF", "ECOD"]
MODELS = ["SVM", "ISOF", "SVDD"]
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
    df_hs = df_hs[df_hs.model.isin(MODELS)].reset_index(drop=True)

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
    df_plain = df_plain[df_plain.model.isin(MODELS)].reset_index(drop=True)

    df_plain_auc = (
        df_plain.groupby(["model", "snr"]).apply(auc_group).reset_index(drop=False)
    ).rename(columns={0: "auc"})

    df_prednet_test = pd.read_csv(f"{BASE_PATH}/data/processed/prednet_test.csv")

    df_prednet_test = df_prednet_test[["id", "snr", "label"]]
    # df_prednet_test["id"] = df_prednet_test["id"].astype(str)

    df_svm = pd.read_csv(f"{root_path}/SVM_prednet_preds.csv")
    df_svm = df_svm.merge(df_prednet_test, how="left", on="id")
    df_svm["model"] = "SVM"

    df_isof = pd.read_csv(f"{root_path}/Isolation_Forest_prednet_preds.csv")
    df_isof = df_isof.merge(df_prednet_test, how="left", on="id")
    df_isof["model"] = "ISOF"

    df_svdd = pd.read_csv(f"{root_path}/Deep_SVDD_prednet_preds.csv")
    df_svdd = df_svdd.merge(df_prednet_test, how="left", on="id")
    df_svdd["model"] = "SVDD"

    df_prednet = pd.concat([df_svm, df_isof, df_svdd], ignore_index=True).reset_index(
        drop=True
    )
    df_prednet["snr"] = df_prednet["snr"].astype(int)
    df_prednet = df_prednet[df_prednet.model.isin(MODELS)].reset_index(drop=True)

    df_prednet["snr"] = df_prednet["snr"].astype(int)
    df_prednet["label"] = df_prednet["label"].astype(int)
    df_prednet["pred_label"] = df_prednet["pred_label"].astype(float)

    df_prednet_auc = (
        df_prednet.groupby(["model", "snr"]).apply(auc_group).reset_index(drop=False)
    ).rename(columns={0: "auc"})

    plot_overall_auc_comparison(df_hs, df_plain, df_prednet=df_prednet)
    plot_overall_roc_curve(df_hs, df_plain, df_prednet=df_prednet)

    for snr in SNR:
        plot_snr_comparisons(df_hs_auc, df_plain_auc, snr, df_prednet=df_prednet_auc)
        plot_snr_curves(df_hs, df_plain, snr, df_prednet=df_prednet)


def plot_snr_comparisons(df_hs, df_plain, snr, df_prednet=None):
    df_hs = df_hs[df_hs.snr == snr].reset_index(drop=True)
    df_plain = df_plain[df_plain.snr == snr].reset_index(drop=True)
    if df_prednet is not None:
        df_prednet = df_prednet[df_prednet.snr == snr].reset_index(drop=True)
    labels = list(df_hs["model"].unique())

    df_plot = pd.DataFrame()
    df_plot["Algorithm"] = df_hs["model"]
    df_plot["H_Score"] = df_hs["auc"]
    df_plot["Without_H_Score"] = df_plain["auc"]
    if df_prednet is not None:
        df_plot["Prednet"] = df_prednet["auc"]

    ax = df_plot.plot(
        x="Algorithm",
        # xlabel="AUC",
        kind="bar",
        stacked=False,
        title=f"SIR: {snr} AUC Comparison",
    )
    ax.set_ylabel("AUC")
    plt.tight_layout()
    ax.figure.savefig(f"{BASE_PATH}/plots/snr_{snr}_comparison.jpeg")
    plt.close()


def plot_snr_curves(df_hs, df_plain, snr, df_prednet=None):
    df_hs = df_hs[df_hs.snr == snr].reset_index(drop=True)
    df_plain = df_plain[df_plain.snr == snr].reset_index(drop=True)
    if df_prednet is not None:
        df_prednet = df_prednet[df_prednet.snr == snr].reset_index(drop=True)
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
    plt.title(f"ROC Curve for SIR: {snr} With H-Score")
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
    plt.title(f"ROC Curve for SIR: {snr} Without H-Score")
    plt.legend(loc="lower right")
    # plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/plots/ROC_curve_snr_{snr}_Without_Score.jpeg")
    plt.close()

    if df_prednet is not None:
        for model in MODELS:
            temp_plain = df_prednet[df_prednet.model == model].reset_index(drop=True)
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
        plt.title(f"ROC Curve for SIR: {snr} Prednet")
        plt.legend(loc="lower right")
        # plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.tight_layout()
        plt.savefig(f"{BASE_PATH}/plots/ROC_curve_snr_{snr}_Prednet.jpeg")
        plt.close()


def plot_overall_auc_comparison(df_hs, df_plain, df_prednet=None):
    df_hs_auc_overall = (
        df_hs.groupby(["model"])
        .apply(auc_group)
        .reset_index(drop=False)
        .rename(columns={0: "With H-Score"})
    )

    df_plain_auc_overall = (
        df_plain.groupby(["model"]).apply(auc_group).reset_index(drop=False)
    ).rename(columns={0: "Without H-Score"})

    df_both_auc_overall = df_hs_auc_overall.merge(
        df_plain_auc_overall, how="left", on="model"
    )

    if df_prednet is not None:
        df_prednet_auc_overall = (
            df_prednet.groupby(["model"]).apply(auc_group).reset_index(drop=False)
        ).rename(columns={0: "Prednet"})
        df_both_auc_overall = df_both_auc_overall.merge(
            df_prednet_auc_overall, how="left", on="model"
        )

    y = ["With H-Score", "Without H-Score"]
    if df_prednet is not None:
        y = ["With H-Score", "Without H-Score", "Prednet"]
    ax = df_both_auc_overall.plot(
        x="model",
        y=y,
        kind="bar",
        title=f"Overall AUC Comparison",
        # xlabel="AUC",
    )
    ax.set_xlabel("Models")
    ax.set_ylabel("AUC")
    plt.tight_layout()
    ax.figure.savefig(f"{BASE_PATH}/plots/overall_auc_comparison.jpeg")
    plt.close()


def plot_overall_roc_curve(df_hs, df_plain, df_prednet=None):
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
    plt.title(f"Overall ROC Curve With H-Score")
    plt.legend(loc="lower right")
    # plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/plots/Overall_ROC_curve_With_Score.jpeg")
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
    plt.title(f"Overall ROC Curve Without H-Score")
    plt.legend(loc="lower right")
    # plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel("True Positive Rate")
    plt.xlabel("False Positive Rate")
    plt.tight_layout()
    plt.savefig(f"{BASE_PATH}/plots/Overall_ROC_curve_Without_Score.jpeg")
    plt.close()

    if df_prednet is not None:
        for model in MODELS:
            temp_plain = df_prednet[df_prednet.model == model].reset_index(drop=True)
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
        plt.title(f"Overall ROC Curve for Prednet")
        plt.legend(loc="lower right")
        # plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel("True Positive Rate")
        plt.xlabel("False Positive Rate")
        plt.tight_layout()
        plt.savefig(f"{BASE_PATH}/plots/Overall_ROC_curve_Prednet.jpeg")
        plt.close()
