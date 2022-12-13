from sklearnex import patch_sklearn

patch_sklearn()

import pandas as pd
from wanda import config
from wanda.model.od_algos import DeepSVDDModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.model.svm import SVMModel
from wanda.infer.predict import Evaluator
from wanda.trainer.optimizer import (
    optimize_iso_forest,
    optimize_svm,
    optimize_svdd,
)
from wanda.utils.util import load_object, save_object
from sklearn.preprocessing import StandardScaler

DATA_DIR = f"{config.BASE_PATH}/data"


def main():
    train_prednet()
    evaluate_best_models_prednet()

def train_prednet():
    df_train = pd.read_csv(f"{DATA_DIR}/processed/prednet_train.csv")
    df_test = pd.read_csv(f"{DATA_DIR}/processed/prednet_test.csv")

    df_train["label"] = df_train["label"].astype(int)
    df_test["label"] = df_test["label"].astype(int)

    print("*" * 100)
    print("Models Performance with Prednet")
    print("*" * 100)

    train_dict = {}
    test_dict = {}
    y_train = df_train["label"].to_numpy()
    X_train = df_train.drop(["label", "id"], axis=1).reset_index(drop=True).to_numpy()

    y_test = df_test["label"].to_numpy()
    X_test = df_test.drop(["label", "id"], axis=1).reset_index(drop=True).to_numpy()


    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_dict["transformed_X"] = X_train
    train_dict["labels"] = y_train
    train_dict["ids"] = df_train["id"]

    test_dict["transformed_X"] = X_test
    test_dict["labels"] = y_test
    test_dict["ids"] = df_test["id"]

    best_study_iso = optimize_iso_forest(
        train_dict, test_dict, n_trials=config.N_OPT_TRIALS, prednet=True
    )
    best_model = best_study_iso.__getattribute__("_best_model")
    save_object(best_model, IsoForestModel.model_path)
    print(
        f"Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}"
    )
    best_study_svm = optimize_svm(
        train_dict, test_dict, n_trials=config.N_OPT_TRIALS, prednet=True
    )
    best_model = best_study_svm.__getattribute__("_best_model")
    save_object(best_model, SVMModel.model_path)
    print(
        f"SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}"
    )
    n_trials = 20
    if config.ENV == "dev":
        n_trials = 1
    best_study_svdd = optimize_svdd(
        train_dict, test_dict, n_trials=n_trials, prednet=True
    )
    best_model = best_study_svdd.__getattribute__("_best_model")
    save_object(best_model, DeepSVDDModel.model_path)
    print(
        f"Deep SVDD Best Params: {best_study_svdd.best_params}. Best Value: {-1*best_study_svdd.best_value}"
    )

    print("*" * 100)
    print("*" * 100)
    result_string = (
        f"Summary: Prednet Models Performance"
        f"Isolation Forest Best Params: {best_study_iso.best_params}. Best Value: {-1*best_study_iso.best_value}"
        f"SVM Best Params: {best_study_svm.best_params}. Best Value: {-1*best_study_svm.best_value}"
        f"Deep SVDD Best Params: {best_study_svdd.best_params}. Best Value: {-1*best_study_svdd.best_value}"
    )
    print("*" * 100)
    print("*" * 100)


def evaluate_best_models_prednet(test_dict):
    print("*" * 100)
    print("Evaluating Prednet Models Performance")
    print("*" * 100)

    transformed_X = test_dict["transformed_X"]
    labels = test_dict["labels"]
    ids = test_dict["ids"]

    clf = load_object(SVMModel.model_path)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)

    clf = load_object(IsoForestModel.model_path)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)

    clf = DeepSVDDModel(contamination=0.3206670971813005)
    clf.fit(transformed_X)
    model_evaluator = Evaluator(model=clf)
    model_evaluator.evaulate(transformed_X, labels, ids)


if __name__ == "__main__":
    main()
