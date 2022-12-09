import optuna
from optuna import Trial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from wanda.model.od_algos import DeepSVDDModel, ECODModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.utils.util import switch_labels, get_auc_score
from wanda.model.svm import SVMModel
from wanda import config

optuna.logging.set_verbosity(optuna.logging.ERROR)

DEFAULT_N_TRIALS = 200

trial_model_dict = {
    "SVM": list(range(DEFAULT_N_TRIALS)),
    "SVDD": list(range(DEFAULT_N_TRIALS)),
    "ECOD": list(range(DEFAULT_N_TRIALS)),
    "ISOF": list(range(DEFAULT_N_TRIALS)),
}


def optimize_iso_forest_fn(trial, train_dict, test_dict, prednet=False):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_features = trial.suggest_float("max_features", 0.2, 0.95)
    contamination = trial.suggest_float("contamination", 0.1, 0.5)
    iso_forest_clf = IsoForestModel(
        random_state=42,
        n_estimators=n_estimators,
        max_features=max_features,
        contamination=contamination,
        n_jobs=config.N_JOBS,
        prednet=prednet,
    )
    transformed_X_train = train_dict["transformed_X"]
    transformed_X_test = test_dict["transformed_X"]
    labels_test = test_dict["labels"]
    iso_forest_clf.fit(transformed_X_train)
    y_pred = iso_forest_clf.decision_function(transformed_X_test)
    auc_score = get_auc_score(labels_test, y_pred, model_name="Isolation_Forest")
    trial_model_dict["ISOF"][trial.number] = iso_forest_clf
    return -1 * auc_score


def optimize_svm_fn(trial, train_dict, test_dict, prednet=False):
    nu = trial.suggest_float("nu", 0.05, 0.95)
    power_t = trial.suggest_float("power_t", 0.1, 0.9)
    svm_clf = SVMModel(nu=nu, power_t=power_t, prednet=prednet)
    transformed_X_train = train_dict["transformed_X"]
    transformed_X_test = test_dict["transformed_X"]
    labels_test = test_dict["labels"]
    svm_clf.fit(transformed_X_train)
    y_pred = svm_clf.decision_function(transformed_X_test)
    auc_score = get_auc_score(labels_test, y_pred, model_name="SVM")
    trial_model_dict["SVM"][trial.number] = svm_clf
    return -1 * auc_score


def optimize_deep_svdd_fn(trial, train_dict, test_dict, prednet=False):
    contamination = trial.suggest_float("contamination", 0.1, 0.45)
    svdd_clf = DeepSVDDModel(contamination=contamination, prednet=prednet)
    transformed_X_train = train_dict["transformed_X"]
    transformed_X_test = test_dict["transformed_X"]
    labels_test = test_dict["labels"]
    svdd_clf.fit(transformed_X_train)
    y_pred = svdd_clf.decision_function(transformed_X_test)
    auc_score = get_auc_score(labels_test, y_pred, model_name="Deep_SVDD")
    trial_model_dict["SVDD"][trial.number] = svdd_clf
    return -1 * auc_score


def optimize_ecod_fn(trial, train_dict, test_dict):
    contamination = trial.suggest_float("contamination", 0.1, 0.45)
    ecod_clf = ECODModel(contamination=contamination)
    transformed_X_train = train_dict["transformed_X"]
    transformed_X_test = test_dict["transformed_X"]
    labels_test = test_dict["labels"]
    ecod_clf.fit(transformed_X_train)
    y_pred = ecod_clf.decision_function(transformed_X_test)
    auc_score = get_auc_score(labels_test, y_pred, model_name="ECOD")
    trial_model_dict["ECOD"][trial.number] = ecod_clf
    return -1 * auc_score


def optimize_iso_forest(train_dict, test_dict, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_iso_forest_fn(trial, train_dict, test_dict),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
    )
    study.__setattr__("_best_model", trial_model_dict["ISOF"][study.best_trial.number])
    return study


def optimize_svm(train_dict, test_dict, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_svm_fn(trial, train_dict, test_dict),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
    )
    study.__setattr__("_best_model", trial_model_dict["SVM"][study.best_trial.number])
    return study


def optimize_svdd(train_dict, test_dict, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_deep_svdd_fn(trial, train_dict, test_dict),
        n_trials=n_trials,
        n_jobs=4,
    )
    study.__setattr__("_best_model", trial_model_dict["SVDD"][study.best_trial.number])
    return study


def optimize_ecod(train_dict, test_dict, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_ecod_fn(trial, train_dict, test_dict),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
    )
    study.__setattr__("_best_model", trial_model_dict["ECOD"][study.best_trial.number])
    return study
