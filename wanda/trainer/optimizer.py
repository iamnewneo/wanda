import optuna
from optuna import Trial
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from wanda.model.od_algos import DeepSVDDModel, ECODModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.utils.util import switch_labels
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


def optimize_iso_forest_fn(trial, transformed_X, labels):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_features = trial.suggest_uniform("max_features", 0.2, 0.95)
    contamination = trial.suggest_uniform("contamination", 0.1, 0.5)
    scores = []
    k_fold = StratifiedKFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(transformed_X, labels):
        iso_forest_clf = IsoForestModel(
            random_state=42,
            n_estimators=n_estimators,
            max_features=max_features,
            contamination=contamination,
            n_jobs=config.N_JOBS,
        )
        iso_forest_clf.fit(transformed_X[train_indices], labels[train_indices])
        y_pred = iso_forest_clf.decision_function(transformed_X[test_indices])
        scores.append(roc_auc_score(switch_labels(labels[test_indices]), y_pred))
    auc_score = sum(scores) / len(scores)
    trial_model_dict["ISOF"][trial.number] = iso_forest_clf
    return -1 * auc_score


def optimize_svm_fn(trial, transformed_X, labels):
    nu = trial.suggest_uniform("nu", 0.05, 0.95)
    power_t = trial.suggest_uniform("power_t", 0.1, 0.9)
    scores = []
    k_fold = StratifiedKFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(transformed_X, labels):
        svm_clf = SVMModel(nu=nu, power_t=power_t)
        svm_clf.fit(transformed_X[train_indices], labels[train_indices])
        y_pred = svm_clf.decision_function(transformed_X[test_indices])
        scores.append(roc_auc_score(switch_labels(labels[test_indices]), y_pred))
    auc_score = sum(scores) / len(scores)
    trial_model_dict["SVM"][trial.number] = svm_clf
    return -1 * auc_score


def optimize_deep_svdd_fn(trial, transformed_X, labels):
    contamination = trial.suggest_uniform("contamination", 0.1, 0.45)
    scores = []
    k_fold = StratifiedKFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(transformed_X, labels):
        svdd_clf = DeepSVDDModel(contamination=contamination)
        svdd_clf.fit(transformed_X[train_indices], labels[train_indices])
        y_pred = svdd_clf.decision_function(transformed_X[test_indices])
        scores.append(roc_auc_score(switch_labels(labels[test_indices]), y_pred))
    auc_score = sum(scores) / len(scores)
    trial_model_dict["SVDD"][trial.number] = svdd_clf
    return -1 * auc_score


def optimize_ecod_fn(trial, transformed_X, labels):
    contamination = trial.suggest_uniform("contamination", 0.1, 0.45)
    scores = []
    k_fold = StratifiedKFold(n_splits=5)
    for train_indices, test_indices in k_fold.split(transformed_X, labels):
        ecod_clf = ECODModel(contamination=contamination)
        ecod_clf.fit(transformed_X[train_indices], labels[train_indices])
        y_pred = ecod_clf.decision_function(transformed_X[test_indices])
        scores.append(roc_auc_score(switch_labels(labels[test_indices]), y_pred))
    auc_score = sum(scores) / len(scores)
    trial_model_dict["ECOD"][trial.number] = ecod_clf
    scores = cross_val_score(ecod_clf, transformed_X, labels, cv=5, scoring="roc_auc")
    auc_score = scores.mean()
    trial.set_user_attr(key="best_model", value=ecod_clf)
    return -1 * auc_score


def optimize_iso_forest(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_iso_forest_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
    )
    study.__setattr__("_best_model", trial_model_dict["ISOF"][study.best_trial.number])
    return study


def optimize_svm(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_svm_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
    )
    study.__setattr__("_best_model", trial_model_dict["SVM"][study.best_trial.number])
    return study


def optimize_svdd(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_deep_svdd_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=4,
    )
    study.__setattr__("_best_model", trial_model_dict["SVDD"][study.best_trial.number])
    return study


def optimize_ecod(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_ecod_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
    )
    study.__setattr__("_best_model", trial_model_dict["ECOD"][study.best_trial.number])
    return study
