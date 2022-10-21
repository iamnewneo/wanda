import optuna
from optuna import Trial
from sklearn.svm import OneClassSVM
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from wanda.model.od_algos import DeepSVDDModel, ECODModel
from wanda.model.isolation_forest import IsoForestModel
from wanda.model.svm import SVMModel
from wanda import config

optuna.logging.set_verbosity(optuna.logging.WARNING)


def callback(study, trial):
    if study.best_trial.number == trial.number:
        study.set_user_attr(key="best_model", value=trial.user_attrs["best_model"])


def optimize_iso_forest_fn(trial, transformed_X, labels):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_features = trial.suggest_uniform("max_features", 0.2, 0.95)
    contamination = trial.suggest_uniform("contamination", 0.1, 0.5)
    iso_forest_clf = IsoForestModel(
        random_state=42,
        n_estimators=n_estimators,
        max_features=max_features,
        contamination=contamination,
        n_jobs=config.N_JOBS,
    )
    scores = cross_val_score(
        iso_forest_clf, transformed_X, labels, cv=5, scoring="roc_auc"
    )
    auc_score = scores.mean()
    trial.set_user_attr(key="best_model", value=iso_forest_clf)
    return -1 * auc_score


def optimize_svm_fn(trial, transformed_X, labels):
    svm_clf = None
    nu = trial.suggest_uniform("nu", 0.05, 0.95)
    power_t = trial.suggest_uniform("power_t", 0.1, 0.9)
    svm_clf = SVMModel(nu=nu, power_t=power_t)
    scores = cross_val_score(svm_clf, transformed_X, labels, cv=5, scoring="roc_auc")
    auc_score = scores.mean()
    trial.set_user_attr(key="best_model", value=svm_clf)
    return -1 * auc_score


def optimize_deep_svdd_fn(trial, transformed_X, labels):
    contamination = trial.suggest_uniform("contamination", 0.1, 0.45)
    svdd_clf = DeepSVDDModel(contamination=contamination)
    scores = cross_val_score(svdd_clf, transformed_X, labels, cv=5, scoring="roc_auc")
    auc_score = scores.mean()
    trial.set_user_attr(key="best_model", value=svdd_clf)
    return -1 * auc_score


def optimize_ecod_fn(trial, transformed_X, labels):
    contamination = trial.suggest_uniform("contamination", 0.1, 0.45)
    ecod_clf = ECODModel(contamination=contamination)
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
        callbacks=[callback],
    )
    return study


def optimize_svm(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_svm_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
        callbacks=[callback],
    )
    return study


def optimize_svdd(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_deep_svdd_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=4,
        callbacks=[callback],
    )
    return study


def optimize_ecod(transformed_X, labels, n_trials):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_ecod_fn(trial, transformed_X, labels),
        n_trials=n_trials,
        n_jobs=config.N_JOBS,
        callbacks=[callback],
    )
    return study
