import torch
import optuna
from optuna import Trial
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


def optimize_iso_forest_fn(trial, transformed_X, labels):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    max_features = trial.suggest_uniform("max_features", 0.2, 0.95)
    contamination = trial.suggest_uniform("contamination", 0.1, 0.5)
    X_train, X_val, y_train, y_val = train_test_split(
        transformed_X, labels, random_state=0
    )
    iso_forest_clf = IsolationForest(
        random_state=42,
        n_estimators=n_estimators,
        max_features=max_features,
        contamination=contamination,
        n_jobs=-1,
    )
    iso_forest_clf.fit(X_train)
    y_pred = iso_forest_clf.predict(X_val)

    auc_score = roc_auc_score(y_val, y_pred)

    return -1 * auc_score


def optimize_iso_forest(transformed_X, labels):
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: optimize_iso_forest_fn(trial, transformed_X, labels), n_trials=50
    )
    return study

