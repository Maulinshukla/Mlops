"""
Model Building with Experimentation Tracking
--------------------------------------------
1. Loads the train / test splits produced by the data-prep job (workflow artifact).
2. Defines six candidate models (Decision Tree, Bagging, Random Forest, AdaBoost,
   Gradient Boosting, XGBoost) with hyper-parameter grids.
3. Tunes each model with 5-fold stratified GridSearchCV (scoring = F1).
4. Logs EVERY tuned parameter combination to MLflow as a nested run, plus the
   best parameters and train/test metrics on the parent run.
5. Selects the best model on cross-validated F1 (the test set is used only for
   final evaluation, never for selection) and saves the full pipeline
   (preprocessing + model) to tourism_project/deployment/ so the workflow can
   commit it to the repository.
"""
import logging
import os
import time
import warnings
from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

RANDOM_STATE = 42

# Keep CI / notebook logs readable
logging.getLogger("mlflow").setLevel(logging.WARNING)
os.environ.setdefault("MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT", "true")  # no per-run URLs
warnings.filterwarnings("ignore")

MODEL_DIR = Path("tourism_project/deployment")
MODEL_PATH = MODEL_DIR / "best_tourism_model_v1.joblib"

# ---------------- MLflow setup ----------------
# In GitHub Actions the workflow starts an MLflow server on port 5000 and sets
# MLFLOW_TRACKING_URI; locally we fall back to a file store (./mlruns).
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
mlflow.set_experiment("tourism-wellness-package")

# ---------------- Load splits (workflow artifact) ----------------
Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").squeeze("columns")
ytest = pd.read_csv("ytest.csv").squeeze("columns")
print(f"Train {Xtrain.shape} | Test {Xtest.shape}")

# ---------------- Preprocessing ----------------
categorical_features = ["TypeofContact", "Occupation", "Gender", "ProductPitched",
                        "MaritalStatus", "Designation"]
numeric_features = [c for c in Xtrain.columns if c not in categorical_features]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown="ignore"), categorical_features),
)

# Class imbalance (~19% buyers): weight the minority class
scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()

# ---------------- Candidate models + hyper-parameter grids ----------------
candidates = {
    "DecisionTree": (
        DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
        {"model__max_depth": [4, 6, 8, None],
         "model__min_samples_leaf": [1, 5, 10]},
    ),
    "Bagging": (
        BaggingClassifier(estimator=DecisionTreeClassifier(class_weight="balanced"),
                          random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [50, 100],
         "model__max_samples": [0.7, 1.0],
         "model__max_features": [0.7, 1.0]},
    ),
    "RandomForest": (
        RandomForestClassifier(class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [10, 15, None],
         "model__min_samples_leaf": [1, 3]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(estimator=DecisionTreeClassifier(class_weight="balanced"),
                           random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.1, 0.5, 1.0],
         "model__estimator__max_depth": [1, 2]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=RANDOM_STATE),
        {"model__n_estimators": [100, 200],
         "model__learning_rate": [0.05, 0.1],
         "model__max_depth": [3, 5],
         "model__subsample": [0.8, 1.0]},
    ),
    "XGBoost": (
        XGBClassifier(scale_pos_weight=scale_pos_weight, eval_metric="logloss",
                      random_state=RANDOM_STATE, n_jobs=-1),
        {"model__n_estimators": [100, 200],
         "model__max_depth": [3, 5, 7],
         "model__learning_rate": [0.05, 0.1],
         "model__subsample": [0.8, 1.0]},
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)


def evaluate(model, X, y):
    """Classification metrics for a fitted pipeline."""
    pred = model.predict(X)
    proba = model.predict_proba(X)[:, 1]
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred),
        "recall": recall_score(y, pred),
        "f1": f1_score(y, pred),
        "roc_auc": roc_auc_score(y, proba),
    }


results, fitted = [], {}

for name, (estimator, grid) in candidates.items():
    start = time.time()
    pipe = Pipeline([("preprocessor", preprocessor), ("model", estimator)])
    search = GridSearchCV(pipe, grid, cv=cv, scoring="f1", n_jobs=-1)

    with mlflow.start_run(run_name=name):
        search.fit(Xtrain, ytrain)

        # --- Log EVERY parameter combination that was tuned (nested runs) ---
        cvres = search.cv_results_
        for i, params in enumerate(cvres["params"]):
            with mlflow.start_run(run_name=f"{name}_combo_{i}", nested=True):
                mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
                mlflow.log_metric("cv_mean_f1", cvres["mean_test_score"][i])
                mlflow.log_metric("cv_std_f1", cvres["std_test_score"][i])

        # --- Best configuration for this algorithm ---
        best = search.best_estimator_
        train_m = evaluate(best, Xtrain, ytrain)
        test_m = evaluate(best, Xtest, ytest)

        mlflow.log_param("algorithm", name)
        mlflow.log_params({f"best_{k.replace('model__', '')}": v
                           for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_f1", search.best_score_)
        mlflow.log_metrics({f"train_{k}": v for k, v in train_m.items()})
        mlflow.log_metrics({f"test_{k}": v for k, v in test_m.items()})

    fitted[name] = best
    results.append({"model": name, "cv_f1": search.best_score_,
                    **{f"train_{k}": v for k, v in train_m.items()},
                    **{f"test_{k}": v for k, v in test_m.items()},
                    "best_params": {k.replace("model__", ""): v
                                    for k, v in search.best_params_.items()},
                    "fit_seconds": round(time.time() - start, 1)})
    print(f"{name:<17} combos={len(cvres['params']):>2}  cv_f1={search.best_score_:.3f}  "
          f"test_f1={test_m['f1']:.3f}  test_recall={test_m['recall']:.3f}  "
          f"({time.time() - start:.0f}s)")

# ---------------- Compare & select ----------------
comparison = pd.DataFrame(results).sort_values("cv_f1", ascending=False)
pd.set_option("display.width", 200)
print("\nModel comparison (sorted by cross-validated F1):")
print(comparison[["model", "cv_f1", "train_f1", "test_f1", "test_precision",
                  "test_recall", "test_roc_auc", "test_accuracy"]].round(3).to_string(index=False))
comparison.to_csv("model_comparison.csv", index=False)

best_name = comparison.iloc[0]["model"]
best_model = fitted[best_name]
print(f"\nBest model: {best_name}")
print(f"Best params: {comparison.iloc[0]['best_params']}")
print("\nClassification report on held-out test set:")
print(classification_report(ytest, best_model.predict(Xtest), digits=3))

# ---------------- Save best model for deployment ----------------
MODEL_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(best_model, MODEL_PATH)

with mlflow.start_run(run_name=f"BEST_{best_name}"):
    mlflow.log_param("selected_model", best_name)
    mlflow.log_params({k: v for k, v in comparison.iloc[0]["best_params"].items()})
    mlflow.log_metrics({k: float(comparison.iloc[0][k]) for k in comparison.columns
                        if k.startswith(("test_", "cv_"))})
    mlflow.log_artifact(str(MODEL_PATH))
    mlflow.log_artifact("model_comparison.csv")

print(f"Best model saved to {MODEL_PATH}")
