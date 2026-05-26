from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATASET_PATH = Path("data/processed/model_dataset_v1.csv")
METRICS_PATH = Path("results/metrics/model_v1_metrics.csv")
CONFUSION_MATRIX_PATH = Path("results/metrics/model_v1_confusion_matrix.csv")
BALANCED_METRICS_PATH = Path("results/metrics/model_v1_balanced_metrics.csv")
BALANCED_CONFUSION_MATRIX_PATH = Path("results/metrics/model_v1_balanced_confusion_matrix.csv")
TUNED_METRICS_PATH = Path("results/metrics/model_v1_tuned_draw_weight_metrics.csv")
TUNED_CONFUSION_MATRIX_PATH = Path("results/metrics/model_v1_tuned_draw_weight_confusion_matrix.csv")
TUNED_CANDIDATES_PATH = Path("results/metrics/model_v1_tuned_draw_weight_candidates.csv")
VALIDATION_START_DATE = "2019-01-01"
TEST_START_DATE = "2022-01-01"

BASE_FEATURE_COLUMNS = [
    "elo_difference",
    "home_advantage",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
]

RECENT_FORM_FEATURE_COLUMNS = [
    "recent_points_difference",
    "home_recent_points_avg",
    "away_recent_points_avg",
    "home_recent_draw_rate",
    "away_recent_draw_rate",
    "recent_goal_difference_difference",
]

FEATURE_COLUMNS = BASE_FEATURE_COLUMNS

DRAW_WEIGHT_CANDIDATES = [round(1.0 + step * 0.1, 1) for step in range(11)]
DEFAULT_DRAW_WEIGHT = 1.6


def train_model(
    dataset_path=DATASET_PATH,
    metrics_path=METRICS_PATH,
    confusion_matrix_path=CONFUSION_MATRIX_PATH,
    validation_start_date=VALIDATION_START_DATE,
    test_start_date=TEST_START_DATE,
    class_weight=None,
    feature_columns=FEATURE_COLUMNS,
):
    dataset = pd.read_csv(dataset_path)
    dataset["date"] = pd.to_datetime(dataset["date"], errors="coerce")
    dataset = dataset.dropna(subset=["date", "result", *feature_columns])

    validation_start = pd.to_datetime(validation_start_date)
    test_start = pd.to_datetime(test_start_date)

    train_data = dataset[dataset["date"] < validation_start]
    validation_data = dataset[
        (dataset["date"] >= validation_start)
        & (dataset["date"] < test_start)
    ]
    test_data = dataset[dataset["date"] >= pd.to_datetime(test_start_date)]

    if train_data.empty or validation_data.empty or test_data.empty:
        raise ValueError("Train, validation, and test sets must all contain rows")

    x_train = train_data[feature_columns]
    y_train = train_data["result"]
    x_validation = validation_data[feature_columns]
    y_validation = validation_data["result"]
    x_test = test_data[feature_columns]
    y_test = test_data["result"]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000, class_weight=class_weight),
    )
    model.fit(x_train, y_train)

    labels = list(model.classes_)
    validation_predictions = model.predict(x_validation)
    validation_probabilities = model.predict_proba(x_validation)
    test_predictions = model.predict(x_test)
    test_probabilities = model.predict_proba(x_test)

    metrics = {
        "train_rows": len(train_data),
        "validation_rows": len(validation_data),
        "test_rows": len(test_data),
        "validation_start_date": validation_start_date,
        "test_start_date": test_start_date,
        "validation_accuracy": accuracy_score(y_validation, validation_predictions),
        "validation_log_loss": log_loss(y_validation, validation_probabilities, labels=labels),
        "validation_macro_precision": precision_score(
            y_validation, validation_predictions, average="macro", zero_division=0
        ),
        "validation_macro_recall": recall_score(
            y_validation, validation_predictions, average="macro", zero_division=0
        ),
        "validation_macro_f1": f1_score(
            y_validation, validation_predictions, average="macro", zero_division=0
        ),
        "validation_weighted_precision": precision_score(
            y_validation, validation_predictions, average="weighted", zero_division=0
        ),
        "validation_weighted_recall": recall_score(
            y_validation, validation_predictions, average="weighted", zero_division=0
        ),
        "validation_weighted_f1": f1_score(
            y_validation, validation_predictions, average="weighted", zero_division=0
        ),
        "validation_roc_auc_ovr": roc_auc_score(
            y_validation, validation_probabilities, multi_class="ovr", labels=labels
        ),
        "test_accuracy": accuracy_score(y_test, test_predictions),
        "test_log_loss": log_loss(y_test, test_probabilities, labels=labels),
        "test_macro_precision": precision_score(
            y_test, test_predictions, average="macro", zero_division=0
        ),
        "test_macro_recall": recall_score(
            y_test, test_predictions, average="macro", zero_division=0
        ),
        "test_macro_f1": f1_score(
            y_test, test_predictions, average="macro", zero_division=0
        ),
        "test_weighted_precision": precision_score(
            y_test, test_predictions, average="weighted", zero_division=0
        ),
        "test_weighted_recall": recall_score(
            y_test, test_predictions, average="weighted", zero_division=0
        ),
        "test_weighted_f1": f1_score(
            y_test, test_predictions, average="weighted", zero_division=0
        ),
        "test_roc_auc_ovr": roc_auc_score(
            y_test, test_probabilities, multi_class="ovr", labels=labels
        ),
    }

    metrics_path = Path(metrics_path)
    confusion_matrix_path = Path(confusion_matrix_path)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_matrix_path.parent.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    matrix = confusion_matrix(y_test, test_predictions, labels=labels)
    matrix_df = pd.DataFrame(matrix, index=labels, columns=labels)
    matrix_df.to_csv(confusion_matrix_path)

    return metrics


def train_balanced_model(
    dataset_path=DATASET_PATH,
    metrics_path=BALANCED_METRICS_PATH,
    confusion_matrix_path=BALANCED_CONFUSION_MATRIX_PATH,
    validation_start_date=VALIDATION_START_DATE,
    test_start_date=TEST_START_DATE,
):
    return train_model(
        dataset_path=dataset_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        validation_start_date=validation_start_date,
        test_start_date=test_start_date,
        class_weight="balanced",
    )


def train_tuned_draw_weight_model(
    dataset_path=DATASET_PATH,
    metrics_path=TUNED_METRICS_PATH,
    confusion_matrix_path=TUNED_CONFUSION_MATRIX_PATH,
    candidates_path=TUNED_CANDIDATES_PATH,
    validation_start_date=VALIDATION_START_DATE,
    test_start_date=TEST_START_DATE,
    draw_weight_candidates=DRAW_WEIGHT_CANDIDATES,
    target_draw_weight=DEFAULT_DRAW_WEIGHT,
):
    candidate_results = []

    for draw_weight in draw_weight_candidates:
        class_weight = {
            "away_win": 1.0,
            "draw": draw_weight,
            "home_win": 1.0,
        }
        metrics = train_model(
            dataset_path=dataset_path,
            metrics_path=Path("/tmp/model_v1_tuned_candidate_metrics.csv"),
            confusion_matrix_path=Path("/tmp/model_v1_tuned_candidate_confusion_matrix.csv"),
            validation_start_date=validation_start_date,
            test_start_date=test_start_date,
            class_weight=class_weight,
        )
        metrics["draw_weight"] = draw_weight
        candidate_results.append(metrics)

    candidates = pd.DataFrame(candidate_results)
    candidates_path = Path(candidates_path)
    candidates_path.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_path, index=False)

    if target_draw_weight is None:
        target_draw_weight = candidates.sort_values(
            ["validation_log_loss", "validation_macro_f1"],
            ascending=[True, False],
        ).iloc[0]["draw_weight"]
    elif target_draw_weight not in list(candidates["draw_weight"]):
        raise ValueError("target_draw_weight must be included in draw_weight_candidates")

    best_metrics = train_model(
        dataset_path=dataset_path,
        metrics_path=metrics_path,
        confusion_matrix_path=confusion_matrix_path,
        validation_start_date=validation_start_date,
        test_start_date=test_start_date,
        class_weight={
            "away_win": 1.0,
            "draw": float(target_draw_weight),
            "home_win": 1.0,
        },
    )
    best_metrics["draw_weight"] = float(target_draw_weight)
    pd.DataFrame([best_metrics]).to_csv(metrics_path, index=False)

    return best_metrics


def main():
    metrics = train_model()
    balanced_metrics = train_balanced_model()
    tuned_metrics = train_tuned_draw_weight_model()

    print("Logistic Regression v1")
    print(f"Validation accuracy: {metrics['validation_accuracy']:.3f}")
    print(f"Validation macro F1: {metrics['validation_macro_f1']:.3f}")
    print(f"Validation log loss: {metrics['validation_log_loss']:.3f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.3f}")
    print(f"Test macro F1: {metrics['test_macro_f1']:.3f}")
    print(f"Test log loss: {metrics['test_log_loss']:.3f}")
    print(f"Saved metrics to {METRICS_PATH}")
    print(f"Saved confusion matrix to {CONFUSION_MATRIX_PATH}")

    print("\nBalanced Logistic Regression v1")
    print(f"Validation accuracy: {balanced_metrics['validation_accuracy']:.3f}")
    print(f"Validation macro F1: {balanced_metrics['validation_macro_f1']:.3f}")
    print(f"Validation log loss: {balanced_metrics['validation_log_loss']:.3f}")
    print(f"Test accuracy: {balanced_metrics['test_accuracy']:.3f}")
    print(f"Test macro F1: {balanced_metrics['test_macro_f1']:.3f}")
    print(f"Test log loss: {balanced_metrics['test_log_loss']:.3f}")
    print(f"Saved metrics to {BALANCED_METRICS_PATH}")
    print(f"Saved confusion matrix to {BALANCED_CONFUSION_MATRIX_PATH}")

    print("\nTuned Draw-Weight Logistic Regression v1")
    print(f"Draw weight: {tuned_metrics['draw_weight']:.2f}")
    print(f"Validation accuracy: {tuned_metrics['validation_accuracy']:.3f}")
    print(f"Validation macro F1: {tuned_metrics['validation_macro_f1']:.3f}")
    print(f"Validation log loss: {tuned_metrics['validation_log_loss']:.3f}")
    print(f"Test accuracy: {tuned_metrics['test_accuracy']:.3f}")
    print(f"Test macro F1: {tuned_metrics['test_macro_f1']:.3f}")
    print(f"Test log loss: {tuned_metrics['test_log_loss']:.3f}")
    print(f"Saved metrics to {TUNED_METRICS_PATH}")
    print(f"Saved confusion matrix to {TUNED_CONFUSION_MATRIX_PATH}")


if __name__ == "__main__":
    main()
