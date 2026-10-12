from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
)


def evaluate_predictions(y_true, predictions):
    return {
        "test_accuracy": accuracy_score(y_true, predictions),
        "test_balanced_accuracy": balanced_accuracy_score(
            y_true, predictions
        ),
        "test_precision": precision_score(
            y_true, predictions, average="macro", zero_division=0
        ),
        "test_recall": recall_score(
            y_true, predictions, average="macro", zero_division=0
        ),
    }