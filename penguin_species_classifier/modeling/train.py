import argparse
import json
import math
import random

from matplotlib import pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from penguin_species_classifier.config import RAW_DATA_DIR, REPORTS_DIR
from penguin_species_classifier.dataset import clean_data, load_data, split_data, split_features_target
from penguin_species_classifier.features import build_preprocessor
from penguin_species_classifier.modeling.evaluate import evaluate_predictions


def positive_float(value):
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Expected a positive number.") from exc
    
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("Expected a finite number greater than 0.")
    
    return number


def parse_gamma(value):
    if value in {"scale", "auto"}:
        return value
    
    return positive_float(value)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--C", "--c", dest="C", type=positive_float,
                        default=1.0)
    
    parser.add_argument("--kernel", choices=["linear", "poly", "rbf", "sigmoid"],
                        default="rbf")
    parser.add_argument("--gamma", type=parse_gamma, default="scale")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_name", type=str, default="svc")
    parser.add_argument("--model_name", type=str, default="penguin_species_svc")
    args = parser.parse_args(argv)

    if not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must be between 0 and 4294967295.")
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def build_pipeline(C, kernel, gamma, seed):
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("classifier", SVC(
            C=C,
            kernel=kernel,
            gamma=gamma,
            class_weight=None,
            random_state=seed,
        )),
    ])


def main():
    args = parse_args()

    set_seed(args.seed)
    mlflow.set_experiment("three_class_svc")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    with mlflow.start_run(run_name=args.run_name) as run:
        params = {
            "C": args.C,
            "kernel": args.kernel,
            "gamma": args.gamma,
            "seed": args.seed,
            "class_weight": "None",
        }

        for name, value in params.items():
            mlflow.log_param(name, value)

        data = clean_data(load_data(f"{RAW_DATA_DIR}/data.csv"))
        X, y = split_features_target(data)
        X_train, X_test, y_train, y_test = split_data(
            X, y, seed=args.seed, test_size=0.2
        )

        pipeline = build_pipeline(args.C, args.kernel, args.gamma, args.seed)
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        metrics = {
            name: float(value)
            for name, value in evaluate_predictions(y_test, predictions).items()
        }
        for name, value in metrics.items():
            mlflow.log_metric(name, value)

        output_dir = REPORTS_DIR / run.info.run_id
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2, sort_keys=True), encoding="utf-8"
        )
        mlflow.log_artifact(str(metrics_path), artifact_path="evaluation")

        display = ConfusionMatrixDisplay.from_predictions(
            y_test, predictions, labels=pipeline.classes_
        )
        figure_path = output_dir / "test_confusion_matrix.png"

        display.figure_.savefig(figure_path, bbox_inches="tight")
     
        plt.close(display.figure_)
        mlflow.log_artifact(str(figure_path), artifact_path="evaluation")

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name=args.model_name,
            input_example=X_test.head(5),
        )


if __name__ == "__main__":
    main()
