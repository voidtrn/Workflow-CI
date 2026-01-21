from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path
import mlflow.sklearn
import pandas as pd
import numpy as np
import argparse
import mlflow
import os

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student-performance_preprocessing")
    parser.add_argument("--target", default="G3")
    args = parser.parse_args()

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    os.environ.pop("MLFLOW_RUN_ID", None)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train_processed.csv"
    test_path = data_dir / "test_processed.csv"

    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"Missing processed files in {data_dir.resolve()}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X_train = train_df.drop(columns=[args.target])
    y_train = train_df[args.target]
    X_test = test_df.drop(columns=[args.target])
    y_test = test_df[args.target]

    mlflow.set_experiment("Student Performance CI")

    with mlflow.start_run(run_name="linear_regression_train"):
        mlflow.autolog()

        model = LinearRegression()
        model.fit(X_train, y_train)
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train.head(5),
            registered_model_name="student_performance_linear_regression"
        )

        model.fit(X_train, y_train)
    
        pred_test = model.predict(X_test)

        metrics = {
            "test_mean_squared_error": float(mean_absolute_error(y_test, pred_test)),
            "test_mean_absolute_error": rmse(y_test, pred_test),
            "test_r2_score": float(r2_score(y_test, pred_test)),
        }
        mlflow.log_metrics(metrics)

if __name__ == "__main__":
    main()