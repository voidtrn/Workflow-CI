from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import mlflow

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="student-performance_preprocessing")
    parser.add_argument("--target", default="G3")
    parser.add_argument("--model", default="linear")
    args = parser.parse_args()

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
    mlflow.sklearn.autolog(silent=True)

    model = LinearRegression()

    mlflow.log_param("target", args.target)
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_param("n_train", X_train.shape[0])
    mlflow.log_param("n_test", X_test.shape[0])
    mlflow.log_param("n_features", X_train.shape[1])

    model.fit(X_train, y_train)
    pred_test = model.predict(X_test)

    mlflow.log_metrics({
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "test_rmse": rmse(y_test, pred_test),
        "test_r2": float(r2_score(y_test, pred_test)),
    })

    mlflow.sklearn.log_model(model, artifact_path="model")


if __name__ == "__main__":
    main()