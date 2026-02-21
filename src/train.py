import joblib
import mlflow
import mlflow.sklearn

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from src.preprocess import load_data, clean_data
    from src.config import DATA_PATH, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_PATH
except ModuleNotFoundError:
    from preprocess import load_data, clean_data
    from config import DATA_PATH, MLFLOW_EXPERIMENT, MLFLOW_TRACKING_URI, MODEL_PATH


def build_pipeline(df):
    X = df.drop("is_cancelled", axis=1)
    y = df["is_cancelled"]

    categorical_cols = X.select_dtypes(include=["object"]).columns
    numerical_cols = X.select_dtypes(include=["int64", "float64"]).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
            ("num", "passthrough", numerical_cols)
        ]
    )

    model = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        class_weight="balanced_subsample",
        n_jobs=-1,
        min_samples_leaf=2,
    )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    return pipeline, X_train, X_test, y_train, y_test


def main() -> None:
    df = load_data(DATA_PATH)
    df = clean_data(df)
    pipeline, X_train, X_test, y_train, y_test = build_pipeline(df)

    pipeline.fit(X_train, y_train)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    with mlflow.start_run():
        y_pred = pipeline.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
        }

        y_proba = pipeline.predict_proba(X_test)[:, 1]
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)
        mlflow.log_metrics(metrics)
        mlflow.log_params(
            {
                "model_type": "RandomForestClassifier",
                "n_estimators": 300,
                "min_samples_leaf": 2,
                "class_weight": "balanced_subsample",
                "train_rows": len(X_train),
                "test_rows": len(X_test),
                "feature_count": X_train.shape[1],
            }
        )
        mlflow.sklearn.log_model(pipeline, name="model")

    joblib.dump(pipeline, MODEL_PATH)
    print(f"Model trained and saved to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
