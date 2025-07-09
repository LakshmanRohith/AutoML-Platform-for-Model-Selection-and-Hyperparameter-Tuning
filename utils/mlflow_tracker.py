import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.keras

def log_experiment(model_name, params, metrics, model, tuning_method):
    # ⚠️ DO NOT start a new run here — it's already started in app.py

    mlflow.log_param("model_name", model_name)
    mlflow.log_param("tuning_method", tuning_method)

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)

    # Log model depending on type
    if model_name == "SVM":
        mlflow.sklearn.log_model(model, "model")
    elif model_name == "XGBoost":
        mlflow.xgboost.log_model(model, "model")
    elif model_name == "DNN":
        mlflow.keras.log_model(model, "model")
