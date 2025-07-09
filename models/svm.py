from sklearn.svm import SVC, SVR
import yaml
import os
import joblib
from tuning.grid_search import grid_search_tune
from tuning.optuna_search import optuna_tune
from tuning.bayesian_opt import bayesian_tune

def create_svm(C, kernel, task_type):
    if task_type.lower() == "classification":
        return SVC(C=C, kernel=kernel, probability=True)
    else:
        return SVR(C=C, kernel=kernel)

def train_svm(X_train, y_train, task_type, tuning_method):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = config["tuning"]["svm"]

    def model_wrapper(**kwargs):
        C = kwargs.get("C", 1.0)
        kernel = kwargs.get("kernel", "rbf")
        return create_svm(C, kernel, task_type)

    if tuning_method == "Grid Search":
        best_params = grid_search_tune(model_wrapper, X_train, y_train, params["grid_search"], task_type)
    elif tuning_method == "Optuna":
        best_params = optuna_tune(model_wrapper, X_train, y_train, params["optuna"], task_type)
    elif tuning_method == "Bayesian Optimization":
        best_params = bayesian_tune(model_wrapper, X_train, y_train, params["bayesian"], task_type)
    else:
        best_params = {"C": 1.0, "kernel": "rbf"}

    model = create_svm(best_params["C"], best_params["kernel"], task_type)
    model.fit(X_train, y_train)

    os.makedirs("artifacts", exist_ok=True)
    joblib.dump(model, "artifacts/SVM_model.pkl")

    return model, best_params
