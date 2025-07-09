import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import xgboost as xgb
from tuning.grid_search import grid_search_tune
from tuning.optuna_search import optuna_tune
from tuning.bayesian_opt import bayesian_tune
import yaml

def train_xgboost(X_train, y_train, task_type, tuning_method):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    model = xgb.XGBClassifier() if task_type.lower() == "classification" else xgb.XGBRegressor()
    params = config["tuning"]["xgboost"]

    if tuning_method == "Grid Search":
        best_params = grid_search_tune(model, X_train, y_train, params["grid_search"])
    elif tuning_method == "Optuna":
        best_params = optuna_tune(model, X_train, y_train, params["optuna"], task_type)
    elif tuning_method == "Bayesian Optimization":
        best_params = bayesian_tune(model, X_train, y_train, params["bayesian"], task_type)
    else:
        best_params = {}  # Default fallback if tuning method is unrecognized

    model.set_params(**best_params)
    model.fit(X_train, y_train)
    return model, best_params

if __name__ == "__main__":
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    # Example config simulation
    config = {
        "tuning": {
            "xgboost": {
                "grid_search": {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
                "optuna": {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]},
                "bayesian": {"max_depth": [3, 5], "learning_rate": [0.01, 0.1], "n_iter": 5}
            }
        }
    }
    import io
    with io.StringIO(yaml.dump(config)) as f:
        with open("config.yaml", "w") as file:
            file.write(f.getvalue())
    
    model, best_params = train_xgboost(X, y, "classification", "Bayesian Optimization")
    print("Model trained with parameters:", best_params)