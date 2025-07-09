# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tuning.grid_search import grid_search_tune
# from tuning.optuna_search import optuna_tune
# from tuning.bayesian_opt import bayesian_tune
# import yaml

# def train_dnn(X_train, y_train, task_type, tuning_method):
#     with open("config.yaml", "r") as file:
#         config = yaml.safe_load(file)
    
#     def create_model(units, layers):
#         model = Sequential()
#         model.add(Dense(units, activation="relu", input_shape=(X_train.shape[1],)))
#         for _ in range(layers - 1):
#             model.add(Dense(units, activation="relu"))
#         model.add(Dense(1, activation="sigmoid" if task_type.lower() == "classification" else "linear"))
#         model.compile(optimizer="adam", loss="binary_crossentropy" if task_type.lower() == "classification" else "mse", metrics=["accuracy" if task_type.lower() == "classification" else "mae"])
#         return model

#     params = config["tuning"]["dnn"]
#     if tuning_method == "Grid Search":
#         best_params = grid_search_tune(create_model, X_train, y_train, params["grid_search"])
#     elif tuning_method == "Optuna":
#         best_params = optuna_tune(create_model, X_train, y_train, params["optuna"], task_type)
#     elif tuning_method == "Bayesian Optimization":
#         best_params = bayesian_tune(create_model, X_train, y_train, params["bayesian"], task_type)
#     else:
#         best_params = {"units": 64, "layers": 2}  # Default fallback if tuning method is unrecognized

#     model = create_model(**best_params)
#     model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
#     return model, best_params

# if __name__ == "__main__":
#     from sklearn.datasets import make_classification
#     import numpy as np
#     X, y = make_classification(n_samples=100, n_features=4, random_state=42)
#     y = np.where(y == 0, 0, 1)  # Binary classification
#     # Example config simulation
#     config = {
#         "tuning": {
#             "dnn": {
#                 "grid_search": {"units": [32, 64], "layers": [1, 2]},
#                 "optuna": {"units": [32, 64], "layers": [1, 2]},
#                 "bayesian": {"units": [32, 64], "layers": [1, 2], "n_iter": 5}
#             }
#         }
#     }
#     import io
#     with io.StringIO(yaml.dump(config)) as f:
#         with open("config.yaml", "w") as file:
#             file.write(f.getvalue())
    
#     model, best_params = train_dnn(X, y, "classification", "Bayesian Optimization")
#     print("Model trained with parameters:", best_params)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tuning.grid_search import grid_search_tune
from tuning.optuna_search import optuna_tune
from tuning.bayesian_opt import bayesian_tune
import yaml
import os
from tensorflow.keras.models import save_model

# ✅ 1. Make create_model globally accessible
def create_model(units, layers, input_dim, task_type):
    model = Sequential()
    model.add(Dense(units, activation="relu", input_shape=(input_dim,)))
    for _ in range(layers - 1):
        model.add(Dense(units, activation="relu"))
    model.add(Dense(1, activation="sigmoid" if task_type.lower() == "classification" else "linear"))
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy" if task_type.lower() == "classification" else "mse",
        metrics=["accuracy" if task_type.lower() == "classification" else "mae"]
    )
    return model

def train_dnn(X_train, y_train, task_type, tuning_method):
    

    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)

    params = config["tuning"]["dnn"]

    # ✅ Wrap create_model to fix input_dim for tuning functions
    model_wrapper = lambda units, layers: create_model(units, layers, X_train.shape[1], task_type)

    if tuning_method == "Grid Search":
        best_params = grid_search_tune(model_wrapper, X_train, y_train, params["grid_search"], task_type)
    elif tuning_method == "Optuna":
        best_params = optuna_tune(model_wrapper, X_train, y_train, params["optuna"], task_type)
    elif tuning_method == "Bayesian Optimization":
        best_params = bayesian_tune(model_wrapper, X_train, y_train, params["bayesian"], task_type)
    else:
        best_params = {"units": 64, "layers": 2}  # Default fallback

    model = create_model(best_params["units"], best_params["layers"], X_train.shape[1], task_type)
    model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

    # ✅ SAVE the trained model to correct location for Streamlit to load later
    os.makedirs("artifacts", exist_ok=True)
    model.save("artifacts/DNN_model")

    return model, best_params
# ✅ 3. For standalone testing
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    import numpy as np

    X, y = make_classification(n_samples=100, n_features=4, random_state=42)
    y = np.where(y == 0, 0, 1)

    # Fake config just for test
    config = {
        "tuning": {
            "dnn": {
                "grid_search": {"units": [32, 64], "layers": [1, 2]},
                "optuna": {"units": [32, 64], "layers": [1, 2]},
                "bayesian": {"units": [32, 64], "layers": [1, 2], "n_iter": 5}
            }
        }
    }

    import io
    with io.StringIO(yaml.dump(config)) as f:
        with open("config.yaml", "w") as file:
            file.write(f.getvalue())

    model, best_params = train_dnn(X, y, "classification", "Bayesian Optimization")
    print("Model trained with parameters:", best_params)
