import optuna
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from xgboost import XGBClassifier, XGBRegressor

def optuna_tune(model_func, X_train, y_train, params, task_type):
    def objective(trial):
        # ✅ DNN Case
        if "units" in params and "layers" in params:
            units = trial.suggest_int("units", *params["units"])
            layers = trial.suggest_int("layers", *params["layers"])
            
            # ✅ FIX: Use keyword arguments
            model = model_func(units=units, layers=layers)
            model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)
            
            if task_type.lower() == "classification":
                score = model.evaluate(X_train, y_train, verbose=0)[1]  # accuracy
            else:
                score = -model.evaluate(X_train, y_train, verbose=0)[1]  # minimize MAE
            return score

        # ✅ SVM Case
        elif "C" in params and "kernel" in params:
            C = trial.suggest_float("C", *params["C"], log=True)
            kernel = trial.suggest_categorical("kernel", params["kernel"])
            model = SVC(C=C, kernel=kernel) if task_type == "classification" else SVR(C=C, kernel=kernel)
            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            if task_type.lower() == "classification":
                return accuracy_score(y_train, preds)
            else:
                return -mean_squared_error(y_train, preds)

        # ✅ XGBoost Case
        elif "learning_rate" in params and "max_depth" in params:
            learning_rate = trial.suggest_float("learning_rate", *params["learning_rate"], log=True)
            max_depth = trial.suggest_int("max_depth", *params["max_depth"])
            
            if task_type.lower() == "classification":
                model = XGBClassifier(learning_rate=learning_rate, max_depth=max_depth,
                                      use_label_encoder=False, eval_metric="logloss")
            else:
                model = XGBRegressor(learning_rate=learning_rate, max_depth=max_depth)

            model.fit(X_train, y_train)
            preds = model.predict(X_train)
            if task_type.lower() == "classification":
                return accuracy_score(y_train, preds)
            else:
                return -mean_squared_error(y_train, preds)

        else:
            raise ValueError("Unsupported model or missing hyperparameters")

    # 🔁 Run the Optuna optimization
    study = optuna.create_study(direction="maximize" if task_type.lower() == "classification" else "minimize")
    study.optimize(objective, n_trials=params.get("n_trials", 10))
    return study.best_params
