from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np

# Wrapper class for Keras models to work with GridSearchCV
class KerasModelWrapper(BaseEstimator):
    def __init__(self, model_fn=None, task_type="classification", **params):
        self.model_fn = model_fn
        self.task_type = task_type
        self.params = params
        self.model = None

    def set_params(self, **params):
        if "model_fn" in params:
            self.model_fn = params.pop("model_fn")
        if "task_type" in params:
            self.task_type = params.pop("task_type")
        self.params.update(params)
        return self

    def get_params(self, deep=True):
        return {
            "model_fn": self.model_fn,
            "task_type": self.task_type,
            **self.params
        }

    def fit(self, X, y):
        self.model = self.model_fn(**self.params)
        self.model.fit(X, y, epochs=10, batch_size=32, verbose=0)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return np.round(y_pred).flatten() if self.task_type == "classification" else y_pred.flatten()

    def score(self, X, y):
        y_pred = self.predict(X)
        if self.task_type == "classification":
            return accuracy_score(y, y_pred)
        else:
            return -mean_squared_error(y, y_pred)  # Negative for minimization


# ✅ Grid search function — auto-handles DNN (callable) vs sklearn models (object)
def grid_search_tune(model_fn_or_obj, X_train, y_train, param_grid, task_type="classification"):
    scoring = "accuracy" if task_type == "classification" else "neg_mean_squared_error"

    # ✅ Detect if model is a callable (like for DNN)
    if callable(model_fn_or_obj):
        wrapped_model = KerasModelWrapper(model_fn=model_fn_or_obj, task_type=task_type)
    else:
        wrapped_model = model_fn_or_obj  # SVM, XGBoost etc.

    grid = GridSearchCV(
        estimator=wrapped_model,
        param_grid=param_grid,
        cv=3,
        n_jobs=-1,
        scoring=scoring,
        error_score='raise'  # better debugging
    )

    grid.fit(X_train, y_train)
    return grid.best_params_
