try:
    from bayes_opt import BayesianOptimization
except ImportError:
    BayesianOptimization = None

from sklearn.svm import SVC, SVR
import numpy as np

def bayesian_tune(model, X_train, y_train, params, task_type):
    if BayesianOptimization is None:
        raise ImportError("Bayesian Optimization package not available. Please install it or use another tuning method.")

    is_callable_model = callable(model)  # True for DNN, False for SVC/XGB, etc.

    def objective(**param):
        param_dict = param.copy()

        # Handle special preprocessing for SVM/SVR kernel
        if not is_callable_model and isinstance(model, (SVC, SVR)):
            kernel_val = param_dict.get("kernel_continuous", 0.5)
            kernels = ['rbf', 'linear', 'sigmoid', 'poly']
            kernel = kernels[int(kernel_val * len(kernels)) % len(kernels)]
            param_dict['kernel'] = kernel
            param_dict.pop('kernel_continuous', None)

        # Convert relevant params to int
        for key in ["max_depth", "units", "layers"]:
            if key in param_dict:
                param_dict[key] = int(round(param_dict[key]))

        try:
            # Build model
            if is_callable_model:
                model_instance = model(**param_dict)
            else:
                model.set_params(**param_dict)
                model_instance = model

            # Fit model
            model_instance.fit(X_train, y_train)

            # Return score
            if hasattr(model_instance, "evaluate"):
                # Keras model
                _, metric = model_instance.evaluate(X_train, y_train, verbose=0)
                return metric if task_type == "classification" else -metric
            else:
                return model_instance.score(X_train, y_train)
        except Exception as e:
            print(f"Error during objective evaluation: {e}")
            return 0.0  # Return neutral score on failure

    # Set parameter bounds
    pbounds = {}
    if isinstance(model, (SVC, SVR)):
        pbounds["C"] = (0.1, 10)
        pbounds["kernel_continuous"] = (0, 1)
    for k in ["max_depth", "learning_rate", "units", "layers"]:
        if k in params:
            vals = params[k]
            if isinstance(vals, list) or isinstance(vals, tuple):
                if len(vals) == 2:
                    pbounds[k] = (float(vals[0]), float(vals[1]))

    optimizer = BayesianOptimization(f=objective, pbounds=pbounds, random_state=1)
    optimizer.maximize(n_iter=params.get("n_iter", 5))

    best_params = optimizer.max["params"]

    # Postprocess best_params
    if "max_depth" in best_params:
        best_params["max_depth"] = int(round(best_params["max_depth"]))
    if "units" in best_params:
        best_params["units"] = int(round(best_params["units"]))
    if "layers" in best_params:
        best_params["layers"] = int(round(best_params["layers"]))
    if "kernel_continuous" in best_params:
        kernels = ['rbf', 'linear', 'sigmoid', 'poly']
        best_params["kernel"] = kernels[int(best_params["kernel_continuous"] * len(kernels)) % len(kernels)]
        del best_params["kernel_continuous"]

    return best_params

# Optional test
if __name__ == "__main__":
    if BayesianOptimization is None:
        print("Bayesian Optimization not available.")
    else:
        from sklearn.svm import SVC
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        model = SVC()
        params = {"C": (0.1, 10), "n_iter": 5}
        best_params = bayesian_tune(model, X, y, params, "classification")
        print("Best parameters:", best_params)