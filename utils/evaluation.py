from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, mean_squared_error
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def evaluate_model(model, X_test, y_test, task_type, metrics):
    y_pred = model.predict(X_test)

    # 🔧 Convert continuous/probabilistic outputs to class labels only for DNN
    if task_type.lower() == "classification":
        y_pred = np.array(y_pred)

        # 👇 Only apply thresholding for DNN (i.e., when predict returns probabilities)
        if not hasattr(model, "predict_proba") and y_pred.ndim > 1:
            y_pred = y_pred[:, 0]
            y_pred = (y_pred > 0.5).astype(int)

        elif hasattr(model, "predict_proba"):
            # Ensure it's integer labels, as SVC's predict gives correct labels
            y_pred = y_pred.astype(int)

    results = {}

    if task_type.lower() == "classification":
        if "Accuracy" in metrics:
            results["Accuracy"] = accuracy_score(y_test, y_pred)
        if "Precision" in metrics:
            results["Precision"] = precision_score(y_test, y_pred, average="weighted")
        if "Recall" in metrics:
            results["Recall"] = recall_score(y_test, y_pred, average="weighted")
        if "F1" in metrics:
            results["F1"] = f1_score(y_test, y_pred, average="weighted")
        if "AUC" in metrics and hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
            results["AUC"] = roc_auc_score(y_test, y_proba)

    elif task_type.lower() == "regression":
        if "MSE" in metrics or "RMSE" in metrics:
            mse = mean_squared_error(y_test, y_pred)
            if "MSE" in metrics:
                results["MSE"] = mse
            if "RMSE" in metrics:
                results["RMSE"] = mse ** 0.5

    return results


def plot_confusion_matrix(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = np.array(y_pred)
    if y_pred.ndim > 1:
        y_pred = y_pred[:, 0]
    y_pred = (y_pred > 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title("Confusion Matrix")
    return fig


def plot_roc_curve(model, X_test, y_test):
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # fallback for models like DNN
        y_proba = model.predict(X_test)
        if isinstance(y_proba, (list, np.ndarray)):
            y_proba = np.array(y_proba)
            if y_proba.ndim > 1:
                y_proba = y_proba[:, 0]

    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        fig, ax = plt.subplots()
        ax.plot(fpr, tpr, label="ROC Curve")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve")
        ax.legend()
        return fig
    return None


def plot_feature_importance(model, X_test, model_name):
    if model_name == "XGBoost" and hasattr(model, "feature_importances_"):
        fig, ax = plt.subplots()
        ax.bar(range(X_test.shape[1]), model.feature_importances_)
        ax.set_title("Feature Importance")
        ax.set_xlabel("Feature Index")
        ax.set_ylabel("Importance")
        return fig
    return None
