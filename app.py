

import streamlit as st
import pandas as pd
import os
import joblib
import mlflow

from models.svm import train_svm
from models.xgboost_model import train_xgboost
from models.dnn import train_dnn, create_model

from tuning.grid_search import grid_search_tune
from tuning.optuna_search import optuna_tune
from utils.preprocessing import preprocess_data
from utils.evaluation import evaluate_model
from utils.mlflow_tracker import log_experiment

from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet
from tensorflow.keras.models import load_model
import zipfile

# Set page configuration
st.set_page_config(page_title="AutoML Platform", layout="wide")

# Title and description
st.title("AutoML Platform")
st.write("Upload a dataset and configure your AutoML run.")

# File uploader
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", df.head())

    # Select target column
    target_column = st.selectbox("Select target column", df.columns)

    # Task type
    task_type = st.selectbox("Select task type", ["Classification", "Regression"])

    # Model selection
    models = st.multiselect("Select models", ["SVM", "XGBoost", "DNN"])
    if not models:
        st.warning("Please select at least one model.")
        st.stop()

    # Tuning method
    tuning_method = st.selectbox("Select hyperparameter tuning method", ["Grid Search", "Optuna", "Bayesian Optimization"])

    # Metrics
    metrics = st.multiselect("Select evaluation metrics", ["Accuracy", "F1", "Precision", "Recall", "RMSE"], default=["Accuracy", "F1"])

    # Run button
    if st.button("Run AutoML"):
        # Preprocess data
        X_train, X_test, y_train, y_test = preprocess_data(df, target_column, task_type)

        # Initialize results dictionary
        results = {}

        # Train and tune models
        for model_name in models:
            with mlflow.start_run(run_name=model_name, nested=True):
                
                
                if model_name == "SVM":
                    try:
                        import yaml
                        with open("config.yaml", "r") as file:
                            config = yaml.safe_load(file)

                        from sklearn.svm import SVC
                        model = SVC(probability=True)

                        if tuning_method == "Grid Search":
                            best_params = grid_search_tune(
                                model, X_train, y_train,
                                config["tuning"]["svm"]["grid_search"], task_type
                            )
                        elif tuning_method == "Optuna":
                            best_params = optuna_tune(
                                model, X_train, y_train,
                                config["tuning"]["svm"]["optuna"], task_type
                            )
                        elif tuning_method == "Bayesian Optimization":
                            from tuning.bayesian_opt import bayesian_tune
                            best_params = bayesian_tune(
                                model, X_train, y_train,
                                config["tuning"]["svm"]["bayesian"], task_type
                            )
                        else:
                            best_params = {}

                        model.set_params(**best_params)
                        model.fit(X_train, y_train)

                        # ✅ Save model for download
                        import os
                        import joblib
                        os.makedirs("artifacts", exist_ok=True)
                        joblib.dump(model, "artifacts/SVM_model.pkl")

                    except Exception as e:
                        st.error(f"SVM training failed: {e}")
                        continue





                # elif model_name == "XGBoost":
                #     try:
                #         best_params = {"max_depth": 3, "learning_rate": 0.1}
                #         if tuning_method == "Grid Search":
                #             best_params = grid_search_tune(train_xgboost, X_train, y_train, {"max_depth": [3, 5], "learning_rate": [0.01, 0.1]}, task_type)
                #         elif tuning_method == "Optuna":
                #             best_params = optuna_tune(train_xgboost, X_train, y_train, {"max_depth": (3, 5), "learning_rate": (0.01, 0.1)}, task_type)
                #         elif tuning_method == "Bayesian Optimization":
                #             from tuning.bayesian_opt import bayesian_tune
                #             best_params = bayesian_tune(train_xgboost, X_train, y_train, {"max_depth": (3, 5), "learning_rate": (0.01, 0.1), "n_iter": 10}, task_type)
                #     except ImportError:
                #         st.warning("Bayesian Optimization not available for XGBoost. Using default parameters.")
                #     model = train_xgboost(X_train, y_train, task_type, **best_params)
                elif model_name == "XGBoost":
                    try:
                        import yaml
                        with open("config.yaml", "r") as file:
                            config = yaml.safe_load(file)

                        from xgboost import XGBClassifier, XGBRegressor

                        # Use correct model type
                        model = XGBClassifier(use_label_encoder=False, eval_metric="logloss") if task_type.lower() == "classification" \
                                else XGBRegressor()

                        if tuning_method == "Grid Search":
                            best_params = grid_search_tune(
                                model, X_train, y_train,
                                config["tuning"]["xgboost"]["grid_search"], task_type
                            )
                        elif tuning_method == "Optuna":
                            best_params = optuna_tune(
                                model, X_train, y_train,
                                config["tuning"]["xgboost"]["optuna"], task_type
                            )
                        elif tuning_method == "Bayesian Optimization":
                            from tuning.bayesian_opt import bayesian_tune
                            best_params = bayesian_tune(
                                model, X_train, y_train,
                                config["tuning"]["xgboost"]["bayesian"], task_type
                            )
                        else:
                            best_params = {}

                        model.set_params(**best_params)
                        model.fit(X_train, y_train)

                        # ✅ Save model for download
                        import os
                        import joblib
                        os.makedirs("artifacts", exist_ok=True)
                        joblib.dump(model, "artifacts/XGBoost_model.pkl")

                    except Exception as e:
                        st.error(f"XGBoost training failed: {e}")
                        continue

                    
                    
                    
                elif model_name == "DNN":
                    try:
                        # Make it compatible with bayes_opt
                        def wrapped_model_fn(**kwargs):
                            units = int(kwargs.get("units", 64))
                            layers = int(kwargs.get("layers", 2))
                            return create_model(units, layers, X_train.shape[1], task_type)

                        if tuning_method == "Grid Search":
                            best_params = grid_search_tune(
                                wrapped_model_fn, X_train, y_train,
                                {"units": [32, 64], "layers": [1, 2]}, task_type
                            )
                        elif tuning_method == "Optuna":
                            best_params = optuna_tune(
                                wrapped_model_fn, X_train, y_train,
                                {"units": (32, 128), "layers": (1, 3)}, task_type
                            )
                        elif tuning_method == "Bayesian Optimization":
                            from tuning.bayesian_opt import bayesian_tune
                            best_params = bayesian_tune(
                                wrapped_model_fn, X_train, y_train,
                                {"units": (32, 128), "layers": (1, 3), "n_iter": 10}, task_type
                            )
                    except ImportError:
                        st.warning("Bayesian Optimization not available for DNN. Using default parameters.")
                        best_params = {"units": 64, "layers": 2}

                    model, _ = train_dnn(X_train, y_train, task_type, tuning_method)



                # Train and evaluate
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                # eval_results = evaluate_model(y_test, y_pred, metrics, task_type)
                eval_results = evaluate_model(model, X_test, y_test, task_type, metrics)


                # Log to MLflow
                log_experiment(model_name, best_params, eval_results, model, tuning_method)

                # Store results
                results[model_name] = {"params": best_params, "metrics": eval_results}

        # Display results
        st.write("Results:", results)

        # Generate and download report
        def generate_report(results):
            pdf_file = "automl_report.pdf"
            doc = SimpleDocTemplate(pdf_file, pagesize=letter)
            styles = getSampleStyleSheet()
            story = [Paragraph("AutoML Report", styles['Title'])]
            for model_name, result in results.items():
                story.append(Paragraph(f"Model: {model_name}", styles['Heading2']))
                story.append(Paragraph(f"Parameters: {result['params']}", styles['Normal']))
                story.append(Paragraph(f"Metrics: {result['metrics']}", styles['Normal']))
                story.append(Spacer(1, 12))
            doc.build(story)
            return pdf_file

        if results:
            pdf_path = generate_report(results)
            with open(pdf_path, "rb") as pdf_file:
                st.download_button("Download Report (PDF)", pdf_file, "automl_report.pdf", mime="application/pdf")


        best_model_name = max(results, key=lambda x: results[x]["metrics"].get("Accuracy", 0))
        # Load model based on type
        if best_model_name == "DNN":
            best_model_path = f"artifacts/{best_model_name}_model"
            best_model = load_model(best_model_path)
            # Create zip archive of the model directory for download
            zip_filename = f"{best_model_name}_model.zip"
            with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
                for root, dirs, files in os.walk(best_model_path):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, os.path.dirname(best_model_path))
                        zipf.write(file_path, arcname)
                        # Provide download button
            with open(zip_filename, "rb") as f:
                st.download_button("Download Best DNN Model (ZIP)", f, file_name=zip_filename, mime="application/zip")
            os.remove(zip_filename)
        else:
            # For SVM, XGBoost (pkl models)
            best_model = joblib.load(f"artifacts/{best_model_name}_model.pkl")
            joblib.dump(best_model, "temp_model.pkl")
            with open("temp_model.pkl", "rb") as model_file:
                st.download_button("Download Best Model", model_file, f"{best_model_name}_model.pkl", mime="application/octet-stream")
            os.remove("temp_model.pkl")
