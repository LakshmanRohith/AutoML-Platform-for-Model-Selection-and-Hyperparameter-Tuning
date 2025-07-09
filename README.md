AutoML Platform for Model Selection and Hyperparameter Tuning

Overview

This project is a web-based AutoML platform that allows users to upload a dataset, preprocess it, train machine learning models (SVM, XGBoost, DNN), tune hyperparameters, and track experiments using MLflow. The frontend is built with Streamlit.

Installation

Clone the repository:

git clone <repository_url>
cd automl-platform

Create and activate a virtual environment:

python -m venv automl_env
source automl_env/bin/activate # On Windows: automl_env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Running the Application

Start the MLflow server:

mlflow ui --backend-store-uri sqlite:///mlflow.db

Run the Streamlit app:

streamlit run app.py

Open your browser and go to http://localhost:8501.

Usage

Upload a CSV dataset.

Select the target column, task type (Classification/Regression), models, tuning method, and evaluation metrics.

Click "Run AutoML" to preprocess, train, and evaluate models.

View results, download the best model, and export a PDF report.

Project Structure

app.py: Main Streamlit app interface.

config.yaml: Configuration for tuning and preprocessing.

models/: Contains training logic for SVM, XGBoost, DNN.

tuning/: Contains hyperparameter tuning logic.

utils/: Contains preprocessing, evaluation, and MLflow tracking functions.

artifacts/: Stores saved models and plots.

requirements.txt: Lists required packages.

README.md: Project documentation.

Future Scope

Add time-series model support.

Integrate cloud storage (S3, GCP).

Deploy with Docker.

Add XAI support (LIME, SHAP).
