# 🚀 AutoML Platform for Model Selection and Hyperparameter Tuning

An end-to-end, no-code AutoML web application for automating the ML workflow—from **data preprocessing**, **model selection** (SVM, XGBoost, DNN), to **hyperparameter tuning** using **Grid Search**, **Optuna**, and **Bayesian Optimization**.

---

## 📌 Features

- ✅ Automated Data Preprocessing (scaling, encoding, imputation, outlier removal)
- ✅ Model Selection: SVM, XGBoost, Deep Neural Networks (via PyCaret)
- ✅ Hyperparameter Tuning with:
  - Grid Search
  - Optuna (Bayesian Optimization)
  - Random Search
- ✅ Visual performance comparison: Accuracy, F1-Score, ROC-AUC
- ✅ Track experiments with MLflow
- ✅ Interactive UI built using Streamlit
- ✅ Export trained models and tuning reports

---

## 🛠️ Tech Stack

| Component        | Technology Used               |
|------------------|-------------------------------|
| Backend / ML     | PyCaret, Optuna, Scikit-Learn |
| UI               | Streamlit                     |
| Experiment Logs  | MLflow                        |
| Model Export     | joblib, pickle                |
| Optimization     | Optuna, GridSearchCV          |

---

## 📂 Project Structure

```bash
📁 automl-platform/
│
├── app.py                  # Main Streamlit app
├── requirements.txt        # Python dependencies
├── README.md               # Project overview
│
├── 📁 modules/              # Custom Python scripts
│   ├── data_handler.py     # Data loading & preprocessing
│   ├── model_selector.py   # PyCaret model setup
│   ├── tuner.py            # Grid/Optuna/BO tuner functions
│   ├── mlflow_utils.py     # MLflow integration
│
├── 📁 artifacts/
│   └── best_model.pkl      # Exported model (example)
├── 📁 logs/                # MLflow tracking directory
└── 📁 datasets/            # Sample CSV datasets
