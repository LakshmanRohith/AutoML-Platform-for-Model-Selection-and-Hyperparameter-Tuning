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
```

---

## 🚦 How It Works

1. **Upload a CSV dataset**
2. **Select your target column**
3. **Choose model type (SVM, XGBoost, DNN)**
4. **Select tuning strategy (Grid Search / Optuna / BO)**
5. **Train & optimize**
6. **View performance metrics and download the best model**

---

## 🖥️ Demo Screenshot

> _Insert screenshot of your UI here if available_

![AutoML Streamlit UI](./assets/demo_ui.png)

---

## 🚀 Quick Start

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### ▶️ 2. Run the App

```bash
streamlit run app.py
```

---

## 📊 MLFlow Dashboard

To monitor your experiment logs:

```bash
mlflow ui
```

Visit at: `http://localhost:5000`

---

## 📈 Supported Models & Tuners

| Model         | Supported         |
|---------------|------------------|
| SVM           | ✅                |
| XGBoost       | ✅                |
| DNN (via PyCaret) | ✅            |

| Tuning Method       | Library        |
|---------------------|----------------|
| Grid Search         | Scikit-learn   |
| Random Search       | PyCaret        |
| Optuna (Bayesian)   | Optuna         |

---

## 📚 Citations & References

This project is inspired by the following papers and tools:

- [Optuna: A Next‑generation Hyperparameter Optimization Framework](https://arxiv.org/abs/1907.10902)
- [PyCaret Documentation](https://pycaret.gitbook.io/docs/)
- [Streamlit Docs](https://docs.streamlit.io/)
- [MLflow Docs](https://mlflow.org/)
- [Automating the Machine Learning Process using PyCaret and Streamlit (ResearchGate)](https://www.researchgate.net/publication/370150310_Automating_the_Machine_Learning_Process_using_PyCaret_and_Streamlit)
- [Benchmark and Survey of AutoML Frameworks (arXiv)](https://arxiv.org/abs/1904.12054)

---

## 🧪 Example Datasets



---

## 🧠 Future Work

- ✨ Auto Feature Engineering with FeatureTools
- 🧪 Multi-objective optimization (Accuracy + Speed)
- 🧠 NAS (Neural Architecture Search) for DNN tuning
- 🌐 Deploy as a public web app using Hugging Face Spaces or Streamlit Cloud

---

## 🤝 Contributing

Pull requests and feature suggestions are welcome!  
Please fork the repo and submit a PR with a clear description.

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙋 Contact

Created with ❤️ by [Lakshman Rohith]  
📧 lakshmansanagapalli@gmail.com
🔗 [LinkedIn]([https://linkedin.com/in/yourprofile](https://linkedin.com/in/lakshman-rohith-sanagapalli)) 
