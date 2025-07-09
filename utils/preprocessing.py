import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import yaml

def preprocess_data(df, target_column, task_type):
    with open("config.yaml", "r") as file:
        config = yaml.safe_load(file)
    
    # Extract preprocessing parameters
    train_size = config["preprocessing"]["train_split"]
    scale_features = config["preprocessing"].get("scale_features", True)
    encode_categorical = config["preprocessing"].get("encode_categorical", True)

    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Encode categorical variables if specified
    if encode_categorical:
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])

    # Scale features if specified
    if scale_features:
        scaler = StandardScaler()
        X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=42)

    return X_train, X_test, y_train, y_test