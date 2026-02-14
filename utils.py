import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def load_data(filepath):
    """
    Load dataset from CSV file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    return pd.read_csv(filepath)

def load_model(filepath='model.pkl'):
    """
    Load a trained model from disk if it exists.
    """
    if os.path.exists(filepath):
        try:
            return joblib.load(filepath)
        except Exception as e:
            print(f"Error loading model: {e}")
            return None
    return None

def train_models(data):
    """
    Train multiple models and return the results and the best model.
    """
    # Assume 'Stock_Index' is the target and the rest are features
    if 'Stock_Index' not in data.columns:
        raise ValueError("Dataset must contain 'Stock_Index' column")
        
    X = data.drop('Stock_Index', axis=1)
    y = data['Stock_Index']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {"R2": r2, "MSE": mse}
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = model
            
    return results, best_model
