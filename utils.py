import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error
import joblib
import os

def load_data(filepath="economic_indicator.csv"):
    """
    Load dataset from CSV file and perform cleaning.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    # Cleaning logic alignment with test2.ipynb
    df.columns = df.columns.str.strip()
    df.columns = df.columns.str.replace(" ", "_")
    df.columns = df.columns.str.replace("(", "")
    df.columns = df.columns.str.replace(")", "")
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date')
        df = df.drop(columns=['Date'])
        
    return df

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
    Train multiple models and return the results, best model, and test data for visualization.
    """
    # Features match cleaned columns from economic_indicator.csv
    feature_cols = [
        'GDP_Growth_%', 'Inflation_Rate_%', 'Unemployment_Rate_%', 
        'Interest_Rate_%', 'Exchange_Rate_TZS'
    ]
    
    if 'Stock_Index' not in data.columns:
        raise ValueError("Dataset must contain 'Stock_Index' column")
        
    X = data[feature_cols]
    y = data['Stock_Index']
    
    # Time-based split (80/20) alignment with test2.ipynb
    split_index = int(len(data) * 0.8)
    X_train = X.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_train = y.iloc[:split_index]
    y_test = y.iloc[split_index:]
    
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    
    results = {}
    best_model = None
    best_r2 = -float('inf')
    
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    for name, model in models.items():
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', model)
        ])
        
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        
        results[name] = {
            "R2": r2, 
            "MSE": mse,
            "y_test": y_test,
            "y_pred": y_pred
        }
        
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
            
    return results, best_model

