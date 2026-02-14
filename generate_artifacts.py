import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# 1. Generate Data
print("Generating data...")
np.random.seed(42)
n_samples = 1000
df = pd.DataFrame({
    'GDP_Growth': np.random.uniform(0, 10, n_samples),
    'Inflation_Rate': np.random.uniform(0, 15, n_samples),
    'Unemployment_Rate': np.random.uniform(0, 20, n_samples),
    'Interest_Rate': np.random.uniform(0, 15, n_samples),
    'Exchange_Rate': np.random.uniform(1000, 5000, n_samples)
})

df['Stock_Index'] = (
    1000 + 
    (df['GDP_Growth'] * 150) + 
    (df['Exchange_Rate'] * 0.05) - 
    (df['Inflation_Rate'] * 80) - 
    (df['Unemployment_Rate'] * 50) - 
    (df['Interest_Rate'] * 60) + 
    np.random.normal(0, 50, n_samples)
)

df.to_csv('economic_indicators_1000.csv', index=False)
print("Data saved.")

# 2. Train Model
print("Training model...")
X = df.drop('Stock_Index', axis=1)
y = df['Stock_Index']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"Model R2: {r2:.4f}")

# 3. Save Model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")
