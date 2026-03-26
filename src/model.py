import os
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import pickle # <-- NEW: The library used to save the AI

def train_model():
    data_path = r"C:\Users\admin\Desktop\PPG_Diabetes_Project\data\raw\clean-dataset.csv"
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find the dataset at {data_path}")
        return

    features_to_drop = ['PPG_Signal', 'Patient_Id', 'index', 'Glucose_level']
    if 'pl' in df.columns:
        features_to_drop.append('pl')
        
    X = df.drop(columns=features_to_drop, errors='ignore')
    y = df['Glucose_level']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print("\n" + "="*30)
    print("🎯 MODEL EVALUATION RESULTS 🎯")
    print("="*30)
    print(f"Mean Absolute Error (MAE): {mae:.2f} mg/dL")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f} mg/dL")
    print("="*30)

    # --- NEW CODE: SAVE THE AI'S BRAIN ---
    # Create a 'models' folder if it doesn't exist yet
    os.makedirs('models', exist_ok=True)
    
    # Save the trained XGBoost model to a file
    model_path = 'models/xgboost_blood_glucose_model.pkl'
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
        
    print(f"✅ SUCCESS: AI model successfully saved to {model_path}!")
    # -------------------------------------

    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, color='dodgerblue', alpha=0.8, edgecolors='k', s=80, label='AI Predictions')
    
    min_val = min(y_test.min(), predictions.min()) - 10
    max_val = max(y_test.max(), predictions.max()) + 10
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Accuracy Line')
    
    plt.title('Non-Invasive Blood Glucose Prediction\nActual vs. Predicted', fontsize=14, fontweight='bold')
    plt.xlabel('True Blood Glucose (mg/dL)', fontsize=12)
    plt.ylabel('AI Predicted Glucose (mg/dL)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.show()

if __name__ == "__main__":
    train_model()