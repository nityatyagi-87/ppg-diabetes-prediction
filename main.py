
from src.model import train_xgboost

def main():
    
    
    print("Training the model...")
    # This function reads the processed data and trains XGBoost
    train_xgboost()

if __name__ == "__main__":
    main()