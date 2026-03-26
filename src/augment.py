import pandas as pd
import numpy as np

def augment_dataset(multiplier=5):
    """
    Takes the original dataset and multiplies it by adding slight 
    random noise to the physiological sensor readings.
    """
    # 1. Load the original clean data
    data_path = r'data/raw/clean-data.csv'
    
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Could not find {data_path}")
        return

    # 2. Separate columns by how we want to treat them
    # These stay exactly the same (demographics and target)
    static_cols = ['Age', 'Gender', 'Height', 'Weight', 'Glucose_level', 'Patient_Id', 'index', 'PPG_Signal']
    
    # These get a tiny bit of realistic mathematical noise (sensor readings)
    sensor_cols = ['Heart_Rate', 'Systolic_Peak', 'Diastolic_Peak', 'Pulse_Area']

    print(f"Original dataset size: {len(df)} rows.")
    
    all_data = [df] # Start our list with the original real data

    # 3. Generate synthetic data
    for i in range(multiplier - 1):
        synthetic_df = df.copy()
        
        for col in sensor_cols:
            if col in synthetic_df.columns:
                # Add +/- 2% random variation to the sensor readings
                noise = np.random.normal(0, 0.02 * synthetic_df[col].mean(), size=len(synthetic_df))
                synthetic_df[col] = synthetic_df[col] + noise
                
        all_data.append(synthetic_df)

    # 4. Combine real and synthetic data together
    augmented_df = pd.concat(all_data, ignore_index=True)

    # 5. Save the new massive dataset
    save_path = r'data/raw/clean-data-augmented.csv'
    augmented_df.to_csv(save_path, index=False)
    
    print("\n" + "="*40)
    print("🧬 DATA AUGMENTATION COMPLETE 🧬")
    print("="*40)
    print(f"New dataset size: {len(augmented_df)} rows!")
    print(f"Saved successfully to: {save_path}")

if __name__ == "__main__":
    # Change the multiplier if you want even more data!
    augment_dataset(multiplier=5)