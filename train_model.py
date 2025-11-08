import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pickle
import os

def extract_features_from_attempt(df):
    """Extract features from a single password attempt"""
    dwell = df['dwell_time'].dropna()
    flight = df['flight_time'].dropna()
    
    features = {
        'dwell_mean': dwell.mean(),
        'dwell_std': dwell.std(),
        'dwell_median': dwell.median(),
        'dwell_min': dwell.min(),
        'dwell_max': dwell.max(),
        'dwell_q25': dwell.quantile(0.25),
        'dwell_q75': dwell.quantile(0.75),
        'flight_mean': flight.mean(),
        'flight_std': flight.std(),
        'flight_median': flight.median(),
        'flight_min': flight.min(),
        'flight_max': flight.max(),
        'flight_q25': flight.quantile(0.25),
        'flight_q75': flight.quantile(0.75),
    }
    
    return features

# Load all 5 CSV files
print("Loading training data from 5 CSV files...")
print("="*50)

csv_files = ['data/sample1.csv', 'data/sample2.csv', 'data/sample3.csv', 'data/sample4.csv', 'data/sample5.csv']
attempts = []
features_list = []

for i, csv_file in enumerate(csv_files, 1):
    if not os.path.exists(csv_file):
        print(f"✗ ERROR: {csv_file} not found!")
        print(f"Please make sure all 5 files exist:")
        for f in csv_files:
            print(f"  - {f}")
        exit(1)
    

    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    
    attempts.append(df)
    print(f"✓ Loaded {csv_file}: {len(df)} keystrokes")
    

    features = extract_features_from_attempt(df)
    features_list.append(features)

print("="*50)
print(f"Total attempts loaded: {len(attempts)}")


lengths = [len(att) for att in attempts]
avg_length = np.mean(lengths)
max_diff = max(lengths) - min(lengths)

print(f"\nPassword length analysis:")
print(f"  Lengths: {lengths}")
print(f"  Average: {avg_length:.1f} keystrokes")

if max_diff > 3:
    print(f"  ⚠ WARNING: Length variation is {max_diff} keystrokes")
    print(f"  This might indicate different passwords or typing errors!")
else:
    print(f"  ✓ Length consistency: Good (±{max_diff} keystrokes)")


X = pd.DataFrame(features_list).values
print(f"\nGenerated {len(X)} training samples (one per CSV)")
print(f"Features per sample: {X.shape[1]}")


print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


print("Training Isolation Forest model...")
model = IsolationForest(
    contamination=0.2, 
    random_state=42,
    n_estimators=100,
    bootstrap=True
)
model.fit(X_scaled)


model_data = {
    'model': model,
    'scaler': scaler,
    'password_length': int(avg_length),
    'num_training_attempts': len(attempts),
    'training_files': csv_files
}

with open('Model/password_keystroke_model.pkl', 'wb') as f:
    pickle.dump(model_data, f)

print("✓ Training completed!")
print("✓ Model saved to 'password_keystroke_model.pkl'")




print("\nTiming Statistics Across All 5 Attempts:")
feature_df = pd.DataFrame(features_list)
stats_df = feature_df[['dwell_mean', 'dwell_std', 'flight_mean', 'flight_std']].describe()
print(stats_df.round(2))