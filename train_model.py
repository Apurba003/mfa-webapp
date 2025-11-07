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
    
    # Load CSV
    df = pd.read_csv(csv_file)
    df.columns = df.columns.str.strip().str.lower()
    
    attempts.append(df)
    print(f"✓ Loaded {csv_file}: {len(df)} keystrokes")
    
    # Extract features
    features = extract_features_from_attempt(df)
    features_list.append(features)

print("="*50)
print(f"Total attempts loaded: {len(attempts)}")

# Check if all attempts have similar length (should be same password)
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

# Convert features to array
X = pd.DataFrame(features_list).values
print(f"\nGenerated {len(X)} training samples (one per CSV)")
print(f"Features per sample: {X.shape[1]}")

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train Isolation Forest
print("Training Isolation Forest model...")
model = IsolationForest(
    contamination=0.2,  # Expect 1 out of 5 might be slightly different
    random_state=42,
    n_estimators=100,
    bootstrap=True
)
model.fit(X_scaled)

# Save model, scaler, and metadata
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

# Display training summary
print("\n" + "="*60)
print("TRAINING SUMMARY")
print("="*60)
print(f"Training files used:")
for f in csv_files:
    print(f"  ✓ {f}")
print(f"\nPassword attempts: {len(attempts)}")
print(f"Average keystrokes per attempt: {avg_length:.1f}")
print(f"Total keystrokes: {sum(lengths)}")
print(f"Training samples: {len(X)}")
print(f"Features per sample: {X.shape[1]}")
print(f"Model type: Isolation Forest")
print(f"Model file: password_keystroke_model.pkl")
print("="*60)

# Show feature statistics across all attempts
print("\nTiming Statistics Across All 5 Attempts:")
feature_df = pd.DataFrame(features_list)
stats_df = feature_df[['dwell_mean', 'dwell_std', 'flight_mean', 'flight_std']].describe()
print(stats_df.round(2))

print("\n" + "="*60)
print("Training complete! You can now verify with: python verify_password.py")
print("="*60)

"""
HOW TO USE:

1. PREPARE 5 CSV FILES:
   Create these files with password attempts:
   - sample1.csv
   - sample2.csv
   - sample3.csv
   - sample4.csv
   - sample5.csv
   
   Each CSV format:
   key,down,up,dwell time,flight time
   p,100,150,50,0
   a,160,200,40,10
   s,210,250,40,10
   ...

2. RUN TRAINING:
   python train_password.py
   
3. RESULT:
   Creates password_keystroke_model.pkl

REQUIREMENTS:
- All 5 CSV files must exist in same directory
- Each file = 1 complete password attempt
- Same password typed 5 times
- Similar length across all attempts (for best results)

TIPS:
- Type naturally, not too carefully
- Use same keyboard/device
- Don't rush or go too slow
- If you made errors, redo that attempt
"""