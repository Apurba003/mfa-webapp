import pandas as pd
import numpy as np
import pickle

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

# Load the trained model
print("Loading trained model...")
with open('Model/password_keystroke_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

model = model_data['model']
scaler = model_data['scaler']
expected_length = model_data['password_length']

print("✓ Model loaded successfully")
print(f"Expected password length: {expected_length} keystrokes")

# Load verification data (single password attempt)
print("\nLoading verification data...")
df = pd.read_csv('data/sample6.csv')
df.columns = df.columns.str.strip().str.lower()

print(f"Loaded {len(df)} keystrokes for verification")

# Check password length
if abs(len(df) - expected_length) > 2:  # Allow small variation
    print(f"⚠ WARNING: Expected ~{expected_length} keystrokes, got {len(df)}")
    print("This might be a different password!")

# Extract features
features = extract_features_from_attempt(df)
X = pd.DataFrame([features]).values

# Scale features
X_scaled = scaler.transform(X)

# Predict
prediction = model.predict(X_scaled)[0]
score = model.decision_function(X_scaled)[0]

# Interpret results
is_authentic = (prediction == 1)
confidence = round((score + 0.5) * 100, 2)
confidence = max(0, min(100, confidence))  # Clamp to 0-100

# Display results
print("\n" + "="*60)
print("VERIFICATION RESULTS")
print("="*60)

if is_authentic:
    print("✓ AUTHENTICATION SUCCESSFUL")
    print("Status: VERIFIED ✓")
    print(f"Confidence: {confidence}%")
    print(f"Decision Score: {score:.4f}")
    print("\n➜ This typing pattern matches the enrolled user!")
else:
    print("✗ AUTHENTICATION FAILED")
    print("Status: REJECTED ✗")
    print(f"Confidence: {confidence}%")
    print(f"Decision Score: {score:.4f}")
    print("\n➜ This typing pattern does NOT match the enrolled user!")
    print("   Possible reasons:")
    print("   - Different person typing")
    print("   - Typing too fast/slow")
    print("   - Typing errors or corrections")

print("="*60)

# Additional statistics comparison
print("\nVerification Sample Statistics:")
print(f"  Avg Dwell Time: {features['dwell_mean']:.2f} ms")
print(f"  Avg Flight Time: {features['flight_mean']:.2f} ms")


# Security recommendation
print("\nSecurity Level:")
if is_authentic and confidence > 70:
    print("  🟢 HIGH - Strong match")
elif is_authentic and confidence > 50:
    print("  🟡 MEDIUM - Acceptable match")
elif is_authentic:
    print("  🟠 LOW - Weak match, consider re-enrollment")
else:
    print("  🔴 REJECTED - No match")

"""
HOW TO USE:

1. TRAINING PHASE (Do once):
   - Collect 5 password attempts in training_data.csv
   - Run: python train_password.py
   - Creates: password_keystroke_model.pkl

2. VERIFICATION PHASE (Every login):
   - User types password once
   - Save as verification_data.csv
   - Run: python verify_password.py
   - Check result: VERIFIED or REJECTED

CSV FORMAT (same for both):
key,down,up,dwell time,flight time
p,100,150,50,0
a,160,200,40,10
s,210,250,40,10
s,260,300,40,10
w,310,360,50,10
...

TIPS:
- User should type naturally (not too careful)
- Same keyboard/device for best results
- Retry if user made typing errors
- Re-train if false rejections are common

SECURITY NOTES:
- This is a SECONDARY authentication factor
- Always combine with traditional password check
- False Acceptance Rate (FAR): ~5-10%
- False Rejection Rate (FRR): ~10-15%
"""