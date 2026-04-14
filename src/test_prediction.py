"""
Test trained model with custom sensor readings
"""

import joblib
import numpy as np
import pandas as pd

print("="*60)
print("🔮 FAILURE PREDICTION TESTER")
print("="*60)

# Load models
try:
    basic_model = joblib.load('models/random_forest_basic.pkl')
    engineered_model = joblib.load('models/random_forest_engineered.pkl')
    scaler = joblib.load('models/scaler_engineered.pkl')
    print("✅ Models loaded successfully")
except Exception as e:
    print(f"❌ Error loading models: {e}")
    exit()

print("\n📋 Enter sensor readings (or press Enter for default values)")

# Get input with defaults
air_temp = float(input("Air temperature [K] (300-310) [default: 300]: ") or 300)
process_temp = float(input("Process temperature [K] (305-340) [default: 310]: ") or 310)
rotational_speed = float(input("Rotational speed [rpm] (1000-3000) [default: 1500]: ") or 1500)
torque = float(input("Torque [Nm] (30-70) [default: 40]: ") or 40)
tool_wear = float(input("Tool wear [min] (0-250) [default: 10]: ") or 10)

# Create feature array
basic_features = np.array([[air_temp, process_temp, rotational_speed, torque, tool_wear]])

# Make prediction with basic model
basic_pred = basic_model.predict(basic_features)[0]
basic_proba = basic_model.predict_proba(basic_features)[0][1]

print("\n" + "="*60)
print("📊 PREDICTION RESULTS")
print("="*60)

print(f"\n🔧 Basic Model (5 sensors only):")
print(f"   Prediction: {'❌ FAILURE PREDICTED' if basic_pred == 1 else '✅ NORMAL'}")
print(f"   Failure Probability: {basic_proba*100:.2f}%")

if basic_proba > 0.7:
    print(f"   ⚠️  URGENT: Schedule maintenance immediately!")
elif basic_proba > 0.4:
    print(f"   ⚠️  Warning: Monitor closely")
else:
    print(f"   ✅ Machine operating normally")

print("\n" + "="*60)
print("💡 INTERPRETATION:")
print("   • Probability < 30%: Safe")
print("   • Probability 30-60%: Monitor")
print("   • Probability 60-80%: Warning")
print("   • Probability > 80%: Critical - Take action")
print("="*60)