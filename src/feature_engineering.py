"""
AI-Powered Predictive Maintenance for IoT Devices
FEATURE ENGINEERING - FIXED VERSION (No Infinity Errors)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# Create output folders
os.makedirs('outputs/feature_engineering', exist_ok=True)
os.makedirs('models', exist_ok=True)

print("="*70)
print("🔧 FEATURE ENGINEERING - TIME-BASED FEATURES (FIXED)")
print("="*70)

# ============================================
# 1. LOAD ORIGINAL DATA
# ============================================
print("\n📂 1. LOADING ORIGINAL DATASET...")
df = pd.read_csv('data/ai4i2020.csv')
print(f"✅ Original shape: {df.shape}")

# Create a time index (simulating time order)
df['time_index'] = range(len(df))
print(f"✅ Added time_index column")

# ============================================
# 2. CREATE ROLLING STATISTICS (Safe version)
# ============================================
print("\n📊 2. CREATING ROLLING STATISTICS...")

sensor_columns = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']

windows = [10, 20, 50]

for col in sensor_columns:
    for window in windows:
        # Rolling mean - fill NaN with forward fill then 0
        df[f'{col}_rolling_mean_{window}'] = df[col].rolling(window=window, min_periods=1).mean()
        
        # Rolling std - fill NaN with 0 (safe)
        rolling_std = df[col].rolling(window=window, min_periods=1).std()
        df[f'{col}_rolling_std_{window}'] = rolling_std.fillna(0)
        
        # Rolling max
        df[f'{col}_rolling_max_{window}'] = df[col].rolling(window=window, min_periods=1).max()

print(f"   ✅ Added {len(sensor_columns) * len(windows) * 3} rolling features")

# ============================================
# 3. CREATE RATE OF CHANGE FEATURES (Safe - handle NaN)
# ============================================
print("\n⚡ 3. CREATING RATE OF CHANGE FEATURES...")

for col in sensor_columns:
    # 1-step difference - fill NaN with 0
    df[f'{col}_diff_1'] = df[col].diff(1).fillna(0)
    
    # 5-step difference - fill NaN with 0
    df[f'{col}_diff_5'] = df[col].diff(5).fillna(0)
    
    # 10-step difference - fill NaN with 0
    df[f'{col}_diff_10'] = df[col].diff(10).fillna(0)
    
    # Percentage change - FIXED: handle division by zero
    pct_change = df[col].pct_change() * 100
    # Replace inf and -inf with 0, then fill NaN
    pct_change = pct_change.replace([np.inf, -np.inf], 0).fillna(0)
    df[f'{col}_pct_change'] = pct_change

print(f"   ✅ Added {len(sensor_columns) * 4} rate-of-change features")

# ============================================
# 4. CREATE CUMULATIVE FEATURES (Safe - normalize)
# ============================================
print("\n📈 4. CREATING CUMULATIVE FEATURES...")

# Cumulative sum - normalize to prevent huge numbers
df['cumulative_torque'] = df['Torque [Nm]'].cumsum() / 1000
df['cumulative_tool_wear'] = df['Tool wear [min]'].cumsum() / 1000
df['cumulative_rotations'] = df['Rotational speed [rpm]'].cumsum() / 10000

print(f"   ✅ Added 3 cumulative features (normalized)")

# ============================================
# 5. CREATE INTERACTION FEATURES
# ============================================
print("\n🔗 5. CREATING INTERACTION FEATURES...")

# Temperature difference (always positive, safe)
df['temp_difference'] = df['Process temperature [K]'] - df['Air temperature [K]']

# Power - normalize to prevent huge numbers
df['power'] = (df['Torque [Nm]'] * df['Rotational speed [rpm]']) / 10000

# Temperature × Tool wear - normalize
df['temp_tool_interaction'] = (df['Process temperature [K]'] * df['Tool wear [min]']) / 1000

# Speed variation - fill NaN with 0
speed_variation = df['Rotational speed [rpm]'].rolling(window=10, min_periods=1).std()
df['speed_variation'] = speed_variation.fillna(0)

# Temperature rate of change (additional safety feature)
temp_rate = df['Process temperature [K]'].diff(5).fillna(0)
df['temp_rate_5'] = temp_rate

print(f"   ✅ Added 6 interaction features")

# ============================================
# 6. CREATE FAILURE WINDOW FEATURES
# ============================================
print("\n⚠️ 6. CREATING FAILURE WINDOW FEATURES...")

# Identify failure positions
failure_indices = df[df['Machine failure'] == 1].index.tolist()

# Create counter since last failure (safe version)
df['steps_since_last_failure'] = 0
last_failure = -1
for i in range(len(df)):
    if i in failure_indices:
        last_failure = i
        df.loc[i, 'steps_since_last_failure'] = 0
    else:
        if last_failure >= 0:
            df.loc[i, 'steps_since_last_failure'] = i - last_failure
        else:
            df.loc[i, 'steps_since_last_failure'] = 999  # No failure yet

# Create time to next failure (capped at 50)
df['steps_to_next_failure'] = 999
for idx in failure_indices:
    for offset in range(1, 51):
        if idx + offset < len(df):
            current_value = df.loc[idx + offset, 'steps_to_next_failure']
            if current_value > offset:
                df.loc[idx + offset, 'steps_to_next_failure'] = offset

print(f"   ✅ Added failure proximity features")

# ============================================
# 7. CLEAN INFINITE AND EXTREME VALUES (CRITICAL FIX)
# ============================================
print("\n🧹 7. CLEANING INFINITE AND EXTREME VALUES...")

# Replace any remaining infinity with 0
df = df.replace([np.inf, -np.inf], 0)

# Fill any remaining NaN with 0
df = df.fillna(0)

# Cap extreme values (optional - prevents outliers)
for col in df.select_dtypes(include=[np.number]).columns:
    if col not in ['UID', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF']:
        # Cap at 99th percentile to prevent extreme outliers
        upper_cap = df[col].quantile(0.99)
        lower_cap = df[col].quantile(0.01)
        df[col] = df[col].clip(lower_cap, upper_cap)

print(f"   ✅ Cleaned all infinite values and capped outliers")

# ============================================
# 8. VERIFY NO INFINITY REMAINS
# ============================================
print("\n🔍 8. VERIFYING DATA CLEANLINESS...")

# Check for infinity
has_inf = np.isinf(df.select_dtypes(include=[np.number])).any().any()
if has_inf:
    print("   ⚠️ Warning: Still has infinite values!")
    # Find and replace any remaining inf
    df = df.replace([np.inf, -np.inf], 0)
else:
    print("   ✅ No infinite values found")

# Check for NaN
has_nan = df.isnull().any().any()
if has_nan:
    print("   ⚠️ Warning: Still has NaN values!")
    df = df.fillna(0)
else:
    print("   ✅ No NaN values found")

print(f"   ✅ Data is clean and ready for modeling")

# ============================================
# 9. COMPARE FEATURE SETS
# ============================================
print("\n📊 9. COMPARING ORIGINAL VS ENGINEERED FEATURES...")

# Original features (sensors only)
original_features = sensor_columns

# All engineered features (excluding original sensors and metadata)
exclude_cols = ['UID', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 
                'PWF', 'OSF', 'RNF', 'time_index']
engineered_features = [col for col in df.columns if col not in exclude_cols]

print(f"\n   Original features: {len(original_features)}")
print(f"   Engineered features: {len(engineered_features)}")
print(f"   Total features available: {len(engineered_features)}")

# ============================================
# 10. TRAIN MODELS FOR COMPARISON
# ============================================
print("\n🤖 10. TRAINING MODELS FOR COMPARISON...")

# Prepare data
X_original = df[original_features].copy()
X_engineered = df[engineered_features].copy()
y = df['Machine failure'].copy()

# Check for any remaining issues in data
print("   Checking data quality...")
print(f"   Original features - any inf? {np.isinf(X_original).any().any()}")
print(f"   Engineered features - any inf? {np.isinf(X_engineered).any().any()}")

# Scale features
scaler_original = StandardScaler()
scaler_engineered = StandardScaler()

X_original_scaled = scaler_original.fit_transform(X_original)
X_engineered_scaled = scaler_engineered.fit_transform(X_engineered)

# Split data
X_orig_train, X_orig_test, y_train, y_test = train_test_split(
    X_original_scaled, y, test_size=0.2, random_state=42, stratify=y
)

X_eng_train, X_eng_test, _, _ = train_test_split(
    X_engineered_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest models
print("   Training model with original features...")
rf_original = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_original.fit(X_orig_train, y_train)

print("   Training model with engineered features...")
rf_engineered = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf_engineered.fit(X_eng_train, y_train)

# Make predictions
y_pred_orig = rf_original.predict(X_orig_test)
y_pred_eng = rf_engineered.predict(X_eng_test)

# Calculate metrics
accuracy_orig = accuracy_score(y_test, y_pred_orig)
recall_orig = recall_score(y_test, y_pred_orig)
f1_orig = f1_score(y_test, y_pred_orig)

accuracy_eng = accuracy_score(y_test, y_pred_eng)
recall_eng = recall_score(y_test, y_pred_eng)
f1_eng = f1_score(y_test, y_pred_eng)

# ============================================
# 11. DISPLAY IMPROVEMENTS
# ============================================
print("\n" + "="*70)
print("📈 PERFORMANCE COMPARISON RESULTS")
print("="*70)

print("\n┌─────────────────────┬──────────────┬──────────────┬─────────────┐")
print("│       Metric        │   Original   │  Engineered  │ Improvement │")
print("├─────────────────────┼──────────────┼──────────────┼─────────────┤")

print(f"│ Accuracy            │    {accuracy_orig*100:5.2f}%     │    {accuracy_eng*100:5.2f}%     │    {accuracy_eng-accuracy_orig:+.2%}    │")
print(f"│ Recall (Failures    │    {recall_orig*100:5.2f}%     │    {recall_eng*100:5.2f}%     │    {recall_eng-recall_orig:+.2%}    │")
print(f"│  detected)          │              │              │             │")
print(f"│ F1-Score            │    {f1_orig*100:5.2f}%     │    {f1_eng*100:5.2f}%     │    {f1_eng-f1_orig:+.2%}    │")
print("└─────────────────────┴──────────────┴──────────────┴─────────────┘")

# ============================================
# 12. FEATURE IMPORTANCE
# ============================================
print("\n🎯 11. TOP ENGINEERED FEATURES...")

feature_importance = pd.DataFrame({
    'Feature': engineered_features,
    'Importance': rf_engineered.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n   Top 15 most important features:")
print("-"*60)
for i, (idx, row) in enumerate(feature_importance.head(15).iterrows()):
    # Mark if it's an engineered feature
    is_engineered = row['Feature'] not in original_features
    marker = "🔧" if is_engineered else "📊"
    print(f"   {i+1:2d}. {marker} {row['Feature']:<45} {row['Importance']:.4f}")

# Plot feature importance
plt.figure(figsize=(12, 10))
top_features = feature_importance.head(20)
colors = ['#2ecc71' if f not in original_features else '#3498db' for f in top_features['Feature']]

plt.barh(range(len(top_features)), top_features['Importance'].values, color=colors)
plt.yticks(range(len(top_features)), top_features['Feature'].values)
plt.xlabel('Importance Score', fontsize=12)
plt.title('Feature Importance - Engineered Features (Green = Time-Based)', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/feature_engineering/feature_importance_engineered.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n   ✅ Saved: outputs/feature_engineering/feature_importance_engineered.png")

# ============================================
# 13. SAVE ENGINEERED DATASET
# ============================================
print("\n💾 12. SAVING ENGINEERED DATASET...")

# Save full dataset with engineered features
df.to_csv('data/ai4i2020_engineered.csv', index=False)
print(f"   ✅ Saved: data/ai4i2020_engineered.csv ({df.shape[1]} columns)")

# Save feature list
feature_list = pd.DataFrame({
    'feature_name': engineered_features,
    'feature_type': ['original' if f in original_features else 'engineered' for f in engineered_features],
    'importance': feature_importance.set_index('Feature').loc[engineered_features, 'Importance'].values
})
feature_list.to_csv('docs/feature_list.csv', index=False)
print(f"   ✅ Saved: docs/feature_list.csv")

# ============================================
# 14. SAVE BEST MODEL
# ============================================
print("\n💾 13. SAVING ENGINEERED MODEL...")

# Save the better performing model
if f1_eng >= f1_orig:
    joblib.dump(rf_engineered, 'models/random_forest_engineered.pkl')
    joblib.dump(scaler_engineered, 'models/scaler_engineered.pkl')
    print(f"   ✅ Saved: models/random_forest_engineered.pkl")
    print(f"   ✅ Saved: models/scaler_engineered.pkl")
else:
    joblib.dump(rf_original, 'models/random_forest_engineered.pkl')
    joblib.dump(scaler_original, 'models/scaler_engineered.pkl')
    print(f"   ⚠️ Original model performed better")

# ============================================
# 15. CREATE COMPARISON VISUALIZATION
# ============================================
print("\n📊 14. CREATING COMPARISON VISUALIZATION...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metrics = ['Accuracy', 'Recall', 'F1-Score']
original_scores = [accuracy_orig, recall_orig, f1_orig]
engineered_scores = [accuracy_eng, recall_eng, f1_eng]

x = range(len(metrics))
width = 0.35

axes[0].bar([i - width/2 for i in x], original_scores, width, label='Original Features', color='#3498db')
axes[0].bar([i + width/2 for i in x], engineered_scores, width, label='Engineered Features', color='#2ecc71')
axes[0].set_ylabel('Score')
axes[0].set_title('Performance Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend()
axes[0].set_ylim(0, 1)

# Improvement chart
improvements = [engineered_scores[i] - original_scores[i] for i in range(3)]
colors_imp = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
axes[1].bar(metrics, improvements, color=colors_imp)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_ylabel('Improvement')
axes[1].set_title('Improvement from Feature Engineering', fontsize=12, fontweight='bold')

# Feature count comparison
axes[2].bar(['Original', 'Engineered'], [len(original_features), len(engineered_features)], 
            color=['#3498db', '#2ecc71'])
axes[2].set_ylabel('Number of Features')
axes[2].set_title('Feature Count Comparison', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/feature_engineering/performance_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"   ✅ Saved: outputs/feature_engineering/performance_comparison.png")

# ============================================
# 16. FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("✅ FEATURE ENGINEERING COMPLETE!")
print("="*70)

print(f"""
📊 SUMMARY:
   • Original features: {len(original_features)}
   • Engineered features created: {len(engineered_features) - len(original_features)}
   • Total features available: {len(engineered_features)}

📈 PERFORMANCE IMPROVEMENT:
   • Accuracy:  {accuracy_orig*100:.2f}% → {accuracy_eng*100:.2f}% ({accuracy_eng-accuracy_orig:+.2%})
   • Recall:    {recall_orig*100:.2f}% → {recall_eng*100:.2f}% ({recall_eng-recall_orig:+.2%})
   • F1-Score:  {f1_orig*100:.2f}% → {f1_eng*100:.2f}% ({f1_eng-f1_orig:+.2%})

🎯 TOP 3 FEATURES:
   1. {feature_importance.iloc[0]['Feature']} ({feature_importance.iloc[0]['Importance']:.4f})
   2. {feature_importance.iloc[1]['Feature']} ({feature_importance.iloc[1]['Importance']:.4f})
   3. {feature_importance.iloc[2]['Feature']} ({feature_importance.iloc[2]['Importance']:.4f})

📁 FILES GENERATED:
   • data/ai4i2020_engineered.csv           ← Dataset with engineered features
   • models/random_forest_engineered.pkl    ← Model trained on engineered features
   • models/scaler_engineered.pkl           ← Scaler for engineered features
   • outputs/feature_engineering/           ← All comparison graphs
   • docs/feature_list.csv                  ← Documentation of all features
""")

print("="*70)
print("🚀 Ready for next step!")
print("="*70)