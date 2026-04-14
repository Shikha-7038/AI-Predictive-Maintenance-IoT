"""
AI-Powered Predictive Maintenance for IoT Devices
RANDOM FOREST MODEL TRAINING
Run this file to train your first failure prediction model
"""

# ============================================
# 1. IMPORT LIBRARIES
# ============================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report,
                             roc_curve, roc_auc_score)
from sklearn.preprocessing import StandardScaler
import warnings
import os
import joblib

warnings.filterwarnings('ignore')

# Create necessary folders
os.makedirs('outputs/model', exist_ok=True)


print("="*70)
print("🤖 RANDOM FOREST MODEL TRAINING - PREDICTIVE MAINTENANCE")
print("="*70)

# ============================================
# 2. LOAD THE DATASET
# ============================================
print("\n📂 1. LOADING DATASET...")
print("-"*50)

df = pd.read_csv('data/ai4i2020.csv')
print(f"✅ Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns")

# ============================================
# 3. PREPARE FEATURES AND TARGET
# ============================================
print("\n🔧 2. PREPARING FEATURES...")
print("-"*50)

# Select features (sensor readings and machine parameters)
feature_columns = [
    'Air temperature [K]',
    'Process temperature [K]', 
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Add failure type columns as features (they help predict future failures)
# But we need to be careful - these are also failure indicators
# For a realistic model, we'll use only sensor data
feature_columns_simple = [
    'Air temperature [K]',
    'Process temperature [K]', 
    'Rotational speed [rpm]',
    'Torque [Nm]',
    'Tool wear [min]'
]

# Target variable
target_column = 'Machine failure'

X = df[feature_columns_simple].copy()
y = df[target_column].copy()

print(f"✅ Features shape: {X.shape}")
print(f"✅ Target shape: {y.shape}")
print(f"\n📋 Features used:")
for col in feature_columns_simple:
    print(f"   • {col}")

print(f"\n📋 Target distribution:")
print(f"   Normal (0): {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"   Failure (1): {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)")

# ============================================
# 4. HANDLE CLASS IMBALANCE (IMPORTANT!)
# ============================================
print("\n⚖️ 3. HANDLING CLASS IMBALANCE...")
print("-"*50)

# Calculate class weights to give more importance to failure cases
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"✅ Class weights computed:")
print(f"   Normal (0): {weight_dict[0]:.3f}")
print(f"   Failure (1): {weight_dict[1]:.3f}")
print(f"   (Higher weight for failures = model pays more attention to them)")

# ============================================
# 5. SCALE FEATURES (IMPORTANT FOR SOME MODELS)
# ============================================
print("\n📏 4. SCALING FEATURES...")
print("-"*50)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use
joblib.dump(scaler, 'models/scaler.pkl')
print("✅ Scaler saved to: models/scaler.pkl")

print(f"\n📊 Feature statistics after scaling:")
print(f"   Mean of each feature: ~0")
print(f"   Standard deviation of each feature: ~1")

# ============================================
# 6. SPLIT DATA INTO TRAIN AND TEST SETS
# ============================================
print("\n✂️ 5. SPLITTING DATA...")
print("-"*50)

# Stratify ensures same failure percentage in train and test
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Training set: {X_train.shape[0]} samples")
print(f"✅ Test set: {X_test.shape[0]} samples")
print(f"\n📊 Training set failure rate: {y_train.mean()*100:.2f}%")
print(f"📊 Test set failure rate: {y_test.mean()*100:.2f}%")

# ============================================
# 7. TRAIN RANDOM FOREST MODEL (BASIC)
# ============================================
print("\n🌲 6. TRAINING RANDOM FOREST MODEL...")
print("-"*50)

# Create model with class_weight to handle imbalance
rf_basic = RandomForestClassifier(
    n_estimators=100,           # Number of trees
    max_depth=10,               # Maximum depth of each tree
    min_samples_split=5,        # Minimum samples to split a node
    min_samples_leaf=2,         # Minimum samples in a leaf
    class_weight=weight_dict,   # Handle class imbalance
    random_state=42,            # For reproducibility
    n_jobs=-1                   # Use all CPU cores
)

# Train the model
print("🔄 Training model... (this may take 10-20 seconds)")
rf_basic.fit(X_train, y_train)
print("✅ Model training complete!")

# Save the model
joblib.dump(rf_basic, 'models/random_forest_basic.pkl')
print("✅ Model saved to: models/random_forest_basic.pkl")

# ============================================
# 8. MAKE PREDICTIONS
# ============================================
print("\n🎯 7. MAKING PREDICTIONS...")
print("-"*50)

# Predictions on test set
y_pred_basic = rf_basic.predict(X_test)
y_pred_proba_basic = rf_basic.predict_proba(X_test)[:, 1]  # Probability of failure

print("✅ Predictions complete!")

# ============================================
# 9. EVALUATE MODEL PERFORMANCE
# ============================================
print("\n📊 8. MODEL EVALUATION...")
print("-"*50)

# Calculate metrics
accuracy_basic = accuracy_score(y_test, y_pred_basic)
precision_basic = precision_score(y_test, y_pred_basic)
recall_basic = recall_score(y_test, y_pred_basic)
f1_basic = f1_score(y_test, y_pred_basic)
auc_basic = roc_auc_score(y_test, y_pred_proba_basic)

print("\n📈 PERFORMANCE METRICS:")
print("-"*40)
print(f"   Accuracy:  {accuracy_basic*100:.2f}%")
print(f"   Precision: {precision_basic*100:.2f}%")
print(f"   Recall:    {recall_basic*100:.2f}%")
print(f"   F1-Score:  {f1_basic*100:.2f}%")
print(f"   AUC-ROC:   {auc_basic*100:.2f}%")
print("-"*40)

print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred_basic, 
                           target_names=['Normal (0)', 'Failure (1)']))

# ============================================
# 10. CONFUSION MATRIX
# ============================================
print("\n🔢 9. CONFUSION MATRIX...")
print("-"*50)

cm_basic = confusion_matrix(y_test, y_pred_basic)

print("\n   Predicted")
print("          Normal  Failure")
print(f"   Actual Normal:   {cm_basic[0,0]:5d}   {cm_basic[0,1]:5d}")
print(f"         Failure:   {cm_basic[1,0]:5d}   {cm_basic[1,1]:5d}")

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Confusion Matrix as numbers
sns.heatmap(cm_basic, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Normal (0)', 'Failure (1)'],
            yticklabels=['Normal (0)', 'Failure (1)'],
            ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

# Plot 2: Normalized confusion matrix
cm_norm = cm_basic.astype('float') / cm_basic.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
            xticklabels=['Normal (0)', 'Failure (1)'],
            yticklabels=['Normal (0)', 'Failure (1)'],
            ax=axes[1])
axes[1].set_title('Confusion Matrix (Percentages)', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig('outputs/model/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/model/confusion_matrix.png")

# ============================================
# 11. FEATURE IMPORTANCE ANALYSIS
# ============================================
print("\n🎯 10. FEATURE IMPORTANCE ANALYSIS...")
print("-"*50)

feature_importance = pd.DataFrame({
    'Feature': feature_columns_simple,
    'Importance': rf_basic.feature_importances_
}).sort_values('Importance', ascending=False)

print("\n📊 Feature Importance (higher = more important for prediction):")
for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['Importance'] * 50)
    print(f"   {row['Feature']:<30} {row['Importance']:.3f}  {bar}")

# Plot feature importance
plt.figure(figsize=(10, 6))
colors = plt.cm.RdYlGn_r(feature_importance['Importance'] / feature_importance['Importance'].max())
plt.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors)
plt.xlabel('Importance Score')
plt.title('Feature Importance - Random Forest Model', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('outputs/model/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✅ Saved: outputs/model/feature_importance.png")

# ============================================
# 12. ROC CURVE (Model Quality Visualization)
# ============================================
print("\n📈 11. ROC CURVE...")
print("-"*50)

fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba_basic)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'Random Forest (AUC = {auc_basic:.3f})')
plt.plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Classifier (AUC = 0.5)')
plt.xlabel('False Positive Rate (FPR)', fontsize=12)
plt.ylabel('True Positive Rate (TPR)', fontsize=12)
plt.title('ROC Curve - Model Performance', fontsize=14, fontweight='bold')
plt.legend(loc='lower right')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('outputs/model/roc_curve.png', dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: outputs/model/roc_curve.png")

# ============================================
# 13. CROSS-VALIDATION (Robustness Check)
# ============================================
print("\n🔄 12. CROSS-VALIDATION...")
print("-"*50)

cv_scores = cross_val_score(rf_basic, X_scaled, y, cv=5, scoring='f1')
print(f"\n📊 5-Fold Cross-Validation F1 Scores:")
for i, score in enumerate(cv_scores, 1):
    print(f"   Fold {i}: {score*100:.2f}%")
print(f"\n   Mean F1 Score: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")

# ============================================
# 14. HYPERPARAMETER TUNING (Optional - Improves Model)
# ============================================
print("\n🔧 13. HYPERPARAMETER TUNING...")
print("-"*50)

print("Would you like to tune hyperparameters for better performance?")
print("This takes 2-3 minutes but can improve accuracy by 5-10%")

# Uncomment the following code if you want to run hyperparameter tuning
"""
print("\n🔄 Running Grid Search for best parameters...")
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(
    RandomForestClassifier(class_weight=weight_dict, random_state=42),
    param_grid,
    cv=3,
    scoring='f1',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"\n✅ Best parameters: {grid_search.best_params_}")
print(f"✅ Best F1 score: {grid_search.best_score_*100:.2f}%")

# Train model with best parameters
rf_tuned = grid_search.best_estimator_
joblib.dump(rf_tuned, 'models/random_forest_tuned.pkl')
print("✅ Tuned model saved to: models/random_forest_tuned.pkl")

# Evaluate tuned model
y_pred_tuned = rf_tuned.predict(X_test)
y_pred_proba_tuned = rf_tuned.predict_proba(X_test)[:, 1]
accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
f1_tuned = f1_score(y_test, y_pred_tuned)

print(f"\n📈 Tuned Model Performance:")
print(f"   Accuracy: {accuracy_tuned*100:.2f}%")
print(f"   F1-Score: {f1_tuned*100:.2f}%")
"""

# ============================================
# 15. BASELINE COMPARISON (Simple Rule)
# ============================================
print("\n📊 14. BASELINE COMPARISON...")
print("-"*50)

# Simple baseline: Always predict "Normal"
baseline_accuracy = (y_test == 0).sum() / len(y_test)
baseline_recall = 0  # No failures detected

print(f"\n📈 Simple Baseline (Always predict 'Normal'):")
print(f"   Accuracy: {baseline_accuracy*100:.2f}%")
print(f"   Recall (failures found): 0.00%")

print(f"\n📈 Our Random Forest Model:")
print(f"   Accuracy: {accuracy_basic*100:.2f}%")
print(f"   Recall (failures found): {recall_basic*100:.2f}%")

improvement = ((accuracy_basic - baseline_accuracy) / baseline_accuracy) * 100
print(f"\n🎯 Improvement over baseline: +{improvement:.1f}%")

# ============================================
# 16. SAVE RESULTS SUMMARY
# ============================================
print("\n💾 15. SAVING RESULTS...")
print("-"*50)

# Create results dictionary
results = {
    'model_type': 'Random Forest',
    'accuracy': accuracy_basic,
    'precision': precision_basic,
    'recall': recall_basic,
    'f1_score': f1_basic,
    'auc_roc': auc_basic,
    'features_used': feature_columns_simple,
    'train_size': len(X_train),
    'test_size': len(X_test),
    'failure_rate_train': y_train.mean(),
    'failure_rate_test': y_test.mean(),
    'cross_val_f1_mean': cv_scores.mean(),
    'cross_val_f1_std': cv_scores.std()
}

# Save as JSON
import json
with open('outputs/model/model_results.json', 'w') as f:
    json.dump(results, f, indent=4)
print("✅ Results saved to: outputs/model/model_results.json")

# Create a text summary
with open('outputs/model/model_summary.txt', 'w') as f:
    f.write("="*60 + "\n")
    f.write("RANDOM FOREST MODEL - PREDICTIVE MAINTENANCE\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model Type: Random Forest Classifier\n")
    f.write(f"Training Samples: {len(X_train)}\n")
    f.write(f"Test Samples: {len(X_test)}\n\n")
    f.write(f"PERFORMANCE METRICS:\n")
    f.write(f"   Accuracy:  {accuracy_basic*100:.2f}%\n")
    f.write(f"   Precision: {precision_basic*100:.2f}%\n")
    f.write(f"   Recall:    {recall_basic*100:.2f}%\n")
    f.write(f"   F1-Score:  {f1_basic*100:.2f}%\n")
    f.write(f"   AUC-ROC:   {auc_basic*100:.2f}%\n\n")
    f.write(f"Cross-Validation (5-fold) F1: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)\n\n")
    f.write(f"FEATURE IMPORTANCE:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"   {row['Feature']}: {row['Importance']:.3f}\n")
print("✅ Summary saved to: outputs/model/model_summary.txt")

# ============================================
# 17. FINAL SUMMARY
# ============================================
print("\n" + "="*70)
print("🎉 RANDOM FOREST MODEL TRAINING COMPLETE!")
print("="*70)

print("""
📁 FILES GENERATED:
   models/
   ├── random_forest_basic.pkl    ← Trained model
   └── scaler.pkl                  ← Feature scaler
   
   outputs/
   ├── confusion_matrix.png        ← Prediction accuracy visualization
   ├── feature_importance.png      ← Which sensors matter most
   ├── roc_curve.png              ← Model quality curve
   ├── model_results.json         ← Numerical results
   └── model_summary.txt          ← Text summary

📊 MODEL PERFORMANCE SUMMARY:
""")
print(f"   ✅ Accuracy:  {accuracy_basic*100:.2f}%  (Higher is better)")
print(f"   ✅ Recall:    {recall_basic*100:.2f}%  (% of actual failures caught)")
print(f"   ✅ Precision: {precision_basic*100:.2f}%  (When it says failure, how often right)")
print(f"   ✅ F1-Score:  {f1_basic*100:.2f}%  (Balance of precision and recall)")

print("""
🎯 INTERPRETATION:
   • Accuracy tells us overall correctness
   • RECALL is CRITICAL for maintenance - we want to catch as many failures as possible
   • If recall is >80%, the model catches most failures before they happen
   • If precision is low, the model gives false alarms (still better than missing failures)

✅ This is your BASELINE model. You can now:
   1. Try different algorithms (XGBoost, LSTM)
   2. Add more features (time-based, rolling averages)
   3. Tune hyperparameters for better performance
""")

print("="*70)
print("🚀 Ready for next step: Testing predictions on new data!")
print("="*70)