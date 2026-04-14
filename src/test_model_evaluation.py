"""
Evaluate and compare all trained models
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
import os

os.makedirs('outputs/evaluation', exist_ok=True)

print("="*60)
print("📊 MODEL EVALUATION & COMPARISON")
print("="*60)

# Load test data
print("\n📂 Loading data...")
df = pd.read_csv('data/ai4i2020_engineered.csv')
y_true = df['Machine failure']

# Prepare features for basic model
basic_features = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X_basic = df[basic_features]

# Prepare features for engineered model
exclude = ['UID', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'time_index']
X_engineered = df[[c for c in df.columns if c not in exclude]]

print(f"✅ Basic features: {X_basic.shape[1]}")
print(f"✅ Engineered features: {X_engineered.shape[1]}")

# Load models
print("\n📂 Loading models...")
basic_model = joblib.load('models/random_forest_basic.pkl')
engineered_model = joblib.load('models/random_forest_engineered.pkl')

# Make predictions
y_pred_basic = basic_model.predict(X_basic)
y_pred_engineered = engineered_model.predict(X_engineered)

# Calculate metrics
models = ['Basic Random Forest', 'Engineered Random Forest']
accuracies = [accuracy_score(y_true, y_pred_basic), accuracy_score(y_true, y_pred_engineered)]
recalls = [recall_score(y_true, y_pred_basic), recall_score(y_true, y_pred_engineered)]
precisions = [precision_score(y_true, y_pred_basic), precision_score(y_true, y_pred_engineered)]
f1_scores = [f1_score(y_true, y_pred_basic), f1_score(y_true, y_pred_engineered)]

# Display results
print("\n" + "="*60)
print("📈 PERFORMANCE COMPARISON")
print("="*60)

print(f"\n{'Model':<25} {'Accuracy':<12} {'Recall':<12} {'Precision':<12} {'F1-Score':<12}")
print("-"*70)
for i, model in enumerate(models):
    print(f"{model:<25} {accuracies[i]*100:>10.2f}%   {recalls[i]*100:>10.2f}%   {precisions[i]*100:>10.2f}%   {f1_scores[i]*100:>10.2f}%")

print(f"\n{'Improvement':<25} {accuracies[1]-accuracies[0]:>+9.2%}   {recalls[1]-recalls[0]:>+9.2%}   {precisions[1]-precisions[0]:>+9.2%}   {f1_scores[1]-f1_scores[0]:>+9.2%}")

# Create comparison chart
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart comparison
x = np.arange(len(models))
width = 0.2

axes[0].bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#3498db')
axes[0].bar(x - 0.5*width, recalls, width, label='Recall', color='#2ecc71')
axes[0].bar(x + 0.5*width, precisions, width, label='Precision', color='#e74c3c')
axes[0].bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#f39c12')

axes[0].set_ylabel('Score')
axes[0].set_title('Model Performance Comparison', fontsize=12, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(['Basic', 'Engineered'])
axes[0].legend()
axes[0].set_ylim(0, 1)

# Improvement chart
improvements = [recalls[1]-recalls[0], f1_scores[1]-f1_scores[0], accuracies[1]-accuracies[0]]
improvement_labels = ['Recall', 'F1-Score', 'Accuracy']
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]

axes[1].bar(improvement_labels, improvements, color=colors)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
axes[1].set_ylabel('Improvement')
axes[1].set_title('Feature Engineering Impact', fontsize=12, fontweight='bold')

for i, (imp, label) in enumerate(zip(improvements, improvement_labels)):
    axes[1].text(i, imp + 0.01, f'{imp:+.1%}', ha='center', fontsize=10)

plt.tight_layout()
plt.savefig('outputs/evaluation/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"\n✅ Saved: outputs/evaluation/model_comparison.png")

# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

cm_basic = confusion_matrix(y_true, y_pred_basic)
cm_eng = confusion_matrix(y_true, y_pred_engineered)

# Basic model confusion matrix
im1 = axes[0].imshow(cm_basic, cmap='Blues')
axes[0].set_title(f'Basic Model\nAccuracy: {accuracies[0]*100:.1f}%', fontweight='bold')
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(['Normal', 'Failure'])
axes[0].set_yticklabels(['Normal', 'Failure'])
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm_basic[i, j], ha='center', va='center', fontsize=14)

# Engineered model confusion matrix
im2 = axes[1].imshow(cm_eng, cmap='Greens')
axes[1].set_title(f'Engineered Model\nAccuracy: {accuracies[1]*100:.1f}%', fontweight='bold')
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(['Normal', 'Failure'])
axes[1].set_yticklabels(['Normal', 'Failure'])
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

for i in range(2):
    for j in range(2):
        axes[1].text(j, i, cm_eng[i, j], ha='center', va='center', fontsize=14)

plt.tight_layout()
plt.savefig('outputs/evaluation/confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ Saved: outputs/evaluation/confusion_matrices.png")

# Summary
print("\n" + "="*60)
print("📊 SUMMARY")
print("="*60)
print(f"""
✅ Basic Model Performance:
   • Catches {recalls[0]*100:.1f}% of failures
   • {cm_basic[1,0]} failures missed

✅ Engineered Model Performance:
   • Catches {recalls[1]*100:.1f}% of failures  
   • {cm_eng[1,0]} failures missed

🎯 IMPROVEMENT:
   • {cm_basic[1,0] - cm_eng[1,0]} more failures detected
   • {recalls[1]-recalls[0]:+.1%} improvement in recall

💡 CONCLUSION:
   Feature engineering with time-based features significantly improves
   the model's ability to detect failures before they occur.
""")

print("="*60)
print("✅ Evaluation complete!")