"""
Generate model comparison image for README
Fixed version - no pie chart errors
"""

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import recall_score, accuracy_score, f1_score
import warnings
warnings.filterwarnings('ignore')

print("="*50)
print("📊 Generating Model Comparison Images")
print("="*50)

# Load data
print("\n📂 Loading data...")
df = pd.read_csv('data/ai4i2020_engineered.csv')
y_true = df['Machine failure']
print(f"   Dataset: {len(df)} rows")

# Basic features (original 5 sensors)
basic_features = ['Air temperature [K]', 'Process temperature [K]', 
                  'Rotational speed [rpm]', 'Torque [Nm]', 'Tool wear [min]']
X_basic = df[basic_features]

# Engineered features (all except metadata)
exclude = ['UID', 'Product ID', 'Type', 'Machine failure', 'TWF', 'HDF', 'PWF', 'OSF', 'RNF', 'time_index']
X_eng = df[[c for c in df.columns if c not in exclude]]

print(f"   Basic features: {X_basic.shape[1]}")
print(f"   Engineered features: {X_eng.shape[1]}")

# Load models
print("\n📂 Loading models...")
basic_model = joblib.load('models/random_forest_basic.pkl')
eng_model = joblib.load('models/random_forest_engineered.pkl')
print("   ✅ Models loaded")

# Make predictions
print("\n🔄 Making predictions...")
y_pred_basic = basic_model.predict(X_basic)
y_pred_eng = eng_model.predict(X_eng)

# Calculate metrics
recall_basic = recall_score(y_true, y_pred_basic)
recall_eng = recall_score(y_true, y_pred_eng)
accuracy_basic = accuracy_score(y_true, y_pred_basic)
accuracy_eng = accuracy_score(y_true, y_pred_eng)
f1_basic = f1_score(y_true, y_pred_basic)
f1_eng = f1_score(y_true, y_pred_eng)

print("\n" + "="*50)
print("📈 PERFORMANCE METRICS")
print("="*50)
print(f"\nBasic Model:")
print(f"   Recall: {recall_basic*100:.1f}%")
print(f"   Accuracy: {accuracy_basic*100:.1f}%")
print(f"   F1-Score: {f1_basic*100:.1f}%")
print(f"\nEngineered Model:")
print(f"   Recall: {recall_eng*100:.1f}%")
print(f"   Accuracy: {accuracy_eng*100:.1f}%")
print(f"   F1-Score: {f1_eng*100:.1f}%")
print(f"\nImprovement:")
print(f"   Recall: +{(recall_eng - recall_basic)*100:.1f}%")
print(f"   Accuracy: +{(accuracy_eng - accuracy_basic)*100:.1f}%")
print(f"   F1-Score: +{(f1_eng - f1_basic)*100:.1f}%")

# ============================================
# CREATE COMPARISON BAR CHART
# ============================================
print("\n📊 Creating comparison bar chart...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Metric Comparison
metrics = ['Recall', 'Accuracy', 'F1-Score']
basic_scores = [recall_basic, accuracy_basic, f1_basic]
engineered_scores = [recall_eng, accuracy_eng, f1_eng]

x = np.arange(len(metrics))
width = 0.35

bars1 = axes[0].bar(x - width/2, basic_scores, width, label='Basic Model (5 features)', color='#3498db', edgecolor='black')
bars2 = axes[0].bar(x + width/2, engineered_scores, width, label='Engineered Model (80+ features)', color='#2ecc71', edgecolor='black')

axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(metrics)
axes[0].legend(loc='lower right')
axes[0].set_ylim(0, 1)
axes[0].grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height*100:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Chart 2: Improvement Chart
improvements = [
    recall_eng - recall_basic,
    accuracy_eng - accuracy_basic,
    f1_eng - f1_basic
]
colors = ['#2ecc71' if x > 0 else '#e74c3c' for x in improvements]
bars3 = axes[1].bar(metrics, improvements, color=colors, edgecolor='black', linewidth=1.5)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1].set_ylabel('Improvement', fontsize=12)
axes[1].set_title('Feature Engineering Impact', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, imp in zip(bars3, improvements):
    axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{imp*100:+.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('images/model_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: images/model_comparison.png")

# ============================================
# CREATE SUMMARY TABLE IMAGE
# ============================================
print("\n📊 Creating summary table...")

fig, ax = plt.subplots(figsize=(10, 4))
ax.axis('off')

# Create table data
table_data = [
    ['Metric', 'Basic Model', 'Engineered Model', 'Improvement'],
    ['Recall (Failures Caught)', f'{recall_basic*100:.1f}%', f'{recall_eng*100:.1f}%', f'+{(recall_eng-recall_basic)*100:.1f}%'],
    ['Accuracy', f'{accuracy_basic*100:.1f}%', f'{accuracy_eng*100:.1f}%', f'+{(accuracy_eng-accuracy_basic)*100:.1f}%'],
    ['F1-Score', f'{f1_basic*100:.1f}%', f'{f1_eng*100:.1f}%', f'+{(f1_eng-f1_basic)*100:.1f}%'],
    ['Features Used', '5', f'{X_eng.shape[1]}', f'+{X_eng.shape[1]-5}'],
]

# Create table
table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

# Color header row
for i in range(4):
    table[(0, i)].set_facecolor('#2c3e50')
    table[(0, i)].set_text_props(weight='bold', color='white')

# Color improvement column
for i in range(1, 5):
    if '+' in table_data[i][3]:
        table[(i, 3)].set_facecolor('#2ecc71')
        table[(i, 3)].set_text_props(weight='bold')

ax.set_title('Model Performance Summary', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('images/performance_summary.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: images/performance_summary.png")

# ============================================
# CREATE SIMPLE INFOGRAPHIC
# ============================================
print("\n📊 Creating infographic...")

fig, ax = plt.subplots(figsize=(10, 5))
ax.axis('off')

# Title
ax.text(0.5, 0.95, '🎯 KEY ACHIEVEMENT', transform=fig.transFigure, 
        fontsize=16, fontweight='bold', ha='center', color='#2c3e50')

# Main metric
ax.text(0.5, 0.7, f'+{(recall_eng-recall_basic)*100:.1f}%', transform=fig.transFigure,
        fontsize=48, fontweight='bold', ha='center', color='#2ecc71')

ax.text(0.5, 0.55, 'IMPROVEMENT IN FAILURE DETECTION', transform=fig.transFigure,
        fontsize=12, ha='center', color='#555555')

# Sub metrics
ax.text(0.25, 0.35, f'{recall_basic*100:.1f}%', transform=fig.transFigure,
        fontsize=24, fontweight='bold', ha='center', color='#3498db')
ax.text(0.25, 0.22, 'Basic Model\nRecall', transform=fig.transFigure,
        fontsize=10, ha='center', color='#555555')

ax.text(0.5, 0.35, '→', transform=fig.transFigure,
        fontsize=30, ha='center', color='#2c3e50')

ax.text(0.75, 0.35, f'{recall_eng*100:.1f}%', transform=fig.transFigure,
        fontsize=24, fontweight='bold', ha='center', color='#2ecc71')
ax.text(0.75, 0.22, 'Engineered Model\nRecall', transform=fig.transFigure,
        fontsize=10, ha='center', color='#555555')

# Bottom note
ax.text(0.5, 0.08, 'Feature engineering with time-based features (rolling stats, rate of change)\nsignificantly improves failure detection capability', 
        transform=fig.transFigure, fontsize=9, ha='center', color='#888888', style='italic')

plt.savefig('images/key_achievement.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.close()
print("✅ Saved: images/key_achievement.png")

# ============================================
# FINAL SUMMARY
# ============================================
print("\n" + "="*50)
print("✅ ALL IMAGES GENERATED SUCCESSFULLY!")
print("="*50)
print("\n📁 Images saved in 'images/' folder:")
print("   1. model_comparison.png     - Bar chart comparison")
print("   2. performance_summary.png  - Results table")
print("   3. key_achievement.png      - Infographic")
print("\n📊 Performance Summary:")
print(f"   Basic Model Recall:     {recall_basic*100:.1f}%")
print(f"   Engineered Model Recall: {recall_eng*100:.1f}%")
print(f"   IMPROVEMENT:            +{(recall_eng-recall_basic)*100:.1f}%")
print("\n🚀 Ready to add these images to your README!")