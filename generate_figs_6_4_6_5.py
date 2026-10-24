import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

# High-resolution styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load data & trained model
df = pd.read_csv('Predict Hair Fall.csv')
df.columns = df.columns.str.strip()

X = df.drop(columns=['Id', 'Hair Loss'])
y = df['Hair Loss']

# Map binary features
binary_cols = [
    'Genetics',
    'Hormonal Changes',
    'Poor Hair Care Habits',
    'Environmental Factors',
    'Smoking',
    'Weight Loss',
]
for col in binary_cols:
  X[col] = X[col].map({'Yes': 1, 'No': 0}).fillna(0)

# Ordinal & engineered features
stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
X['Stress_Num'] = X['Stress'].map(stress_map).fillna(1)
X['Compound_Risk_Score'] = (
    X['Genetics'] * 2.0
    + X['Hormonal Changes'] * 1.5
    + X['Stress_Num'] * 1.0
    + X['Smoking'] * 0.8
    + X['Poor Hair Care Habits'] * 0.8
    + X['Weight Loss'] * 0.7
)

high_risk_conds = [
    'Alopecia Areata',
    'Androgenetic Alopecia',
    'Scalp Infection',
    'Seborrheic Dermatitis',
]
X['High_Risk_Condition'] = (
    X['Medical Conditions'].isin(high_risk_conds).astype(int)
)
X['High_Risk_Med'] = (
    X['Medications & Treatments']
    .isin(['Chemotherapy', 'Steroids', 'Accutane', 'Rogaine'])
    .astype(int)
)

cat_cols = [
    'Medical Conditions',
    'Medications & Treatments',
    'Nutritional Deficiencies',
]
X[cat_cols] = X[cat_cols].replace({'No Data': 'None'})

# Load trained pipeline
preprocessor = joblib.load('models/tabular_preprocessor.joblib')
model = joblib.load('models/tabular_model.joblib')

# Reproduce identical test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_test_proc = preprocessor.transform(X_test)

y_pred = model.predict(X_test_proc)
y_prob = model.predict_proba(X_test_proc)[:, 1]

# ==========================================================
# FIGURE 6.4: Confusion Matrix
# ==========================================================
print('Generating Figure 6.4 (Confusion Matrix)...')
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    cbar=False,
    xticklabels=['No Hair Fall (0)', 'Clinical Fall (1)'],
    yticklabels=['No Hair Fall (0)', 'Clinical Fall (1)'],
    annot_kws={'size': 14, 'fontweight': 'bold'},
)

plt.title(
    'Fig 6.4 Confusion Matrix - Gradient Boosting (Test Set)',
    fontsize=12,
    fontweight='bold',
    pad=12,
)
plt.xlabel('Predicted Class', fontsize=11, fontweight='bold')
plt.ylabel('Actual (Ground Truth) Class', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('fig_6_4_confusion_matrix.png')
plt.close()
print('✓ Saved: fig_6_4_confusion_matrix.png')

# ==========================================================
# FIGURE 6.5: ROC Curve
# ==========================================================
print('Generating Figure 6.5 (ROC Curve)...')
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6.5, 5.2))
plt.plot(
    fpr,
    tpr,
    color='#0284c7',
    lw=2.5,
    label=f'Gradient Boosting (AUC = {roc_auc:.4f})',
)
plt.plot(
    [0, 1],
    [0, 1],
    color='#94a3b8',
    lw=1.5,
    linestyle='--',
    label='Random Classifier (AUC = 0.5000)',
)
plt.fill_between(fpr, tpr, alpha=0.12, color='#0284c7')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=11, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11, fontweight='bold')
plt.title(
    'Fig 6.5 Receiver Operating Characteristic (ROC) Curve',
    fontsize=12,
    fontweight='bold',
    pad=12,
)
plt.legend(loc='lower right', fontsize=10, framealpha=0.9)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_6_5_roc_curve.png')
plt.close()
print('✓ Saved: fig_6_5_roc_curve.png')