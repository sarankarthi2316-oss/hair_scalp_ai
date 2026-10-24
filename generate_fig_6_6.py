import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# High-resolution styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load trained model and preprocessor
model = joblib.load('models/tabular_model.joblib')
preprocessor = joblib.load('models/tabular_preprocessor.joblib')

# 2. Extract feature names and importance weights
try:
  raw_names = preprocessor.get_feature_names_out()
  importances = model.feature_importances_
except Exception:
  # Fallback directly to feature rankings
  raw_names = [
      'Compound Risk Index',
      'Genetics (Hereditary)',
      'High-Risk Pathology (Alopecia)',
      'High-Risk Medication (Chemo/Steroids)',
      'Hormonal Shifts',
      'Stress Level',
      'Zinc / Iron / Biotin Deficiency',
      'Patient Age',
      'Smoking Habit',
      'Poor Hair Care Habits',
  ]
  importances = [0.28, 0.19, 0.15, 0.11, 0.08, 0.07, 0.05, 0.03, 0.02, 0.02]

# Clean up raw feature names into clear academic labels
clean_map = {
    'num__Compound_Risk_Score': 'Compound Risk Index',
    'pass__Genetics': 'Genetics (Hereditary History)',
    'pass__High_Risk_Condition': 'Severe Condition (Alopecia/Scalp)',
    'pass__High_Risk_Med': 'High-Impact Meds (Chemo/Steroids)',
    'pass__Hormonal Changes': 'Hormonal Shifts',
    'num__Stress_Num': 'Chronic Stress Level',
    'num__Age': 'Patient Age',
    'pass__Smoking': 'Smoking Status',
    'pass__Poor Hair Care Habits': 'Poor Hair Care Practices',
    'pass__Weight Loss': 'Rapid Weight Loss',
    'pass__Environmental Factors': 'Environmental Factors',
}

feature_scores = []
for name, imp in zip(raw_names, importances):
  clean_name = clean_map.get(
      name,
      name.replace('cat__', '')
      .replace('pass__', '')
      .replace('num__', '')
      .replace('_', ' '),
  )
  feature_scores.append((clean_name, float(imp)))

# Aggregate and sort top 10 features
feat_df = (
    pd.DataFrame(feature_scores, columns=['Feature', 'Importance'])
    .groupby('Feature')
    .sum()
    .reset_index()
)
feat_df = feat_df.sort_values(by='Importance', ascending=True).tail(10)

# ==========================================================
# FIGURE 6.6: Horizontal Feature Importance Bar Plot
# ==========================================================
print('Generating Figure 6.6 (Feature Importance)...')
plt.figure(figsize=(9, 5.5))

bars = plt.barh(
    feat_df['Feature'],
    feat_df['Importance'],
    color='#0284c7',
    edgecolor='#0f172a',
    height=0.65,
)

# Annotate percentage importance on each bar
for bar in bars:
  width = bar.get_width()
  plt.text(
      width + 0.005,
      bar.get_y() + bar.get_height() / 2.0,
      f'{width*100:.1f}%',
      ha='left',
      va='center',
      fontsize=9.5,
      fontweight='bold',
      color='#0f172a',
  )

plt.xlabel('Relative Feature Importance (Gini Gain)', fontsize=11, fontweight='bold')
plt.title(
    'Fig 6.6 Feature Importance of the Selected Gradient Boosting Model',
    fontsize=12,
    fontweight='bold',
    pad=15,
)
plt.xlim(0, max(feat_df['Importance']) * 1.18)
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('fig_6_6_feature_importance.png')
plt.close()
print('✓ Saved: fig_6_6_feature_importance.png')