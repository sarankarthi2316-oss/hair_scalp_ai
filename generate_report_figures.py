import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Set high-resolution academic styling
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.dpi'] = 300

# 1. Load data
df = pd.read_csv('Predict Hair Fall.csv')
df.columns = df.columns.str.strip()

# ==========================================================
# FIGURE 6.1: Distribution of Target Variable
# ==========================================================
print('Generating Figure 6.1 (Target Distribution)...')
plt.figure(figsize=(7, 4.5))

counts = df['Hair Loss'].value_counts()
labels = [
    'No Significant Hair Fall (Class 0)',
    'Clinical Hair Fall (Class 1)',
]
colors = ['#2563eb', '#ef4444']

ax = sns.barplot(
    x=counts.index, y=counts.values, palette=colors, edgecolor='#0f172a'
)

# Annotate counts and percentages on bars
total = len(df)
for i, p in enumerate(ax.patches):
  height = p.get_height()
  ax.annotate(
      f'{int(height)} ({height/total*100:.1f}%)',
      (p.get_x() + p.get_width() / 2.0, height / 2.0),
      ha='center',
      va='center',
      xytext=(0, 0),
      textcoords='offset points',
      fontsize=12,
      fontweight='bold',
      color='white',
  )

plt.title(
    'Distribution of Target Variable (Hair Loss Status)',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.xlabel('Diagnosis Classification', fontsize=11, fontweight='bold')
plt.ylabel('Patient Record Count', fontsize=11, fontweight='bold')
plt.xticks(ticks=[0, 1], labels=labels, fontsize=11)
plt.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('fig_6_1_target_distribution.png')
plt.close()
print('✓ Saved: fig_6_1_target_distribution.png')

# ==========================================================
# FIGURE 6.2: Correlation Heat-map of Input Features
# ==========================================================
print('Generating Figure 6.2 (Correlation Heat-map)...')
plt.figure(figsize=(10, 8))

# Map features to numeric for correlation matrix
corr_df = pd.DataFrame()
corr_df['Age'] = df['Age']

binary_cols = [
    'Genetics',
    'Hormonal Changes',
    'Poor Hair Care Habits',
    'Smoking',
    'Weight Loss',
    'Environmental Factors',
]
for col in binary_cols:
  corr_df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)

corr_df['Stress'] = df['Stress'].map({'Low': 0, 'Moderate': 1, 'High': 2})

# Key clinical indicators
high_risk_conds = [
    'Alopecia Areata',
    'Androgenetic Alopecia',
    'Scalp Infection',
    'Seborrheic Dermatitis',
]
corr_df['High_Risk_Condition'] = (
    df['Medical Conditions'].isin(high_risk_conds).astype(int)
)
corr_df['Chemo_or_Steroids'] = (
    df['Medications & Treatments']
    .isin(['Chemotherapy', 'Steroids', 'Accutane'])
    .astype(int)
)
corr_df['Hair_Loss_Target'] = df['Hair Loss']

# Compute Correlation Matrix
corr_matrix = corr_df.corr()

# Rename columns for cleaner academic presentation
clean_labels = [
    'Age',
    'Genetics',
    'Hormonal Shifts',
    'Poor Hair Care',
    'Smoking',
    'Weight Loss',
    'Environmental',
    'Stress Level',
    'Severe Condition',
    'High-Risk Meds',
    'Hair Loss (Target)',
]

sns.heatmap(
    corr_matrix,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    vmin=-0.2,
    vmax=0.8,
    xticklabels=clean_labels,
    yticklabels=clean_labels,
    linewidths=0.6,
    cbar_kws={'label': 'Pearson Correlation Coefficient (r)'},
    annot_kws={'size': 9, 'fontweight': 'bold'},
)

plt.title(
    'Feature Correlation Matrix (Clinical & Lifestyle Factors)',
    fontsize=14,
    fontweight='bold',
    pad=15,
)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.savefig('fig_6_2_correlation_heatmap.png')
plt.close()
print('✓ Saved: fig_6_2_correlation_heatmap.png')