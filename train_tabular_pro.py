import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

print('1. Loading and engineering features...')
df = pd.read_csv('Predict Hair Fall.csv')
df.columns = df.columns.str.strip()

X = df.drop(columns=['Id', 'Hair Loss'])
y = df['Hair Loss']

# Binary mapping
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

# Ordinal mapping for Stress
stress_map = {'Low': 0, 'Moderate': 1, 'High': 2}
X['Stress_Num'] = X['Stress'].map(stress_map).fillna(1)

# Feature Engineering: Compounded Risk Score
X['Compound_Risk_Score'] = (
    X['Genetics'] * 2.0
    + X['Hormonal Changes'] * 1.5
    + X['Stress_Num'] * 1.0
    + X['Smoking'] * 0.8
    + X['Poor Hair Care Habits'] * 0.8
    + X['Weight Loss'] * 0.7
)

# High-Risk Clinical Condition Indicators
high_risk_conditions = [
    'Alopecia Areata',
    'Androgenetic Alopecia',
    'Scalp Infection',
    'Seborrheic Dermatitis',
]
X['High_Risk_Condition'] = (
    X['Medical Conditions'].isin(high_risk_conditions).astype(int)
)

high_risk_meds = ['Chemotherapy', 'Steroids', 'Accutane', 'Rogaine']
X['High_Risk_Med'] = (
    X['Medications & Treatments'].isin(high_risk_meds).astype(int)
)

# Clean missing categories
cat_cols = [
    'Medical Conditions',
    'Medications & Treatments',
    'Nutritional Deficiencies',
]
X[cat_cols] = X[cat_cols].replace({'No Data': 'None'})

# Preprocessing Pipeline
num_cols = ['Age', 'Stress_Num', 'Compound_Risk_Score']
passthrough_cols = [
    'Genetics',
    'Hormonal Changes',
    'Poor Hair Care Habits',
    'Environmental Factors',
    'Smoking',
    'Weight Loss',
    'High_Risk_Condition',
    'High_Risk_Med',
]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        (
            'cat',
            OneHotEncoder(handle_unknown='ignore', sparse_output=False),
            cat_cols,
        ),
        ('pass', 'passthrough', passthrough_cols),
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print('2. Training Tuned XGBoost Classifier...')
model = XGBClassifier(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.03,
    subsample=0.85,
    colsample_bytree=0.85,
    gamma=0.2,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    eval_metric='logloss',
)

model.fit(X_train_proc, y_train)

# Evaluation
y_pred = model.predict(X_test_proc)
y_prob = model.predict_proba(X_test_proc)[:, 1]

print('\n--- Enhanced Model Evaluation ---')
print(classification_report(y_test, y_pred))
print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}')

# Save optimized models
joblib.dump(preprocessor, 'models/tabular_preprocessor.joblib')
joblib.dump(model, 'models/tabular_model.joblib')
print('\n✓ Successfully saved optimized models to models/')