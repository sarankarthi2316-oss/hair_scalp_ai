import joblib
import pandas as pd

# Load updated 95% accuracy model & preprocessor
preprocessor = joblib.load('models/tabular_preprocessor.joblib')
model = joblib.load('models/tabular_model.joblib')

HIGH_RISK_CONDITIONS = [
    'Alopecia Areata',
    'Androgenetic Alopecia',
    'Scalp Infection',
    'Seborrheic Dermatitis',
]
HIGH_RISK_MEDS = ['Chemotherapy', 'Steroids', 'Accutane', 'Rogaine']
STRESS_MAP = {'Low': 0, 'Moderate': 1, 'High': 2}


def predict_hair_loss_risk(form_data):
  """Takes user questionnaire responses, applies clinical feature engineering,

  and returns the predicted hair loss probability (0–100%).
  """
  genetics = 1 if form_data.get('genetics') == 'Yes' else 0
  hormonal = 1 if form_data.get('hormonal_changes') == 'Yes' else 0
  habits = 1 if form_data.get('poor_hair_care') == 'Yes' else 0
  smoking = 1 if form_data.get('smoking') == 'Yes' else 0
  weight_loss = 1 if form_data.get('weight_loss') == 'Yes' else 0
  stress_num = STRESS_MAP.get(form_data.get('stress', 'Moderate'), 1)

  med_condition = form_data.get('medical_condition', 'None')
  medication = form_data.get('medications_treatments', 'None')
  nutrition = form_data.get('nutritional_deficiencies', 'None')
  age = float(form_data.get('age', 25))

  # Compute engineered features
  compound_risk = (
      genetics * 2.0
      + hormonal * 1.5
      + stress_num * 1.0
      + smoking * 0.8
      + habits * 0.8
      + weight_loss * 0.7
  )
  high_risk_cond = 1 if med_condition in HIGH_RISK_CONDITIONS else 0
  high_risk_med = 1 if medication in HIGH_RISK_MEDS else 0

  data = {
      'Age': [age],
      'Stress_Num': [stress_num],
      'Compound_Risk_Score': [compound_risk],
      'Medical Conditions': [med_condition],
      'Medications & Treatments': [medication],
      'Nutritional Deficiencies': [nutrition],
      'Genetics': [genetics],
      'Hormonal Changes': [hormonal],
      'Poor Hair Care Habits': [habits],
      'Environmental Factors': [
          1 if form_data.get('environmental_factors') == 'Yes' else 0
      ],
      'Smoking': [smoking],
      'Weight Loss': [weight_loss],
      'High_Risk_Condition': [high_risk_cond],
      'High_Risk_Med': [high_risk_med],
  }

  df = pd.DataFrame(data)
  X_proc = preprocessor.transform(df)
  prob = model.predict_proba(X_proc)[0][1]

  return round(float(prob) * 100, 2)