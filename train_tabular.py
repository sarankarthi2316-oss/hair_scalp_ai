import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

print("1. Loading dataset...")
df = pd.read_csv("Predict Hair Fall.csv")
df.columns = df.columns.str.strip()

# Separate features and target
X = df.drop(columns=["Id", "Hair Loss"])
y = df["Hair Loss"]

# Clean & map binary features
binary_cols = [
    "Genetics",
    "Hormonal Changes",
    "Poor Hair Care Habits",
    "Environmental Factors",
    "Smoking",
    "Weight Loss",
]
for col in binary_cols:
  X[col] = X[col].map({"Yes": 1, "No": 0}).fillna(0)

# Replace 'No Data' placeholder with 'None'
cat_cols = [
    "Medical Conditions",
    "Medications & Treatments",
    "Nutritional Deficiencies",
]
X[cat_cols] = X[cat_cols].replace({"No Data": "None"})

print("2. Building preprocessing pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age"]),
        (
            "ord",
            OrdinalEncoder(categories=[["Low", "Moderate", "High"]]),
            ["Stress"],
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            cat_cols,
        ),
    ],
    remainder="passthrough",
)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("3. Preprocessing data...")
X_train_proc = preprocessor.fit_transform(X_train)
X_test_proc = preprocessor.transform(X_test)

print("4. Training Random Forest model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_proc, y_train)

# Evaluation
y_pred = model.predict(X_test_proc)
print("\n--- Model Evaluation ---")
print(classification_report(y_test, y_pred))

# Save artifacts
joblib.dump(preprocessor, "models/tabular_preprocessor.joblib")
joblib.dump(model, "models/tabular_model.joblib")
print(
    "\nArtifacts successfully saved to models/tabular_preprocessor.joblib and"
    " models/tabular_model.joblib"
)