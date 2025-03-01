import pandas as pd
import xgboost as xgb
import pickle
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the datasets
file1 = "general_information.xlsx"
file2 = "general_and_medical_information.xlsx"
df1 = pd.read_excel(file1)
df2 = pd.read_excel(file2)

# Ensure target column exists
target_col = 'PCOS'
if target_col not in df1.columns or target_col not in df2.columns:
    raise ValueError(f"Column '{target_col}' not found in datasets!")

# Function to preprocess data
def preprocess_data(df, target):
    df.fillna(df.median(numeric_only=True), inplace=True)
    label_encoders = {}
    for col in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    X = df.drop(columns=[target])
    y = df[target]
    return X, y, label_encoders

# Train Model 1
X1, y1, label_encoders1 = preprocess_data(df1, target_col)
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
model1 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model1.fit(X1_train, y1_train)
preds1 = model1.predict(X1_test)
print(f"Model 1 Accuracy: {accuracy_score(y1_test, preds1):.4f}")

# Train Model 2
X2, y2, label_encoders2 = preprocess_data(df2, target_col)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
model2 = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss')
model2.fit(X2_train, y2_train)
preds2 = model2.predict(X2_test)
print(f"Model 2 Accuracy: {accuracy_score(y2_test, preds2):.4f}")

# Save Models
with open("general_information.pkl", "wb") as f:
    pickle.dump(model1, f, protocol=pickle.HIGHEST_PROTOCOL)
with open("general_and_medical_information.pkl", "wb") as f:
    pickle.dump(model2, f, protocol=pickle.HIGHEST_PROTOCOL)

# Save using joblib (recommended for large models)
joblib.dump(model1, "general_information.joblib")
joblib.dump(model2, "general_and_medical_information.joblib")

print("Models trained and saved successfully!")
