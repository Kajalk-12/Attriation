import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("employee_attrition.csv")  # Replace with your actual dataset

# Encode categorical variables
label_encoders = {}
for col in ['JobSatisfaction', 'WorkLifeBalance', 'PerformanceRating', 'OverTime',
            'RelationshipSatisfaction', 'CareerGrowthOpportunity', 'StockOptionLevel', 'JobLevel']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for later use

# Define features and target
X = df.drop(columns=['Attrition'])  # Features
y = df['Attrition']  # Target (Attrition: Yes/No)

# Train a model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save the trained model and encoders
joblib.dump(model, "attrition_model.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")