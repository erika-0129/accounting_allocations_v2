"""
Training set for AI integration.
Will eventually be used to replace the Gemini API calls.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

# Sample dataset (replace with actual transaction data)
data = {
    "Vendor": ["Amazon", "Walmart", "Starbucks", "Uber", "Shell", "Netflix"],
    "Amount": [50.25, 120.99, 5.75, 15.60, 40.00, 9.99],
    "Description": [
        "Amazon Purchase - Electronics",
        "Grocery shopping at Walmart",
        "Morning coffee at Starbucks",
        "Uber ride to downtown",
        "Gas station refuel",
        "Monthly Netflix subscription"
    ],
    "Category": ["Shopping", "Groceries", "Dining", "Transport", "Gas", "Entertainment"]
}

df = pd.DataFrame(data)

# Encode categories
label_encoder = LabelEncoder()
df["Category_encoded"] = label_encoder.fit_transform(df["Category"])

# Text feature extraction
vectorizer = TfidfVectorizer()
X_text = vectorizer.fit_transform(df["Description"])

# Combine features
X = np.hstack((X_text.toarray(), df[["Amount"]].values))
y = df["Category_encoded"].values

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoders
joblib.dump(model, "transaction_classifier.pkl")
joblib.dump(label_encoder, "label_encoder.pkl")
joblib.dump(vectorizer, "vectorizer.pkl")


# Function to classify new transactions
def classify_transaction(vendor, amount, description):
    model = joblib.load("transaction_classifier.pkl")
    label_encoder = joblib.load("label_encoder.pkl")
    vectorizer = joblib.load("vectorizer.pkl")

    X_new_text = vectorizer.transform([description])
    X_new = np.hstack((X_new_text.toarray(), [[amount]]))
    category_encoded = model.predict(X_new)[0]
    category = label_encoder.inverse_transform([category_encoded])[0]

    return category


# Example usage
print(classify_transaction("Uber", 20.00, "Rideshare trip to airport"))
