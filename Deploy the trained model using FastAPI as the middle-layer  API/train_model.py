# train_model.py
import pickle
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Simulated churn dataset
np.random.seed(42)
n = 500

X = np.column_stack([
    np.random.randint(1, 72, n),        # tenure (months)
    np.random.uniform(20, 120, n),      # monthly_charges
    np.random.uniform(100, 8000, n),    # total_charges
    np.random.randint(0, 2, n),         # has_internet (0/1)
    np.random.randint(0, 2, n),         # has_phone (0/1)
])

# Churn is more likely if high charges + low tenure
churn_prob = (X[:, 1] / 120) * 0.5 + (1 - X[:, 0] / 72) * 0.5
y = (churn_prob + np.random.normal(0, 0.1, n) > 0.5).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))
print(f"Model Accuracy: {accuracy:.2f}")

# Save model
with open("model/churn_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved to model/churn_model.pkl")