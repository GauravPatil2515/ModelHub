import pickle, os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Training Customer Churn Predictor...")
np.random.seed(42)
n = 1000
tenure        = np.random.randint(1, 72, n)
charges       = np.random.uniform(20, 120, n)
total         = tenure * charges + np.random.randn(n) * 50
products      = np.random.randint(1, 5, n)
calls         = np.random.poisson(2, n)
prob          = 1 / (1 + np.exp(-(-3.5 + 0.03*(charges-65) - 0.04*tenure + 0.3*calls - 0.2*products)))
churn         = (np.random.rand(n) < prob).astype(int)

X = np.column_stack([tenure, charges, total, products, calls])
X_tr, X_te, y_tr, y_te = train_test_split(X, churn, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42))
])
model.fit(X_tr, y_tr)
acc = round(accuracy_score(y_te, model.predict(X_te)) * 100, 1)

print(f"✅ Accuracy: {acc}%")
print(classification_report(y_te, model.predict(X_te), target_names=['Stay','Churn']))

os.makedirs('pkl', exist_ok=True)
pickle.dump(model, open('pkl/churn.pkl', 'wb'))
print(f"✅ Saved: models/pkl/churn.pkl")
