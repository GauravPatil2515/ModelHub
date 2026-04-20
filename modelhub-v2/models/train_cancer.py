import pickle, os
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Training Breast Cancer Detector...")
bc = load_breast_cancer()
X_tr, X_te, y_tr, y_te = train_test_split(bc.data, bc.target, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', GradientBoostingClassifier(n_estimators=150, random_state=42))
])
model.fit(X_tr, y_tr)
acc = round(accuracy_score(y_te, model.predict(X_te)) * 100, 1)

print(f"✅ Accuracy: {acc}%")
print(classification_report(y_te, model.predict(X_te), target_names=bc.target_names))

os.makedirs('pkl', exist_ok=True)
pickle.dump(model, open('pkl/cancer.pkl', 'wb'))
print(f"✅ Saved: models/pkl/cancer.pkl")
