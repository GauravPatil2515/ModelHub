import pickle, os
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

print("Training Wine Cultivar Classifier...")
wine = load_wine()
X_tr, X_te, y_tr, y_te = train_test_split(wine.data, wine.target, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', SVC(kernel='rbf', C=10, probability=True, random_state=42))
])
model.fit(X_tr, y_tr)
acc = round(accuracy_score(y_te, model.predict(X_te)) * 100, 1)

print(f"✅ Accuracy: {acc}%")
print(classification_report(y_te, model.predict(X_te), target_names=wine.target_names))

os.makedirs('pkl', exist_ok=True)
pickle.dump(model, open('pkl/wine.pkl', 'wb'))
print(f"✅ Saved: models/pkl/wine.pkl")
