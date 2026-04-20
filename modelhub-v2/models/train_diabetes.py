import pickle, os
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("Training Diabetes Progression Predictor...")
diab = load_diabetes()
X_tr, X_te, y_tr, y_te = train_test_split(diab.data, diab.target, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42))
])
model.fit(X_tr, y_tr)
r2  = round(r2_score(y_te, model.predict(X_te)), 3)
mae = round(mean_absolute_error(y_te, model.predict(X_te)), 2)

print(f"✅ R² Score: {r2}")
print(f"✅ MAE:      {mae}")

os.makedirs('pkl', exist_ok=True)
pickle.dump(model, open('pkl/diabetes.pkl', 'wb'))
print(f"✅ Saved: models/pkl/diabetes.pkl")
