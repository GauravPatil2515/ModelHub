import pickle, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

print("Training Air Quality Index Predictor...")
np.random.seed(0)
n    = 800
co   = np.random.uniform(0.1, 10, n)
no2  = np.random.uniform(5, 200, n)
o3   = np.random.uniform(10, 180, n)
pm25 = np.random.uniform(5, 250, n)
temp = np.random.uniform(10, 45, n)
hum  = np.random.uniform(20, 95, n)
aqi  = (0.35*pm25 + 0.25*no2 + 0.20*o3 + 0.10*co*10
        + 0.05*temp - 0.05*hum + np.random.randn(n)*8).clip(0, 500)

X = np.column_stack([co, no2, o3, pm25, temp, hum])
X_tr, X_te, y_tr, y_te = train_test_split(X, aqi, test_size=0.2, random_state=42)

model = Pipeline([
    ('scaler', StandardScaler()),
    ('reg', RandomForestRegressor(n_estimators=200, random_state=42))
])
model.fit(X_tr, y_tr)
r2  = round(r2_score(y_te, model.predict(X_te)), 3)
mae = round(mean_absolute_error(y_te, model.predict(X_te)), 2)

print(f"✅ R² Score: {r2}")
print(f"✅ MAE:      {mae}")

os.makedirs('pkl', exist_ok=True)
pickle.dump(model, open('pkl/aqi.pkl', 'wb'))
print(f"✅ Saved: models/pkl/aqi.pkl")
