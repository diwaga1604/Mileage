from pathlib import Path
import json
import sys
import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

BASE = Path(__file__).resolve().parent
csv_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else BASE / 'auto-mpg.csv'
df = pd.read_csv(csv_path)
df.columns = [str(c).strip().lower().replace('_', ' ') for c in df.columns]
df = df.loc[:, ~df.columns.str.startswith('unnamed')]
features = ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'origin']
required = features + ['mpg']
missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError('Missing CSV columns: ' + ', '.join(missing))
df = df[required].copy()
for c in ['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model year', 'mpg']:
    df[c] = pd.to_numeric(df[c].replace('?', np.nan), errors='coerce')
df['origin'] = df['origin'].astype('string').str.strip()
df = df.dropna(subset=['mpg'])
if len(df) < 10:
    raise ValueError('Please provide at least 10 valid rows in auto-mpg.csv')
X, y = df[features], df['mpg']
numeric = features[:-1]
preprocess = ColumnTransformer([
    ('num', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), ['origin'])
])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
models = {'Linear Regression': LinearRegression(), 'Random Forest': RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1), 'Gradient Boosting': GradientBoostingRegressor(random_state=42)}
results, fitted = [], []
for name, estimator in models.items():
    pipe = Pipeline([('preprocess', preprocess), ('model', estimator)])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    mae = float(mean_absolute_error(y_test, pred))
    results.append({'model': name, 'MAE': round(mae, 3), 'RMSE': round(float(np.sqrt(mean_squared_error(y_test, pred))), 3), 'R2': round(float(r2_score(y_test, pred)), 3)})
    fitted.append((mae, name, pipe))
_, best_name, best_model = min(fitted, key=lambda x: x[0])
joblib.dump(best_model, BASE / 'mpg_model.joblib')
ranges = {}
for c in numeric:
    vals = df[c].dropna()
    ranges[c] = [float(vals.min()), float(vals.median()), float(vals.max())] if not vals.empty else [0.0, 0.0, 0.0]
meta = {'best_model': best_name, 'results': results, 'ranges': ranges, 'data_rows': int(len(df)), 'data_note': 'Trained from the supplied auto-mpg.csv dataset.'}
(BASE / 'model_meta.json').write_text(json.dumps(meta, indent=2), encoding='utf-8')
print('Training complete! Best model:', best_name)
print(pd.DataFrame(results).to_string(index=False))
print('Created mpg_model.joblib and model_meta.json in', BASE)