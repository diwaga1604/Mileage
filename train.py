"""Train and compare regressors for MPG prediction; save the best full pipeline.

Usage: python train.py [path/to/auto-mpg.csv]
"""
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["cylinders", "displacement", "horsepower", "weight", "acceleration", "model year"]
CATEGORICAL = ["origin"]  # 1 = USA, 2 = Europe, 3 = Japan
TARGET = "mpg"


def load_data(path):
    df = pd.read_csv(path, na_values=["?", "NA", ""])
    # tolerate underscore column names (e.g. seaborn's mpg.csv)
    df.columns = [c.strip().replace("_", " ") for c in df.columns]
    df = df.rename(columns={"name": "car name"})
    if df["origin"].dtype == object:
        df["origin"] = df["origin"].str.lower().map({"usa": 1, "europe": 2, "japan": 3})
    df["origin"] = df["origin"].astype("Int64").astype(str)  # categorical
    df.loc[df["origin"] == "<NA>", "origin"] = np.nan
    for c in NUMERIC + [TARGET]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=[TARGET])  # cannot learn from rows without a target
    return df


def make_preprocessor():
    num = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    cat = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([("num", num, NUMERIC), ("cat", cat, CATEGORICAL)])


def main(path):
    df = load_data(path)
    X, y = df[NUMERIC + CATEGORICAL], df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": (LinearRegression(), {}),
        "Random Forest": (
            RandomForestRegressor(random_state=42),
            {"model__n_estimators": [200, 400], "model__max_depth": [None, 8, 12]},
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=42),
            {"model__n_estimators": [200, 400], "model__learning_rate": [0.03, 0.1],
             "model__max_depth": [2, 3]},
        ),
    }
    cv = KFold(5, shuffle=True, random_state=42)
    results, fitted = [], {}
    for name, (model, grid) in candidates.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", model)])
        if grid:
            search = GridSearchCV(pipe, grid, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=-1)
            search.fit(X_tr, y_tr)
            pipe = search.best_estimator_
        else:
            pipe.fit(X_tr, y_tr)
        cv_rmse = -cross_val_score(pipe, X_tr, y_tr, cv=cv, scoring="neg_root_mean_squared_error").mean()
        pred = pipe.predict(X_te)
        results.append({
            "model": name,
            "cv_rmse": round(cv_rmse, 3),
            "test_rmse": round(float(np.sqrt(mean_squared_error(y_te, pred))), 3),
            "test_mae": round(float(mean_absolute_error(y_te, pred)), 3),
            "test_r2": round(float(r2_score(y_te, pred)), 3),
        })
        fitted[name] = pipe

    res = pd.DataFrame(results).sort_values("cv_rmse")
    print(res.to_string(index=False))
    best = res.iloc[0]["model"]  # chosen on CV only, not on the test set
    print(f"\nBest model (lowest CV RMSE): {best}")

    final = fitted[best].fit(X, y)  # refit on all data for deployment
    joblib.dump(final, "mpg_model.joblib")
    meta = {
        "best_model": best,
        "results": results,
        "ranges": {c: [float(X[c].min()), float(X[c].max()), float(X[c].median())] for c in NUMERIC},
    }
    with open("model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "auto-mpg.csv")
