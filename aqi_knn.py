"""
Hourly AQI prediction with K-Nearest Neighbours.

CSV columns expected (names are cleaned automatically):
    date, time, location, pm2.5, o3, co, temp, humidity, aqi

Install:  pip install pandas numpy scikit-learn joblib
Run:      python aqi_knn.py
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

CSV_PATH = "aqi_data.csv"   # <- your file
DAYFIRST = True            # set True if your dates look like 31/12/2026
LAGS = [1, 2, 3]            # previous hours of AQI used as features
TEST_FRACTION = 0.2         # most recent 20% of data is held out for testing


# ---------------------------------------------------------------- load & clean
def load_data(path):
    df = pd.read_csv(path)
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.replace(".", "", regex=False).str.replace(" ", "_")
    )
    df = df.rename(columns={"temperature": "temp", "pm_25": "pm25"})

    df["datetime"] = pd.to_datetime(
        df["date"].astype(str) + " " + df["time"].astype(str),
        errors="coerce", dayfirst=DAYFIRST,
    )
    df = df.dropna(subset=["datetime"])
    df["location"] = df["location"].astype(str).str.strip()

    num_cols = ["pm25", "o3", "co", "temp", "humidity", "aqi"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["location", "datetime"]).reset_index(drop=True)
    # fill small gaps within each location, then drop whatever is left
    df[num_cols] = df.groupby("location")[num_cols].transform(
        lambda s: s.interpolate(limit=3, limit_direction="both")
    )
    return df.dropna(subset=num_cols)


# ---------------------------------------------------------------- features
def build_features(df, for_training=True):
    df = df.copy()
    df["hour"] = df["datetime"].dt.hour
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)   # 23:00 is close to 00:00
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dayofweek"] = df["datetime"].dt.dayofweek
    df["month"] = df["datetime"].dt.month

    g = df.groupby("location")["aqi"]
    for lag in LAGS:
        df[f"aqi_lag{lag}"] = g.shift(lag)

    df["target"] = g.shift(-1)          # AQI one hour ahead
    lag_cols = [f"aqi_lag{l}" for l in LAGS]
    if for_training:
        return df.dropna(subset=["target"] + lag_cols)
    return df.dropna(subset=lag_cols)   # latest rows have no target yet


NUMERIC = (["pm25", "o3", "co", "temp", "humidity", "aqi",
            "hour_sin", "hour_cos", "dayofweek", "month"]
           + [f"aqi_lag{l}" for l in LAGS])
CATEGORICAL = ["location"]


def make_pipeline():
    pre = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),          # KNN needs scaled features
        ("loc", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])
    return Pipeline([("prep", pre), ("knn", KNeighborsRegressor())])


# ---------------------------------------------------------------- train
def main():
    df = build_features(load_data(CSV_PATH))
    df = df.sort_values("datetime").reset_index(drop=True)

    split = int(len(df) * (1 - TEST_FRACTION))       # chronological split
    train, test = df.iloc[:split], df.iloc[split:]
    X_train, y_train = train[NUMERIC + CATEGORICAL], train["target"]
    X_test, y_test = test[NUMERIC + CATEGORICAL], test["target"]

    grid = GridSearchCV(
        make_pipeline(),
        param_grid={
            "knn__n_neighbors": [3, 5, 7, 9, 15, 21],
            "knn__weights": ["uniform", "distance"],
            "knn__p": [1, 2],                         # Manhattan / Euclidean
        },
        cv=TimeSeriesSplit(n_splits=5),
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)
    model = grid.best_estimator_
    print("Best parameters:", grid.best_params_)

    pred = model.predict(X_test)
    print(f"MAE : {mean_absolute_error(y_test, pred):.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, pred)):.2f}")
    print(f"R^2 : {r2_score(y_test, pred):.3f}")

    # baseline: "next hour = this hour"
    base = mean_absolute_error(y_test, test["aqi"])
    print(f"Baseline MAE (persistence): {base:.2f}")

    joblib.dump(model, "aqi_knn_model.joblib")
    print("Model saved to aqi_knn_model.joblib")

    # ---- forecast next hour for each location using its latest row
    full = build_features(load_data(CSV_PATH), for_training=False)
    latest = full.sort_values("datetime").groupby("location").tail(1)
    nxt = model.predict(latest[NUMERIC + CATEGORICAL])
    for loc, t, p in zip(latest["location"], latest["datetime"], nxt):
        print(f"{loc}: predicted AQI for {t + pd.Timedelta(hours=1)} -> {p:.0f}")


if __name__ == "__main__":
    main()
