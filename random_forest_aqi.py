# Chennai AQI Prediction using Random Forest Regression
# Put "Processed Data.xlsx" in the same folder as this file.
# Run: python3 random_forest_aqi.py

from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

DATA_FILE = Path("Processed Data.xlsx")
if not DATA_FILE.exists():
    raise FileNotFoundError(
        'Could not find "Processed Data.xlsx". Put it in the same folder as this Python file.'
    )

df = pd.read_excel(DATA_FILE)
df.columns = df.columns.astype(str).str.strip()
print("Original dataset shape:", df.shape)
print("Columns:", df.columns.tolist())

required = {"Location", "AQI"}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Keep Chennai records and remove exact duplicates.
df["Location_clean"] = df["Location"].astype(str).str.strip().str.lower()
df = df[df["Location_clean"] == "chennai"].copy()
before = len(df)
df = df.drop_duplicates()
print("Exact duplicate rows removed:", before - len(df))

# Use whichever of these columns are present in the dataset.
candidate_features = ["PM2.5", "CO", "O3", "Temp", "Humidity"]
features = [column for column in candidate_features if column in df.columns]
if not features:
    raise ValueError("No expected predictor columns found. Check the column names printed above.")

for column in features + ["AQI"]:
    df[column] = pd.to_numeric(df[column], errors="coerce")

before = len(df)
df = df.dropna(subset=features + ["AQI"])
print("Rows removed due to missing/non-numeric values:", before - len(df))

# This follows the original project's cleaning rule; change if zero is valid in your source.
before = len(df)
df = df[df["AQI"] > 0].copy()
print("Rows removed because AQI <= 0:", before - len(df))

if len(df) < 2:
    raise ValueError("Not enough valid Chennai rows to train and test the model.")

X = df[features]
y = df["AQI"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Features used: {features}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")

model = RandomForestRegressor(
    n_estimators=300,
    max_features=1.0,
    min_samples_leaf=2,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n--- Random Forest Evaluation ---")
print(f"MAE : {mae:.3f}")
print(f"MSE : {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²  : {r2:.3f}")

pd.DataFrame({
    "Actual_AQI": y_test.to_numpy(),
    "Predicted_AQI": y_pred,
}).to_csv("random_forest_test_predictions.csv", index=False)
print("Saved: random_forest_test_predictions.csv")

plt.figure(figsize=(7, 6))
plt.scatter(y_test, y_pred)
lower = min(float(y_test.min()), float(y_pred.min()))
upper = max(float(y_test.max()), float(y_pred.max()))
plt.plot([lower, upper], [lower, upper], linestyle="--")
plt.xlabel("Actual AQI")
plt.ylabel("Predicted AQI")
plt.title("Random Forest: Actual vs Predicted AQI")
plt.tight_layout()
plt.savefig("random_forest_actual_vs_predicted.png", dpi=300)
plt.close()
print("Saved: random_forest_actual_vs_predicted.png")

importance = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
plt.figure(figsize=(7, 5))
importance.plot(kind="barh")
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig("random_forest_feature_importance.png", dpi=300)
plt.close()
print("Saved: random_forest_feature_importance.png")

joblib.dump({"model": model, "features": features}, "chennai_aqi_random_forest.joblib")
print("Saved: chennai_aqi_random_forest.joblib")
