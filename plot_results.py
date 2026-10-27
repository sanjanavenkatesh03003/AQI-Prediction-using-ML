"""
Two plots:
  1. Correlation heatmap of all inputs and AQI
  2. Predicted vs actual AQI for the KNN and SVR models on the held-out test rows
     (the most recent 20% of your data, which the models never trained on)

Needs, in the same folder:
  aqi_data.csv, aqi_knn.py + aqi_knn_model.joblib, aqi_svr.py + aqi_svr_model.joblib
(if you only trained one of the models, the other is skipped)

Install:  pip install pandas numpy scikit-learn joblib matplotlib
Run:      python plot_results.py
Output:   aqi_correlation_heatmap.png, aqi_pred_vs_actual.png
"""

import importlib

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score

MODELS = [
    ("KNN", "aqi_knn", "aqi_knn_model.joblib"),
    ("SVR", "aqi_svr", "aqi_svr_model.joblib"),
]

# ---------------------------------------------------------------- load what is available
available = []
for name, module_name, model_file in MODELS:
    try:
        module = importlib.import_module(module_name)
        model = joblib.load(model_file)
        available.append((name, module, model))
    except (ImportError, FileNotFoundError) as e:
        print(f"Skipping {name}: {e}")

if not available:
    raise SystemExit("No trained model found. Run aqi_knn.py and/or aqi_svr.py first.")

# ---------------------------------------------------------------- 1. correlation heatmap
base = available[0][1]
df = base.load_data(base.CSV_PATH)
df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
df["aqi_prev_hour"] = df.groupby("location")["aqi"].shift(1)

cols = ["pm25", "o3", "co", "temp", "humidity", "hour", "aqi_prev_hour", "aqi"]
labels = ["PM2.5", "O3", "CO", "Temp", "Humidity", "Hour", "Prev-hour AQI", "AQI"]
corr = df[cols].corr().to_numpy()

fig1, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels)
for i in range(len(labels)):
    for j in range(len(labels)):
        ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                color="white" if abs(corr[i, j]) > 0.6 else "black", fontsize=9)
fig1.colorbar(im, ax=ax, label="Correlation")
ax.set_title("Correlation between inputs and AQI")
fig1.tight_layout()
fig1.savefig("aqi_correlation_heatmap.png", dpi=150)
print("Saved aqi_correlation_heatmap.png")

# ---------------------------------------------------------------- 2. predicted vs actual
fig2, axes = plt.subplots(1, len(available), figsize=(7 * len(available), 6), squeeze=False)

for ax, (name, module, model) in zip(axes[0], available):
    data = module.build_features(module.load_data(module.CSV_PATH), for_training=True)
    data = data.sort_values("datetime").reset_index(drop=True)
    split = int(len(data) * (1 - module.TEST_FRACTION))
    test = data.iloc[split:]

    y_true = test["target"].to_numpy()
    y_pred = model.predict(test[module.NUMERIC + module.CATEGORICAL])
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    base_mae = mean_absolute_error(y_true, test["aqi"])   # "next hour = this hour"

    ax.scatter(y_true, y_pred, alpha=0.6, color="blue", label="Test predictions")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, color="red", linewidth=2, label="Perfect prediction")
    ax.set_xlabel("Actual AQI")
    ax.set_ylabel("Predicted AQI")
    ax.set_title(f"{name}: predicted vs actual\n"
                 f"MAE {mae:.1f} | R² {r2:.2f} | baseline MAE {base_mae:.1f}")
    ax.grid(True)
    ax.legend(loc="upper left")
    print(f"{name}: test rows {len(test)}, MAE {mae:.2f}, R2 {r2:.3f}, baseline MAE {base_mae:.2f}")

fig2.tight_layout()
fig2.savefig("aqi_pred_vs_actual.png", dpi=150)
print("Saved aqi_pred_vs_actual.png")
plt.show()
