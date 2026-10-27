"""
Plot every input used by the model against AQI: one panel per parameter, all in one figure.

Install:  pip install pandas numpy matplotlib
Run:      python plot_aqi_features.py
Output:   aqi_feature_plots.png (also shown on screen)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = "aqi_data.csv"
DAYFIRST = False            # set True if your dates look like 31/12/2026

# ---------------------------------------------------------------- load
df = pd.read_csv(CSV_PATH)
df.columns = (df.columns.str.strip().str.lower()
              .str.replace(".", "", regex=False).str.replace(" ", "_"))
df = df.rename(columns={"temperature": "temp", "pm_25": "pm25"})

df["datetime"] = pd.to_datetime(df["date"].astype(str) + " " + df["time"].astype(str),
                                errors="coerce", dayfirst=DAYFIRST)
df = df.dropna(subset=["datetime"])
df["location"] = df["location"].astype(str).str.strip()
for c in ["pm25", "o3", "co", "temp", "humidity", "aqi"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.sort_values(["location", "datetime"])
df["hour"] = df["datetime"].dt.hour + df["datetime"].dt.minute / 60
df["aqi_prev_hour"] = df.groupby("location")["aqi"].shift(1)

# ---------------------------------------------------------------- panels
numeric_panels = [
    ("pm25", "PM2.5"),
    ("o3", "O3"),
    ("co", "CO"),
    ("temp", "Temperature"),
    ("humidity", "Humidity"),
    ("aqi_prev_hour", "Previous-hour AQI"),
]

fig, axes = plt.subplots(2, 4, figsize=(20, 9))
axes = axes.flatten()

# scatter + best-fit line for the numeric inputs
for ax, (col, label) in zip(axes, numeric_panels):
    d = df[[col, "aqi"]].dropna()
    x, y = d[col].to_numpy(), d["aqi"].to_numpy()
    ax.scatter(x, y, alpha=0.5, color="blue", label="Actual data")
    if len(d) > 2 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ax.plot(xs, slope * xs + intercept, color="red", linewidth=2, label="Best fit")
        r = np.corrcoef(x, y)[0, 1]
        ax.set_title(f"{label} vs AQI  (r = {r:.2f})")
    else:
        ax.set_title(f"{label} vs AQI")
    ax.set_xlabel(label)
    ax.set_ylabel("AQI")
    ax.grid(True)
    ax.legend(loc="upper left", fontsize=8)

# hour of day: scatter + average AQI for each hour
ax = axes[6]
ax.scatter(df["hour"], df["aqi"], alpha=0.4, color="blue", label="Actual data")
hourly = df.groupby(df["hour"].astype(int))["aqi"].mean()
ax.plot(hourly.index + 0.5, hourly.values, color="red", linewidth=2, marker="o",
        label="Average per hour")
ax.set_title("Hour of day vs AQI")
ax.set_xlabel("Hour of day")
ax.set_ylabel("AQI")
ax.set_xticks(range(0, 25, 4))
ax.grid(True)
ax.legend(loc="upper left", fontsize=8)

# location: box plot of AQI per location
ax = axes[7]
locs = sorted(df["location"].unique())
ax.boxplot([df.loc[df["location"] == l, "aqi"].dropna() for l in locs])
ax.set_xticks(range(1, len(locs) + 1))
ax.set_xticklabels(locs)
ax.set_title("Location vs AQI")
ax.set_xlabel("Location")
ax.set_ylabel("AQI")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, axis="y")

fig.suptitle("Model inputs vs AQI", fontsize=16)
fig.tight_layout()
fig.savefig("aqi_feature_plots.png", dpi=150)
print("Saved aqi_feature_plots.png")
plt.show()
