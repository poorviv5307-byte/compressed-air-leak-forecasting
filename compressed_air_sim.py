"""
AI-based compressed-air leakage prediction and consumption forecasting
Simulation study: plant data generation -> leak detection -> demand forecasting
"""
import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, mean_squared_error

rng = np.random.default_rng(42)
OUT = os.path.dirname(os.path.abspath(__file__))
PLOTS = os.path.join(OUT, "plots")
os.makedirs(PLOTS, exist_ok=True)

# =====================================================================
# 1. SIMULATE THE COMPRESSED-AIR PLANT
# =====================================================================
DAYS = 60
N_DAY = 96                       # 15-minute samples
idx = pd.date_range("2026-01-01", periods=DAYS * N_DAY, freq="15min")
df = pd.DataFrame(index=idx)
df["day"] = np.asarray((idx - idx[0]).days)
h = idx.hour + idx.minute / 60
dow = idx.dayofweek
df["hour"] = h
df["dow"] = dow

# Production schedule: two shifts Mon-Fri 06-22h, half shift Sat 06-14h, Sunday off
prod = np.zeros(len(df))
prod[(dow < 5) & (h >= 6) & (h < 22)] = 1.0
prod[(dow == 5) & (h >= 6) & (h < 14)] = 0.5

# Air demand (m3/min): shift-shaped load + purge/instrument baseline + noise
shape = np.clip(np.sin(np.pi * (h - 6) / 16), 0, None)
demand = prod * (7.5 + 2.0 * shape) + 0.4
demand = demand + rng.normal(0, 0.35, len(df)) * (prod + 0.2)
demand = np.clip(demand, 0.1, None)

# Leak schedule (m3/min at reference pressure):
#   days 0-19 : 1.2  (normal, pre-existing leakage)
#   days 20-39: 2.5  (new leak appears)
#   days 40-59: 4.0  (leak grows)
leak_ref = np.where(df.day < 20, 1.2, np.where(df.day < 40, 2.5, 4.0))

# Pressure: receiver pressure falls with total flow; slow drift; sensor noise
drift = 0.05 * np.sin(2 * np.pi * df.day / 30)
P0 = 7.0 - 0.02 * (demand + leak_ref) + drift
leak = leak_ref * np.sqrt(P0 / 7.0)          # orifice law: leak ~ sqrt(pressure)
total = demand + leak

df["pressure"] = 7.0 - 0.02 * total + drift + rng.normal(0, 0.02, len(df))
df["flow"] = total + rng.normal(0, 0.15, len(df))          # measured compressed air flow
df["power"] = 4.5 + 0.9 * total + rng.normal(0, 0.3, len(df))  # kW
df["temp"] = 24 + 4 * np.sin(2 * np.pi * (h - 9) / 24) + rng.normal(0, 0.5, len(df))
df["demand_true"] = demand
df["leak_true"] = leak
df["anomaly_true"] = (df.day >= 20).astype(int)

df.to_csv(os.path.join(OUT, "simulated_plant_data_15min.csv"))

# =====================================================================
# 2. LEAK DETECTION (night-time minimum-flow method + Isolation Forest)
# =====================================================================
# At 01:00-04:00 no production runs, so measured flow ~ purge + leak
night = df[(df.hour >= 1) & (df.hour < 4)]
daily = night.groupby("day")[["flow", "pressure", "power"]].mean()
PURGE = 0.4
daily["leak_est"] = daily["flow"] - PURGE

# Baseline from the first 14 days (before any leak), threshold = mean + 3 sigma
base = daily.loc[:13, "leak_est"]
thr = max(base.mean() + 3 * base.std(), base.mean() + 0.5)
daily["alert"] = daily["leak_est"] > thr
daily["alert_2day"] = daily["alert"] & daily["alert"].shift(1, fill_value=False)
first_alert_day = int(daily.index[daily["alert_2day"]][0])
truth_daily = (daily.index >= 20).astype(int)

def prf(y, p):
    tp = int(((y == 1) & (p == 1)).sum())
    fp = int(((y == 0) & (p == 1)).sum())
    fn = int(((y == 1) & (p == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1

rule_pred = daily["alert_2day"].astype(int).values
rule_prf = prf(truth_daily, rule_pred)

iso = IsolationForest(contamination=0.35, random_state=42)
iso_feat = daily[["leak_est", "pressure", "power"]].values
iso_pred = (iso.fit_predict(iso_feat) == -1).astype(int)
iso_prf = prf(truth_daily, iso_pred)

# Leak size estimate from the last 20 days, and the cost of that leak
est_leak_now = daily.loc[40:, "leak_est"].mean() - base.mean()
est_leak_true = 4.0 - 1.2
KW_PER_M3MIN = 0.9            # from the power model above
HOURS = 8760 * 0.70           # 70 % operating hours
TARIFF = 8.0                  # INR per kWh (assumed)
annual_kwh = est_leak_now * KW_PER_M3MIN * HOURS
annual_cost = annual_kwh * TARIFF

# =====================================================================
# 3. CONSUMPTION FORECASTING (1-hour ahead, 15-min data)
# =====================================================================
H = 4  # forecast horizon: 4 steps = 1 hour
X = pd.DataFrame(index=df.index)
X["lag1"] = df.flow.shift(1)
X["lag4"] = df.flow.shift(4)
X["lag96"] = df.flow.shift(96)
X["lag672"] = df.flow.shift(672)
X["roll4"] = df.flow.shift(1).rolling(4).mean()
X["roll96"] = df.flow.shift(1).rolling(96).mean()
X["p_lag1"] = df.pressure.shift(1)
X["temp"] = df.temp.shift(1)
X["hour"] = df.hour
X["dow"] = df.dow
y = df.flow.shift(-H).rename("y")

data = pd.concat([X, y], axis=1).dropna()
split_time = data.index[-7 * N_DAY]          # last 7 days = test set
train = data[data.index < split_time]
test = data[data.index >= split_time]

model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, random_state=42)
model.fit(train.drop(columns="y"), train["y"])
pred_ml = model.predict(test.drop(columns="y"))

pred_naive = X.loc[test.index, "lag96"].values + 0.0 * 0  # same time yesterday (approx for t+H)
pred_naive = df.flow.shift(96 - H).loc[test.index].values
pred_pers = X.loc[test.index, "lag1"].values              # last observed value

def metrics(yt, yp):
    return {
        "MAE": round(float(mean_absolute_error(yt, yp)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "MAPE_%": round(float(np.mean(np.abs((yt - yp) / yt)) * 100), 2),
    }

yt = test["y"].values
m_ml = metrics(yt, pred_ml)
m_naive = metrics(yt, pred_naive)
m_pers = metrics(yt, pred_pers)

# =====================================================================
# 4. PLOTS
# =====================================================================
# Fig 1: 3-day zoom during the grown-leak period
z = df.loc["2026-02-10":"2026-02-13"]
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(z.index, z.demand_true + z.leak_true, label="Total air flow (true)", lw=0.8)
ax.plot(z.index, z.demand_true, label="Production demand (true)", lw=0.8)
ax.plot(z.index, z.flow, label="Measured flow (sensor)", lw=0.5, alpha=0.5)
ax.set_ylabel("Flow (m³/min)")
ax.set_title("Fig 1. Plant air flow over 3 days (leak 4.0 m³/min present)")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig1_flow_zoom.png"), dpi=150); plt.close(fig)

# Fig 2: nightly leak estimate with threshold
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(daily.index, daily.leak_est, "o-", ms=3, label="Nightly leak estimate")
ax.plot(daily.index, df.groupby("day").leak_true.mean(), "k--", label="True leak")
ax.axhline(thr, color="r", ls=":", label=f"Alert threshold ({thr:.2f})")
ax.axvline(first_alert_day, color="r", alpha=0.4)
ax.axvline(20, color="g", alpha=0.4)
ax.text(first_alert_day + 0.5, ax.get_ylim()[1] * 0.95, f"Alert day {first_alert_day}", color="r")
ax.text(20.5, ax.get_ylim()[0] + 0.2, "Leak starts (day 20)", color="g")
ax.set_xlabel("Day"); ax.set_ylabel("Leak (m³/min)")
ax.set_title("Fig 2. Leak detection from night-time flow")
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig2_leak_detection.png"), dpi=150); plt.close(fig)

# Fig 3: forecast vs actual, test week
fig, ax = plt.subplots(figsize=(11, 4))
ax.plot(test.index, yt, label="Actual", lw=0.8)
ax.plot(test.index, pred_ml, label=f"Gradient boosting (MAE {m_ml['MAE']})", lw=0.8)
ax.plot(test.index, pred_naive, label=f"Seasonal naive (MAE {m_naive['MAE']})", lw=0.6, alpha=0.6)
ax.set_ylabel("Flow 1 h ahead (m³/min)")
ax.set_title("Fig 3. 1-hour-ahead consumption forecast, last 7 days")
ax.legend(loc="upper right")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig3_forecast.png"), dpi=150); plt.close(fig)

# Fig 4: daily pressure and power trends
daily_pp = df.groupby("day")[["pressure", "power"]].mean()
fig, ax1 = plt.subplots(figsize=(11, 4))
ax1.plot(daily_pp.index, daily_pp.pressure, "b-", label="Mean pressure (bar)")
ax1.set_ylabel("Pressure (bar)", color="b"); ax1.set_xlabel("Day")
ax2 = ax1.twinx()
ax2.plot(daily_pp.index, daily_pp.power, "r-", label="Mean power (kW)")
ax2.set_ylabel("Power (kW)", color="r")
ax1.axvline(20, color="g", alpha=0.4); ax1.axvline(40, color="g", alpha=0.4)
ax1.set_title("Fig 4. Daily pressure and power (leak events at days 20 and 40)")
fig.tight_layout(); fig.savefig(os.path.join(PLOTS, "fig4_pressure_power.png"), dpi=150); plt.close(fig)

# =====================================================================
# 5. RESULTS
# =====================================================================
daily.reset_index().to_csv(os.path.join(OUT, "daily_leak_detection.csv"), index=False)
results = {
    "simulation": {"days": DAYS, "interval_min": 15, "leak_schedule_m3min": {"d0-19": 1.2, "d20-39": 2.5, "d40-59": 4.0}},
    "leak_detection": {
        "baseline_mean_m3min": round(float(base.mean()), 3),
        "threshold_m3min": round(float(thr), 3),
        "first_alert_day": first_alert_day,
        "detection_delay_days": first_alert_day - 20,
        "rule_precision_recall_f1": [round(v, 3) for v in rule_prf],
        "isolation_forest_precision_recall_f1": [round(v, 3) for v in iso_prf],
        "estimated_extra_leak_m3min": round(float(est_leak_now), 3),
        "true_extra_leak_m3min": round(float(est_leak_true), 3),
        "est_annual_leak_cost_INR": round(float(annual_cost)),
    },
    "forecasting_1h_ahead": {"gradient_boosting": m_ml, "seasonal_naive": m_naive, "persistence": m_pers},
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(results, f, indent=2)

print(json.dumps(results, indent=2))
print("\nFiles written to:", OUT)
