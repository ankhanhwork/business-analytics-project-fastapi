import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
import os

# =========================================================
# 1. DATA GENERATION (Reusing logic from user's code)
# =========================================================
np.random.seed(42)
start_date = "2024-01-01"
end_date = "2026-03-31"

holiday_ranges = [
    ("2024-01-01", "2024-01-01"), ("2024-02-08", "2024-02-14"),
    ("2024-04-18", "2024-04-18"), ("2024-04-30", "2024-04-30"),
    ("2024-05-01", "2024-05-01"), ("2024-08-31", "2024-09-03"),
    ("2025-01-01", "2025-01-01"), ("2025-01-25", "2025-02-02"),
    ("2025-04-05", "2025-04-07"), ("2025-04-30", "2025-05-04"),
    ("2025-08-30", "2025-09-02"), ("2026-01-01", "2026-01-04"),
    ("2026-02-14", "2026-02-22"),
]
holiday_dates = set()
for start_h, end_h in holiday_ranges:
    for d in pd.date_range(start_h, end_h, freq="D"):
        holiday_dates.add(pd.Timestamp(d))

all_dates = pd.date_range(start=start_date, end=end_date, freq="D")
daily_df = pd.DataFrame({"date": all_dates})
daily_df["day_of_week"] = daily_df["date"].dt.dayofweek
daily_df["is_holiday"] = daily_df["date"].isin(holiday_dates)
daily_df = daily_df[(daily_df["day_of_week"] <= 5) & (~daily_df["is_holiday"])].copy()
daily_df["month"] = daily_df["date"].dt.month
daily_df["day_of_year"] = daily_df["date"].dt.dayofyear
days_from_start = (daily_df["date"] - daily_df["date"].min()).dt.days
daily_df["trend_factor"] = 1 + (days_from_start / days_from_start.max()) * 0.05

daily_base = 760
weekday_factor = daily_df["day_of_week"].map({0: 1.18, 1: 1.10, 2: 1.03, 3: 0.95, 4: 0.88, 5: 0.62})
month_factor = daily_df["month"].map({1: 1.03, 2: 0.90, 3: 1.00, 4: 0.98, 5: 1.00, 6: 1.03, 7: 1.08, 8: 1.10, 9: 1.06, 10: 1.01, 11: 1.00, 12: 1.02})
smooth_seasonality = 1 + 0.04 * np.sin(2 * np.pi * daily_df["day_of_year"] / 365.25)
daily_noise = np.clip(np.random.normal(loc=1.0, scale=0.15, size=len(daily_df)), 0.68, 1.38)

daily_df["daily_total_patients"] = np.round(daily_base * daily_df["trend_factor"] * weekday_factor * month_factor * smooth_seasonality * daily_noise).astype(int).clip(350, 1200)

booking_share_base = 0.225
booking_weekday_adj = daily_df["day_of_week"].map({0: 0.015, 1: 0.010, 2: 0.000, 3: -0.005, 4: -0.010, 5: -0.015})
booking_month_adj = daily_df["month"].map({1: 0.000, 2: -0.015, 3: 0.000, 4: 0.000, 5: 0.000, 6: 0.005, 7: 0.010, 8: 0.010, 9: 0.005, 10: 0.000, 11: 0.000, 12: 0.000})
booking_trend_adj = 0.015 * (days_from_start / days_from_start.max())
booking_share_noise = np.random.normal(loc=0.0, scale=0.035, size=len(daily_df))
daily_df["booking_share"] = (booking_share_base + booking_weekday_adj + booking_month_adj + booking_trend_adj + booking_share_noise).clip(0.10, 0.36)
daily_df["daily_appointments_booked"] = np.round(daily_df["daily_total_patients"] * daily_df["booking_share"]).astype(int)
daily_df["daily_appointments_booked"] = np.minimum(daily_df["daily_appointments_booked"], daily_df["daily_total_patients"] - 25)

morning_share = np.clip(np.random.normal(loc=0.60, scale=0.06, size=len(daily_df)), 0.48, 0.72)
daily_df["morning_share"] = np.where(daily_df["day_of_week"] == 5, 1.0, morning_share)
daily_df["morning_total"] = np.round(daily_df["daily_total_patients"] * daily_df["morning_share"]).astype(int)
daily_df["afternoon_total"] = daily_df["daily_total_patients"] - daily_df["morning_total"]

morning_booking_share = np.clip(np.random.normal(loc=0.62, scale=0.07, size=len(daily_df)), 0.45, 0.78)
daily_df["morning_booking_share"] = np.where(daily_df["day_of_week"] == 5, 1.0, morning_booking_share)
daily_df["morning_booked"] = np.minimum(np.round(daily_df["daily_appointments_booked"] * daily_df["morning_booking_share"]).astype(int), np.maximum(daily_df["morning_total"] - 5, 0))
daily_df["afternoon_booked"] = np.minimum(daily_df["daily_appointments_booked"] - daily_df["morning_booked"], np.maximum(daily_df["afternoon_total"] - 5, 0))

morning_df = daily_df[["date", "morning_booked", "morning_total"]].copy().rename(columns={"morning_booked": "appointments_booked", "morning_total": "total_patients"})
morning_df["shift"] = "Morning"
afternoon_df = daily_df[daily_df["day_of_week"] != 5][["date", "afternoon_booked", "afternoon_total"]].copy().rename(columns={"afternoon_booked": "appointments_booked", "afternoon_total": "total_patients"})
afternoon_df["shift"] = "Afternoon"
final_df = pd.concat([morning_df, afternoon_df], ignore_index=True)
shift_order_map = {"Morning": 0, "Afternoon": 1}
final_df["shift_order"] = final_df["shift"].map(shift_order_map)
final_df = final_df.sort_values(["date", "shift_order"]).reset_index(drop=True)

# =========================================================
# 2. FEATURE ENGINEERING (Refactored to be reusable)
# =========================================================
def add_time_features(data):
    data = data.copy()
    data["date"] = pd.to_datetime(data["date"])
    data["day_of_week"] = data["date"].dt.dayofweek
    data["month"] = data["date"].dt.month
    data["year"] = data["date"].dt.year
    data["day_of_month"] = data["date"].dt.day
    data["week_of_year"] = data["date"].dt.isocalendar().week.astype(int)
    data["quarter"] = data["date"].dt.quarter
    data["is_morning"] = (data["shift"] == "Morning").astype(int)
    data["is_afternoon"] = (data["shift"] == "Afternoon").astype(int)
    data["is_saturday"] = (data["day_of_week"] == 5).astype(int)
    data["is_month_start"] = data["date"].dt.is_month_start.astype(int)
    data["is_month_end"] = data["date"].dt.is_month_end.astype(int)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 6)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 6)
    data["month_sin"] = np.sin(2 * np.pi * data["month"] / 12)
    data["month_cos"] = np.cos(2 * np.pi * data["month"] / 12)
    return data

df = add_time_features(final_df)
target_lags = [1, 2, 3, 6, 11, 12, 22]
appt_lags = [1, 2, 6, 11]
for lag in target_lags: df[f"lag_{lag}"] = df["total_patients"].shift(lag)
for lag in appt_lags: df[f"appt_lag_{lag}"] = df["appointments_booked"].shift(lag)
for window in [3, 6, 11, 22]:
    df[f"roll_mean_{window}"] = df["total_patients"].shift(1).rolling(window).mean()
    df[f"roll_std_{window}"] = df["total_patients"].shift(1).rolling(window).std()
    df[f"roll_min_{window}"] = df["total_patients"].shift(1).rolling(window).min()
    df[f"roll_max_{window}"] = df["total_patients"].shift(1).rolling(window).max()
for window in [3, 6, 11]:
    df[f"appt_roll_mean_{window}"] = df["appointments_booked"].shift(1).rolling(window).mean()
    df[f"appt_roll_std_{window}"] = df["appointments_booked"].shift(1).rolling(window).std()
df["appt_diff_1"] = df["appointments_booked"] - df["appointments_booked"].shift(1)
df["appt_diff_6"] = df["appointments_booked"] - df["appointments_booked"].shift(6)
df["appt_diff_11"] = df["appointments_booked"] - df["appointments_booked"].shift(11)
df["appt_to_recent_mean_6"] = df["appointments_booked"] / df["roll_mean_6"]
df["appt_to_recent_mean_11"] = df["appointments_booked"] / df["roll_mean_11"]
df["appt_x_morning"] = df["appointments_booked"] * df["is_morning"]
df["appt_x_saturday"] = df["appointments_booked"] * df["is_saturday"]
df["appt_x_dow"] = df["appointments_booked"] * df["day_of_week"]
df["lag1_x_morning"] = df["lag_1"] * df["is_morning"]
df["lag6_x_morning"] = df["lag_6"] * df["is_morning"]

df_model = df.dropna().reset_index(drop=True)

feature_cols = [
    "appointments_booked", "is_morning", "is_afternoon", "is_saturday", "day_of_week", "month", "year", "day_of_month", "week_of_year", "quarter", "is_month_start", "is_month_end", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "lag_1", "lag_2", "lag_3", "lag_6", "lag_11", "lag_12", "lag_22", "appt_lag_1", "appt_lag_2", "appt_lag_6", "appt_lag_11",
    "roll_mean_3", "roll_mean_6", "roll_mean_11", "roll_mean_22", "roll_std_3", "roll_std_6", "roll_std_11", "roll_std_22",
    "roll_min_3", "roll_min_6", "roll_min_11", "roll_min_22", "roll_max_3", "roll_max_6", "roll_max_11", "roll_max_22",
    "appt_roll_mean_3", "appt_roll_mean_6", "appt_roll_mean_11", "appt_roll_std_3", "appt_roll_std_6", "appt_roll_std_11",
    "appt_diff_1", "appt_diff_6", "appt_diff_11", "appt_to_recent_mean_6", "appt_to_recent_mean_11", "appt_x_morning", "appt_x_saturday", "appt_x_dow", "lag1_x_morning", "lag6_x_morning",
]
target_col = "total_patients"

# =========================================================
# 3. MODEL TRAINING (Using best params from user)
# =========================================================
best_params = {'n_estimators': 700, 'min_samples_split': 20, 'min_samples_leaf': 1, 'max_samples': 0.9, 'max_features': 'sqrt', 'max_depth': 20, 'bootstrap': True}
model = RandomForestRegressor(**best_params, random_state=42, n_jobs=-1)
model.fit(df_model[feature_cols], df_model[target_col])

# =========================================================
# 4. SAVE MODEL AND RECENT DATA
# =========================================================
joblib.dump(model, "model.joblib")
joblib.dump(feature_cols, "feature_cols.joblib")

# Lưu lại các dòng cuối cùng để phục vụ việc tính toán feature động trong API
df_model[["date", "shift", "appointments_booked", "total_patients", "shift_order"]].tail(50).to_csv("historical_data.csv", index=False)

print("Model and historical data saved successfully.")
