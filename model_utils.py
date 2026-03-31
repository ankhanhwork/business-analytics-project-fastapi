import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# =========================================================
# CONFIG
# =========================================================
MODEL_PATH = "model.joblib"
FEATURE_COLS_PATH = "feature_cols.joblib"
HISTORICAL_DATA_PATH = "historical_data.csv"

# =========================================================
# UTILITIES
# =========================================================
def load_assets():
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_COLS_PATH)
    history_df = pd.read_csv(HISTORICAL_DATA_PATH)
    history_df["date"] = pd.to_datetime(history_df["date"])
    return model, feature_cols, history_df

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

def prepare_future_row(history_df, input_date, input_shift, appointments_booked, feature_cols):
    input_date = pd.to_datetime(input_date)
    shift_order_map = {"Morning": 0, "Afternoon": 1}

    # Validate inputs
    if input_date.dayofweek == 6:
        raise ValueError("Sunday is not a working day.")
    if input_date.dayofweek == 5 and input_shift == "Afternoon":
        raise ValueError("Saturday Afternoon is not a working shift.")

    new_row = pd.DataFrame({
        "date": [input_date],
        "shift": [input_shift],
        "appointments_booked": [appointments_booked],
        "shift_order": [shift_order_map[input_shift]]
    })

    # Combine with enough history for window calculations (max window is 22)
    temp = pd.concat([history_df.tail(30), new_row], ignore_index=True)
    temp = temp.sort_values(["date", "shift_order"]).reset_index(drop=True)
    temp = add_time_features(temp)

    # Re-calculate lags and rolling features for the last row
    target_lags = [1, 2, 3, 6, 11, 12, 22]
    appt_lags = [1, 2, 6, 11]
    
    for lag in target_lags: temp[f"lag_{lag}"] = temp["total_patients"].shift(lag)
    for lag in appt_lags: temp[f"appt_lag_{lag}"] = temp["appointments_booked"].shift(lag)
    
    for window in [3, 6, 11, 22]:
        temp[f"roll_mean_{window}"] = temp["total_patients"].shift(1).rolling(window).mean()
        temp[f"roll_std_{window}"] = temp["total_patients"].shift(1).rolling(window).std()
        temp[f"roll_min_{window}"] = temp["total_patients"].shift(1).rolling(window).min()
        temp[f"roll_max_{window}"] = temp["total_patients"].shift(1).rolling(window).max()
    
    for window in [3, 6, 11]:
        temp[f"appt_roll_mean_{window}"] = temp["appointments_booked"].shift(1).rolling(window).mean()
        temp[f"appt_roll_std_{window}"] = temp["appointments_booked"].shift(1).rolling(window).std()
    
    temp["appt_diff_1"] = temp["appointments_booked"] - temp["appointments_booked"].shift(1)
    temp["appt_diff_6"] = temp["appointments_booked"] - temp["appointments_booked"].shift(6)
    temp["appt_diff_11"] = temp["appointments_booked"] - temp["appointments_booked"].shift(11)
    
    temp["appt_to_recent_mean_6"] = temp["appointments_booked"] / temp["roll_mean_6"]
    temp["appt_to_recent_mean_11"] = temp["appointments_booked"] / temp["roll_mean_11"]
    
    temp["appt_x_morning"] = temp["appointments_booked"] * temp["is_morning"]
    temp["appt_x_saturday"] = temp["appointments_booked"] * temp["is_saturday"]
    temp["appt_x_dow"] = temp["appointments_booked"] * temp["day_of_week"]
    
    temp["lag1_x_morning"] = temp["lag_1"] * temp["is_morning"]
    temp["lag6_x_morning"] = temp["lag_6"] * temp["is_morning"]

    # Extract features for the new row
    predict_row = temp.iloc[[-1]].copy()

    # Fill potential NaNs with median/mean if history is too short (unlikely due to tail(30))
    for col in feature_cols:
        if col not in predict_row.columns:
             predict_row[col] = 0 # Safety fallback
        if predict_row[col].isna().any():
            predict_row[col] = predict_row[col].fillna(0)

    return predict_row[feature_cols]

def predict(model, history_df, input_date, input_shift, appointments_booked, feature_cols):
    future_X = prepare_future_row(history_df, input_date, input_shift, appointments_booked, feature_cols)
    pred = model.predict(future_X)[0]
    return int(round(max(0, pred)))
