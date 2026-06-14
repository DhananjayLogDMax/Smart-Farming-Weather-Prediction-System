"""
=============================================================
  AgriSense – Smart Farming Weather Prediction
  ML Model Training Script
  Models: RandomForest (rain + crop) + GradientBoosting (rainfall amount)
=============================================================
Run from project root:  python ml_model/train_model.py
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report
import joblib

# ── Reproducibility ────────────────────────────────────────────────────────────
np.random.seed(42)
N_SAMPLES = 3000   # increase for better accuracy


# ── 1. Synthetic Dataset Generation ───────────────────────────────────────────
def generate_dataset(n: int) -> pd.DataFrame:
    """
    Generates a realistic synthetic weather + crop dataset.
    In production, replace this with real sensor / API data.
    """
    months     = np.random.randint(1, 13, n)
    # Temperature: seasonal sine wave + noise
    temp       = 15 + 15 * np.sin((months - 3) * np.pi / 6) + np.random.normal(0, 3, n)
    # Humidity: inverse seasonal cosine + noise
    humidity   = 40 + 30 * np.cos((months - 6) * np.pi / 6) + np.random.normal(0, 8, n)
    humidity   = np.clip(humidity, 10, 100)
    pressure   = 1013 + np.random.normal(0, 12, n)
    wind_speed = np.abs(np.random.normal(10, 5, n))
    soil_moist = np.clip(humidity * 0.6 + np.random.normal(0, 10, n), 0, 100)

    # Rain probability depends on humidity, temperature, pressure
    rain_prob  = (0.25
                  + 0.45 * (humidity / 100)
                  - 0.08 * (temp / 40)
                  + 0.10 * (pressure < 1005).astype(float)
                  + 0.08 * np.isin(months, [6, 7, 8, 9]).astype(float))
    rain_prob  = np.clip(rain_prob, 0.05, 0.95)
    will_rain  = (np.random.random(n) < rain_prob).astype(int)
    rainfall   = np.where(will_rain, np.random.exponential(15, n), 0.0)

    def weather_condition(r, h, t):
        if r > 0:    return "Rainy"
        if h > 70:   return "Cloudy"
        if t > 30:   return "Sunny"
        return "Partly Cloudy"

    def recommend_crop(t, h, r, month):
        if month in [6, 7, 8] and r > 10:          return "Rice"
        if month in [10, 11, 12] and t < 25:        return "Wheat"
        if t > 28 and h < 50:                        return "Cotton"
        if month in [2, 3, 4, 5] and h > 50:        return "Vegetables"
        return "Maize"

    conditions = [weather_condition(r, h, t)
                  for r, h, t in zip(rainfall, humidity, temp)]
    crops      = [recommend_crop(t, h, r, m)
                  for t, h, r, m in zip(temp, humidity, rainfall, months)]

    return pd.DataFrame({
        "month":              months,
        "temperature":        np.round(temp, 2),
        "humidity":           np.round(humidity, 2),
        "pressure":           np.round(pressure, 2),
        "wind_speed":         np.round(wind_speed, 2),
        "soil_moisture":      np.round(soil_moist, 2),
        "will_rain":          will_rain,
        "rainfall_amount":    np.round(rainfall, 2),
        "condition":          conditions,
        "crop_recommendation":crops,
    })


# ── 2. Generate & Save Dataset ─────────────────────────────────────────────────
print("=" * 55)
print("  AgriSense ML Training Pipeline")
print("=" * 55)
print(f"\n[1/4] Generating {N_SAMPLES}-sample dataset ...")

os.makedirs("data", exist_ok=True)
df = generate_dataset(N_SAMPLES)
df.to_csv("data/weather_data.csv", index=False)
print(f"      Saved → data/weather_data.csv")
print(f"      Columns: {list(df.columns)}")
print(f"      Rain %: {df['will_rain'].mean()*100:.1f}%")
print(f"      Crops : {df['crop_recommendation'].value_counts().to_dict()}")


# ── 3. Feature Engineering ─────────────────────────────────────────────────────
FEATURES = ["month", "temperature", "humidity", "pressure", "wind_speed", "soil_moisture"]
X        = df[FEATURES].values
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)


# ── 4. Train Models ────────────────────────────────────────────────────────────
print("\n[2/4] Training Rain Prediction model (Random Forest) ...")
y_rain              = df["will_rain"].values
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y_rain, test_size=0.2, random_state=42)
rain_clf            = RandomForestClassifier(n_estimators=150, max_depth=10,
                                             min_samples_split=5, random_state=42, n_jobs=-1)
rain_clf.fit(Xtr, ytr)
rain_acc            = accuracy_score(yte, rain_clf.predict(Xte))
print(f"      Accuracy : {rain_acc:.2%}")
print(classification_report(yte, rain_clf.predict(Xte),
                             target_names=["No Rain", "Rain"], zero_division=0))

print("[3/4] Training Rainfall Amount model (Gradient Boosting) ...")
y_amt                  = df["rainfall_amount"].values
Xtr2, Xte2, ytr2, yte2 = train_test_split(X_scaled, y_amt, test_size=0.2, random_state=42)
rain_reg               = GradientBoostingRegressor(n_estimators=150, learning_rate=0.08,
                                                    max_depth=5, random_state=42)
rain_reg.fit(Xtr2, ytr2)
rain_mae               = mean_absolute_error(yte2, rain_reg.predict(Xte2))
print(f"      MAE      : {rain_mae:.2f} mm")

print("[4/4] Training Crop Recommendation model (Random Forest) ...")
le_crop                = LabelEncoder()
y_crop                 = le_crop.fit_transform(df["crop_recommendation"].values)
Xtr3, Xte3, ytr3, yte3 = train_test_split(X_scaled, y_crop, test_size=0.2, random_state=42)
crop_clf               = RandomForestClassifier(n_estimators=200, max_depth=12,
                                                 min_samples_split=4, random_state=42, n_jobs=-1)
crop_clf.fit(Xtr3, ytr3)
crop_acc               = accuracy_score(yte3, crop_clf.predict(Xte3))
print(f"      Accuracy : {crop_acc:.2%}")
print(classification_report(yte3, crop_clf.predict(Xte3),
                             target_names=le_crop.classes_, zero_division=0))


# ── 5. Save All Artifacts ──────────────────────────────────────────────────────
SAVE_DIR = "ml_model/saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)
joblib.dump(rain_clf,  f"{SAVE_DIR}/rain_classifier.pkl")
joblib.dump(rain_reg,  f"{SAVE_DIR}/rainfall_regressor.pkl")
joblib.dump(crop_clf,  f"{SAVE_DIR}/crop_classifier.pkl")
joblib.dump(scaler,    f"{SAVE_DIR}/scaler.pkl")
joblib.dump(le_crop,   f"{SAVE_DIR}/crop_label_encoder.pkl")
joblib.dump(FEATURES,  f"{SAVE_DIR}/feature_names.pkl")

print("\n" + "=" * 55)
print("  ✅  Training Complete — Summary")
print("=" * 55)
print(f"  Rain Prediction Accuracy  : {rain_acc:.2%}")
print(f"  Rainfall Amount MAE       : {rain_mae:.2f} mm")
print(f"  Crop Recommendation Acc   : {crop_acc:.2%}")
print(f"  Models saved to           : {SAVE_DIR}/")
print("=" * 55)
