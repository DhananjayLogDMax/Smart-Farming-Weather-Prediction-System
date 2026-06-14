# 🌾 AgriSense — Smart Farming Weather Prediction System

> An end-to-end **AI/ML web application** that predicts weather conditions, estimates rainfall, and recommends the best crops — built with Python, scikit-learn, Flask, and pure HTML/CSS/JS.

---

## 📸 Preview

```
┌─────────────────────────────────────────────────────────────┐
│  🌾 AgriSense  ·  AI-Powered Smart Farming                  │
├────────────────────┬────────────────────────────────────────┤
│  Field Parameters  │  🌧 Rainy                    72%       │
│  ─────────────     │  "Avoid pesticide spraying"  Rain      │
│  Month: July       │                                        │
│  Temp:  28°C   ── │  🌡28°C  💧75%  💨15km  🌱60%         │
│  Humid: 75%    ── │                                        │
│  Press: 1005   ── │  💧 Rainfall: 18.3 mm  ████████░░      │
│  Wind:  15     ── │                                        │
│  Soil:  60%    ── │  🌾 Crop: Rice  (Kharif, High water)   │
│                    │  Also: Maize, Vegetables               │
│  [ Run Prediction ]│                                        │
│                    │  ⚠ No alerts — conditions look good!  │
└────────────────────┴────────────────────────────────────────┘
```

---

## 🚀 Features

| Feature | Details |
|---------|---------|
| 🌧️ Rain Prediction | Random Forest classifier — probability + yes/no |
| 📊 Rainfall Estimation | Gradient Boosting Regressor — mm amount |
| 🌱 Crop Recommendation | Random Forest — 95% accuracy, top-3 crops |
| ⚠️ Farm Alerts | 7 smart alerts (heat, frost, drought, wind, flood…) |

---

## 🤖 ML Models Explained

### Input Features
| Feature | Unit | Description |
|---------|------|-------------|
| month | 1–12 | Calendar month |
| temperature | °C | Air temperature |
| humidity | % | Relative humidity |
| pressure | hPa | Atmospheric pressure |
| wind_speed | km/h | Wind speed |
| soil_moisture | % | Soil moisture level |

### Models

| # | Model | Algorithm | Target | Performance |
|---|-------|-----------|--------|-------------|
| 1 | Rain Classifier | Random Forest (150 trees) | Will it rain? (0/1) | ~70% accuracy |
| 2 | Rainfall Regressor | Gradient Boosting (150 est.) | Rain amount (mm) | MAE ~8 mm |
| 3 | Crop Recommender | Random Forest (200 trees) | Best crop | **95% accuracy** |

> **To improve accuracy:** Replace the synthetic dataset in `train_model.py` with real historical weather data from your region (IMD, OpenWeatherMap, etc.)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built with ❤️ for smart farming and sustainable agriculture.

---

*If you found this helpful, please ⭐ star the repository!*
