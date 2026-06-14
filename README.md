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
| 🖥️ Live Dashboard | Interactive sliders, animated results, zero page reload |
| 🔌 Offline Mode | Built-in JS fallback — works without backend |
| 🔗 REST API | Flask JSON API — easily connect mobile apps or IoT |

---

## 🏗️ Project Structure

```
agrisense/
│
├── frontend/
│   └── index.html              # Dashboard UI (HTML + CSS + JS, zero framework)
│
├── backend/
│   └── app.py                  # Flask REST API server
│
├── ml_model/
│   ├── train_model.py          # Full ML training pipeline
│   └── saved_models/           # Auto-created after training
│       ├── rain_classifier.pkl
│       ├── rainfall_regressor.pkl
│       ├── crop_classifier.pkl
│       ├── scaler.pkl
│       └── crop_label_encoder.pkl
│
├── data/
│   └── weather_data.csv        # Auto-generated training dataset
│
├── requirements.txt
├── .gitignore
└── README.md
```

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

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9 or higher
- pip
- Git
- A modern web browser

---

### Step 1 — Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/agrisense.git
cd agrisense
```

---

### Step 2 — Create a virtual environment

```bash
# Create
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS / Linux)
source venv/bin/activate
```

---

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4 — Train the ML models

```bash
python ml_model/train_model.py
```

Expected output:
```
[1/4] Generating 3000-sample dataset ...
[2/4] Training Rain Prediction model ...
      Accuracy : 70.33%
[3/4] Training Rainfall Amount model ...
      MAE      : 7.85 mm
[4/4] Training Crop Recommendation model ...
      Accuracy : 95.00%
Models saved to ml_model/saved_models/
```

---

### Step 5 — Start the Flask API

```bash
python backend/app.py
```

API is now running at: `http://localhost:5000`

---

### Step 6 — Open the dashboard

Open `frontend/index.html` in your browser.

> ✅ **The dashboard works fully offline too** — it has a built-in JavaScript model that runs in the browser even without the Flask backend.

---

## 🌐 API Reference

### Base URL
```
http://localhost:5000
```

---

### `POST /api/predict`

Run all three ML models and get weather + crop predictions.

**Request:**
```json
{
  "month": 7,
  "temperature": 28,
  "humidity": 75,
  "pressure": 1005,
  "wind_speed": 15,
  "soil_moisture": 60
}
```

**Response:**
```json
{
  "success": true,
  "data": {
    "inputs": {
      "month": 7,
      "month_name": "July",
      "temperature": 28,
      "humidity": 75,
      "pressure": 1005,
      "wind_speed": 15,
      "soil_moisture": 60
    },
    "weather": {
      "condition": "Rainy",
      "rain_probability": 72.4,
      "will_rain": true,
      "rainfall_mm": 18.3,
      "tip": "Avoid pesticide spraying. Check field drainage channels."
    },
    "crop": {
      "recommended": "Rice",
      "top_3": ["Rice", "Maize", "Vegetables"],
      "info": {
        "icon": "🌾",
        "water_need": "High",
        "season": "Kharif (Jun–Nov)",
        "temp_range": "22–32°C"
      }
    },
    "alerts": [
      "🌊 Heavy rainfall forecast – inspect drainage channels."
    ],
    "alert_count": 1
  }
}
```

---

### `GET /api/health`

```json
{
  "status": "ok",
  "models_loaded": true,
  "models": ["rain_classifier", "rainfall_regressor", "crop_classifier", "scaler", "crop_label_encoder"]
}
```

---

### `GET /api/crops`

Returns all crop metadata.

---

## 📤 Upload to GitHub — Step-by-Step

### 1. Create a GitHub account
Go to [github.com](https://github.com) and sign up (free).

---

### 2. Create a new repository

1. Click the **+** icon (top right) → **New repository**
2. Repository name: `agrisense`
3. Description: `AI-powered Smart Farming Weather Prediction System`
4. Set to **Public**
5. ❌ Do NOT check "Add a README file" (we already have one)
6. Click **Create repository**

---

### 3. Initialize git locally

Open terminal in the project folder:

```bash
cd agrisense
git init
git add .
git commit -m "🌾 Initial commit: AgriSense Smart Farming Weather Prediction"
```

---

### 4. Connect to GitHub and push

Copy the commands GitHub shows you, or use:

```bash
git remote add origin https://github.com/YOUR_USERNAME/agrisense.git
git branch -M main
git push -u origin main
```

---

### 5. Verify on GitHub

Go to `https://github.com/YOUR_USERNAME/agrisense` — you should see all your files! ✅

---

## 🚀 Deploy Online (Optional)

### Option A — Frontend on GitHub Pages (free)

1. Go to your repo → **Settings** → **Pages**
2. Source: **Deploy from a branch**
3. Branch: `main`, folder: `/frontend`
4. Click Save
5. Your dashboard will be live at:
   `https://YOUR_USERNAME.github.io/agrisense/`

> Note: GitHub Pages is static — it will use the offline JS model (no Flask needed).

---

### Option B — Full Stack on Render.com (free)

1. Sign up at [render.com](https://render.com)
2. New → **Web Service** → Connect your GitHub repo
3. Settings:
   - **Build Command:** `pip install -r requirements.txt && python ml_model/train_model.py`
   - **Start Command:** `python backend/app.py`
4. Click **Deploy**
5. Update `fetch()` URL in `frontend/index.html` to your Render URL

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Training | Python, scikit-learn, pandas, numpy |
| Models | Random Forest, Gradient Boosting |
| Backend | Flask, Flask-CORS |
| Frontend | HTML5, CSS3, Vanilla JavaScript |
| Serialization | joblib (.pkl files) |

---

## 📈 Roadmap / Improvements

- [ ] Connect to real weather API (OpenWeatherMap, IMD)
- [ ] Add IoT sensor data ingestion
- [ ] Add 7-day forecast view
- [ ] Add historical charts (Chart.js)
- [ ] Add more crops (Sugarcane, Soybean, Turmeric…)
- [ ] Export predictions as PDF report
- [ ] Mobile app (PWA)

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Commit changes: `git commit -m "Add my feature"`
4. Push: `git push origin feature/my-feature`
5. Open a Pull Request

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 👤 Author

Built with ❤️ for smart farming and sustainable agriculture.

---

*If you found this helpful, please ⭐ star the repository!*
