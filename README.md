# 🌍 AI-Powered Carbon Emission Analyzer

An AI-driven tool that helps organisations measure, analyse, and reduce their carbon footprint. The system accepts monthly activity data, predicts CO₂ emissions using machine-learning regression, rates the organisation's sustainability, and provides actionable AI recommendations.

---

## ✨ Features

| Feature | Description |
|---|---|
| 📊 **Emission Dashboard** | Pie chart, bar chart, and 12-month trend for all emission categories |
| 🤖 **AI Prediction** | Random Forest / Gradient Boosting / Linear Regression models trained on synthetic data |
| 🏷️ **Carbon Score Rating** | A (Sustainable) → D (Critical) grading system |
| 🌱 **Carbon Simulator** | What-if sliders to model solar, EV, remote-work, and other interventions |
| 💡 **AI Recommendations** | Category-specific, priority-ranked sustainability suggestions |
| 🤖 **Sustainability Chatbot** | Keyword-driven Q&A chatbot covering emissions, offsets, EVs, cloud, and more |
| 📄 **PDF Report Export** | Downloadable professional report with all data and recommendations |

---

## 🏗️ Architecture

```
┌──────────────────────┐
│ Organisation Inputs  │  electricity · fuel · travel · cloud · waste
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Data Preprocessing  │  Normalisation via StandardScaler
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│   AI Prediction      │  RandomForest / GradientBoosting / LinearRegression
│   (scikit-learn)     │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Carbon Footprint     │  Emission factor calculation + ML prediction
│ Score & Rating A–D   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ AI Sustainability    │  Rule-based recommendation engine
│ Advisor              │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Dashboard & Report   │  Streamlit + Plotly + PDF (ReportLab)
└──────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the application

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📦 Tech Stack

| Layer | Technology |
|---|---|
| Frontend / UI | [Streamlit](https://streamlit.io) |
| AI / ML | [scikit-learn](https://scikit-learn.org) (Random Forest, Gradient Boosting, Linear Regression) |
| Visualisation | [Plotly](https://plotly.com) |
| PDF Reports | [ReportLab](https://www.reportlab.com) |
| Data Processing | [pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) |

---

## 📁 Project Structure

```
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── backend/
│   ├── carbon_calculator.py   # Emission factor calculations & scoring
│   ├── model.py               # ML model training & prediction
│   ├── recommendations.py     # AI recommendation engine & simulator
│   └── report_generator.py    # PDF report generation
└── tests/
    └── test_carbon_analyzer.py # Unit tests
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

---

## 📊 Emission Factors Used

| Category | Factor |
|---|---|
| Electricity | 0.233 kg CO₂ / kWh (US grid average) |
| Fuel (diesel) | 2.64 kg CO₂ / litre |
| Business Travel | 0.089 kg CO₂ / km (average air) |
| Cloud / Servers | 0.40 kg CO₂ / server-hour |
| Office Waste | 0.50 kg CO₂ / kg |

---

## 🏷️ Carbon Score Ratings

| Grade | Label | Monthly CO₂ |
|---|---|---|
| **A** | 🌱 Sustainable | < 5 tons |
| **B** | 🌿 Moderate | 5 – 15 tons |
| **C** | ⚠️ High Emissions | 15 – 30 tons |
| **D** | 🚨 Critical | ≥ 30 tons |

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
