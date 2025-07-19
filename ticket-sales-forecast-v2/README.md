# 🎭 Ticket Sales Forecasting App

This Streamlit app analyzes past ticket sales data, attributes sales to specific shows, and forecasts future ticket sales using polynomial regression (degree 2 and 5).

---

## 🚀 Features
- Upload your ticket sales CSV (`date`, `tickets_sold`).
- Attribute sales to shows by setting start and end dates.
- Visualize ticket sales from first sale day to show day (0).
- Fit polynomial regression (degree 2 or 5) and show R² score.
- Forecast future shows with customizable days range.

---

## 📦 Installation & Run

1. Clone this repository:
   ```bash
   git clone https://github.com/YOUR_USERNAME/ticket-sales-forecast.git
   cd ticket-sales-forecast
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

4. Open your browser at `http://localhost:8501`.

---

## ☁️ Deploy to Streamlit Cloud
Click the button below to deploy your own version:

[![Deploy to Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

---

## 📂 Sample Data
See [sample_data.csv](sample_data.csv) for an example dataset.

---

## 🛠 Future Improvements
- Add logistic/exponential regression models.
- Implement ARIMA/Prophet for time-series forecasting.
- Store and load show configurations from a database.
- Add interactive dashboards with filters and cumulative curves.
- Allow real-time sales data input for live tracking.
- Export forecasts as CSV or PDF reports.

- Integrate seat map visualization for per-section sales tracking.
- Implement machine learning models for ticket price optimization.
- Add anomaly detection for unusual sales patterns.
- Include comparative analytics between shows (sales velocity, peak days).
- Enable automated daily data refresh from CSV or API.
