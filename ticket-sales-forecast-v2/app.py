import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy.optimize import curve_fit

st.set_page_config(page_title="Ticket Sales Forecast", layout="wide")
st.title("🎭 Ticket Sales Forecasting App")

# --- Helper functions ---
def logistic_fn(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

def fit_model(X, y, model_type):
    """Fits a model and returns predict_fn, y_pred, r2."""
    if model_type.startswith("Polynomial"):
        degree = 2 if "deg 2" in model_type else 5
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model = LinearRegression().fit(X_poly, y)
        y_pred = model.predict(X_poly)
        return lambda x: np.maximum(model.predict(poly.transform(x)), 0), y_pred, r2_score(y, y_pred)

    elif model_type == "Linear Trend":
        model = LinearRegression().fit(X, y)
        y_pred = model.predict(X)
        return lambda x: np.maximum(model.predict(x), 0), y_pred, r2_score(y, y_pred)

    elif model_type == "Logistic Growth":
        p0 = [y.max() * 1.1, 0.1, np.median(X)]
        try:
            params, _ = curve_fit(logistic_fn, X.flatten(), y, p0=p0, maxfev=10000)
            y_pred = logistic_fn(X.flatten(), *params)
            return lambda x: np.maximum(logistic_fn(x.flatten(), *params), 0), y_pred, r2_score(y, y_pred)
        except RuntimeError:
            return lambda x: np.zeros_like(x.flatten()), np.zeros_like(y), -np.inf

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV with columns: `date`, `tickets_sold`", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    if 'date' not in df.columns or 'tickets_sold' not in df.columns:
        st.error("CSV must have columns: 'date' and 'tickets_sold'")
    else:
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        st.write("### Uploaded Data", df)

        # --- Show Date Ranges ---
        st.subheader("Show Date Attribution")
        n_shows = st.number_input("Number of past shows to analyze:", min_value=1, value=1, step=1)

        show_ranges = []
        for i in range(n_shows):
            st.write(f"**Show {i+1} Dates:**")
            col1, col2 = st.columns(2)
            start = col1.date_input(f"Show {i+1} Start Date")
            end = col2.date_input(f"Show {i+1} End Date")
            show_ranges.append((start, end))

        # --- Process Data ---
        processed_data = []
        for i, (start, end) in enumerate(show_ranges):
            mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
            show_df = df[mask].copy()
            show_df['days_to_show'] = (pd.to_datetime(end) - show_df['date']).dt.days * -1
            show_df['cumulative_sales'] = show_df['tickets_sold'].cumsum()
            processed_data.append(show_df)
            st.write(f"### Show {i+1} Data", show_df)

        # --- Daily vs Cumulative Toggle ---
        view_type = st.radio("View & Forecast Type", ["Daily Sales", "Cumulative Sales"])

        # --- Plot Sales Curves ---
        fig, ax = plt.subplots()
        for i, show_df in enumerate(processed_data):
            y_values = show_df['cumulative_sales'] if view_type == "Cumulative Sales" else show_df['tickets_sold']
            ax.scatter(show_df['days_to_show'], y_values, label=f"Show {i+1}")
        ax.set_xlabel("Days to Show (0 = Show Day)")
        ax.set_ylabel("Cumulative Tickets" if view_type == "Cumulative Sales" else "Daily Tickets")
        ax.legend()
        st.pyplot(fig)

        # --- Prepare Data ---
        all_sales = pd.concat(processed_data)
        y_col = 'cumulative_sales' if view_type == "Cumulative Sales" else 'tickets_sold'
        X = all_sales['days_to_show'].values.reshape(-1, 1)
        y = all_sales[y_col].values

        # --- Model Comparison ---
        st.subheader("Model Comparison")
        models = ["Polynomial (deg 2)", "Polynomial (deg 5)", "Linear Trend", "Logistic Growth"]
        results = []

        fig_comp, ax_comp = plt.subplots()
        ax_comp.scatter(X, y, label="Historical Data", color='blue')

        for model_type in models:
            predict_fn, y_pred, r2 = fit_model(X, y, model_type)
            results.append((model_type, r2))
            x_sorted = np.sort(X.flatten()).reshape(-1, 1)
            ax_comp.plot(x_sorted, predict_fn(x_sorted), label=f"{model_type} (R²={r2:.3f})")

        ax_comp.set_xlabel("Days to Show")
        ax_comp.set_ylabel("Tickets")
        ax_comp.legend()
        st.pyplot(fig_comp)

        st.write("### Model R² Scores")
        st.write(pd.DataFrame(results, columns=["Model", "R²"]).sort_values("R²", ascending=False))

        # --- Best Model Selection ---
        best_model = max(results, key=lambda x: x[1])[0]
        st.success(f"Best model based on R²: **{best_model}**")

        # --- Forecast range slider ---
        forecast_days = st.slider("Forecast range (days before show)", min_value=10, max_value=120, value=30)
        future_days = np.arange(X.min(), 1).reshape(-1, 1)

        # --- Best Model Forecast ---
        predict_fn, _, _ = fit_model(X, y, best_model)
        forecast_sales = predict_fn(future_days)
        forecast_df = pd.DataFrame({
            "days_to_show": future_days.flatten(),
            f"predicted_{y_col}": forecast_sales
        })
        st.write("### Forecasted Ticket Sales (Best Model)", forecast_df)

        fig_forecast, ax_forecast = plt.subplots()
        ax_forecast.plot(future_days, forecast_sales, color='green', label="Forecast (Best Model)")
        ax_forecast.set_xlabel("Days to Show")
        ax_forecast.set_ylabel("Predicted " + ("Cumulative" if view_type == "Cumulative Sales" else "Daily") + " Tickets")
        ax_forecast.legend()
        st.pyplot(fig_forecast)

        # --- Ensemble Forecasting Option ---
        ensemble_enabled = st.checkbox("Enable Ensemble Forecast (average of all models)", value=False)

        if ensemble_enabled:
            preds = []
            for model_type in models:
                predict_fn_ens, _, _ = fit_model(X, y, model_type)
                preds.append(predict_fn_ens(future_days))
            ensemble_forecast = np.maximum(np.mean(preds, axis=0), 0)

            ensemble_df = pd.DataFrame({
                "days_to_show": future_days.flatten(),
                f"predicted_{y_col}": ensemble_forecast
            })
            st.write("### Ensemble Forecasted Ticket Sales", ensemble_df)

            fig_ensemble, ax_ens = plt.subplots()
            ax_ens.plot(future_days, ensemble_forecast, color='purple', label="Ensemble Forecast")
            ax_ens.set_xlabel("Days to Show")
            ax_ens.set_ylabel("Predicted " + ("Cumulative" if view_type == "Cumulative Sales" else "Daily") + " Tickets")
            ax_ens.legend()
            st.pyplot(fig_ensemble)

else:
    st.info("Please upload a CSV to get started.")
