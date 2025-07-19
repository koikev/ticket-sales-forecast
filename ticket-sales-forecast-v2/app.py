import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Ticket Sales Forecast", layout="wide")
st.title("🎭 Ticket Sales Forecasting App")

# --- File Upload ---
uploaded_file = st.file_uploader("Upload CSV with columns: `date`, `tickets_sold`", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['date'])
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

    # --- Process Data for Each Show ---
    processed_data = []
    for i, (start, end) in enumerate(show_ranges):
        mask = (df['date'] >= pd.to_datetime(start)) & (df['date'] <= pd.to_datetime(end))
        show_df = df[mask].copy()
        show_df['days_to_show'] = (pd.to_datetime(end) - show_df['date']).dt.days * -1  # 0 is show day
        processed_data.append(show_df)
        st.write(f"### Show {i+1} Data", show_df)

    # --- Plot Sales Curves ---
    fig, ax = plt.subplots()
    for i, show_df in enumerate(processed_data):
        ax.scatter(show_df['days_to_show'], show_df['tickets_sold'], label=f"Show {i+1}")
    ax.set_xlabel("Days to Show (0 = Show Day)")
    ax.set_ylabel("Tickets Sold")
    ax.legend()
    st.pyplot(fig)

    # --- Regression ---
    st.subheader("Regression & Forecast")
    degree = st.selectbox("Select Polynomial Degree:", [2, 5], index=0)
    
    all_sales = pd.concat(processed_data)
    X = all_sales['days_to_show'].values.reshape(-1, 1)
    y = all_sales['tickets_sold'].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)
    y_pred = model.predict(X_poly)
    
    r2 = r2_score(y, y_pred)
    st.write(f"**R² for degree {degree}: {r2:.4f}**")

    # Plot regression curve
    x_fit = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    x_fit_poly = poly.transform(x_fit)
    y_fit = model.predict(x_fit_poly)

    fig2, ax2 = plt.subplots()
    ax2.scatter(X, y, color='blue', label="Historical Data")
    ax2.plot(x_fit, y_fit, color='red', label=f"Poly Deg {degree} Fit")
    ax2.set_xlabel("Days to Show")
    ax2.set_ylabel("Tickets Sold")
    ax2.legend()
    st.pyplot(fig2)

    # --- Forecast ---
    st.subheader("Forecast Future Show")
    forecast_days = st.slider("Forecast range (days before show)", min_value=10, max_value=120, value=30)
    future_days = np.arange(-forecast_days, 1).reshape(-1, 1)
    future_days_poly = poly.transform(future_days)
    forecast_sales = model.predict(future_days_poly)
    
    forecast_df = pd.DataFrame({"days_to_show": future_days.flatten(), "predicted_tickets": forecast_sales})
    st.write("### Forecasted Ticket Sales", forecast_df)

    fig3, ax3 = plt.subplots()
    ax3.plot(forecast_df['days_to_show'], forecast_df['predicted_tickets'], color='green')
    ax3.set_xlabel("Days to Show")
    ax3.set_ylabel("Predicted Tickets Sold")
    st.pyplot(fig3)
else:
    st.info("Please upload a CSV to get started.")
