import streamlit as st
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
from datetime import timedelta
from io import BytesIO

st.set_page_config(page_title="Ticket Sales Forecasting + Marketing Insights", layout="wide")
st.title("🎭 Ticket Sales Forecasting & Marketing Suggestions")

# --- Upload Global CSV ---
st.sidebar.header("Upload Ticket Sales CSV")
uploaded_file = st.sidebar.file_uploader("Upload a CSV with 'date' and 'tickets_sold' columns", type=["csv"])

if uploaded_file is None:
    st.info("Please upload a CSV to begin.")
    st.stop()

# Load CSV
try:
    data_all = pd.read_csv(uploaded_file, parse_dates=["date"])
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Ensure columns are correct
if not {"date", "tickets_sold"}.issubset(data_all.columns):
    st.error("CSV must have columns: 'date', 'tickets_sold'")
    st.stop()

data_all = data_all.sort_values("date").reset_index(drop=True)
st.sidebar.success(f"CSV loaded: {len(data_all)} rows, from {data_all['date'].min().date()} to {data_all['date'].max().date()}")

# --- Shows Definition ---
if "shows" not in st.session_state:
    st.session_state.shows = []

def add_show():
    st.session_state.shows.append({
        "name": "",
        "avg_price": 0.0,
        "start_date": data_all["date"].min().date(),
        "end_date": data_all["date"].max().date()
    })

st.sidebar.header("Manage Shows")
if st.sidebar.button("Add New Show"):
    add_show()

for idx, show in enumerate(st.session_state.shows):
    with st.sidebar.expander(f"Show {idx+1} Details", expanded=True):
        show["name"] = st.text_input(f"Show Name {idx+1}", value=show.get("name", ""), key=f"name_{idx}")
        show["avg_price"] = st.number_input(
            f"Average Ticket Price ({show['name']})",
            min_value=0.0,
            format="%.2f",
            value=show.get("avg_price", 0.0),
            key=f"price_{idx}"
        )
        show["start_date"], show["end_date"] = st.date_input(
            f"Date Range ({show['name']})",
            value=(show.get("start_date", data_all["date"].min().date()),
                   show.get("end_date", data_all["date"].max().date())),
            min_value=data_all["date"].min().date(),
            max_value=data_all["date"].max().date(),
            key=f"range_{idx}"
        )

if len(st.session_state.shows) == 0:
    st.info("Add at least one show from the sidebar to continue.")
    st.stop()

# --- Prepare Data for Selected Show ---
shows_list = [s["name"] for s in st.session_state.shows]
selected_show = st.sidebar.selectbox("Select Show for Analysis", shows_list)

current_show = next(s for s in st.session_state.shows if s["name"] == selected_show)
df_show = data_all[
    (data_all["date"].dt.date >= current_show["start_date"]) &
    (data_all["date"].dt.date <= current_show["end_date"])
].copy()

if df_show.empty:
    st.warning(f"No ticket sales data found between {current_show['start_date']} and {current_show['end_date']} for {selected_show}.")
    st.stop()

df_show["avg_price"] = current_show["avg_price"]
df_show = df_show.sort_values("date")
df_show["days_to_show"] = (pd.to_datetime(current_show["end_date"]) - df_show["date"]).dt.days * -1
df_show["cumulative_sales"] = df_show["tickets_sold"].cumsum()

st.header(f"Data for {selected_show}")
st.dataframe(df_show[["date", "tickets_sold", "avg_price", "cumulative_sales"]])

# --- Isotonic Regression Forecast (t-days) ---
def isotonic_with_ci(x, y, confidence=0.95, n_bootstrap=200):
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    y_pred = ir.fit_transform(x, y)
    residuals = y - y_pred
    preds_bootstrap = []
    for _ in range(n_bootstrap):
        resampled = np.random.choice(residuals, size=len(residuals), replace=True)
        y_boot = y_pred + resampled
        y_boot_pred = IsotonicRegression(increasing=True, out_of_bounds='clip').fit_transform(x, y_boot)
        preds_bootstrap.append(y_boot_pred)
    preds_bootstrap = np.array(preds_bootstrap)
    lower = np.percentile(preds_bootstrap, (1-confidence)/2*100, axis=0)
    upper = np.percentile(preds_bootstrap, (1+confidence)/2*100, axis=0)
    return y_pred, lower, upper

x = df_show["days_to_show"].values
y = df_show["cumulative_sales"].values
if len(x) < 2:
    st.warning("Not enough data points to run forecasting.")
    st.stop()

y_pred, lower_ci, upper_ci = isotonic_with_ci(x, y)
forecast_df = pd.DataFrame({
    "days_to_show": x,
    "actual": y,
    "predicted": y_pred,
    "lower_ci": lower_ci,
    "upper_ci": upper_ci,
})

# --- Global Prophet Forecast (calendar dates) ---
global_df = data_all.groupby("date", as_index=False)["tickets_sold"].sum().rename(columns={"tickets_sold": "y"})
global_df = global_df.rename(columns={"date": "ds"})

m_global = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=True)
m_global.fit(global_df)

future_global = m_global.make_future_dataframe(periods=30)
forecast_global = m_global.predict(future_global)

# --- Overlay Prophet Global Forecast on Current Show ---
prophet_overlay = forecast_global[
    (forecast_global["ds"].dt.date >= current_show["start_date"]) &
    (forecast_global["ds"].dt.date <= current_show["end_date"])
][["ds", "yhat", "yhat_lower", "yhat_upper"]].reset_index(drop=True)

# --- Show Performance vs. Baseline ---
show_total = df_show["cumulative_sales"].iloc[-1]
baseline_total = prophet_overlay["yhat"].cumsum().iloc[-1]

perf_pct = ((show_total - baseline_total) / baseline_total) * 100
if perf_pct > 0:
    perf_msg = f"**{selected_show} is performing {perf_pct:.1f}% ABOVE the global baseline forecast.** 🎉"
else:
    perf_msg = f"**{selected_show} is performing {abs(perf_pct):.1f}% BELOW the global baseline forecast.** ⚠️"

st.subheader("Show Performance vs. Global Baseline")
st.success(perf_msg)

# --- Performance Breakdown ---
st.subheader("Performance Breakdown vs. Global Baseline")

# Merge actual vs baseline daily
performance_df = df_show[["date", "tickets_sold"]].merge(
    prophet_overlay[["ds", "yhat"]], left_on="date", right_on="ds", how="left"
)
performance_df["daily_variance"] = performance_df["tickets_sold"] - performance_df["yhat"]
performance_df["cumulative_variance"] = performance_df["daily_variance"].cumsum()

# Daily variance bar chart
fig_daily = go.Figure()
fig_daily.add_trace(go.Bar(
    x=performance_df["date"],
    y=performance_df["daily_variance"],
    marker_color=np.where(performance_df["daily_variance"] >= 0, "green", "red"),
    name="Daily Variance"
))
fig_daily.update_layout(title="Daily Variance (Actual vs. Baseline)", xaxis_title="Date", yaxis_title="Tickets (±)")
st.plotly_chart(fig_daily, use_container_width=True)

# Cumulative variance line chart
fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=performance_df["date"], y=performance_df["cumulative_variance"],
    mode="lines+markers", name="Cumulative Variance", line=dict(color="blue")
))
fig_cum.update_layout(title="Cumulative Variance (Actual vs. Baseline)", xaxis_title="Date", yaxis_title="Cumulative Tickets (±)")
st.plotly_chart(fig_cum, use_container_width=True)

# --- Remaining Plots ---
st.subheader("Isotonic Regression Forecast with Confidence Intervals (t-days)")
fig_iso = go.Figure()
fig_iso.add_trace(go.Scatter(x=forecast_df["days_to_show"], y=forecast_df["actual"], mode='markers', name="Actual"))
fig_iso.add_trace(go.Scatter(x=forecast_df["days_to_show"], y=forecast_df["predicted"], mode='lines', name="Predicted"))
fig_iso.add_trace(go.Scatter(
    x=np.concatenate([forecast_df["days_to_show"], forecast_df["days_to_show"][::-1]]),
    y=np.concatenate([forecast_df["lower_ci"], forecast_df["upper_ci"][::-1]]),
    fill='toself',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=False,
))
fig_iso.update_layout(xaxis_title="Days to Show (0 = Show Day)", yaxis_title="Cumulative Tickets Sold")
st.plotly_chart(fig_iso, use_container_width=True)
