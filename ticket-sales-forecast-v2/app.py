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

try:
    data_all = pd.read_csv(uploaded_file, parse_dates=["date"])
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

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
shows_list = [f"{i+1}: {s['name']}" for i, s in enumerate(st.session_state.shows)]
selected_show_str = st.sidebar.selectbox("Select Show for Analysis", shows_list)
selected_idx = int(selected_show_str.split(":")[0]) - 1
current_show = st.session_state.shows[selected_idx]

df_show = data_all[
    (data_all["date"].dt.date >= current_show["start_date"]) &
    (data_all["date"].dt.date <= current_show["end_date"])
].copy()

if df_show.empty:
    st.warning(f"No ticket sales data found between {current_show['start_date']} and {current_show['end_date']} for {current_show['name']}.")
    st.stop()

df_show["avg_price"] = current_show["avg_price"]
df_show = df_show.sort_values("date")
df_show["days_to_show"] = (pd.to_datetime(current_show["end_date"]) - df_show["date"]).dt.days
df_show["cumulative_sales"] = df_show["tickets_sold"].cumsum()

st.header(f"Data for {current_show['name']}")
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
with st.spinner("Fitting global Prophet model..."):
    m_global.fit(global_df)

with st.spinner("Generating global future forecast..."):
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

if baseline_total == 0:
    perf_msg = "Baseline forecast total is zero, unable to calculate performance percentage."
else:
    perf_pct = ((show_total - baseline_total) / baseline_total) * 100
    if perf_pct > 0:
        perf_msg = f"**{current_show['name']} is performing {perf_pct:.1f}% ABOVE the global baseline forecast.** 🎉"
    else:
        perf_msg = f"**{current_show['name']} is performing {abs(perf_pct):.1f}% BELOW the global baseline forecast.** ⚠️"

st.subheader("Show Performance vs. Global Baseline")
st.success(perf_msg)

# --- Performance Breakdown ---
st.subheader("Performance Breakdown vs. Global Baseline")

performance_df = df_show[["date", "tickets_sold"]].merge(
    prophet_overlay[["ds", "yhat"]], left_on="date", right_on="ds", how="left"
)
performance_df["daily_variance"] = performance_df["tickets_sold"] - performance_df["yhat"]
performance_df["cumulative_variance"] = performance_df["daily_variance"].cumsum()

fig_daily = go.Figure()
fig_daily.add_trace(go.Bar(
    x=performance_df["date"],
    y=performance_df["daily_variance"],
    marker_color=np.where(performance_df["daily_variance"] >= 0, "green", "red"),
    name="Daily Variance"
))
fig_daily.update_layout(title="Daily Variance (Actual vs. Baseline)", xaxis_title="Date", yaxis_title="Tickets (±)")
st.plotly_chart(fig_daily, use_container_width=True)

fig_cum = go.Figure()
fig_cum.add_trace(go.Scatter(
    x=performance_df["date"], y=performance_df["cumulative_variance"],
    mode="lines+markers", name="Cumulative Variance", line=dict(color="blue")
))
fig_cum.update_layout(title="Cumulative Variance (Actual vs. Baseline)", xaxis_title="Date", yaxis_title="Cumulative Tickets (±)")
st.plotly_chart(fig_cum, use_container_width=True)

# --- GLOBAL BENCHMARK CUMULATIVE SALES CURVE (days_to_show) ---
def assign_show_end_date(row, shows):
    for show in shows:
        if show["start_date"] <= row["date"].date() <= show["end_date"]:
            return pd.to_datetime(show["end_date"])
    return pd.NaT

def assign_show_name(row, shows):
    for show in shows:
        if show["start_date"] <= row["date"].date() <= show["end_date"]:
            return show["name"]
    return None

data_all["show_end_date"] = data_all.apply(assign_show_end_date, axis=1, shows=st.session_state.shows)
data_all["show_name"] = data_all.apply(assign_show_name, axis=1, shows=st.session_state.shows)
data_filtered = data_all.dropna(subset=["show_end_date", "show_name"]).copy()
data_filtered["days_to_show"] = (data_filtered["show_end_date"] - data_filtered["date"]).dt.days

data_filtered = data_filtered.sort_values(["show_name", "date"])
data_filtered["cumulative_sales"] = data_filtered.groupby("show_name")["tickets_sold"].cumsum()

benchmark_df = data_filtered.groupby("days_to_show")["cumulative_sales"].median().reset_index()
benchmark_df = benchmark_df.sort_values("days_to_show")

ir_global = IsotonicRegression(increasing=True, out_of_bounds='clip')
benchmark_df["cumulative_sales_pred"] = ir_global.fit_transform(benchmark_df["days_to_show"], benchmark_df["cumulative_sales"])

# Predict benchmark cumulative sales for selected show
df_show["benchmark_cumulative_sales"] = ir_global.predict(df_show["days_to_show"])

# --- Isotonic Regression Forecast with Confidence Intervals (t-days) ---
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

fig_iso.add_trace(go.Scatter(
    x=df_show["days_to_show"],
    y=df_show["benchmark_cumulative_sales"],
    mode='lines',
    name="Benchmark (All Shows)",
    line=dict(color="orange", dash="dash")
))

fig_iso.update_layout(xaxis_title="Days to Show (0 = Show Day)", yaxis_title="Cumulative Tickets Sold")
st.plotly_chart(fig_iso, use_container_width=True)

# --- TODAY'S BENCHMARK CUMULATIVE SALES FOR SELECTED SHOW ---
today = pd.Timestamp.now().normalize()
if current_show["start_date"] <= today.date() <= current_show["end_date"]:
    days_left = (pd.to_datetime(current_show["end_date"]) - today).days
    predicted_benchmark = ir_global.predict([days_left])[0]
    st.markdown(f"### Benchmark cumulative tickets sold by today ({today.date()}): **{int(predicted_benchmark)}** tickets")
else:
    st.markdown("### Today's date is outside the show date range; benchmark not applicable.")

# --- NEW FEATURE: Predictive Baseline for ANY FUTURE SHOW ---

st.sidebar.header("Predict Baseline for New/Future Show")

future_show_date = st.sidebar.date_input(
    "Select Future Show Date for Baseline Prediction",
    value=pd.Timestamp.today().date() + timedelta(days=30),
    min_value=pd.Timestamp.today().date()
)

max_days_before_show = st.sidebar.slider(
    "Days Before Show to Forecast", min_value=7, max_value=90, value=60
)

def get_benchmark_curve(ir_model, max_days=60):
    days_range = np.arange(max_days, -1, -1)  # from max_days down to 0 (show day)
    predicted_cum_sales = ir_model.predict(days_range)
    return pd.DataFrame({
        "days_to_show": days_range,
        "predicted_cumulative_sales": predicted_cum_sales
    })

benchmark_curve_df = get_benchmark_curve(ir_global, max_days_before_show)

st.header(f"Predicted Baseline Cumulative Sales for New Show on {future_show_date}")
st.write(f"Expected cumulative ticket sales from {max_days_before_show} days before show day to show day.")

fig_benchmark = go.Figure()
fig_benchmark.add_trace(go.Scatter(
    x=benchmark_curve_df["days_to_show"],
    y=benchmark_curve_df["predicted_cumulative_sales"],
    mode="lines+markers",
    name="Baseline Cumulative Sales"
))
fig_benchmark.update_layout(
    title="Baseline Cumulative Ticket Sales by Days to Show",
    xaxis_title="Days to Show (0 = Show Day)",
    yaxis_title="Cumulative Tickets Sold",
    yaxis=dict(range=[0, benchmark_curve_df["predicted_cumulative_sales"].max() * 1.1])
)
st.plotly_chart(fig_benchmark, use_container_width=True)

st.dataframe(benchmark_curve_df)
