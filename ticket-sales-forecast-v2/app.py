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

if "shows" not in st.session_state:
    st.session_state.shows = []

def add_show():
    st.session_state.shows.append({"name": "", "data": None, "avg_price": 0.0, "start_date": None, "end_date": None, "history": {}})

st.sidebar.header("Manage Shows")
if st.sidebar.button("Add New Show"):
    add_show()

for idx, show in enumerate(st.session_state.shows):
    with st.sidebar.expander(f"Show {idx+1} Details", expanded=True):
        show["name"] = st.text_input(f"Show Name {idx+1}", value=show.get("name", ""), key=f"name_{idx}")
        uploaded_file = st.file_uploader(f"Upload CSV for {show['name']} (date,tickets_sold)", type=["csv"], key=f"upload_{idx}")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file, parse_dates=["date"])
            if not {"date", "tickets_sold"}.issubset(df.columns):
                st.error("CSV must contain 'date' and 'tickets_sold' columns")
            else:
                df = df.sort_values("date").reset_index(drop=True)
                show["data"] = df
                show["start_date"] = df["date"].min().date()
                show["end_date"] = df["date"].max().date()
                st.success(f"Loaded data from {show['start_date']} to {show['end_date']} ({len(df)} rows)")
        else:
            st.info(f"Upload CSV for {show['name']} or add data manually below")

        # Allow manual avg price input (single price per show)
        show["avg_price"] = st.number_input(f"Average Ticket Price for {show['name']}", min_value=0.0, format="%.2f", value=show.get("avg_price", 0.0), key=f"price_{idx}")

        # If no data, allow manual daily entry too (optional)
        if show["data"] is None:
            st.write("No CSV uploaded - enter ticket sales data manually (optional)")
            start_date = st.date_input(f"Start Date for {show['name']}", key=f"start_manual_{idx}")
            end_date = st.date_input(f"End Date for {show['name']}", key=f"end_manual_{idx}")
            if start_date > end_date:
                st.error("Start date must be before end date")
            else:
                days = (end_date - start_date).days + 1
                rows = []
                for i in range(days):
                    rows.append({"date": start_date + timedelta(i), "tickets_sold": 0})
                manual_df = pd.DataFrame(rows)
                for i, row in manual_df.iterrows():
                    tickets = st.number_input(f"Tickets sold on {row['date']} for {show['name']}", min_value=0, key=f"manual_tickets_{idx}_{i}")
                    manual_df.at[i, "tickets_sold"] = tickets
                show["data"] = manual_df
                show["start_date"] = start_date
                show["end_date"] = end_date

if len(st.session_state.shows) == 0:
    st.info("Add at least one show on the left sidebar.")
    st.stop()

# Combine all shows data for analysis
all_data = []
for show in st.session_state.shows:
    if show["data"] is not None:
        df = show["data"].copy()
        df["show_name"] = show["name"]
        df["avg_price"] = show["avg_price"]
        df["days_to_show"] = (pd.to_datetime(show["end_date"]) - pd.to_datetime(df["date"])).dt.days * -1
        df["cumulative_sales"] = df["tickets_sold"].cumsum()
        all_data.append(df)

if not all_data:
    st.warning("No valid data loaded for any shows.")
    st.stop()

data = pd.concat(all_data).reset_index(drop=True)

# Sidebar filtering
shows_list = [s["name"] for s in st.session_state.shows]
selected_show = st.sidebar.selectbox("Select Show", shows_list)
df_show = data[data["show_name"] == selected_show]

date_min = df_show["date"].min()
date_max = df_show["date"].max()
selected_dates = st.sidebar.date_input("Filter dates", value=(date_min, date_max), min_value=date_min, max_value=date_max)
if len(selected_dates) == 2:
    df_show = df_show[(df_show["date"] >= pd.to_datetime(selected_dates[0])) & (df_show["date"] <= pd.to_datetime(selected_dates[1]))]

st.header(f"Data for show: {selected_show}")
st.dataframe(df_show[["date", "tickets_sold", "avg_price", "cumulative_sales"]])

# --- Forecasting isotonic regression ---
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

# Prophet daily forecast
m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
df_prophet = df_show[["date", "tickets_sold"]].rename(columns={"date":"ds","tickets_sold":"y"})
m.fit(df_prophet)
future = m.make_future_dataframe(periods=30)
forecast = m.predict(future)

# Velocity & acceleration
df_show = df_show.sort_values("date")
df_show["velocity"] = df_show["tickets_sold"].diff().fillna(0)
df_show["acceleration"] = df_show["velocity"].diff().fillna(0)

# Surge detection
vel_thresh = np.percentile(df_show["velocity"], 90)
accel_thresh = np.percentile(df_show["acceleration"], 90)
surge_days = df_show[(df_show["velocity"] >= vel_thresh) | (df_show["acceleration"] >= accel_thresh)]

def marketing_suggestions(surge_days, show_end_date, avg_price):
    suggestions = []
    if surge_days.empty:
        suggestions.append("No significant surge detected. Consider early promotions to boost initial sales.")
        if avg_price > 100:
            suggestions.append(f"High average ticket price (${avg_price:.2f}). Consider early bird discounts.")
        else:
            suggestions.append(f"Lower average ticket price (${avg_price:.2f}). Consider bundle/group offers.")
    else:
        first_surge = surge_days.iloc[0]["date"]
        days_before_show = (pd.to_datetime(show_end_date) - pd.to_datetime(first_surge)).days
        if days_before_show > 20:
            suggestions.append(f"Early surge detected on {first_surge.date()}. Maintain strong marketing & consider VIP packages.")
        elif 7 < days_before_show <= 20:
            suggestions.append(f"Mid-cycle surge on {first_surge.date()}. Run targeted ads & segment-specific offers.")
        else:
            suggestions.append(f"Late surge on {first_surge.date()}. Push last-minute deals and group discounts.")
        # Price trend
        suggestions.append("Communicate scarcity or price changes to boost urgency.")
    return suggestions

st.subheader("Sales, Price, Velocity & Acceleration")

import plotly.graph_objs as go
fig = go.Figure()
fig.add_trace(go.Scatter(x=df_show["date"], y=df_show["tickets_sold"], mode='lines+markers', name="Tickets Sold"))
fig.add_trace(go.Scatter(x=df_show["date"], y=[df_show["avg_price"].iloc[0]]*len(df_show), mode='lines', name="Average Price", yaxis="y2"))
fig.add_trace(go.Scatter(x=df_show["date"], y=df_show["velocity"], mode='lines', name="Velocity", yaxis="y3"))
fig.add_trace(go.Scatter(x=df_show["date"], y=df_show["acceleration"], mode='lines', name="Acceleration", yaxis="y4"))

fig.update_layout(
    yaxis=dict(title="Tickets Sold / Velocity"),
    yaxis2=dict(title="Price", overlaying="y", side="right", position=0.85),
    yaxis3=dict(title="Velocity", overlaying="y", side="right", position=0.90, showgrid=False, zeroline=False),
    yaxis4=dict(title="Acceleration", overlaying="y", side="right", position=0.95, showgrid=False, zeroline=False),
    legend=dict(x=0, y=1.2),
    height=500,
    margin=dict(l=40, r=120, t=40, b=40)
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Isotonic Regression Forecast with Confidence Intervals")
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

st.subheader("Prophet Forecast of Daily Tickets Sold")
from prophet.plot import plot_plotly
fig_prophet = plot_plotly(m, forecast)
st.plotly_chart(fig_prophet, use_container_width=True)

st.subheader("Surge Day Detection & Marketing Suggestions")
if surge_days.empty:
    st.write("No surge days detected.")
else:
    st.write("Detected Surge Days:")
    st.dataframe(surge_days[["date", "tickets_sold", "velocity", "acceleration"]])

suggestions = marketing_suggestions(surge_days, st.session_state.shows[shows_list.index(selected_show)]["end_date"], df_show["avg_price"].iloc[0])
st.write("Marketing Suggestions:")
for s in suggestions:
    st.info(s)

# --- Export ---
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    return output.getvalue()

st.subheader("Export Data & Forecasts")
if st.button("Download Raw Data (Filtered)"):
    tmp_download_link = to_excel(df_show)
    st.download_button(label="Download Excel", data=tmp_download_link, file_name=f"{selected_show}_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.button("Download Forecast Data"):
    tmp_forecast_link = to_excel(forecast_df)
    st.download_button(label="Download Excel", data=tmp_forecast_link, file_name=f"{selected_show}_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
