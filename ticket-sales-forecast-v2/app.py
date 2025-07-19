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

# --- Session state setup ---
if "shows" not in st.session_state:
    st.session_state.shows = []

def add_show():
    st.session_state.shows.append({"name": "", "start_date": None, "end_date": None, "data": None, "history": {}})

st.sidebar.header("Show & Date Filters")
st.sidebar.button("Add New Show", on_click=add_show)

# --- Add / Edit shows ---
for idx, show in enumerate(st.session_state.shows):
    with st.sidebar.expander(f"Show {idx+1} Details", expanded=True):
        show["name"] = st.text_input(f"Show Name {idx+1}", value=show.get("name", ""), key=f"name_{idx}")
        show["start_date"] = st.date_input(f"Start Date {idx+1}", value=show.get("start_date") or pd.Timestamp.today().date(), key=f"start_{idx}")
        show["end_date"] = st.date_input(f"End Date {idx+1}", value=show.get("end_date") or pd.Timestamp.today().date(), key=f"end_{idx}")

        if show["start_date"] > show["end_date"]:
            st.sidebar.error("Start date must be before end date")
            continue

        days = (show["end_date"] - show["start_date"]).days + 1

        # --- Segmentation: customer groups ---
        if "segments" not in show:
            show["segments"] = ["General"]  # default one segment

        segments_str = st.text_input(f"Segments (comma separated) for Show {idx+1}", value=",".join(show["segments"]), key=f"segments_{idx}")
        show["segments"] = [s.strip() for s in segments_str.split(",") if s.strip()]

        # Initialize or update data
        if show["data"] is None or len(show["data"]) != days * len(show["segments"]):
            rows = []
            for seg in show["segments"]:
                for i in range(days):
                    rows.append({
                        "date": show["start_date"] + timedelta(i),
                        "tickets_sold": 0,
                        "price": 0.0,
                        "segment": seg
                    })
            show["data"] = pd.DataFrame(rows)

        st.write(f"### Enter daily tickets sold & price for {show['name']} ({days} days) and segments")

        df = show["data"]

        # Editable table: tickets_sold and price per date & segment
        for seg in show["segments"]:
            st.markdown(f"**Segment: {seg}**")
            seg_df = df[df["segment"] == seg].copy()
            for i, row in seg_df.iterrows():
                cols = st.columns(2)
                with cols[0]:
                    tickets = st.number_input(f"Tickets sold {row['date']} ({seg})", min_value=0, value=int(row["tickets_sold"]), key=f"tickets_{idx}_{seg}_{i}")
                    df.at[i, "tickets_sold"] = tickets
                with cols[1]:
                    price = st.number_input(f"Price {row['date']} ({seg})", min_value=0.0, format="%.2f", value=float(row["price"]), key=f"price_{idx}_{seg}_{i}")
                    df.at[i, "price"] = price

        show["data"] = df

st.write("---")

if len(st.session_state.shows) == 0:
    st.info("Add at least one show on the left sidebar to enter ticket sales, price, and segmentation.")
    st.stop()

# Combine all shows & segments into one DataFrame
all_data = []
for show in st.session_state.shows:
    df = show["data"].copy()
    df["show_name"] = show["name"]
    df["days_to_show"] = (pd.to_datetime(show["end_date"]) - pd.to_datetime(df["date"])).dt.days * -1
    df["cumulative_sales"] = df.groupby("segment")["tickets_sold"].cumsum()
    all_data.append(df)

data = pd.concat(all_data).reset_index(drop=True)

# Sidebar filters
shows_list = [s["name"] for s in st.session_state.shows]
selected_show = st.sidebar.selectbox("Select Show", shows_list)
df_show = data[data["show_name"] == selected_show]

segments_list = df_show["segment"].unique().tolist()
selected_segments = st.sidebar.multiselect("Select Segments", options=segments_list, default=segments_list)
df_show = df_show[df_show["segment"].isin(selected_segments)]

date_min = df_show["date"].min()
date_max = df_show["date"].max()
selected_dates = st.sidebar.date_input("Filter dates", value=(date_min, date_max), min_value=date_min, max_value=date_max)
if len(selected_dates) == 2:
    df_show = df_show[(df_show["date"] >= pd.to_datetime(selected_dates[0])) & (df_show["date"] <= pd.to_datetime(selected_dates[1]))]

st.header(f"Data for show: {selected_show} (Segments: {', '.join(selected_segments)})")
st.dataframe(df_show[["date", "segment", "tickets_sold", "price", "cumulative_sales"]])

# --- Forecasting per segment with isotonic regression and CI ---
def isotonic_with_ci(x, y, confidence=0.95, n_bootstrap=200):
    ir = IsotonicRegression(increasing=True, out_of_bounds='clip')
    y_pred = ir.fit_transform(x, y)

    # Bootstrap residuals for CI
    residuals = y - y_pred
    preds_bootstrap = []
    for _ in range(n_bootstrap):
        resampled = np.random.choice(residuals, size=len(residuals), replace=True)
        y_boot = y_pred + resampled
        y_boot_pred = IsotonicRegression(increasing=True, out_of_bounds='clip').fit_transform(x, y_boot)
        preds_bootstrap.append(y_boot_pred)
    preds_bootstrap = np.array(preds_bootstrap)
    lower = np.percentile(preds_bootstrap, (1-confidence)/2*100, axis=0)
    upper = np.percentile(preds_bootstrap, (1+(confidence))/2*100, axis=0)

    return y_pred, lower, upper

# Aggregate forecasts for all selected segments
forecast_dfs = []
for seg in selected_segments:
    seg_df = df_show[df_show["segment"] == seg].sort_values("days_to_show")
    x = seg_df["days_to_show"].values
    y = seg_df["cumulative_sales"].values
    if len(x) < 2:
        continue
    y_pred, lower_ci, upper_ci = isotonic_with_ci(x, y)
    temp_df = pd.DataFrame({
        "days_to_show": x,
        "actual": y,
        "predicted": y_pred,
        "lower_ci": lower_ci,
        "upper_ci": upper_ci,
        "segment": seg
    })
    forecast_dfs.append(temp_df)
forecast_df = pd.concat(forecast_dfs) if forecast_dfs else pd.DataFrame()

# Prophet model for daily tickets per segment and overall (optional)
prophet_forecasts = []
for seg in selected_segments:
    seg_df = df_show[df_show["segment"] == seg][["date", "tickets_sold"]].rename(columns={"date":"ds", "tickets_sold":"y"})
    if len(seg_df) < 2:
        continue
    m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
    m.fit(seg_df)
    future = m.make_future_dataframe(periods=30)
    forecast = m.predict(future)
    forecast["segment"] = seg
    prophet_forecasts.append(forecast)
prophet_df = pd.concat(prophet_forecasts) if prophet_forecasts else pd.DataFrame()

# --- Velocity & Acceleration ---
df_show = df_show.sort_values(["segment", "date"])
df_show["velocity"] = df_show.groupby("segment")["tickets_sold"].diff().fillna(0)
df_show["acceleration"] = df_show.groupby("segment")["velocity"].diff().fillna(0)

# Surge day detection (per segment)
surge_results = []
for seg in selected_segments:
    seg_df = df_show[df_show["segment"] == seg]
    vel_thresh = np.percentile(seg_df["velocity"], 90)
    accel_thresh = np.percentile(seg_df["acceleration"], 90)
    surge_days = seg_df[(seg_df["velocity"] >= vel_thresh) | (seg_df["acceleration"] >= accel_thresh)]
    surge_results.append((seg, surge_days))

def marketing_suggestions_advanced(surge_days, show_end_date, prices, segment):
    suggestions = []
    if surge_days.empty:
        suggestions.append("No significant surge detected. Consider running early promotions to boost initial sales.")
        # Price-based suggestion
        avg_price = prices.mean() if not prices.empty else 0
        if avg_price > 100:
            suggestions.append(f"High average ticket price (${avg_price:.2f}). Consider offering early bird discounts.")
        else:
            suggestions.append(f"Lower average ticket price (${avg_price:.2f}). Consider bundle/group offers.")
    else:
        first_surge_day = surge_days.iloc[0]["date"]
        days_before_show = (pd.to_datetime(show_end_date) - pd.to_datetime(first_surge_day)).days
        if days_before_show > 20:
            suggestions.append(f"Early surge detected on {first_surge_day.date()}. Maintain strong marketing and consider VIP packages.")
        elif 7 < days_before_show <= 20:
            suggestions.append(f"Mid-cycle surge on {first_surge_day.date()}. Run targeted ads and segment-specific offers.")
        else:
            suggestions.append(f"Late surge on {first_surge_day.date()}. Push last-minute deals and group discounts.")
        # Price trend suggestion
        if prices.is_monotonic_increasing:
            suggestions.append("Ticket prices rising — consider communicating scarcity to boost urgency.")
        elif prices.is_monotonic_decreasing:
            suggestions.append("Ticket prices dropping — good for last-minute sales boost.")
    return suggestions

# --- Plots with Plotly ---
import plotly.express as px
import plotly.graph_objects as go

st.subheader("Sales, Price, Velocity & Acceleration")

fig = go.Figure()

for seg in selected_segments:
    seg_df = df_show[df_show["segment"] == seg]
    fig.add_trace(go.Scatter(x=seg_df["date"], y=seg_df["tickets_sold"], mode='lines+markers', name=f"Tickets Sold ({seg})"))
    fig.add_trace(go.Scatter(x=seg_df["date"], y=seg_df["price"], mode='lines+markers', name=f"Price ({seg})", yaxis="y2"))
    fig.add_trace(go.Scatter(x=seg_df["date"], y=seg_df["velocity"], mode='lines', name=f"Velocity ({seg})", yaxis="y3"))
    fig.add_trace(go.Scatter(x=seg_df["date"], y=seg_df["acceleration"], mode='lines', name=f"Acceleration ({seg})", yaxis="y4"))

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
for seg in selected_segments:
    seg_forecast = forecast_df[forecast_df["segment"] == seg]
    if seg_forecast.empty:
        continue
    fig_iso.add_trace(go.Scatter(x=seg_forecast["days_to_show"], y=seg_forecast["actual"], mode='markers', name=f"Actual ({seg})"))
    fig_iso.add_trace(go.Scatter(x=seg_forecast["days_to_show"], y=seg_forecast["predicted"], mode='lines', name=f"Predicted ({seg})"))
    fig_iso.add_trace(go.Scatter(
        x=np.concatenate([seg_forecast["days_to_show"], seg_forecast["days_to_show"][::-1]]),
        y=np.concatenate([seg_forecast["lower_ci"], seg_forecast["upper_ci"][::-1]]),
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False,
        name=f"CI ({seg})"
    ))
fig_iso.update_layout(xaxis_title="Days to Show (0 = Show Day)", yaxis_title="Cumulative Tickets Sold")
st.plotly_chart(fig_iso, use_container_width=True)

st.subheader("Prophet Forecast of Daily Tickets Sold")
if not prophet_df.empty:
    for seg in selected_segments:
        seg_forecast = prophet_df[prophet_df["segment"] == seg]
        if seg_forecast.empty:
            continue
        m = Prophet(daily_seasonality=True, yearly_seasonality=False, weekly_seasonality=False)
        # Refit model to plot again (could optimize)
        seg_hist = df_show[(df_show["segment"] == seg)][["date","tickets_sold"]].rename(columns={"date":"ds","tickets_sold":"y"})
        m.fit(seg_hist)
        fig_prophet = plot_plotly(m, seg_forecast)
        st.plotly_chart(fig_prophet, use_container_width=True)
else:
    st.info("Not enough data for Prophet forecast.")

st.subheader("Surge Day Detection & Marketing Suggestions")
for seg, surge_days in surge_results:
    st.markdown(f"**Segment: {seg}**")
    if surge_days.empty:
        st.write("No surge days detected.")
    else:
        st.write("Detected Surge Days:")
        st.dataframe(surge_days[["date", "tickets_sold", "velocity", "acceleration"]])
    seg_prices = df_show[df_show["segment"] == seg]["price"]
    suggestions = marketing_suggestions_advanced(surge_days, st.session_state.shows[shows_list.index(selected_show)]["end_date"], seg_prices, seg)
    st.write("Marketing Suggestions:")
    for s in suggestions:
        st.info(s)

# --- Model retraining & accuracy monitoring ---

st.subheader("Model Retraining & Forecast Accuracy Monitoring")

# For demo: simple error metric comparing isotonic prediction to actuals on filtered data
if not forecast_df.empty:
    merged = pd.merge(df_show, forecast_df, how="inner", on=["days_to_show", "segment"])
    merged["abs_error"] = (merged["cumulative_sales"] - merged["predicted"]).abs()
    mae = merged.groupby("segment")["abs_error"].mean()
    st.write("Mean Absolute Error (MAE) by Segment for Isotonic Regression Forecast:")
    st.dataframe(mae)

    fig_err = go.Figure()
    for seg in selected_segments:
        seg_err = merged[merged["segment"] == seg]
        if seg_err.empty:
            continue
        fig_err.add_trace(go.Scatter(x=seg_err["days_to_show"], y=seg_err["abs_error"], mode="lines+markers", name=f"Error ({seg})"))
    fig_err.update_layout(title="Forecast Absolute Errors Over Time", xaxis_title="Days to Show", yaxis_title="Absolute Error")
    st.plotly_chart(fig_err, use_container_width=True)
else:
    st.info("Not enough forecast data to calculate accuracy metrics.")

# Button to retrain all models (here we just rerun everything, but could add persistence / versioning)
if st.button("Retrain Models Now"):
    st.experimental_rerun()

# --- Export data & forecasts ---
def to_excel(df):
    output = BytesIO()
    writer = pd.ExcelWriter(output, engine='xlsxwriter')
    df.to_excel(writer, index=False)
    writer.save()
    processed_data = output.getvalue()
    return processed_data

st.subheader("Export Data & Forecasts")

if st.button("Download Raw Data (Filtered)"):
    tmp_download_link = to_excel(df_show)
    st.download_button(label="Download Excel", data=tmp_download_link, file_name=f"{selected_show}_data.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

if st.button("Download Forecast Data"):
    if not forecast_df.empty:
        tmp_forecast_link = to_excel(forecast_df)
        st.download_button(label="Download Excel", data=tmp_forecast_link, file_name=f"{selected_show}_forecast.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    else:
        st.warning("No forecast data to download.")
