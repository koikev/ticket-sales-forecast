import streamlit as st
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import plotly.graph_objs as go

st.title("🎭 Simple Ticket Sales Baseline Forecasting")

uploaded_file = st.file_uploader("Upload CSV with columns: date, tickets_sold", type=["csv"])
if uploaded_file is None:
    st.stop()

df_all = pd.read_csv(uploaded_file, parse_dates=["date"])
if not {"date", "tickets_sold"}.issubset(df_all.columns):
    st.error("CSV must have 'date' and 'tickets_sold' columns")
    st.stop()

df_all = df_all.sort_values("date")

# Shows management
if "shows" not in st.session_state:
    st.session_state.shows = []

def add_show():
    st.session_state.shows.append({
        "name": "",
        "start_date": df_all["date"].min().date(),
        "end_date": df_all["date"].max().date()
    })

st.sidebar.header("Manage Shows")
if st.sidebar.button("Add Show"):
    add_show()

for i, show in enumerate(st.session_state.shows):
    with st.sidebar.expander(f"Show {i+1}"):
        show["name"] = st.text_input("Show Name", value=show.get("name", ""), key=f"name_{i}")
        show["start_date"], show["end_date"] = st.date_input(
            "Date Range",
            value=(show.get("start_date", df_all["date"].min().date()),
                   show.get("end_date", df_all["date"].max().date())),
            min_value=df_all["date"].min().date(),
            max_value=df_all["date"].max().date(),
            key=f"range_{i}"
        )

if len(st.session_state.shows) == 0 or any(s["name"] == "" for s in st.session_state.shows):
    st.info("Please add at least one show with a valid name in the sidebar")
    st.stop()

# Select show
show_names = [s["name"] for s in st.session_state.shows]
selected_show = st.selectbox("Select Show to Analyze", show_names)
current_show = next(s for s in st.session_state.shows if s["name"] == selected_show)

# Filter show data
df_show = df_all[(df_all["date"].dt.date >= current_show["start_date"]) & (df_all["date"].dt.date <= current_show["end_date"])].copy()
if df_show.empty:
    st.warning("No sales data in this date range for this show")
    st.stop()

df_show = df_show.sort_values("date")
df_show["days_to_show"] = (pd.to_datetime(current_show["end_date"]) - df_show["date"]).dt.days
df_show["cumulative_sales"] = df_show["tickets_sold"].cumsum()

st.subheader(f"Sales Data for Show '{selected_show}'")
st.dataframe(df_show[["date", "tickets_sold", "days_to_show", "cumulative_sales"]])

# Build baseline from all shows
all_cum = []
for show in st.session_state.shows:
    if show["name"] == "":
        continue
    tmp = df_all[(df_all["date"].dt.date >= show["start_date"]) & (df_all["date"].dt.date <= show["end_date"])].copy()
    if tmp.empty:
        continue
    tmp = tmp.sort_values("date")
    tmp["days_to_show"] = (pd.to_datetime(show["end_date"]) - tmp["date"]).dt.days
    tmp["cumulative_sales"] = tmp["tickets_sold"].cumsum()
    tmp["show_name"] = show["name"]
    all_cum.append(tmp[["show_name", "days_to_show", "cumulative_sales"]])

if not all_cum:
    st.error("No shows with sales data to create baseline")
    st.stop()

df_all_cum = pd.concat(all_cum)

# Calculate median cumulative sales per day across all shows
baseline = df_all_cum.groupby("days_to_show")["cumulative_sales"].median().reset_index().sort_values("days_to_show")

# Isotonic regression for smooth baseline curve
ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
baseline["baseline_sales"] = ir.fit_transform(baseline["days_to_show"], baseline["cumulative_sales"])

# Plot actual vs baseline
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_show["days_to_show"],
    y=df_show["cumulative_sales"],
    mode="lines+markers",
    name="Actual Cumulative Sales"
))
fig.add_trace(go.Scatter(
    x=baseline["days_to_show"],
    y=baseline["baseline_sales"],
    mode="lines",
    name="Baseline Median",
    line=dict(dash="dash", color="orange")
))
fig.update_layout(
    title=f"Cumulative Tickets Sold vs Days to Show for '{selected_show}'",
    xaxis_title="Days to Show (0 = Show Day)",
    yaxis_title="Cumulative Tickets Sold",
    yaxis=dict(range=[0, max(df_show["cumulative_sales"].max(), baseline["baseline_sales"].max()) * 1.1])
)
st.plotly_chart(fig, use_container_width=True)

# Predict for future show
st.header("Predict Baseline for Future Show")
future_show_days = st.slider("Days to Show (Forecast Horizon)", 7, 90, 60)
future_days = np.arange(future_show_days, -1, -1)
future_preds = ir.predict(future_days)

future_df = pd.DataFrame({
    "days_to_show": future_days,
    "predicted_cumulative_sales": future_preds
})

fig_future = go.Figure()
fig_future.add_trace(go.Scatter(
    x=future_df["days_to_show"],
    y=future_df["predicted_cumulative_sales"],
    mode="lines+markers",
    name="Baseline Prediction"
))
fig_future.update_layout(
    title="Baseline Predicted Cumulative Sales for Future Show",
    xaxis_title="Days to Show",
    yaxis_title="Cumulative Tickets Sold"
)
st.plotly_chart(fig_future, use_container_width=True)
st.dataframe(future_df)
