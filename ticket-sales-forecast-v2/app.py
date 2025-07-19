import streamlit as st
import pandas as pd
import numpy as np
from sklearn.isotonic import IsotonicRegression
import plotly.graph_objs as go

st.set_page_config(page_title="Ticket Sales Forecasting", layout="wide")
st.title("🎭 Ticket Sales Forecasting & Baseline")

# Upload CSV
uploaded_file = st.file_uploader("Upload CSV with columns: date, tickets_sold", type=["csv"])
if uploaded_file is None:
    st.info("Upload your CSV to start")
    st.stop()

df_all = pd.read_csv(uploaded_file, parse_dates=["date"])
if not {"date", "tickets_sold"}.issubset(df_all.columns):
    st.error("CSV must have columns: date, tickets_sold")
    st.stop()

df_all = df_all.sort_values("date").reset_index(drop=True)

# Manage shows in sidebar
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
        show["name"] = st.text_input(f"Name", value=show.get("name", ""), key=f"name_{i}")
        show["start_date"], show["end_date"] = st.date_input(
            f"Date Range",
            value=(show.get("start_date", df_all["date"].min().date()),
                   show.get("end_date", df_all["date"].max().date())),
            min_value=df_all["date"].min().date(),
            max_value=df_all["date"].max().date(),
            key=f"range_{i}"
        )

if len(st.session_state.shows) == 0:
    st.info("Add at least one show in the sidebar")
    st.stop()

# Select show to analyze
show_names = [s["name"] for s in st.session_state.shows if s["name"].strip() != ""]
selected_show = st.selectbox("Select Show", show_names)
if not selected_show:
    st.warning("Please enter a name for at least one show")
    st.stop()

current_show = next(s for s in st.session_state.shows if s["name"] == selected_show)

# Filter data for selected show
df_show = df_all[
    (df_all["date"].dt.date >= current_show["start_date"]) &
    (df_all["date"].dt.date <= current_show["end_date"])
].copy()

if df_show.empty:
    st.warning(f"No data found for show '{selected_show}' in given date range")
    st.stop()

df_show = df_show.sort_values("date")
df_show["days_to_show"] = (pd.to_datetime(current_show["end_date"]) - df_show["date"]).dt.days
df_show["cumulative_sales"] = df_show["tickets_sold"].cumsum()

st.subheader(f"Sales Data for '{selected_show}'")
st.dataframe(df_show[["date", "tickets_sold", "days_to_show", "cumulative_sales"]])

# Build baseline from all shows
all_cum_sales = []
for show in st.session_state.shows:
    if not show["name"]:
        continue
    df_tmp = df_all[
        (df_all["date"].dt.date >= show["start_date"]) &
        (df_all["date"].dt.date <= show["end_date"])
    ].copy()
    if df_tmp.empty:
        continue
    df_tmp = df_tmp.sort_values("date")
    df_tmp["days_to_show"] = (pd.to_datetime(show["end_date"]) - df_tmp["date"]).dt.days
    df_tmp["cumulative_sales"] = df_tmp["tickets_sold"].cumsum()
    df_tmp["show_name"] = show["name"]
    all_cum_sales.append(df_tmp[["show_name", "days_to_show", "cumulative_sales"]])

if not all_cum_sales:
    st.error("No shows with sales data to build baseline")
    st.stop()

df_all_cum = pd.concat(all_cum_sales)

# Calculate median cumulative sales per days_to_show
baseline_df = df_all_cum.groupby("days_to_show")["cumulative_sales"].median().reset_index()
baseline_df = baseline_df.sort_values("days_to_show")

# Fit isotonic regression for smooth baseline
ir = IsotonicRegression(increasing=True, out_of_bounds="clip")
baseline_df["predicted_cumulative_sales"] = ir.fit_transform(baseline_df["days_to_show"], baseline_df["cumulative_sales"])

# Plot selected show + baseline
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_show["days_to_show"],
    y=df_show["cumulative_sales"],
    mode="lines+markers",
    name="Actual Cumulative Sales"
))
fig.add_trace(go.Scatter(
    x=baseline_df["days_to_show"],
    y=baseline_df["predicted_cumulative_sales"],
    mode="lines",
    name="Baseline Median (All Shows)",
    line=dict(dash="dash", color="orange")
))
fig.update_layout(
    title=f"Cumulative Tickets Sold vs Days to Show for '{selected_show}'",
    xaxis_title="Days to Show (0 = Show Day)",
    yaxis_title="Cumulative Tickets Sold",
    yaxis=dict(range=[0, max(df_show["cumulative_sales"].max(), baseline_df["predicted_cumulative_sales"].max()) * 1.1])
)
st.plotly_chart(fig, use_container_width=True)

# Predict baseline for a future show
st.header("Predict Baseline for Future Show")

future_show_date = st.date_input(
    "Future Show Date",
    value=pd.Timestamp.today().date() + pd.Timedelta(days=30),
    min_value=pd.Timestamp.today().date()
)

max_days_before = st.slider("Days Before Show to Forecast", 7, 90, 60)

future_days = np.arange(max_days_before, -1, -1)
future_preds = ir.predict(future_days)

future_pred_df = pd.DataFrame({
    "days_to_show": future_days,
    "predicted_cumulative_sales": future_preds
})

fig_future = go.Figure()
fig_future.add_trace(go.Scatter(
    x=future_pred_df["days_to_show"],
    y=future_pred_df["predicted_cumulative_sales"],
    mode="lines+markers",
    name="Baseline Prediction"
))
fig_future.update_layout(
    title=f"Baseline Predicted Cumulative Sales for Future Show on {future_show_date}",
    xaxis_title="Days to Show (0 = Show Day)",
    yaxis_title="Cumulative Tickets Sold"
)
st.plotly_chart(fig_future, use_container_width=True)

st.dataframe(future_pred_df)
