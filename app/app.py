"""Streamlit app: forecast + recommended order for a store/item. Reads the precomputed files in artifacts/.
Run with: streamlit run app.py"""
import json
import math
from pathlib import Path
from statistics import NormalDist

import altair as alt
import pandas as pd
import streamlit as st

ARTIFACTS = Path(__file__).parent / "artifacts"
KEYS = ["store", "item"]

st.set_page_config(page_title="Demand Forecast & Inventory Support", layout="wide")


# ---- load data ----
@st.cache_data
def load_artifacts():
    forecast = pd.read_csv(ARTIFACTS / "forecast.csv", parse_dates=["date"])
    history = pd.read_csv(ARTIFACTS / "history_last365.csv.gz", parse_dates=["date"])
    calib = pd.read_csv(ARTIFACTS / "calibration.csv").set_index(KEYS)
    metrics = json.loads((ARTIFACTS / "metrics.json").read_text())
    assumptions = json.loads((ARTIFACTS / "assumptions.json").read_text())
    return forecast, history, calib, metrics, assumptions


forecast_all, history_all, calib, metrics, defaults = load_artifacts()


# ---- calculations ----
def interval_bounds(fc: pd.DataFrame, sigma_1: float, level: float) -> pd.DataFrame:
    """forecast +/- z * sigma_1 for a single day (sigma_1 = std of out-of-sample errors)"""
    z = NormalDist().inv_cdf(0.5 + level / 2)
    out = fc.copy()
    out["lower"] = (out["forecast"] - z * sigma_1).clip(lower=0)  # can't be negative
    out["upper"] = out["forecast"] + z * sigma_1
    return out


def recommend_order(fc_values, i0, lead, review, z, sigma_lr, sigma_l, on_hand, on_order, pack):
    # order-up-to policy
    # S = forecast over lead + review days + z * sigma(lead + review)
    # order = max(0, S - (on hand + on order)), rounded up to the pack size
    horizon = lead + review
    demand_lr = float(fc_values[i0:i0 + horizon].sum())
    demand_l = float(fc_values[i0:i0 + lead].sum())
    safety = z * sigma_lr
    s_level = demand_lr + safety
    position = on_hand + on_order
    raw = max(0.0, s_level - position)
    qty = math.ceil(raw / pack) * pack if raw > 0 else 0
    return {"demand_lr": demand_lr, "safety_stock": safety, "order_up_to": s_level,
            "position": position, "raw": raw, "qty": int(qty), "reorder_point": demand_l + z * sigma_l}


# ---- sidebar ----
st.sidebar.header("1. Pick a series")
stores = sorted(forecast_all["store"].unique())
items = sorted(forecast_all["item"].unique())
store = st.sidebar.selectbox("Store", stores, index=0)
item = st.sidebar.selectbox("Item", items, index=0)

st.sidebar.header("2. Forecast interval")
level_label = st.sidebar.radio("Prediction interval", ["80%", "90%", "95%"], index=0, horizontal=True)
level = int(level_label.strip("%")) / 100

st.sidebar.header("3. Inventory policy")
lead = st.sidebar.slider("Lead time (days)", 1, 14, int(defaults["lead_time_days"]))
review = st.sidebar.slider("Review period (days)", 1, 14, int(defaults["review_period_days"]))
service = st.sidebar.slider("Target service level", 0.80, 0.995, float(defaults["service_level"]), 0.005,
                            help="Probability of no stockout during a replenishment cycle.")

st.sidebar.header("4. Current stock")
on_hand = st.sidebar.number_input("Units on hand", min_value=0.0, value=100.0, step=10.0)
on_order = st.sidebar.number_input("Units already on order", min_value=0.0, value=0.0, step=10.0)
pack = st.sidebar.number_input("Pack size (order multiple)", min_value=1, value=1, step=1)

with st.sidebar.expander("Cost assumptions (illustrative)"):
    hold_cost = st.number_input("Holding cost, $ per unit per day", min_value=0.001,
                                value=float(defaults["unit_holding_cost_per_day"]), step=0.01, format="%.3f")
    stock_cost = st.number_input("Stockout cost, $ per unit", min_value=0.01,
                                 value=float(defaults["unit_stockout_cost"]), step=0.5)
suggested = stock_cost / (stock_cost + hold_cost * review)

# ---- data for the selected series ----
fc = (forecast_all[(forecast_all["store"] == store) & (forecast_all["item"] == item)]
      .sort_values("date").reset_index(drop=True))
hist = history_all[(history_all["store"] == store) & (history_all["item"] == item)].sort_values("date")
cal = calib.loc[(store, item)]
fc = interval_bounds(fc, float(cal["sigma_1"]), level)

# same weekday last year (364 days back), the seasonal naive line
fc["last_year"] = hist.set_index("date")["sales"].reindex(fc["date"] - pd.Timedelta(days=364)).to_numpy()

n_days = len(fc)
max_i0 = n_days - (lead + review)  # protection interval has to fit inside the forecast window
order_dates = list(fc["date"].dt.date[: max_i0 + 1])

# ---- page ----
st.title("Retail Demand Forecast & Inventory Decision Support")
st.info("**Demonstration only.** The dataset has no prices, costs, stock levels or lead times, so the policy "
        "inputs and cost figures are illustrative assumptions you can change. Forecasts are 90 days ahead.")

order_date = st.sidebar.selectbox("Order date (review day)", order_dates, index=0,
                                  help="The protection interval starts here.")
i0 = order_dates.index(order_date)

z = NormalDist().inv_cdf(service)
rec = recommend_order(fc["forecast"].to_numpy(), i0, lead, review, z,
                      float(cal[f"sigma_{lead + review}"]), float(cal[f"sigma_{lead}"]),
                      on_hand, on_order, int(pack))

st.subheader(f"Store {store} · Item {item}")
c1, c2, c3, c4 = st.columns(4)
c1.metric(f"Forecast demand, next {lead + review} days", f"{rec['demand_lr']:.0f} units")
c2.metric("Safety stock", f"{rec['safety_stock']:.0f} units")
c3.metric("Order-up-to level", f"{rec['order_up_to']:.0f} units")
c4.metric("Recommended order", f"{rec['qty']} units")

if rec["qty"] == 0:
    st.success(f"No order needed: inventory position ({rec['position']:.0f}) already covers the "
               f"order-up-to level ({rec['order_up_to']:.0f}).")
else:
    st.warning(f"Order **{rec['qty']} units** on {order_date}: order-up-to level {rec['order_up_to']:.0f} "
               f"minus inventory position {rec['position']:.0f}.")
st.caption(f"Reorder point (continuous-review view): {rec['reorder_point']:.0f} units. "
           f"With your cost inputs, the cost-optimal service level would be about {suggested:.1%} "
           f"(s / (s + h·R)); you selected {service:.1%}.")

# ---- chart ----
recent = hist.tail(120)
lines = pd.concat([
    recent.assign(series="Actual sales (history)").rename(columns={"sales": "value"})[["date", "value", "series"]],
    fc.assign(series="Forecast").rename(columns={"forecast": "value"})[["date", "value", "series"]],
    fc.assign(series="Same weekday last year").rename(columns={"last_year": "value"})[["date", "value", "series"]],
])
color = alt.Color("series:N", scale=alt.Scale(
    domain=["Actual sales (history)", "Forecast", "Same weekday last year"],
    range=["#4c78a8", "#e45756", "#9d9d9d"]), legend=alt.Legend(title=None, orient="bottom"))
base = alt.Chart(lines).encode(x=alt.X("date:T", title=None))
line_layer = base.mark_line().encode(
    y=alt.Y("value:Q", title="Units per day"), color=color,
    strokeDash=alt.condition(alt.datum.series == "Same weekday last year", alt.value([4, 3]), alt.value([1, 0])))
band = alt.Chart(fc).mark_area(opacity=0.2, color="#e45756").encode(
    x="date:T", y=alt.Y("lower:Q"), y2="upper:Q")
window = pd.DataFrame({"start": [fc["date"].iloc[i0]],
                       "end": [fc["date"].iloc[i0 + lead + review - 1] + pd.Timedelta(days=1)]})
shade = alt.Chart(window).mark_rect(opacity=0.12, color="#54a24b").encode(x="start:T", x2="end:T")
st.altair_chart((shade + band + line_layer).properties(
    height=380, title=f"Daily sales and {level_label} prediction interval; green = protection interval "
                      f"({lead + review} days from order date)").interactive(), width="stretch")

# ---- details ----
with st.expander("How this recommendation is calculated"):
    st.markdown(
        f"""
1. **Forecast** daily demand with a LightGBM model (predicts sales relative to a recent level), trained on 2013-2017.
2. **Interval:** forecast ± z × σ, where σ = {cal['sigma_1']:.2f} units is the std of this series' *out-of-sample* daily errors.
3. **Protection interval** = lead time + review period = **{lead + review} days**. Forecast demand over it: **{rec['demand_lr']:.0f}**.
4. **Safety stock** = z × σ({lead + review}-day error) = {z:.2f} × {float(cal[f'sigma_{lead + review}']):.1f} = **{rec['safety_stock']:.0f}** units.
   σ is measured on *summed* multi-day errors, not scaled from the daily value.
5. **Order-up-to level** = demand + safety stock = **{rec['order_up_to']:.0f}**.
6. **Order** = max(0, level − (on hand + on order)) = max(0, {rec['order_up_to']:.0f} − {rec['position']:.0f}),
   rounded up to the pack size = **{rec['qty']}** units.
        """)

with st.expander("Forecast table"):
    table = fc[["date", "forecast", "lower", "upper", "last_year"]].copy()
    table[["forecast", "lower", "upper", "last_year"]] = table[["forecast", "lower", "upper", "last_year"]].round(1)
    st.dataframe(table, width="stretch", hide_index=True)
    st.download_button("Download CSV", table.to_csv(index=False), file_name=f"forecast_store{store}_item{item}.csv")

with st.expander("Model accuracy and limitations"):
    m, b = metrics["lgbm_ratio"], metrics["seasonal_naive_364"]
    st.markdown(f"**Hold-out ({metrics['holdout_period']}, all 500 series, never used for model selection):**")
    st.table(pd.DataFrame({"LightGBM": m, "Seasonal naive (364d)": b}).loc[["MAE", "RMSE", "sMAPE", "Bias"]].round(2))
    cov = metrics.get("interval_coverage")
    if cov:
        st.markdown("**Interval check on the hold-out (nominal vs. observed coverage):** " +
                    ", ".join(f"{k}: {v:.1%}" for k, v in cov.items()))
    st.markdown("""
**Limitations.** No real cost, price, stock or lead-time data (all inputs are assumptions). Sales are treated as demand,
so any past stockouts understate true demand. Forecasts are 90 days ahead; a live system would use fresher data.
The dataset behaves like clean, count-noise data, so accuracy here probably overstates what real retail data would give.
Safety stock assumes roughly normal errors and one lead time per series.
        """)
