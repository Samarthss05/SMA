import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Algorithmic Pricing Profit Optimizer", layout="wide")

st.title("Algorithmic Pricing Simulation + Profit Optimization")
st.markdown(
    r"""
This app simulates a market where:
- a pricing bot publishes a price at time **t**
- consumers buy if **price <= willingness-to-pay**
- demand is observed
- the bot updates price using

\[
P_{t+1} = r \cdot P_t \cdot Demand_t
\]

It also adds **profit analysis** and scans across different values of **r**
to find the setting that gives the **highest long-run profit**.
"""
)

# ----------------------------
# Helpers
# ----------------------------
def generate_wtp(n, distribution="uniform", seed=42):
    rng = np.random.default_rng(seed)

    if distribution == "uniform":
        wtp = rng.uniform(0.0, 1.0, n)
    elif distribution == "normal":
        wtp = rng.normal(loc=0.55, scale=0.18, size=n)
        wtp = np.clip(wtp, 0.0, 1.0)
    elif distribution == "beta":
        wtp = rng.beta(a=2.0, b=2.5, size=n)
    else:
        wtp = rng.uniform(0.0, 1.0, n)

    return wtp


def seasonal_multiplier(t, amplitude=0.0, period=30):
    return 1.0 + amplitude * np.sin(2 * np.pi * t / period)


def run_simulation(
    r=2.0,
    n_consumers=1000,
    steps=200,
    initial_price=0.5,
    distribution="uniform",
    seed=42,
    dynamic_wtp=False,
    wtp_noise=0.02,
    use_seasonality=False,
    season_amplitude=0.0,
    season_period=30,
    clip_price=True,
    unit_cost=0.20,
    fixed_cost_per_step=0.0,
):
    rng = np.random.default_rng(seed)
    base_wtp = generate_wtp(n_consumers, distribution=distribution, seed=seed)
    current_wtp = base_wtp.copy()

    prices = np.zeros(steps + 1)
    demands = np.zeros(steps)
    buyers = np.zeros(steps, dtype=int)
    revenue = np.zeros(steps)
    cost = np.zeros(steps)
    profit = np.zeros(steps)

    prices[0] = initial_price

    for t in range(steps):
        current_price = prices[t]

        if use_seasonality:
            effective_price = current_price / seasonal_multiplier(
                t, amplitude=season_amplitude, period=season_period
            )
        else:
            effective_price = current_price

        buy_mask = effective_price <= current_wtp
        buyers[t] = buy_mask.sum()
        demands[t] = buyers[t] / n_consumers

        units_sold = buyers[t]
        revenue[t] = current_price * units_sold
        cost[t] = unit_cost * units_sold + fixed_cost_per_step
        profit[t] = revenue[t] - cost[t]

        next_price = r * current_price * demands[t]

        if clip_price:
            next_price = np.clip(next_price, 0.0, 1.5)

        prices[t + 1] = next_price

        if dynamic_wtp:
            noise = rng.normal(0.0, wtp_noise, size=n_consumers)
            current_wtp = np.clip(current_wtp + noise, 0.0, 1.0)

    cumulative_profit = np.cumsum(profit)

    return {
        "prices": prices,
        "demands": demands,
        "buyers": buyers,
        "revenue": revenue,
        "cost": cost,
        "profit": profit,
        "cumulative_profit": cumulative_profit,
        "base_wtp": base_wtp,
    }


def classify_regime(prices, transient_fraction=0.5):
    start = int(len(prices) * transient_fraction)
    tail = prices[start:]

    if len(tail) < 10:
        return "Insufficient data"

    tail_std = np.std(tail)
    rounded_unique = len(np.unique(np.round(tail, 3)))

    if tail_std < 0.005:
        return "Stable / Convergent"
    elif rounded_unique <= 8:
        return "Cyclical / Periodic"
    else:
        return "Chaos-like / Irregular"


def compute_bifurcation(
    r_values,
    n_consumers=1000,
    steps=300,
    keep_last=80,
    initial_price=0.5,
    distribution="uniform",
    seed=42,
    dynamic_wtp=False,
    wtp_noise=0.02,
    use_seasonality=False,
    season_amplitude=0.0,
    season_period=30,
    unit_cost=0.20,
    fixed_cost_per_step=0.0,
):
    xs = []
    ys = []

    for r in r_values:
        result = run_simulation(
            r=r,
            n_consumers=n_consumers,
            steps=steps,
            initial_price=initial_price,
            distribution=distribution,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )
        tail = result["prices"][-keep_last:]
        xs.extend([r] * len(tail))
        ys.extend(tail.tolist())

    return np.array(xs), np.array(ys)


def optimize_r(
    r_min,
    r_max,
    num_r,
    n_consumers,
    steps,
    initial_price,
    distribution,
    seed,
    dynamic_wtp,
    wtp_noise,
    use_seasonality,
    season_amplitude,
    season_period,
    unit_cost,
    fixed_cost_per_step,
):
    r_values = np.linspace(r_min, r_max, num_r)
    rows = []

    for r in r_values:
        result = run_simulation(
            r=float(r),
            n_consumers=n_consumers,
            steps=steps,
            initial_price=initial_price,
            distribution=distribution,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )

        tail_start = max(1, int(steps * 0.5))
        avg_profit = float(np.mean(result["profit"][tail_start:]))
        total_profit = float(np.sum(result["profit"]))
        volatility = float(np.std(result["prices"][tail_start:]))
        avg_price = float(np.mean(result["prices"][tail_start:]))
        avg_demand = float(np.mean(result["demands"][tail_start:]))
        regime = classify_regime(result["prices"])

        rows.append(
            {
                "r": float(r),
                "avg_profit_long_run": avg_profit,
                "total_profit": total_profit,
                "price_volatility": volatility,
                "avg_price": avg_price,
                "avg_demand": avg_demand,
                "regime": regime,
            }
        )

    rows = sorted(rows, key=lambda x: x["avg_profit_long_run"], reverse=True)
    return rows


# ----------------------------
# Sidebar controls
# ----------------------------
st.sidebar.header("Simulation Controls")

r = st.sidebar.slider("Aggression parameter r", min_value=0.0, max_value=4.0, value=2.0, step=0.01)
n_consumers = st.sidebar.slider("Number of consumers", min_value=100, max_value=5000, value=1000, step=100)
steps = st.sidebar.slider("Number of time steps", min_value=50, max_value=1000, value=250, step=10)
initial_price = st.sidebar.slider("Initial price P₀", min_value=0.01, max_value=1.0, value=0.5, step=0.01)
distribution = st.sidebar.selectbox("WTP distribution", options=["uniform", "normal", "beta"])
seed = st.sidebar.number_input("Random seed", min_value=0, max_value=999999, value=42, step=1)

st.sidebar.subheader("Profit Assumptions")
unit_cost = st.sidebar.slider("Unit cost", min_value=0.0, max_value=1.0, value=0.20, step=0.01)
fixed_cost_per_step = st.sidebar.slider("Fixed cost per step", min_value=0.0, max_value=500.0, value=0.0, step=5.0)

st.sidebar.subheader("Optional extensions")
dynamic_wtp = st.sidebar.checkbox("Dynamic WTP over time", value=False)
wtp_noise = st.sidebar.slider("WTP noise level", min_value=0.0, max_value=0.2, value=0.02, step=0.005)

use_seasonality = st.sidebar.checkbox("Use seasonality", value=False)
season_amplitude = st.sidebar.slider("Seasonality amplitude", min_value=0.0, max_value=0.5, value=0.15, step=0.01)
season_period = st.sidebar.slider("Seasonality period", min_value=5, max_value=120, value=30, step=1)

run_button = st.sidebar.button("Run simulation", type="primary")
show_bifurcation = st.sidebar.checkbox("Show bifurcation plot", value=True)

st.sidebar.subheader("Optimize r for Profit")
opt_r_min = st.sidebar.slider("Optimization r min", min_value=0.0, max_value=4.0, value=0.1, step=0.1)
opt_r_max = st.sidebar.slider("Optimization r max", min_value=0.1, max_value=4.0, value=4.0, step=0.1)
opt_num_r = st.sidebar.slider("Optimization grid size", min_value=10, max_value=200, value=60, step=5)
run_optimization = st.sidebar.button("Find best r")

# ----------------------------
# Run simulation
# ----------------------------
if run_button or "already_ran" not in st.session_state:
    st.session_state["already_ran"] = True
    result = run_simulation(
        r=r,
        n_consumers=n_consumers,
        steps=steps,
        initial_price=initial_price,
        distribution=distribution,
        seed=seed,
        dynamic_wtp=dynamic_wtp,
        wtp_noise=wtp_noise,
        use_seasonality=use_seasonality,
        season_amplitude=season_amplitude,
        season_period=season_period,
        unit_cost=unit_cost,
        fixed_cost_per_step=fixed_cost_per_step,
    )
    st.session_state["result"] = result

result = st.session_state["result"]

prices = result["prices"]
demands = result["demands"]
buyers = result["buyers"]
profit = result["profit"]
cumulative_profit = result["cumulative_profit"]

tail_start_price = int(len(prices) * 0.5)
tail_start_profit = int(len(profit) * 0.5)

price_volatility = float(np.std(prices[tail_start_price:]))
demand_volatility = float(np.std(demands[tail_start_profit:]))
avg_price = float(np.mean(prices[tail_start_price:]))
avg_demand = float(np.mean(demands[tail_start_profit:]))
avg_profit = float(np.mean(profit[tail_start_profit:]))
total_profit = float(np.sum(profit))
regime = classify_regime(prices)

# ----------------------------
# KPIs
# ----------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Regime", regime)
col2.metric("Price volatility", f"{price_volatility:.4f}")
col3.metric("Avg long-run price", f"{avg_price:.4f}")
col4.metric("Avg long-run demand", f"{avg_demand:.4f}")
col5.metric("Avg long-run profit", f"{avg_profit:.2f}")
col6.metric("Total profit", f"{total_profit:.2f}")

# ----------------------------
# Charts
# ----------------------------
row1_col1, row1_col2 = st.columns(2)

with row1_col1:
    fig1, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(prices, linewidth=1.5)
    ax1.set_title("Price Time Series")
    ax1.set_xlabel("Time step")
    ax1.set_ylabel("Price")
    ax1.grid(True, alpha=0.3)
    st.pyplot(fig1)

with row1_col2:
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(range(len(demands)), demands, linewidth=1.5)
    ax2.set_title("Demand Time Series")
    ax2.set_xlabel("Time step")
    ax2.set_ylabel("Demand")
    ax2.grid(True, alpha=0.3)
    st.pyplot(fig2)

row2_col1, row2_col2 = st.columns(2)

with row2_col1:
    fig3, ax3 = plt.subplots(figsize=(8, 4.5))
    ax3.plot(range(len(profit)), profit, linewidth=1.5)
    ax3.set_title("Profit Per Time Step")
    ax3.set_xlabel("Time step")
    ax3.set_ylabel("Profit")
    ax3.grid(True, alpha=0.3)
    st.pyplot(fig3)

with row2_col2:
    fig4, ax4 = plt.subplots(figsize=(8, 4.5))
    ax4.plot(range(len(cumulative_profit)), cumulative_profit, linewidth=1.5)
    ax4.set_title("Cumulative Profit")
    ax4.set_xlabel("Time step")
    ax4.set_ylabel("Cumulative profit")
    ax4.grid(True, alpha=0.3)
    st.pyplot(fig4)

row3_col1, row3_col2 = st.columns(2)

with row3_col1:
    fig5, ax5 = plt.subplots(figsize=(8, 4.5))
    ax5.hist(result["base_wtp"], bins=30)
    ax5.set_title("Consumer WTP Distribution")
    ax5.set_xlabel("WTP")
    ax5.set_ylabel("Number of consumers")
    ax5.grid(True, alpha=0.3)
    st.pyplot(fig5)

with row3_col2:
    fig6, ax6 = plt.subplots(figsize=(8, 4.5))
    ax6.scatter(prices[:-1], profit, s=10, alpha=0.7)
    ax6.set_title("Profit vs Published Price")
    ax6.set_xlabel("Published price")
    ax6.set_ylabel("Profit")
    ax6.grid(True, alpha=0.3)
    st.pyplot(fig6)

# ----------------------------
# Bifurcation plot
# ----------------------------
if show_bifurcation:
    st.subheader("Bifurcation-Style Summary Plot")

    bif_col1, bif_col2, bif_col3 = st.columns(3)
    with bif_col1:
        r_min = st.slider("r min", min_value=0.0, max_value=4.0, value=0.0, step=0.1, key="rmin")
    with bif_col2:
        r_max = st.slider("r max", min_value=0.1, max_value=4.0, value=4.0, step=0.1, key="rmax")
    with bif_col3:
        num_r = st.slider("Number of r values", min_value=20, max_value=300, value=120, step=10, key="numr")

    if r_max <= r_min:
        st.warning("Please keep r max greater than r min.")
    else:
        r_values = np.linspace(r_min, r_max, num_r)
        xs, ys = compute_bifurcation(
            r_values=r_values,
            n_consumers=n_consumers,
            steps=max(steps, 250),
            keep_last=min(80, max(30, steps // 3)),
            initial_price=initial_price,
            distribution=distribution,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )

        fig7, ax7 = plt.subplots(figsize=(10, 5))
        ax7.scatter(xs, ys, s=1, alpha=0.5)
        ax7.set_title("Bifurcation-Style Plot: Long-Run Price vs r")
        ax7.set_xlabel("Aggression parameter r")
        ax7.set_ylabel("Long-run price values")
        ax7.grid(True, alpha=0.3)
        st.pyplot(fig7)

# ----------------------------
# Optimization
# ----------------------------
st.subheader("Profit Optimization Over r")

if run_optimization:
    if opt_r_max <= opt_r_min:
        st.error("Optimization r max must be greater than r min.")
    else:
        ranking = optimize_r(
            r_min=opt_r_min,
            r_max=opt_r_max,
            num_r=opt_num_r,
            n_consumers=n_consumers,
            steps=steps,
            initial_price=initial_price,
            distribution=distribution,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )

        best = ranking[0]
        st.success(
            f"Best r found: {best['r']:.4f} | "
            f"Avg long-run profit: {best['avg_profit_long_run']:.2f} | "
            f"Total profit: {best['total_profit']:.2f} | "
            f"Regime: {best['regime']}"
        )

        top_n = min(10, len(ranking))
        top_rows = ranking[:top_n]

        st.markdown("### Top r values by long-run average profit")
        st.dataframe(top_rows, use_container_width=True)

        # Plot profit vs r
        profit_curve_x = [row["r"] for row in ranking]
        profit_curve_y = [row["avg_profit_long_run"] for row in ranking]
        profit_curve_vol = [row["price_volatility"] for row in ranking]

        fig8, ax8 = plt.subplots(figsize=(10, 5))
        ax8.plot(profit_curve_x, profit_curve_y, linewidth=1.5)
        ax8.set_title("Average Long-Run Profit vs r")
        ax8.set_xlabel("r")
        ax8.set_ylabel("Average long-run profit")
        ax8.grid(True, alpha=0.3)
        st.pyplot(fig8)

        fig9, ax9 = plt.subplots(figsize=(10, 5))
        ax9.plot(profit_curve_x, profit_curve_vol, linewidth=1.5)
        ax9.set_title("Price Volatility vs r")
        ax9.set_xlabel("r")
        ax9.set_ylabel("Price volatility")
        ax9.grid(True, alpha=0.3)
        st.pyplot(fig9)

        best_run = run_simulation(
            r=best["r"],
            n_consumers=n_consumers,
            steps=steps,
            initial_price=initial_price,
            distribution=distribution,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )

        st.markdown("### Best-r simulation output")
        best_col1, best_col2 = st.columns(2)

        with best_col1:
            fig10, ax10 = plt.subplots(figsize=(8, 4.5))
            ax10.plot(best_run["prices"], linewidth=1.5)
            ax10.set_title(f"Price Time Series at Best r = {best['r']:.4f}")
            ax10.set_xlabel("Time step")
            ax10.set_ylabel("Price")
            ax10.grid(True, alpha=0.3)
            st.pyplot(fig10)

        with best_col2:
            fig11, ax11 = plt.subplots(figsize=(8, 4.5))
            ax11.plot(best_run["cumulative_profit"], linewidth=1.5)
            ax11.set_title(f"Cumulative Profit at Best r = {best['r']:.4f}")
            ax11.set_xlabel("Time step")
            ax11.set_ylabel("Cumulative profit")
            ax11.grid(True, alpha=0.3)
            st.pyplot(fig11)

# ----------------------------
# Interpretation
# ----------------------------
st.subheader("Interpretation")
st.markdown(
    f"""
- With **r = {r:.2f}**, the current simulation is classified as **{regime}**.
- The long-run **average profit** is **{avg_profit:.2f}** and the **total profit** is **{total_profit:.2f}**.
- The model assumes **profit = price × units sold − unit cost × units sold − fixed cost per step**.
- This lets you compare not only stability and volatility, but also whether a given pricing aggressiveness actually improves profitability.
"""
)

st.subheader("How to run locally")
st.code(
    """pip install streamlit numpy matplotlib
streamlit run pricing_profit_optimizer.py""",
    language="bash",
)
