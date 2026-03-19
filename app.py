import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================================================
# Utility helpers
# =========================================================
def clip(value, low, high):
    return max(low, min(high, value))


def classify_regime(series: np.ndarray) -> str:
    """
    Rough automatic regime classification from the tail of a series.
    """
    tail = np.array(series[len(series) // 2 :], dtype=float)

    if len(tail) < 10:
        return "Too short"

    rounded = np.round(tail, 3)
    unique_count = len(np.unique(rounded))
    std_val = np.std(tail)

    if std_val < 0.01:
        return "Stable"
    if unique_count <= 6:
        return "Cycle-like"
    return "Chaos-like"


# =========================================================
# Consumer population generation
# =========================================================
def generate_consumers(n_consumers: int, seed: int = 42):
    """
    Create heterogeneous consumers:
    - budget
    - regular
    - premium

    Each consumer has:
    - base_wtp
    - shopping frequency
    - loyalty bias toward our bot
    """
    rng = np.random.default_rng(seed)

    segments = rng.choice(
        ["budget", "regular", "premium"],
        size=n_consumers,
        p=[0.35, 0.45, 0.20]
    )

    base_wtp = np.zeros(n_consumers)
    shop_prob = np.zeros(n_consumers)
    loyalty = np.zeros(n_consumers)

    for i, seg in enumerate(segments):
        if seg == "budget":
            base_wtp[i] = clip(rng.normal(0.35, 0.08), 0.05, 0.95)
            shop_prob[i] = clip(rng.normal(0.30, 0.05), 0.05, 0.95)
            loyalty[i] = clip(rng.normal(0.02, 0.03), -0.08, 0.12)

        elif seg == "regular":
            base_wtp[i] = clip(rng.normal(0.55, 0.10), 0.05, 0.95)
            shop_prob[i] = clip(rng.normal(0.45, 0.08), 0.05, 0.95)
            loyalty[i] = clip(rng.normal(0.04, 0.04), -0.10, 0.15)

        else:  # premium
            base_wtp[i] = clip(rng.normal(0.75, 0.10), 0.05, 0.99)
            shop_prob[i] = clip(rng.normal(0.60, 0.08), 0.05, 0.99)
            loyalty[i] = clip(rng.normal(0.06, 0.05), -0.10, 0.20)

    return {
        "segment": segments,
        "base_wtp": base_wtp,
        "shop_prob": shop_prob,
        "loyalty": loyalty
    }


# =========================================================
# Core brief model
# =========================================================
def run_core_brief_model(n_consumers, r, initial_price, n_days, seed=42):
    """
    Exact brief-style model:
    1. Publish Price_t
    2. Consumer buys if Price_t <= WTP_i
    3. Demand_t = buyers / N
    4. Price_{t+1} = r * Price_t * Demand_t
    """
    rng = np.random.default_rng(seed)
    wtp = rng.uniform(0.0, 1.0, n_consumers)

    prices = []
    demands = []
    buyers_hist = []
    revenues = []
    profits = []

    price = initial_price
    unit_cost = 0.20

    for _ in range(n_days):
        buyers = int(np.sum(wtp >= price))
        demand = buyers / n_consumers

        revenue = price * buyers
        profit = (price - unit_cost) * buyers

        prices.append(price)
        demands.append(demand)
        buyers_hist.append(buyers)
        revenues.append(revenue)
        profits.append(profit)

        next_price = r * price * demand
        price = clip(next_price, 0.0, 1.0)

    df = pd.DataFrame({
        "day": np.arange(1, n_days + 1),
        "our_price": prices,
        "our_demand": demands,
        "our_buyers": buyers_hist,
        "our_revenue": revenues,
        "our_profit": profits
    })

    return df


# =========================================================
# Enhanced ABM
# =========================================================
def run_enhanced_abm(
    n_consumers=1000,
    n_days=180,
    our_r=2.0,
    initial_our_price=0.50,
    enable_competitor=True,
    competitor_r=1.6,
    initial_comp_price=0.48,
    target_demand=0.28,
    season_amp=0.10,
    shock_std=0.03,
    unit_cost=0.20,
    seed=42
):
    """
    Enhanced ABM:
    - heterogeneous consumers
    - shopping probability
    - dynamic WTP via seasonality + shocks
    - optional competitor
    - our pricing bot updates based on realized demand
    - competitor can also adapt
    - concrete outputs: revenue, profit, market share, surplus, volatility
    """
    rng = np.random.default_rng(seed)
    consumers = generate_consumers(n_consumers=n_consumers, seed=seed)

    base_wtp = consumers["base_wtp"]
    shop_prob = consumers["shop_prob"]
    loyalty = consumers["loyalty"]

    our_price = initial_our_price
    comp_price = initial_comp_price

    records = []

    for day in range(1, n_days + 1):
        # Seasonality and demand shock
        season_factor = 1.0 + season_amp * np.sin(2 * np.pi * day / 30.0)
        market_shock = float(rng.normal(0.0, shock_std))

        # Which consumers are active today
        active_prob = np.clip(shop_prob * season_factor, 0.01, 0.99)
        is_active = rng.random(n_consumers) < active_prob
        active_count = int(np.sum(is_active))

        # Dynamic WTP for active consumers
        daily_wtp = np.clip(
            base_wtp + rng.normal(0.0, 0.04, n_consumers) + market_shock,
            0.01,
            1.20
        )

        our_buyers = 0
        comp_buyers = 0
        no_buyers = 0

        our_surplus_total = 0.0
        comp_surplus_total = 0.0

        for i in range(n_consumers):
            if not is_active[i]:
                continue

            # Utility from our store vs competitor
            our_utility = daily_wtp[i] - our_price + loyalty[i]
            comp_utility = daily_wtp[i] - comp_price

            if enable_competitor:
                if our_utility > 0 and our_utility >= comp_utility:
                    our_buyers += 1
                    our_surplus_total += our_utility
                elif comp_utility > 0:
                    comp_buyers += 1
                    comp_surplus_total += comp_utility
                else:
                    no_buyers += 1
            else:
                if our_utility > 0:
                    our_buyers += 1
                    our_surplus_total += our_utility
                else:
                    no_buyers += 1

        total_buyers = our_buyers + comp_buyers
        our_demand = our_buyers / n_consumers
        comp_demand = comp_buyers / n_consumers if enable_competitor else 0.0
        our_share = our_buyers / total_buyers if total_buyers > 0 else 0.0

        our_revenue = our_price * our_buyers
        our_profit = (our_price - unit_cost) * our_buyers

        comp_revenue = comp_price * comp_buyers if enable_competitor else 0.0
        comp_profit = (comp_price - unit_cost) * comp_buyers if enable_competitor else 0.0

        avg_our_surplus = our_surplus_total / our_buyers if our_buyers > 0 else 0.0

        records.append({
            "day": day,
            "active_consumers": active_count,
            "our_price": our_price,
            "comp_price": comp_price if enable_competitor else np.nan,
            "our_buyers": our_buyers,
            "comp_buyers": comp_buyers,
            "no_buyers": no_buyers,
            "our_demand": our_demand,
            "comp_demand": comp_demand,
            "our_market_share": our_share,
            "our_revenue": our_revenue,
            "our_profit": our_profit,
            "comp_revenue": comp_revenue,
            "comp_profit": comp_profit,
            "avg_our_surplus": avg_our_surplus,
            "season_factor": season_factor,
            "market_shock": market_shock
        })

        # ---------------------------
        # Price update rules
        # ---------------------------
        # If our observed demand > target demand, we increase price.
        # If observed demand < target demand, we decrease price.
        demand_error = our_demand - target_demand

        if enable_competitor:
            competitor_gap = comp_price - our_price
        else:
            competitor_gap = 0.0

        raw_next_our = our_price * np.exp(
            our_r * demand_error + 0.25 * competitor_gap
        )
        our_price = clip(raw_next_our, 0.05, 1.20)

        if enable_competitor:
            comp_error = comp_demand - target_demand
            raw_next_comp = comp_price * np.exp(
                competitor_r * comp_error + 0.20 * (our_price - comp_price)
            )
            comp_price = clip(raw_next_comp, 0.05, 1.20)

    df = pd.DataFrame(records)
    return df


# =========================================================
# Automated experiments
# =========================================================
def summarize_run(df: pd.DataFrame):
    """
    Summarize one simulation run.
    """
    tail = df.iloc[len(df) // 2 :]

    summary = {
        "avg_price": df["our_price"].mean(),
        "avg_demand": df["our_demand"].mean(),
        "avg_revenue": df["our_revenue"].mean(),
        "avg_profit": df["our_profit"].mean(),
        "avg_market_share": df["our_market_share"].mean() if "our_market_share" in df else np.nan,
        "avg_surplus": df["avg_our_surplus"].mean() if "avg_our_surplus" in df else np.nan,
        "price_volatility": df["our_price"].std(),
        "profit_volatility": df["our_profit"].std(),
        "tail_price_volatility": tail["our_price"].std(),
        "regime": classify_regime(df["our_price"].values)
    }
    return summary


def run_parameter_sweep(
    model_mode,
    r_values,
    seeds,
    n_consumers,
    n_days,
    initial_price,
    enable_competitor,
    competitor_r,
    season_amp,
    shock_std,
    target_demand
):
    """
    Run multiple simulations automatically across many r values and seeds.
    """
    rows = []

    for r in r_values:
        for seed in seeds:
            if model_mode == "Core Brief":
                df = run_core_brief_model(
                    n_consumers=n_consumers,
                    r=r,
                    initial_price=initial_price,
                    n_days=n_days,
                    seed=seed
                )
                summary = {
                    "avg_market_share": np.nan,
                    "avg_surplus": np.nan
                }
            else:
                df = run_enhanced_abm(
                    n_consumers=n_consumers,
                    n_days=n_days,
                    our_r=r,
                    initial_our_price=initial_price,
                    enable_competitor=enable_competitor,
                    competitor_r=competitor_r,
                    target_demand=target_demand,
                    season_amp=season_amp,
                    shock_std=shock_std,
                    seed=seed
                )
                summary = {}

            s = summarize_run(df)
            summary.update(s)

            rows.append({
                "r": r,
                "seed": seed,
                **summary
            })

    results = pd.DataFrame(rows)

    grouped = (
        results
        .groupby("r", as_index=False)
        .agg(
            avg_price=("avg_price", "mean"),
            avg_demand=("avg_demand", "mean"),
            avg_revenue=("avg_revenue", "mean"),
            avg_profit=("avg_profit", "mean"),
            avg_market_share=("avg_market_share", "mean"),
            avg_surplus=("avg_surplus", "mean"),
            price_volatility=("price_volatility", "mean"),
            tail_price_volatility=("tail_price_volatility", "mean")
        )
    )

    # Simple stability score: higher is better
    grouped["stability_score"] = grouped["avg_profit"] - 120 * grouped["tail_price_volatility"]

    return results, grouped


# =========================================================
# Plotting
# =========================================================
def plot_time_series(df: pd.DataFrame, show_competitor: bool):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["day"], df["our_price"], label="Our price")
    if show_competitor and "comp_price" in df.columns:
        ax.plot(df["day"], df["comp_price"], label="Competitor price", alpha=0.85)
    ax.set_title("Price Over Time")
    ax.set_xlabel("Day")
    ax.set_ylabel("Price")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_demand_profit(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(df["day"], df["our_demand"], label="Our demand")
    ax.plot(df["day"], df["our_profit"], label="Our profit")
    ax.set_title("Demand and Profit Over Time")
    ax.set_xlabel("Day")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


def plot_sweep(grouped: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(grouped["r"], grouped["avg_profit"], label="Average profit")
    ax.plot(grouped["r"], grouped["tail_price_volatility"], label="Tail price volatility")
    ax.plot(grouped["r"], grouped["stability_score"], label="Stability score")
    ax.set_title("Automated Sweep Across r")
    ax.set_xlabel("r")
    ax.grid(True, alpha=0.3)
    ax.legend()
    return fig


# =========================================================
# Streamlit app
# =========================================================
st.set_page_config(page_title="Volatility Loop ABM - Enhanced", layout="wide")
st.title("Volatility Loop: Enhanced Agent-Based Market Simulation")

st.markdown(
    """
This version gives you **concrete outputs**, not just a toy plot:
- revenue
- profit
- market share
- consumer surplus
- volatility
- automatic regime classification
- automated parameter sweeps over `r`
"""
)

with st.sidebar:
    st.header("Model mode")
    model_mode = st.radio("Choose model", ["Core Brief", "Enhanced ABM"])

    st.header("Base settings")
    n_consumers = st.slider("Consumers", 200, 5000, 1000, step=100)
    n_days = st.slider("Days", 60, 365, 180, step=10)
    initial_price = st.slider("Initial price", 0.05, 1.00, 0.50, step=0.01)
    our_r = st.slider("Our aggressiveness r", 0.1, 4.0, 2.0, step=0.05)
    seed = st.number_input("Random seed", 0, 999999, 42, 1)

    enable_competitor = False
    competitor_r = 1.6
    season_amp = 0.10
    shock_std = 0.03
    target_demand = 0.28

    if model_mode == "Enhanced ABM":
        st.header("Enhanced market features")
        enable_competitor = st.checkbox("Enable competitor bot", value=True)
        competitor_r = st.slider("Competitor aggressiveness", 0.1, 4.0, 1.6, step=0.05)
        season_amp = st.slider("Seasonality amplitude", 0.00, 0.30, 0.10, step=0.01)
        shock_std = st.slider("Shock standard deviation", 0.00, 0.10, 0.03, step=0.01)
        target_demand = st.slider("Target demand", 0.05, 0.60, 0.28, step=0.01)

    st.header("Automated sweep")
    sweep_min = st.slider("Sweep r min", 0.1, 4.0, 0.5, step=0.1)
    sweep_max = st.slider("Sweep r max", 0.1, 4.0, 3.5, step=0.1)
    sweep_points = st.slider("Number of r values", 5, 40, 20, step=1)
    sweep_runs = st.slider("Seeds per r", 1, 10, 4, step=1)

run_single = st.button("Run single simulation")
run_sweep = st.button("Run automated sweep")

if run_single:
    if model_mode == "Core Brief":
        df = run_core_brief_model(
            n_consumers=n_consumers,
            r=our_r,
            initial_price=initial_price,
            n_days=n_days,
            seed=seed
        )
        show_comp = False
    else:
        df = run_enhanced_abm(
            n_consumers=n_consumers,
            n_days=n_days,
            our_r=our_r,
            initial_our_price=initial_price,
            enable_competitor=enable_competitor,
            competitor_r=competitor_r,
            target_demand=target_demand,
            season_amp=season_amp,
            shock_std=shock_std,
            seed=seed
        )
        show_comp = enable_competitor

    summary = summarize_run(df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Average profit", f"{summary['avg_profit']:.2f}")
    col2.metric("Average revenue", f"{summary['avg_revenue']:.2f}")
    col3.metric("Price volatility", f"{summary['price_volatility']:.4f}")
    col4.metric("Regime", summary["regime"])

    if model_mode == "Enhanced ABM":
        col5, col6, col7 = st.columns(3)
        col5.metric("Average market share", f"{summary['avg_market_share']:.3f}")
        col6.metric("Average consumer surplus", f"{summary['avg_surplus']:.3f}")
        col7.metric("Tail volatility", f"{summary['tail_price_volatility']:.4f}")

    st.pyplot(plot_time_series(df, show_competitor=show_comp))
    st.pyplot(plot_demand_profit(df))

    st.subheader("Simulation table")
    st.dataframe(df, use_container_width=True)

    st.download_button(
        label="Download simulation CSV",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="simulation_output.csv",
        mime="text/csv"
    )

if run_sweep:
    r_values = np.linspace(sweep_min, sweep_max, sweep_points)
    seeds = list(range(100, 100 + sweep_runs))

    raw_results, grouped_results = run_parameter_sweep(
        model_mode=model_mode,
        r_values=r_values,
        seeds=seeds,
        n_consumers=n_consumers,
        n_days=n_days,
        initial_price=initial_price,
        enable_competitor=enable_competitor,
        competitor_r=competitor_r,
        season_amp=season_amp,
        shock_std=shock_std,
        target_demand=target_demand
    )

    best_row = grouped_results.loc[grouped_results["stability_score"].idxmax()]

    st.subheader("Best r from automated sweep")
    st.write(
        f"Best `r` by stability score: **{best_row['r']:.2f}**  |  "
        f"Average profit: **{best_row['avg_profit']:.2f}**  |  "
        f"Tail volatility: **{best_row['tail_price_volatility']:.4f}**"
    )

    st.pyplot(plot_sweep(grouped_results))

    st.subheader("Grouped sweep results")
    st.dataframe(grouped_results, use_container_width=True)

    st.subheader("Raw sweep results")
    st.dataframe(raw_results, use_container_width=True)

    st.download_button(
        label="Download grouped sweep CSV",
        data=grouped_results.to_csv(index=False).encode("utf-8"),
        file_name="grouped_sweep_results.csv",
        mime="text/csv"
    )