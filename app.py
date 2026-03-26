"""
Volatility Loop: Nonlinear Dynamics in Algorithmic Pricing
Full research-grade implementation — Tiers 1–6.

Tier 1 : model fidelity (seasonality fix, OU dynamic WTP, reference-price anchoring,
          error-correction update rule)
Tier 2 : agent heterogeneity (3-segment consumers, churn & re-entry, 2-bot competition)
Tier 3 : rigorous nonlinear analysis (Lyapunov exponent, return map, Feigenbaum ratio)
Tier 4 : adaptive pricing (Q-learning bot, MPC with smoothness penalty)
Tier 5 : controllability & welfare (OGY chaos control, consumer surplus, Gini, price CV)
"""

import json
from textwrap import dedent

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd

try:
    from scipy.optimize import minimize, brentq
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

# ── matplotlib style ─────────────────────────────────────────
plt.rcParams.update({
    "axes.grid": True, "grid.alpha": 0.25,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110,
})

def apply_custom_ui():
    st.markdown(
        dedent(
            """
            <style>
                .stApp {
                    background: #000000;
                }

                .block-container {
                    padding-top: 1.2rem;
                    padding-bottom: 2rem;
                    max-width: 1400px;
                }

                [data-testid="stSidebar"] {
                    background: #000000;
                    border-right: 1px solid rgba(255,255,255,0.08);
                }

                h1, h2, h3 {
                    letter-spacing: -0.02em;
                }

                .hero-card {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.10);
                    border-radius: 18px;
                    padding: 1.1rem 1.2rem;
                    margin: 0.5rem 0 1rem 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.18);
                }

                .section-card {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 16px;
                    padding: 0.9rem 1rem;
                    margin-bottom: 0.8rem;
                }

                .small-note {
                    color: rgba(255,255,255,0.72);
                    font-size: 0.93rem;
                    line-height: 1.45;
                }

                div[data-testid="stMetric"] {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 14px;
                    padding: 0.65rem 0.8rem;
                    min-height: 132px;
                }

                div[data-testid="stMetricLabel"] {
                    color: rgba(255,255,255,0.75);
                }

                div[data-testid="stMetricLabel"] > div {
                    font-size: 0.95rem;
                    line-height: 1.25rem;
                    white-space: normal !important;
                    word-break: break-word;
                }

                div[data-testid="stMetricValue"] {
                    font-size: 2.15rem;
                    line-height: 1.05;
                }

                .stTabs [data-baseweb="tab-list"] {
                    gap: 0.4rem;
                    flex-wrap: wrap;
                }

                .stTabs [data-baseweb="tab"] {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 12px;
                    padding: 0.45rem 0.8rem;
                }

                .stTabs [aria-selected="true"] {
                    background: rgba(255,255,255,0.08) !important;
                    border-color: rgba(255,255,255,0.20) !important;
                }

                .stButton > button {
                    border-radius: 12px;
                    font-weight: 600;
                }

                .stDataFrame, div[data-testid="stTable"] {
                    border-radius: 14px;
                    overflow: hidden;
                }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

# ════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Volatility Loop — Nonlinear Pricing Dynamics",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_custom_ui()
st.title("Volatility Loop: Nonlinear Dynamics in Algorithmic Pricing")
st.markdown(
    """
    <div class="hero-card">
        <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.35rem;">
            Research sandbox for dynamic pricing, nonlinear instability, and control.
        </div>
        <div class="small-note">
            Explore how pricing aggressiveness, consumer behavior, competition, and control rules affect
            stability, profit, welfare, bifurcations, and chaos.
            <br><br>
            <strong>Includes:</strong> Lyapunov exponent · Feigenbaum ratio · segmented consumers · churn &amp; re-entry ·
            two-bot competition · Q-learning · MPC · OGY chaos control · consumer welfare &amp; Gini.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ════════════════════════════════════════════════════════════════
# CONSUMER GENERATION
# ════════════════════════════════════════════════════════════════

def generate_wtp_simple(n, distribution="uniform", seed=42):
    rng = np.random.default_rng(seed)
    if distribution == "uniform":
        return rng.uniform(0.0, 1.0, n)
    elif distribution == "normal":
        return np.clip(rng.normal(0.55, 0.18, n), 0.0, 1.0)
    elif distribution == "beta":
        return rng.beta(2.0, 2.5, n)
    return rng.uniform(0.0, 1.0, n)


def generate_segmented_wtp(n, seed=42):
    """Three-segment market: price-sensitive (50 %), mainstream (35 %), loyal (15 %)."""
    rng = np.random.default_rng(seed)
    n_s = int(0.50 * n)
    n_m = int(0.35 * n)
    n_l = n - n_s - n_m
    wtp = np.concatenate([
        rng.beta(1.5, 4.0, n_s),
        np.clip(rng.normal(0.55, 0.15, n_m), 0.0, 1.0),
        np.clip(rng.normal(0.80, 0.08, n_l), 0.0, 1.0),
    ])
    segs = np.array(
        ["price_sensitive"] * n_s + ["mainstream"] * n_m + ["loyal"] * n_l,
        dtype=object,
    )
    return wtp, segs


def seasonal_multiplier(t, amplitude=0.0, period=30):
    return 1.0 + amplitude * np.sin(2 * np.pi * t / period)


# ════════════════════════════════════════════════════════════════
# MAIN ABM SIMULATION  (Tiers 1 + 2)
# ════════════════════════════════════════════════════════════════

def run_simulation(
    r=2.5, n_consumers=800, steps=300, initial_price=0.5,
    distribution="uniform", seed=42,
    # Tier 1 – OU dynamic WTP
    dynamic_wtp=False, wtp_noise=0.02, ou_theta=0.10,
    # Tier 1 – Seasonality (FIXED: scales WTP, not price)
    use_seasonality=False, season_amplitude=0.2, season_period=30,
    # Tier 1 – Reference-price anchoring
    use_reference_price=False, ref_alpha=0.7, ref_beta=0.15,
    # Tier 1 – Update rule
    update_rule="original", d_target=0.5,
    # Tier 2 – Segmented consumers
    use_segments=False,
    # Tier 2 – Churn & re-entry
    use_churn=False, churn_patience=3, reentry_rate=0.005,
    # Cost structure
    unit_cost=0.20, fixed_cost_per_step=0.0,
    clip_price=True,
):
    rng = np.random.default_rng(seed)

    if use_segments:
        base_wtp, segments = generate_segmented_wtp(n_consumers, seed=seed)
    else:
        base_wtp = generate_wtp_simple(n_consumers, distribution, seed=seed)
        segments = np.array(["all"] * n_consumers, dtype=object)

    current_wtp   = base_wtp.copy()
    active        = np.ones(n_consumers, dtype=bool)
    consec_above  = np.zeros(n_consumers, dtype=int)

    prices     = np.zeros(steps + 1)
    demands    = np.zeros(steps)
    buyers_arr = np.zeros(steps, dtype=int)
    revenue    = np.zeros(steps)
    cost_arr   = np.zeros(steps)
    profit     = np.zeros(steps)
    welfare_cs = np.zeros(steps)
    lyap_terms = []
    frames = []

    prices[0] = initial_price
    ref_price = initial_price

    for t in range(steps):
        p = prices[t]

        # Tier 1 FIX: scale WTP by season (demand rises in peak), not divide price
        if use_seasonality:
            eff_wtp = np.clip(
                current_wtp * seasonal_multiplier(t, season_amplitude, season_period),
                0.0, 1.0,
            )
        else:
            eff_wtp = current_wtp

        # Reference-price anchoring: consumers lower WTP when price < reference
        if use_reference_price:
            eff_wtp = np.clip(
                eff_wtp - ref_beta * np.maximum(0.0, ref_price - p),
                0.0, 1.0,
            )

        # Purchase decisions (only active consumers)
        buy_mask  = active & (p <= eff_wtp)
        n_buyers  = int(buy_mask.sum())
        buyers_arr[t] = n_buyers
        demands[t]    = n_buyers / n_consumers   # normalised by total N for stability

        # Financials
        revenue[t]  = p * n_buyers
        cost_arr[t] = unit_cost * n_buyers + fixed_cost_per_step
        profit[t]   = revenue[t] - cost_arr[t]

        # Consumer surplus
        welfare_cs[t] = float(np.sum(np.maximum(0.0, eff_wtp[buy_mask] - p)))

        # Frames for JS live visualisation
        viz_n = min(240, n_consumers)
        frames.append({
            "step": int(t),
            "price": float(p),
            "demand": float(demands[t]),
            "profit": float(profit[t]),
            "buyers": buy_mask[:viz_n].astype(int).tolist(),
            "active": active[:viz_n].astype(int).tolist(),
            "segments": segments[:viz_n].tolist(),
        })

        # Tier 3: Lyapunov term via numerical derivative of the map
        dp_eps  = 1e-4
        d_fwd   = float(np.mean(active & (eff_wtp >= p + dp_eps)))
        d_prime = (d_fwd - demands[t]) / dp_eps      # ∂D/∂P ≤ 0
        d_t     = demands[t]
        if update_rule == "error_correction":
            # f(P) = P·exp(r·(D−D*))  →  f'(P) = exp(r·(D−D*))·(1 + r·P·D'(P))
            deriv = np.exp(r * (d_t - d_target)) * (1.0 + r * p * d_prime)
        else:
            # f(P) = r·P·D(P)  →  f'(P) = r·D + r·P·D'(P)
            deriv = r * d_t + r * p * d_prime
        lyap_terms.append(np.log(max(abs(deriv), 1e-12)))

        # Price update
        if update_rule == "error_correction":
            next_price = p * np.exp(r * (d_t - d_target))
        else:
            next_price = r * p * d_t

        if clip_price:
            next_price = float(np.clip(next_price, 1e-4, 2.0))
        prices[t + 1] = next_price

        # Reference price EMA update
        if use_reference_price:
            ref_price = ref_alpha * ref_price + (1.0 - ref_alpha) * p

        # Ornstein–Uhlenbeck dynamic WTP (Tier 1)
        if dynamic_wtp:
            noise = rng.normal(0.0, wtp_noise, size=n_consumers)
            current_wtp = np.clip(
                current_wtp + ou_theta * (base_wtp - current_wtp) + noise,
                0.0, 1.0,
            )

        # Churn & re-entry (Tier 2)
        if use_churn:
            consec_above[~buy_mask & active] += 1
            consec_above[buy_mask]            = 0
            churned = active & (consec_above >= churn_patience)
            active[churned]       = False
            consec_above[churned] = 0
            inactive_idx = np.where(~active)[0]
            if len(inactive_idx):
                n_re = rng.binomial(len(inactive_idx), reentry_rate)
                if n_re > 0:
                    re_idx = rng.choice(inactive_idx, size=n_re, replace=False)
                    active[re_idx]       = True
                    consec_above[re_idx] = 0
                    current_wtp[re_idx]  = np.clip(
                        base_wtp[re_idx] + rng.normal(0.0, 0.05, n_re),
                        0.0, 1.0,
                    )

    return {
        "prices":            prices,
        "demands":           demands,
        "buyers":            buyers_arr,
        "revenue":           revenue,
        "cost":              cost_arr,
        "profit":            profit,
        "cumulative_profit": np.cumsum(profit),
        "welfare_cs":        welfare_cs,
        "base_wtp":          base_wtp,
        "segments":          segments,
        "frames":            frames,
        "lyapunov_exponent": float(np.mean(lyap_terms)) if lyap_terms else 0.0,
    }


# ════════════════════════════════════════════════════════════════
# REGIME CLASSIFICATION  (Tier 3 — Lyapunov-based)
# ════════════════════════════════════════════════════════════════

def classify_regime(prices, lyapunov=None, transient_fraction=0.5):
    start = int(len(prices) * transient_fraction)
    tail  = prices[start:]
    if len(tail) < 10:
        return "Insufficient data"
    if lyapunov is not None:
        if lyapunov < -0.05:
            return "Stable / Convergent  (λ < 0)"
        if lyapunov < 0.05:
            return "Bifurcation boundary  (λ ≈ 0)"
        u = len(np.unique(np.round(tail, 3)))
        return "Cyclical / Periodic  (λ > 0)" if u <= 10 else "Chaotic  (λ > 0)"
    # Heuristic fallback
    tail_std = np.std(tail)
    u = len(np.unique(np.round(tail, 3)))
    if tail_std < 0.005:
        return "Stable / Convergent"
    return "Cyclical / Periodic" if u <= 8 else "Chaos-like / Irregular"


# ════════════════════════════════════════════════════════════════
# ANALYTICAL DEMAND FUNCTION  (used by deterministic sweep)
# ════════════════════════════════════════════════════════════════

def _make_demand_fn(distribution):
    """Return D(P) = P(WTP ≥ P) analytically for fast bifurcation sweeps."""
    if distribution == "uniform":
        return lambda p: float(max(0.0, 1.0 - float(np.clip(p, 0.0, 1.0))))
    elif distribution == "normal":
        if HAS_SCIPY:
            from scipy.stats import norm
            return lambda p: float(1.0 - norm.cdf(float(p), loc=0.55, scale=0.18))
        # tanh approximation to complementary normal CDF
        return lambda p: float(max(0.0, 0.5 * (1.0 + np.tanh(-(p - 0.55) / 0.18 * 1.11))))
    elif distribution == "beta":
        if HAS_SCIPY:
            from scipy.stats import beta as _beta
            return lambda p: float(1.0 - _beta.cdf(float(np.clip(p, 0.0, 1.0)), 2.0, 2.5))
        # Precomputed grid fallback
        xs  = np.linspace(0.0, 1.0, 2001)
        pdf = xs * (1.0 - xs) ** 1.5               # beta(2,2.5) unnormalised pdf
        cdf = np.cumsum(pdf); cdf /= cdf[-1]
        return lambda p: float(1.0 - np.interp(np.clip(p, 0.0, 1.0), xs, cdf))
    return lambda p: float(max(0.0, 1.0 - float(np.clip(p, 0.0, 1.0))))


# ════════════════════════════════════════════════════════════════
# FAST DETERMINISTIC SIMULATION  (Tier 3 bifurcation / Feigenbaum)
# ════════════════════════════════════════════════════════════════

def run_deterministic(r, steps=700, initial_price=0.5,
                      distribution="uniform", update_rule="original", d_target=0.5):
    """Noiseless simulation via analytical D(P).  Note: for uniform WTP + original rule
    this is exactly the logistic map  f(P) = r·P·(1−P)."""
    D      = _make_demand_fn(distribution)
    prices = np.zeros(steps + 1)
    prices[0] = initial_price
    for t in range(steps):
        p = prices[t]
        d = D(p)
        if update_rule == "error_correction":
            prices[t + 1] = float(np.clip(p * np.exp(r * (d - d_target)), 1e-7, 2.0))
        else:
            prices[t + 1] = float(np.clip(r * p * d, 1e-7, 2.0))
    return prices


# ════════════════════════════════════════════════════════════════
# BIFURCATION + LYAPUNOV SWEEP  (Tier 3)
# ════════════════════════════════════════════════════════════════

def compute_bifurcation_and_lyapunov(
    r_values, steps=700, keep_last=200,
    distribution="uniform", update_rule="original", d_target=0.5,
):
    D = _make_demand_fn(distribution)
    bif_x, bif_y, lyap_x, lyap_y = [], [], [], []

    for r in r_values:
        prices = run_deterministic(
            r, steps=steps, distribution=distribution,
            update_rule=update_rule, d_target=d_target,
        )
        tail = prices[-keep_last:]
        bif_x.extend([r] * len(tail))
        bif_y.extend(tail.tolist())

        # Analytical Lyapunov over attractor
        lsum = 0.0
        dp_e = 1e-5
        for p in tail:
            d_now   = D(p)
            d_fwd   = D(p + dp_e)
            d_prime = (d_fwd - d_now) / dp_e
            if update_rule == "error_correction":
                deriv = np.exp(r * (d_now - d_target)) * (1.0 + r * p * d_prime)
            else:
                deriv = r * d_now + r * p * d_prime
            lsum += np.log(max(abs(deriv), 1e-12))
        lyap_x.append(r)
        lyap_y.append(lsum / len(tail) if len(tail) else 0.0)

    return np.array(bif_x), np.array(bif_y), np.array(lyap_x), np.array(lyap_y)


# ════════════════════════════════════════════════════════════════
# FEIGENBAUM RATIO  (Tier 3)
# ════════════════════════════════════════════════════════════════

def compute_feigenbaum(
    r_min=0.5, r_max=4.1, n_pts=600,
    steps=900, keep_last=300, tol=0.005,
    distribution="uniform", update_rule="original", d_target=0.5,
):
    """
    Estimate period-doubling bifurcation points and Feigenbaum ratios.
    For uniform WTP + original rule the map is exactly the logistic map;
    the universal constant δ ≈ 4.669 emerges from the period-doubling cascade.
    """
    r_vals  = np.linspace(r_min, r_max, n_pts)
    periods = []

    for r in r_vals:
        tail = run_deterministic(
            r, steps=steps, distribution=distribution,
            update_rule=update_rule, d_target=d_target,
        )[-keep_last:]
        s       = np.sort(tail)
        n_uniq  = 1 + int(np.sum(np.diff(s) > tol))
        periods.append(min(n_uniq, 64))   # cap chaotic tails at 64

    periods = np.array(periods)

    # Find first r at each period-doubling transition
    bpts = {}
    for p_from, p_to in [(1, 2), (2, 4), (4, 8), (8, 16)]:
        for i in range(len(periods) - 1):
            if periods[i] <= p_from and periods[i + 1] >= p_to:
                bpts[f"{p_from}→{p_to}"] = float(r_vals[i])
                break

    # Successive interval ratios → Feigenbaum constant
    blist  = [v for _, v in sorted(bpts.items())]
    ratios = []
    for k in range(2, len(blist)):
        d_prev = blist[k - 1] - blist[k - 2]
        d_curr = blist[k]     - blist[k - 1]
        if abs(d_curr) > 1e-8:
            ratios.append(d_prev / d_curr)

    return r_vals, periods, bpts, ratios


# ════════════════════════════════════════════════════════════════
# TWO-BOT COMPETITION  (Tier 2)
# ════════════════════════════════════════════════════════════════

def run_two_bot_simulation(
    r1=2.5, r2=3.0,
    n_consumers=800, steps=300,
    initial_price_1=0.6, initial_price_2=0.5,
    distribution="uniform", seed=42,
    unit_cost=0.20,
    update_rule="original", d_target=0.5,
    choice_model="logit", logit_beta=8.0,
):
    rng      = np.random.default_rng(seed)
    base_wtp = generate_wtp_simple(n_consumers, distribution, seed)

    p1  = np.zeros(steps + 1); p2  = np.zeros(steps + 1)
    d1  = np.zeros(steps);     d2  = np.zeros(steps)
    pr1 = np.zeros(steps);     pr2 = np.zeros(steps)
    p1[0] = initial_price_1;   p2[0] = initial_price_2

    for t in range(steps):
        # Consumers who can afford at least one option
        willing = np.minimum(p1[t], p2[t]) <= base_wtp

        if choice_model == "logit":
            # Lower price → higher probability of being chosen
            prob1   = 1.0 / (1.0 + np.exp(-logit_beta * (p2[t] - p1[t])))
            buys1   = willing & (rng.random(n_consumers) < prob1)
            buys2   = willing & ~buys1
        else:
            # Bertrand: lowest price takes all; ties to bot 1
            buys1 = willing & (p1[t] <= p2[t])
            buys2 = willing & (p2[t] <  p1[t])

        d1[t]  = buys1.sum() / n_consumers
        d2[t]  = buys2.sum() / n_consumers
        pr1[t] = (p1[t] - unit_cost) * buys1.sum()
        pr2[t] = (p2[t] - unit_cost) * buys2.sum()

        def _nxt(p, d, r):
            if update_rule == "error_correction":
                return float(np.clip(p * np.exp(r * (d - d_target)), 1e-4, 2.0))
            return float(np.clip(r * p * d, 1e-4, 2.0))

        p1[t + 1] = _nxt(p1[t], d1[t], r1)
        p2[t + 1] = _nxt(p2[t], d2[t], r2)

    return {
        "prices_1": p1, "prices_2": p2,
        "demands_1": d1, "demands_2": d2,
        "profits_1": pr1, "profits_2": pr2,
        "cumprof_1": np.cumsum(pr1), "cumprof_2": np.cumsum(pr2),
    }


# ════════════════════════════════════════════════════════════════
# Q-LEARNING BOT  (Tier 4)
# ════════════════════════════════════════════════════════════════

def train_q_bot(
    n_consumers=500, n_episodes=80, episode_steps=200,
    distribution="uniform", seed=42, unit_cost=0.20,
    n_price_bins=12, n_demand_bins=10, n_actions=7,
    gamma=0.95, alpha_lr=0.1,
    epsilon_start=1.0, epsilon_end=0.05,
):
    rng      = np.random.default_rng(seed)
    base_wtp = generate_wtp_simple(n_consumers, distribution, seed)
    deltas   = np.linspace(-0.30, 0.30, n_actions)   # price increments
    Q        = np.zeros((n_price_bins, n_demand_bins, n_actions))

    def sp(v): return int(np.clip(v / 1.5 * n_price_bins,   0, n_price_bins - 1))
    def sd(v): return int(np.clip(v       * n_demand_bins,   0, n_demand_bins - 1))

    rewards_hist = []
    for ep in range(n_episodes):
        eps    = epsilon_start - (epsilon_start - epsilon_end) * ep / max(1, n_episodes - 1)
        p      = float(rng.uniform(0.1, 1.0))
        ep_rew = 0.0
        for _ in range(episode_steps):
            d     = float(np.mean(base_wtp >= p))
            s_p   = sp(p); s_d = sd(d)
            a     = rng.integers(n_actions) if rng.random() < eps else int(np.argmax(Q[s_p, s_d]))
            p2    = float(np.clip(p + deltas[a], 1e-3, 1.5))
            d2    = float(np.mean(base_wtp >= p2))
            rew   = (p2 - unit_cost) * d2 * n_consumers
            s_p2  = sp(p2); s_d2 = sd(d2)
            Q[s_p, s_d, a] += alpha_lr * (rew + gamma * np.max(Q[s_p2, s_d2]) - Q[s_p, s_d, a])
            p = p2; ep_rew += rew
        rewards_hist.append(ep_rew / episode_steps)

    return Q, deltas, np.array(rewards_hist)


def run_q_bot_simulation(Q, deltas, n_consumers=500, steps=300,
                          distribution="uniform", seed=42, unit_cost=0.20):
    base_wtp = generate_wtp_simple(n_consumers, distribution, seed)
    n_pb = Q.shape[0]; n_db = Q.shape[1]

    def sp(v): return int(np.clip(v / 1.5 * n_pb, 0, n_pb - 1))
    def sd(v): return int(np.clip(v       * n_db, 0, n_db - 1))

    prices = [0.5]; demands = []; profits = []
    for _ in range(steps):
        p  = prices[-1]
        d  = float(np.mean(base_wtp >= p))
        a  = int(np.argmax(Q[sp(p), sd(d)]))
        p2 = float(np.clip(p + deltas[a], 1e-3, 1.5))
        demands.append(d)
        profits.append((p - unit_cost) * d * n_consumers)
        prices.append(p2)

    return {
        "prices":            np.array(prices),
        "demands":           np.array(demands),
        "profits":           np.array(profits),
        "cumulative_profit": np.cumsum(profits),
    }


# ════════════════════════════════════════════════════════════════
# MPC BOT — K-step lookahead with smoothness penalty  (Tier 4)
# ════════════════════════════════════════════════════════════════

def run_mpc_simulation(
    n_consumers=500, steps=300, K=6, lambda_smooth=0.15,
    distribution="uniform", seed=42, unit_cost=0.20,
):
    if not HAS_SCIPY:
        return None

    base_wtp = generate_wtp_simple(n_consumers, distribution, seed)

    def profit_fn(p):
        d = float(np.mean(base_wtp >= np.clip(p, 0.0, 2.0)))
        return (float(p) - unit_cost) * d * n_consumers

    prices = [0.5]; demands = []; profits = []

    for _ in range(steps):
        p0 = prices[-1]

        def neg_obj(pv):
            total = sum(-profit_fn(pv[k]) for k in range(K))
            # Smoothness penalty: penalise large price jumps
            total += lambda_smooth * sum((pv[k + 1] - pv[k]) ** 2 for k in range(K - 1))
            return total

        res = minimize(
            neg_obj, np.full(K, p0), method="L-BFGS-B",
            bounds=[(1e-3, 1.5)] * K,
            options={"maxiter": 40, "ftol": 1e-5},
        )
        p_next = float(np.clip(res.x[0], 1e-3, 1.5))
        d      = float(np.mean(base_wtp >= p_next))
        demands.append(d)
        profits.append(profit_fn(p_next))
        prices.append(p_next)

    return {
        "prices":            np.array(prices),
        "demands":           np.array(demands),
        "profits":           np.array(profits),
        "cumulative_profit": np.cumsum(profits),
    }


# ════════════════════════════════════════════════════════════════
# OGY CHAOS CONTROL  (Tier 5)
# ════════════════════════════════════════════════════════════════

def run_ogy_simulation(
    r_base=3.5, delta_r_max=0.20,
    n_consumers=500, steps=400,
    distribution="uniform", seed=42, unit_cost=0.20,
    update_rule="original", d_target=0.5,
):
    """
    Ott–Grebogi–Yorke chaos control.
    At each step apply a tiny δr to steer the trajectory toward the unstable
    fixed point P* embedded in the chaotic attractor:

        δr = −(P_t − P*) / (∂f/∂r |_{P*})    clamped to [−Δr_max, Δr_max]
    """
    if not HAS_SCIPY:
        return None

    base_wtp = generate_wtp_simple(n_consumers, distribution, seed)

    def demand_fn(p):
        return float(np.mean(base_wtp >= np.clip(p, 0.0, 2.0)))

    def map_fn(p, r):
        d = demand_fn(p)
        if update_rule == "error_correction":
            return float(np.clip(p * np.exp(r * (d - d_target)), 1e-4, 2.0))
        return float(np.clip(r * p * d, 1e-4, 2.0))

    # Find fixed point P*: map_fn(P*, r_base) = P*
    def fp_eq(p):
        return map_fn(p, r_base) - p
    try:
        p_star = brentq(fp_eq, 0.01, 1.49, xtol=1e-7)
    except Exception:
        p_star = 1.0 / r_base   # analytic fallback for original rule

    prices = [0.5]; r_used = []; profits = []; demands_list = []

    for _ in range(steps):
        p = prices[-1]
        d = demand_fn(p)
        profits.append((p - unit_cost) * d * n_consumers)
        demands_list.append(d)

        # OGY correction  (∂f/∂r ≈ P*·D(P*) for the original rule)
        df_dr   = p_star * demand_fn(p_star)
        delta_r = 0.0
        if abs(df_dr) > 1e-8:
            delta_r = float(np.clip(-(p - p_star) / df_dr, -delta_r_max, delta_r_max))

        r_eff = r_base + delta_r
        r_used.append(r_eff)
        prices.append(map_fn(p, r_eff))

    return {
        "prices":            np.array(prices),
        "demands":           np.array(demands_list),
        "profits":           np.array(profits),
        "cumulative_profit": np.cumsum(profits),
        "r_used":            np.array(r_used),
        "p_star":            p_star,
    }


# ════════════════════════════════════════════════════════════════
# WELFARE METRICS  (Tier 5)
# ════════════════════════════════════════════════════════════════

def compute_welfare_summary(result, transient=0.5):
    n     = len(result["profit"])
    start = int(n * transient)
    cs    = result["welfare_cs"][start:]
    prf   = result["profit"][start:]
    prc   = result["prices"][start:-1]

    # Per-step-profit Gini coefficient
    s  = np.sort(np.maximum(prf, 0.0))
    ns = len(s)
    gini = float(
        2.0 * np.sum(np.arange(1, ns + 1) * s) / (ns * (np.sum(s) + 1e-12)) - (ns + 1) / ns
    )

    return {
        "Avg consumer surplus / step": round(float(np.mean(cs)), 2),
        "Total surplus  (CS + profit)": round(float(np.sum(cs) + np.sum(prf)), 1),
        "Avg profit / step":            round(float(np.mean(prf)), 2),
        "Price CoV  (fairness)":        round(float(np.std(prc) / (np.mean(prc) + 1e-12)), 4),
        "Profit Gini":                  round(gini, 4),
        "Lyapunov exponent λ":          round(result.get("lyapunov_exponent", 0.0), 4),
    }


# ════════════════════════════════════════════════════════════════
# PROFIT OPTIMISATION SCAN
# ════════════════════════════════════════════════════════════════

def optimize_r(r_min, r_max, num_r, n_consumers, steps, initial_price,
               distribution, seed, dynamic_wtp, wtp_noise,
               use_seasonality, season_amplitude, season_period,
               unit_cost, fixed_cost_per_step, update_rule, d_target):
    rows = []
    for r in np.linspace(r_min, r_max, num_r):
        res = run_simulation(
            r=float(r), n_consumers=n_consumers, steps=steps,
            initial_price=initial_price, distribution=distribution, seed=seed,
            dynamic_wtp=dynamic_wtp, wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude, season_period=season_period,
            unit_cost=unit_cost, fixed_cost_per_step=fixed_cost_per_step,
            update_rule=update_rule, d_target=d_target,
        )
        ts = max(1, int(steps * 0.5))
        rows.append({
            "r":                   round(float(r), 4),
            "avg_profit_long_run": round(float(np.mean(res["profit"][ts:])), 2),
            "total_profit":        round(float(np.sum(res["profit"])), 1),
            "price_volatility":    round(float(np.std(res["prices"][ts:])), 4),
            "avg_price":           round(float(np.mean(res["prices"][ts:])), 4),
            "avg_demand":          round(float(np.mean(res["demands"][ts:])), 4),
            "lyapunov":            round(res["lyapunov_exponent"], 4),
            "regime":              classify_regime(res["prices"], res["lyapunov_exponent"]),
        })
    return sorted(rows, key=lambda x: x["avg_profit_long_run"], reverse=True)


def find_best_r_for_given_price_cost(
    given_price,
    given_unit_cost,
    given_fixed_cost,
    r_min=0.0,
    r_max=4.5,
    num_r=100,
    n_consumers=800,
    steps=300,
    distribution="uniform",
    seed=42,
    dynamic_wtp=False,
    wtp_noise=0.02,
    ou_theta=0.10,
    use_seasonality=False,
    season_amplitude=0.2,
    season_period=30,
    use_reference_price=False,
    ref_alpha=0.7,
    ref_beta=0.15,
    use_churn=False,
    churn_patience=3,
    reentry_rate=0.005,
    use_segments=False,
    update_rule="original",
    d_target=0.5,
):
    """
    For a fixed starting price and cost structure, scan over r and return:
    - full ranking table
    - best row
    - best simulation result
    """
    ranking = []
    r_grid = np.linspace(r_min, r_max, num_r)

    for r_val in r_grid:
        sim = run_simulation(
            r=float(r_val),
            n_consumers=n_consumers,
            steps=steps,
            initial_price=given_price,
            distribution=distribution,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            wtp_noise=wtp_noise,
            ou_theta=ou_theta,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            use_reference_price=use_reference_price,
            ref_alpha=ref_alpha,
            ref_beta=ref_beta,
            use_churn=use_churn,
            churn_patience=churn_patience,
            reentry_rate=reentry_rate,
            use_segments=use_segments,
            update_rule=update_rule,
            d_target=d_target,
            unit_cost=given_unit_cost,
            fixed_cost_per_step=given_fixed_cost,
        )

        tail_start = max(1, int(steps * 0.5))

        ranking.append({
            "r": round(float(r_val), 4),
            "avg_profit_long_run": round(float(np.mean(sim["profit"][tail_start:])), 4),
            "total_profit": round(float(np.sum(sim["profit"])), 4),
            "avg_price_long_run": round(float(np.mean(sim["prices"][tail_start:])), 4),
            "avg_demand_long_run": round(float(np.mean(sim["demands"][tail_start:])), 4),
            "price_volatility": round(float(np.std(sim["prices"][tail_start:])), 4),
            "lyapunov": round(float(sim["lyapunov_exponent"]), 4),
            "regime": classify_regime(sim["prices"], sim["lyapunov_exponent"]),
        })

    ranking = sorted(ranking, key=lambda x: x["avg_profit_long_run"], reverse=True)
    best = ranking[0]

    best_result = run_simulation(
        r=float(best["r"]),
        n_consumers=n_consumers,
        steps=steps,
        initial_price=given_price,
        distribution=distribution,
        seed=seed,
        dynamic_wtp=dynamic_wtp,
        wtp_noise=wtp_noise,
        ou_theta=ou_theta,
        use_seasonality=use_seasonality,
        season_amplitude=season_amplitude,
        season_period=season_period,
        use_reference_price=use_reference_price,
        ref_alpha=ref_alpha,
        ref_beta=ref_beta,
        use_churn=use_churn,
        churn_patience=churn_patience,
        reentry_rate=reentry_rate,
        use_segments=use_segments,
        update_rule=update_rule,
        d_target=d_target,
        unit_cost=given_unit_cost,
        fixed_cost_per_step=given_fixed_cost,
    )

    return ranking, best, best_result


# ════════════════════════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════════════════════════

st.sidebar.header("Core Parameters")
r           = st.sidebar.slider("Aggression r", 0.0, 4.5, 2.5, 0.01)
n_consumers = st.sidebar.slider("Consumers N", 200, 2000, 800, 100)
steps       = st.sidebar.slider("Time steps",   50, 800, 300, 10)
init_price  = st.sidebar.slider("Initial price P₀", 0.01, 1.5, 0.5, 0.01)
seed        = int(st.sidebar.number_input("Random seed", 0, 999999, 42, 1))

st.sidebar.subheader("Update Rule")
_rule_label = st.sidebar.selectbox(
    "Price update rule",
    ["original  ·  P(t+1) = r·P·D", "error_correction  ·  P(t+1) = P·exp(r·(D−D*))"],
)
update_rule = "error_correction" if "error" in _rule_label else "original"
d_target    = st.sidebar.slider(
    "Target demand D*", 0.1, 0.9, 0.5, 0.05,
    help="Used only by the error-correction rule",
    disabled=(update_rule == "original"),
)

st.sidebar.subheader("Consumer WTP")
use_segments = st.sidebar.checkbox("3-segment consumers (price-sensitive / mainstream / loyal)", False)
distribution = st.sidebar.selectbox(
    "WTP distribution  (non-segmented)", ["uniform", "normal", "beta"],
    disabled=use_segments,
)
dynamic_wtp = st.sidebar.checkbox("Dynamic WTP — Ornstein–Uhlenbeck process", False)
wtp_noise   = st.sidebar.slider("OU noise σ",       0.001, 0.10, 0.02, 0.001, disabled=not dynamic_wtp)
ou_theta    = st.sidebar.slider("OU mean-rev speed θ", 0.01, 0.50, 0.10, 0.01, disabled=not dynamic_wtp)

st.sidebar.subheader("Market Features")
use_seasonality  = st.sidebar.checkbox("Seasonality  (scales WTP — corrected)", False)
season_amplitude = st.sidebar.slider("Season amplitude", 0.0, 0.5, 0.20, 0.01, disabled=not use_seasonality)
season_period    = st.sidebar.slider("Season period (steps)", 5, 90, 30, 1, disabled=not use_seasonality)

use_ref_price = st.sidebar.checkbox("Reference-price anchoring", False)
ref_alpha     = st.sidebar.slider("Anchor EMA α", 0.10, 0.99, 0.70, 0.01, disabled=not use_ref_price)
ref_beta      = st.sidebar.slider("WTP sensitivity β", 0.0, 0.5, 0.15, 0.01, disabled=not use_ref_price)

use_churn      = st.sidebar.checkbox("Churn & re-entry", False)
churn_patience = st.sidebar.slider("Churn patience (steps)", 1, 20, 3, 1, disabled=not use_churn)
reentry_rate   = st.sidebar.slider("Re-entry rate / step", 0.001, 0.05, 0.005, 0.001, disabled=not use_churn)

st.sidebar.subheader("Cost Structure")
unit_cost           = st.sidebar.slider("Unit cost c", 0.00, 0.50, 0.20, 0.01)
fixed_cost_per_step = st.sidebar.slider("Fixed cost / step", 0.0, 50.0, 0.0, 1.0)

if st.sidebar.button("▶  Run simulation", type="primary", use_container_width=True):
    with st.spinner("Running simulation…"):
        st.session_state["result"] = run_simulation(
            r=r, n_consumers=n_consumers, steps=steps, initial_price=init_price,
            distribution=distribution, seed=seed,
            dynamic_wtp=dynamic_wtp, wtp_noise=wtp_noise, ou_theta=ou_theta,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude, season_period=season_period,
            use_reference_price=use_ref_price, ref_alpha=ref_alpha, ref_beta=ref_beta,
            use_churn=use_churn, churn_patience=churn_patience, reentry_rate=reentry_rate,
            use_segments=use_segments,
            update_rule=update_rule, d_target=d_target,
            unit_cost=unit_cost, fixed_cost_per_step=fixed_cost_per_step,
        )

if "result" not in st.session_state:
    st.info("👈  Configure parameters in the sidebar and click ▶ Run simulation to begin.")
    st.stop()

# ── Shorthand references ─────────────────────────────────────
result    = st.session_state["result"]
prices    = result["prices"]
demands   = result["demands"]
profit    = result["profit"]
cum_prf   = result["cumulative_profit"]
lam       = result["lyapunov_exponent"]
regime    = classify_regime(prices, lam)

ts        = int(len(profit) * 0.5)          # tail start
price_vol = float(np.std(prices[ts:]))
avg_price = float(np.mean(prices[ts:]))
avg_dem   = float(np.mean(demands[ts:]))
avg_prf   = float(np.mean(profit[ts:]))
tot_prf   = float(np.sum(profit))

# ════════════════════════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
    "📈 Main Simulation",
    "🔬 Nonlinear Analysis",
    "⚔️  Two-Bot Competition",
    "🤖 Adaptive Bots",
    "⚖️  Welfare Analysis",
    "🎯 Profit Optimisation",
    "🧮 Best r for Given Price & Cost",
    "🎬 Live Market View",
])
# ────────────────────────────────────────────────────────────────
# TAB 8 — LIVE MARKET VIEW
# ────────────────────────────────────────────────────────────────
with tab8:
    st.subheader("Live Market View")
    st.caption("Use Play / Pause / Next / Reset to watch the pricing loop unfold over time.")

    frames_json = json.dumps(result["frames"])

    components.html(
        f"""
        <html>
        <head>
          <style>
            body {{
              margin: 0;
              background: #060b18;
              color: white;
              font-family: Arial, sans-serif;
            }}

            .wrap {{
              padding: 16px;
            }}

            .top {{
              display: flex;
              flex-wrap: wrap;
              gap: 16px;
              margin-bottom: 16px;
              font-size: 17px;
              font-weight: 600;
            }}

            .stat {{
              background: rgba(255,255,255,0.04);
              border: 1px solid rgba(255,255,255,0.08);
              border-radius: 12px;
              padding: 10px 14px;
            }}

            .legend {{
              display: flex;
              flex-wrap: wrap;
              gap: 16px;
              margin: 8px 0 16px 0;
              font-size: 14px;
              color: rgba(255,255,255,0.78);
            }}

            .legend-item {{
              display: flex;
              align-items: center;
              gap: 8px;
            }}

            .legend-dot {{
              width: 12px;
              height: 12px;
              border-radius: 50%;
            }}

            .grid {{
              display: grid;
              grid-template-columns: repeat(20, 18px);
              gap: 8px;
              margin: 18px 0 20px 0;
              justify-content: start;
            }}

            .dot {{
              width: 18px;
              height: 18px;
              border-radius: 50%;
              opacity: 0.35;
              transition: all 0.18s ease;
            }}

            .price-sensitive {{ background: #ef4444; }}
            .mainstream {{ background: #3b82f6; }}
            .loyal {{ background: #22c55e; }}

            .bought {{
              opacity: 1;
              transform: scale(1.2);
              box-shadow: 0 0 10px rgba(255,255,255,0.35);
            }}

            .inactive {{
              opacity: 0.08;
            }}

            .controls {{
              display: flex;
              flex-wrap: wrap;
              align-items: center;
              gap: 10px;
              margin-top: 16px;
            }}

            .speed-wrap {{
              display: flex;
              align-items: center;
              gap: 10px;
              margin-left: 8px;
              color: rgba(255,255,255,0.82);
              font-size: 14px;
            }}

            input[type="range"] {{
              width: 180px;
              accent-color: #ffffff;
            }}

            button {{
              background: #111111;
              color: white;
              border: 1px solid #333333;
              padding: 8px 14px;
              border-radius: 10px;
              cursor: pointer;
              font-weight: 600;
            }}

            button:hover {{
              background: #1f1f1f;
            }}
          </style>
        </head>
        <body>
          <div class="wrap">
            <div class="top">
              <div id="step" class="stat">Step: 0</div>
              <div id="price" class="stat">Price: 0</div>
              <div id="demand" class="stat">Demand: 0</div>
              <div id="profit" class="stat">Profit: 0</div>
            </div>

            <div class="legend">
              <div class="legend-item"><span class="legend-dot price-sensitive"></span>Price-sensitive</div>
              <div class="legend-item"><span class="legend-dot mainstream"></span>Mainstream</div>
              <div class="legend-item"><span class="legend-dot loyal"></span>Loyal</div>
              <div class="legend-item"><span class="legend-dot" style="background:#ffffff;"></span>Bright = bought this step</div>
              <div class="legend-item"><span class="legend-dot" style="background:#666666;"></span>Faded = inactive</div>
            </div>

            <div id="grid" class="grid"></div>

            <div class="controls">
              <button onclick="playSim()">Play</button>
              <button onclick="pauseSim()">Pause</button>
              <button onclick="nextFrame()">Next</button>
              <button onclick="resetSim()">Reset</button>

              <div class="speed-wrap">
                <span>Speed</span>
                <input id="speedSlider" type="range" min="80" max="1200" value="450" step="20" oninput="updateSpeed(this.value)">
                <span id="speedLabel">450 ms</span>
              </div>
            </div>
          </div>

          <script>
            const frames = {frames_json};
            let idx = 0;
            let timer = null;
            let frameDelay = 450;

            function renderFrame(frame) {{
              document.getElementById("step").innerText = "Step: " + frame.step;
              document.getElementById("price").innerText = "Price: " + frame.price.toFixed(3);
              document.getElementById("demand").innerText = "Demand: " + frame.demand.toFixed(3);
              document.getElementById("profit").innerText = "Profit: " + frame.profit.toFixed(2);

              const grid = document.getElementById("grid");
              grid.innerHTML = "";

              for (let i = 0; i < frame.buyers.length; i++) {{
                const dot = document.createElement("div");
                dot.classList.add("dot");

                const seg = frame.segments[i];
                if (seg === "price_sensitive") dot.classList.add("price-sensitive");
                else if (seg === "mainstream") dot.classList.add("mainstream");
                else if (seg === "loyal") dot.classList.add("loyal");
                else dot.classList.add("mainstream");

                if (frame.buyers[i] === 1) dot.classList.add("bought");
                if (frame.active[i] === 0) dot.classList.add("inactive");

                grid.appendChild(dot);
              }}
            }}

            function nextFrame() {{
              if (idx < frames.length) {{
                renderFrame(frames[idx]);
                idx += 1;
              }} else {{
                pauseSim();
              }}
            }}

            function playSim() {{
              if (timer) return;
              timer = setInterval(nextFrame, frameDelay);
            }}

            function pauseSim() {{
              clearInterval(timer);
              timer = null;
            }}

            function updateSpeed(value) {{
              frameDelay = Number(value);
              document.getElementById("speedLabel").innerText = frameDelay + " ms";
              if (timer) {{
                pauseSim();
                playSim();
              }}
            }}

            function resetSim() {{
              pauseSim();
              idx = 0;
              if (frames.length > 0) renderFrame(frames[0]);
            }}

            if (frames.length > 0) renderFrame(frames[0]);
            updateSpeed(frameDelay);
          </script>
        </body>
        </html>
        """,
        height=620,
    )


# ────────────────────────────────────────────────────────────────
# TAB 1 — MAIN SIMULATION
# ────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Key Performance Indicators")
    c = st.columns(7)
    c[0].metric("Regime",              regime.split("(")[0].strip())
    c[1].metric("Lyapunov λ",          f"{lam:.4f}")
    c[2].metric("Price volatility",    f"{price_vol:.4f}")
    c[3].metric("Avg long-run price",  f"{avg_price:.4f}")
    c[4].metric("Avg long-run demand", f"{avg_dem:.4f}")
    c[5].metric("Avg long-run profit", f"{avg_prf:.2f}")
    c[6].metric("Total profit",        f"{tot_prf:.1f}")

    r1a, r1b = st.columns(2)
    with r1a:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(prices, lw=1.4, color="#3366cc")
        ax.set_title("Price time series"); ax.set_xlabel("Step"); ax.set_ylabel("Price")
        st.pyplot(fig); plt.close()
    with r1b:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(demands, lw=1.4, color="#e65c00")
        ax.set_title("Demand time series"); ax.set_xlabel("Step"); ax.set_ylabel("Demand")
        st.pyplot(fig); plt.close()

    r2a, r2b = st.columns(2)
    with r2a:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(profit, lw=1.1, color="#2ca02c", alpha=0.75)
        ax.set_title("Profit per step"); ax.set_xlabel("Step"); ax.set_ylabel("Profit")
        st.pyplot(fig); plt.close()
    with r2b:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(cum_prf, lw=1.5, color="#9467bd")
        ax.set_title("Cumulative profit"); ax.set_xlabel("Step"); ax.set_ylabel("Cum. profit")
        st.pyplot(fig); plt.close()

    r3a, r3b = st.columns(2)
    with r3a:
        fig, ax = plt.subplots(figsize=(7, 4))
        if use_segments:
            seg_colours = {"price_sensitive": "#d62728", "mainstream": "#1f77b4", "loyal": "#2ca02c"}
            for seg, col in seg_colours.items():
                wtp_seg = result["base_wtp"][result["segments"] == seg]
                ax.hist(wtp_seg, bins=30, alpha=0.6, label=seg, color=col)
            ax.legend()
        else:
            ax.hist(result["base_wtp"], bins=30, color="#1f77b4", alpha=0.75)
        ax.set_title("Consumer WTP distribution"); ax.set_xlabel("WTP"); ax.set_ylabel("Count")
        st.pyplot(fig); plt.close()
    with r3b:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(prices[:-1], profit, s=7, alpha=0.45, color="#ff7f0e")
        ax.set_title("Profit vs published price"); ax.set_xlabel("Price"); ax.set_ylabel("Profit")
        st.pyplot(fig); plt.close()

    st.subheader("Interpretation")
    _lam_desc = ("stable" if lam < -0.05
                 else "at a period-doubling boundary" if lam < 0.05
                 else "in a chaotic regime")
    st.markdown(
        f"With **r = {r:.2f}** and update rule **{update_rule}**, the system is "
        f"classified as **{regime}** — the Lyapunov exponent λ = {lam:.4f} indicates "
        f"the system is {_lam_desc}.  "
        f"Long-run avg profit: **{avg_prf:.2f}** · total profit: **{tot_prf:.1f}**."
    )
    st.info(
        "**Note on Lyapunov sign**: λ < 0 → stable fixed point or periodic orbit. "
        "λ ≈ 0 → period-doubling bifurcation point. λ > 0 → deterministic chaos, "
        "exponential sensitivity to initial price."
    )


# ────────────────────────────────────────────────────────────────
# TAB 2 — NONLINEAR ANALYSIS
# ────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Return Map  (P_{t+1} vs P_t)")
    st.caption(
        "A single dot → stable fixed point. Two dots → period-2 orbit. "
        "A filled curve or cloud → chaos."
    )
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(prices[:-1], prices[1:], s=4, alpha=0.4, color="#333333")
    ax.plot([0, max(prices)], [0, max(prices)], lw=0.7, ls="--", color="#aaa")
    ax.set_xlabel("P_t"); ax.set_ylabel("P_{t+1}"); ax.set_title("Return map (attractor)")
    ax.set_aspect("equal")
    st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Bifurcation Diagram + Lyapunov Exponent Sweep")
    st.caption(
        "Uses the **deterministic** (analytical) demand function D(P) = 1 − CDF_WTP(P). "
        "For uniform WTP with the original rule this is exactly the logistic map f(P) = r·P·(1−P)."
    )
    with st.expander("Sweep settings", expanded=True):
        bc1, bc2, bc3 = st.columns(3)
        bif_rmin = bc1.slider("r min", 0.0, 4.5, 0.5, 0.05, key="bifmin")
        bif_rmax = bc2.slider("r max", 0.1, 4.5, 4.0, 0.05, key="bifmax")
        bif_npts = bc3.slider("# r values", 50, 500, 220, 10,  key="bifnpts")

    if st.button("Compute bifurcation + Lyapunov", type="primary", key="run_bif"):
        if bif_rmax <= bif_rmin:
            st.error("r max must exceed r min.")
        else:
            with st.spinner("Sweeping r values with analytical demand…"):
                r_arr = np.linspace(bif_rmin, bif_rmax, bif_npts)
                bx, by, lx, ly = compute_bifurcation_and_lyapunov(
                    r_arr, distribution=distribution,
                    update_rule=update_rule, d_target=d_target,
                )
                st.session_state["bif"] = (bx, by, lx, ly)

    if "bif" in st.session_state:
        bx, by, lx, ly = st.session_state["bif"]
        ly_np = np.array(ly)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.scatter(bx, by, s=0.8, alpha=0.35, color="#222222")
        ax1.set_ylabel("Long-run price  P∞")
        ax1.set_title("Bifurcation diagram  (deterministic analytical map)")

        ax2.plot(lx, ly, lw=1.1, color="#c0392b")
        ax2.axhline(0, color="black", lw=0.9, ls="--")
        ax2.fill_between(lx, ly, 0, where=ly_np  > 0, alpha=0.18, color="#c0392b")
        ax2.fill_between(lx, ly, 0, where=ly_np <= 0, alpha=0.12, color="#27ae60")
        ax2.set_xlabel("Aggression r")
        ax2.set_ylabel("Lyapunov λ")
        ax2.set_title("Lyapunov exponent  (red = chaotic, green = stable / periodic)")

        plt.tight_layout()
        st.pyplot(fig); plt.close()

    st.divider()
    st.subheader("Feigenbaum Ratio Estimation")
    st.markdown(
        r"""
For **uniform WTP + original rule**, the map is $f(P) = rP(1-P)$ — the classical
**logistic map**. The universal Feigenbaum constant $\delta \approx 4.669$ emerges from
successive period-doubling bifurcation intervals:

$$\delta = \lim_{n\to\infty} \frac{r_{n} - r_{n-1}}{r_{n+1} - r_n} \approx 4.669\,201\ldots$$

This constant is the *same* for any smooth 1D map with a single quadratic maximum —
demonstrating that your pricing system is in the same universality class as the logistic
growth model, the Belousov–Zhabotinsky reaction, and many other physical systems.
"""
    )
    if st.button("Compute Feigenbaum  (≈ 5 s)", type="primary", key="run_feig"):
        with st.spinner("Scanning period-doubling cascade…"):
            f_rvals, f_periods, f_bpts, f_ratios = compute_feigenbaum(
                distribution=distribution,
                update_rule=update_rule, d_target=d_target,
            )
            st.session_state["feig"] = (f_rvals, f_periods, f_bpts, f_ratios)

    if "feig" in st.session_state:
        f_rvals, f_periods, f_bpts, f_ratios = st.session_state["feig"]
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.plot(f_rvals, f_periods, lw=0.9, color="#5533cc")
        for label, rv in f_bpts.items():
            ax.axvline(rv, color="red", lw=0.8, ls="--", alpha=0.75)
            ax.text(rv + 0.01, max(f_periods) * 0.8, label, fontsize=8, color="red")
        ax.set_xlabel("Aggression r"); ax.set_ylabel("Attractor period")
        ax.set_title("Attractor period vs r  (red dashes = period-doubling bifurcations)")
        st.pyplot(fig); plt.close()

        if f_bpts:
            st.markdown("**Detected bifurcation points:**")
            for k, v in sorted(f_bpts.items()):
                st.write(f"  Period {k}:  r = **{v:.5f}**")
        if f_ratios:
            st.markdown("**Feigenbaum interval ratios**  (converging to δ ≈ 4.669):")
            for i, ratio in enumerate(f_ratios):
                delta = ratio - 4.669
                sign  = "+" if delta >= 0 else ""
                st.write(f"  Ratio {i+1}: **{ratio:.4f}**  ({sign}{delta:.4f} from ideal)")
            best_r = f_ratios[-1]
            st.metric("Best Feigenbaum estimate δ", f"{best_r:.4f}",
                      delta=f"{best_r - 4.669:+.4f} vs ideal 4.669201…")
        else:
            st.info(
                "Not enough period-doubling transitions detected. "
                "For best results use uniform WTP + original update rule."
            )


# ────────────────────────────────────────────────────────────────
# TAB 3 — TWO-BOT COMPETITION
# ────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Two-Bot Bertrand / Logit Competition")
    st.markdown(
        "Two pricing bots compete for the same pool of consumers. "
        "Each observes only its own demand and updates independently. "
        "Use the **logit** choice model for probabilistic splitting "
        "(more realistic) or **Bertrand** (lowest price takes all)."
    )
    cc1, cc2 = st.columns(2)
    with cc1:
        r1_bot  = st.slider("Bot 1 aggression r₁", 0.0, 4.5, 2.5, 0.05, key="r1b")
        p1_init = st.slider("Bot 1 initial price",  0.01, 1.5, 0.6, 0.01, key="p1i")
    with cc2:
        r2_bot  = st.slider("Bot 2 aggression r₂", 0.0, 4.5, 3.0, 0.05, key="r2b")
        p2_init = st.slider("Bot 2 initial price",  0.01, 1.5, 0.5, 0.01, key="p2i")

    _cm_label    = st.selectbox("Consumer choice model",
                                ["logit  (probabilistic softmax)", "lowest_price  (Bertrand)"])
    choice_model = "logit" if "logit" in _cm_label else "lowest_price"
    logit_beta_v = st.slider("Logit price sensitivity β", 1.0, 20.0, 8.0, 0.5,
                             disabled=(choice_model == "lowest_price"), key="lgb")

    if st.button("▶  Run competition", type="primary", key="run_comp"):
        with st.spinner("Simulating two-bot market…"):
            st.session_state["comp"] = run_two_bot_simulation(
                r1=r1_bot, r2=r2_bot,
                n_consumers=n_consumers, steps=steps,
                initial_price_1=p1_init, initial_price_2=p2_init,
                distribution=distribution, seed=seed, unit_cost=unit_cost,
                update_rule=update_rule, d_target=d_target,
                choice_model=choice_model, logit_beta=logit_beta_v,
            )

    if "comp" in st.session_state:
        comp = st.session_state["comp"]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].plot(comp["prices_1"], lw=1.2, color="#1f77b4", label="Bot 1")
        axes[0, 0].plot(comp["prices_2"], lw=1.2, color="#d62728", ls="--", label="Bot 2")
        axes[0, 0].set_title("Price time series"); axes[0, 0].legend()

        # Fidelity: the user may change the sidebar `steps` after running the
        # competition; keep plotting dimensions consistent with cached results.
        n_pts = min(len(comp["demands_1"]), len(comp["demands_2"]))
        x_share = range(n_pts)
        d1 = comp["demands_1"][:n_pts]
        d2 = comp["demands_2"][:n_pts]
        axes[0, 1].stackplot(
            x_share, d1, d2,
            labels=["Bot 1 share", "Bot 2 share"],
            colors=["#1f77b4", "#d62728"], alpha=0.7,
        )
        axes[0, 1].set_title("Market share split"); axes[0, 1].legend(loc="upper right")

        axes[1, 0].plot(comp["profits_1"], lw=1.0, color="#1f77b4", alpha=0.8, label="Bot 1")
        axes[1, 0].plot(comp["profits_2"], lw=1.0, color="#d62728", alpha=0.8, ls="--", label="Bot 2")
        axes[1, 0].set_title("Profit per step"); axes[1, 0].legend()

        axes[1, 1].plot(comp["cumprof_1"], lw=1.5, color="#1f77b4", label="Bot 1")
        axes[1, 1].plot(comp["cumprof_2"], lw=1.5, color="#d62728", ls="--", label="Bot 2")
        axes[1, 1].set_title("Cumulative profit"); axes[1, 1].legend()

        for ax in axes.flat:
            ax.set_xlabel("Step"); ax.grid(True, alpha=0.25)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        _tail = int(steps * 0.5)
        summary_df = pd.DataFrame({
            "Metric": ["Avg long-run profit", "Total profit",
                       "Avg market share", "Price volatility"],
            "Bot 1": [
                round(float(np.mean(comp["profits_1"][_tail:])), 2),
                round(float(np.sum(comp["profits_1"])), 1),
                round(float(np.mean(comp["demands_1"][_tail:])), 3),
                round(float(np.std(comp["prices_1"][_tail:])), 4),
            ],
            "Bot 2": [
                round(float(np.mean(comp["profits_2"][_tail:])), 2),
                round(float(np.sum(comp["profits_2"])), 1),
                round(float(np.mean(comp["demands_2"][_tail:])), 3),
                round(float(np.std(comp["prices_2"][_tail:])), 4),
            ],
        }).set_index("Metric")
        st.dataframe(summary_df, use_container_width=True)


# ────────────────────────────────────────────────────────────────
# TAB 4 — ADAPTIVE BOTS
# ────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Adaptive Pricing Strategies")

    # ── Q-learning ──────────────────────────────────────────────
    with st.expander("Q-learning bot", expanded=True):
        st.markdown(
            "**Q-learning**: discretised (price bucket × demand bucket) state space, "
            "ε-greedy exploration with linear schedule, TD(0) updates. "
            "Trained from scratch in a noiseless version of the market, then deployed."
        )
        qc1, qc2 = st.columns(2)
        q_episodes  = qc1.slider("Training episodes",    20,  200, 80,  10, key="q_ep")
        q_ep_steps  = qc1.slider("Steps per episode",    50,  500, 200, 10, key="q_es")
        q_n_actions = qc2.slider("Price action bins",     5,  11,  7,   2,  key="q_na")
        q_n_pb      = qc2.slider("Price state bins",      6,  20,  12,  2,  key="q_pb")

        if st.button("Train + deploy Q-bot", type="primary", key="run_q"):
            _n = min(n_consumers, 500)
            with st.spinner(f"Training {q_episodes} episodes…"):
                Q_table, deltas, rew_hist = train_q_bot(
                    n_consumers=_n, n_episodes=q_episodes,
                    episode_steps=q_ep_steps, distribution=distribution,
                    seed=seed, unit_cost=unit_cost,
                    n_price_bins=q_n_pb, n_demand_bins=10,
                    n_actions=q_n_actions,
                )
            with st.spinner("Deploying trained Q-bot…"):
                q_run = run_q_bot_simulation(
                    Q_table, deltas, n_consumers=_n, steps=steps,
                    distribution=distribution, seed=seed, unit_cost=unit_cost,
                    n_price_bins=q_n_pb, n_demand_bins=10,
                )
            st.session_state["q_run"]  = q_run
            st.session_state["q_hist"] = rew_hist
            st.session_state["q_n"]    = _n

        if "q_run" in st.session_state:
            q_run  = st.session_state["q_run"]
            q_hist = st.session_state["q_hist"]
            _n     = st.session_state.get("q_n", 500)
            naive  = run_simulation(r=r, n_consumers=_n, steps=steps,
                                    distribution=distribution, seed=seed, unit_cost=unit_cost)

            fig, axes = plt.subplots(1, 3, figsize=(14, 4))
            axes[0].plot(q_hist, lw=1.5, color="#2ca02c")
            axes[0].set_title("Training reward / step"); axes[0].set_xlabel("Episode")

            axes[1].plot(q_run["prices"], label="Q-bot",         lw=1.2, color="#2ca02c")
            axes[1].plot(naive["prices"], label=f"Fixed r={r:.2f}", lw=1.0, ls="--",
                         alpha=0.6, color="#999")
            axes[1].set_title("Price comparison"); axes[1].legend()

            axes[2].plot(q_run["cumulative_profit"], label="Q-bot",         lw=1.5, color="#2ca02c")
            axes[2].plot(naive["cumulative_profit"],  label=f"Fixed r={r:.2f}", lw=1.2, ls="--",
                         alpha=0.7, color="#999")
            axes[2].set_title("Cumulative profit"); axes[2].legend()

            for ax in axes:
                ax.set_xlabel("Step"); ax.grid(True, alpha=0.25)
            plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── MPC ─────────────────────────────────────────────────────
    with st.expander("MPC — Model Predictive Control", expanded=False):
        if not HAS_SCIPY:
            st.warning("MPC requires scipy.  `pip install scipy`")
        else:
            st.markdown(
                "**MPC**: at each step solve a K-step lookahead profit optimisation "
                r"with smoothness penalty $\lambda\sum_k(\Delta P_k)^2$. "
                "This directly suppresses the chaotic oscillations that emerge at high r, "
                "because the smoothness constraint acts as an effective chaos damper."
            )
            mc1, mc2 = st.columns(2)
            mpc_K      = mc1.slider("Lookahead horizon K", 1, 12, 6, 1, key="mpc_k")
            mpc_lambda = mc1.slider("Smoothness penalty λ", 0.0, 2.0, 0.15, 0.05, key="mpc_l")
            mpc_steps  = mc2.slider("Steps to simulate",   50, 400, 200, 10, key="mpc_s")

            if st.button("Run MPC bot", type="primary", key="run_mpc"):
                _n = min(n_consumers, 500)
                with st.spinner(f"Running MPC (K={mpc_K}, {mpc_steps} steps — may take ~8 s)…"):
                    mpc_res = run_mpc_simulation(
                        n_consumers=_n, steps=mpc_steps, K=mpc_K,
                        lambda_smooth=mpc_lambda,
                        distribution=distribution, seed=seed, unit_cost=unit_cost,
                    )
                if mpc_res:
                    st.session_state["mpc_res"] = mpc_res
                    st.session_state["mpc_n"]   = _n
                    st.session_state["mpc_s"]   = mpc_steps

            if "mpc_res" in st.session_state:
                mpc_res  = st.session_state["mpc_res"]
                _n       = st.session_state.get("mpc_n", 500)
                _s       = st.session_state.get("mpc_s", 200)
                naive_m  = run_simulation(r=r, n_consumers=_n, steps=_s,
                                          distribution=distribution, seed=seed, unit_cost=unit_cost)
                fig, axes = plt.subplots(1, 3, figsize=(14, 4))
                axes[0].plot(mpc_res["prices"], label="MPC",             lw=1.3, color="#ff7f0e")
                axes[0].plot(naive_m["prices"], label=f"Fixed r={r:.2f}", lw=1.0, ls="--",
                             alpha=0.55, color="#999")
                axes[0].set_title("Price comparison"); axes[0].legend()

                axes[1].plot(mpc_res["demands"], label="MPC",             lw=1.0, color="#ff7f0e")
                axes[1].plot(naive_m["demands"], label=f"Fixed r={r:.2f}", lw=1.0, ls="--",
                             alpha=0.55, color="#999")
                axes[1].set_title("Demand comparison"); axes[1].legend()

                axes[2].plot(mpc_res["cumulative_profit"], label="MPC",             lw=1.5, color="#ff7f0e")
                axes[2].plot(naive_m["cumulative_profit"],  label=f"Fixed r={r:.2f}", lw=1.2, ls="--",
                             alpha=0.7, color="#999")
                axes[2].set_title("Cumulative profit"); axes[2].legend()

                for ax in axes:
                    ax.set_xlabel("Step"); ax.grid(True, alpha=0.25)
                plt.tight_layout(); st.pyplot(fig); plt.close()

    # ── OGY Chaos Control ────────────────────────────────────────
    with st.expander("OGY Chaos Control", expanded=False):
        if not HAS_SCIPY:
            st.warning("OGY requires scipy.  `pip install scipy`")
        else:
            st.markdown(
                r"""
**OGY (Ott–Grebogi–Yorke) method**: Apply a tiny perturbation δr at each step to
steer the trajectory toward the unstable fixed point P\* that is embedded in the
chaotic attractor.  The correction is:

$$\delta r_t = -\frac{P_t - P^*}{\partial f/\partial r\big|_{P^*}} \qquad
\text{clamped to } [-\Delta r_{\max},\, \Delta r_{\max}]$$

The fixed point P\* is found by Brent's method.  The bot operates at a nominally
chaotic r yet produces near-stationary prices — *controllability through tiny nudges*.
"""
            )
            oc1, oc2 = st.columns(2)
            ogy_r_base  = oc1.slider("Base r  (chaotic regime)", 3.0, 4.5, 3.5, 0.05, key="ogy_r")
            ogy_dmax    = oc1.slider("Max perturbation Δr_max",  0.01, 0.5, 0.20, 0.01, key="ogy_d")
            ogy_steps   = oc2.slider("Steps to simulate",        100,  600, 300,  20,   key="ogy_s")

            if st.button("Run OGY control", type="primary", key="run_ogy"):
                _n = min(n_consumers, 500)
                with st.spinner("Finding fixed point + running OGY simulation…"):
                    ogy_res = run_ogy_simulation(
                        r_base=ogy_r_base, delta_r_max=ogy_dmax,
                        n_consumers=_n, steps=ogy_steps,
                        distribution=distribution, seed=seed, unit_cost=unit_cost,
                        update_rule=update_rule, d_target=d_target,
                    )
                    unc = run_simulation(
                        r=ogy_r_base, n_consumers=_n, steps=ogy_steps,
                        distribution=distribution, seed=seed,
                    )
                if ogy_res:
                    st.session_state["ogy_res"]  = ogy_res
                    st.session_state["ogy_unc"]  = unc

            if "ogy_res" in st.session_state:
                ogy_res  = st.session_state["ogy_res"]
                unc      = st.session_state["ogy_unc"]
                p_star   = ogy_res["p_star"]
                st.info(f"Fixed point  P* = **{p_star:.5f}**  (solved via Brent's method)")

                fig, axes = plt.subplots(2, 2, figsize=(12, 8))

                axes[0, 0].plot(unc["prices"],     lw=0.9,  color="#d62728", alpha=0.85,
                                label=f"Uncontrolled  r={ogy_r_base}")
                axes[0, 0].plot(ogy_res["prices"], lw=1.3,  color="#2ca02c", label="OGY-controlled")
                axes[0, 0].axhline(p_star, ls=":", color="black", lw=1.0, label=f"P*={p_star:.3f}")
                axes[0, 0].set_title("Price: uncontrolled vs OGY"); axes[0, 0].legend(fontsize=8)

                axes[0, 1].plot(ogy_res["r_used"], lw=0.8, color="#1f77b4")
                axes[0, 1].axhline(ogy_r_base, ls="--", color="gray", lw=0.8)
                axes[0, 1].set_title("Effective r  (OGY nudges)")
                axes[0, 1].set_ylim(ogy_r_base - ogy_dmax * 1.5, ogy_r_base + ogy_dmax * 1.5)

                axes[1, 0].plot(unc["cumulative_profit"],     lw=1.2, color="#d62728",
                                alpha=0.85, label="Uncontrolled")
                axes[1, 0].plot(ogy_res["cumulative_profit"], lw=1.2, color="#2ca02c",
                                label="OGY")
                axes[1, 0].set_title("Cumulative profit"); axes[1, 0].legend()

                axes[1, 1].hist(ogy_res["prices"][10:], bins=30, alpha=0.65,
                                label="OGY prices",          color="#2ca02c")
                axes[1, 1].hist(unc["prices"][10:],     bins=30, alpha=0.55,
                                label="Uncontrolled prices", color="#d62728")
                axes[1, 1].axvline(p_star, ls=":", color="black", lw=1.2)
                axes[1, 1].set_title("Price distribution"); axes[1, 1].legend()

                for ax in axes.flat:
                    ax.set_xlabel("Step"); ax.grid(True, alpha=0.25)
                plt.tight_layout(); st.pyplot(fig); plt.close()

                std_unc = np.std(unc["prices"][10:])
                std_ogy = np.std(ogy_res["prices"][10:])
                pct     = 100.0 * (1.0 - std_ogy / (std_unc + 1e-12))
                st.success(
                    f"Price std-dev:  uncontrolled **{std_unc:.4f}**  →  "
                    f"OGY-controlled **{std_ogy:.4f}**  "
                    f"({pct:.1f}% reduction in price variance)"
                )


# ────────────────────────────────────────────────────────────────
# TAB 5 — WELFARE ANALYSIS
# ────────────────────────────────────────────────────────────────
with tab5:
    st.subheader("Consumer Welfare & Fairness Metrics")
    wf   = compute_welfare_summary(result)
    wcols = st.columns(len(wf))
    for col, (k, v) in zip(wcols, wf.items()):
        col.metric(k, v)

    st.divider()
    w1, w2 = st.columns(2)
    with w1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["welfare_cs"], lw=1.2, color="#9467bd")
        ax.set_title("Consumer surplus per step"); ax.set_xlabel("Step"); ax.set_ylabel("CS")
        st.pyplot(fig); plt.close()
    with w2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["profit"],     lw=1.2,  color="#2ca02c", label="Seller profit")
        ax.plot(result["welfare_cs"], lw=1.0, ls="--", alpha=0.8, color="#9467bd",
                label="Consumer surplus")
        ax.set_title("Profit vs consumer surplus"); ax.set_xlabel("Step"); ax.legend()
        st.pyplot(fig); plt.close()

    # Total surplus decomposition (tail only)
    tail_cs  = float(np.sum(result["welfare_cs"][ts:]))
    tail_prf = float(np.sum(result["profit"][ts:]))
    total_s  = tail_cs + tail_prf
    if total_s > 0:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.bar(["Consumer surplus", "Seller profit"], [tail_cs, tail_prf],
               color=["#9467bd", "#2ca02c"], alpha=0.8, width=0.5)
        ax.set_title("Long-run surplus split  (tail 50%)"); ax.set_ylabel("Total value")
        st.pyplot(fig); plt.close()

    if use_segments:
        st.subheader("Welfare by Consumer Segment")
        segs_arr = result["segments"]
        wtp_arr  = result["base_wtp"]
        avg_p    = avg_price
        seg_rows = []
        for seg in ["price_sensitive", "mainstream", "loyal"]:
            mask     = segs_arr == seg
            wtp_seg  = wtp_arr[mask]
            buy_frac = float(np.mean(wtp_seg >= avg_p))
            avg_wtp  = float(np.mean(wtp_seg))
            cs_unit  = float(max(0.0, avg_wtp - avg_p) * buy_frac)
            seg_rows.append({
                "Segment":                      seg,
                "N":                            int(mask.sum()),
                "Avg WTP":                      round(avg_wtp, 3),
                "Buy fraction at avg price":    round(buy_frac, 3),
                "Est. avg CS per consumer":     round(cs_unit, 4),
            })
        st.dataframe(pd.DataFrame(seg_rows), use_container_width=True)

    st.markdown(
        """
---
**Metric guide**

| Metric | Interpretation |
|---|---|
| Price CoV | Standard deviation / mean of long-run prices. Higher = more erratic. Consumer-unfair. |
| Profit Gini | 0 = uniform profit across time. 1 = all profit concentrated in one step. |
| Consumer surplus | Value captured by buyers above the price they paid. |
| λ > 0 | Deterministic chaos — exponential sensitivity to initial price. |

**Distributional effect of chaos**: In chaotic regimes, high-WTP (loyal) consumers
always buy — even at price peaks — while low-WTP consumers can only buy during troughs.
This creates a **regressive distributional effect**: chaos transfers surplus from
price-sensitive to loyal segments and from consumers to the seller at price peaks.
"""
    )


# ────────────────────────────────────────────────────────────────
# TAB 6 — PROFIT OPTIMISATION
# ────────────────────────────────────────────────────────────────
with tab6:
    st.subheader("Profit Optimisation Over r")
    with st.expander("Scan settings", expanded=True):
        oc1, oc2, oc3 = st.columns(3)
        opt_rmin = oc1.slider("r min", 0.0, 4.5, 0.5, 0.1, key="opt_rmin")
        opt_rmax = oc2.slider("r max", 0.1, 4.5, 4.0, 0.1, key="opt_rmax")
        opt_nr   = oc3.slider("# r values", 20, 200, 80, 5,  key="opt_nr")

    if st.button("Run optimisation scan", type="primary", key="run_opt"):
        if opt_rmax <= opt_rmin:
            st.error("r max must exceed r min.")
        else:
            with st.spinner(f"Scanning {opt_nr} values of r…"):
                ranking = optimize_r(
                    r_min=opt_rmin, r_max=opt_rmax, num_r=opt_nr,
                    n_consumers=n_consumers, steps=steps, initial_price=init_price,
                    distribution=distribution, seed=seed,
                    dynamic_wtp=dynamic_wtp, wtp_noise=wtp_noise,
                    use_seasonality=use_seasonality,
                    season_amplitude=season_amplitude, season_period=season_period,
                    unit_cost=unit_cost, fixed_cost_per_step=fixed_cost_per_step,
                    update_rule=update_rule, d_target=d_target,
                )
                st.session_state["ranking"] = ranking

    if "ranking" in st.session_state:
        ranking = st.session_state["ranking"]
        best    = ranking[0]
        st.success(
            f"Best r = **{best['r']:.4f}**  ·  regime: {best['regime'].split('(')[0].strip()}  ·  "
            f"avg long-run profit: **{best['avg_profit_long_run']:.2f}**  ·  λ = {best['lyapunov']:.4f}"
        )

        ranking_sorted = sorted(ranking, key=lambda x: x["r"])
        rs    = [row["r"]                   for row in ranking_sorted]
        profs = [row["avg_profit_long_run"] for row in ranking_sorted]
        vols  = [row["price_volatility"]    for row in ranking_sorted]
        lyaps = [row["lyapunov"]            for row in ranking_sorted]
        ly_np = np.array(lyaps)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(rs, profs, lw=1.5, color="#1f77b4")
        axes[0].axvline(best["r"], color="red", ls="--", lw=1.0, label=f"Best r={best['r']:.3f}")
        axes[0].set_title("Avg long-run profit vs r"); axes[0].set_xlabel("r")
        axes[0].legend()

        axes[1].plot(rs, vols, lw=1.5, color="#d62728")
        axes[1].set_title("Price volatility vs r"); axes[1].set_xlabel("r")

        axes[2].plot(rs, lyaps, lw=1.2, color="#333333")
        axes[2].axhline(0, color="black", ls="--", lw=0.8)
        axes[2].fill_between(rs, lyaps, 0, where=ly_np  > 0, alpha=0.18, color="#d62728")
        axes[2].fill_between(rs, lyaps, 0, where=ly_np <= 0, alpha=0.12, color="#2ca02c")
        axes[2].set_title("Lyapunov λ vs r  (red = chaotic)"); axes[2].set_xlabel("r")

        for ax in axes:
            ax.grid(True, alpha=0.25)
        plt.tight_layout(); st.pyplot(fig); plt.close()

        st.markdown(f"### Top {min(15, len(ranking))} r values by long-run profit")
        st.dataframe(pd.DataFrame(ranking[:15]), use_container_width=True)

        # Best-r re-simulation
        best_run = run_simulation(
            r=best["r"], n_consumers=n_consumers, steps=steps,
            initial_price=init_price, distribution=distribution, seed=seed,
            dynamic_wtp=dynamic_wtp, wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude, season_period=season_period,
            unit_cost=unit_cost, fixed_cost_per_step=fixed_cost_per_step,
            update_rule=update_rule, d_target=d_target,
        )
        bc1, bc2 = st.columns(2)
        with bc1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(best_run["prices"], lw=1.3, color="#1f77b4")
            ax.set_title(f"Price time series at best r = {best['r']:.4f}")
            ax.set_xlabel("Step"); ax.set_ylabel("Price")
            st.pyplot(fig); plt.close()
        with bc2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(best_run["cumulative_profit"], lw=1.5, color="#2ca02c")
            ax.set_title(f"Cumulative profit at best r = {best['r']:.4f}")
            ax.set_xlabel("Step"); ax.set_ylabel("Cumulative profit")
            st.pyplot(fig); plt.close()

# ────────────────────────────────────────────────────────────────
# TAB 7 — BEST r FOR GIVEN PRICE & COST
# ────────────────────────────────────────────────────────────────
with tab7:
    st.subheader("Find the Best r for a Given Starting Price and Cost Structure")
    st.markdown(
        "This tool fixes the **starting price** and **cost inputs**, then sweeps over "
        "`r` values to find the one that gives the best long-run profit."
    )

    g1, g2, g3 = st.columns(3)
    with g1:
        given_price = st.slider(
            "Given starting price",
            0.01, 1.5, float(init_price), 0.01,
            key="given_price_r_search"
        )
    with g2:
        given_unit_cost = st.slider(
            "Given unit cost",
            0.0, 0.5, float(unit_cost), 0.01,
            key="given_unit_cost_r_search"
        )
    with g3:
        given_fixed_cost = st.slider(
            "Given fixed cost / step",
            0.0, 50.0, float(fixed_cost_per_step), 1.0,
            key="given_fixed_cost_r_search"
        )

    s1, s2, s3 = st.columns(3)
    with s1:
        r_search_min = st.slider("r search min", 0.0, 4.5, 0.5, 0.05, key="r_search_min")
    with s2:
        r_search_max = st.slider("r search max", 0.1, 4.5, 4.0, 0.05, key="r_search_max")
    with s3:
        r_search_n = st.slider("# r values to test", 20, 250, 100, 5, key="r_search_n")

    if st.button("Find best r for this price + cost setup", type="primary", key="run_best_r_fixed"):
        if r_search_max <= r_search_min:
            st.error("r search max must be greater than r search min.")
        else:
            with st.spinner("Searching for best r..."):
                ranking_fixed, best_fixed, best_fixed_run = find_best_r_for_given_price_cost(
                    given_price=given_price,
                    given_unit_cost=given_unit_cost,
                    given_fixed_cost=given_fixed_cost,
                    r_min=r_search_min,
                    r_max=r_search_max,
                    num_r=r_search_n,
                    n_consumers=n_consumers,
                    steps=steps,
                    distribution=distribution,
                    seed=seed,
                    dynamic_wtp=dynamic_wtp,
                    wtp_noise=wtp_noise,
                    ou_theta=ou_theta,
                    use_seasonality=use_seasonality,
                    season_amplitude=season_amplitude,
                    season_period=season_period,
                    use_reference_price=use_ref_price,
                    ref_alpha=ref_alpha,
                    ref_beta=ref_beta,
                    use_churn=use_churn,
                    churn_patience=churn_patience,
                    reentry_rate=reentry_rate,
                    use_segments=use_segments,
                    update_rule=update_rule,
                    d_target=d_target,
                )
                st.session_state["ranking_fixed"] = ranking_fixed
                st.session_state["best_fixed"] = best_fixed
                st.session_state["best_fixed_run"] = best_fixed_run
                st.session_state["given_price_fixed"] = given_price
                st.session_state["given_unit_cost_fixed"] = given_unit_cost
                st.session_state["given_fixed_cost_fixed"] = given_fixed_cost

    if "ranking_fixed" in st.session_state:
        ranking_fixed = st.session_state["ranking_fixed"]
        best_fixed = st.session_state["best_fixed"]
        best_fixed_run = st.session_state["best_fixed_run"]

        saved_given_price = st.session_state.get("given_price_fixed", given_price)
        saved_given_unit_cost = st.session_state.get("given_unit_cost_fixed", given_unit_cost)
        saved_given_fixed_cost = st.session_state.get("given_fixed_cost_fixed", given_fixed_cost)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Best r", f"{best_fixed['r']:.4f}")
        c2.metric("Avg long-run profit", f"{best_fixed['avg_profit_long_run']:.4f}")
        c3.metric("Total profit", f"{best_fixed['total_profit']:.4f}")
        c4.metric("Regime", best_fixed["regime"].split("(")[0].strip())

        st.success(
            f"For starting price = **{saved_given_price:.2f}**, unit cost = **{saved_given_unit_cost:.2f}**, "
            f"and fixed cost/step = **{saved_given_fixed_cost:.2f}**, the best r is "
            f"**{best_fixed['r']:.4f}**."
        )

        ranking_fixed_sorted = sorted(ranking_fixed, key=lambda x: x["r"])
        rs_fixed = [row["r"] for row in ranking_fixed_sorted]
        profs_fixed = [row["avg_profit_long_run"] for row in ranking_fixed_sorted]
        vols_fixed = [row["price_volatility"] for row in ranking_fixed_sorted]
        lyaps_fixed = [row["lyapunov"] for row in ranking_fixed_sorted]
        lyaps_fixed_np = np.array(lyaps_fixed)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        axes[0].plot(rs_fixed, profs_fixed, lw=1.5, color="#1f77b4")
        axes[0].axvline(best_fixed["r"], color="red", ls="--", lw=1.0, label=f"Best r = {best_fixed['r']:.4f}")
        axes[0].set_title("Avg long-run profit vs r")
        axes[0].set_xlabel("r")
        axes[0].set_ylabel("Avg long-run profit")
        axes[0].legend()

        axes[1].plot(rs_fixed, vols_fixed, lw=1.5, color="#d62728")
        axes[1].axvline(best_fixed["r"], color="red", ls="--", lw=1.0)
        axes[1].set_title("Price volatility vs r")
        axes[1].set_xlabel("r")
        axes[1].set_ylabel("Price volatility")

        axes[2].plot(rs_fixed, lyaps_fixed, lw=1.2, color="#333333")
        axes[2].axhline(0, color="black", ls="--", lw=0.8)
        axes[2].axvline(best_fixed["r"], color="red", ls="--", lw=1.0)
        axes[2].fill_between(rs_fixed, lyaps_fixed, 0, where=lyaps_fixed_np > 0, alpha=0.18, color="#d62728")
        axes[2].fill_between(rs_fixed, lyaps_fixed, 0, where=lyaps_fixed_np <= 0, alpha=0.12, color="#2ca02c")
        axes[2].set_title("Lyapunov exponent vs r")
        axes[2].set_xlabel("r")
        axes[2].set_ylabel("λ")

        for ax in axes:
            ax.grid(True, alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

        b1, b2 = st.columns(2)
        with b1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(best_fixed_run["prices"], lw=1.3, color="#1f77b4")
            ax.set_title(f"Price path at best r = {best_fixed['r']:.4f}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Price")
            st.pyplot(fig)
            plt.close()

        with b2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(best_fixed_run["cumulative_profit"], lw=1.4, color="#2ca02c")
            ax.set_title("Cumulative profit at best r")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative profit")
            st.pyplot(fig)
            plt.close()

        st.markdown(f"### Top {min(15, len(ranking_fixed))} r values for this fixed price/cost setup")
        st.dataframe(pd.DataFrame(ranking_fixed[:15]), use_container_width=True)
# ════════════════════════════════════════════════════════════════
# RUN INSTRUCTIONS
# ════════════════════════════════════════════════════════════════
st.divider()
st.subheader("How to run locally")
st.code(
    "pip install streamlit numpy matplotlib pandas scipy\n"
    "streamlit run app.py",
    language="bash",
)
