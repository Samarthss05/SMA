# Algorithmic Pricing Simulation + Profit Optimization

## Overview

This project is an interactive **Streamlit-based simulation app** that models a simple algorithmic pricing market and evaluates the resulting **dynamics, volatility, and profitability**.

We simulate a feedback-based algorithmic pricing market, study how the aggressiveness parameter changes system dynamics, and then optimize that parameter for long-run profit while observing the associated volatility and stability trade-offs.

The core idea is that a pricing bot repeatedly updates its price based on observed demand. Consumers decide whether to buy depending on whether the published price is below their personal willingness-to-pay (WTP). Over time, this creates a feedback loop:

- the bot sets a price
- consumers react to that price
- demand is measured
- the bot uses that demand to update the next price

Because the pricing update is nonlinear, the system can show different behaviors depending on the value of the aggressiveness parameter `r`:

- stable convergence
- cyclical oscillations
- irregular or chaos-like fluctuations

This version of the project also adds **profit analysis** and an **optimization routine** that searches for the value of `r` that produces the highest long-run profit under the model assumptions.

---

## Main Objective

The app helps answer two key questions:

1. **How does a nonlinear pricing rule affect market behavior over time?**
2. **Which value of the pricing aggressiveness parameter `r` gives the best profit outcome?**

So this is not just a simulation of price movement. It is also a tool for analyzing whether more aggressive pricing improves profit or instead creates instability that reduces performance.

---

## Core Model Logic

At each time step:

1. The pricing bot publishes a price `P_t`
2. Each consumer compares that price to their own willingness-to-pay
3. If `P_t <= WTP`, the consumer buys
4. Demand is computed as the fraction of consumers who buy
5. The bot updates the next price using the rule:

\[
P_{t+1} = r \cdot P_t \cdot Demand_t
\]

Where:

- `P_t` = price at time step `t`
- `Demand_t` = fraction of consumers who buy at time step `t`
- `r` = pricing aggressiveness parameter

This means the next price depends on:
- the current price
- how much demand that price generated
- how aggressively the bot reacts to demand

---

## Economic Intuition

This model captures a simple feedback loop often seen in algorithmic markets:

- when demand is strong, the bot may increase price
- when demand is weak, the bot may decrease price
- if the bot reacts too aggressively, the system may overshoot and become unstable

So the project is useful for exploring the trade-off between:

- responsiveness
- stability
- volatility
- profitability

---

## Features of the App

The app includes the following major features:

### 1. Interactive Simulation
You can choose model parameters such as:
- aggressiveness parameter `r`
- number of consumers
- number of time steps
- initial price
- WTP distribution
- random seed

This allows you to test how the market behaves under different assumptions.

### 2. Multiple Consumer WTP Distributions
Consumers' willingness-to-pay can be generated using:
- **Uniform distribution**
- **Normal distribution**
- **Beta distribution**

This lets you model different kinds of customer populations.

### 3. Optional Dynamic WTP
You can allow willingness-to-pay to evolve over time using random noise. This makes the market more realistic by allowing customer preferences to shift gradually.

### 4. Optional Seasonality
You can apply a seasonal multiplier so that demand sensitivity changes over time. This simulates periodic fluctuations such as:
- weekly demand cycles
- holiday effects
- promotional seasons

### 5. Profit Analysis
The app calculates:
- revenue
- cost
- profit per time step
- cumulative profit
- long-run average profit
- total profit

### 6. Optimization of `r`
The app scans across a range of `r` values and ranks them based on **average long-run profit**. It then reports the best-performing `r`.

### 7. Regime Detection
The app uses a simple heuristic to classify the observed price behavior as:
- Stable / Convergent
- Cyclical / Periodic
- Chaos-like / Irregular

### 8. Visualization
The app generates multiple plots to help interpret the system:
- price time series
- demand time series
- profit per step
- cumulative profit
- WTP histogram
- profit vs price scatter plot
- bifurcation-style plot
- long-run profit vs `r`
- volatility vs `r`
- best-`r` simulation output

---

## File Structure

This project currently contains the main application file:

- `pricing_profit_optimizer.py`

You can also create this README file and keep it in the same folder as the Python script.

---

## How the Code Works

## 1. Import Section

The code imports three core libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
```

### Why these are used:
- `numpy` is used for numerical simulation and array-based calculations
- `matplotlib` is used for plotting charts
- `streamlit` is used to build the interactive web app interface

---

## 2. Streamlit Page Setup

The app begins with:

```python
st.set_page_config(page_title="Algorithmic Pricing Profit Optimizer", layout="wide")
```

This sets:
- the app title shown in the browser tab
- a wide layout so charts and controls are easier to view

The title and markdown below it explain the purpose of the application to the user.

---

## 3. WTP Generation Function

### Function:
```python
generate_wtp(n, distribution="uniform", seed=42)
```

### Purpose:
This function creates the willingness-to-pay values for all consumers.

### Inputs:
- `n`: number of consumers
- `distribution`: the type of probability distribution
- `seed`: random seed for reproducibility

### Output:
An array of WTP values between 0 and 1.

### Supported distributions:
- **Uniform**: spreads values evenly between 0 and 1
- **Normal**: centered around a mean, clipped to stay in range
- **Beta**: flexible bounded distribution useful for skewed demand

### Why it matters:
Different WTP distributions lead to different demand responses and therefore different price dynamics.

---

## 4. Seasonal Multiplier Function

### Function:
```python
seasonal_multiplier(t, amplitude=0.0, period=30)
```

### Purpose:
This function creates a repeating seasonal adjustment using a sine wave.

### Formula:
\[
1 + A \sin\left(\frac{2\pi t}{T}\right)
\]

Where:
- `A` = amplitude
- `T` = period

### Why it matters:
If seasonality is enabled, the effective price sensitivity of consumers changes over time. This adds a realistic time-varying pattern to demand.

---

## 5. Main Simulation Function

### Function:
```python
run_simulation(...)
```

This is the heart of the project.

### Inputs:
- `r`: pricing aggressiveness
- `n_consumers`: number of consumers
- `steps`: number of time periods
- `initial_price`: starting price
- `distribution`: WTP distribution type
- `seed`: random seed
- `dynamic_wtp`: whether WTP changes over time
- `wtp_noise`: amount of random WTP movement
- `use_seasonality`: whether seasonality is enabled
- `season_amplitude`: seasonal strength
- `season_period`: seasonal cycle length
- `clip_price`: whether prices are kept within a bounded range
- `unit_cost`: variable cost per unit sold
- `fixed_cost_per_step`: fixed cost incurred each period

### What happens inside:
The function:
1. generates the initial WTP population
2. stores arrays for price, demand, buyers, revenue, cost, and profit
3. loops across all time steps
4. calculates who buys
5. computes demand
6. computes revenue and cost
7. computes profit
8. updates the next price using the nonlinear pricing rule
9. optionally modifies WTP over time

### Revenue Formula
\[
Revenue_t = Price_t \times UnitsSold_t
\]

### Cost Formula
\[
Cost_t = UnitCost \times UnitsSold_t + FixedCostPerStep
\]

### Profit Formula
\[
Profit_t = Revenue_t - Cost_t
\]

### Why this function is important:
It transforms the pricing rule into a full market simulation and produces all the output data used in the app.

---

## 6. Regime Classification Function

### Function:
```python
classify_regime(prices, transient_fraction=0.5)
```

### Purpose:
This function attempts to classify the long-run behavior of the price series.

### Logic used:
It ignores the early transient portion of the price path, then examines:
- the standard deviation of the tail
- the number of distinct rounded values in the tail

### Output categories:
- **Stable / Convergent**: price settles near one level
- **Cyclical / Periodic**: price repeats among a few values
- **Chaos-like / Irregular**: price behaves in a complex, unstable way

### Important note:
This is a heuristic, not a rigorous chaos test. It is designed for interpretation and demonstration purposes.

---

## 7. Bifurcation Plot Function

### Function:
```python
compute_bifurcation(...)
```

### Purpose:
This function generates data for a bifurcation-style plot.

### What it does:
- loops over many values of `r`
- runs the simulation for each one
- keeps only the long-run tail of the price series
- plots those tail values against `r`

### Why it matters:
This helps visualize the transition from:
- stable behavior
- to cycles
- to complex or fragmented dynamics

It is one of the strongest visuals in the project because it summarizes how market behavior changes across different pricing aggressiveness levels.

---

## 8. Optimization Function

### Function:
```python
optimize_r(...)
```

### Purpose:
This function searches across a grid of `r` values and ranks them by profit.

### What it computes for each `r`:
- long-run average profit
- total profit
- price volatility
- average price
- average demand
- regime classification

### Output:
A sorted ranking from best to worst based on **average long-run profit**.

### Why long-run average profit?
Because early time steps may be unstable or transitional. Long-run average profit focuses on the sustained behavior of the system rather than only the starting phase.

---

## User Interface Layout

The Streamlit sidebar lets the user control the simulation.

## A. Main Simulation Controls
These define the core market setup:
- `r`
- number of consumers
- number of time steps
- initial price
- WTP distribution
- random seed

## B. Profit Assumptions
These define the business side:
- `unit cost`
- `fixed cost per step`

## C. Optional Extensions
These allow realism to be added:
- dynamic WTP
- WTP noise
- seasonality
- seasonal amplitude
- seasonal period

## D. Optimization Controls
These define the search range for `r`:
- minimum `r`
- maximum `r`
- number of grid points
- button to run the optimization

---

## Key Outputs Shown in the App

## 1. KPI Summary Cards
These show:
- regime classification
- price volatility
- average long-run price
- average long-run demand
- average long-run profit
- total profit

These give an instant summary of the system state.

---

## 2. Price Time Series
This chart shows how the price evolves over time.

### What to look for:
- does the price settle?
- does it oscillate?
- does it become erratic?

---

## 3. Demand Time Series
This shows the fraction of consumers buying at each step.

### What to look for:
- stable demand
- repeating demand cycles
- large swings in demand

---

## 4. Profit Per Time Step
This shows profitability on each step individually.

### What to look for:
- consistent profits
- sharp losses
- unstable earnings patterns

---

## 5. Cumulative Profit
This shows total accumulated profit over time.

### What to look for:
- steady upward growth
- flattening profit
- drawdowns or poor long-run performance

---

## 6. WTP Distribution Histogram
This visualizes the consumer population.

### Why it matters:
It helps explain why demand behaves the way it does. For example, if many consumers have low WTP, the system may struggle to support high prices.

---

## 7. Profit vs Published Price Scatter Plot
This shows the relationship between price and profit.

### Why it matters:
It helps identify whether higher prices are consistently better or whether there is an intermediate region that performs best.

---

## 8. Bifurcation-Style Plot
This shows long-run price behavior across a full range of `r` values.

### Why it matters:
It visually demonstrates regime changes across the control parameter.

---

## 9. Optimization Results Table
When the user clicks **Find best r**, the app shows:
- the best `r`
- average long-run profit at that `r`
- total profit
- regime type

It also displays a table of the top-ranked `r` values.

---

## 10. Profit vs r Plot
This chart shows how average long-run profit changes as `r` changes.

### Why it matters:
It directly answers the optimization question:
**which pricing aggressiveness maximizes long-run profit?**

---

## 11. Volatility vs r Plot
This shows how price volatility changes as `r` changes.

### Why it matters:
It helps compare profitability and stability at the same time.

Sometimes the most profitable `r` may also create more volatility.

---

## 12. Best-r Simulation Output
The app reruns the simulation using the best `r` found and shows:
- price time series
- cumulative profit

This helps the user visually verify what the optimal result looks like.

---

## Interpretation of Fixed Cost Per Step

`fixed_cost_per_step` represents the cost incurred at every simulation period regardless of how many units are sold.

Examples:
- server costs
- software operating cost
- staff/admin overhead per day
- platform maintenance cost

It is different from `unit_cost`, which only applies when units are sold.

### Example:
If:
- price = 0.60
- units sold = 100
- unit cost = 0.20
- fixed cost per step = 10

Then:

\[
Profit = (0.60 \times 100) - (0.20 \times 100) - 10 = 30
\]

---

## Suggested Workflow for Using the App

A good way to use the app is:

### Step 1
Set a baseline market:
- 1000 consumers
- 250 steps
- initial price 0.5
- uniform WTP
- unit cost 0.2
- fixed cost 0

### Step 2
Try different values of `r` manually:
- low `r` for stability
- medium `r` for oscillation
- high `r` for irregularity

### Step 3
Observe:
- price dynamics
- demand dynamics
- profit per step
- cumulative profit

### Step 4
Run the optimization scan over `r`

### Step 5
Compare:
- best profit
- regime type
- volatility level

### Step 6
Interpret the trade-off:
- is the most profitable policy also the most stable?
- does higher aggressiveness create more volatility?
- is there a sweet spot where profit is high but instability is still manageable?

---

## Strengths of This Model

This model is useful because it is:

- simple enough to understand
- nonlinear enough to produce interesting dynamics
- visually interpretable
- interactive
- extendable
- suitable for demonstrations, coursework, and experiments

It combines ideas from:
- agent-based simulation
- pricing strategy
- nonlinear dynamics
- optimization
- business profitability analysis

---

## Limitations

This model is intentionally simplified.

Some limitations include:
- consumers only use a basic buy/no-buy rule
- no competitors are included
- no inventory constraints are modeled
- no learning or memory effects are included for consumers
- regime detection is heuristic, not mathematically rigorous
- optimization is grid-based, not an advanced continuous optimization method

These simplifications are acceptable for a project or proof-of-concept, but they should be acknowledged.

---

## Possible Extensions

Here are some good improvements you can add later:

### Market realism
- multiple competing pricing bots
- brand loyalty
- consumer memory
- delayed demand response
- inventory depletion
- stockouts

### Optimization realism
- optimize profit subject to a volatility cap
- optimize total profit instead of long-run average profit
- risk-adjusted objective function
- multi-objective optimization

### Mathematical rigor
- Lyapunov exponent estimation
- better periodicity detection
- more rigorous chaos diagnostics

### Business realism
- discounting
- marketing costs
- per-customer segmentation
- promotion campaigns
- retention-based demand

---

## Installation

Install the required packages:

```bash
pip install streamlit numpy matplotlib
```

---

## Running the App

Run the following command in the terminal:

```bash
streamlit run pricing_profit_optimizer.py
```

Then open the local Streamlit URL shown in the terminal.

---

## Recommended Demo Parameters

A clean demo setup is:

- `N = 1000`
- `P0 = 0.5`
- `steps = 250`
- `unit_cost = 0.20`
- `fixed_cost_per_step = 0`
- optimization range `r = 0.1` to `4.0`

This setup is usually good enough to demonstrate:
- stable behavior at low `r`
- oscillation at medium `r`
- irregularity at higher `r`
- an identifiable best `r` for profit

---

## Summary

This project is a nonlinear algorithmic pricing simulator with profit analysis and parameter optimization.

In simple terms, it does four things:

1. simulates how a pricing bot updates price based on demand
2. shows how consumer demand reacts to those prices
3. calculates revenue, cost, profit, and cumulative profit
4. finds the value of `r` that maximizes long-run profit under the model assumptions

It is both:
- a simulation of complex market dynamics
- and a decision-support tool for choosing a pricing aggressiveness level

---

## Author Notes

This README is designed to help:
- understand the code in detail
- explain the project in class
- document the logic for submissions
- make the app easier to extend later


