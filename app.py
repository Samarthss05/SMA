import json
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import streamlit.components.v1 as components
from textwrap import dedent


# Initialize Page 
st.set_page_config(
    page_title="ABM Pricing Bot — Optimal r",
    layout="wide",
    initial_sidebar_state="expanded",
)

plt.rcParams.update({
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 110,
})

def apply_custom_ui():
    st.markdown(
        dedent(
            """
            <style>
                .stApp {
                    background: #0b0b0b;
                }

                [data-testid="stSidebar"] {
                    background: #050505;
                    border-right: 1px solid rgba(255,255,255,0.08);
                }

                .block-container {
                    max-width: 1400px;
                    padding-top: 1.1rem;
                    padding-bottom: 2rem;
                }

                h1, h2, h3 {
                    letter-spacing: -0.02em;
                }

                .hero-card {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 18px;
                    padding: 1rem 1.15rem;
                    margin: 0.4rem 0 1rem 0;
                }

                .small-note {
                    color: rgba(255,255,255,0.72);
                    font-size: 0.95rem;
                    line-height: 1.5;
                }

                div[data-testid="stMetric"] {
                    background: rgba(255,255,255,0.03);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 14px;
                    padding: 0.7rem 0.85rem;
                    min-height: 125px;
                }

                div[data-testid="stMetricLabel"] > div {
                    white-space: normal !important;
                    word-break: break-word;
                    line-height: 1.2rem;
                    font-size: 0.94rem;
                }

                div[data-testid="stMetricValue"] {
                    font-size: 2.05rem;
                    line-height: 1.05;
                }

                .stTabs [data-baseweb="tab-list"] {
                    gap: 0.45rem;
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
                    border-color: rgba(255,255,255,0.18) !important;
                }

                .stButton > button {
                    border-radius: 12px;
                    font-weight: 600;
                }
            </style>
            """
        ),
        unsafe_allow_html=True,
    )

apply_custom_ui()

st.title("ABM Pricing Bot: Finding the Optimal Aggressiveness `r`")
st.markdown(
    """
    <div class="hero-card">
        <div style="font-size:1.05rem; font-weight:700; margin-bottom:0.3rem;">
            An agent-based model for dynamic pricing
        </div>
        <div class="small-note">
            This application hopes to discover how pricing aggressiveness parameter <code>r</code> affects
            long-run profit, volatility, demand stability, and consumer welfare.
            <br><br>
            The pricing bot updates according to:
            <code>P(t+1) = r · P(t) · D(t)</code>
            <br><br>
            Under a very simple demand assumption, the equation reduces to the logistic map.<br>
            To model the randomness in the world, our ABM becomes a more realistic nonlinear pricing system with
            heterogeneous consumers, seasonality, and noisy willingness-to-pay.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
######################################################################################

def generate_segmented_market(n_consumers: int, seed: int = 42):
    rng = np.random.default_rng(seed)

    n_sensitive = int(0.50 * n_consumers) # 50% of customers are senstive to price changes, meaning if it increased from 0.1 to 0.2 they will consider
    n_mainstream = int(0.35 * n_consumers) # 35% are mainstream customers, see, they like, then they buy. Else its chill
    n_loyal = n_consumers - n_sensitive - n_mainstream # regardless of my price, i will buy 

    wtp = np.concatenate([
        rng.beta(1.5, 4.0, n_sensitive),
        np.clip(rng.normal(0.55, 0.15, n_mainstream), 0.0, 1.0),
        np.clip(rng.normal(0.80, 0.08, n_loyal), 0.0, 1.0),
    ])

    segments = np.array(
        ["price_sensitive"] * n_sensitive +
        ["mainstream"] * n_mainstream +
        ["loyal"] * n_loyal,
        dtype=object
    )

    return wtp, segments


def seasonal_multiplier(t: int, amplitude: float = 0.0, period: int = 30):
    return 1.0 + amplitude * np.sin(2 * np.pi * t / period)


######################################################################################

def build_agent_grid(n_agents: int):
    cols = int(np.ceil(np.sqrt(n_agents)))
    rows = int(np.ceil(n_agents / cols))
    xs, ys = [], []
    for idx in range(n_agents):
        xs.append(idx % cols)
        ys.append(-(idx // cols))
    return np.array(xs, dtype=float), np.array(ys, dtype=float)



def build_abm_dashboard_html(result, width=1200, height=1380):
    agent_viz = result["agent_viz"]
    prices = np.asarray(result["prices"][:-1], dtype=float)
    demands = np.asarray(result["demands"], dtype=float)
    profits = np.asarray(result["profit"], dtype=float)
    buy_history = np.asarray(agent_viz["buy_history"], dtype=int)
    wtp_history = np.asarray(agent_viz["wtp_history"], dtype=float)
    sample_segments = np.asarray(agent_viz["sample_segments"], dtype=object)
    sample_x = np.asarray(agent_viz["sample_x"], dtype=float)
    sample_y = np.asarray(agent_viz["sample_y"], dtype=float)

    segment_labels = sample_segments.tolist()
    xs = sample_x.tolist()
    ys = sample_y.tolist()

    segment_colors = {
        "price_sensitive": "#ef4444",
        "mainstream": "#3b82f6",
        "loyal": "#22c55e",
    }

    frames = []
    n_steps = min(len(demands), buy_history.shape[0], wtp_history.shape[0], len(prices), len(profits))

    for i in range(n_steps):
        buy_now = buy_history[i].astype(int)
        wtp_now = wtp_history[i]

        seg_summary = []
        for seg in ["price_sensitive", "mainstream", "loyal"]:
            mask = sample_segments == seg
            n_seg = int(np.sum(mask))
            if n_seg == 0:
                seg_summary.append({
                    "segment": seg,
                    "count": 0,
                    "buyers": 0,
                    "buy_rate": 0.0,
                    "avg_wtp": 0.0,
                })
            else:
                buyers = int(np.sum(buy_now[mask]))
                seg_summary.append({
                    "segment": seg,
                    "count": n_seg,
                    "buyers": buyers,
                    "buy_rate": float(buyers / n_seg),
                    "avg_wtp": float(np.mean(wtp_now[mask])),
                })

        # which segment reacts most? = lowest buy rate at this price, among segments with members
        valid_seg = [row for row in seg_summary if row["count"] > 0]
        reacting_segment = min(valid_seg, key=lambda row: row["buy_rate"])["segment"] if valid_seg else "mainstream"

        # stability signal from recent price oscillation
        start = max(0, i - 24)
        recent_prices = prices[start:i + 1]
        if len(recent_prices) >= 8:
            recent_std = float(np.std(recent_prices))
            recent_mean = float(np.mean(recent_prices))
            recent_cv = recent_std / (recent_mean + 1e-12)
            if recent_cv < 0.02:
                regime_text = "Stable"
            elif recent_cv < 0.06:
                regime_text = "Mildly oscillating"
            else:
                regime_text = "Clearly oscillating"
        else:
            regime_text = "Building pattern"

        if reacting_segment == "price_sensitive":
            interpretation = "Price-sensitive consumers are reacting the most at this step."
        elif reacting_segment == "mainstream":
            interpretation = "Mainstream consumers are reacting the most at this step."
        else:
            interpretation = "Loyal consumers are reacting the most at this step."

        frames.append({
            "step": int(i),
            "price": float(prices[i]),
            "demand": float(demands[i]),
            "profit": float(profits[i]),
            "buyers": int(np.sum(buy_now)),
            "buy_now": buy_now.tolist(),
            "wtp_now": [float(v) for v in wtp_now.tolist()],
            "segment_summary": seg_summary,
            "reacting_segment": reacting_segment,
            "regime_text": regime_text,
            "interpretation": interpretation,
        })

    payload = {
        "frames": frames,
        "segment_labels": segment_labels,
        "segment_colors": segment_colors,
        "x": xs,
        "y": ys,
        "prices": [float(v) for v in prices.tolist()],
        "demands": [float(v) for v in demands.tolist()],
        "profits": [float(v) for v in profits.tolist()],
    }

    data_json = json.dumps(payload)
    max_step = max(0, len(frames) - 1)

    html = f"""
    <div id="abm-root"></div>
    <script>
    const ABM_DATA = {data_json};

    const root = document.getElementById('abm-root');
    root.innerHTML = `
      <style>
        :root {{
          --bg: #050505;
          --panel: #0b0f19;
          --panel2: #0f172a;
          --border: rgba(255,255,255,0.10);
          --muted: rgba(255,255,255,0.72);
          --text: #f8fafc;
          --accent: #38bdf8;
        }}
        * {{ box-sizing: border-box; }}
        body {{ margin: 0; background: transparent; color: var(--text); font-family: Inter, system-ui, sans-serif; }}
        .wrap {{ background: transparent; color: var(--text); padding: 4px 0 18px 0; max-width: 100%; overflow: hidden; }}
        .intro {{ color: var(--muted); font-size: 15px; line-height: 1.6; margin-bottom: 14px; }}
        .stats {{ display: grid; grid-template-columns: repeat(5, minmax(0, 1fr)); gap: 12px; margin-bottom: 14px; }}
        .card {{ background: linear-gradient(180deg, #0b1120, #0a0a0a); border: 1px solid var(--border); border-radius: 16px; padding: 14px 16px; min-height: 96px; box-shadow: 0 6px 20px rgba(0,0,0,0.25); }}
        .card .label {{ font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
        .card .value {{ font-size: 30px; font-weight: 800; line-height: 1.0; word-break: break-word; }}
        .controls {{ display: grid; grid-template-columns: 1.2fr 1fr repeat(4, minmax(90px, 1fr)); gap: 12px; align-items: end; margin-bottom: 14px; }}
        .control {{ background: rgba(255,255,255,0.02); border: 1px solid var(--border); border-radius: 14px; padding: 10px 12px; min-width: 0; }}
        .control label {{ display: block; font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
        .control input[type=range] {{ width: 100%; }}
        .btn {{ border: 1px solid var(--border); background: linear-gradient(180deg, #111827, #0a0a0a); color: var(--text); border-radius: 14px; padding: 12px 14px; font-weight: 700; cursor: pointer; min-width: 0; width: 100%; }}
        .btn:hover {{ border-color: rgba(255,255,255,0.18); }}
        .main {{ display: grid; grid-template-columns: 1fr; gap: 14px; }}
        .panel {{ background: linear-gradient(180deg, #0b1120, #060606); border: 1px solid var(--border); border-radius: 18px; padding: 14px; overflow: hidden; }}
        .panel h3 {{ margin: 0 0 12px 0; font-size: 18px; }}
        .legend {{ display: flex; flex-wrap: wrap; gap: 16px; font-size: 13px; color: var(--muted); margin-bottom: 8px; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; }}
        .dot {{ width: 12px; height: 12px; border-radius: 999px; display: inline-block; flex: 0 0 auto; }}
        .grid-wrap {{ border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; background: radial-gradient(circle at 50% 0%, rgba(56,189,248,0.06), transparent 30%), #05070d; padding: 10px; }}
        .svg-wrap {{ width: 100%; height: 620px; display: block; }}
        .bottom-grid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
        .right-stack {{ display: grid; grid-template-columns: 1fr 1fr 1.15fr; gap: 14px; }}
        .mini-title {{ font-size: 15px; font-weight: 700; margin-bottom: 8px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 10px; margin-top: 10px; }}
        .summary-box {{ background: rgba(255,255,255,0.03); border: 1px solid rgba(255,255,255,0.07); border-radius: 14px; padding: 12px; }}
        .summary-box .seg {{ font-size: 13px; color: var(--muted); margin-bottom: 6px; text-transform: capitalize; }}
        .summary-box .big {{ font-size: 22px; font-weight: 800; }}
        .summary-box .small {{ font-size: 13px; color: var(--muted); margin-top: 4px; line-height: 1.4; }}
        .interpret {{ background: linear-gradient(180deg, rgba(56,189,248,0.10), rgba(255,255,255,0.03)); border: 1px solid rgba(56,189,248,0.18); border-radius: 14px; padding: 14px; margin-top: 10px; }}
        .interpret .title {{ font-weight: 700; margin-bottom: 6px; }}
        .interpret .body {{ color: var(--muted); line-height: 1.55; font-size: 14px; }}
        .gridline {{ stroke: rgba(255,255,255,0.08); stroke-width: 1; }}
        .current-line {{ stroke: #ef4444; stroke-width: 1.5; stroke-dasharray: 4 4; }}
        @media (max-width: 1200px) {{
          .stats {{ grid-template-columns: repeat(3, minmax(0, 1fr)); }}
          .controls {{ grid-template-columns: 1fr 1fr; }}
          .bottom-grid {{ grid-template-columns: 1fr; }}
          .right-stack {{ grid-template-columns: 1fr; }}
          .summary-grid {{ grid-template-columns: 1fr; }}
        }}
        @media (max-width: 720px) {{
          .stats {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
          .controls {{ grid-template-columns: 1fr; }}
          .card .value {{ font-size: 24px; }}
          .svg-wrap {{ height: 520px; }}
        }}
      </style>

      <div class="wrap">
        <div class="intro">
          A polished ABM view should answer four questions instantly: what is the current price, who is buying, which segment is reacting the most, and whether the market is stable or oscillating. This dashboard is designed around those four answers.
        </div>

        <div class="stats">
          <div class="card"><div class="label">Current step</div><div class="value" id="stat-step">0</div></div>
          <div class="card"><div class="label">Current price</div><div class="value" id="stat-price">0.000</div></div>
          <div class="card"><div class="label">Current demand</div><div class="value" id="stat-demand">0.000</div></div>
          <div class="card"><div class="label">Buyers this step</div><div class="value" id="stat-buyers">0</div></div>
          <div class="card"><div class="label">Market state</div><div class="value" id="stat-regime" style="font-size:24px;">Stable</div></div>
        </div>

        <div class="controls">
          <div class="control">
            <label for="stepRange">Step</label>
            <input id="stepRange" type="range" min="0" max="{max_step}" value="0" step="1">
          </div>
          <div class="control">
            <label for="speedRange">Speed (ms)</label>
            <input id="speedRange" type="range" min="40" max="700" value="180" step="20">
          </div>
          <button class="btn" id="playBtn">Play</button>
          <button class="btn" id="pauseBtn">Pause</button>
          <button class="btn" id="nextBtn">Next</button>
          <button class="btn" id="resetBtn">Reset</button>
        </div>

        <div class="main">
          <div class="panel">
            <h3>Who is buying right now?</h3>
            <div class="legend">
              <div class="legend-item"><span class="dot" style="background:#ef4444"></span> Price-sensitive</div>
              <div class="legend-item"><span class="dot" style="background:#3b82f6"></span> Mainstream</div>
              <div class="legend-item"><span class="dot" style="background:#22c55e"></span> Loyal</div>
              <div class="legend-item"><span class="dot" style="background:#ffffff"></span> Bright = bought this step</div>
              <div class="legend-item"><span class="dot" style="background:#6b7280"></span> Faded = not buying</div>
            </div>
            <div class="grid-wrap">
              <svg id="agentSvg" class="svg-wrap" viewBox="0 0 900 620"></svg>
            </div>
          </div>

          <div class="bottom-grid">
            <div class="panel">
              <div class="mini-title">Where are we in the price path?</div>
              <svg id="priceSvg" viewBox="0 0 520 230" style="width:100%; height:220px;"></svg>
            </div>

            <div class="panel">
              <div class="mini-title">How is demand moving?</div>
              <svg id="demandSvg" viewBox="0 0 520 230" style="width:100%; height:220px;"></svg>
            </div>
          </div>

          <div class="panel">
            <div class="mini-title">Which segment is reacting most?</div>
            <div class="summary-grid" id="segmentSummary"></div>
            <div class="interpret">
              <div class="title" id="interpretTitle">Live market read</div>
              <div class="body" id="interpretBody"></div>
            </div>
          </div>
        </div>
      </div>
    `;

    const frames = ABM_DATA.frames;
    const segmentLabels = ABM_DATA.segment_labels;
    const segColors = ABM_DATA.segment_colors;
    const xs = ABM_DATA.x;
    const ys = ABM_DATA.y;
    const prices = ABM_DATA.prices;
    const demands = ABM_DATA.demands;

    const stepRange = document.getElementById('stepRange');
    const speedRange = document.getElementById('speedRange');
    const playBtn = document.getElementById('playBtn');
    const pauseBtn = document.getElementById('pauseBtn');
    const nextBtn = document.getElementById('nextBtn');
    const resetBtn = document.getElementById('resetBtn');

    const priceSvg = document.getElementById('priceSvg');
    const demandSvg = document.getElementById('demandSvg');
    const agentSvg = document.getElementById('agentSvg');
    const segmentSummary = document.getElementById('segmentSummary');

    let currentStep = 0;
    let timer = null;

    function clearSvg(svg) {{ while (svg.firstChild) svg.removeChild(svg.firstChild); }}

    function makeEl(tag, attrs = {{}}) {{
      const el = document.createElementNS('http://www.w3.org/2000/svg', tag);
      Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, v));
      return el;
    }}

    function drawLineChart(svg, series, currentIndex, color, yLabel, minY = null, maxY = null) {{
      clearSvg(svg);
      const W = 520, H = 230;
      const m = {{ left: 48, right: 16, top: 16, bottom: 34 }};
      const pw = W - m.left - m.right;
      const ph = H - m.top - m.bottom;

      const yMin = (minY !== null) ? minY : Math.min(...series);
      const yMax = (maxY !== null) ? maxY : Math.max(...series);
      const safeMin = yMin;
      const safeMax = yMax + (yMax === yMin ? 1 : 0);

      for (let i = 0; i < 5; i++) {{
        const y = m.top + (ph * i / 4);
        svg.appendChild(makeEl('line', {{ x1: m.left, y1: y, x2: W - m.right, y2: y, class: 'gridline' }}));
      }}

      const path = series.map((v, i) => {{
        const x = m.left + (pw * i / Math.max(1, series.length - 1));
        const y = m.top + ph * (1 - ((v - safeMin) / (safeMax - safeMin)));
        return `${{i === 0 ? 'M' : 'L'}} ${{x.toFixed(2)}} ${{y.toFixed(2)}}`;
      }}).join(' ');
      svg.appendChild(makeEl('path', {{ d: path, fill: 'none', stroke: color, 'stroke-width': 2.5 }}));

      const cx = m.left + (pw * currentIndex / Math.max(1, series.length - 1));
      svg.appendChild(makeEl('line', {{ x1: cx, y1: m.top, x2: cx, y2: H - m.bottom, class: 'current-line' }}));

      const dotY = m.top + ph * (1 - ((series[currentIndex] - safeMin) / (safeMax - safeMin)));
      svg.appendChild(makeEl('circle', {{ cx, cy: dotY, r: 4.8, fill: color }}));

      const xAxis = makeEl('line', {{ x1: m.left, y1: H - m.bottom, x2: W - m.right, y2: H - m.bottom, stroke: 'rgba(255,255,255,0.25)' }});
      const yAxis = makeEl('line', {{ x1: m.left, y1: m.top, x2: m.left, y2: H - m.bottom, stroke: 'rgba(255,255,255,0.25)' }});
      svg.appendChild(xAxis); svg.appendChild(yAxis);

      const label = makeEl('text', {{ x: 10, y: 24, fill: 'rgba(255,255,255,0.72)', 'font-size': 12 }});
      label.textContent = yLabel;
      svg.appendChild(label);
    }}

    function drawAgents(frame) {{
      clearSvg(agentSvg);
      const W = 900, H = 620;
      const m = 34;
      const minX = Math.min(...xs), maxX = Math.max(...xs);
      const minY = Math.min(...ys), maxY = Math.max(...ys);
      const scaleX = (W - 2 * m) / Math.max(1, maxX - minX + 1);
      const scaleY = (H - 2 * m) / Math.max(1, maxY - minY + 1);
      const scale = Math.min(scaleX, scaleY);

      frame.buy_now.forEach((buy, i) => {{
        const seg = segmentLabels[i];
        const base = segColors[seg] || '#94a3b8';
        const cx = m + (xs[i] - minX + 0.5) * scale;
        const cy = m + (maxY - ys[i] + 0.5) * scale;
        const bright = !!buy;
        const fill = bright ? base : 'rgba(148,163,184,0.34)';
        const stroke = bright ? 'rgba(255,255,255,0.35)' : 'rgba(255,255,255,0.10)';
        const size = Math.max(7.5, Math.min(11.5, scale * 0.22));

        if (seg === 'mainstream') {{
          const rect = makeEl('rect', {{ x: cx - size, y: cy - size, width: size * 2, height: size * 2, fill, stroke, 'stroke-width': 1 }});
          if (bright) rect.setAttribute('filter', 'url(#glow)');
          agentSvg.appendChild(rect);
        }} else if (seg === 'loyal') {{
          const points = `${{cx}},${{cy - size*1.15}} ${{cx - size}},${{cy + size}} ${{cx + size}},${{cy + size}}`;
          const tri = makeEl('polygon', {{ points, fill, stroke, 'stroke-width': 1 }});
          if (bright) tri.setAttribute('filter', 'url(#glow)');
          agentSvg.appendChild(tri);
        }} else {{
          const circle = makeEl('circle', {{ cx, cy, r: size, fill, stroke, 'stroke-width': 1 }});
          if (bright) circle.setAttribute('filter', 'url(#glow)');
          agentSvg.appendChild(circle);
        }}
      }});

      const defs = makeEl('defs');
      const filter = makeEl('filter', {{ id: 'glow', x: '-50%', y: '-50%', width: '200%', height: '200%' }});
      filter.appendChild(makeEl('feGaussianBlur', {{ in: 'SourceGraphic', stdDeviation: 2.4, result: 'blur' }}));
      const merge = makeEl('feMerge');
      merge.appendChild(makeEl('feMergeNode', {{ in: 'blur' }}));
      merge.appendChild(makeEl('feMergeNode', {{ in: 'SourceGraphic' }}));
      filter.appendChild(merge);
      defs.appendChild(filter);
      agentSvg.insertBefore(defs, agentSvg.firstChild);
    }}

    function renderSegmentSummary(frame) {{
      segmentSummary.innerHTML = '';
      frame.segment_summary.forEach(row => {{
        const div = document.createElement('div');
        div.className = 'summary-box';
        const pct = (100 * row.buy_rate).toFixed(0);
        div.innerHTML = `
          <div class="seg" style="color:${{segColors[row.segment]}}">${{row.segment.replace('_', ' ')}}</div>
          <div class="big">${{pct}}%</div>
          <div class="small">${{row.buyers}} / ${{row.count}} buying now<br>Avg effective WTP: ${{row.avg_wtp.toFixed(3)}}</div>
        `;
        segmentSummary.appendChild(div);
      }});
    }}

    function renderFrame(idx) {{
      currentStep = idx;
      stepRange.value = idx;
      const frame = frames[idx];

      document.getElementById('stat-step').textContent = frame.step;
      document.getElementById('stat-price').textContent = frame.price.toFixed(3);
      document.getElementById('stat-demand').textContent = frame.demand.toFixed(3);
      document.getElementById('stat-buyers').textContent = frame.buyers;
      document.getElementById('stat-regime').textContent = frame.regime_text;
      document.getElementById('interpretTitle').textContent = `Live market read · ${{frame.regime_text}}`;
      document.getElementById('interpretBody').textContent = `${{frame.interpretation}} Current price is ${{frame.price.toFixed(3)}}, demand is ${{frame.demand.toFixed(3)}}, and profit is ${{frame.profit.toFixed(2)}} at this step.`;

      drawAgents(frame);
      drawLineChart(priceSvg, prices, idx, '#60a5fa', 'Price');
      drawLineChart(demandSvg, demands, idx, '#f59e0b', 'Demand', 0, 1);
      renderSegmentSummary(frame);
    }}

    function play() {{
      if (timer) return;
      timer = setInterval(() => {{
        if (currentStep >= frames.length - 1) {{ pause(); return; }}
        renderFrame(currentStep + 1);
      }}, Number(speedRange.value));
    }}

    function pause() {{
      if (timer) {{ clearInterval(timer); timer = null; }}
    }}

    playBtn.onclick = play;
    pauseBtn.onclick = pause;
    nextBtn.onclick = () => {{ pause(); renderFrame(Math.min(frames.length - 1, currentStep + 1)); }};
    resetBtn.onclick = () => {{ pause(); renderFrame(0); }};
    stepRange.oninput = (e) => {{ pause(); renderFrame(Number(e.target.value)); }};
    speedRange.oninput = () => {{ if (timer) {{ pause(); play(); }} }};

    renderFrame(Math.min(10, frames.length - 1));
    </script>
    """

    return html, height


# This is my simulation 

def run_simulation(
    r=1.2,
    n_consumers=1000,
    steps=300,
    initial_price=0.50,
    seed=42,
    dynamic_wtp=True,
    ou_theta=0.10,
    wtp_noise=0.02,
    use_seasonality=True,
    season_amplitude=0.15,
    season_period=30,
    unit_cost=0.20,
    fixed_cost_per_step=0.0,
    n_visual_agents=180,
):
    """
    Core rule:
        P_{t+1} = r * P_t * D_t

    where D_t is the fraction of consumers who buy at time t.
    """
    rng = np.random.default_rng(seed)

    base_wtp, segments = generate_segmented_market(n_consumers, seed=seed)
    current_wtp = base_wtp.copy()

    n_visual_agents = int(max(20, min(n_visual_agents, n_consumers)))
    sample_idx = np.linspace(0, n_consumers - 1, n_visual_agents, dtype=int)
    sample_x, sample_y = build_agent_grid(n_visual_agents)
    buy_history = np.zeros((steps, n_visual_agents), dtype=int)
    wtp_history = np.zeros((steps, n_visual_agents), dtype=float)

    prices = np.zeros(steps + 1)
    demands = np.zeros(steps)
    buyers = np.zeros(steps, dtype=int)
    revenue = np.zeros(steps)
    costs = np.zeros(steps)
    profit = np.zeros(steps)
    consumer_surplus = np.zeros(steps)

    prices[0] = initial_price

    for t in range(steps):
        p = prices[t]

        if use_seasonality:
            effective_wtp = np.clip(
                current_wtp * seasonal_multiplier(t, season_amplitude, season_period),
                0.0,
                1.2
            )
        else:
            effective_wtp = current_wtp

        buy_mask = p <= effective_wtp
        n_buy = int(buy_mask.sum())
        d_t = n_buy / n_consumers

        buyers[t] = n_buy
        demands[t] = d_t
        revenue[t] = p * n_buy
        costs[t] = unit_cost * n_buy + fixed_cost_per_step
        profit[t] = revenue[t] - costs[t]
        consumer_surplus[t] = float(np.sum(np.maximum(0.0, effective_wtp[buy_mask] - p)))

        buy_history[t] = buy_mask[sample_idx].astype(int)
        wtp_history[t] = effective_wtp[sample_idx]


        next_price = r * p * d_t
        next_price = float(np.clip(next_price, 1e-4, 2.0))
        prices[t + 1] = next_price

        if dynamic_wtp:
            noise = rng.normal(0.0, wtp_noise, size=n_consumers)
            current_wtp = np.clip(
                current_wtp + ou_theta * (base_wtp - current_wtp) + noise,
                0.0,
                1.0
            )

    tail_price = float(np.mean(prices[int(0.5 * len(prices)):]))
    segment_rows = []

    for seg in ["price_sensitive", "mainstream", "loyal"]:
        mask = segments == seg
        seg_wtp = base_wtp[mask]
        buy_proxy = float(np.mean(seg_wtp >= tail_price))
        segment_rows.append({
            "Segment": seg,
            "Consumers": int(mask.sum()),
            "Avg base WTP": round(float(np.mean(seg_wtp)), 4),
            "Tail buy-rate proxy": round(buy_proxy, 4),
        })

    return {
        "prices": prices,
        "demands": demands,
        "buyers": buyers,
        "revenue": revenue,
        "costs": costs,
        "profit": profit,
        "cumulative_profit": np.cumsum(profit),
        "consumer_surplus": consumer_surplus,
        "base_wtp": base_wtp,
        "segments": segments,
        "segment_summary": pd.DataFrame(segment_rows),
        "agent_viz": {
            "sample_idx": sample_idx,
            "sample_segments": segments[sample_idx],
            "sample_x": sample_x,
            "sample_y": sample_y,
            "buy_history": buy_history,
            "wtp_history": wtp_history,
        },
    }



def compute_tail_diagnostics(prices, demands, profit, transient_fraction=0.5):
    start = int(len(profit) * transient_fraction)

    tail_prices = np.asarray(prices[start:], dtype=float)
    tail_demands = np.asarray(demands[start:], dtype=float)
    tail_profit = np.asarray(profit[start:], dtype=float)

    mean_price = float(np.mean(tail_prices))
    std_price = float(np.std(tail_prices))
    cv_price = float(std_price / (mean_price + 1e-12))

    mean_demand = float(np.mean(tail_demands))
    std_demand = float(np.std(tail_demands))

    mean_profit = float(np.mean(tail_profit))
    std_profit = float(np.std(tail_profit))

    centered = tail_prices - np.mean(tail_prices)
    lag_autocorr = {}
    denom = float(np.dot(centered, centered))

    for lag in range(1, min(12, len(centered) - 2) + 1):
        if denom <= 1e-12:
            lag_autocorr[lag] = 0.0
        else:
            lag_autocorr[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denom)

    # demand stability: lower std means more stable
    demand_stability = float(1.0 / (1.0 + std_demand))

    # fairness proxy: lower price CV = more fair/less erratic
    fairness_index = float(1.0 / (1.0 + cv_price))

    # simple spectral analysis
    if len(centered) > 5:
        fft_vals = np.fft.rfft(centered)
        power = np.abs(fft_vals) ** 2
        freqs = np.fft.rfftfreq(len(centered), d=1.0)

        if len(power) > 1 and np.sum(power[1:]) > 1e-12:
            dom_idx = int(np.argmax(power[1:]) + 1)
            dominant_frequency = float(freqs[dom_idx])
            dominant_period = float(1.0 / dominant_frequency) if dominant_frequency > 1e-12 else np.nan
        else:
            dominant_frequency = 0.0
            dominant_period = np.nan
    else:
        dominant_frequency = 0.0
        dominant_period = np.nan

    return {
        "tail_prices": tail_prices,
        "tail_demands": tail_demands,
        "tail_profit": tail_profit,
        "mean_price": mean_price,
        "std_price": std_price,
        "cv_price": cv_price,
        "mean_demand": mean_demand,
        "std_demand": std_demand,
        "mean_profit": mean_profit,
        "std_profit": std_profit,
        "demand_stability": demand_stability,
        "fairness_index": fairness_index,
        "lag_autocorr": lag_autocorr,
        "dominant_frequency": dominant_frequency,
        "dominant_period": dominant_period,
        "min_price": float(np.min(tail_prices)),
        "max_price": float(np.max(tail_prices)),
    }


def estimate_cycle_count(tail_prices, tol=0.005):
    """
    Rough estimate of how many distinct long-run price levels / cycle states exist.
    This is a simple, interpretable replacement for formal bifurcation-period detection.
    """
    tail_prices = np.asarray(tail_prices, dtype=float)
    if len(tail_prices) == 0:
        return 0

    s = np.sort(tail_prices)
    distinct = 1
    for i in range(1, len(s)):
        if abs(s[i] - s[i - 1]) > tol:
            distinct += 1

    return int(distinct)


def classify_regime(price_cv: float, demand_stability: float, dominant_period):
    if price_cv < 0.03 and demand_stability > 0.92:
        return "Stable"
    if price_cv < 0.08 and np.isfinite(dominant_period):
        return "Cyclical"
    if price_cv < 0.14:
        return "Volatile"
    return "Highly volatile"


def build_main_takeaway(diagnostics):
    regime = classify_regime(
        diagnostics["cv_price"],
        diagnostics["demand_stability"],
        diagnostics["dominant_period"],
    )

    if regime == "Stable":
        regime_text = (
            "The system is operating in a stable region. Prices stay relatively smooth, demand remains consistent, and the bot is not overreacting to short-run fluctuations."
        )
    elif regime == "Cyclical":
        regime_text = (
            "The system appears cyclical rather than fully stable. Prices are not random, but they move in a repeating pattern over time."
        )
    elif regime == "Volatile":
        regime_text = (
            "The system is becoming volatile. The bot is reacting strongly enough that prices fluctuate meaningfully, even if they are still bounded."
        )
    else:
        regime_text = (
            "The system is highly volatile. Price movements are large relative to the average price level, which makes the policy harder to justify operationally."
        )

    if np.isfinite(diagnostics["dominant_period"]):
        cycle_text = f"The long-run price path shows a dominant cycle of roughly {diagnostics['dominant_period']:.2f} steps."
    else:
        cycle_text = "There is no single dominant repeating cycle in the long-run window."

    profit_text = (
        f"Average long-run profit is {diagnostics['mean_profit']:.2f} per step, with average price {diagnostics['mean_price']:.4f} and average demand {diagnostics['mean_demand']:.4f}."
    )

    return regime_text + "\n\n" + cycle_text + "\n\n" + profit_text


def build_sweep_takeaway(best_row, ranking_df):
    median_profit = float(ranking_df["avg_profit_long_run"].median())
    uplift = float(best_row["avg_profit_long_run"] - median_profit)

    if best_row["regime"] == "Stable":
        stability_note = "The best-profit r also lies in a stable operating region, which makes it easier to defend in practice."
    elif best_row["regime"] == "Cyclical":
        stability_note = "The best-profit r lies in a cyclical region, so the policy may still be usable but will produce recurring swings over time."
    elif best_row["regime"] == "Volatile":
        stability_note = "The best-profit r lies in a volatile region, so there is a trade-off between higher profit and smoother operations."
    else:
        stability_note = "The best-profit r lies in a highly volatile region, so it may look attractive in simulation but could be difficult to deploy operationally."

    return (
        f"The best tested value is r = {best_row['r']:.4f}. Compared with the median tested policy, it improves long-run profit by about {uplift:.2f} per step.\n\n"
        + stability_note
    )


def sweep_r_values(
    r_min,
    r_max,
    n_r,
    n_consumers,
    steps,
    initial_price,
    seed,
    dynamic_wtp,
    ou_theta,
    wtp_noise,
    use_seasonality,
    season_amplitude,
    season_period,
    unit_cost,
    fixed_cost_per_step,
):
    rows = []
    r_grid = np.linspace(r_min, r_max, n_r)

    for r in r_grid:
        sim = run_simulation(
            r=float(r),
            n_consumers=n_consumers,
            steps=steps,
            initial_price=initial_price,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            ou_theta=ou_theta,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )

        diag = compute_tail_diagnostics(sim["prices"], sim["demands"], sim["profit"])
        cycle_count = estimate_cycle_count(diag["tail_prices"])

        rows.append({
            "r": round(float(r), 4),
            "avg_profit_long_run": round(diag["mean_profit"], 4),
            "total_profit": round(float(np.sum(sim["profit"])), 4),
            "price_volatility": round(diag["std_price"], 4),
            "avg_price_long_run": round(diag["mean_price"], 4),
            "avg_demand_long_run": round(diag["mean_demand"], 4),
            "demand_stability": round(diag["demand_stability"], 4),
            "fairness_index": round(diag["fairness_index"], 4),
            "consumer_surplus_long_run": round(float(np.mean(sim["consumer_surplus"][int(0.5 * len(sim["consumer_surplus"])):])), 4),
            "dominant_period": round(float(diag["dominant_period"]), 4) if np.isfinite(diag["dominant_period"]) else np.nan,
            "cycle_count": int(cycle_count),
            "regime": classify_regime(diag["cv_price"], diag["demand_stability"], diag["dominant_period"]),
        })

    ranking_df = pd.DataFrame(rows).sort_values("avg_profit_long_run", ascending=False).reset_index(drop=True)
    best_row = ranking_df.iloc[0].to_dict()

    best_run = run_simulation(
        r=float(best_row["r"]),
        n_consumers=n_consumers,
        steps=steps,
        initial_price=initial_price,
        seed=seed,
        dynamic_wtp=dynamic_wtp,
        ou_theta=ou_theta,
        wtp_noise=wtp_noise,
        use_seasonality=use_seasonality,
        season_amplitude=season_amplitude,
        season_period=season_period,
        unit_cost=unit_cost,
        fixed_cost_per_step=fixed_cost_per_step,
    )

    return ranking_df, best_row, best_run

# Bifurcation diagram helper
def build_bifurcation_points(
    r_min,
    r_max,
    n_r,
    n_consumers,
    steps,
    initial_price,
    seed,
    dynamic_wtp,
    ou_theta,
    wtp_noise,
    use_seasonality,
    season_amplitude,
    season_period,
    unit_cost,
    fixed_cost_per_step,
    tail_keep=120,
):
    xs = []
    ys = []
    r_grid = np.linspace(r_min, r_max, n_r)

    for r in r_grid:
        sim = run_simulation(
            r=float(r),
            n_consumers=n_consumers,
            steps=steps,
            initial_price=initial_price,
            seed=seed,
            dynamic_wtp=dynamic_wtp,
            ou_theta=ou_theta,
            wtp_noise=wtp_noise,
            use_seasonality=use_seasonality,
            season_amplitude=season_amplitude,
            season_period=season_period,
            unit_cost=unit_cost,
            fixed_cost_per_step=fixed_cost_per_step,
        )

        tail_prices = np.asarray(sim["prices"][-tail_keep:], dtype=float)
        xs.extend([float(r)] * len(tail_prices))
        ys.extend(tail_prices.tolist())

    return np.array(xs, dtype=float), np.array(ys, dtype=float)



st.sidebar.header("Core Parameters")
r = st.sidebar.slider("Aggressiveness r", 0.0, 4.0, 1.20, 0.01)
n_consumers = st.sidebar.slider("Consumers N", 300, 3000, 1000, 100)
steps = st.sidebar.slider("Time steps", 80, 800, 300, 20)
initial_price = st.sidebar.slider("Initial price P₀", 0.01, 1.50, 0.50, 0.01)
seed = int(st.sidebar.number_input("Random seed", 0, 999999, 42, 1))

st.sidebar.subheader("Consumer Dynamics")
dynamic_wtp = st.sidebar.checkbox("Dynamic WTP (mean-reverting)", True)
ou_theta = st.sidebar.slider("OU mean reversion θ", 0.01, 0.50, 0.10, 0.01, disabled=not dynamic_wtp)
wtp_noise = st.sidebar.slider("WTP noise σ", 0.001, 0.10, 0.02, 0.001, disabled=not dynamic_wtp)

st.sidebar.subheader("Seasonality")
use_seasonality = st.sidebar.checkbox("Use seasonality", True)
season_amplitude = st.sidebar.slider("Season amplitude", 0.0, 0.50, 0.15, 0.01, disabled=not use_seasonality)
season_period = st.sidebar.slider("Season period", 5, 90, 30, 1, disabled=not use_seasonality)

st.sidebar.subheader("Cost Structure")
unit_cost = st.sidebar.slider("Unit cost", 0.00, 0.80, 0.20, 0.01)
fixed_cost_per_step = st.sidebar.slider("Fixed cost / step", 0.0, 100.0, 0.0, 1.0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    <div class="small-note">
        <strong>Interpretation:</strong> low r means the bot is cautious, high r means it reacts aggressively to current demand.
        The objective is to find the value of r that gives the best long-run balance of profit, stability, and fairness.
    </div>
    """,
    unsafe_allow_html=True,
)

if st.sidebar.button("▶ Run simulation", type="primary", width="stretch"):
    st.session_state["result"] = run_simulation(
        r=r,
        n_consumers=n_consumers,
        steps=steps,
        initial_price=initial_price,
        seed=seed,
        dynamic_wtp=dynamic_wtp,
        ou_theta=ou_theta,
        wtp_noise=wtp_noise,
        use_seasonality=use_seasonality,
        season_amplitude=season_amplitude,
        season_period=season_period,
        unit_cost=unit_cost,
        fixed_cost_per_step=fixed_cost_per_step,
        n_visual_agents=180,
    )

if "result" not in st.session_state:
    st.info("Configure the model in the sidebar and click ▶ Run simulation.")
    st.stop()

result = st.session_state["result"]
diagnostics = compute_tail_diagnostics(result["prices"], result["demands"], result["profit"])
regime = classify_regime(
    diagnostics["cv_price"],
    diagnostics["demand_stability"],
    diagnostics["dominant_period"],
)


tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📈 Main Simulation",
    "🔬 Stability Diagnostics",
    "🎯 r Sweep Results",
    "⚖️ Consumer Welfare",
    "🎬 ABM Visual",
])


with tab1:
    st.subheader("Main Simulation Summary")

    m = st.columns(7)
    m[0].metric("Regime", regime)
    m[1].metric("Demand stability", f"{diagnostics['demand_stability']:.4f}")
    m[2].metric("Avg long-run price", f"{diagnostics['mean_price']:.4f}")
    m[3].metric("Avg long-run demand", f"{diagnostics['mean_demand']:.4f}")
    m[4].metric("Avg long-run profit", f"{diagnostics['mean_profit']:.2f}")
    m[5].metric("Price volatility", f"{diagnostics['std_price']:.4f}")

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["prices"], lw=1.4, color="#3b82f6")
        ax.set_title("Price over time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Price")
        st.pyplot(fig)
        plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["demands"], lw=1.4, color="#f97316")
        ax.set_title("Demand over time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Demand")
        st.pyplot(fig)
        plt.close()

    c3, c4 = st.columns(2)
    with c3:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["profit"], lw=1.1, color="#22c55e", alpha=0.8)
        ax.set_title("Profit per step")
        ax.set_xlabel("Step")
        ax.set_ylabel("Profit")
        st.pyplot(fig)
        plt.close()

    with c4:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["cumulative_profit"], lw=1.4, color="#a855f7")
        ax.set_title("Cumulative profit")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative profit")
        st.pyplot(fig)
        plt.close()

    st.subheader("Interpretation")
    st.markdown(build_main_takeaway(diagnostics))


with tab2:
    st.subheader("Stability Diagnostics")

    d = st.columns(6)
    d[0].metric("Regime", regime)
    d[1].metric("Price CoV", f"{diagnostics['cv_price']:.4f}")
    d[2].metric("Demand stability", f"{diagnostics['demand_stability']:.4f}")
    d[3].metric(
        "Dominant period",
        "—" if not np.isfinite(diagnostics["dominant_period"]) else f"{diagnostics['dominant_period']:.2f}"
    )
    d[4].metric("Mean profit", f"{diagnostics['mean_profit']:.4f}")
    d[5].metric("Price range", f"{diagnostics['min_price']:.3f} – {diagnostics['max_price']:.3f}")

    st.markdown(
        """
        This section explains whether the pricing bot is behaving in a stable and interpretable way.

        - **Price CoV** shows how erratic prices are relative to their average level
        - **Demand stability** measures how smooth demand remains over time
        - **Dominant period** checks whether the system falls into a repeating cycle
        - **Return map** and **autocorrelation** help you see whether the system converges, cycles, or becomes irregular
        """
    )

    a1, a2 = st.columns(2)
    with a1:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(result["prices"][:-1], result["prices"][1:], s=6, alpha=0.4, color="#e5e7eb")
        mx = max(result["prices"])
        ax.plot([0, mx], [0, mx], ls="--", lw=0.8, color="#777777")
        ax.set_title("Return map")
        ax.set_xlabel("P_t")
        ax.set_ylabel("P_(t+1)")
        ax.set_aspect("equal")
        st.pyplot(fig)
        plt.close()

    with a2:
        lags = list(diagnostics["lag_autocorr"].keys())
        vals = list(diagnostics["lag_autocorr"].values())
        fig, ax = plt.subplots(figsize=(6, 5))
        if lags:
            ax.bar(lags, vals, width=0.7)
        ax.axhline(0, color="black", lw=0.8)
        ax.set_title("Price autocorrelation")
        ax.set_xlabel("Lag")
        ax.set_ylabel("Autocorrelation")
        st.pyplot(fig)
        plt.close()


with tab5:
    st.subheader("Agent-Based Modelling Visual")
    st.markdown(
        "A polished ABM visual should answer four questions instantly: **what is the current price, who is buying, which segment is reacting most, and whether the market is stable or oscillating**."
    )

    html, html_height = build_abm_dashboard_html(result)
    components.html(html, height=html_height, scrolling=True)


with tab3:
    st.subheader("Sweep r and Find the Best Region")

    s1, s2, s3 = st.columns(3)
    with s1:
        r_min = st.slider("r min", 0.0, 4.0, 0.10, 0.05)
    with s2:
        r_max = st.slider("r max", 0.1, 4.0, 3.00, 0.05)
    with s3:
        n_r = st.slider("# of r values", 20, 200, 80, 5)

    if st.button("Run r sweep", type="primary", width="stretch"):
        if r_max <= r_min:
            st.error("r max must be greater than r min.")
        else:
            with st.spinner("Sweeping across r values..."):
                ranking_df, best_row, best_run = sweep_r_values(
                    r_min=r_min,
                    r_max=r_max,
                    n_r=n_r,
                    n_consumers=n_consumers,
                    steps=steps,
                    initial_price=initial_price,
                    seed=seed,
                    dynamic_wtp=dynamic_wtp,
                    ou_theta=ou_theta,
                    wtp_noise=wtp_noise,
                    use_seasonality=use_seasonality,
                    season_amplitude=season_amplitude,
                    season_period=season_period,
                    unit_cost=unit_cost,
                    fixed_cost_per_step=fixed_cost_per_step,
                )
                bif_x, bif_y = build_bifurcation_points(
                    r_min=r_min,
                    r_max=r_max,
                    n_r=n_r,
                    n_consumers=n_consumers,
                    steps=steps,
                    initial_price=initial_price,
                    seed=seed,
                    dynamic_wtp=dynamic_wtp,
                    ou_theta=ou_theta,
                    wtp_noise=wtp_noise,
                    use_seasonality=use_seasonality,
                    season_amplitude=season_amplitude,
                    season_period=season_period,
                    unit_cost=unit_cost,
                    fixed_cost_per_step=fixed_cost_per_step,
                )
                st.session_state["ranking_df"] = ranking_df
                st.session_state["best_row"] = best_row
                st.session_state["best_run"] = best_run
                st.session_state["bif_x"] = bif_x
                st.session_state["bif_y"] = bif_y

    if "ranking_df" in st.session_state:
        ranking_df = st.session_state["ranking_df"]
        best_row = st.session_state["best_row"]
        best_run = st.session_state["best_run"]

        k = st.columns(4)
        k[0].metric("Best r", f"{best_row['r']:.4f}")
        k[1].metric("Best avg profit", f"{best_row['avg_profit_long_run']:.4f}")
        k[2].metric("Best regime", best_row["regime"])
        k[3].metric("Best demand stability", f"{best_row['demand_stability']:.4f}")
        st.caption(f"Estimated cycle count at best r: {int(best_row['cycle_count'])}")

        st.markdown(build_sweep_takeaway(best_row, ranking_df))

        ranking_sorted = ranking_df.sort_values("r")

        fig, axes = plt.subplots(4, 2, figsize=(14, 16))

        axes[0, 0].plot(ranking_sorted["r"], ranking_sorted["avg_profit_long_run"], lw=1.5, color="#3b82f6")
        axes[0, 0].axvline(best_row["r"], color="red", ls="--", lw=1.0)
        axes[0, 0].set_title("r vs profit")
        axes[0, 0].set_xlabel("r")
        axes[0, 0].set_ylabel("Avg long-run profit")

        axes[0, 1].plot(ranking_sorted["r"], ranking_sorted["price_volatility"], lw=1.5, color="#ef4444")
        axes[0, 1].axvline(best_row["r"], color="red", ls="--", lw=1.0)
        axes[0, 1].set_title("r vs volatility")
        axes[0, 1].set_xlabel("r")
        axes[0, 1].set_ylabel("Std price")

        axes[1, 0].plot(ranking_sorted["r"], ranking_sorted["demand_stability"], lw=1.5, color="#22c55e")
        axes[1, 0].axvline(best_row["r"], color="red", ls="--", lw=1.0)
        axes[1, 0].set_title("r vs demand stability")
        axes[1, 0].set_xlabel("r")
        axes[1, 0].set_ylabel("Demand stability")

        axes[1, 1].plot(ranking_sorted["r"], ranking_sorted["fairness_index"], lw=1.5, color="#a855f7")
        axes[1, 1].axvline(best_row["r"], color="red", ls="--", lw=1.0)
        axes[1, 1].set_title("r vs fairness / consumer welfare")
        axes[1, 1].set_xlabel("r")
        axes[1, 1].set_ylabel("Fairness index")

        axes[2, 0].plot(ranking_sorted["r"], ranking_sorted["cycle_count"], lw=1.5, color="#f59e0b")
        axes[2, 0].axvline(best_row["r"], color="red", ls="--", lw=1.0)
        axes[2, 0].set_title("r vs number of cycles")
        axes[2, 0].set_xlabel("r")
        axes[2, 0].set_ylabel("Estimated cycle count")

        axes[2, 1].plot(ranking_sorted["r"], ranking_sorted["avg_price_long_run"], lw=1.5, color="#14b8a6")
        axes[2, 1].axvline(best_row["r"], color="red", ls="--", lw=1.0)
        axes[2, 1].set_title("r vs average long-run price")
        axes[2, 1].set_xlabel("r")
        axes[2, 1].set_ylabel("Avg price")

        if "bif_x" in st.session_state and "bif_y" in st.session_state:
            axes[3, 0].scatter(st.session_state["bif_x"], st.session_state["bif_y"], s=1.0, alpha=0.20, color="#e5e7eb")
            axes[3, 0].axvline(best_row["r"], color="red", ls="--", lw=1.0)
            axes[3, 0].set_title("Bifurcation diagram: r vs long-run price levels")
            axes[3, 0].set_xlabel("r")
            axes[3, 0].set_ylabel("Long-run price")
        else:
            axes[3, 0].set_visible(False)

        axes[3, 1].axis("off")
        axes[3, 1].text(
            0.0,
            0.95,
            "How to read the bifurcation diagram:\n\n"
            "• One narrow band = stable long-run pricing\n"
            "• Two or more branches = repeating cycles\n"
            "• A dense cloud = irregular / highly complex behavior\n\n"
            "This is the closest graph in the app to the textbook logistic-map picture.",
            va="top",
            fontsize=11,
        )

        for ax in axes.flat:
            if ax.get_visible():
                ax.grid(True, alpha=0.25)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close()



        st.markdown("### Top tested r values")
        st.dataframe(ranking_df.head(15), width="stretch")

        b1, b2 = st.columns(2)
        with b1:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(best_run["prices"], lw=1.3, color="#3b82f6")
            ax.set_title(f"Price path at best r = {best_row['r']:.4f}")
            ax.set_xlabel("Step")
            ax.set_ylabel("Price")
            st.pyplot(fig)
            plt.close()

        with b2:
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(best_run["cumulative_profit"], lw=1.4, color="#22c55e")
            ax.set_title("Cumulative profit at best r")
            ax.set_xlabel("Step")
            ax.set_ylabel("Cumulative profit")
            st.pyplot(fig)
            plt.close()

with tab4:
    st.subheader("Consumer Welfare and Segment View")

    w = st.columns(4)
    w[0].metric("Avg consumer surplus / step", f"{np.mean(result['consumer_surplus'][int(0.5*len(result['consumer_surplus'])):]):.2f}")
    w[1].metric("Total consumer surplus", f"{np.sum(result['consumer_surplus']):.1f}")
    w[2].metric("Avg seller profit / step", f"{diagnostics['mean_profit']:.2f}")
    w[3].metric("Fairness index", f"{diagnostics['fairness_index']:.4f}")

    p1, p2 = st.columns(2)
    with p1:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["consumer_surplus"], lw=1.2, color="#a855f7")
        ax.set_title("Consumer surplus over time")
        ax.set_xlabel("Step")
        ax.set_ylabel("Consumer surplus")
        st.pyplot(fig)
        plt.close()

    with p2:
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(result["profit"], lw=1.2, color="#22c55e", label="Seller profit")
        ax.plot(result["consumer_surplus"], lw=1.0, ls="--", alpha=0.85, color="#a855f7", label="Consumer surplus")
        ax.set_title("Seller vs consumer value")
        ax.set_xlabel("Step")
        ax.legend()
        st.pyplot(fig)
        plt.close()

    st.markdown("### Segment summary")
    st.dataframe(result["segment_summary"], width="stretch")

    fig, ax = plt.subplots(figsize=(7, 4))
    seg_colors = {
        "price_sensitive": "#ef4444",
        "mainstream": "#3b82f6",
        "loyal": "#22c55e",
    }

    for seg, color in seg_colors.items():
        seg_wtp = result["base_wtp"][result["segments"] == seg]
        ax.hist(seg_wtp, bins=30, alpha=0.55, label=seg, color=color)

    ax.set_title("Base WTP distribution by segment")
    ax.set_xlabel("WTP")
    ax.set_ylabel("Count")
    ax.legend()
    st.pyplot(fig)
    plt.close()

    st.markdown(
        """
        **How to read this section**
        
        - If `r` is too low, the bot reacts too slowly and may leave profit unrealized.
        - If `r` is too high, the bot may become unstable and create erratic prices.
        - A good `r` should balance **profit**, **stability**, **demand smoothness**, and **consumer experience**.
        """
    )


st.divider()
st.subheader("How to run locally")
st.code(
    "pip install streamlit numpy pandas matplotlib\n"
    "streamlit run app.py",
    language="bash",
)
