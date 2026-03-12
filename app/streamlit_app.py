"""
Ligue 1 Predictor — Premium Betting Analytics Platform
"""

import json
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
UPCOMING_PATH = PROJECT_ROOT / "data" / "raw" / "fixtures" / "upcoming.json"
ODDS_PATH = PROJECT_ROOT / "data" / "raw" / "odds" / "upcoming_odds.json"

# ── Design Tokens ──
BG = "#0A0E1A"
CARD = "#111827"
BORDER = "#1F2937"
SURFACE = "#1E293B"
BLUE = "#3B82F6"
GREEN = "#10B981"
RED = "#EF4444"
YELLOW = "#F59E0B"
WHITE = "#F9FAFB"
MUTED = "#6B7280"
SUBTLE = "#374151"

LABEL_ORDER = ["H", "D", "A"]
LABEL_NAMES = {"H": "Home", "D": "Draw", "A": "Away"}

# ── Page Config ──
st.set_page_config(
    page_title="Ligue 1 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ──
st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

    .stApp {{
        background-color: {BG};
        color: {WHITE};
        font-family: 'Inter', -apple-system, sans-serif;
    }}
    #MainMenu, footer, header {{visibility: hidden;}}
    .block-container {{padding-top: 1rem; max-width: 1100px;}}

    /* Hero */
    .hero {{
        background: linear-gradient(135deg, {CARD} 0%, {SURFACE} 100%);
        border: 1px solid {BORDER};
        border-radius: 16px;
        padding: 28px 36px;
        margin-bottom: 20px;
    }}
    .hero-row {{
        display: flex;
        justify-content: space-between;
        align-items: center;
    }}
    .hero-title {{
        font-size: 32px;
        font-weight: 900;
        letter-spacing: -0.5px;
    }}
    .hero-sub {{
        font-size: 13px;
        color: {MUTED};
        margin-top: 2px;
    }}
    .hero-update {{
        font-size: 12px;
        color: {MUTED};
        text-align: right;
    }}
    .hero-update strong {{ color: {GREEN}; }}

    /* Summary Row */
    .summary-row {{
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 12px;
        margin-bottom: 24px;
    }}
    .summary-box {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
        padding: 16px 20px;
        text-align: center;
    }}
    .summary-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 800;
    }}
    .summary-label {{
        font-size: 11px;
        color: {MUTED};
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }}

    /* Match Card */
    .mc {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 14px;
        padding: 22px 26px;
        margin-bottom: 14px;
    }}
    .mc-edge {{ border-left: 4px solid {GREEN}; }}
    .mc-skip {{ border-left: 4px solid {SUBTLE}; opacity: 0.7; }}

    .mc-head {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 14px;
    }}
    .mc-date {{
        font-size: 13px;
        color: {MUTED};
    }}
    .mc-round {{
        font-size: 11px;
        color: {MUTED};
        background: {SURFACE};
        padding: 3px 10px;
        border-radius: 20px;
    }}

    /* Teams */
    .teams {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 20px;
        margin-bottom: 18px;
    }}
    .tm {{
        display: flex;
        align-items: center;
        gap: 10px;
        flex: 1;
    }}
    .tm-h {{ justify-content: flex-end; text-align: right; }}
    .tm-a {{ justify-content: flex-start; }}
    .tm-name {{
        font-size: 18px;
        font-weight: 700;
    }}
    .tm-logo {{
        width: 36px;
        height: 36px;
    }}
    .vs {{
        font-size: 13px;
        font-weight: 600;
        color: {MUTED};
        background: {SURFACE};
        padding: 4px 10px;
        border-radius: 6px;
    }}

    /* Probability Bar */
    .pb {{
        height: 40px;
        border-radius: 8px;
        display: flex;
        overflow: hidden;
        font-weight: 700;
        font-size: 12px;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 14px;
    }}
    .pb > div {{
        display: flex;
        align-items: center;
        justify-content: center;
        min-width: 36px;
    }}

    /* Odds Grid */
    .og {{
        display: grid;
        grid-template-columns: 70px minmax(0,1fr) minmax(0,1fr) minmax(0,1fr);
        gap: 2px 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 12px;
        background: {SURFACE};
        border-radius: 8px;
        padding: 10px 14px;
        margin-bottom: 14px;
    }}
    .og-h {{
        color: {MUTED};
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        padding-bottom: 6px;
        border-bottom: 1px solid {BORDER};
    }}
    .og-c {{ padding: 5px 0; text-align: center; }}
    .og-l {{ text-align: left; color: {MUTED}; padding: 5px 0; }}
    .og-pos {{ color: {GREEN}; font-weight: 700; }}
    .og-neg {{ color: {MUTED}; }}

    /* Edge + Kelly */
    .edge-tag {{
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: {GREEN}15;
        border: 1px solid {GREEN}40;
        border-radius: 8px;
        padding: 6px 14px;
        font-size: 13px;
        font-weight: 700;
        color: {GREEN};
        font-family: 'JetBrains Mono', monospace;
    }}
    .kelly {{
        background: linear-gradient(135deg, #064E3B, #065F46);
        border: 1px solid {GREEN}30;
        border-radius: 10px;
        padding: 14px 20px;
        margin-top: 10px;
        display: flex;
        align-items: center;
        gap: 14px;
    }}
    .kelly-val {{
        font-family: 'JetBrains Mono', monospace;
        font-size: 22px;
        font-weight: 800;
        color: {GREEN};
    }}
    .kelly-txt {{
        font-size: 13px;
        color: #9CA3AF;
    }}
    .kelly-txt strong {{ color: {WHITE}; }}
    .skip {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 10px 18px;
        margin-top: 10px;
        color: {MUTED};
        font-size: 13px;
        text-align: center;
    }}

    /* Details */
    .det-grid {{
        display: grid;
        grid-template-columns: 1fr 80px 1fr;
        gap: 6px;
        font-size: 12px;
    }}
    .det-l {{
        text-align: right;
        font-family: 'JetBrains Mono', monospace;
    }}
    .det-c {{
        text-align: center;
        color: {MUTED};
        font-size: 11px;
    }}
    .det-r {{
        text-align: left;
        font-family: 'JetBrains Mono', monospace;
    }}
    .det-sep {{
        grid-column: 1 / -1;
        border-top: 1px solid {BORDER};
        margin: 2px 0;
    }}
    .form-dot {{
        display: inline-block;
        width: 20px;
        height: 20px;
        border-radius: 50%;
        text-align: center;
        line-height: 20px;
        font-size: 9px;
        font-weight: 800;
        margin: 1px;
    }}

    /* Bet Recommendation */
    .bet-rec {{
        border-radius: 10px;
        padding: 14px 20px;
        margin-top: 10px;
        font-family: 'JetBrains Mono', monospace;
    }}
    .bet-green {{
        background: linear-gradient(135deg, #064E3B, #065F46);
        border: 1px solid {GREEN}30;
        color: {GREEN};
    }}
    .bet-yellow {{
        background: linear-gradient(135deg, #78350F, #92400E);
        border: 1px solid {YELLOW}30;
        color: {YELLOW};
    }}
    .bet-red {{
        background: linear-gradient(135deg, #7F1D1D, #991B1B);
        border: 1px solid {RED}30;
        color: {RED};
    }}
    .bet-rec-title {{
        font-size: 15px;
        font-weight: 800;
        margin-bottom: 4px;
    }}
    .bet-rec-detail {{
        font-size: 13px;
        color: #9CA3AF;
    }}
    .bet-rec-detail strong {{ color: {WHITE}; }}

    /* Disclaimer banner */
    .disclaimer-banner {{
        background: {SURFACE};
        border: 1px solid {YELLOW}40;
        border-radius: 10px;
        padding: 14px 20px;
        margin-bottom: 20px;
        font-size: 13px;
        color: {YELLOW};
        text-align: center;
    }}

    /* Footer */
    .foot {{
        text-align: center;
        color: {MUTED};
        font-size: 11px;
        margin-top: 40px;
        padding: 20px;
        border-top: 1px solid {BORDER};
    }}

    /* Streamlit overrides */
    div[data-testid="stExpander"] {{
        background: {SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        margin-bottom: 14px;
    }}
    div[data-testid="stExpander"] summary {{
        font-size: 13px;
        color: {MUTED};
    }}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@st.cache_data
def load_features():
    return pd.read_parquet(FEATURES_PATH)


@st.cache_resource
def load_stacking():
    config = joblib.load(MODELS_DIR / "stack_config.joblib")
    meta = joblib.load(MODELS_DIR / "stack_meta.joblib")
    bases = {}
    for name in config["base_model_names"]:
        bases[name] = joblib.load(MODELS_DIR / f"stack_base_{name}.joblib")
    return config, meta, bases


def load_upcoming():
    if not UPCOMING_PATH.exists():
        return []
    with open(UPCOMING_PATH) as f:
        return json.load(f)


def load_odds():
    if not ODDS_PATH.exists():
        return {}
    with open(ODDS_PATH) as f:
        return json.load(f)


def get_last_update():
    if UPCOMING_PATH.exists():
        mtime = os.path.getmtime(UPCOMING_PATH)
        return datetime.fromtimestamp(mtime).strftime("%b %d, %Y %H:%M")
    return "N/A"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PREDICTION LOGIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def get_team_latest_features(df, team_id, is_home=True):
    prefix = "home" if is_home else "away"
    id_col = f"{prefix}_team_id"
    matches = df[df[id_col] == team_id].sort_values("date")
    if len(matches) == 0:
        return None
    return matches.iloc[-1]


def build_feature_vector(df, feat_cols, home_name, away_name):
    home_rows = df[df["home_team"] == home_name]
    away_rows = df[df["away_team"] == away_name]
    if len(home_rows) == 0 or len(away_rows) == 0:
        return None

    home_id = home_rows["home_team_id"].iloc[0]
    away_id = away_rows["away_team_id"].iloc[0]

    latest_home = get_team_latest_features(df, home_id, is_home=True)
    latest_away = get_team_latest_features(df, away_id, is_home=False)

    if latest_home is None or latest_away is None:
        return None

    vector = {}
    for col in feat_cols:
        if col.startswith("home_"):
            vector[col] = latest_home.get(col, np.nan)
        elif col.startswith("away_"):
            vector[col] = latest_away.get(col, np.nan)
        elif col == "elo_diff":
            vector[col] = latest_home.get("home_elo", 1500) - latest_away.get("away_elo", 1500)
        elif col in latest_home.index:
            vector[col] = latest_home.get(col, np.nan)
        else:
            vector[col] = np.nan

    series = pd.Series(vector)
    for col in series.index:
        if pd.isna(series[col]):
            col_data = df[col] if col in df.columns else pd.Series([0])
            series[col] = col_data.median()

    return series[feat_cols].values.astype(np.float64).reshape(1, -1)


def predict_match(X, config, meta, bases):
    meta_features = []
    for name in config["base_model_names"]:
        meta_features.append(bases[name].predict_proba(X))
    X_meta = np.hstack(meta_features)
    proba = meta.predict_proba(X_meta)[0]
    return dict(zip(LABEL_ORDER, proba))


def get_team_form(df, team_name, n=5):
    mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    team_matches = df[mask].sort_values("date").tail(n)
    results = []
    for _, row in team_matches.iterrows():
        if row["home_team"] == team_name:
            results.append(row["result"])
        else:
            results.append({"H": "A", "D": "D", "A": "H"}[row["result"]])
    return results


def get_team_elo(df, team_name):
    hm = df[df["home_team"] == team_name].sort_values("date")
    am = df[df["away_team"] == team_name].sort_values("date")
    latest_h = hm.iloc[-1] if len(hm) > 0 else None
    latest_a = am.iloc[-1] if len(am) > 0 else None
    if latest_h is not None and latest_a is not None:
        if latest_h["date"] >= latest_a["date"]:
            return latest_h["home_elo"]
        return latest_a["away_elo"]
    if latest_h is not None:
        return latest_h["home_elo"]
    if latest_a is not None:
        return latest_a["away_elo"]
    return 1500


def get_team_stats(df, team_name):
    hm = df[df["home_team"] == team_name].sort_values("date")
    am = df[df["away_team"] == team_name].sort_values("date")
    latest_h = hm.iloc[-1] if len(hm) > 0 else None
    latest_a = am.iloc[-1] if len(am) > 0 else None

    if latest_h is not None and latest_a is not None:
        if latest_h["date"] >= latest_a["date"]:
            r, p = latest_h, "home"
        else:
            r, p = latest_a, "away"
    elif latest_h is not None:
        r, p = latest_h, "home"
    elif latest_a is not None:
        r, p = latest_a, "away"
    else:
        return {}

    return {
        "goals": r.get(f"{p}_goals_avg", 0),
        "xg": r.get(f"{p}_xg_avg", 0),
        "pts": r.get(f"{p}_points_last_n", 0),
        "gd": r.get(f"{p}_goal_diff_rolling", 0),
        "sot": r.get(f"{p}_shots_on_target_avg", 0),
        "poss": r.get(f"{p}_possession_avg", 0),
    }


def kelly(prob, odds):
    if odds <= 1:
        return 0.0
    return max(0.0, (prob * odds - 1) / (odds - 1))


def decimal_to_american(decimal_odds):
    if decimal_odds >= 2.0:
        return f"+{(decimal_odds - 1) * 100:.0f}"
    else:
        return f"{-100 / (decimal_odds - 1):.0f}"


def american_to_decimal(american_odds):
    if american_odds > 0:
        return american_odds / 100.0 + 1.0
    elif american_odds < 0:
        return 100.0 / abs(american_odds) + 1.0
    return 1.0


def fmt_odds(decimal_odds):
    return f"{decimal_odds:.2f} / {decimal_to_american(decimal_odds)}"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# ANALYSIS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def analyze_all(df, config, meta, bases, upcoming, odds_data):
    feat_cols = config["feature_cols"]
    results = []

    for fx in upcoming:
        home = fx["teams"]["home"]["name"]
        away = fx["teams"]["away"]["name"]
        fid = str(fx["fixture"]["id"])

        X = build_feature_vector(df, feat_cols, home, away)
        if X is None:
            continue

        probs = predict_match(X, config, meta, bases)
        odds_match = odds_data.get(fid, {})
        oh = odds_match.get("home")
        od = odds_match.get("draw")
        oa = odds_match.get("away")

        best_edge = 0.0
        best_outcome = None
        best_kelly = 0.0
        best_odds = 0.0
        edges = {}

        if oh and od and oa:
            tot = 1 / oh + 1 / od + 1 / oa
            impl = {"H": (1 / oh) / tot, "D": (1 / od) / tot, "A": (1 / oa) / tot}
            odds_map = {"H": oh, "D": od, "A": oa}
            for o in LABEL_ORDER:
                edges[o] = probs[o] - impl[o]
                k = kelly(probs[o], odds_map[o])
                if edges[o] > best_edge:
                    best_edge = edges[o]
                    best_outcome = o
                    best_kelly = k
                    best_odds = odds_map[o]

        round_str = fx["league"].get("round", "")
        round_num = round_str.split(" - ")[-1] if " - " in round_str else round_str

        results.append({
            "home": home,
            "away": away,
            "home_logo": fx["teams"]["home"]["logo"],
            "away_logo": fx["teams"]["away"]["logo"],
            "date": fx["fixture"].get("date_display", ""),
            "date_sort": fx["fixture"].get("date", ""),
            "round": round_num,
            "probs": probs,
            "oh": oh, "od": od, "oa": oa,
            "bookie": odds_match.get("bookmaker", ""),
            "edges": edges,
            "best_edge": best_edge,
            "best_outcome": best_outcome,
            "best_kelly": best_kelly,
            "best_odds": best_odds,
            "home_form": get_team_form(df, home),
            "away_form": get_team_form(df, away),
            "home_elo": get_team_elo(df, home),
            "away_elo": get_team_elo(df, away),
            "home_stats": get_team_stats(df, home),
            "away_stats": get_team_stats(df, away),
        })

    return results


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# RENDERING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def form_dots(results):
    colors = {"H": GREEN, "D": YELLOW, "A": RED}
    letters = {"H": "W", "D": "D", "A": "L"}
    text_colors = {"H": "#000", "D": "#000", "A": "#fff"}
    html = ""
    for r in results:
        html += f'<span class="form-dot" style="background:{colors[r]};color:{text_colors[r]};">{letters[r]}</span>'
    return html


def render_card_html(m, bankroll):
    has_edge = m["best_edge"] > 0.05
    cls = "mc mc-edge" if has_edge else "mc mc-skip"

    html = f"""<div class="{cls}">
    <div class="mc-head">
        <span class="mc-date">{m['date']} (CDMX)</span>
        <span class="mc-round">Jornada {m['round']}</span>
    </div>
    <div class="teams">
        <div class="tm tm-h">
            <span class="tm-name">{m['home']}</span>
            <img class="tm-logo" src="{m['home_logo']}" alt="">
        </div>
        <span class="vs">VS</span>
        <div class="tm tm-a">
            <img class="tm-logo" src="{m['away_logo']}" alt="">
            <span class="tm-name">{m['away']}</span>
        </div>
    </div>
    <div class="pb">
        <div style="width:{m['probs']['H']*100:.0f}%;background:{BLUE};">H {m['probs']['H']*100:.0f}%</div>
        <div style="width:{m['probs']['D']*100:.0f}%;background:{YELLOW};color:#000;">D {m['probs']['D']*100:.0f}%</div>
        <div style="width:{m['probs']['A']*100:.0f}%;background:{RED};">A {m['probs']['A']*100:.0f}%</div>
    </div>"""

    if m["oh"] and m["od"] and m["oa"]:
        tot = 1 / m["oh"] + 1 / m["od"] + 1 / m["oa"]
        ih = (1 / m["oh"]) / tot * 100
        id_ = (1 / m["od"]) / tot * 100
        ia = (1 / m["oa"]) / tot * 100
        e = m["edges"]

        def ec(v):
            return "og-pos" if v > 0 else "og-neg"

        html += f"""<div class="og">
        <div class="og-h"></div>
        <div class="og-h" style="text-align:center">Home</div>
        <div class="og-h" style="text-align:center">Draw</div>
        <div class="og-h" style="text-align:center">Away</div>
        <div class="og-l">Modelo</div>
        <div class="og-c">{m['probs']['H']*100:.1f}%</div>
        <div class="og-c">{m['probs']['D']*100:.1f}%</div>
        <div class="og-c">{m['probs']['A']*100:.1f}%</div>
        <div class="og-l">Odds</div>
        <div class="og-c">{fmt_odds(m['oh'])}</div>
        <div class="og-c">{fmt_odds(m['od'])}</div>
        <div class="og-c">{fmt_odds(m['oa'])}</div>
        <div class="og-l">Casa</div>
        <div class="og-c">{ih:.1f}%</div>
        <div class="og-c">{id_:.1f}%</div>
        <div class="og-c">{ia:.1f}%</div>
        <div class="og-l">Edge</div>
        <div class="og-c {ec(e.get('H',0))}">{e.get('H',0)*100:+.1f}%</div>
        <div class="og-c {ec(e.get('D',0))}">{e.get('D',0)*100:+.1f}%</div>
        <div class="og-c {ec(e.get('A',0))}">{e.get('A',0)*100:+.1f}%</div>
        </div>"""

    if has_edge:
        oname = LABEL_NAMES[m["best_outcome"]]
        kelly_pct = m["best_kelly"] * 100
        model_prob = m["probs"][m["best_outcome"]] * 100
        # Implied probability from bookmaker
        impl_prob = (1 / m["best_odds"]) / (1 / m["oh"] + 1 / m["od"] + 1 / m["oa"]) * 100

        html += f"""<div style="background:{SURFACE};border-radius:8px;padding:14px 18px;margin-bottom:10px;font-size:13px;line-height:1.8;">
        <span style="color:{MUTED};">Nuestro modelo:</span> <strong style="color:{WHITE};">{oname} {model_prob:.1f}%</strong><br>
        <span style="color:{MUTED};">Casa de apuestas:</span> <strong style="color:{WHITE};">{oname} {impl_prob:.1f}%</strong><br>
        <span style="color:{MUTED};">Edge detectado:</span> <strong style="color:{GREEN};">+{m['best_edge']*100:.1f}%</strong><br>
        <span style="color:{MUTED};">Momio:</span> <strong style="color:{WHITE};">{fmt_odds(m['best_odds'])}</strong>
        </div>"""

        # Bet recommendation with bankroll
        bet_full = bankroll * m["best_kelly"]
        bet_min = bet_full * 0.5
        bet_max = bet_full

        if kelly_pct < 2:
            rec_cls = "bet-rec bet-red"
            rec_icon = "NO APUESTES — Edge insuficiente para cubrir varianza"
            bet_line = ""
        elif kelly_pct <= 5:
            rec_cls = "bet-rec bet-yellow"
            rec_icon = "APUESTA PEQUENA — Riesgo moderado"
            bet_line = f'<div class="bet-rec-detail">Apostar entre <strong>${bet_min:,.0f}</strong> y <strong>${bet_max:,.0f}</strong> pesos<br>(Kelly {kelly_pct:.1f}% de tu bankroll de ${bankroll:,.0f})</div>'
        else:
            rec_cls = "bet-rec bet-green"
            rec_icon = "APUESTA RECOMENDADA"
            bet_line = f'<div class="bet-rec-detail">Apostar entre <strong>${bet_min:,.0f}</strong> y <strong>${bet_max:,.0f}</strong> pesos<br>(Kelly {kelly_pct:.1f}% de tu bankroll de ${bankroll:,.0f})</div>'

        html += f"""<div class="{rec_cls}">
            <div class="bet-rec-title">{rec_icon}</div>
            {bet_line}
        </div>"""
    else:
        html += '<div class="skip">SKIP — No edge &gt; 5%</div>'

    html += "</div>"
    return html


def render_details(m):
    """Render match details inside a Streamlit expander."""
    c1, c2 = st.columns(2)

    with c1:
        st.markdown(f"**{m['home']}**")
        st.markdown(form_dots(m["home_form"]), unsafe_allow_html=True)
        wins = sum(1 for r in m["home_form"] if r == "H")
        draws = sum(1 for r in m["home_form"] if r == "D")
        losses = sum(1 for r in m["home_form"] if r == "A")
        st.caption(f"{wins}W {draws}D {losses}L")

    with c2:
        st.markdown(f"**{m['away']}**")
        st.markdown(form_dots(m["away_form"]), unsafe_allow_html=True)
        wins = sum(1 for r in m["away_form"] if r == "H")
        draws = sum(1 for r in m["away_form"] if r == "D")
        losses = sum(1 for r in m["away_form"] if r == "A")
        st.caption(f"{wins}W {draws}D {losses}L")

    hs = m["home_stats"]
    aws = m["away_stats"]

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "—"
        if isinstance(v, float):
            return f"{v:.1f}" if v != int(v) else f"{int(v)}"
        return str(v)

    detail_html = f"""<div class="det-grid">
    <div class="det-l" style="font-weight:700;color:{GREEN};">{m['home_elo']:.0f}</div>
    <div class="det-c">ELO</div>
    <div class="det-r" style="font-weight:700;color:{BLUE};">{m['away_elo']:.0f}</div>
    <div class="det-sep"></div>
    <div class="det-l">{fmt(hs.get('goals'))}</div>
    <div class="det-c">Goals/g</div>
    <div class="det-r">{fmt(aws.get('goals'))}</div>
    <div class="det-l">{fmt(hs.get('xg'))}</div>
    <div class="det-c">xG/g</div>
    <div class="det-r">{fmt(aws.get('xg'))}</div>
    <div class="det-sep"></div>
    <div class="det-l">{fmt(hs.get('pts'))}</div>
    <div class="det-c">Pts L5</div>
    <div class="det-r">{fmt(aws.get('pts'))}</div>
    <div class="det-l">{fmt(hs.get('gd'))}</div>
    <div class="det-c">GD L5</div>
    <div class="det-r">{fmt(aws.get('gd'))}</div>
    <div class="det-sep"></div>
    <div class="det-l">{fmt(hs.get('sot'))}</div>
    <div class="det-c">SoT/g</div>
    <div class="det-r">{fmt(aws.get('sot'))}</div>
    <div class="det-l">{fmt(hs.get('poss'))}</div>
    <div class="det-c">Poss %</div>
    <div class="det-r">{fmt(aws.get('poss'))}</div>
    </div>"""
    st.markdown(detail_html, unsafe_allow_html=True)


def render_calculadora(m, bankroll):
    """Calculadora de momio — user inputs their sportsbook odds."""
    key_base = f"{m['home']}_{m['away']}".replace(" ", "_").replace(".", "")

    # Outcome selector
    labels = [f"Home ({m['home']})", "Empate", f"Away ({m['away']})"]
    keys_list = ["H", "D", "A"]

    if m["best_outcome"]:
        def_idx = keys_list.index(m["best_outcome"])
    else:
        def_idx = keys_list.index(max(keys_list, key=lambda k: m["probs"][k]))

    c_sel, c_fmt = st.columns([3, 2])
    with c_sel:
        sel_label = st.selectbox(
            "Resultado a apostar:", labels, index=def_idx, key=f"{key_base}_sel"
        )
    outcome_key = keys_list[labels.index(sel_label)]
    model_prob = m["probs"][outcome_key]

    with c_fmt:
        momio_fmt = st.radio(
            "Formato momio:", ["Decimal", "Americano"],
            horizontal=True, key=f"{key_base}_fmt"
        )

    if momio_fmt == "Decimal":
        momio_dec = st.number_input(
            "Momio actual (decimal):", min_value=1.01, max_value=100.0,
            value=2.00, step=0.05, format="%.2f", key=f"{key_base}_dec"
        )
    else:
        momio_am = st.number_input(
            "Momio actual (americano):", min_value=-10000, max_value=10000,
            value=100, step=5, key=f"{key_base}_am"
        )
        if momio_am == 0:
            st.warning("Momio americano no puede ser 0.")
            return
        momio_dec = american_to_decimal(momio_am)

    am_str = decimal_to_american(momio_dec)
    st.caption(f"{momio_dec:.2f} decimal = {am_str} americano")

    # Calculations
    implied_prob = 1.0 / momio_dec
    edge = model_prob - implied_prob
    edge_pct = edge * 100
    k = kelly(model_prob, momio_dec)
    kelly_pct = k * 100
    stake = bankroll * k
    win_net = stake * (momio_dec - 1)
    breakeven_ratio = momio_dec

    # Edge classification
    if edge_pct > 5:
        edge_color = GREEN
        edge_verdict = "✅ Sigue siendo buen bet"
    elif edge_pct > 2:
        edge_color = YELLOW
        edge_verdict = "⚠️ Edge marginal, precaución"
    else:
        edge_color = RED
        edge_verdict = "❌ Con ese momio ya NO conviene apostar"

    # Results display
    calc_html = f"""<div style="background:{SURFACE};border-radius:8px;padding:14px 18px;
    font-family:'JetBrains Mono',monospace;font-size:13px;line-height:2.2;">
    <strong style="color:{WHITE};">Con ese momio:</strong><br>
    <span style="color:{MUTED};">Edge:</span>
    <span style="color:{edge_color};font-weight:700;">{edge_pct:+.1f}%</span> {edge_verdict}<br>"""

    if k > 0.001:
        calc_html += f"""<span style="color:{MUTED};">Apostar:</span>
    <strong style="color:{WHITE};">${stake:,.0f} pesos</strong>
    <span style="color:{MUTED};">({kelly_pct:.1f}% de ${bankroll:,.2f})</span><br>
    <span style="color:{MUTED};">Si ganas:</span>
    <strong style="color:{GREEN};">+${win_net:,.0f} pesos netos</strong>
    <span style="color:{MUTED};">(recibes ${stake + win_net:,.0f} total)</span><br>
    <span style="color:{MUTED};">Si pierdes:</span>
    <strong style="color:{RED};">-${stake:,.0f} pesos</strong><br>"""
    else:
        calc_html += f"""<span style="color:{MUTED};">Apostar:</span>
    <strong style="color:{RED};">$0 — Kelly no recomienda</strong><br>"""

    calc_html += f"""<span style="color:{MUTED};">Break-even:</span>
    <span style="color:{WHITE};">necesitas acertar 1 de cada {breakeven_ratio:.1f} veces para ser rentable</span><br>
    <span style="color:{MUTED};">Prob. mínima:</span>
    <span style="color:{WHITE};">{implied_prob*100:.1f}%</span>
    <span style="color:{MUTED};">· Modelo dice:</span>
    <strong style="color:{WHITE};">{model_prob*100:.1f}%</strong>
    </div>"""

    st.markdown(calc_html, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN PAGE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

df = load_features()
config, meta, bases = load_stacking()
upcoming = load_upcoming()
odds_data = load_odds()
last_update = get_last_update()

# ── Sidebar: Bankroll ──
with st.sidebar:
    bankroll = st.number_input(
        "Mi bankroll (MXN $)",
        min_value=100.0,
        max_value=1_000_000.0,
        value=668.93,
        step=50.0,
    )

# ── Hero Header ──
st.markdown(f"""
<div class="hero">
    <div class="hero-row">
        <div>
            <div class="hero-title">⚽ Ligue 1 Predictor</div>
            <div class="hero-sub">Stacking Ensemble &middot; LR + RF + XGBoost + MLP &middot; {len(df)} matches trained</div>
        </div>
        <div class="hero-update">
            Last update<br><strong>{last_update}</strong>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Analyze ──
if upcoming:
    matches = analyze_all(df, config, meta, bases, upcoming, odds_data)
else:
    matches = []

if not matches:
    st.warning("No upcoming fixtures found. Run `python src/data/update.py` to fetch data.")
    st.stop()

# ── Disclaimer Banner ──
st.markdown(f"""
<div class="disclaimer-banner">
    Este modelo acierta el 55.7% de los partidos en datos historicos.
    Las apuestas deportivas implican riesgo real. Nunca apuestes mas de lo que puedes perder.
</div>
""", unsafe_allow_html=True)

# ── Summary Metrics ──
n_edge = sum(1 for m in matches if m["best_edge"] > 0.05)
best_edge_match = max(matches, key=lambda x: x["best_edge"])
best_edge_pct = best_edge_match["best_edge"] * 100

st.markdown(f"""
<div class="summary-row">
    <div class="summary-box">
        <div class="summary-val">{len(matches)}</div>
        <div class="summary-label">Matches</div>
    </div>
    <div class="summary-box">
        <div class="summary-val" style="color:{GREEN};">{n_edge}</div>
        <div class="summary-label">With Edge</div>
    </div>
    <div class="summary-box">
        <div class="summary-val" style="color:{BLUE};">{best_edge_pct:.1f}%</div>
        <div class="summary-label">Best Edge</div>
    </div>
    <div class="summary-box">
        <div class="summary-val" style="color:{GREEN};">55.7%</div>
        <div class="summary-label">Model Accuracy</div>
    </div>
</div>
""", unsafe_allow_html=True)

st.caption("55.7% Accuracy (vs 33% aleatorio) — En futbol, predecir 1 de cada 3 resultados correctamente es el azar. Nuestro modelo acierta 1 de cada 1.8.")

# ── Sort Toggle ──
rounds = sorted(set(m["round"] for m in matches))
rounds_str = ", ".join(f"J{r}" for r in rounds)
st.markdown(f"### Upcoming Fixtures — {rounds_str}")

sort_mode = st.radio(
    "Ordenar por:",
    options=["Fecha", "Edge"],
    horizontal=True,
    index=0,
)

if sort_mode == "Fecha":
    matches.sort(key=lambda x: x["date_sort"])
    st.caption("Ordenado por fecha (mas proximo primero). Verde = apuesta recomendada. Gris = skip.")
else:
    matches.sort(key=lambda x: x["best_edge"], reverse=True)
    st.caption("Ordenado por mayor edge. Verde = apuesta recomendada. Gris = skip.")

# ── Match Cards ──
for m in matches:
    card_html = render_card_html(m, bankroll)
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"Details — {m['home']} vs {m['away']}"):
        render_details(m)

    with st.expander(f"🧮 Calculadora de Momio — {m['home']} vs {m['away']}"):
        render_calculadora(m, bankroll)

# ── Footer ──
st.markdown(f"""
<div class="foot">
    <strong>Disclaimer:</strong> Este modelo acierta el 55.7% de los partidos en datos historicos.
    Las apuestas deportivas implican riesgo real. Nunca apuestes mas de lo que puedes perder.<br><br>
    Ligue 1 Predictor v2.0 &middot; Stacking Ensemble (LR + RF + XGB + MLP)
    &middot; Data: API-Football 2021-2025 ({len(df)} matches)
</div>
""", unsafe_allow_html=True)
