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
# RECOMMENDATION ENGINE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DOUBLE_OPP_LABELS = {
    "1X": "Local o Empate",
    "X2": "Empate o Visitante",
    "12": "Local o Visitante",
}


def get_recommendation(probs, edges, kelly_vals, implied, odds_map):
    """
    5-rule decision engine. Returns:
      (action, reason, bet_type, outcome_key, bet_odds, bet_prob, bet_edge, bet_kelly)
      action: "BET" or "SKIP"
      bet_type: "directa" | "doble_oportunidad" | None
    """
    ph, pd_, pa = probs["H"], probs["D"], probs["A"]
    eh, ed, ea = edges.get("H", 0), edges.get("D", 0), edges.get("A", 0)
    kh, kd, ka = kelly_vals.get("H", 0), kelly_vals.get("D", 0), kelly_vals.get("A", 0)

    # REGLA 1: ningún outcome tiene edge > 5% → check doble oportunidad first
    max_single_edge = max(eh, ed, ea)

    # REGLA 2: Si el mejor outcome tiene prob > 55% Y edge > 5% → APUESTA DIRECTA
    for key, prob, edge, k, odds in [
        ("H", ph, eh, kh, odds_map.get("H", 0)),
        ("D", pd_, ed, kd, odds_map.get("D", 0)),
        ("A", pa, ea, ka, odds_map.get("A", 0)),
    ]:
        if prob > 0.55 and edge > 0.05:
            # REGLA 5: Kelly < 2% → SKIP
            if k < 0.02:
                return ("SKIP", "Kelly insuficiente — varianza muy alta", None, None, 0, 0, 0, 0)
            return ("BET", None, "directa", key, odds, prob, edge, k)

    # REGLA 3: top 2 probs within 10% → SKIP
    top_two = sorted([ph, pd_, pa], reverse=True)[:2]
    if top_two[0] - top_two[1] < 0.10:
        return ("SKIP", "Modelo sin conviccion — probabilidades muy parejas", None, None, 0, 0, 0, 0)

    # REGLA 4: DOBLE OPORTUNIDAD
    double_opps = {
        "1X": {"prob": ph + pd_, "keys": ("H", "D"), "implied": implied.get("H", 0) + implied.get("D", 0)},
        "X2": {"prob": pd_ + pa, "keys": ("D", "A"), "implied": implied.get("D", 0) + implied.get("A", 0)},
        "12": {"prob": ph + pa, "keys": ("H", "A"), "implied": implied.get("H", 0) + implied.get("A", 0)},
    }

    best_do = None
    best_do_edge = 0
    for do_key, do_data in double_opps.items():
        do_edge = do_data["prob"] - do_data["implied"]
        if do_data["prob"] > 0.72 and do_edge > 0.05:
            # Compute combined odds: 1 / (raw sum of implied probs with vig)
            k1, k2 = do_data["keys"]
            raw_combined = (1 / odds_map.get(k1, 999)) + (1 / odds_map.get(k2, 999))
            do_odds = 1 / raw_combined if raw_combined > 0 else 1.0
            do_kelly = kelly(do_data["prob"], do_odds)
            if do_kelly >= 0.02 and do_edge > best_do_edge:
                best_do = (do_key, do_odds, do_data["prob"], do_edge, do_kelly)
                best_do_edge = do_edge

    if best_do:
        do_key, do_odds, do_prob, do_edge, do_kelly = best_do
        return ("BET", None, "doble_oportunidad", do_key, do_odds, do_prob, do_edge, do_kelly)

    # If we have a single edge > 5% but prob <= 55% and no double opp qualifies
    if max_single_edge > 0.05:
        # Find the best single outcome
        for key, prob, edge, k, odds in sorted(
            [("H", ph, eh, kh, odds_map.get("H", 0)),
             ("D", pd_, ed, kd, odds_map.get("D", 0)),
             ("A", pa, ea, ka, odds_map.get("A", 0))],
            key=lambda x: x[2], reverse=True
        ):
            if edge > 0.05:
                if k < 0.02:
                    return ("SKIP", "Kelly insuficiente — varianza muy alta", None, None, 0, 0, 0, 0)
                return ("BET", None, "directa", key, odds, prob, edge, k)

    # REGLA 1: No edge
    return ("SKIP", "Sin edge suficiente vs la casa", None, None, 0, 0, 0, 0)


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

        edges = {}
        kelly_vals = {}
        implied = {}
        odds_map = {}

        if oh and od and oa:
            tot = 1 / oh + 1 / od + 1 / oa
            implied = {"H": (1 / oh) / tot, "D": (1 / od) / tot, "A": (1 / oa) / tot}
            odds_map = {"H": oh, "D": od, "A": oa}
            for o in LABEL_ORDER:
                edges[o] = probs[o] - implied[o]
                kelly_vals[o] = kelly(probs[o], odds_map[o])

        # Run recommendation engine
        rec = get_recommendation(probs, edges, kelly_vals, implied, odds_map)
        action, skip_reason, bet_type, bet_key, bet_odds, bet_prob, bet_edge, bet_kelly = rec

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
            "kelly_vals": kelly_vals,
            "implied": implied,
            "odds_map": odds_map,
            # Recommendation
            "action": action,
            "skip_reason": skip_reason,
            "bet_type": bet_type,
            "bet_key": bet_key,
            "bet_odds": bet_odds,
            "bet_prob": bet_prob,
            "bet_edge": bet_edge,
            "bet_kelly": bet_kelly,
            # Context
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


def _outcome_display_name(m, key):
    """Human-readable outcome name."""
    if key == "H":
        return m["home"]
    elif key == "A":
        return m["away"]
    elif key == "D":
        return "Empate"
    # Double opportunity keys
    elif key in DOUBLE_OPP_LABELS:
        return DOUBLE_OPP_LABELS[key]
    return key


def _outcome_side_label(key):
    """Short side label."""
    return {"H": "Local", "D": "Empate", "A": "Visitante"}.get(key, key)


def render_bet_card(m, bankroll):
    """Render a BET card — green border, action-oriented."""
    stake = bankroll * m["bet_kelly"]
    win_net = stake * (m["bet_odds"] - 1)

    # Bet description
    if m["bet_type"] == "directa":
        bet_label = f"{_outcome_display_name(m, m['bet_key'])} ({_outcome_side_label(m['bet_key'])})"
        tipo = "Apuesta directa"
        momio_str = f"{m['bet_odds']:.2f} / {decimal_to_american(m['bet_odds'])}"
    else:
        # Doble oportunidad
        do_key = m["bet_key"]
        do_map = {"1X": "1X", "X2": "X2", "12": "12"}
        if do_key == "1X":
            bet_label = f"{m['home']} o Empate (Doble Oportunidad)"
        elif do_key == "X2":
            bet_label = f"Empate o {m['away']} (Doble Oportunidad)"
        else:
            bet_label = f"{m['home']} o {m['away']} (Doble Oportunidad)"
        tipo = f"Doble Oportunidad {do_map[do_key]}"
        momio_str = f"{m['bet_odds']:.2f} / {decimal_to_american(m['bet_odds'])}"

    html = f"""<div class="mc mc-edge">
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
    <div style="background:linear-gradient(135deg,#064E3B,#065F46);border:1px solid {GREEN}30;border-radius:10px;padding:16px 20px;font-family:'JetBrains Mono',monospace;">
        <div style="font-size:16px;font-weight:800;color:{GREEN};margin-bottom:8px;">APOSTAR: {bet_label}</div>
        <div style="font-size:12px;color:{MUTED};margin-bottom:12px;">Tipo: {tipo}</div>
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;font-size:13px;">
            <div>
                <span style="color:{MUTED};">Momio:</span>
                <strong style="color:{WHITE};">{momio_str}</strong>
            </div>
            <div>
                <span style="color:{MUTED};">Edge vs casa:</span>
                <strong style="color:{GREEN};">+{m['bet_edge']*100:.1f}%</strong>
            </div>
        </div>
        <div style="border-top:1px solid {GREEN}30;margin:12px 0;"></div>
        <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:13px;">
            <div>
                <span style="color:{MUTED};">Apostar:</span><br>
                <strong style="color:{WHITE};font-size:16px;">${stake:,.0f}</strong>
                <span style="color:{MUTED};font-size:11px;"> pesos</span>
            </div>
            <div>
                <span style="color:{MUTED};">Si ganas:</span><br>
                <strong style="color:{GREEN};font-size:16px;">+${win_net:,.0f}</strong>
                <span style="color:{MUTED};font-size:11px;"> pesos</span>
            </div>
            <div>
                <span style="color:{MUTED};">Si pierdes:</span><br>
                <strong style="color:{RED};font-size:16px;">-${stake:,.0f}</strong>
                <span style="color:{MUTED};font-size:11px;"> pesos</span>
            </div>
        </div>
    </div>
    </div>"""
    return html


def render_skip_card(m):
    """Render a SKIP card — grey, muted, with reason."""
    reason = m["skip_reason"] or "Sin edge suficiente vs la casa"

    html = f"""<div class="mc mc-skip">
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
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:8px;padding:14px 18px;text-align:center;">
        <div style="font-size:14px;font-weight:700;color:{MUTED};margin-bottom:6px;">SKIP</div>
        <div style="font-size:13px;color:{MUTED};">{reason}</div>
        <div style="font-size:12px;color:{SUBTLE};margin-top:6px;">
            ({m['home'][:3].upper()} {m['probs']['H']*100:.0f}% · Emp {m['probs']['D']*100:.0f}% · {m['away'][:3].upper()} {m['probs']['A']*100:.0f}%)
        </div>
    </div>
    </div>"""
    return html


def form_dots(results):
    colors = {"H": GREEN, "D": YELLOW, "A": RED}
    letters = {"H": "W", "D": "D", "A": "L"}
    text_colors = {"H": "#000", "D": "#000", "A": "#fff"}
    html = ""
    for r in results:
        html += f'<span class="form-dot" style="background:{colors[r]};color:{text_colors[r]};">{letters[r]}</span>'
    return html


def render_details(m):
    """Full details inside expander: probs, odds grid, form, stats."""
    # Probability bar
    pb_html = f"""<div class="pb">
        <div style="width:{m['probs']['H']*100:.0f}%;background:{BLUE};">H {m['probs']['H']*100:.0f}%</div>
        <div style="width:{m['probs']['D']*100:.0f}%;background:{YELLOW};color:#000;">D {m['probs']['D']*100:.0f}%</div>
        <div style="width:{m['probs']['A']*100:.0f}%;background:{RED};">A {m['probs']['A']*100:.0f}%</div>
    </div>"""
    st.markdown(pb_html, unsafe_allow_html=True)

    # Odds grid
    if m["oh"] and m["od"] and m["oa"]:
        imp = m["implied"]
        e = m["edges"]

        def ec(v):
            return "og-pos" if v > 0 else "og-neg"

        og_html = f"""<div class="og">
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
        <div class="og-c">{imp.get('H',0)*100:.1f}%</div>
        <div class="og-c">{imp.get('D',0)*100:.1f}%</div>
        <div class="og-c">{imp.get('A',0)*100:.1f}%</div>
        <div class="og-l">Edge</div>
        <div class="og-c {ec(e.get('H',0))}">{e.get('H',0)*100:+.1f}%</div>
        <div class="og-c {ec(e.get('D',0))}">{e.get('D',0)*100:+.1f}%</div>
        <div class="og-c {ec(e.get('A',0))}">{e.get('A',0)*100:+.1f}%</div>
        </div>"""
        st.markdown(og_html, unsafe_allow_html=True)

    # Kelly values
    kv = m.get("kelly_vals", {})
    if kv:
        kelly_html = f"""<div style="background:{SURFACE};border-radius:8px;padding:10px 14px;font-size:12px;font-family:'JetBrains Mono',monospace;margin-bottom:10px;">
        <span style="color:{MUTED};">Kelly:</span>
        H {kv.get('H',0)*100:.1f}% · D {kv.get('D',0)*100:.1f}% · A {kv.get('A',0)*100:.1f}%
        </div>"""
        st.markdown(kelly_html, unsafe_allow_html=True)

    # Form and stats
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
            <div class="hero-title">Ligue 1 Predictor</div>
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

# ── Sort: BET first (by edge desc), then SKIP ──
bet_matches = [m for m in matches if m["action"] == "BET"]
skip_matches = [m for m in matches if m["action"] == "SKIP"]
bet_matches.sort(key=lambda x: x["bet_edge"], reverse=True)
skip_matches.sort(key=lambda x: x["date_sort"])
sorted_matches = bet_matches + skip_matches

n_bet = len(bet_matches)
n_total = len(matches)

# ── Header: apostable count ──
if n_bet > 0:
    header_color = GREEN
    header_text = f"{n_bet} de {n_total} partidos recomendados"
else:
    header_color = MUTED
    header_text = f"0 de {n_total} partidos recomendados esta jornada"

rounds = sorted(set(m["round"] for m in matches))
rounds_str = ", ".join(f"J{r}" for r in rounds)

st.markdown(f"""
<div style="background:{CARD};border:1px solid {BORDER};border-radius:12px;padding:20px 28px;margin-bottom:20px;">
    <div style="display:flex;justify-content:space-between;align-items:center;">
        <div>
            <div style="font-size:24px;font-weight:800;color:{header_color};font-family:'JetBrains Mono',monospace;">
                {header_text}
            </div>
            <div style="font-size:13px;color:{MUTED};margin-top:4px;">Jornada {rounds_str} &middot; Bankroll: ${bankroll:,.2f}</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Disclaimer Banner ──
st.markdown(f"""
<div class="disclaimer-banner">
    Este modelo acierta el 55.7% de los partidos en datos historicos.
    Las apuestas deportivas implican riesgo real. Nunca apuestes mas de lo que puedes perder.
</div>
""", unsafe_allow_html=True)

# ── Match Cards ──
for m in sorted_matches:
    if m["action"] == "BET":
        card_html = render_bet_card(m, bankroll)
    else:
        card_html = render_skip_card(m)
    st.markdown(card_html, unsafe_allow_html=True)

    with st.expander(f"ver detalles — {m['home']} vs {m['away']}"):
        render_details(m)

# ── Footer ──
st.markdown(f"""
<div class="foot">
    <strong>Disclaimer:</strong> Este modelo acierta el 55.7% de los partidos en datos historicos.
    Las apuestas deportivas implican riesgo real. Nunca apuestes mas de lo que puedes perder.<br><br>
    Ligue 1 Predictor v3.0 &middot; Stacking Ensemble (LR + RF + XGB + MLP)
    &middot; Data: API-Football 2021-2025 ({len(df)} matches)
</div>
""", unsafe_allow_html=True)
