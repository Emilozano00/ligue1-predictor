"""
Ligue 1 Predictor — Streamlit App
3 pages: Predictor, Model Comparison, Bankroll Tracker
"""

import os
import sys
import warnings

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Paths ──
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"
MODELS_DIR = PROJECT_ROOT / "models" / "saved"
BANKROLL_PATH = PROJECT_ROOT / "data" / "bankroll.csv"

# ── Colors ──
GREEN = "#00C853"
YELLOW = "#FFD600"
RED = "#D50000"
BG_DARK = "#0E1117"
CARD_BG = "#1A1D23"
TEXT = "#FAFAFA"
TEXT_MUTED = "#8B949E"

LABEL_ORDER = ["H", "D", "A"]
LABEL_NAMES = {"H": "Local", "D": "Empate", "A": "Visita"}
LABEL_COLORS = {"H": GREEN, "D": YELLOW, "A": RED}

# ── Page config ──
st.set_page_config(
    page_title="Ligue 1 Predictor",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(f"""
<style>
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT};
    }}
    .metric-card {{
        background: {CARD_BG};
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #30363D;
    }}
    .prob-bar {{
        height: 40px;
        border-radius: 8px;
        display: flex;
        overflow: hidden;
        font-weight: 700;
        font-size: 14px;
    }}
    .prob-segment {{
        display: flex;
        align-items: center;
        justify-content: center;
        color: #000;
        min-width: 30px;
    }}
    .form-dot {{
        display: inline-block;
        width: 24px;
        height: 24px;
        border-radius: 50%;
        text-align: center;
        line-height: 24px;
        font-size: 11px;
        font-weight: 700;
        margin: 2px;
        color: #000;
    }}
    .kelly-box {{
        background: linear-gradient(135deg, #1B5E20, #2E7D32);
        border-radius: 12px;
        padding: 16px 24px;
        margin: 8px 0;
        font-size: 18px;
        font-weight: 700;
        color: white;
    }}
    .no-edge-box {{
        background: {CARD_BG};
        border: 1px solid #30363D;
        border-radius: 12px;
        padding: 16px 24px;
        margin: 8px 0;
        color: {TEXT_MUTED};
    }}
</style>
""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# DATA LOADING
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


@st.cache_data
def load_features():
    df = pd.read_parquet(FEATURES_PATH)
    df = df[df["season"] >= 2022].copy()
    return df


@st.cache_resource
def load_stacking():
    config = joblib.load(MODELS_DIR / "stack_config.joblib")
    meta = joblib.load(MODELS_DIR / "stack_meta.joblib")
    bases = {}
    for name in config["base_model_names"]:
        bases[name] = joblib.load(MODELS_DIR / f"stack_base_{name}.joblib")
    return config, meta, bases


@st.cache_data
def load_comparison():
    return pd.read_csv(MODELS_DIR / "comparison.csv")


@st.cache_data
def get_teams(df):
    latest_season = df["season"].max()
    latest = df[df["season"] == latest_season]
    teams = sorted(set(latest["home_team"].unique()) | set(latest["away_team"].unique()))
    return teams


def get_team_form(df, team_name, n=5):
    """Get last N results for a team."""
    mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    team_matches = df[mask].sort_values("date").tail(n)
    results = []
    for _, row in team_matches.iterrows():
        if row["home_team"] == team_name:
            results.append(row["result"])
        else:
            results.append({"H": "A", "D": "D", "A": "H"}[row["result"]])
    return results


def get_team_latest_features(df, team_id, is_home=True):
    """Get the most recent rolling features for a team."""
    prefix = "home" if is_home else "away"
    id_col = f"{prefix}_team_id"
    matches = df[df[id_col] == team_id].sort_values("date")
    if len(matches) == 0:
        return None
    return matches.iloc[-1]


def build_feature_vector(df, feat_cols, home_team, away_team):
    """Build feature vector for a prediction from latest available data."""
    home_id = df[df["home_team"] == home_team]["home_team_id"].iloc[0]
    away_id = df[df["away_team"] == away_team]["away_team_id"].iloc[0]

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

    # Fill NaN with median from training data
    series = pd.Series(vector)
    for col in series.index:
        if pd.isna(series[col]):
            col_data = df[col] if col in df.columns else pd.Series([0])
            series[col] = col_data.median()

    return series[feat_cols].values.astype(np.float64).reshape(1, -1)


def predict_match(X):
    """Run stacking ensemble prediction."""
    config, meta, bases = load_stacking()
    meta_features = []
    for name in config["base_model_names"]:
        meta_features.append(bases[name].predict_proba(X))
    X_meta = np.hstack(meta_features)
    proba = meta.predict_proba(X_meta)[0]
    return dict(zip(LABEL_ORDER, proba))


def kelly_criterion(prob_model, odds_decimal):
    """Kelly fraction: f* = (p * odds - 1) / (odds - 1)"""
    if odds_decimal <= 1:
        return 0.0
    f = (prob_model * odds_decimal - 1) / (odds_decimal - 1)
    return max(0.0, f)


def render_form_dots(results):
    """Render W/D/L dots for form."""
    html = ""
    for r in results:
        if r == "H":
            html += f'<span class="form-dot" style="background:{GREEN}">W</span>'
        elif r == "D":
            html += f'<span class="form-dot" style="background:{YELLOW}">D</span>'
        else:
            html += f'<span class="form-dot" style="background:{RED}">L</span>'
    return html


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SIDEBAR NAVIGATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

st.sidebar.title("Ligue 1 Predictor")
st.sidebar.markdown("---")
page = st.sidebar.radio(
    "Navegacion",
    ["Predictor", "Comparacion de Modelos", "Bankroll Tracker"],
    label_visibility="collapsed",
)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 1: PREDICTOR
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

if page == "Predictor":
    st.title("Predictor de Partidos")
    st.caption("Stacking Ensemble: LR + RF + XGBoost + MLP")

    df = load_features()
    config, _, _ = load_stacking()
    feat_cols = config["feature_cols"]
    teams = get_teams(df)

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Equipo Local", teams, index=teams.index("Paris Saint Germain") if "Paris Saint Germain" in teams else 0)
    with col2:
        away_options = [t for t in teams if t != home_team]
        away_team = st.selectbox("Equipo Visitante", away_options)

    if st.button("Predecir", type="primary", use_container_width=True):
        X = build_feature_vector(df, feat_cols, home_team, away_team)
        if X is not None:
            probs = predict_match(X)
            ph, pd_, pa = probs["H"], probs["D"], probs["A"]

            # ── Probability bar ──
            st.markdown("### Probabilidades")
            bar_html = f"""
            <div class="prob-bar">
                <div class="prob-segment" style="width:{ph*100:.0f}%; background:{GREEN};">
                    {ph*100:.1f}%
                </div>
                <div class="prob-segment" style="width:{pd_*100:.0f}%; background:{YELLOW};">
                    {pd_*100:.1f}%
                </div>
                <div class="prob-segment" style="width:{pa*100:.0f}%; background:{RED};">
                    {pa*100:.1f}%
                </div>
            </div>
            """
            st.markdown(bar_html, unsafe_allow_html=True)
            st.caption("Local | Empate | Visita")

            # ── Prediction badge ──
            pred = max(probs, key=probs.get)
            conf = probs[pred]
            if conf > 0.55:
                conf_label = "ALTA"
                conf_color = GREEN
            elif conf > 0.45:
                conf_label = "MEDIA"
                conf_color = YELLOW
            else:
                conf_label = "BAJA"
                conf_color = RED

            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {LABEL_COLORS[pred]};">
                <div style="font-size:14px; color:{TEXT_MUTED};">PREDICCION</div>
                <div style="font-size:32px; font-weight:800; color:{LABEL_COLORS[pred]};">
                    {LABEL_NAMES[pred]}
                </div>
                <div style="font-size:14px; margin-top:4px;">
                    Confianza: <span style="color:{conf_color}; font-weight:700;">{conf_label} ({conf*100:.1f}%)</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Odds comparison + Kelly ──
            st.markdown("### Modelo vs Casa de Apuestas")
            st.caption("Ingresa los momios decimales de tu casa de apuestas")

            oc1, oc2, oc3 = st.columns(3)
            with oc1:
                odds_h = st.number_input("Momio Local", min_value=1.01, value=2.00, step=0.05, format="%.2f")
            with oc2:
                odds_d = st.number_input("Momio Empate", min_value=1.01, value=3.30, step=0.05, format="%.2f")
            with oc3:
                odds_a = st.number_input("Momio Visita", min_value=1.01, value=3.50, step=0.05, format="%.2f")

            implied_h = 1 / odds_h
            implied_d = 1 / odds_d
            implied_a = 1 / odds_a
            total_implied = implied_h + implied_d + implied_a

            comp_data = pd.DataFrame({
                "Resultado": ["Local (H)", "Empate (D)", "Visita (A)"],
                "Modelo %": [f"{ph*100:.1f}%", f"{pd_*100:.1f}%", f"{pa*100:.1f}%"],
                "Momio": [f"{odds_h:.2f}", f"{odds_d:.2f}", f"{odds_a:.2f}"],
                "Implícita %": [
                    f"{implied_h/total_implied*100:.1f}%",
                    f"{implied_d/total_implied*100:.1f}%",
                    f"{implied_a/total_implied*100:.1f}%",
                ],
                "Edge %": [
                    f"{(ph - implied_h/total_implied)*100:+.1f}%",
                    f"{(pd_ - implied_d/total_implied)*100:+.1f}%",
                    f"{(pa - implied_a/total_implied)*100:+.1f}%",
                ],
            })
            st.dataframe(comp_data, use_container_width=True, hide_index=True)

            # ── Kelly Criterion ──
            st.markdown("### Kelly Criterion")
            edges = {
                "H": (ph, odds_h, kelly_criterion(ph, odds_h)),
                "D": (pd_, odds_d, kelly_criterion(pd_, odds_d)),
                "A": (pa, odds_a, kelly_criterion(pa, odds_a)),
            }
            has_edge = False
            for outcome, (prob, odds, kelly) in edges.items():
                edge = prob - (1 / odds)
                if edge > 0.05 and kelly > 0:
                    has_edge = True
                    st.markdown(f"""
                    <div class="kelly-box">
                        Apostar <span style="font-size:24px;">{kelly*100:.1f}%</span> del bankroll
                        a {LABEL_NAMES[outcome]} @ {odds:.2f}
                        &nbsp;(edge: {edge*100:+.1f}%)
                    </div>
                    """, unsafe_allow_html=True)

            if not has_edge:
                st.markdown("""
                <div class="no-edge-box">
                    Sin edge > 5% — No apostar
                </div>
                """, unsafe_allow_html=True)

            # ── Recent form ──
            st.markdown("### Forma Reciente (ultimos 5)")
            fc1, fc2 = st.columns(2)
            with fc1:
                form_home = get_team_form(df, home_team)
                st.markdown(f"**{home_team}**")
                st.markdown(render_form_dots(form_home), unsafe_allow_html=True)
                wins = sum(1 for r in form_home if r == "H")
                draws = sum(1 for r in form_home if r == "D")
                losses = sum(1 for r in form_home if r == "A")
                st.caption(f"{wins}W {draws}D {losses}L")

            with fc2:
                form_away = get_team_form(df, away_team)
                st.markdown(f"**{away_team}**")
                st.markdown(render_form_dots(form_away), unsafe_allow_html=True)
                wins = sum(1 for r in form_away if r == "H")
                draws = sum(1 for r in form_away if r == "D")
                losses = sum(1 for r in form_away if r == "A")
                st.caption(f"{wins}W {draws}D {losses}L")

        else:
            st.error("No hay datos suficientes para estos equipos.")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 2: MODEL COMPARISON
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "Comparacion de Modelos":
    st.title("Comparacion de Modelos")
    st.caption("Holdout: Temporada 2024 (307 partidos)")

    comp = load_comparison()

    # ── Metrics table ──
    st.markdown("### Metricas por Modelo")
    display_comp = comp.copy()
    for col in ["accuracy", "f1_macro", "f1_H", "f1_D", "f1_A", "roc_auc", "log_loss", "brier"]:
        display_comp[col] = display_comp[col].map("{:.4f}".format)
    st.dataframe(display_comp, use_container_width=True, hide_index=True)

    # ── F1 Macro + ROC AUC bar chart ──
    st.markdown("### F1 Macro vs ROC-AUC")

    fig_bars = go.Figure()
    fig_bars.add_trace(go.Bar(
        x=comp["model"], y=comp["f1_macro"],
        name="F1 Macro", marker_color=GREEN,
    ))
    fig_bars.add_trace(go.Bar(
        x=comp["model"], y=comp["roc_auc"],
        name="ROC AUC", marker_color="#448AFF",
    ))
    fig_bars.update_layout(
        barmode="group",
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font_color=TEXT,
        yaxis_range=[0, 0.8],
        height=400,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig_bars, use_container_width=True)

    # ── F1 per class heatmap ──
    st.markdown("### F1 por Clase (H / D / A)")

    f1_data = comp[["model", "f1_H", "f1_D", "f1_A"]].set_index("model")
    fig_heat = go.Figure(data=go.Heatmap(
        z=f1_data.values,
        x=["Local (H)", "Empate (D)", "Visita (A)"],
        y=f1_data.index,
        colorscale=[[0, RED], [0.5, YELLOW], [1, GREEN]],
        text=[[f"{v:.3f}" for v in row] for row in f1_data.values],
        texttemplate="%{text}",
        textfont_size=14,
        zmin=0, zmax=0.7,
    ))
    fig_heat.update_layout(
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font_color=TEXT,
        height=350,
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Confusion matrix (best model = stacking) ──
    st.markdown("### Confusion Matrix — Stacking Ensemble")

    df = load_features()
    config, meta, bases = load_stacking()
    feat_cols = config["feature_cols"]

    meta_cols = [c for c in df.columns if c not in [
        "fixture_id", "season", "date", "matchday",
        "home_team_id", "home_team", "away_team_id", "away_team",
        "referee", "result",
    ]]

    test = df[df["season"] == 2024].dropna(subset=["home_goals_avg"]).copy()
    for col in meta_cols:
        if test[col].isna().any():
            test[col] = test[col].fillna(df[col].median())

    X_test = test[feat_cols].values.astype(np.float64)
    le = LabelEncoder()
    le.classes_ = np.array(LABEL_ORDER)
    y_test = le.transform(test["result"].values)

    # Predict with stacking
    meta_features = []
    for name in config["base_model_names"]:
        meta_features.append(bases[name].predict_proba(X_test))
    X_meta_test = np.hstack(meta_features)
    y_pred = meta.predict(X_meta_test)

    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])
    fig_cm = go.Figure(data=go.Heatmap(
        z=cm,
        x=["Pred H", "Pred D", "Pred A"],
        y=["Real H", "Real D", "Real A"],
        colorscale=[[0, CARD_BG], [1, GREEN]],
        text=cm,
        texttemplate="%{text}",
        textfont_size=18,
    ))
    fig_cm.update_layout(
        plot_bgcolor=BG_DARK,
        paper_bgcolor=BG_DARK,
        font_color=TEXT,
        height=400,
        yaxis_autorange="reversed",
    )
    st.plotly_chart(fig_cm, use_container_width=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# PAGE 3: BANKROLL TRACKER
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

elif page == "Bankroll Tracker":
    st.title("Bankroll Tracker")

    # Load or init bankroll data
    if BANKROLL_PATH.exists():
        bets = pd.read_csv(BANKROLL_PATH)
        bets["date"] = pd.to_datetime(bets["date"])
    else:
        bets = pd.DataFrame(columns=["date", "match", "market", "odds", "stake", "result", "pnl"])

    # ── Config ──
    initial_bankroll = st.number_input(
        "Bankroll Inicial ($)", min_value=0.0, value=1000.0, step=100.0,
    )

    st.markdown("---")
    st.markdown("### Registrar Apuesta")

    with st.form("bet_form"):
        bc1, bc2 = st.columns(2)
        with bc1:
            bet_match = st.text_input("Partido", placeholder="PSG vs Lyon")
            bet_market = st.selectbox("Mercado", ["Local (H)", "Empate (D)", "Visita (A)", "Over 2.5", "Under 2.5", "Otro"])
        with bc2:
            bet_odds = st.number_input("Momio Decimal", min_value=1.01, value=2.00, step=0.05)
            bet_stake = st.number_input("Stake ($)", min_value=0.0, value=50.0, step=10.0)
        bet_result = st.selectbox("Resultado", ["Pendiente", "Ganada", "Perdida", "Push"])
        submitted = st.form_submit_button("Agregar Apuesta", use_container_width=True)

        if submitted and bet_match:
            if bet_result == "Ganada":
                pnl = bet_stake * (bet_odds - 1)
            elif bet_result == "Perdida":
                pnl = -bet_stake
            elif bet_result == "Push":
                pnl = 0.0
            else:
                pnl = 0.0

            new_bet = pd.DataFrame([{
                "date": pd.Timestamp.now().strftime("%Y-%m-%d"),
                "match": bet_match,
                "market": bet_market,
                "odds": bet_odds,
                "stake": bet_stake,
                "result": bet_result,
                "pnl": pnl,
            }])
            bets = pd.concat([bets, new_bet], ignore_index=True)
            bets.to_csv(BANKROLL_PATH, index=False)
            st.success(f"Apuesta registrada: {bet_match} — P&L: ${pnl:+.2f}")
            st.rerun()

    # ── Stats ──
    if len(bets) > 0:
        resolved = bets[bets["result"].isin(["Ganada", "Perdida"])]

        st.markdown("---")
        st.markdown("### Resumen")

        mc1, mc2, mc3, mc4 = st.columns(4)
        total_pnl = bets["pnl"].sum()
        total_staked = resolved["stake"].sum()
        win_rate = (resolved["result"] == "Ganada").mean() * 100 if len(resolved) > 0 else 0
        roi = (total_pnl / total_staked * 100) if total_staked > 0 else 0
        current_bankroll = initial_bankroll + total_pnl

        with mc1:
            st.metric("Bankroll Actual", f"${current_bankroll:,.2f}", f"${total_pnl:+,.2f}")
        with mc2:
            st.metric("ROI", f"{roi:+.1f}%")
        with mc3:
            st.metric("Win Rate", f"{win_rate:.0f}%")
        with mc4:
            st.metric("Apuestas", f"{len(resolved)} / {len(bets)}")

        # ── Bankroll curve ──
        st.markdown("### Evolucion del Bankroll")
        bets_sorted = bets.sort_values("date").copy()
        bets_sorted["cumulative_pnl"] = bets_sorted["pnl"].cumsum()
        bets_sorted["bankroll"] = initial_bankroll + bets_sorted["cumulative_pnl"]

        fig_bankroll = go.Figure()
        fig_bankroll.add_trace(go.Scatter(
            x=bets_sorted["date"],
            y=bets_sorted["bankroll"],
            mode="lines+markers",
            line=dict(color=GREEN, width=3),
            marker=dict(size=8),
            fill="tozeroy",
            fillcolor="rgba(0,200,83,0.1)",
        ))
        fig_bankroll.add_hline(
            y=initial_bankroll, line_dash="dash",
            line_color=YELLOW, annotation_text="Bankroll Inicial",
        )
        fig_bankroll.update_layout(
            plot_bgcolor=BG_DARK,
            paper_bgcolor=BG_DARK,
            font_color=TEXT,
            yaxis_title="Bankroll ($)",
            height=400,
        )
        st.plotly_chart(fig_bankroll, use_container_width=True)

        # ── Bet history ──
        st.markdown("### Historial de Apuestas")
        display_bets = bets.copy()
        display_bets["pnl"] = display_bets["pnl"].map("${:+,.2f}".format)
        display_bets["odds"] = display_bets["odds"].map("{:.2f}".format)
        display_bets["stake"] = display_bets["stake"].map("${:,.2f}".format)
        st.dataframe(display_bets, use_container_width=True, hide_index=True)

        # Delete button
        if st.button("Borrar todo el historial", type="secondary"):
            if BANKROLL_PATH.exists():
                BANKROLL_PATH.unlink()
            st.rerun()
    else:
        st.info("No hay apuestas registradas. Usa el formulario de arriba para comenzar.")


# ── Footer ──
st.sidebar.markdown("---")
st.sidebar.caption("Ligue 1 Predictor v1.0")
st.sidebar.caption("Stacking: LR + RF + XGB + MLP")
st.sidebar.caption("Data: API-Football 2022-2024")
