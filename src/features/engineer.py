"""
Ligue 1 Feature Engineering
Computes all rolling/historical features WITHOUT data leakage.
Every feature for match N uses only data from matches < N.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.features.elo import compute_elo_ratings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MATCHES_PATH = PROJECT_ROOT / "data" / "processed" / "matches.parquet"
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "features.parquet"

WINDOW = 5  # Rolling window size

# Stats to compute rolling averages for (column suffix without home_/away_ prefix)
ROLLING_STATS = [
    "goals",           # scored
    "goals_conceded",  # conceded
    "xg",
    "xg_conceded",
    "shots_on_target",
    "corners",
    "fouls",
    "possession",
    "pass_pct",
    "shots_insidebox",
    "yellow_cards",
    "red_cards",
    "gk_saves",
]

# Player-level stats with custom windows
PLAYER_ROLLING = {
    "avg_rating": 3,           # last 3 matches
    "top_scorer_goals": 5,     # last 5 matches (sum, not mean)
    "key_players_shots": 3,    # last 3 matches
}


def _build_team_match_history(df: pd.DataFrame) -> dict[int, pd.DataFrame]:
    """
    Build per-team match history with normalized columns.
    Each row represents one match from that team's perspective.
    """
    records: dict[int, list[dict]] = {}

    for _, row in df.iterrows():
        date = row["date"]
        fixture_id = row["fixture_id"]

        # Home team perspective
        home_id = row["home_team_id"]
        records.setdefault(home_id, []).append({
            "fixture_id": fixture_id,
            "date": date,
            "is_home": True,
            "goals": row["home_goals"],
            "goals_conceded": row["away_goals"],
            "xg": row.get("home_xg"),
            "xg_conceded": row.get("away_xg"),
            "shots_on_target": row.get("home_shots_on_target"),
            "corners": row.get("home_corners"),
            "fouls": row.get("home_fouls"),
            "possession": row.get("home_possession"),
            "pass_pct": row.get("home_pass_pct"),
            "shots_insidebox": row.get("home_shots_insidebox"),
            "yellow_cards": row.get("home_yellow_cards"),
            "red_cards": row.get("home_red_cards"),
            "gk_saves": row.get("home_gk_saves"),
            "points": 3 if row["result"] == "H" else (1 if row["result"] == "D" else 0),
            "win": 1 if row["result"] == "H" else 0,
            # Player-level aggregates
            "avg_rating": row.get("home_avg_rating"),
            "top_scorer_goals": row.get("home_top_scorer_goals"),
            "key_players_shots": row.get("home_key_players_shots"),
        })

        # Away team perspective
        away_id = row["away_team_id"]
        records.setdefault(away_id, []).append({
            "fixture_id": fixture_id,
            "date": date,
            "is_home": False,
            "goals": row["away_goals"],
            "goals_conceded": row["home_goals"],
            "xg": row.get("away_xg"),
            "xg_conceded": row.get("home_xg"),
            "shots_on_target": row.get("away_shots_on_target"),
            "corners": row.get("away_corners"),
            "fouls": row.get("away_fouls"),
            "possession": row.get("away_possession"),
            "pass_pct": row.get("away_pass_pct"),
            "shots_insidebox": row.get("away_shots_insidebox"),
            "yellow_cards": row.get("away_yellow_cards"),
            "red_cards": row.get("away_red_cards"),
            "gk_saves": row.get("away_gk_saves"),
            "points": 3 if row["result"] == "A" else (1 if row["result"] == "D" else 0),
            "win": 1 if row["result"] == "A" else 0,
            # Player-level aggregates
            "avg_rating": row.get("away_avg_rating"),
            "top_scorer_goals": row.get("away_top_scorer_goals"),
            "key_players_shots": row.get("away_key_players_shots"),
        })

    # Convert to DataFrames sorted by date
    team_dfs = {}
    for team_id, recs in records.items():
        tdf = pd.DataFrame(recs).sort_values("date").reset_index(drop=True)
        team_dfs[team_id] = tdf
    return team_dfs


def _get_rolling_features(team_df: pd.DataFrame, fixture_id: int, window: int) -> dict:
    """
    Get rolling features for a team BEFORE a specific fixture.
    Returns dict of feature values.
    """
    # Find the index of this fixture
    idx_mask = team_df["fixture_id"] == fixture_id
    if not idx_mask.any():
        return {}
    idx = team_df[idx_mask].index[0]

    # Get previous matches (strictly before this one)
    start = max(0, idx - window)
    prev = team_df.iloc[start:idx]

    if len(prev) == 0:
        return {}

    features = {}
    for stat in ROLLING_STATS:
        if stat in prev.columns:
            vals = prev[stat].dropna()
            features[f"{stat}_avg"] = vals.mean() if len(vals) > 0 else np.nan
        else:
            features[f"{stat}_avg"] = np.nan

    # Points last N
    features["points_last_n"] = prev["points"].sum()
    # Goal diff rolling
    features["goal_diff_rolling"] = (prev["goals"] - prev["goals_conceded"]).sum()

    # Home/away specific win rate (from previous matches of same venue type)
    is_home_now = team_df.loc[team_df["fixture_id"] == fixture_id, "is_home"].iloc[0]
    venue_prev = team_df.iloc[:idx]
    if is_home_now:
        home_matches = venue_prev[venue_prev["is_home"]]
        features["venue_win_rate"] = home_matches["win"].mean() if len(home_matches) > 0 else np.nan
    else:
        away_matches = venue_prev[~venue_prev["is_home"]]
        features["venue_win_rate"] = away_matches["win"].mean() if len(away_matches) > 0 else np.nan

    # Red cards in last 5
    rc_vals = prev["red_cards"].dropna()
    features["red_cards_last_n"] = rc_vals.sum() if len(rc_vals) > 0 else 0.0

    # Player-level rolling features (custom windows)
    for stat, w in PLAYER_ROLLING.items():
        if stat not in team_df.columns:
            features[f"{stat}_rolling"] = np.nan
            continue

        p_start = max(0, idx - w)
        p_prev = team_df.iloc[p_start:idx]
        vals = p_prev[stat].dropna()

        if stat == "top_scorer_goals":
            # Sum for top scorer goals (cumulative over window)
            features[f"{stat}_rolling"] = vals.sum() if len(vals) > 0 else np.nan
        else:
            # Mean for rating and shots
            features[f"{stat}_rolling"] = vals.mean() if len(vals) > 0 else np.nan

    return features


def _compute_referee_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute referee_home_win_rate and referee_cards_avg using only
    historical data (matches before the current one).
    """
    ref_home_win_rate = []
    ref_cards_avg = []

    # Pre-sort by date
    for i, row in df.iterrows():
        ref = row["referee"]
        if pd.isna(ref) or ref == "":
            ref_home_win_rate.append(np.nan)
            ref_cards_avg.append(np.nan)
            continue

        # All matches by this referee strictly before this date
        past = df[(df["referee"] == ref) & (df["date"] < row["date"])]

        if len(past) == 0:
            ref_home_win_rate.append(np.nan)
            ref_cards_avg.append(np.nan)
            continue

        # Home win rate
        home_wins = (past["result"] == "H").sum()
        ref_home_win_rate.append(home_wins / len(past))

        # Cards avg (total yellow + red per match)
        total_cards = (
            past["home_yellow_cards"].fillna(0)
            + past["away_yellow_cards"].fillna(0)
            + past["home_red_cards"].fillna(0)
            + past["away_red_cards"].fillna(0)
        )
        ref_cards_avg.append(total_cards.mean())

    df = df.copy()
    df["referee_home_win_rate"] = ref_home_win_rate
    df["referee_cards_avg"] = ref_cards_avg
    return df


def _compute_days_rest(df: pd.DataFrame) -> pd.DataFrame:
    """Compute days since last match for each team."""
    # Build last-match-date lookup per team
    last_match: dict[int, pd.Timestamp] = {}
    home_rest = []
    away_rest = []

    for _, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        match_date = row["date"]

        if home_id in last_match:
            home_rest.append((match_date - last_match[home_id]).days)
        else:
            home_rest.append(np.nan)

        if away_id in last_match:
            away_rest.append((match_date - last_match[away_id]).days)
        else:
            away_rest.append(np.nan)

        last_match[home_id] = match_date
        last_match[away_id] = match_date

    df = df.copy()
    df["home_days_rest"] = home_rest
    df["away_days_rest"] = away_rest
    return df


def _compute_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """Head-to-head record in last 3 meetings (from home team perspective)."""
    h2h_points = []

    for _, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        match_date = row["date"]

        # Find previous matches between these teams (either side)
        mask = (
            (df["date"] < match_date)
            & (
                ((df["home_team_id"] == home_id) & (df["away_team_id"] == away_id))
                | ((df["home_team_id"] == away_id) & (df["away_team_id"] == home_id))
            )
        )
        past = df[mask].tail(3)

        if len(past) == 0:
            h2h_points.append(np.nan)
            continue

        points = 0
        for _, pmatch in past.iterrows():
            if pmatch["home_team_id"] == home_id:
                # Home team was home in past match
                if pmatch["result"] == "H":
                    points += 3
                elif pmatch["result"] == "D":
                    points += 1
            else:
                # Home team was away in past match
                if pmatch["result"] == "A":
                    points += 3
                elif pmatch["result"] == "D":
                    points += 1

        h2h_points.append(points / len(past))  # Normalize by number of matches

    df = df.copy()
    df["h2h_points_avg"] = h2h_points
    return df


def build_features(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Main feature engineering pipeline."""
    if df is None:
        df = pd.read_parquet(MATCHES_PATH)

    print(f"Input: {len(df)} matches")

    # Ensure sorted by date
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    # 1. ELO ratings
    print("Computing ELO ratings...")
    df = compute_elo_ratings(df)

    # 2. Build team histories for rolling features
    print("Building team match histories...")
    team_dfs = _build_team_match_history(df)

    # 3. Compute rolling features per team per match
    print("Computing rolling features...")
    home_features_list = []
    away_features_list = []

    for _, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]
        fid = row["fixture_id"]

        hf = _get_rolling_features(team_dfs[home_id], fid, WINDOW)
        af = _get_rolling_features(team_dfs[away_id], fid, WINDOW)

        home_features_list.append(hf)
        away_features_list.append(af)

    # Convert to DataFrames and prefix
    home_feat_df = pd.DataFrame(home_features_list)
    away_feat_df = pd.DataFrame(away_features_list)

    home_feat_df.columns = [f"home_{c}" for c in home_feat_df.columns]
    away_feat_df.columns = [f"away_{c}" for c in away_feat_df.columns]

    df = pd.concat([df, home_feat_df, away_feat_df], axis=1)

    # 4. Referee features
    print("Computing referee features...")
    df = _compute_referee_features(df)

    # 5. Days rest
    print("Computing days rest...")
    df = _compute_days_rest(df)

    # 6. Head-to-head
    print("Computing head-to-head...")
    df = _compute_h2h(df)

    # 7. Select final feature columns + target
    feature_cols = [
        # Identifiers (for reference, not model input)
        "fixture_id", "season", "date", "matchday",
        "home_team_id", "home_team", "away_team_id", "away_team",
        "referee",
        # Target
        "result",
        # ELO
        "home_elo", "away_elo", "elo_diff",
        # Rolling features (home)
        "home_goals_avg", "home_goals_conceded_avg",
        "home_xg_avg", "home_xg_conceded_avg",
        "home_shots_on_target_avg", "home_corners_avg",
        "home_fouls_avg", "home_possession_avg", "home_pass_pct_avg",
        "home_shots_insidebox_avg",
        "home_yellow_cards_avg", "home_gk_saves_avg",
        "home_points_last_n", "home_goal_diff_rolling",
        "home_venue_win_rate", "home_red_cards_last_n",
        # Rolling features (away)
        "away_goals_avg", "away_goals_conceded_avg",
        "away_xg_avg", "away_xg_conceded_avg",
        "away_shots_on_target_avg", "away_corners_avg",
        "away_fouls_avg", "away_possession_avg", "away_pass_pct_avg",
        "away_shots_insidebox_avg",
        "away_yellow_cards_avg", "away_gk_saves_avg",
        "away_points_last_n", "away_goal_diff_rolling",
        "away_venue_win_rate", "away_red_cards_last_n",
        # Referee
        "referee_home_win_rate", "referee_cards_avg",
        # Rest
        "home_days_rest", "away_days_rest",
        # H2H
        "h2h_points_avg",
        # Player-level rolling
        "home_avg_rating_rolling", "away_avg_rating_rolling",
        "home_top_scorer_goals_rolling", "away_top_scorer_goals_rolling",
        "home_key_players_shots_rolling", "away_key_players_shots_rolling",
    ]

    # Only keep columns that exist
    feature_cols = [c for c in feature_cols if c in df.columns]
    df_out = df[feature_cols].copy()

    # Save
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(FEATURES_PATH, index=False)
    print(f"\nSaved {len(df_out)} rows, {len(df_out.columns)} columns → {FEATURES_PATH}")

    return df_out


if __name__ == "__main__":
    df = build_features()

    print(f"\n{'='*60}")
    print(f"FINAL DATASET")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")

    print(f"\nTarget distribution:")
    print(df["result"].value_counts().to_string())

    print(f"\nNull counts per column:")
    nulls = df.isnull().sum()
    print(nulls[nulls > 0].to_string())

    print(f"\nFirst 3 rows (feature columns only):")
    feat_only = df.drop(columns=["fixture_id", "season", "date", "home_team", "away_team",
                                  "home_team_id", "away_team_id", "referee", "matchday"],
                         errors="ignore")
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.width", 200)
    print(feat_only.head(3).to_string())
