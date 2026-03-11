"""
Dynamic ELO rating system for Ligue 1 teams.
K=20, initial rating=1500. Home advantage built into expected score.
"""

import pandas as pd

K = 20
INITIAL_ELO = 1500
HOME_ADVANTAGE = 60  # ELO points added to home expected score


def expected_score(rating_a: float, rating_b: float) -> float:
    """Expected score for player A against player B."""
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def compute_elo_ratings(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ELO ratings for each match. Returns df with columns:
    home_elo, away_elo, elo_diff (all BEFORE the match — no leakage).

    df must be sorted by date and have: home_team_id, away_team_id, result (H/D/A).
    """
    ratings: dict[int, float] = {}
    home_elos = []
    away_elos = []

    for _, row in df.iterrows():
        home_id = row["home_team_id"]
        away_id = row["away_team_id"]

        # Get current ratings (before match)
        home_r = ratings.get(home_id, INITIAL_ELO)
        away_r = ratings.get(away_id, INITIAL_ELO)

        # Store pre-match ratings
        home_elos.append(home_r)
        away_elos.append(away_r)

        # Actual scores
        if row["result"] == "H":
            actual_home, actual_away = 1.0, 0.0
        elif row["result"] == "A":
            actual_home, actual_away = 0.0, 1.0
        else:  # Draw
            actual_home, actual_away = 0.5, 0.5

        # Expected scores (home gets advantage)
        exp_home = expected_score(home_r + HOME_ADVANTAGE, away_r)
        exp_away = 1 - exp_home

        # Update ratings
        ratings[home_id] = home_r + K * (actual_home - exp_home)
        ratings[away_id] = away_r + K * (actual_away - exp_away)

    df = df.copy()
    df["home_elo"] = home_elos
    df["away_elo"] = away_elos
    df["elo_diff"] = df["home_elo"] - df["away_elo"]

    return df
