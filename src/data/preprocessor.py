"""
Ligue 1 Preprocessor
Reads raw JSON fixtures + stats → flat DataFrame → parquet.
"""

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
FIXTURES_DIR = RAW_DIR / "fixtures"
STATS_DIR = RAW_DIR / "team_stats"
PLAYERS_DIR = RAW_DIR / "player_stats"
OUT_PATH = PROJECT_ROOT / "data" / "processed" / "matches.parquet"

STAT_KEYS = [
    "Shots on Goal",
    "Shots off Goal",
    "Total Shots",
    "Blocked Shots",
    "Shots insidebox",
    "Shots outsidebox",
    "Fouls",
    "Corner Kicks",
    "Offsides",
    "Ball Possession",
    "Yellow Cards",
    "Red Cards",
    "Goalkeeper Saves",
    "Total passes",
    "Passes accurate",
    "Passes %",
    "expected_goals",
]

# Maps raw stat names → clean column suffixes
STAT_COL_MAP = {
    "Shots on Goal": "shots_on_target",
    "Shots off Goal": "shots_off_target",
    "Total Shots": "total_shots",
    "Blocked Shots": "blocked_shots",
    "Shots insidebox": "shots_insidebox",
    "Shots outsidebox": "shots_outsidebox",
    "Fouls": "fouls",
    "Corner Kicks": "corners",
    "Offsides": "offsides",
    "Ball Possession": "possession",
    "Yellow Cards": "yellow_cards",
    "Red Cards": "red_cards",
    "Goalkeeper Saves": "gk_saves",
    "Total passes": "total_passes",
    "Passes accurate": "accurate_passes",
    "Passes %": "pass_pct",
    "expected_goals": "xg",
}


def _parse_stat_value(value):
    """Convert stat value to float. Handles '78%', None, etc."""
    if value is None:
        return 0.0
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip().replace("%", "")
    try:
        return float(s)
    except ValueError:
        return None


def _extract_stats(stats_list: list[dict]) -> dict:
    """Extract stats dict from API response for one team."""
    result = {}
    for item in stats_list:
        stat_type = item["type"]
        if stat_type in STAT_COL_MAP:
            col = STAT_COL_MAP[stat_type]
            result[col] = _parse_stat_value(item["value"])
    return result


def _parse_round(round_str: str) -> int | None:
    """Extract matchday number from 'Regular Season - 5'."""
    if not round_str or "Regular Season" not in round_str:
        return None
    try:
        return int(round_str.split(" - ")[-1])
    except ValueError:
        return None


def _extract_player_aggregates(players_data: list[dict], team_id: int) -> dict:
    """Extract per-team player aggregates from fixture player stats.

    Returns dict with avg_rating, top_scorer_goals, key_players_shots.
    """
    team_entry = None
    for entry in players_data:
        if entry["team"]["id"] == team_id:
            team_entry = entry
            break

    if team_entry is None:
        return {}

    ratings = []
    player_goals = []
    player_shots = []

    for p in team_entry["players"]:
        stats = p["statistics"][0]

        # Rating (string like "7" or "6.3")
        r = stats["games"].get("rating")
        if r is not None:
            try:
                ratings.append(float(r))
            except (ValueError, TypeError):
                pass

        # Goals
        g = stats["goals"].get("total")
        player_goals.append(int(g) if g is not None else 0)

        # Shots total
        s = stats["shots"].get("total")
        player_shots.append(int(s) if s is not None else 0)

    result = {}
    result["avg_rating"] = sum(ratings) / len(ratings) if ratings else None
    result["top_scorer_goals"] = max(player_goals) if player_goals else 0
    # Key players: top 3 by shots
    sorted_shots = sorted(player_shots, reverse=True)[:3]
    result["key_players_shots"] = sum(sorted_shots)

    return result


def build_matches_df() -> pd.DataFrame:
    """Build flat DataFrame from all raw data."""
    rows = []

    fixture_files = sorted(FIXTURES_DIR.glob("ligue1_*_fixtures.json"))
    print(f"Found {len(fixture_files)} fixture files")

    for fpath in fixture_files:
        with open(fpath) as f:
            fixtures = json.load(f)

        season = int(fpath.stem.split("_")[1])
        season_stats_dir = STATS_DIR / str(season)
        loaded, skipped = 0, 0

        for fx in fixtures:
            fid = fx["fixture"]["id"]
            stats_path = season_stats_dir / f"{fid}.json"

            # Base fixture info
            row = {
                "fixture_id": fid,
                "season": season,
                "date": fx["fixture"]["date"],
                "referee": fx["fixture"].get("referee"),
                "round": fx["league"].get("round", ""),
                "matchday": _parse_round(fx["league"].get("round", "")),
                "home_team_id": fx["teams"]["home"]["id"],
                "home_team": fx["teams"]["home"]["name"],
                "away_team_id": fx["teams"]["away"]["id"],
                "away_team": fx["teams"]["away"]["name"],
                "home_goals": fx["goals"]["home"],
                "away_goals": fx["goals"]["away"],
                "ht_home_goals": fx["score"]["halftime"]["home"],
                "ht_away_goals": fx["score"]["halftime"]["away"],
            }

            # Target
            hg, ag = fx["goals"]["home"], fx["goals"]["away"]
            if hg is None or ag is None:
                skipped += 1
                continue
            if hg > ag:
                row["result"] = "H"
            elif hg < ag:
                row["result"] = "A"
            else:
                row["result"] = "D"

            # Stats
            if stats_path.exists():
                with open(stats_path) as sf:
                    stats_data = json.load(sf)

                if len(stats_data) >= 2:
                    # Match stats team to fixture team by ID
                    if stats_data[0]["team"]["id"] == row["home_team_id"]:
                        home_stats = _extract_stats(stats_data[0]["statistics"])
                        away_stats = _extract_stats(stats_data[1]["statistics"])
                    else:
                        home_stats = _extract_stats(stats_data[1]["statistics"])
                        away_stats = _extract_stats(stats_data[0]["statistics"])

                    for col, val in home_stats.items():
                        row[f"home_{col}"] = val
                    for col, val in away_stats.items():
                        row[f"away_{col}"] = val

                loaded += 1

            # Player stats (may not exist for all seasons)
            players_path = PLAYERS_DIR / str(season) / f"{fid}.json"
            if players_path.exists():
                with open(players_path) as pf:
                    players_data = json.load(pf)

                if len(players_data) >= 2:
                    home_pagg = _extract_player_aggregates(players_data, row["home_team_id"])
                    away_pagg = _extract_player_aggregates(players_data, row["away_team_id"])

                    for col, val in home_pagg.items():
                        row[f"home_{col}"] = val
                    for col, val in away_pagg.items():
                        row[f"away_{col}"] = val

            rows.append(row)

        print(f"  Season {season}: {len(fixtures)} fixtures, {loaded} with stats, {skipped} skipped")

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "fixture_id"]).reset_index(drop=True)

    # Impute missing xG with median per team+season
    for side in ["home", "away"]:
        col = f"{side}_xg"
        team_col = f"{side}_team_id"
        if col in df.columns:
            mask = df[col].isna() | (df[col] == 0.0)
            # Compute median per team+season (excluding zeros that are likely missing)
            non_zero = df[~mask]
            medians = non_zero.groupby(["season", team_col])[col].median()

            for idx in df[mask].index:
                key = (df.loc[idx, "season"], df.loc[idx, team_col])
                if key in medians.index:
                    df.loc[idx, col] = medians[key]
                else:
                    # Fallback: season median
                    season_med = non_zero[non_zero["season"] == df.loc[idx, "season"]][col].median()
                    if pd.notna(season_med):
                        df.loc[idx, col] = season_med

    # Save
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print(f"\nSaved {len(df)} matches → {OUT_PATH}")
    print(f"Columns: {list(df.columns)}")
    return df


if __name__ == "__main__":
    df = build_matches_df()
    print(f"\nShape: {df.shape}")
    print(f"\nTarget distribution:\n{df['result'].value_counts()}")
    print(f"\nNull counts:\n{df.isnull().sum()[df.isnull().sum() > 0]}")
