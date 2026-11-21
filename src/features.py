from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
RESULTS_PATH = PROCESSED_DIR / "cleaned_international_results.csv"
ELO_SNAPSHOTS_PATH = PROCESSED_DIR / "cleaned_elo_ratings_by_year.csv"
MODEL_DATASET_PATH = PROCESSED_DIR / "model_dataset_v1.csv"
START_DATE = "2010-01-01"
FORM_WINDOW = 5
LONG_FORM_WINDOW = 10

CONTINENTAL_TOURNAMENTS = {
    "AFC Asian Cup",
    "African Cup of Nations",
    "CONCACAF Championship",
    "Copa América",
    "Gold Cup",
    "Oceania Nations Cup",
    "UEFA Euro",
}


def build_model_dataset(
    results_path=RESULTS_PATH,
    elo_path=ELO_SNAPSHOTS_PATH,
    output_path=MODEL_DATASET_PATH,
    start_date=START_DATE,
    form_window=FORM_WINDOW,
):
    results = pd.read_csv(results_path)
    elo = pd.read_csv(elo_path)

    if "tournament" not in results.columns:
        results["tournament"] = ""

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    results = results[results["date"] >= pd.to_datetime(start_date)].copy()
    results = results.sort_values("date").reset_index(drop=True)

    results = add_recent_form_features(results, form_window)
    results = add_tournament_context_features(results)
    results = add_elo_features(results, elo)
    results = results.dropna(
        subset=[
            "home_elo",
            "away_elo",
            "home_goals_for_avg",
            "home_goals_against_avg",
            "away_goals_for_avg",
            "away_goals_against_avg",
            "home_recent_points_avg",
            "away_recent_points_avg",
            "recent_points_difference",
            "home_recent_draw_rate",
            "away_recent_draw_rate",
            "home_recent_goal_difference_avg",
            "away_recent_goal_difference_avg",
            "recent_goal_difference_difference",
            "home_recent_close_match_rate",
            "away_recent_close_match_rate",
            "home_recent_low_scoring_rate",
            "away_recent_low_scoring_rate",
            "home_recent_draw_rate_10",
            "away_recent_draw_rate_10",
            "home_recent_close_match_rate_10",
            "away_recent_close_match_rate_10",
            "home_recent_low_scoring_rate_10",
            "away_recent_low_scoring_rate_10",
        ]
    ).copy()

    results["home_elo"] = results["home_elo"].astype(int)
    results["away_elo"] = results["away_elo"].astype(int)
    results["elo_difference"] = results["home_elo"] - results["away_elo"]
    results["home_advantage"] = (~results["neutral"].astype(bool)).astype(int)

    keep_columns = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "result",
        "neutral",
        "home_advantage",
        "home_elo",
        "away_elo",
        "elo_difference",
        "home_goals_for_avg",
        "home_goals_against_avg",
        "away_goals_for_avg",
        "away_goals_against_avg",
        "home_recent_points_avg",
        "away_recent_points_avg",
        "recent_points_difference",
        "home_recent_draw_rate",
        "away_recent_draw_rate",
        "home_recent_goal_difference_avg",
        "away_recent_goal_difference_avg",
        "recent_goal_difference_difference",
        "draw_rate_difference",
        "home_recent_close_match_rate",
        "away_recent_close_match_rate",
        "close_match_rate_difference",
        "home_recent_low_scoring_rate",
        "away_recent_low_scoring_rate",
        "home_recent_draw_rate_10",
        "away_recent_draw_rate_10",
        "draw_rate_difference_10",
        "home_recent_close_match_rate_10",
        "away_recent_close_match_rate_10",
        "close_match_rate_difference_10",
        "home_recent_low_scoring_rate_10",
        "away_recent_low_scoring_rate_10",
        "is_world_cup",
        "is_world_cup_qualifier",
        "is_continental_tournament",
        "is_friendly",
        "match_importance_weight",
    ]

    model_dataset = results[keep_columns].sort_values("date").reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_dataset.to_csv(output_path, index=False)

    return model_dataset


def add_elo_features(matches, elo):
    elo = elo.copy()
    elo["snapshot_date"] = pd.to_datetime(elo["snapshot_date"], errors="coerce")
    elo = elo.dropna(subset=["snapshot_date", "team", "elo"])
    elo = elo[
        (elo["snapshot_date"].dt.month == 12)
        & (elo["snapshot_date"].dt.day == 31)
    ].copy()
    elo = elo.sort_values("snapshot_date")

    home_elo = elo.rename(columns={"team": "home_team", "elo": "home_elo"})
    away_elo = elo.rename(columns={"team": "away_team", "elo": "away_elo"})

    matches = pd.merge_asof(
        matches.sort_values("date"),
        home_elo[["snapshot_date", "home_team", "home_elo"]].sort_values("snapshot_date"),
        left_on="date",
        right_on="snapshot_date",
        by="home_team",
        direction="backward",
    ).drop(columns=["snapshot_date"])

    matches = pd.merge_asof(
        matches.sort_values("date"),
        away_elo[["snapshot_date", "away_team", "away_elo"]].sort_values("snapshot_date"),
        left_on="date",
        right_on="snapshot_date",
        by="away_team",
        direction="backward",
    ).drop(columns=["snapshot_date"])

    return matches


def add_recent_form_features(matches, window):
    team_match_rows = []

    for match in matches.itertuples(index=False):
        home_points, away_points = get_match_points(match.home_score, match.away_score)
        team_match_rows.append(
            {
                "date": match.date,
                "team": match.home_team,
                "goals_for": match.home_score,
                "goals_against": match.away_score,
                "points": home_points,
                "is_draw": int(match.home_score == match.away_score),
                "goal_difference": match.home_score - match.away_score,
                "is_close_match": int(abs(match.home_score - match.away_score) <= 1),
                "is_low_scoring": int((match.home_score + match.away_score) < 2.5),
            }
        )
        team_match_rows.append(
            {
                "date": match.date,
                "team": match.away_team,
                "goals_for": match.away_score,
                "goals_against": match.home_score,
                "points": away_points,
                "is_draw": int(match.home_score == match.away_score),
                "goal_difference": match.away_score - match.home_score,
                "is_close_match": int(abs(match.home_score - match.away_score) <= 1),
                "is_low_scoring": int((match.home_score + match.away_score) < 2.5),
            }
        )

    team_matches = pd.DataFrame(team_match_rows).sort_values(["team", "date"])
    team_matches["goals_for_avg"] = add_rolling_previous_mean(team_matches, "goals_for", window)
    team_matches["goals_against_avg"] = add_rolling_previous_mean(team_matches, "goals_against", window)
    team_matches["recent_points_avg"] = add_rolling_previous_mean(team_matches, "points", window)
    team_matches["recent_draw_rate"] = add_rolling_previous_mean(team_matches, "is_draw", window)
    team_matches["recent_goal_difference_avg"] = add_rolling_previous_mean(
        team_matches, "goal_difference", window
    )
    team_matches["recent_close_match_rate"] = add_rolling_previous_mean(
        team_matches, "is_close_match", window
    )
    team_matches["recent_low_scoring_rate"] = add_rolling_previous_mean(
        team_matches, "is_low_scoring", window
    )
    team_matches["recent_draw_rate_10"] = add_rolling_previous_mean(
        team_matches, "is_draw", LONG_FORM_WINDOW
    )
    team_matches["recent_close_match_rate_10"] = add_rolling_previous_mean(
        team_matches, "is_close_match", LONG_FORM_WINDOW
    )
    team_matches["recent_low_scoring_rate_10"] = add_rolling_previous_mean(
        team_matches, "is_low_scoring", LONG_FORM_WINDOW
    )

    home_features = team_matches.rename(
        columns={
            "team": "home_team",
            "goals_for_avg": "home_goals_for_avg",
            "goals_against_avg": "home_goals_against_avg",
            "recent_points_avg": "home_recent_points_avg",
            "recent_draw_rate": "home_recent_draw_rate",
            "recent_goal_difference_avg": "home_recent_goal_difference_avg",
            "recent_close_match_rate": "home_recent_close_match_rate",
            "recent_low_scoring_rate": "home_recent_low_scoring_rate",
            "recent_draw_rate_10": "home_recent_draw_rate_10",
            "recent_close_match_rate_10": "home_recent_close_match_rate_10",
            "recent_low_scoring_rate_10": "home_recent_low_scoring_rate_10",
        }
    )
    away_features = team_matches.rename(
        columns={
            "team": "away_team",
            "goals_for_avg": "away_goals_for_avg",
            "goals_against_avg": "away_goals_against_avg",
            "recent_points_avg": "away_recent_points_avg",
            "recent_draw_rate": "away_recent_draw_rate",
            "recent_goal_difference_avg": "away_recent_goal_difference_avg",
            "recent_close_match_rate": "away_recent_close_match_rate",
            "recent_low_scoring_rate": "away_recent_low_scoring_rate",
            "recent_draw_rate_10": "away_recent_draw_rate_10",
            "recent_close_match_rate_10": "away_recent_close_match_rate_10",
            "recent_low_scoring_rate_10": "away_recent_low_scoring_rate_10",
        }
    )

    matches = matches.merge(
        home_features[
            [
                "date",
                "home_team",
                "home_goals_for_avg",
                "home_goals_against_avg",
                "home_recent_points_avg",
                "home_recent_draw_rate",
                "home_recent_goal_difference_avg",
                "home_recent_close_match_rate",
                "home_recent_low_scoring_rate",
                "home_recent_draw_rate_10",
                "home_recent_close_match_rate_10",
                "home_recent_low_scoring_rate_10",
            ]
        ],
        on=["date", "home_team"],
        how="left",
    )
    matches = matches.merge(
        away_features[
            [
                "date",
                "away_team",
                "away_goals_for_avg",
                "away_goals_against_avg",
                "away_recent_points_avg",
                "away_recent_draw_rate",
                "away_recent_goal_difference_avg",
                "away_recent_close_match_rate",
                "away_recent_low_scoring_rate",
                "away_recent_draw_rate_10",
                "away_recent_close_match_rate_10",
                "away_recent_low_scoring_rate_10",
            ]
        ],
        on=["date", "away_team"],
        how="left",
    )

    matches["recent_points_difference"] = (
        matches["home_recent_points_avg"] - matches["away_recent_points_avg"]
    )
    matches["recent_goal_difference_difference"] = (
        matches["home_recent_goal_difference_avg"]
        - matches["away_recent_goal_difference_avg"]
    )
    matches["draw_rate_difference"] = (
        matches["home_recent_draw_rate"] - matches["away_recent_draw_rate"]
    )
    matches["close_match_rate_difference"] = (
        matches["home_recent_close_match_rate"] - matches["away_recent_close_match_rate"]
    )
    matches["draw_rate_difference_10"] = (
        matches["home_recent_draw_rate_10"] - matches["away_recent_draw_rate_10"]
    )
    matches["close_match_rate_difference_10"] = (
        matches["home_recent_close_match_rate_10"] - matches["away_recent_close_match_rate_10"]
    )

    return matches


def add_rolling_previous_mean(team_matches, column, window):
    return (
        team_matches.groupby("team")[column]
        .transform(lambda values: values.shift(1).rolling(window, min_periods=1).mean())
    )


def add_tournament_context_features(matches):
    matches = matches.copy()
    matches["tournament"] = matches["tournament"].fillna("").astype(str)

    matches["is_world_cup"] = (matches["tournament"] == "FIFA World Cup").astype(int)
    matches["is_world_cup_qualifier"] = (
        matches["tournament"] == "FIFA World Cup qualification"
    ).astype(int)
    matches["is_continental_tournament"] = (
        matches["tournament"].isin(CONTINENTAL_TOURNAMENTS)
    ).astype(int)
    matches["is_friendly"] = (matches["tournament"] == "Friendly").astype(int)
    matches["match_importance_weight"] = matches["tournament"].map(get_match_importance_weight)

    return matches


def get_match_importance_weight(tournament):
    if tournament == "FIFA World Cup":
        return 5
    if tournament in CONTINENTAL_TOURNAMENTS:
        return 4
    if tournament == "FIFA World Cup qualification":
        return 3
    if "Nations League" in tournament or "qualification" in tournament:
        return 2
    if tournament == "Friendly":
        return 1

    return 2


def get_match_points(home_score, away_score):
    if home_score > away_score:
        return 3, 0
    if home_score < away_score:
        return 0, 3
    return 1, 1


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    model_dataset = build_model_dataset()
    print(f"Saved {len(model_dataset)} rows to {MODEL_DATASET_PATH}")


if __name__ == "__main__":
    main()
