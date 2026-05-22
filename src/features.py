from pathlib import Path

import pandas as pd


PROCESSED_DIR = Path("data/processed")
RESULTS_PATH = PROCESSED_DIR / "cleaned_international_results.csv"
ELO_SNAPSHOTS_PATH = PROCESSED_DIR / "cleaned_elo_ratings_by_year.csv"
MODEL_DATASET_PATH = PROCESSED_DIR / "model_dataset_v1.csv"
START_DATE = "2010-01-01"
FORM_WINDOW = 5


def build_model_dataset(
    results_path=RESULTS_PATH,
    elo_path=ELO_SNAPSHOTS_PATH,
    output_path=MODEL_DATASET_PATH,
    start_date=START_DATE,
    form_window=FORM_WINDOW,
):
    results = pd.read_csv(results_path)
    elo = pd.read_csv(elo_path)

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    results = results[results["date"] >= pd.to_datetime(start_date)].copy()
    results = results.sort_values("date").reset_index(drop=True)

    results = add_recent_goal_features(results, form_window)
    results = add_elo_features(results, elo)
    results = results.dropna(
        subset=[
            "home_elo",
            "away_elo",
            "home_goals_for_avg",
            "home_goals_against_avg",
            "away_goals_for_avg",
            "away_goals_against_avg",
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


def add_recent_goal_features(matches, window):
    team_match_rows = []

    for match in matches.itertuples(index=False):
        team_match_rows.append(
            {
                "date": match.date,
                "team": match.home_team,
                "goals_for": match.home_score,
                "goals_against": match.away_score,
            }
        )
        team_match_rows.append(
            {
                "date": match.date,
                "team": match.away_team,
                "goals_for": match.away_score,
                "goals_against": match.home_score,
            }
        )

    team_matches = pd.DataFrame(team_match_rows).sort_values(["team", "date"])
    team_matches["goals_for_avg"] = (
        team_matches.groupby("team")["goals_for"]
        .transform(lambda scores: scores.shift(1).rolling(window, min_periods=1).mean())
    )
    team_matches["goals_against_avg"] = (
        team_matches.groupby("team")["goals_against"]
        .transform(lambda scores: scores.shift(1).rolling(window, min_periods=1).mean())
    )

    home_features = team_matches.rename(
        columns={
            "team": "home_team",
            "goals_for_avg": "home_goals_for_avg",
            "goals_against_avg": "home_goals_against_avg",
        }
    )
    away_features = team_matches.rename(
        columns={
            "team": "away_team",
            "goals_for_avg": "away_goals_for_avg",
            "goals_against_avg": "away_goals_against_avg",
        }
    )

    matches = matches.merge(
        home_features[["date", "home_team", "home_goals_for_avg", "home_goals_against_avg"]],
        on=["date", "home_team"],
        how="left",
    )
    matches = matches.merge(
        away_features[["date", "away_team", "away_goals_for_avg", "away_goals_against_avg"]],
        on=["date", "away_team"],
        how="left",
    )

    return matches


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    model_dataset = build_model_dataset()
    print(f"Saved {len(model_dataset)} rows to {MODEL_DATASET_PATH}")


if __name__ == "__main__":
    main()
