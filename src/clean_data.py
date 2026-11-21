from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
TEAM_NAME_MAP_PATH = RAW_DIR / "team_name_map.csv"


def clean_international_results():
    input_path = RAW_DIR / "international_results" / "results.csv"
    output_path = PROCESSED_DIR / "cleaned_international_results.csv"

    results = pd.read_csv(input_path)

    results["date"] = pd.to_datetime(results["date"], errors="coerce")
    results = results.dropna(subset=["date", "home_score", "away_score"])

    results["home_score"] = results["home_score"].astype(int)
    results["away_score"] = results["away_score"].astype(int)
    results["neutral"] = results["neutral"].astype(bool)
    results = apply_team_name_map(results, ["home_team", "away_team"])

    results["result"] = results.apply(get_match_result, axis=1)

    results = results.sort_values("date").reset_index(drop=True)
    results.to_csv(output_path, index=False)

    print(f"Saved {len(results)} rows to {output_path}")


def clean_elo_ratings():
    input_path = RAW_DIR / "elo" / "elo_ratings.csv"
    output_path = PROCESSED_DIR / "cleaned_elo_ratings.csv"

    elo = pd.read_csv(input_path)

    elo = elo.drop(columns=["change"], errors="ignore")
    elo = elo.dropna(subset=["team", "elo"])

    elo["rank"] = elo["rank"].astype(int)
    elo["elo"] = elo["elo"].astype(int)
    elo["team"] = elo["team"].str.strip()
    elo = apply_team_name_map(elo, ["team"])

    elo = elo.sort_values("rank").reset_index(drop=True)
    elo.to_csv(output_path, index=False)

    print(f"Saved {len(elo)} rows to {output_path}")


def clean_elo_snapshots():
    clean_elo_snapshot_file(
        RAW_DIR / "elo" / "elo_ratings_by_year.csv",
        PROCESSED_DIR / "cleaned_elo_ratings_by_year.csv",
    )
    clean_elo_snapshot_file(
        RAW_DIR / "elo" / "elo_rankings_for_world_cup_2026.csv",
        PROCESSED_DIR / "cleaned_elo_rankings_for_world_cup_2026.csv",
    )


def clean_elo_snapshot_file(input_path, output_path):
    if not input_path.exists():
        print(f"Skipping missing Elo file: {input_path}")
        return

    elo = pd.read_csv(input_path)

    elo = elo.drop(columns=["change"], errors="ignore")
    elo = elo.dropna(subset=["snapshot_date", "team", "elo"])

    elo["snapshot_date"] = pd.to_datetime(elo["snapshot_date"], errors="coerce")
    elo = elo.dropna(subset=["snapshot_date"])
    elo["snapshot_date"] = elo["snapshot_date"].dt.date.astype(str)

    elo["rank"] = elo["rank"].astype(int)
    elo["elo"] = elo["elo"].astype(int)
    elo["team"] = elo["team"].str.strip()
    elo = apply_team_name_map(elo, ["team"])

    elo = elo.sort_values(["snapshot_date", "rank"]).reset_index(drop=True)
    elo.to_csv(output_path, index=False)

    print(f"Saved {len(elo)} rows to {output_path}")


def clean_world_cup_files():
    input_dir = RAW_DIR / "world_cup"
    output_path = PROCESSED_DIR / "cleaned_world_cup_matches.csv"

    cleaned_files = []

    for file_path in sorted(input_dir.glob("*.csv")):
        world_cup = pd.read_csv(file_path)
        cleaned = clean_single_world_cup_file(world_cup, file_path.name)
        cleaned_files.append(cleaned)

    all_world_cup = pd.concat(cleaned_files, ignore_index=True)
    all_world_cup = all_world_cup.sort_values("date").reset_index(drop=True)
    all_world_cup.to_csv(output_path, index=False)

    print(f"Saved {len(all_world_cup)} rows to {output_path}")


def clean_single_world_cup_file(world_cup, source_file):
    world_cup = world_cup.copy()
    world_cup.columns = normalize_columns(world_cup.columns)

    rename_map = {
        "competition": "competition",
        "home": "home_team",
        "away": "away_team",
        "date": "date",
        "time": "time",
        "hgft": "home_goals",
        "agft": "away_goals",
        "hg": "home_goals",
        "ag": "away_goals",
        "hs": "home_shots",
        "as": "away_shots",
        "hst": "home_shots_on_target",
        "ast": "away_shots_on_target",
        "h_max": "home_max_odds",
        "d_max": "draw_max_odds",
        "a_max": "away_max_odds",
        "h_avg": "home_avg_odds",
        "d_avg": "draw_avg_odds",
        "a_avg": "away_avg_odds",
        "h-max": "home_max_odds",
        "d-max": "draw_max_odds",
        "a-max": "away_max_odds",
        "h-avg": "home_avg_odds",
        "d-avg": "draw_avg_odds",
        "a-avg": "away_avg_odds",
    }

    world_cup = world_cup.rename(columns=rename_map)

    keep_columns = [
        "competition",
        "date",
        "time",
        "home_team",
        "away_team",
        "home_goals",
        "away_goals",
        "home_shots",
        "away_shots",
        "home_shots_on_target",
        "away_shots_on_target",
        "home_max_odds",
        "draw_max_odds",
        "away_max_odds",
        "home_avg_odds",
        "draw_avg_odds",
        "away_avg_odds",
    ]

    available_columns = [column for column in keep_columns if column in world_cup.columns]
    world_cup = world_cup[available_columns]

    world_cup["date"] = pd.to_datetime(world_cup["date"], errors="coerce")
    world_cup = world_cup.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    world_cup["home_goals"] = world_cup["home_goals"].astype(int)
    world_cup["away_goals"] = world_cup["away_goals"].astype(int)
    world_cup = apply_team_name_map(world_cup, ["home_team", "away_team"])

    world_cup["result"] = world_cup.apply(
        lambda row: get_match_result(
            {
                "home_score": row["home_goals"],
                "away_score": row["away_goals"],
            }
        ),
        axis=1,
    )
    world_cup["source_file"] = source_file

    return world_cup


def normalize_columns(columns):
    return (
        pd.Index(columns)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )


def apply_team_name_map(dataframe, columns, team_name_map_path=TEAM_NAME_MAP_PATH):
    if not Path(team_name_map_path).exists():
        return dataframe

    name_map = pd.read_csv(team_name_map_path)
    name_map = dict(zip(name_map["source_name"], name_map["standard_name"]))

    dataframe = dataframe.copy()

    for column in columns:
        dataframe[column] = dataframe[column].str.strip().replace(name_map)

    return dataframe


def get_match_result(row):
    if row["home_score"] > row["away_score"]:
        return "home_win"
    if row["home_score"] < row["away_score"]:
        return "away_win"
    return "draw"


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    clean_international_results()
    clean_elo_ratings()
    clean_elo_snapshots()
    clean_world_cup_files()


if __name__ == "__main__":
    main()
