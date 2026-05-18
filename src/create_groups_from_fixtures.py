from pathlib import Path

import pandas as pd


FIXTURES_PATH = Path("data/raw/world_cup_2026/fixtures.csv")
GROUPS_PATH = Path("data/raw/world_cup_2026/groups.csv")


def main():
    fixtures = pd.read_csv(FIXTURES_PATH)

    fixtures.columns = (
        fixtures.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
    )

    required_columns = {"home_team", "away_team", "group"}
    missing_columns = required_columns - set(fixtures.columns)

    if missing_columns:
        raise ValueError(f"Missing columns in fixtures file: {sorted(missing_columns)}")

    group_matches = fixtures.dropna(subset=["home_team", "away_team", "group"]).copy()
    group_matches = group_matches[group_matches["group"].str.startswith("Group")]

    home_teams = group_matches[["group", "home_team"]].rename(columns={"home_team": "team"})
    away_teams = group_matches[["group", "away_team"]].rename(columns={"away_team": "team"})

    groups = pd.concat([home_teams, away_teams], ignore_index=True)
    groups["group"] = groups["group"].str.replace("Group ", "", regex=False).str.strip()
    groups["team"] = groups["team"].str.strip()

    groups = groups.drop_duplicates().sort_values(["group", "team"]).reset_index(drop=True)
    groups.to_csv(GROUPS_PATH, index=False)

    print(f"Saved {len(groups)} teams to {GROUPS_PATH}")


if __name__ == "__main__":
    main()
