from pathlib import Path

import pandas as pd


ELO_DIR = Path("data/raw/elo")
YEARLY_ELO_PATH = ELO_DIR / "elo_ratings_by_year.csv"
WORLD_CUP_ELO_PATH = ELO_DIR / "elo_rankings_for_world_cup_2026.csv"

YEARLY_SNAPSHOT_YEARS = range(2009, 2026)
WORLD_CUP_ELO_DATE = "2026-06-04"


def build_elo_url(snapshot_date):
    date = pd.to_datetime(snapshot_date)

    return (
        "https://www.international-football.net/elo-ratings-table?"
        f"year={date.year}&month={date.month:02d}&day={date.day:02d}"
        f"&confed=&prev-year=&prev-month={date.month:02d}&prev-day={date.day:02d}"
    )


def fetch_elo_table(snapshot_date):
    url = build_elo_url(snapshot_date)
    tables = pd.read_html(url)
    ranking_tables = [table for table in tables if len(table.columns) == 4 and len(table) > 1]

    if not ranking_tables:
        raise ValueError(f"No Elo rankings found for {snapshot_date}")

    elo_table = pd.concat(ranking_tables, ignore_index=True)
    elo_table.columns = ["rank", "change", "team", "elo"]
    elo_table = elo_table.dropna(subset=["team", "elo"])
    elo_table = elo_table.drop(columns=["change"], errors="ignore")
    elo_table["snapshot_date"] = pd.to_datetime(snapshot_date).date().isoformat()

    return elo_table[["snapshot_date", "rank", "team", "elo"]]


def save_yearly_elo_snapshots():
    snapshots = []

    for year in YEARLY_SNAPSHOT_YEARS:
        snapshot_date = f"{year}-12-31"
        print(f"Fetching Elo snapshot for {snapshot_date}")
        snapshots.append(fetch_elo_table(snapshot_date))

    yearly_elo = pd.concat(snapshots, ignore_index=True)
    yearly_elo.to_csv(YEARLY_ELO_PATH, index=False)

    print(f"Saved yearly Elo snapshots to {YEARLY_ELO_PATH}")


def save_world_cup_elo_snapshot():
    print(f"Fetching World Cup Elo snapshot for {WORLD_CUP_ELO_DATE}")
    world_cup_elo = fetch_elo_table(WORLD_CUP_ELO_DATE)
    world_cup_elo.to_csv(WORLD_CUP_ELO_PATH, index=False)

    print(f"Saved World Cup Elo rankings to {WORLD_CUP_ELO_PATH}")


def main():
    ELO_DIR.mkdir(parents=True, exist_ok=True)

    save_yearly_elo_snapshots()
    save_world_cup_elo_snapshot()


if __name__ == "__main__":
    main()
