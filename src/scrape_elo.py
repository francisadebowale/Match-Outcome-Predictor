import pandas as pd
from pathlib import Path

url = "https://www.international-football.net/elo-ratings-table?year=2026&month=06&day=03&confed=&prev-year=&prev-month=06&prev-day=03"
output_path = Path("data/raw/elo/elo_ratings.csv")

output_path.parent.mkdir(parents=True, exist_ok=True)

tables = pd.read_html(url)
print(f"Found {len(tables)} tables")

ranking_tables = [table for table in tables if len(table.columns) == 4 and len(table) > 1]

if not ranking_tables:
    raise ValueError("No Elo ratings tables found on the page")

elo_table = pd.concat(ranking_tables, ignore_index=True)
print(f"Using {len(ranking_tables)} ranking tables with {len(elo_table)} total rows")

elo_table.columns = ["rank", "change", "team", "elo"]
elo_table = elo_table.dropna(subset=["team", "elo"])
elo_table.to_csv(output_path, index=False)

print(f"Saved Elo ratings to {output_path}")
