import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.build_2026_prediction_dataset import build_group_tables, build_prediction_dataset


class Build2026PredictionDatasetTest(unittest.TestCase):
    def test_builds_group_fixture_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fixtures_path = temp_path / "fixtures.csv"
            history_path = temp_path / "history.csv"
            elo_path = temp_path / "elo.csv"
            team_map_path = temp_path / "team_name_map.csv"
            output_path = temp_path / "prediction_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "Match Number": 1,
                        "Round Number": 1,
                        "Date": "11/06/2026 19:00",
                        "Location": "Mexico City Stadium",
                        "Home Team": "Mexico",
                        "Away Team": "South Africa",
                        "Group": "Group A",
                        "Result": "",
                    }
                ]
            ).to_csv(fixtures_path, index=False)

            pd.DataFrame(
                [
                    {
                        "date": "2026-01-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 0,
                    },
                    {
                        "date": "2026-02-01",
                        "home_team": "South Africa",
                        "away_team": "Mexico",
                        "home_score": 1,
                        "away_score": 1,
                    },
                ]
            ).to_csv(history_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2026-06-04", "team": "Mexico", "elo": 1868},
                    {"snapshot_date": "2026-06-04", "team": "South Africa", "elo": 1517},
                ]
            ).to_csv(elo_path, index=False)

            pd.DataFrame(columns=["source_name", "standard_name"]).to_csv(
                team_map_path, index=False
            )

            dataset = build_prediction_dataset(
                fixtures_path=fixtures_path,
                history_path=history_path,
                elo_path=elo_path,
                team_name_map_path=team_map_path,
                output_path=output_path,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.loc[0, "home_elo"], 1868)
            self.assertEqual(dataset.loc[0, "away_elo"], 1517)
            self.assertEqual(dataset.loc[0, "elo_difference"], 351)
            self.assertEqual(dataset.loc[0, "home_advantage"], 1)
            self.assertEqual(dataset.loc[0, "home_recent_points_avg"], 2)
            self.assertEqual(dataset.loc[0, "away_recent_points_avg"], 0.5)
            self.assertEqual(dataset.loc[0, "recent_points_difference"], 1.5)
            self.assertEqual(dataset.loc[0, "home_recent_draw_rate"], 0.5)
            self.assertEqual(dataset.loc[0, "away_recent_draw_rate"], 0.5)
            self.assertEqual(dataset.loc[0, "recent_goal_difference_difference"], 2)

    def test_home_advantage_only_depends_on_home_team_host_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            fixtures_path = temp_path / "fixtures.csv"
            history_path = temp_path / "history.csv"
            elo_path = temp_path / "elo.csv"
            team_map_path = temp_path / "team_name_map.csv"
            output_path = temp_path / "prediction_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "Match Number": 1,
                        "Round Number": 1,
                        "Date": "11/06/2026 19:00",
                        "Location": "Los Angeles Stadium",
                        "Home Team": "Canada",
                        "Away Team": "South Africa",
                        "Group": "Group A",
                        "Result": "",
                    },
                    {
                        "Match Number": 2,
                        "Round Number": 1,
                        "Date": "12/06/2026 19:00",
                        "Location": "Toronto Stadium",
                        "Home Team": "Brazil",
                        "Away Team": "Canada",
                        "Group": "Group A",
                        "Result": "",
                    },
                ]
            ).to_csv(fixtures_path, index=False)

            pd.DataFrame(
                [
                    {
                        "date": "2026-01-01",
                        "home_team": "Canada",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 0,
                    },
                    {
                        "date": "2026-02-01",
                        "home_team": "Brazil",
                        "away_team": "Canada",
                        "home_score": 1,
                        "away_score": 1,
                    },
                    {
                        "date": "2026-03-01",
                        "home_team": "South Africa",
                        "away_team": "Brazil",
                        "home_score": 0,
                        "away_score": 1,
                    },
                ]
            ).to_csv(history_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2026-06-04", "team": "Canada", "elo": 1784},
                    {"snapshot_date": "2026-06-04", "team": "South Africa", "elo": 1517},
                    {"snapshot_date": "2026-06-04", "team": "Brazil", "elo": 1984},
                ]
            ).to_csv(elo_path, index=False)

            pd.DataFrame(columns=["source_name", "standard_name"]).to_csv(
                team_map_path, index=False
            )

            dataset = build_prediction_dataset(
                fixtures_path=fixtures_path,
                history_path=history_path,
                elo_path=elo_path,
                team_name_map_path=team_map_path,
                output_path=output_path,
            )

            canada_home = dataset[dataset["home_team"] == "Canada"].iloc[0]
            brazil_home = dataset[dataset["home_team"] == "Brazil"].iloc[0]

            self.assertEqual(canada_home["home_advantage"], 1)
            self.assertEqual(brazil_home["home_advantage"], 0)

    def test_builds_group_tables_from_predictions(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            predictions_path = temp_path / "predictions.csv"
            group_tables_path = temp_path / "group_tables.csv"

            pd.DataFrame(
                [
                    {
                        "group": "A",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "predicted_result": "home_win",
                    },
                    {
                        "group": "A",
                        "home_team": "South Korea",
                        "away_team": "Czech Republic",
                        "predicted_result": "draw",
                    },
                ]
            ).to_csv(predictions_path, index=False)

            table = build_group_tables(predictions_path, group_tables_path)

            self.assertTrue(group_tables_path.exists())
            mexico = table[table["team"] == "Mexico"].iloc[0]
            south_africa = table[table["team"] == "South Africa"].iloc[0]
            south_korea = table[table["team"] == "South Korea"].iloc[0]

            self.assertEqual(mexico["points"], 3)
            self.assertEqual(south_africa["points"], 0)
            self.assertEqual(south_korea["points"], 1)


if __name__ == "__main__":
    unittest.main()
