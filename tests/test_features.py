import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.features import build_model_dataset


class BuildModelDatasetTest(unittest.TestCase):
    def test_adds_home_and_away_elo_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.csv"
            elo_path = temp_path / "elo.csv"
            output_path = temp_path / "model_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2026-05-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 1,
                        "away_score": 0,
                        "neutral": False,
                        "result": "home_win",
                    },
                    {
                        "date": "2026-06-11",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 1,
                        "neutral": False,
                        "result": "home_win",
                    }
                ]
            ).to_csv(results_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2025-12-31", "team": "Mexico", "elo": 1868},
                    {"snapshot_date": "2025-12-31", "team": "South Africa", "elo": 1517},
                    {"snapshot_date": "2026-06-04", "team": "Mexico", "elo": 1900},
                    {"snapshot_date": "2026-06-04", "team": "South Africa", "elo": 1500},
                ]
            ).to_csv(elo_path, index=False)

            dataset = build_model_dataset(results_path, elo_path, output_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.loc[0, "home_elo"], 1868)
            self.assertEqual(dataset.loc[0, "away_elo"], 1517)
            self.assertEqual(dataset.loc[0, "elo_difference"], 351)
            self.assertTrue(output_path.exists())

    def test_adds_previous_match_goal_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.csv"
            elo_path = temp_path / "elo.csv"
            output_path = temp_path / "model_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 0,
                        "neutral": False,
                        "result": "home_win",
                    },
                    {
                        "date": "2025-02-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 1,
                        "away_score": 1,
                        "neutral": False,
                        "result": "draw",
                    },
                ]
            ).to_csv(results_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2024-12-31", "team": "Mexico", "elo": 1800},
                    {"snapshot_date": "2024-12-31", "team": "South Africa", "elo": 1500},
                ]
            ).to_csv(elo_path, index=False)

            dataset = build_model_dataset(results_path, elo_path, output_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.loc[0, "home_goals_for_avg"], 2)
            self.assertEqual(dataset.loc[0, "home_goals_against_avg"], 0)
            self.assertEqual(dataset.loc[0, "away_goals_for_avg"], 0)
            self.assertEqual(dataset.loc[0, "away_goals_against_avg"], 2)

    def test_adds_previous_match_recent_form_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.csv"
            elo_path = temp_path / "elo.csv"
            output_path = temp_path / "model_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 0,
                        "neutral": False,
                        "result": "home_win",
                    },
                    {
                        "date": "2025-02-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 1,
                        "away_score": 1,
                        "neutral": False,
                        "result": "draw",
                    },
                    {
                        "date": "2025-03-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 0,
                        "away_score": 1,
                        "neutral": False,
                        "result": "away_win",
                    },
                ]
            ).to_csv(results_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2024-12-31", "team": "Mexico", "elo": 1800},
                    {"snapshot_date": "2024-12-31", "team": "South Africa", "elo": 1500},
                ]
            ).to_csv(elo_path, index=False)

            dataset = build_model_dataset(results_path, elo_path, output_path)

            self.assertEqual(len(dataset), 2)
            latest_match = dataset.iloc[1]
            self.assertEqual(latest_match["home_recent_points_avg"], 2)
            self.assertEqual(latest_match["away_recent_points_avg"], 0.5)
            self.assertEqual(latest_match["recent_points_difference"], 1.5)
            self.assertEqual(latest_match["home_recent_draw_rate"], 0.5)
            self.assertEqual(latest_match["away_recent_draw_rate"], 0.5)
            self.assertEqual(latest_match["recent_goal_difference_difference"], 2)

    def test_adds_tournament_context_features(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.csv"
            elo_path = temp_path / "elo.csv"
            output_path = temp_path / "model_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 0,
                        "neutral": False,
                        "tournament": "FIFA World Cup qualification",
                        "result": "home_win",
                    },
                    {
                        "date": "2025-02-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 1,
                        "away_score": 1,
                        "neutral": False,
                        "tournament": "FIFA World Cup",
                        "result": "draw",
                    },
                ]
            ).to_csv(results_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2024-12-31", "team": "Mexico", "elo": 1800},
                    {"snapshot_date": "2024-12-31", "team": "South Africa", "elo": 1500},
                ]
            ).to_csv(elo_path, index=False)

            dataset = build_model_dataset(results_path, elo_path, output_path)

            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset.loc[0, "tournament"], "FIFA World Cup")
            self.assertEqual(dataset.loc[0, "is_world_cup"], 1)
            self.assertEqual(dataset.loc[0, "is_world_cup_qualifier"], 0)
            self.assertEqual(dataset.loc[0, "is_continental_tournament"], 0)
            self.assertEqual(dataset.loc[0, "is_friendly"], 0)
            self.assertEqual(dataset.loc[0, "match_importance_weight"], 5)

    def test_adds_close_match_and_low_scoring_rates_without_current_match(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            results_path = temp_path / "results.csv"
            elo_path = temp_path / "elo.csv"
            output_path = temp_path / "model_dataset.csv"

            pd.DataFrame(
                [
                    {
                        "date": "2025-01-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 2,
                        "away_score": 0,
                        "neutral": False,
                        "tournament": "Friendly",
                        "result": "home_win",
                    },
                    {
                        "date": "2025-02-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 1,
                        "away_score": 1,
                        "neutral": False,
                        "tournament": "Friendly",
                        "result": "draw",
                    },
                    {
                        "date": "2025-03-01",
                        "home_team": "Mexico",
                        "away_team": "South Africa",
                        "home_score": 0,
                        "away_score": 3,
                        "neutral": False,
                        "tournament": "Friendly",
                        "result": "away_win",
                    },
                ]
            ).to_csv(results_path, index=False)

            pd.DataFrame(
                [
                    {"snapshot_date": "2024-12-31", "team": "Mexico", "elo": 1800},
                    {"snapshot_date": "2024-12-31", "team": "South Africa", "elo": 1500},
                ]
            ).to_csv(elo_path, index=False)

            dataset = build_model_dataset(results_path, elo_path, output_path)
            latest_match = dataset.iloc[1]

            self.assertEqual(latest_match["home_recent_close_match_rate"], 0.5)
            self.assertEqual(latest_match["away_recent_close_match_rate"], 0.5)
            self.assertEqual(latest_match["close_match_rate_difference"], 0)
            self.assertEqual(latest_match["home_recent_low_scoring_rate"], 1.0)
            self.assertEqual(latest_match["away_recent_low_scoring_rate"], 1.0)
            self.assertEqual(latest_match["draw_rate_difference"], 0)

if __name__ == "__main__":
    unittest.main()
