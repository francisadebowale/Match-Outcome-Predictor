import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.train_model import (
    DEFAULT_DRAW_WEIGHT,
    DRAW_WEIGHT_CANDIDATES,
    train_balanced_model,
    train_model,
    train_tuned_draw_weight_model,
)


class TrainModelTest(unittest.TestCase):
    def test_trains_model_and_saves_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "model_dataset.csv"
            metrics_path = temp_path / "metrics.csv"
            confusion_matrix_path = temp_path / "confusion_matrix.csv"

            rows = []
            outcomes = ["home_win", "draw", "away_win"]
            for index in range(36):
                rows.append(
                    {
                        "date": f"202{index // 12}-01-01",
                        "result": outcomes[index % 3],
                        "elo_difference": (index % 9) - 4,
                        "home_advantage": index % 2,
                        "home_goals_for_avg": 1.0 + (index % 3),
                        "home_goals_against_avg": 0.5 + (index % 2),
                        "away_goals_for_avg": 1.2 + (index % 3),
                        "away_goals_against_avg": 0.7 + (index % 2),
                        "recent_points_difference": (index % 5) - 2,
                        "home_recent_points_avg": 1.0 + (index % 3),
                        "away_recent_points_avg": 0.8 + (index % 3),
                        "home_recent_draw_rate": (index % 2) / 2,
                        "away_recent_draw_rate": ((index + 1) % 2) / 2,
                        "recent_goal_difference_difference": (index % 7) - 3,
                    }
                )

            pd.DataFrame(rows).to_csv(dataset_path, index=False)

            result = train_model(
                dataset_path=dataset_path,
                metrics_path=metrics_path,
                confusion_matrix_path=confusion_matrix_path,
                validation_start_date="2021-01-01",
                test_start_date="2022-01-01",
            )

            self.assertTrue(metrics_path.exists())
            self.assertTrue(confusion_matrix_path.exists())
            self.assertIn("validation_accuracy", result)
            self.assertIn("validation_log_loss", result)
            self.assertIn("validation_macro_precision", result)
            self.assertIn("validation_macro_recall", result)
            self.assertIn("validation_macro_f1", result)
            self.assertIn("validation_roc_auc_ovr", result)
            self.assertIn("test_accuracy", result)
            self.assertIn("test_log_loss", result)
            self.assertIn("test_macro_precision", result)
            self.assertIn("test_macro_recall", result)
            self.assertIn("test_macro_f1", result)
            self.assertIn("test_roc_auc_ovr", result)

    def test_trains_balanced_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "model_dataset.csv"
            metrics_path = temp_path / "balanced_metrics.csv"
            confusion_matrix_path = temp_path / "balanced_confusion_matrix.csv"

            rows = []
            outcomes = ["home_win", "home_win", "home_win", "draw", "away_win"]
            for index in range(50):
                rows.append(
                    {
                        "date": f"202{index // 15}-01-01",
                        "result": outcomes[index % len(outcomes)],
                        "elo_difference": (index % 9) - 4,
                        "home_advantage": index % 2,
                        "home_goals_for_avg": 1.0 + (index % 3),
                        "home_goals_against_avg": 0.5 + (index % 2),
                        "away_goals_for_avg": 1.2 + (index % 3),
                        "away_goals_against_avg": 0.7 + (index % 2),
                        "recent_points_difference": (index % 5) - 2,
                        "home_recent_points_avg": 1.0 + (index % 3),
                        "away_recent_points_avg": 0.8 + (index % 3),
                        "home_recent_draw_rate": (index % 2) / 2,
                        "away_recent_draw_rate": ((index + 1) % 2) / 2,
                        "recent_goal_difference_difference": (index % 7) - 3,
                    }
                )

            pd.DataFrame(rows).to_csv(dataset_path, index=False)

            result = train_balanced_model(
                dataset_path=dataset_path,
                metrics_path=metrics_path,
                confusion_matrix_path=confusion_matrix_path,
                validation_start_date="2021-01-01",
                test_start_date="2022-01-01",
            )

            self.assertTrue(metrics_path.exists())
            self.assertTrue(confusion_matrix_path.exists())
            self.assertIn("test_macro_f1", result)

    def test_trains_tuned_draw_weight_model(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            dataset_path = temp_path / "model_dataset.csv"
            metrics_path = temp_path / "tuned_metrics.csv"
            confusion_matrix_path = temp_path / "tuned_confusion_matrix.csv"
            candidates_path = temp_path / "tuned_candidates.csv"

            rows = []
            outcomes = ["home_win", "home_win", "home_win", "draw", "away_win"]
            for index in range(50):
                rows.append(
                    {
                        "date": f"202{index // 15}-01-01",
                        "result": outcomes[index % len(outcomes)],
                        "elo_difference": (index % 9) - 4,
                        "home_advantage": index % 2,
                        "home_goals_for_avg": 1.0 + (index % 3),
                        "home_goals_against_avg": 0.5 + (index % 2),
                        "away_goals_for_avg": 1.2 + (index % 3),
                        "away_goals_against_avg": 0.7 + (index % 2),
                        "recent_points_difference": (index % 5) - 2,
                        "home_recent_points_avg": 1.0 + (index % 3),
                        "away_recent_points_avg": 0.8 + (index % 3),
                        "home_recent_draw_rate": (index % 2) / 2,
                        "away_recent_draw_rate": ((index + 1) % 2) / 2,
                        "recent_goal_difference_difference": (index % 7) - 3,
                    }
                )

            pd.DataFrame(rows).to_csv(dataset_path, index=False)

            result = train_tuned_draw_weight_model(
                dataset_path=dataset_path,
                metrics_path=metrics_path,
                confusion_matrix_path=confusion_matrix_path,
                candidates_path=candidates_path,
                validation_start_date="2021-01-01",
                test_start_date="2022-01-01",
                draw_weight_candidates=[1.25, 1.5],
                target_draw_weight=1.5,
            )

            self.assertTrue(metrics_path.exists())
            self.assertTrue(confusion_matrix_path.exists())
            self.assertTrue(candidates_path.exists())
            self.assertIn("draw_weight", result)
            self.assertIn(result["draw_weight"], [1.25, 1.5])

    def test_default_draw_weight_is_moderate(self):
        self.assertEqual(DEFAULT_DRAW_WEIGHT, 1.6)

    def test_draw_weight_candidates_cover_one_to_two(self):
        self.assertEqual(DRAW_WEIGHT_CANDIDATES[0], 1.0)
        self.assertEqual(DRAW_WEIGHT_CANDIDATES[-1], 2.0)
        self.assertEqual(len(DRAW_WEIGHT_CANDIDATES), 11)

if __name__ == "__main__":
    unittest.main()
