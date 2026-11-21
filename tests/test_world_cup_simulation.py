import tempfile
import unittest
from itertools import combinations
from pathlib import Path

import pandas as pd

from src.simulate_world_cup_2026 import (
    ANNEX_C_COLUMNS,
    assign_group_points,
    parse_annex_c_wikitext,
    rank_group_table,
    run_single_simulation,
    validate_annex_allocations,
)


class WorldCupSimulationTest(unittest.TestCase):
    def test_annex_c_parser_returns_valid_495_row_lookup(self):
        source_path = Path("data/raw/world_cup_2026/annex_c_third_place_allocation_wikitext.txt")

        allocations = parse_annex_c_wikitext(source_path.read_text())
        validate_annex_allocations(allocations)

        self.assertEqual(len(allocations), 495)
        self.assertEqual(list(allocations.columns), ANNEX_C_COLUMNS)
        self.assertEqual(allocations["qualified_groups"].nunique(), 495)
        self.assertEqual(allocations.loc[0, "qualified_groups"], "EFGHIJKL")
        self.assertEqual(allocations.loc[0, "1A_vs"], "3E")

    def test_assign_group_points(self):
        self.assertEqual(assign_group_points("home_win"), (3, 0, 1, 0))
        self.assertEqual(assign_group_points("away_win"), (0, 3, 0, 1))
        self.assertEqual(assign_group_points("draw"), (1, 1, 0, 0))

    def test_rank_group_table_uses_points_wins_elo_then_team(self):
        table = pd.DataFrame(
            [
                {"group": "A", "team": "Beta", "points": 4, "wins": 1, "elo": 1500},
                {"group": "A", "team": "Alpha", "points": 4, "wins": 1, "elo": 1500},
                {"group": "A", "team": "Gamma", "points": 4, "wins": 1, "elo": 1600},
            ]
        )

        ranked = rank_group_table(table)

        self.assertEqual(ranked["team"].tolist(), ["Gamma", "Alpha", "Beta"])
        self.assertEqual(ranked["group_position"].tolist(), [1, 2, 3])

    def test_single_simulation_is_reproducible_for_same_seed(self):
        predictions = build_sample_group_predictions()
        annex = build_sample_annex_allocations()

        first = run_single_simulation(
            simulation_id=789,
            base_seed=2026,
            group_predictions=predictions,
            annex_allocations=annex,
        )
        second = run_single_simulation(
            simulation_id=789,
            base_seed=2026,
            group_predictions=predictions,
            annex_allocations=annex,
        )

        self.assertEqual(first["champion"], second["champion"])
        self.assertEqual(first["group_table"].to_dict("records"), second["group_table"].to_dict("records"))
        self.assertEqual(len(first["round_of_32_teams"]), 32)
        self.assertEqual(len(first["third_place_teams"]), 8)


def build_sample_group_predictions():
    rows = []
    match_id = 1
    for group_index, group in enumerate("ABCDEFGHIJKL"):
        teams = [f"{group}{position}" for position in range(1, 5)]
        for home_index, away_index in [(0, 1), (2, 3), (0, 2), (1, 3), (0, 3), (1, 2)]:
            rows.append(
                {
                    "match_id": match_id,
                    "date": "2026-06-01",
                    "group": group,
                    "location": "Test Stadium",
                    "home_team": teams[home_index],
                    "away_team": teams[away_index],
                    "home_elo": 2000 - group_index * 10 - home_index,
                    "away_elo": 2000 - group_index * 10 - away_index,
                    "elo_difference": away_index - home_index,
                    "prob_home_win": 0.70,
                    "prob_draw": 0.20,
                    "prob_away_win": 0.10,
                }
            )
            match_id += 1

    return pd.DataFrame(rows)


def build_sample_annex_allocations():
    rows = []
    for option_number, groups in enumerate(combinations("ABCDEFGHIJKL", 8), start=1):
        qualified_groups = "".join(groups)
        assignments = [f"3{group}" for group in qualified_groups]
        rows.append(
            {
                "option_number": option_number,
                "qualified_groups": qualified_groups,
                "1A_vs": assignments[0],
                "1B_vs": assignments[1],
                "1D_vs": assignments[2],
                "1E_vs": assignments[3],
                "1G_vs": assignments[4],
                "1I_vs": assignments[5],
                "1K_vs": assignments[6],
                "1L_vs": assignments[7],
            }
        )

    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
