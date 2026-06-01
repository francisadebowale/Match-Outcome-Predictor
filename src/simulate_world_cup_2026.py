from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


ANNEX_C_WIKITEXT_PATH = Path("data/raw/world_cup_2026/annex_c_third_place_allocation_wikitext.txt")
ANNEX_C_CSV_PATH = Path("data/raw/world_cup_2026/annex_c_third_place_allocation.csv")
ANNEX_C_JSON_PATH = Path("data/raw/world_cup_2026/annex_c_third_place_allocation.json")
FIXTURES_PATH = Path("data/raw/world_cup_2026/fixtures.csv")
GROUP_PREDICTIONS_PATH = Path("results/predictions/world_cup_2026_group_predictions_v1.csv")
SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_simulation_summary.csv")
MATCH_SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_match_simulation_summary.csv")
METADATA_PATH = Path("results/simulations/world_cup_2026_metadata.json")

NUMBER_OF_SIMULATIONS = 10000
BASE_SEED = 2026
MODEL_NAME = "Multiclass Logistic Regression"
DRAW_WEIGHT = 1.6
FEATURE_SET = [
    "elo_difference",
    "home_advantage",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
]
ANNEX_C_COLUMNS = [
    "option_number",
    "qualified_groups",
    "1A_vs",
    "1B_vs",
    "1D_vs",
    "1E_vs",
    "1G_vs",
    "1I_vs",
    "1K_vs",
    "1L_vs",
]
ANNEX_C_ASSIGNMENT_COLUMNS = ANNEX_C_COLUMNS[2:]
ROUND_OF_32_WINNER_SLOTS = ["1A", "1B", "1D", "1E", "1G", "1I", "1K", "1L"]


def parse_annex_c_wikitext(wikitext):
    rows = []
    row_pattern = re.compile(
        r"!\s*scope=\"row\"\s*\|\s*(\d+)(.*?)(?=\|-\s*!\s*scope=\"row\"\s*\||\|\})",
        flags=re.DOTALL,
    )

    for match in row_pattern.finditer(wikitext):
        option_number = int(match.group(1))
        row_text = match.group(2)
        qualified_groups = "".join(re.findall(r"'''([A-L])'''", row_text))
        opponents = re.findall(r"\b3[A-L]\b", row_text)

        if len(opponents) != 8:
            raise ValueError(
                f"Annex C option {option_number} has {len(opponents)} assigned opponents"
            )

        rows.append(
            {
                "option_number": option_number,
                "qualified_groups": qualified_groups,
                **dict(zip(ANNEX_C_ASSIGNMENT_COLUMNS, opponents)),
            }
        )

    allocations = pd.DataFrame(rows, columns=ANNEX_C_COLUMNS)
    validate_annex_allocations(allocations)
    return allocations


def validate_annex_allocations(allocations):
    if len(allocations) != 495:
        raise ValueError(f"Expected 495 Annex C rows, found {len(allocations)}")

    missing_columns = [column for column in ANNEX_C_COLUMNS if column not in allocations.columns]
    if missing_columns:
        raise ValueError(f"Annex C allocation is missing columns: {missing_columns}")

    if allocations["qualified_groups"].duplicated().any():
        duplicate_keys = allocations.loc[
            allocations["qualified_groups"].duplicated(), "qualified_groups"
        ].tolist()
        raise ValueError(f"Duplicate Annex C qualified_groups keys: {duplicate_keys}")

    for _, row in allocations.iterrows():
        qualified_groups = row["qualified_groups"]
        if len(qualified_groups) != 8:
            raise ValueError(
                f"Annex C option {row['option_number']} has {len(qualified_groups)} groups"
            )
        if len(set(qualified_groups)) != 8:
            raise ValueError(f"Annex C option {row['option_number']} has duplicate groups")

        assignments = [row[column] for column in ANNEX_C_ASSIGNMENT_COLUMNS]
        if any(not assignment for assignment in assignments):
            raise ValueError(f"Annex C option {row['option_number']} has a blank assignment")

        assigned_groups = [assignment[-1] for assignment in assignments]
        if len(assigned_groups) != 8:
            raise ValueError(f"Annex C option {row['option_number']} does not assign 8 teams")

        invalid_groups = sorted(set(assigned_groups) - set(qualified_groups))
        if invalid_groups:
            raise ValueError(
                f"Annex C option {row['option_number']} assigns groups that did not qualify: "
                f"{invalid_groups}"
            )


def write_annex_c_outputs(
    wikitext_path=ANNEX_C_WIKITEXT_PATH,
    csv_path=ANNEX_C_CSV_PATH,
    json_path=ANNEX_C_JSON_PATH,
):
    wikitext = Path(wikitext_path).read_text()
    allocations = parse_annex_c_wikitext(wikitext)

    csv_path = Path(csv_path)
    json_path = Path(json_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)

    allocations.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(allocations.to_dict("records"), indent=2))
    return allocations


def assign_group_points(result):
    if result == "home_win":
        return 3, 0, 1, 0
    if result == "away_win":
        return 0, 3, 0, 1
    if result == "draw":
        return 1, 1, 0, 0
    raise ValueError(f"Unknown match result: {result}")


def rank_group_table(table):
    ranked = table.sort_values(
        ["group", "points", "wins", "elo", "team"],
        ascending=[True, False, False, False, True],
    ).reset_index(drop=True)
    ranked["group_position"] = ranked.groupby("group").cumcount() + 1
    return ranked


def run_single_simulation(
    simulation_id,
    base_seed,
    group_predictions,
    annex_allocations,
    fixtures=None,
):
    rng = np.random.default_rng(base_seed + simulation_id)
    group_table = simulate_group_stage(group_predictions, rng)
    ranked_table = rank_group_table(group_table)
    qualifiers = select_qualified_teams(ranked_table)

    third_place_key = "".join(sorted(qualifiers["third_place_by_group"]))
    annex_match = annex_allocations[annex_allocations["qualified_groups"] == third_place_key]
    if annex_match.empty:
        raise ValueError(f"No Annex C row found for third-place groups: {third_place_key}")
    annex_row = annex_match.iloc[0].to_dict()

    team_elos = build_team_elo_lookup(group_predictions)
    match_probability_lookup = build_match_probability_lookup(group_predictions)
    round_of_32_matches = build_round_of_32_matches(
        qualifiers=qualifiers,
        annex_row=annex_row,
        fixtures=fixtures,
    )

    round_of_32_winners, round_of_32_records = simulate_knockout_round(
        round_name="Round of 32",
        matches=round_of_32_matches,
        team_elos=team_elos,
        match_probability_lookup=match_probability_lookup,
        rng=rng,
    )
    round_of_16_winners, round_of_16_records = simulate_ordered_knockout_round(
        "Round of 16", round_of_32_winners, team_elos, match_probability_lookup, rng
    )
    quarter_final_winners, quarter_final_records = simulate_ordered_knockout_round(
        "Quarter Finals", round_of_16_winners, team_elos, match_probability_lookup, rng
    )
    semi_final_winners, semi_final_records = simulate_ordered_knockout_round(
        "Semi Finals", quarter_final_winners, team_elos, match_probability_lookup, rng
    )
    final_winners, final_records = simulate_ordered_knockout_round(
        "Finals", semi_final_winners, team_elos, match_probability_lookup, rng
    )

    knockout_records = (
        round_of_32_records
        + round_of_16_records
        + quarter_final_records
        + semi_final_records
        + final_records
    )

    return {
        "simulation_id": simulation_id,
        "group_table": ranked_table,
        "group_winners": qualifiers["group_winners"],
        "group_runners_up": qualifiers["group_runners_up"],
        "third_place_teams": list(qualifiers["third_place_by_group"].values()),
        "round_of_32_teams": qualifiers["round_of_32_teams"],
        "round_of_16_teams": round_of_32_winners,
        "quarter_final_teams": round_of_16_winners,
        "semi_final_teams": quarter_final_winners,
        "final_teams": semi_final_winners,
        "champion": final_winners[0],
        "knockout_records": knockout_records,
    }


def simulate_group_stage(group_predictions, rng):
    table_rows = []

    for prediction in group_predictions.itertuples(index=False):
        probabilities = [
            float(prediction.prob_home_win),
            float(prediction.prob_draw),
            float(prediction.prob_away_win),
        ]
        probabilities = normalize_probabilities(probabilities)
        sampled_result = rng.choice(["home_win", "draw", "away_win"], p=probabilities)
        home_points, away_points, home_win, away_win = assign_group_points(sampled_result)

        table_rows.append(
            {
                "group": prediction.group,
                "team": prediction.home_team,
                "played": 1,
                "wins": home_win,
                "draws": int(sampled_result == "draw"),
                "losses": int(sampled_result == "away_win"),
                "points": home_points,
                "elo": float(prediction.home_elo),
            }
        )
        table_rows.append(
            {
                "group": prediction.group,
                "team": prediction.away_team,
                "played": 1,
                "wins": away_win,
                "draws": int(sampled_result == "draw"),
                "losses": int(sampled_result == "home_win"),
                "points": away_points,
                "elo": float(prediction.away_elo),
            }
        )

    return (
        pd.DataFrame(table_rows)
        .groupby(["group", "team"], as_index=False)
        .agg(
            played=("played", "sum"),
            wins=("wins", "sum"),
            draws=("draws", "sum"),
            losses=("losses", "sum"),
            points=("points", "sum"),
            elo=("elo", "max"),
        )
    )


def select_qualified_teams(ranked_table):
    group_winners = {}
    group_runners_up = {}
    third_place_by_group = {}

    for group, group_table in ranked_table.groupby("group"):
        ordered = group_table.sort_values(
            ["points", "wins", "elo", "team"],
            ascending=[False, False, False, True],
        ).reset_index(drop=True)
        group_winners[group] = ordered.loc[0, "team"]
        group_runners_up[group] = ordered.loc[1, "team"]
        third_place_by_group[group] = {
            "group": group,
            "team": ordered.loc[2, "team"],
            "points": ordered.loc[2, "points"],
            "wins": ordered.loc[2, "wins"],
            "elo": ordered.loc[2, "elo"],
        }

    best_third_place_rows = sorted(
        third_place_by_group.values(),
        key=lambda row: (-row["points"], -row["wins"], -row["elo"], row["team"]),
    )[:8]
    best_third_place_by_group = {
        row["group"]: row["team"] for row in best_third_place_rows
    }

    return {
        "group_winners": group_winners,
        "group_runners_up": group_runners_up,
        "third_place_by_group": best_third_place_by_group,
        "round_of_32_teams": (
            list(group_winners.values())
            + list(group_runners_up.values())
            + list(best_third_place_by_group.values())
        ),
    }


def build_round_of_32_matches(qualifiers, annex_row, fixtures=None):
    if fixtures is None:
        teams = qualifiers["round_of_32_teams"]
        return [
            {
                "round": "Round of 32",
                "match_id": index + 1,
                "home_team": teams[index],
                "away_team": teams[index + 16],
            }
            for index in range(16)
        ]

    fixtures = fixtures.copy()
    round_fixtures = fixtures[fixtures["Round Number"] == "Round of 32"].copy()
    round_fixtures["Date"] = pd.to_datetime(round_fixtures["Date"], dayfirst=True, errors="coerce")
    round_fixtures = round_fixtures.sort_values(["Date", "Match Number"])

    matches = []
    for _, fixture in round_fixtures.iterrows():
        home_token = fixture["Home Team"]
        away_token = fixture["Away Team"]
        matches.append(
            {
                "round": "Round of 32",
                "match_id": int(fixture["Match Number"]),
                "home_team": resolve_fixture_token(home_token, away_token, qualifiers, annex_row),
                "away_team": resolve_fixture_token(away_token, home_token, qualifiers, annex_row),
            }
        )

    if len(matches) != 16:
        raise ValueError(f"Expected 16 Round of 32 matches, found {len(matches)}")
    return matches


def resolve_fixture_token(token, paired_token, qualifiers, annex_row):
    token = str(token).strip()
    paired_token = str(paired_token).strip()

    if re.fullmatch(r"1[A-L]", token):
        return qualifiers["group_winners"][token[1]]
    if re.fullmatch(r"2[A-L]", token):
        return qualifiers["group_runners_up"][token[1]]
    if re.fullmatch(r"3[A-L]+", token):
        if re.fullmatch(r"1[A-L]", paired_token):
            assigned_token = annex_row[f"{paired_token}_vs"]
            return qualifiers["third_place_by_group"][assigned_token[1]]

        candidate_groups = token[1:]
        available_groups = sorted(set(candidate_groups) & set(qualifiers["third_place_by_group"]))
        if not available_groups:
            raise ValueError(f"Could not resolve third-place token {token}")
        return qualifiers["third_place_by_group"][available_groups[0]]

    raise ValueError(f"Unsupported fixture token: {token}")


def simulate_ordered_knockout_round(
    round_name, teams, team_elos, match_probability_lookup, rng
):
    matches = []
    for index in range(0, len(teams), 2):
        matches.append(
            {
                "round": round_name,
                "match_id": index // 2 + 1,
                "home_team": teams[index],
                "away_team": teams[index + 1],
            }
        )
    return simulate_knockout_round(
        round_name, matches, team_elos, match_probability_lookup, rng
    )


def simulate_knockout_round(round_name, matches, team_elos, match_probability_lookup, rng):
    winners = []
    records = []

    for match in matches:
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_prob, draw_prob, away_prob = get_match_probabilities(
            home_team, away_team, team_elos, match_probability_lookup
        )
        home_adv_prob, away_adv_prob = convert_to_advancement_probabilities(
            home_prob, draw_prob, away_prob
        )
        winner = rng.choice([home_team, away_team], p=[home_adv_prob, away_adv_prob])
        winners.append(winner)
        records.append(
            {
                "round": round_name,
                "match_id": match["match_id"],
                "home_team": home_team,
                "away_team": away_team,
                "winner": winner,
                "home_probability": home_prob,
                "draw_probability": draw_prob,
                "away_probability": away_prob,
            }
        )

    return winners, records


def get_match_probabilities(home_team, away_team, team_elos, match_probability_lookup):
    exact_key = (home_team, away_team)
    if exact_key in match_probability_lookup:
        return match_probability_lookup[exact_key]

    reversed_key = (away_team, home_team)
    if reversed_key in match_probability_lookup:
        away_prob, draw_prob, home_prob = match_probability_lookup[reversed_key]
        return home_prob, draw_prob, away_prob

    home_elo = team_elos.get(home_team)
    away_elo = team_elos.get(away_team)
    if home_elo is None or away_elo is None:
        return 0.5, 0.0, 0.5

    home_prob = 1 / (1 + 10 ** ((away_elo - home_elo) / 400))
    return home_prob, 0.0, 1 - home_prob


def convert_to_advancement_probabilities(home_prob, draw_prob, away_prob):
    denominator = home_prob + away_prob
    if denominator == 0:
        return 0.5, 0.5
    home_adv_prob = home_prob + draw_prob * home_prob / denominator
    away_adv_prob = away_prob + draw_prob * away_prob / denominator
    return normalize_probabilities([home_adv_prob, away_adv_prob])


def normalize_probabilities(probabilities):
    probabilities = np.array(probabilities, dtype=float)
    total = probabilities.sum()
    if total <= 0:
        return np.repeat(1 / len(probabilities), len(probabilities))
    return probabilities / total


def build_team_elo_lookup(group_predictions):
    team_elos = {}
    for prediction in group_predictions.itertuples(index=False):
        team_elos[prediction.home_team] = float(prediction.home_elo)
        team_elos[prediction.away_team] = float(prediction.away_elo)
    return team_elos


def build_match_probability_lookup(group_predictions):
    lookup = {}
    for prediction in group_predictions.itertuples(index=False):
        lookup[(prediction.home_team, prediction.away_team)] = (
            float(prediction.prob_home_win),
            float(prediction.prob_draw),
            float(prediction.prob_away_win),
        )
    return lookup


def run_simulations(
    number_of_simulations=NUMBER_OF_SIMULATIONS,
    base_seed=BASE_SEED,
    group_predictions_path=GROUP_PREDICTIONS_PATH,
    annex_csv_path=ANNEX_C_CSV_PATH,
    fixtures_path=FIXTURES_PATH,
    summary_path=SIMULATION_SUMMARY_PATH,
    match_summary_path=MATCH_SIMULATION_SUMMARY_PATH,
    metadata_path=METADATA_PATH,
):
    group_predictions = pd.read_csv(group_predictions_path)
    annex_allocations = pd.read_csv(annex_csv_path)
    fixtures = pd.read_csv(fixtures_path)
    validate_annex_allocations(annex_allocations)

    team_groups = get_team_groups(group_predictions)
    teams = sorted(team_groups)
    counters = {team: Counter() for team in teams}
    match_counters = defaultdict(Counter)

    for simulation_id in range(number_of_simulations):
        result = run_single_simulation(
            simulation_id=simulation_id,
            base_seed=base_seed,
            group_predictions=group_predictions,
            annex_allocations=annex_allocations,
            fixtures=fixtures,
        )
        update_team_counters(counters, result)
        update_match_counters(match_counters, result["knockout_records"])

    summary = build_simulation_summary(counters, team_groups, number_of_simulations)
    match_summary = build_match_simulation_summary(match_counters)

    summary_path = Path(summary_path)
    match_summary_path = Path(match_summary_path)
    metadata_path = Path(metadata_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    match_summary_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    summary.to_csv(summary_path, index=False)
    match_summary.to_csv(match_summary_path, index=False)
    metadata_path.write_text(
        json.dumps(
            build_metadata(
                number_of_simulations=number_of_simulations,
                base_seed=base_seed,
                input_files=[fixtures_path, group_predictions_path, annex_csv_path],
                annex_c_rows=len(annex_allocations),
            ),
            indent=2,
        )
    )

    return summary, match_summary


def get_team_groups(group_predictions):
    groups = {}
    for prediction in group_predictions.itertuples(index=False):
        groups[prediction.home_team] = prediction.group
        groups[prediction.away_team] = prediction.group
    return groups


def update_team_counters(counters, result):
    for team in result["group_winners"].values():
        counters[team]["group_winner_count"] += 1
        counters[team]["round_of_32_count"] += 1
    for team in result["group_runners_up"].values():
        counters[team]["group_runner_up_count"] += 1
        counters[team]["round_of_32_count"] += 1
    for team in result["third_place_teams"]:
        counters[team]["third_place_qualified_count"] += 1
        counters[team]["round_of_32_count"] += 1
    for team in result["round_of_16_teams"]:
        counters[team]["round_of_16_count"] += 1
    for team in result["quarter_final_teams"]:
        counters[team]["quarter_final_count"] += 1
    for team in result["semi_final_teams"]:
        counters[team]["semi_final_count"] += 1
    for team in result["final_teams"]:
        counters[team]["final_count"] += 1
    counters[result["champion"]]["champion_count"] += 1


def update_match_counters(match_counters, records):
    for record in records:
        key = (
            record["round"],
            record["home_team"],
            record["away_team"],
            round(record["home_probability"], 6),
            round(record["draw_probability"], 6),
            round(record["away_probability"], 6),
        )
        match_counters[key][record["winner"]] += 1


def build_simulation_summary(counters, team_groups, number_of_simulations):
    rows = []
    count_columns = [
        "group_winner_count",
        "group_runner_up_count",
        "third_place_qualified_count",
        "round_of_32_count",
        "round_of_16_count",
        "quarter_final_count",
        "semi_final_count",
        "final_count",
        "champion_count",
    ]
    probability_columns = {
        "group_winner_count": "group_winner_probability",
        "group_runner_up_count": "group_runner_up_probability",
        "third_place_qualified_count": "third_place_qualification_probability",
        "round_of_32_count": "round_of_32_probability",
        "round_of_16_count": "round_of_16_probability",
        "quarter_final_count": "quarter_final_probability",
        "semi_final_count": "semi_final_probability",
        "final_count": "final_probability",
        "champion_count": "champion_probability",
    }

    for team in sorted(counters):
        row = {
            "team": team,
            "group": team_groups[team],
            "simulations": number_of_simulations,
        }
        for column in count_columns:
            row[column] = counters[team][column]
        for count_column, probability_column in probability_columns.items():
            row[probability_column] = counters[team][count_column] / number_of_simulations
        rows.append(row)

    return pd.DataFrame(rows).sort_values(
        ["champion_probability", "final_probability", "team"],
        ascending=[False, False, True],
    )


def build_match_simulation_summary(match_counters):
    rows = []
    for key, winners in match_counters.items():
        round_name, home_team, away_team, home_probability, draw_probability, away_probability = key
        total = sum(winners.values())
        rows.append(
            {
                "round": round_name,
                "home_team": home_team,
                "away_team": away_team,
                "simulations": total,
                "home_probability": home_probability,
                "draw_probability": draw_probability,
                "away_probability": away_probability,
                "home_win_count": winners[home_team],
                "away_win_count": winners[away_team],
                "home_advancement_probability": winners[home_team] / total,
                "away_advancement_probability": winners[away_team] / total,
            }
        )
    return pd.DataFrame(rows).sort_values(["round", "simulations"], ascending=[True, False])


def build_metadata(number_of_simulations, base_seed, input_files, annex_c_rows):
    input_files = [str(Path(path)) for path in input_files]
    return {
        "base_seed": base_seed,
        "number_of_simulations": number_of_simulations,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_files": input_files,
        "input_fingerprint": fingerprint_files(input_files),
        "model_name": MODEL_NAME,
        "draw_weight": DRAW_WEIGHT,
        "feature_set": FEATURE_SET,
        "annex_c_rows": annex_c_rows,
        "bracket_method": "official_with_annex_c",
        "draw_handling_method": "knockout_draw_probability_redistributed_proportionally",
        "tie_breaker_method": "points_wins_elo_team_name",
        "notes": [
            "Tournament simulation uses outcome-level probabilities, not simulated scores.",
            "Knockout probabilities use exact group-stage prediction rows when available and Elo fallback otherwise.",
            "Full per-simulation match output is aggregated to keep dashboard data manageable.",
        ],
    }


def fingerprint_files(paths):
    digest = hashlib.sha256()
    for path in sorted(str(Path(path)) for path in paths):
        digest.update(path.encode("utf-8"))
        digest.update(Path(path).read_bytes())
    return digest.hexdigest()


def main():
    allocations = write_annex_c_outputs()
    summary, match_summary = run_simulations()
    print(f"Parsed {len(allocations)} Annex C rows to {ANNEX_C_CSV_PATH}")
    print(f"Saved simulation summary to {SIMULATION_SUMMARY_PATH}")
    print(f"Saved match simulation summary to {MATCH_SIMULATION_SUMMARY_PATH}")
    print(summary.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
