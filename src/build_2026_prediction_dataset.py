from pathlib import Path
import sys
import warnings

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.train_model import DEFAULT_DRAW_WEIGHT, FEATURE_COLUMNS


FIXTURES_PATH = Path("data/raw/world_cup_2026/fixtures.csv")
HISTORY_PATH = Path("data/processed/cleaned_international_results.csv")
WORLD_CUP_ELO_PATH = Path("data/processed/cleaned_elo_rankings_for_world_cup_2026.csv")
TEAM_NAME_MAP_PATH = Path("data/raw/team_name_map.csv")
MODEL_DATASET_PATH = Path("data/processed/model_dataset_v1.csv")
PREDICTION_DATASET_PATH = Path("data/processed/world_cup_2026_prediction_dataset_v1.csv")
PREDICTIONS_PATH = Path("results/predictions/world_cup_2026_group_predictions_v1.csv")
GROUP_TABLES_PATH = Path("results/predictions/world_cup_2026_group_tables_v1.csv")

DRAW_WEIGHT = DEFAULT_DRAW_WEIGHT
FORM_WINDOW = 5

LOCATION_COUNTRY_MAP = {
    "BC Place Vancouver": "Canada",
    "Toronto Stadium": "Canada",
    "Guadalajara Stadium": "Mexico",
    "Mexico City Stadium": "Mexico",
    "Monterrey Stadium": "Mexico",
}

HOST_TEAM_BY_COUNTRY = {
    "Canada": "Canada",
    "Mexico": "Mexico",
    "USA": "United States",
}
HOST_TEAMS = {"Canada", "Mexico", "United States"}


def build_prediction_dataset(
    fixtures_path=FIXTURES_PATH,
    history_path=HISTORY_PATH,
    elo_path=WORLD_CUP_ELO_PATH,
    team_name_map_path=TEAM_NAME_MAP_PATH,
    output_path=PREDICTION_DATASET_PATH,
):
    fixtures = pd.read_csv(fixtures_path)
    history = pd.read_csv(history_path)
    elo = pd.read_csv(elo_path)
    team_name_map = load_team_name_map(team_name_map_path)

    fixtures = clean_fixtures(fixtures, team_name_map)
    history = prepare_history(history, team_name_map)
    elo = prepare_world_cup_elo(elo, team_name_map)

    prediction_rows = []

    for fixture in fixtures.itertuples(index=False):
        home_form = get_recent_form(history, fixture.home_team, fixture.date)
        away_form = get_recent_form(history, fixture.away_team, fixture.date)

        home_elo = get_team_elo(elo, fixture.home_team)
        away_elo = get_team_elo(elo, fixture.away_team)

        prediction_rows.append(
            {
                "match_id": fixture.match_id,
                "round_number": fixture.round_number,
                "date": fixture.date,
                "location": fixture.location,
                "venue_country": fixture.venue_country,
                "group": fixture.group,
                "home_team": fixture.home_team,
                "away_team": fixture.away_team,
                "home_advantage": get_home_advantage(fixture.home_team),
                "home_elo": home_elo,
                "away_elo": away_elo,
                "elo_difference": home_elo - away_elo,
                "home_goals_for_avg": home_form["goals_for_avg"],
                "home_goals_against_avg": home_form["goals_against_avg"],
                "away_goals_for_avg": away_form["goals_for_avg"],
                "away_goals_against_avg": away_form["goals_against_avg"],
                "recent_points_difference": (
                    home_form["recent_points_avg"] - away_form["recent_points_avg"]
                ),
                "home_recent_points_avg": home_form["recent_points_avg"],
                "away_recent_points_avg": away_form["recent_points_avg"],
                "home_recent_draw_rate": home_form["recent_draw_rate"],
                "away_recent_draw_rate": away_form["recent_draw_rate"],
                "recent_goal_difference_difference": (
                    home_form["recent_goal_difference_avg"]
                    - away_form["recent_goal_difference_avg"]
                ),
            }
        )

    prediction_dataset = pd.DataFrame(prediction_rows)
    prediction_dataset = prediction_dataset.dropna(subset=FEATURE_COLUMNS).reset_index(drop=True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_dataset.to_csv(output_path, index=False)

    return prediction_dataset


def predict_world_cup_group_fixtures(
    model_dataset_path=MODEL_DATASET_PATH,
    prediction_dataset_path=PREDICTION_DATASET_PATH,
    predictions_path=PREDICTIONS_PATH,
):
    model_dataset = pd.read_csv(model_dataset_path)
    prediction_dataset = pd.read_csv(prediction_dataset_path)

    model_dataset = model_dataset.dropna(subset=["result", *FEATURE_COLUMNS])

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight={
                "away_win": 1.0,
                "draw": DRAW_WEIGHT,
                "home_win": 1.0,
            },
        ),
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        model.fit(model_dataset[FEATURE_COLUMNS], model_dataset["result"])

    probabilities = model.predict_proba(prediction_dataset[FEATURE_COLUMNS])
    predictions = model.predict(prediction_dataset[FEATURE_COLUMNS])
    labels = list(model.classes_)

    output = prediction_dataset[
        [
            "match_id",
            "date",
            "group",
            "location",
            "home_team",
            "away_team",
            "home_elo",
            "away_elo",
            "elo_difference",
        ]
    ].copy()
    output["predicted_result"] = predictions

    for index, label in enumerate(labels):
        output[f"prob_{label}"] = probabilities[:, index]

    output["home_team_result"] = output.apply(format_home_team_result, axis=1)

    predictions_path = Path(predictions_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(predictions_path, index=False)

    return output


def build_group_tables(
    predictions_path=PREDICTIONS_PATH,
    group_tables_path=GROUP_TABLES_PATH,
):
    predictions = pd.read_csv(predictions_path)
    table_rows = []

    for prediction in predictions.itertuples(index=False):
        home_points, away_points = get_points_for_prediction(prediction.predicted_result)

        table_rows.append(
            {
                "group": prediction.group,
                "team": prediction.home_team,
                "played": 1,
                "wins": int(prediction.predicted_result == "home_win"),
                "draws": int(prediction.predicted_result == "draw"),
                "losses": int(prediction.predicted_result == "away_win"),
                "points": home_points,
            }
        )
        table_rows.append(
            {
                "group": prediction.group,
                "team": prediction.away_team,
                "played": 1,
                "wins": int(prediction.predicted_result == "away_win"),
                "draws": int(prediction.predicted_result == "draw"),
                "losses": int(prediction.predicted_result == "home_win"),
                "points": away_points,
            }
        )

    group_tables = (
        pd.DataFrame(table_rows)
        .groupby(["group", "team"], as_index=False)
        .sum(numeric_only=True)
    )
    group_tables = group_tables.sort_values(
        ["group", "points", "wins", "team"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)
    group_tables["group_position"] = group_tables.groupby("group").cumcount() + 1

    group_tables = group_tables[
        [
            "group",
            "group_position",
            "team",
            "played",
            "wins",
            "draws",
            "losses",
            "points",
        ]
    ]

    group_tables_path = Path(group_tables_path)
    group_tables_path.parent.mkdir(parents=True, exist_ok=True)
    group_tables.to_csv(group_tables_path, index=False)

    return group_tables


def get_points_for_prediction(predicted_result):
    if predicted_result == "home_win":
        return 3, 0
    if predicted_result == "away_win":
        return 0, 3
    return 1, 1


def clean_fixtures(fixtures, team_name_map):
    fixtures = fixtures.rename(
        columns={
            "Match Number": "match_id",
            "Round Number": "round_number",
            "Date": "date",
            "Location": "location",
            "Home Team": "home_team",
            "Away Team": "away_team",
            "Group": "group",
        }
    )
    fixtures = fixtures.dropna(subset=["home_team", "away_team", "group"])
    fixtures = fixtures[fixtures["group"].str.startswith("Group")].copy()

    fixtures["date"] = pd.to_datetime(fixtures["date"], dayfirst=True, errors="coerce")
    fixtures["group"] = fixtures["group"].str.replace("Group ", "", regex=False).str.strip()
    fixtures["home_team"] = fixtures["home_team"].str.strip().replace(team_name_map)
    fixtures["away_team"] = fixtures["away_team"].str.strip().replace(team_name_map)
    fixtures["venue_country"] = fixtures["location"].map(get_venue_country)

    fixtures = fixtures.dropna(subset=["date", "venue_country"])
    return fixtures.sort_values(["date", "match_id"]).reset_index(drop=True)


def prepare_history(history, team_name_map):
    history = history.copy()
    history["date"] = pd.to_datetime(history["date"], errors="coerce")
    history["home_team"] = history["home_team"].str.strip().replace(team_name_map)
    history["away_team"] = history["away_team"].str.strip().replace(team_name_map)
    history = history.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    return history


def prepare_world_cup_elo(elo, team_name_map):
    elo = elo.copy()
    elo["team"] = elo["team"].str.strip().replace(team_name_map)
    elo["elo"] = elo["elo"].astype(int)
    return elo


def get_recent_form(history, team, fixture_date, window=FORM_WINDOW):
    previous_matches = history[
        (history["date"] < fixture_date)
        & ((history["home_team"] == team) | (history["away_team"] == team))
    ].sort_values("date").tail(window)

    if previous_matches.empty:
        return {
            "goals_for_avg": None,
            "goals_against_avg": None,
            "recent_points_avg": None,
            "recent_draw_rate": None,
            "recent_goal_difference_avg": None,
        }

    goals_for = []
    goals_against = []
    points = []
    draws = []
    goal_differences = []

    for match in previous_matches.itertuples(index=False):
        if match.home_team == team:
            goals_for.append(match.home_score)
            goals_against.append(match.away_score)
            team_points, _ = get_match_points(match.home_score, match.away_score)
        else:
            goals_for.append(match.away_score)
            goals_against.append(match.home_score)
            _, team_points = get_match_points(match.home_score, match.away_score)

        points.append(team_points)
        draws.append(int(match.home_score == match.away_score))
        goal_differences.append(goals_for[-1] - goals_against[-1])

    return {
        "goals_for_avg": sum(goals_for) / len(goals_for),
        "goals_against_avg": sum(goals_against) / len(goals_against),
        "recent_points_avg": sum(points) / len(points),
        "recent_draw_rate": sum(draws) / len(draws),
        "recent_goal_difference_avg": sum(goal_differences) / len(goal_differences),
    }


def get_match_points(home_score, away_score):
    if home_score > away_score:
        return 3, 0
    if home_score < away_score:
        return 0, 3
    return 1, 1


def get_team_elo(elo, team):
    matches = elo[elo["team"] == team]

    if matches.empty:
        raise ValueError(f"No World Cup Elo rating found for {team}")

    return int(matches.iloc[0]["elo"])


def get_venue_country(location):
    if location in LOCATION_COUNTRY_MAP:
        return LOCATION_COUNTRY_MAP[location]

    return "USA"


def get_home_advantage(team):
    return int(team in HOST_TEAMS)


def load_team_name_map(path):
    if not Path(path).exists():
        return {}

    name_map = pd.read_csv(path)
    return dict(zip(name_map["source_name"], name_map["standard_name"]))


def format_home_team_result(row):
    if row["predicted_result"] == "home_win":
        return f"{row['home_team']} win"
    if row["predicted_result"] == "away_win":
        return f"{row['away_team']} win"
    return "draw"


def main():
    prediction_dataset = build_prediction_dataset()
    predictions = predict_world_cup_group_fixtures()
    group_tables = build_group_tables()

    print(f"Saved {len(prediction_dataset)} prediction rows to {PREDICTION_DATASET_PATH}")
    print(f"Saved fixture predictions to {PREDICTIONS_PATH}")
    print(f"Saved group tables to {GROUP_TABLES_PATH}")
    print(predictions[["date", "group", "home_team", "away_team", "home_team_result"]].to_string(index=False))
    print()
    print(group_tables.to_string(index=False))


if __name__ == "__main__":
    main()
