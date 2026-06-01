from pathlib import Path

import pandas as pd
import streamlit as st


GROUP_PREDICTIONS_PATH = Path("results/predictions/world_cup_2026_group_predictions_v1.csv")
GROUP_TABLES_PATH = Path("results/predictions/world_cup_2026_group_tables_v1.csv")
SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_simulation_summary.csv")
MATCH_SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_match_simulation_summary.csv")
METADATA_PATH = Path("results/simulations/world_cup_2026_metadata.json")
ANNEX_C_PATH = Path("data/raw/world_cup_2026/annex_c_third_place_allocation.csv")

MODEL_METRICS = {
    "Model": "Multiclass Logistic Regression",
    "Draw weight": "1.6",
    "Accuracy": "0.581",
    "Log Loss": "0.890",
    "Macro F1": "0.530",
    "ROC-AUC": "0.740",
}

FEATURE_SET = [
    "elo_difference",
    "home_advantage",
    "home_goals_for_avg",
    "home_goals_against_avg",
    "away_goals_for_avg",
    "away_goals_against_avg",
]

TEAM_FLAGS = {
    "Algeria": "DZ",
    "Argentina": "AR",
    "Australia": "AU",
    "Austria": "AT",
    "Belgium": "BE",
    "Bosnia and Herzegovina": "BA",
    "Brazil": "BR",
    "Canada": "CA",
    "Cape Verde": "CV",
    "Colombia": "CO",
    "Croatia": "HR",
    "Curaçao": "CW",
    "Czech Republic": "CZ",
    "Dem. Rep. of Congo": "CD",
    "Ecuador": "EC",
    "Egypt": "EG",
    "England": "GB-ENG",
    "France": "FR",
    "Germany": "DE",
    "Ghana": "GH",
    "Haiti": "HT",
    "Iran": "IR",
    "Iraq": "IQ",
    "Ivory Coast": "CI",
    "Japan": "JP",
    "Jordan": "JO",
    "Mexico": "MX",
    "Morocco": "MA",
    "Netherlands": "NL",
    "New Zealand": "NZ",
    "Norway": "NO",
    "Panama": "PA",
    "Paraguay": "PY",
    "Portugal": "PT",
    "Qatar": "QA",
    "Saudi Arabia": "SA",
    "Scotland": "GB-SCT",
    "Senegal": "SN",
    "South Africa": "ZA",
    "South Korea": "KR",
    "Spain": "ES",
    "Sweden": "SE",
    "Switzerland": "CH",
    "Tunisia": "TN",
    "Turkey": "TR",
    "United States": "US",
    "Uruguay": "UY",
    "Uzbekistan": "UZ",
}

SUBDIVISION_FLAGS = {
    "GB-ENG": "\U0001F3F4\U000E0067\U000E0062\U000E0065\U000E006E\U000E0067\U000E007F",
    "GB-SCT": "\U0001F3F4\U000E0067\U000E0062\U000E0073\U000E0063\U000E0074\U000E007F",
}


st.set_page_config(
    page_title="World Cup Match Outcome Predictor",
    layout="wide",
)


NAV_ITEMS = [
    "Overview",
    "Group Predictions",
    "Group Tables",
    "Simulation Results",
    "Knockout Summary",
    "Methodology",
]


def inject_dark_theme():
    st.markdown(
        """
        <style>
        :root {
            --wc-bg: #0b1120;
            --wc-panel: #111827;
            --wc-panel-soft: #162033;
            --wc-border: #263244;
            --wc-text: #e5e7eb;
            --wc-muted: #94a3b8;
            --wc-accent: #38bdf8;
            --wc-accent-strong: #22c55e;
        }

        .stApp {
            background: var(--wc-bg);
            color: var(--wc-text);
        }

        [data-testid="stSidebar"] {
            background: #080d18;
            border-right: 1px solid var(--wc-border);
        }

        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] span {
            color: var(--wc-text);
        }

        .block-container {
            padding-top: 1.4rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }

        h1, h2, h3 {
            color: var(--wc-text);
            letter-spacing: 0;
        }

        h1 {
            font-size: 1.85rem;
            line-height: 1.2;
        }

        h2 {
            font-size: 1.25rem;
        }

        h3 {
            font-size: 1.05rem;
        }

        .wc-page-kicker {
            color: var(--wc-accent);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.04em;
            text-transform: uppercase;
            margin-bottom: 0.15rem;
        }

        .wc-page-title {
            font-size: 1.9rem;
            font-weight: 750;
            line-height: 1.15;
            color: var(--wc-text);
            margin-bottom: 0.25rem;
        }

        .wc-page-caption {
            color: var(--wc-muted);
            font-size: 0.95rem;
            margin-bottom: 1.15rem;
        }

        .wc-sidebar-title {
            font-size: 1rem;
            font-weight: 800;
            line-height: 1.2;
            color: var(--wc-text);
            margin-bottom: 0.2rem;
        }

        .wc-sidebar-caption {
            color: var(--wc-muted);
            font-size: 0.82rem;
            line-height: 1.35;
            margin-bottom: 1rem;
        }

        [data-testid="stMetric"] {
            background: var(--wc-panel);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--wc-muted);
            font-size: 0.75rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--wc-text);
            font-size: 1.2rem;
        }

        [data-testid="stDataFrame"] {
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            overflow: hidden;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: var(--wc-panel);
            border-color: var(--wc-border);
            color: var(--wc-text);
        }

        .stTextInput input {
            color: var(--wc-text);
        }

        .stAlert {
            border-radius: 8px;
        }

        hr {
            border-color: var(--wc-border);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar_nav():
    with st.sidebar:
        st.markdown(
            """
            <div class="wc-sidebar-title">World Cup Predictor</div>
            <div class="wc-sidebar-caption">
            Logistic Regression forecast dashboard for group predictions and 10,000 tournament simulations.
            </div>
            """,
            unsafe_allow_html=True,
        )
        selected_page = st.radio("Navigation", NAV_ITEMS, label_visibility="collapsed")
        st.divider()
        st.caption("Model: Logistic Regression")
        st.caption("Draw weight: 1.6")
        st.caption("Mode: dashboard only")
    return selected_page


def render_page_header(title, caption):
    st.markdown(
        f"""
        <div class="wc-page-kicker">2026 World Cup Analytics</div>
        <div class="wc-page-title">{title}</div>
        <div class="wc-page-caption">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def country_code_to_flag(country_code):
    if not country_code:
        return ""
    if country_code in SUBDIVISION_FLAGS:
        return SUBDIVISION_FLAGS[country_code]
    if len(country_code) != 2:
        return ""

    base = 127397
    return "".join(chr(base + ord(character)) for character in country_code.upper())


def team_label(team):
    flag = country_code_to_flag(TEAM_FLAGS.get(str(team), ""))
    if not flag:
        return str(team)
    return f"{flag} {team}"


def team_options(teams):
    return {team_label(team): team for team in sorted(set(teams))}


def add_team_display_columns(data, columns):
    display = data.copy()
    for column in columns:
        if column in display.columns:
            display[f"{column}_display"] = display[column].map(team_label)
    return display


def add_chart_team_labels(data, team_column="team"):
    display = data.copy()
    if team_column in display.columns:
        display["team_display"] = display[team_column].map(team_label)
    return display


def require_file(path, upstream_hint):
    if not Path(path).exists():
        st.error(f"Missing required file: {path}")
        st.info(upstream_hint)
        st.stop()


@st.cache_data
def load_group_predictions():
    require_file(
        GROUP_PREDICTIONS_PATH,
        "Run `python3 src/build_2026_prediction_dataset.py` before opening the dashboard.",
    )
    data = pd.read_csv(GROUP_PREDICTIONS_PATH)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    return data


@st.cache_data
def load_group_tables():
    require_file(
        GROUP_TABLES_PATH,
        "Run `python3 src/build_2026_prediction_dataset.py` before opening the dashboard.",
    )
    return pd.read_csv(GROUP_TABLES_PATH)


@st.cache_data
def load_simulation_summary():
    require_file(
        SIMULATION_SUMMARY_PATH,
        "Run `python3 src/simulate_world_cup_2026.py` before opening the dashboard.",
    )
    return pd.read_csv(SIMULATION_SUMMARY_PATH)


@st.cache_data
def load_match_simulation_summary():
    if not MATCH_SIMULATION_SUMMARY_PATH.exists():
        return None
    return pd.read_csv(MATCH_SIMULATION_SUMMARY_PATH)


@st.cache_data
def load_metadata():
    require_file(
        METADATA_PATH,
        "Run `python3 src/simulate_world_cup_2026.py` before opening the dashboard.",
    )
    return pd.read_json(METADATA_PATH, typ="series").to_dict()


@st.cache_data
def load_annex_c_rows():
    if not ANNEX_C_PATH.exists():
        return None
    return len(pd.read_csv(ANNEX_C_PATH))


def format_probability_columns(data, columns):
    formatted = data.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(lambda value: f"{value:.1%}")
    return formatted


def add_outcome_summary(data):
    data = data.copy()
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    label_map = {
        "prob_home_win": "Home win",
        "prob_draw": "Draw",
        "prob_away_win": "Away win",
    }
    data["highest_probability"] = data[probability_columns].max(axis=1)
    data["highest_probability_outcome"] = data[probability_columns].idxmax(axis=1).map(label_map)
    return data


def filter_by_group_and_team(data, group_column="group"):
    filtered = data.copy()
    groups = sorted(filtered[group_column].dropna().unique())
    selected_group = st.selectbox("Group", ["All", *groups])
    if selected_group != "All":
        filtered = filtered[filtered[group_column] == selected_group]

    team_columns = [column for column in ["team", "home_team", "away_team"] if column in filtered.columns]
    teams = sorted(
        {
            team
            for column in team_columns
            for team in filtered[column].dropna().astype(str).tolist()
        }
    )
    option_map = team_options(teams)
    selected_team_label = st.selectbox("Team", ["All", *option_map.keys()])
    selected_team = option_map.get(selected_team_label, selected_team_label)
    if selected_team != "All":
        if {"home_team", "away_team"}.issubset(filtered.columns):
            filtered = filtered[
                (filtered["home_team"] == selected_team)
                | (filtered["away_team"] == selected_team)
            ]
        elif "team" in filtered.columns:
            filtered = filtered[filtered["team"] == selected_team]

    return filtered


def show_probability_bar_chart(data, label_column, probability_column, title, top_n=10):
    chart_source = data.copy()
    display_label_column = label_column
    if label_column == "team":
        chart_source = add_chart_team_labels(chart_source, team_column=label_column)
        display_label_column = "team_display"

    chart_data = (
        chart_source[[display_label_column, probability_column]]
        .sort_values(probability_column, ascending=False)
        .head(top_n)
        .set_index(display_label_column)
    )
    st.subheader(title)
    st.bar_chart(chart_data)


def render_overview(simulation_summary, metadata):
    render_page_header(
        "World Cup Match Outcome Predictor",
        "Selected model outputs, tournament simulation probabilities, and dashboard-ready metadata.",
    )

    metric_columns = st.columns(6)
    for column, (label, value) in zip(metric_columns, MODEL_METRICS.items()):
        column.metric(label, value)

    st.divider()

    meta_columns = st.columns(5)
    meta_columns[0].metric("Simulations", metadata.get("number_of_simulations", "Unknown"))
    meta_columns[1].metric("Base seed", metadata.get("base_seed", "Unknown"))
    meta_columns[2].metric("Annex C rows", metadata.get("annex_c_rows", load_annex_c_rows() or "Unknown"))
    meta_columns[3].metric("Draw weight", metadata.get("draw_weight", "1.6"))
    meta_columns[4].metric("Bracket", metadata.get("bracket_method", "Unknown"))

    st.text_input("Input fingerprint", value=str(metadata.get("input_fingerprint", "Unknown")), disabled=True)
    st.caption(f"Created at: {metadata.get('created_at', 'Unknown')}")

    show_probability_bar_chart(
        simulation_summary,
        "team",
        "champion_probability",
        "Top 10 Champion Probabilities",
        top_n=10,
    )


def render_group_predictions(group_predictions):
    render_page_header(
        "Group Predictions",
        "Fixture-level home, draw, and away probabilities for the confirmed group stage.",
    )
    predictions = add_outcome_summary(group_predictions)

    filter_columns = st.columns(3)
    with filter_columns[0]:
        groups = sorted(predictions["group"].dropna().unique())
        selected_group = st.selectbox("Group", ["All", *groups], key="fixture_group")
    with filter_columns[1]:
        teams = sorted(set(predictions["home_team"]).union(set(predictions["away_team"])))
        option_map = team_options(teams)
        selected_team_label = st.selectbox("Team", ["All", *option_map.keys()], key="fixture_team")
        selected_team = option_map.get(selected_team_label, selected_team_label)
    with filter_columns[2]:
        results = sorted(predictions["predicted_result"].dropna().unique())
        selected_result = st.selectbox("Predicted result", ["All", *results])

    filtered = predictions.copy()
    if selected_group != "All":
        filtered = filtered[filtered["group"] == selected_group]
    if selected_team != "All":
        filtered = filtered[
            (filtered["home_team"] == selected_team)
            | (filtered["away_team"] == selected_team)
        ]
    if selected_result != "All":
        filtered = filtered[filtered["predicted_result"] == selected_result]

    display_source = add_team_display_columns(filtered, ["home_team", "away_team"])
    display_columns = [
        "date",
        "group",
        "home_team_display",
        "away_team_display",
        "predicted_result",
        "highest_probability_outcome",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
    ]
    display = format_probability_columns(
        display_source[display_columns],
        ["prob_home_win", "prob_draw", "prob_away_win"],
    )
    display = display.rename(
        columns={
            "home_team_display": "home_team",
            "away_team_display": "away_team",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_group_tables(group_tables):
    render_page_header(
        "Predicted Group Tables",
        "Model-selected group outcomes using deterministic points from predicted results.",
    )
    groups = sorted(group_tables["group"].dropna().unique())
    selected_group = st.selectbox("Group view", ["All", *groups])

    if selected_group == "All":
        for group in groups:
            st.subheader(f"Group {group}")
            table = group_tables[group_tables["group"] == group].sort_values("group_position")
            table = add_team_display_columns(table, ["team"])
            table["team"] = table["team_display"]
            table = table.drop(columns=["team_display"])
            st.dataframe(table, use_container_width=True, hide_index=True)
    else:
        table = group_tables[group_tables["group"] == selected_group].sort_values("group_position")
        table = add_team_display_columns(table, ["team"])
        table["team"] = table["team_display"]
        table = table.drop(columns=["team_display"])
        st.dataframe(table, use_container_width=True, hide_index=True)


def render_simulation_results(simulation_summary):
    render_page_header(
        "Simulation Results",
        "Round-by-round advancement probabilities from 10,000 seeded tournament simulations.",
    )

    filtered = filter_by_group_and_team(simulation_summary, group_column="group")
    probability_columns = [
        "round_of_32_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    display_source = add_team_display_columns(filtered, ["team"])
    display_columns = ["team_display", "group", *probability_columns]
    display = format_probability_columns(display_source[display_columns], probability_columns)
    display = display.rename(columns={"team_display": "team"})
    st.dataframe(display, use_container_width=True, hide_index=True)

    chart_columns = st.columns(3)
    with chart_columns[0]:
        show_probability_bar_chart(
            simulation_summary,
            "team",
            "champion_probability",
            "Champion Probability Top 15",
            top_n=15,
        )
    with chart_columns[1]:
        show_probability_bar_chart(
            simulation_summary,
            "team",
            "final_probability",
            "Final Probability Top 15",
            top_n=15,
        )
    with chart_columns[2]:
        show_probability_bar_chart(
            simulation_summary,
            "team",
            "round_of_16_probability",
            "Round of 16 Probability Top 15",
            top_n=15,
        )

    selected_team = st.selectbox(
        "Advancement breakdown",
        list(team_options(simulation_summary["team"].dropna().unique()).keys()),
    )
    selected_team = team_options(simulation_summary["team"].dropna().unique())[selected_team]
    team_row = simulation_summary[simulation_summary["team"] == selected_team].iloc[0]
    breakdown = pd.Series(
        {
            "Round of 32": team_row["round_of_32_probability"],
            "Round of 16": team_row["round_of_16_probability"],
            "Quarter Final": team_row["quarter_final_probability"],
            "Semi Final": team_row["semi_final_probability"],
            "Final": team_row["final_probability"],
            "Champion": team_row["champion_probability"],
        }
    )
    st.subheader(f"{team_label(selected_team)} Advancement Probability")
    st.bar_chart(breakdown)


def render_knockout_summary(match_summary):
    render_page_header(
        "Knockout Simulation Summary",
        "Most common simulated knockout matchups and advancement outcomes.",
    )

    if match_summary is None:
        st.warning(
            f"Optional file missing: {MATCH_SIMULATION_SUMMARY_PATH}. "
            "Run `python3 src/simulate_world_cup_2026.py` to create it."
        )
        return

    filtered = match_summary.copy()
    filter_columns = st.columns(2)
    with filter_columns[0]:
        rounds = sorted(filtered["round"].dropna().unique())
        selected_round = st.selectbox("Round", ["All", *rounds])
    with filter_columns[1]:
        teams = sorted(set(filtered["home_team"]).union(set(filtered["away_team"])))
        option_map = team_options(teams)
        selected_team_label = st.selectbox("Team", ["All", *option_map.keys()], key="knockout_team")
        selected_team = option_map.get(selected_team_label, selected_team_label)

    if selected_round != "All":
        filtered = filtered[filtered["round"] == selected_round]
    if selected_team != "All":
        filtered = filtered[
            (filtered["home_team"] == selected_team)
            | (filtered["away_team"] == selected_team)
        ]

    display_columns = [
        "round",
        "home_team_display",
        "away_team_display",
        "simulations",
        "home_advancement_probability",
        "away_advancement_probability",
    ]
    filtered = filtered.sort_values("simulations", ascending=False)
    filtered = add_team_display_columns(filtered, ["home_team", "away_team"])
    display = format_probability_columns(
        filtered[display_columns],
        ["home_advancement_probability", "away_advancement_probability"],
    )
    display = display.rename(
        columns={
            "home_team_display": "home_team",
            "away_team_display": "away_team",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_methodology(metadata):
    render_page_header(
        "Methodology and Caveats",
        "What the model uses, what the simulation assumes, and where the current limits are.",
    )

    st.subheader("Model")
    st.write("Logistic Regression")

    st.subheader("Feature Set")
    st.code("\n".join(FEATURE_SET))

    st.subheader("Home Advantage")
    st.write("Only United States, Mexico, and Canada receive home advantage for 2026 prediction rows.")

    st.subheader("Simulation")
    st.write("10,000 seeded simulations.")
    st.code("simulation_seed = 2026 + simulation_id")

    st.subheader("Annex C")
    st.write("Official third-place allocation table parsed into 495 combinations.")

    st.subheader("Known Limitations")
    st.write(
        "- Outcome-level simulation only.\n"
        "- No simulated scorelines.\n"
        "- No goal difference or goals scored tie-breakers yet.\n"
        "- Knockout probabilities may use Elo fallback unless exact model probability rows exist.\n"
        "- Predictions are estimates, not betting advice."
    )

    st.subheader("Metadata")
    st.json(metadata)


def main():
    inject_dark_theme()
    group_predictions = load_group_predictions()
    group_tables = load_group_tables()
    simulation_summary = load_simulation_summary()
    match_summary = load_match_simulation_summary()
    metadata = load_metadata()
    selected_page = render_sidebar_nav()

    if selected_page == "Overview":
        render_overview(simulation_summary, metadata)
    elif selected_page == "Group Predictions":
        render_group_predictions(group_predictions)
    elif selected_page == "Group Tables":
        render_group_tables(group_tables)
    elif selected_page == "Simulation Results":
        render_simulation_results(simulation_summary)
    elif selected_page == "Knockout Summary":
        render_knockout_summary(match_summary)
    elif selected_page == "Methodology":
        render_methodology(metadata)


if __name__ == "__main__":
    main()
