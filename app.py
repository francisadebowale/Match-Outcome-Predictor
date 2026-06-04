import html
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

        .wc-insight-strip {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.7rem 0 1.1rem;
        }

        .wc-insight {
            background: var(--wc-panel);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.8rem 0.9rem;
        }

        .wc-insight-label {
            color: var(--wc-muted);
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .wc-insight-value {
            color: var(--wc-text);
            font-size: 1.05rem;
            font-weight: 750;
        }

        .wc-caveat-strip {
            background: var(--wc-panel-soft);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            color: var(--wc-muted);
            font-size: 0.86rem;
            line-height: 1.45;
            padding: 0.85rem 1rem;
            margin-top: 1rem;
        }

        .wc-group-card {
            background: var(--wc-panel);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.85rem;
            margin-bottom: 0.85rem;
        }

        .wc-group-card-title {
            color: var(--wc-text);
            font-size: 1rem;
            font-weight: 800;
            margin-bottom: 0.55rem;
        }

        .wc-group-table {
            border-collapse: collapse;
            width: 100%;
            font-size: 0.84rem;
        }

        .wc-group-table th {
            color: var(--wc-muted);
            font-size: 0.68rem;
            font-weight: 800;
            padding: 0.35rem 0.25rem;
            text-align: right;
            text-transform: uppercase;
            border-bottom: 1px solid var(--wc-border);
        }

        .wc-group-table th:first-child,
        .wc-group-table td:first-child {
            text-align: left;
        }

        .wc-group-table td {
            color: var(--wc-text);
            padding: 0.42rem 0.25rem;
            text-align: right;
            border-bottom: 1px solid rgba(148, 163, 184, 0.12);
        }

        .wc-group-table tr:last-child td {
            border-bottom: 0;
        }

        .wc-position {
            color: var(--wc-muted);
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 800;
            width: 1.2rem;
        }

        @media (max-width: 900px) {
            .wc-insight-strip {
                grid-template-columns: 1fr;
            }
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


def get_top_team(simulation_summary, probability_column):
    top_row = simulation_summary.sort_values(probability_column, ascending=False).iloc[0]
    return top_row["team"], top_row[probability_column]


def render_overview_insights(simulation_summary):
    champion, champion_probability = get_top_team(simulation_summary, "champion_probability")
    finalist, final_probability = get_top_team(simulation_summary, "final_probability")
    knockout_team, knockout_probability = get_top_team(
        simulation_summary, "round_of_16_probability"
    )

    st.markdown(
        f"""
        <div class="wc-insight-strip">
            <div class="wc-insight">
                <div class="wc-insight-label">Most Likely Champion</div>
                <div class="wc-insight-value">{team_label(champion)} · {champion_probability:.1%}</div>
            </div>
            <div class="wc-insight">
                <div class="wc-insight-label">Most Likely Finalist</div>
                <div class="wc-insight-value">{team_label(finalist)} · {final_probability:.1%}</div>
            </div>
            <div class="wc-insight">
                <div class="wc-insight-label">Highest Round of 16 Rate</div>
                <div class="wc-insight-value">{team_label(knockout_team)} · {knockout_probability:.1%}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_top_contenders_table(simulation_summary, top_n=10):
    columns = [
        "team",
        "group",
        "round_of_16_probability",
        "quarter_final_probability",
        "semi_final_probability",
        "final_probability",
        "champion_probability",
    ]
    contenders = (
        simulation_summary[columns]
        .sort_values("champion_probability", ascending=False)
        .head(top_n)
        .copy()
    )
    contenders["team"] = contenders["team"].map(team_label)
    return format_probability_columns(
        contenders,
        [
            "round_of_16_probability",
            "quarter_final_probability",
            "semi_final_probability",
            "final_probability",
            "champion_probability",
        ],
    )


def render_overview_caveats():
    st.markdown(
        """
        <div class="wc-caveat-strip">
            <strong>Read this as a forecast, not certainty.</strong>
            The simulation uses outcome-level match results, not scorelines. Group tie-breakers use points,
            wins, Elo, then team name. Knockout rounds may use Elo fallback when exact model probabilities
            are not available for a simulated matchup.
        </div>
        """,
        unsafe_allow_html=True,
    )


def format_percent(value):
    return f"{value:.1%}"


def build_group_qualification_matrix(simulation_summary, selected_group="All"):
    matrix = simulation_summary[
        [
            "group",
            "team",
            "group_winner_probability",
            "group_runner_up_probability",
            "third_place_qualification_probability",
            "round_of_32_probability",
        ]
    ].copy()

    if selected_group != "All":
        matrix = matrix[matrix["group"] == selected_group]

    matrix["team"] = matrix["team"].map(team_label)
    matrix = matrix.rename(
        columns={
            "group_winner_probability": "winner",
            "group_runner_up_probability": "runner_up",
            "third_place_qualification_probability": "third_place",
            "round_of_32_probability": "knockout",
        }
    )
    matrix = matrix.sort_values(
        ["group", "knockout", "winner", "team"],
        ascending=[True, False, False, True],
    )
    return format_probability_columns(
        matrix,
        ["winner", "runner_up", "third_place", "knockout"],
    )


def build_fixture_display_table(fixtures):
    display = add_team_display_columns(fixtures, ["home_team", "away_team"])
    display["match"] = display["home_team_display"] + " vs " + display["away_team_display"]
    display["date"] = display["date"].dt.strftime("%Y-%m-%d %H:%M")
    display = display[
        [
            "date",
            "group",
            "match",
            "predicted_result",
            "highest_probability_outcome",
            "highest_probability",
            "prob_home_win",
            "prob_draw",
            "prob_away_win",
        ]
    ].rename(
        columns={
            "highest_probability_outcome": "top_outcome",
            "highest_probability": "top_probability",
            "prob_home_win": "home_win",
            "prob_draw": "draw",
            "prob_away_win": "away_win",
        }
    )
    for probability_column in ["top_probability", "home_win", "draw", "away_win"]:
        display[probability_column] = display[probability_column] * 100
    return display


def fixture_probability_column_config():
    return {
        "top_probability": st.column_config.ProgressColumn(
            "top_probability",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "home_win": st.column_config.ProgressColumn(
            "home_win",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "draw": st.column_config.ProgressColumn(
            "draw",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
        "away_win": st.column_config.ProgressColumn(
            "away_win",
            format="%.1f%%",
            min_value=0,
            max_value=100,
        ),
    }


def render_group_card(group, group_table, simulation_summary):
    table = group_table.sort_values("group_position").copy()
    probability_lookup = (
        simulation_summary[simulation_summary["group"] == group]
        .set_index("team")["round_of_32_probability"]
        .to_dict()
    )

    rows = []
    for team in table.itertuples(index=False):
        knockout_probability = probability_lookup.get(team.team, 0)
        rows.append(
            "<tr>"
            f"<td><span class='wc-position'>{int(team.group_position)}</span>{html.escape(team_label(team.team))}</td>"
            f"<td>{int(team.played)}</td>"
            f"<td>{int(team.wins)}</td>"
            f"<td>{int(team.draws)}</td>"
            f"<td>{int(team.losses)}</td>"
            f"<td><strong>{int(team.points)}</strong></td>"
            f"<td>{format_percent(knockout_probability)}</td>"
            "</tr>"
        )

    st.markdown(
        f"""
        <div class="wc-group-card">
            <div class="wc-group-card-title">Group {html.escape(str(group))}</div>
            <table class="wc-group-table">
                <thead>
                    <tr>
                        <th>Team</th>
                        <th>P</th>
                        <th>W</th>
                        <th>D</th>
                        <th>L</th>
                        <th>Pts</th>
                        <th>KO</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_group_cards(group_tables, simulation_summary, selected_group):
    groups = sorted(group_tables["group"].dropna().unique())
    if selected_group != "All":
        render_group_card(
            selected_group,
            group_tables[group_tables["group"] == selected_group],
            simulation_summary,
        )
        return

    for index in range(0, len(groups), 3):
        columns = st.columns(3)
        for column, group in zip(columns, groups[index:index + 3]):
            with column:
                render_group_card(
                    group,
                    group_tables[group_tables["group"] == group],
                    simulation_summary,
                )


def render_overview(simulation_summary, metadata):
    render_page_header(
        "World Cup Match Outcome Predictor",
        "Selected model outputs, tournament simulation probabilities, and dashboard-ready metadata.",
    )

    render_overview_insights(simulation_summary)

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

    chart_columns = st.columns(2)
    with chart_columns[0]:
        show_probability_bar_chart(
            simulation_summary,
            "team",
            "champion_probability",
            "Top 10 Champion Probabilities",
            top_n=10,
        )
    with chart_columns[1]:
        show_probability_bar_chart(
            simulation_summary,
            "team",
            "final_probability",
            "Top 10 Final Probabilities",
            top_n=10,
        )

    st.subheader("Top Contenders")
    st.dataframe(
        build_top_contenders_table(simulation_summary),
        use_container_width=True,
        hide_index=True,
    )
    render_overview_caveats()


def render_group_predictions(group_predictions):
    render_page_header(
        "Group Predictions",
        "Fixture-level home, draw, and away probabilities for the confirmed group stage.",
    )
    predictions = add_outcome_summary(group_predictions)

    filter_columns = st.columns([1, 1, 1, 1.4])
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
    with filter_columns[3]:
        search_query = st.text_input(
            "Search fixtures",
            placeholder="Search team, group, or result",
        ).strip().lower()

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

    if search_query:
        searchable = (
            filtered["home_team"].str.lower()
            + " "
            + filtered["away_team"].str.lower()
            + " group "
            + filtered["group"].astype(str).str.lower()
            + " "
            + filtered["predicted_result"].astype(str).str.lower()
            + " "
            + filtered["highest_probability_outcome"].astype(str).str.lower()
        )
        filtered = filtered[searchable.str.contains(search_query, na=False)]

    filtered = filtered.sort_values(["date", "group", "home_team"]).reset_index(drop=True)
    st.caption(f"{len(filtered)} fixtures shown")

    display = build_fixture_display_table(filtered)
    st.dataframe(
        display,
        use_container_width=True,
        hide_index=True,
        column_config=fixture_probability_column_config(),
    )


def render_group_tables(group_tables, simulation_summary):
    render_page_header(
        "Predicted Group Tables",
        "Predicted standings and qualification probabilities for each World Cup group.",
    )
    groups = sorted(group_tables["group"].dropna().unique())
    selected_group = st.selectbox("Group view", ["All", *groups])

    render_group_cards(group_tables, simulation_summary, selected_group)

    st.subheader("Group Qualification Matrix")
    st.caption("Probabilities come from the 10,000 seeded tournament simulations.")
    matrix = build_group_qualification_matrix(simulation_summary, selected_group)
    st.dataframe(matrix, use_container_width=True, hide_index=True)


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
        render_group_tables(group_tables, simulation_summary)
    elif selected_page == "Simulation Results":
        render_simulation_results(simulation_summary)
    elif selected_page == "Knockout Summary":
        render_knockout_summary(match_summary)
    elif selected_page == "Methodology":
        render_methodology(metadata)


if __name__ == "__main__":
    main()
