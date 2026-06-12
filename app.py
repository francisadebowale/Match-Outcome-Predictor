import html
import textwrap
from pathlib import Path

import pandas as pd
import streamlit as st

from src.simulate_world_cup_2026 import (
    build_match_probability_lookup,
    build_round_of_32_matches,
    build_team_elo_lookup,
    convert_to_advancement_probabilities,
    get_match_probabilities,
)


GROUP_PREDICTIONS_PATH = Path("results/predictions/world_cup_2026_group_predictions_v1.csv")
GROUP_TABLES_PATH = Path("results/predictions/world_cup_2026_group_tables_v1.csv")
SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_simulation_summary.csv")
MATCH_SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_match_simulation_summary.csv")
METADATA_PATH = Path("results/simulations/world_cup_2026_metadata.json")
ANNEX_C_PATH = Path("data/raw/world_cup_2026/annex_c_third_place_allocation.csv")
FIXTURES_PATH = Path("data/raw/world_cup_2026/fixtures.csv")

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
    "Team Page",
    "Match Page",
    "Bracket Page",
    "Simulation Results",
    "Knockout Summary",
    "Methodology",
]

KNOCKOUT_ROUNDS = [
    "Round of 32",
    "Round of 16",
    "Quarter Finals",
    "Semi Finals",
    "Finals",
]

BRACKET_MATCH_COUNTS = {
    "Round of 32": 16,
    "Round of 16": 8,
    "Quarter Finals": 4,
    "Semi Finals": 2,
    "Finals": 1,
}


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

        .wc-bracket-summary-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.65rem;
            margin: 0.55rem 0 0.9rem;
        }

        .wc-bracket-summary-item {
            background: linear-gradient(180deg, rgba(17, 24, 39, 0.98), rgba(15, 23, 42, 0.88));
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.7rem 0.75rem;
        }

        .wc-bracket-summary-label {
            color: var(--wc-muted);
            font-size: 0.66rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            margin-bottom: 0.25rem;
            text-transform: uppercase;
        }

        .wc-bracket-summary-value {
            color: var(--wc-text);
            font-size: 0.9rem;
            font-weight: 800;
            line-height: 1.2;
        }

        .wc-bracket-summary-sub {
            color: var(--wc-muted);
            font-size: 0.72rem;
            font-weight: 700;
            margin-top: 0.2rem;
        }

        .wc-bracket-viewport {
            background: linear-gradient(180deg, rgba(8, 13, 24, 0.86), rgba(11, 17, 32, 0.7));
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            margin-top: 0.75rem;
            overflow-x: auto;
            overflow-y: visible;
            padding: 0.35rem;
        }

        .wc-bracket-board {
            display: grid;
            grid-template-columns: repeat(9, 164px);
            gap: 8px;
            min-width: 1510px;
            padding: 0.7rem;
        }

        .wc-bracket-column {
            background: rgba(15, 23, 42, 0.34);
            border: 1px solid rgba(148, 163, 184, 0.12);
            border-radius: 8px;
            min-width: 0;
            padding: 0.45rem;
        }

        .wc-bracket-round-title {
            color: #cbd5e1;
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            margin: 0 0 0.38rem;
        }

        .wc-bracket-card {
            background: rgba(17, 24, 39, 0.98);
            border: 1px solid rgba(148, 163, 184, 0.2);
            border-radius: 8px;
            box-sizing: border-box;
            height: 4.65rem;
            margin-bottom: 0.34rem;
            overflow: hidden;
            padding: 0.42rem 0.5rem;
            position: relative;
            transition: border-color 120ms ease, transform 120ms ease, background 120ms ease;
        }

        .wc-bracket-card:hover {
            background: rgba(22, 32, 51, 0.98);
            border-color: rgba(56, 189, 248, 0.45);
            transform: translateY(-1px);
        }

        .wc-bracket-card::after {
            content: "";
            position: absolute;
            top: 50%;
            width: 0.48rem;
            border-top: 1px solid rgba(148, 163, 184, 0.28);
        }

        .wc-bracket-left .wc-bracket-card::after,
        .wc-bracket-left-middle .wc-bracket-card::after {
            right: -0.48rem;
        }

        .wc-bracket-right .wc-bracket-card::after,
        .wc-bracket-right-middle .wc-bracket-card::after {
            left: -0.48rem;
        }

        .wc-bracket-center .wc-bracket-card::after {
            display: none;
        }

        .wc-bracket-depth-0 .wc-bracket-card {
            margin-bottom: 0.32rem;
        }

        .wc-bracket-depth-1 .wc-bracket-card:first-of-type {
            margin-top: 1.8rem;
        }

        .wc-bracket-depth-1 .wc-bracket-card {
            margin-bottom: 2.25rem;
        }

        .wc-bracket-depth-2 .wc-bracket-card:first-of-type {
            margin-top: 4.85rem;
        }

        .wc-bracket-depth-2 .wc-bracket-card {
            margin-bottom: 5.65rem;
        }

        .wc-bracket-depth-3 .wc-bracket-card:first-of-type {
            margin-top: 10.2rem;
        }

        .wc-bracket-depth-3 .wc-bracket-card {
            margin-bottom: 0.32rem;
        }

        .wc-bracket-center .wc-bracket-card:first-of-type {
            margin-top: 8.25rem;
        }

        .wc-bracket-center .wc-bracket-card {
            border-color: rgba(56, 189, 248, 0.32);
            margin-bottom: 0.5rem;
        }

        .wc-bracket-third-place {
            border-color: rgba(148, 163, 184, 0.22);
            margin-top: 0.65rem;
            opacity: 0.9;
        }

        .wc-bracket-card-meta {
            color: var(--wc-muted);
            font-size: 0.62rem;
            font-weight: 700;
            line-height: 1.2;
            margin-bottom: 0.25rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .wc-bracket-team {
            color: var(--wc-text);
            font-size: 0.8rem;
            font-weight: 700;
            line-height: 1.18;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .wc-bracket-team-advancing {
            color: #67e8f9;
            font-weight: 900;
        }

        .wc-bracket-team-secondary {
            color: var(--wc-muted);
            font-weight: 650;
        }

        .wc-bracket-flag {
            display: inline-block;
            font-size: 1rem;
            line-height: 1;
            margin-right: 0.28rem;
            vertical-align: -0.06rem;
        }

        .wc-bracket-prob {
            color: var(--wc-accent);
            font-size: 0.66rem;
            font-weight: 800;
            margin-top: 0.24rem;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .wc-bracket-advancing-label {
            color: var(--wc-muted);
            font-size: 0.58rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            margin-top: 0.18rem;
            text-transform: uppercase;
        }

        .wc-bracket-prob-row {
            align-items: center;
            display: grid;
            gap: 0.3rem;
            grid-template-columns: minmax(0, 1fr) 2.6rem;
            margin-top: 0.14rem;
        }

        .wc-bracket-bar {
            background: rgba(148, 163, 184, 0.16);
            border-radius: 999px;
            height: 0.25rem;
            overflow: hidden;
        }

        .wc-bracket-bar-fill {
            background: rgba(56, 189, 248, 0.74);
            border-radius: inherit;
            height: 100%;
        }

        .wc-bracket-prob-value {
            color: var(--wc-muted);
            font-size: 0.62rem;
            font-weight: 800;
            text-align: right;
        }

        .wc-bracket-mobile-note {
            color: var(--wc-muted);
            font-size: 0.78rem;
            margin: 0.35rem 0 0.7rem;
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

            .wc-bracket-summary-grid {
                grid-template-columns: 1fr;
            }

            .wc-bracket-board {
                grid-template-columns: repeat(9, 156px);
                min-width: 1440px;
            }

            .wc-bracket-card {
                height: 4.6rem;
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


def team_bracket_html(team):
    flag = country_code_to_flag(TEAM_FLAGS.get(str(team), ""))
    escaped_team = html.escape(str(team))
    if not flag:
        return escaped_team
    return f'<span class="wc-bracket-flag">{html.escape(flag)}</span>{escaped_team}'


def bracket_team_row_html(team, is_advancing=False):
    class_name = "wc-bracket-team wc-bracket-team-advancing" if is_advancing else "wc-bracket-team wc-bracket-team-secondary"
    return f'<div class="{class_name}">{team_bracket_html(team)}</div>'


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


@st.cache_data
def load_annex_allocations():
    if not ANNEX_C_PATH.exists():
        return None
    return pd.read_csv(ANNEX_C_PATH)


@st.cache_data
def load_fixture_skeleton():
    if not FIXTURES_PATH.exists():
        return None
    fixtures = pd.read_csv(FIXTURES_PATH)
    fixtures["Date"] = pd.to_datetime(
        fixtures["Date"],
        format="%d/%m/%Y %H:%M",
        errors="coerce",
    )
    return fixtures


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


def get_team_selection(teams, label="Team", key=None):
    option_map = team_options(teams)
    selected_label = st.selectbox(label, list(option_map.keys()), key=key)
    return option_map[selected_label]


def get_team_summary_row(simulation_summary, team):
    team_rows = simulation_summary[simulation_summary["team"] == team]
    if team_rows.empty:
        return None
    return team_rows.iloc[0]


def build_team_advancement_series(team_row):
    return pd.Series(
        {
            "Group winner": team_row["group_winner_probability"],
            "Runner-up": team_row["group_runner_up_probability"],
            "Third-place qualify": team_row["third_place_qualification_probability"],
            "Round of 32": team_row["round_of_32_probability"],
            "Round of 16": team_row["round_of_16_probability"],
            "Quarter Final": team_row["quarter_final_probability"],
            "Semi Final": team_row["semi_final_probability"],
            "Final": team_row["final_probability"],
            "Champion": team_row["champion_probability"],
        }
    )


def build_team_group_table(group_tables, selected_team):
    team_group_rows = group_tables[group_tables["team"] == selected_team]
    if team_group_rows.empty:
        return pd.DataFrame()

    selected_group = team_group_rows.iloc[0]["group"]
    group_table = group_tables[group_tables["group"] == selected_group].copy()
    group_table = add_team_display_columns(group_table, ["team"])
    group_table["selected"] = group_table["team"].map(
        lambda team: "Selected" if team == selected_team else ""
    )
    display_columns = [
        "group_position",
        "team_display",
        "played",
        "wins",
        "draws",
        "losses",
        "points",
        "selected",
    ]
    available_columns = [column for column in display_columns if column in group_table.columns]
    return (
        group_table[available_columns]
        .sort_values("group_position")
        .rename(columns={"team_display": "team"})
    )


def build_team_fixtures(group_predictions, selected_team):
    predictions = add_outcome_summary(group_predictions)
    fixtures = predictions[
        (predictions["home_team"] == selected_team)
        | (predictions["away_team"] == selected_team)
    ].copy()
    return fixtures.sort_values(["date", "group"])


def build_team_knockout_matchups(match_summary, selected_team, top_n=12):
    if match_summary is None:
        return None

    matchups = match_summary[
        (match_summary["home_team"] == selected_team)
        | (match_summary["away_team"] == selected_team)
    ].copy()
    if matchups.empty:
        return matchups

    matchups["opponent"] = matchups.apply(
        lambda row: row["away_team"] if row["home_team"] == selected_team else row["home_team"],
        axis=1,
    )
    matchups["advancement_probability"] = matchups.apply(
        lambda row: (
            row["home_advancement_probability"]
            if row["home_team"] == selected_team
            else row["away_advancement_probability"]
        ),
        axis=1,
    )
    matchups = add_team_display_columns(matchups, ["home_team", "away_team", "opponent"])
    display = matchups[
        [
            "round",
            "opponent_display",
            "simulations",
            "advancement_probability",
            "home_team_display",
            "away_team_display",
        ]
    ].sort_values("simulations", ascending=False).head(top_n)
    display = format_probability_columns(display, ["advancement_probability"])
    return display.rename(
        columns={
            "opponent_display": "opponent",
            "home_team_display": "home_team",
            "away_team_display": "away_team",
        }
    )


def format_match_datetime(value):
    if pd.isna(value):
        return "Date unavailable"
    return pd.to_datetime(value).strftime("%Y-%m-%d %H:%M")


def get_match_winner_label(predicted_result, home_team, away_team):
    if predicted_result == "home_win":
        return f"{team_label(home_team)} win"
    if predicted_result == "away_win":
        return f"{team_label(away_team)} win"
    return "Draw"


def build_group_match_options(matches):
    options = {}
    for row in matches.itertuples():
        label = (
            f"{format_match_datetime(row.date)} · Group {row.group} · "
            f"{team_label(row.home_team)} vs {team_label(row.away_team)}"
        )
        options[label] = row.Index
    return options


def build_knockout_match_options(matches):
    options = {}
    for row in matches.itertuples():
        label = (
            f"{row.round} · {team_label(row.home_team)} vs {team_label(row.away_team)} · "
            f"{int(row.simulations)} sims"
        )
        options[label] = row.Index
    return options


def get_knockout_fixtures(fixtures):
    if fixtures is None:
        return None
    return fixtures[fixtures["Round Number"].isin(KNOCKOUT_ROUNDS)].copy()


def get_placeholder_label(value):
    if pd.isna(value):
        return "To be announced"
    return str(value)


def render_official_bracket_card(row):
    date_value = row.get("Date", pd.NaT)
    if pd.isna(date_value):
        date_label = "Date unavailable"
    else:
        date_label = pd.to_datetime(date_value, dayfirst=True).strftime("%b %d · %H:%M")

    st.markdown(
        f"""
        <div class="wc-bracket-card">
            <div class="wc-bracket-card-meta">M{int(row['Match Number'])} · {html.escape(date_label)}</div>
            <div class="wc-bracket-team">{html.escape(get_placeholder_label(row['Home Team']))}</div>
            <div class="wc-bracket-team">{html.escape(get_placeholder_label(row['Away Team']))}</div>
            <div class="wc-bracket-card-meta">{html.escape(get_placeholder_label(row['Location']))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_simulated_bracket_card(row):
    home_probability = row["home_advancement_probability"]
    away_probability = row["away_advancement_probability"]
    favored_team = row["home_team"] if home_probability >= away_probability else row["away_team"]
    favored_probability = max(home_probability, away_probability)

    st.markdown(
        f"""
        <div class="wc-bracket-card">
            <div class="wc-bracket-card-meta">{int(row['simulations'])} simulations</div>
            <div class="wc-bracket-team">{html.escape(team_label(row['home_team']))}</div>
            <div class="wc-bracket-team">{html.escape(team_label(row['away_team']))}</div>
            <div class="wc-bracket-prob">{html.escape(team_label(favored_team))} · {format_percent(favored_probability)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_bracket_card_count(row):
    if "simulations" in row and pd.notna(row["simulations"]):
        return f"{int(row['simulations'])} simulations"
    return "Most likely path"


def official_bracket_card_html(row):
    date_value = row.get("Date", pd.NaT)
    if pd.isna(date_value):
        date_label = "Date unavailable"
    else:
        date_label = pd.to_datetime(date_value).strftime("%b %d · %H:%M")

    tooltip = (
        f"Match {int(row['Match Number'])} | {get_placeholder_label(row['Home Team'])} vs "
        f"{get_placeholder_label(row['Away Team'])} | {get_placeholder_label(row['Location'])}"
    )
    extra_class = " wc-bracket-third-place" if row.get("is_third_place") else ""
    card_label = "Third-Place Match" if row.get("is_third_place") else f"M{int(row['Match Number'])} · {html.escape(date_label)}"
    return textwrap.dedent(f"""
        <div class="wc-bracket-card{extra_class}" title="{html.escape(tooltip)}">
            <div class="wc-bracket-card-meta">{card_label}</div>
            <div class="wc-bracket-team">{html.escape(get_placeholder_label(row['Home Team']))}</div>
            <div class="wc-bracket-team">{html.escape(get_placeholder_label(row['Away Team']))}</div>
            <div class="wc-bracket-card-meta">{html.escape(get_placeholder_label(row['Location']))}</div>
        </div>
    """).strip()


def probability_bar_html(probability):
    bounded_probability = max(0, min(1, probability))
    return textwrap.dedent(f"""
        <div class="wc-bracket-prob-row">
            <div class="wc-bracket-bar">
                <div class="wc-bracket-bar-fill" style="width: {bounded_probability * 100:.1f}%"></div>
            </div>
            <div class="wc-bracket-prob-value">{bounded_probability:.0%}</div>
        </div>
    """).strip()


def simulated_bracket_card_html(row):
    home_probability = row["home_advancement_probability"]
    away_probability = row["away_advancement_probability"]
    favored_team = row.get("winner")
    if not favored_team:
        favored_team = row["home_team"] if home_probability >= away_probability else row["away_team"]
    favored_probability = max(home_probability, away_probability)
    card_count = get_bracket_card_count(row)
    extra_class = " wc-bracket-third-place" if row.get("is_third_place") else ""
    meta_label = "Third-Place Match" if row.get("is_third_place") else card_count

    tooltip = (
        f"{row['home_team']} vs {row['away_team']} | {card_count} | "
        f"{row['home_team']}: {format_percent(home_probability)} | "
        f"{row['away_team']}: {format_percent(away_probability)}"
    )
    return textwrap.dedent(f"""
        <div class="wc-bracket-card{extra_class}" title="{html.escape(tooltip)}">
            <div class="wc-bracket-card-meta">{html.escape(meta_label)}</div>
            {bracket_team_row_html(row['home_team'], row['home_team'] == favored_team)}
            {probability_bar_html(home_probability)}
            {bracket_team_row_html(row['away_team'], row['away_team'] == favored_team)}
            {probability_bar_html(away_probability)}
            <div class="wc-bracket-advancing-label">Advancing</div>
            <div class="wc-bracket-prob">{html.escape(team_label(favored_team))} {format_percent(favored_probability)}</div>
        </div>
    """).strip()


def split_bracket_rows(rows):
    midpoint = (len(rows) + 1) // 2
    return rows[:midpoint], rows[midpoint:]


def bracket_round_html(round_name, rows, depth, card_builder):
    cards = "".join(card_builder(row) for row in rows)
    if not cards:
        cards = '<div class="wc-bracket-card"><div class="wc-bracket-card-meta">No matchups</div></div>'
    return textwrap.dedent(f"""
        <div class="wc-bracket-round wc-bracket-depth-{depth}">
            <div class="wc-bracket-round-title">{html.escape(round_name)}</div>
            {cards}
        </div>
    """).strip()


def bracket_side_html(round_rows, side_classes, card_builder, reverse=False):
    display_rows = list(reversed(round_rows)) if reverse else round_rows
    depth_by_round = {round_name: depth for depth, (round_name, _) in enumerate(round_rows)}
    columns = []
    for index, (round_name, rows) in enumerate(display_rows):
        columns.append(
            textwrap.dedent(f"""
                <div class="wc-bracket-column {side_classes[index]}">
                    {bracket_round_html(round_name, rows, depth_by_round[round_name], card_builder)}
                </div>
            """).strip()
        )
    return "".join(columns)


def bracket_center_html(finals, card_builder, third_place=None):
    cards = "".join(card_builder(row) for row in finals)
    if not cards:
        cards = '<div class="wc-bracket-card"><div class="wc-bracket-card-meta">No final matchups</div></div>'
    third_place_html = card_builder(third_place) if third_place is not None else ""
    return textwrap.dedent(f"""
        <div class="wc-bracket-column wc-bracket-center">
            <div class="wc-bracket-round-title">Finals</div>
            {cards}
            {third_place_html}
        </div>
    """).strip()


def build_most_likely_qualifiers(group_tables, simulation_summary):
    group_winners = {}
    group_runners_up = {}
    third_place_candidates = []
    probability_lookup = simulation_summary.set_index("team").to_dict("index")

    for group, group_table in group_tables.groupby("group"):
        ordered = group_table.sort_values("group_position").reset_index(drop=True)
        group_winners[group] = ordered.loc[0, "team"]
        group_runners_up[group] = ordered.loc[1, "team"]
        third_team = ordered.loc[2, "team"]
        third_summary = probability_lookup.get(third_team, {})
        third_place_candidates.append(
            {
                "group": group,
                "team": third_team,
                "third_place_qualification_probability": third_summary.get(
                    "third_place_qualification_probability",
                    0,
                ),
                "round_of_32_probability": third_summary.get("round_of_32_probability", 0),
            }
        )

    best_thirds = sorted(
        third_place_candidates,
        key=lambda row: (
            -row["third_place_qualification_probability"],
            -row["round_of_32_probability"],
            row["team"],
        ),
    )[:8]
    third_place_by_group = {row["group"]: row["team"] for row in best_thirds}

    return {
        "group_winners": group_winners,
        "group_runners_up": group_runners_up,
        "third_place_by_group": third_place_by_group,
        "round_of_32_teams": (
            list(group_winners.values())
            + list(group_runners_up.values())
            + list(third_place_by_group.values())
        ),
    }


def get_annex_row_for_qualifiers(qualifiers, annex_allocations):
    third_place_key = "".join(sorted(qualifiers["third_place_by_group"]))
    annex_match = annex_allocations[annex_allocations["qualified_groups"] == third_place_key]
    if annex_match.empty:
        raise ValueError(f"No Annex C row found for third-place groups: {third_place_key}")
    return annex_match.iloc[0].to_dict()


def deterministic_knockout_round(round_name, matches, team_elos, match_probability_lookup):
    winners = []
    records = []
    for match_index, match in enumerate(matches, start=1):
        home_team = match["home_team"]
        away_team = match["away_team"]
        home_prob, draw_prob, away_prob = get_match_probabilities(
            home_team,
            away_team,
            team_elos,
            match_probability_lookup,
        )
        home_adv_prob, away_adv_prob = convert_to_advancement_probabilities(
            home_prob,
            draw_prob,
            away_prob,
        )
        winner = home_team if home_adv_prob >= away_adv_prob else away_team
        winners.append(winner)
        records.append(
            {
                "round": round_name,
                "match_id": match.get("match_id", match_index),
                "home_team": home_team,
                "away_team": away_team,
                "winner": winner,
                "home_probability": home_prob,
                "draw_probability": draw_prob,
                "away_probability": away_prob,
                "home_advancement_probability": home_adv_prob,
                "away_advancement_probability": away_adv_prob,
            }
        )
    return winners, records


def build_third_place_match(semi_final_records, team_elos, match_probability_lookup):
    if len(semi_final_records) < 2:
        return None

    semi_losers = []
    for record in semi_final_records[:2]:
        loser = record["away_team"] if record["winner"] == record["home_team"] else record["home_team"]
        semi_losers.append(loser)

    home_team, away_team = semi_losers
    home_prob, draw_prob, away_prob = get_match_probabilities(
        home_team,
        away_team,
        team_elos,
        match_probability_lookup,
    )
    home_adv_prob, away_adv_prob = convert_to_advancement_probabilities(
        home_prob,
        draw_prob,
        away_prob,
    )
    winner = home_team if home_adv_prob >= away_adv_prob else away_team
    return {
        "round": "Third-Place Match",
        "match_id": 0,
        "home_team": home_team,
        "away_team": away_team,
        "winner": winner,
        "home_probability": home_prob,
        "draw_probability": draw_prob,
        "away_probability": away_prob,
        "home_advancement_probability": home_adv_prob,
        "away_advancement_probability": away_adv_prob,
        "is_third_place": True,
    }


def ordered_knockout_matches(round_name, teams):
    return [
        {
            "round": round_name,
            "match_id": index // 2 + 1,
            "home_team": teams[index],
            "away_team": teams[index + 1],
        }
        for index in range(0, len(teams), 2)
    ]


def build_most_likely_tournament_path(
    group_tables,
    group_predictions,
    simulation_summary,
    annex_allocations,
    fixtures,
):
    if annex_allocations is None or fixtures is None:
        return None

    qualifiers = build_most_likely_qualifiers(group_tables, simulation_summary)
    annex_row = get_annex_row_for_qualifiers(qualifiers, annex_allocations)
    team_elos = build_team_elo_lookup(group_predictions)
    match_probability_lookup = build_match_probability_lookup(group_predictions)

    round_of_32_matches = build_round_of_32_matches(
        qualifiers=qualifiers,
        annex_row=annex_row,
        fixtures=fixtures,
    )
    round_of_32_winners, round_of_32_records = deterministic_knockout_round(
        "Round of 32",
        round_of_32_matches,
        team_elos,
        match_probability_lookup,
    )

    round_of_16_winners, round_of_16_records = deterministic_knockout_round(
        "Round of 16",
        ordered_knockout_matches("Round of 16", round_of_32_winners),
        team_elos,
        match_probability_lookup,
    )
    quarter_final_winners, quarter_final_records = deterministic_knockout_round(
        "Quarter Finals",
        ordered_knockout_matches("Quarter Finals", round_of_16_winners),
        team_elos,
        match_probability_lookup,
    )
    semi_final_winners, semi_final_records = deterministic_knockout_round(
        "Semi Finals",
        ordered_knockout_matches("Semi Finals", quarter_final_winners),
        team_elos,
        match_probability_lookup,
    )
    _, final_records = deterministic_knockout_round(
        "Finals",
        ordered_knockout_matches("Finals", semi_final_winners),
        team_elos,
        match_probability_lookup,
    )
    third_place_record = build_third_place_match(
        semi_final_records,
        team_elos,
        match_probability_lookup,
    )

    records = (
        round_of_32_records
        + round_of_16_records
        + quarter_final_records
        + semi_final_records
        + final_records
    )
    if third_place_record is not None:
        records.append(third_place_record)
    return records


def split_records_for_bracket(records):
    records_by_round = {
        round_name: [
            record
            for record in records
            if record["round"] == round_name
        ]
        for round_name in KNOCKOUT_ROUNDS
    }
    left_round_rows = []
    right_round_rows = []
    for round_name in KNOCKOUT_ROUNDS[:4]:
        left_rows, right_rows = split_bracket_rows(records_by_round[round_name])
        left_round_rows.append((round_name, left_rows))
        right_round_rows.append((round_name, right_rows))
    third_place_records = [
        record
        for record in records
        if record["round"] == "Third-Place Match"
    ]
    third_place = third_place_records[0] if third_place_records else None
    return left_round_rows, records_by_round["Finals"], right_round_rows, third_place


def render_two_sided_bracket(left_round_rows, finals, right_round_rows, card_builder, third_place=None):
    bracket_html = textwrap.dedent(f"""
        <div class="wc-bracket-mobile-note">Bracket scrolls horizontally on smaller screens.</div>
        <div class="wc-bracket-viewport">
            <div class="wc-bracket-board">
                {bracket_side_html(left_round_rows, ["wc-bracket-left", "wc-bracket-left", "wc-bracket-left-middle", "wc-bracket-left-middle"], card_builder)}
                {bracket_center_html(finals, card_builder, third_place)}
                {bracket_side_html(right_round_rows, ["wc-bracket-right-middle", "wc-bracket-right-middle", "wc-bracket-right", "wc-bracket-right"], card_builder, reverse=True)}
            </div>
        </div>
    """).strip()
    st.markdown(bracket_html, unsafe_allow_html=True)


def render_official_bracket_grid(fixtures):
    knockout_fixtures = get_knockout_fixtures(fixtures)
    if knockout_fixtures is None:
        st.warning(f"Optional fixture skeleton missing: {FIXTURES_PATH}.")
        return

    left_round_rows = []
    right_round_rows = []
    for round_name in KNOCKOUT_ROUNDS[:4]:
        round_rows = [
            row.to_dict()
            for _, row in knockout_fixtures[
                knockout_fixtures["Round Number"] == round_name
            ].sort_values("Match Number").iterrows()
        ]
        left_rows, right_rows = split_bracket_rows(round_rows)
        left_round_rows.append((round_name, left_rows))
        right_round_rows.append((round_name, right_rows))

    finals = [
        row.to_dict()
        for _, row in knockout_fixtures[
            knockout_fixtures["Round Number"] == "Finals"
        ].sort_values("Match Number").iterrows()
    ]
    final_rows = []
    third_place = None
    for row in finals:
        if int(row["Match Number"]) == 103:
            row["is_third_place"] = True
            row["Home Team"] = "Loser SF1"
            row["Away Team"] = "Loser SF2"
            third_place = row
        else:
            final_rows.append(row)

    render_two_sided_bracket(
        left_round_rows,
        final_rows,
        right_round_rows,
        official_bracket_card_html,
        third_place=third_place,
    )


def render_simulated_bracket_grid(
    group_tables,
    group_predictions,
    simulation_summary,
    annex_allocations,
    fixtures,
):
    path_records = build_most_likely_tournament_path(
        group_tables=group_tables,
        group_predictions=group_predictions,
        simulation_summary=simulation_summary,
        annex_allocations=annex_allocations,
        fixtures=fixtures,
    )
    if path_records is None:
        st.warning(
            f"Required bracket inputs missing. Check {ANNEX_C_PATH} and {FIXTURES_PATH}."
        )
        return

    left_round_rows, finals, right_round_rows, third_place = split_records_for_bracket(path_records)
    render_two_sided_bracket(
        left_round_rows,
        finals,
        right_round_rows,
        simulated_bracket_card_html,
        third_place=third_place,
    )


def build_bracket_round_summary(fixtures, match_summary):
    rows = []
    knockout_fixtures = get_knockout_fixtures(fixtures)
    for round_name in KNOCKOUT_ROUNDS:
        fixture_count = 0
        if knockout_fixtures is not None:
            fixture_count = len(knockout_fixtures[knockout_fixtures["Round Number"] == round_name])

        simulated_matchup_count = 0
        most_common_matchup = "Unavailable"
        if match_summary is not None:
            round_matchups = match_summary[match_summary["round"] == round_name]
            simulated_matchup_count = len(round_matchups)
            if not round_matchups.empty:
                top_matchup = round_matchups.sort_values("simulations", ascending=False).iloc[0]
                most_common_matchup = (
                    f"{team_label(top_matchup['home_team'])} vs "
                    f"{team_label(top_matchup['away_team'])}"
                )

        rows.append(
            {
                "round": round_name,
                "official_fixture_slots": fixture_count,
                "simulated_matchups": simulated_matchup_count,
                "most_common_matchup": most_common_matchup,
            }
        )
    return pd.DataFrame(rows)


def get_probability_row(simulation_summary, sort_column, filtered=None):
    data = simulation_summary if filtered is None else filtered
    if data.empty:
        return None
    return data.sort_values(sort_column, ascending=False).iloc[0]


def build_bracket_summary_items(simulation_summary, match_summary):
    champion = get_probability_row(simulation_summary, "champion_probability")
    final_match = None
    if match_summary is not None:
        final_rows = match_summary[match_summary["round"] == "Finals"]
        if not final_rows.empty:
            final_match = final_rows.sort_values("simulations", ascending=False).iloc[0]

    dark_horse_pool = simulation_summary[simulation_summary["group_winner_probability"] < 0.4]
    dark_horse = get_probability_row(simulation_summary, "champion_probability", dark_horse_pool)

    upset_pool = simulation_summary[simulation_summary["round_of_32_probability"] < 0.85]
    upset_candidate = get_probability_row(simulation_summary, "round_of_16_probability", upset_pool)

    items = []
    if champion is not None:
        items.append(
            (
                "Most Likely Champion",
                team_label(champion["team"]),
                format_percent(champion["champion_probability"]),
            )
        )
    if final_match is not None:
        items.append(
            (
                "Most Likely Final",
                f"{team_label(final_match['home_team'])} vs {team_label(final_match['away_team'])}",
                f"{int(final_match['simulations'])} simulations",
            )
        )
    if dark_horse is not None:
        items.append(
            (
                "Dark Horse",
                team_label(dark_horse["team"]),
                format_percent(dark_horse["champion_probability"]),
            )
        )
    if upset_candidate is not None:
        items.append(
            (
                "Upset Candidate",
                team_label(upset_candidate["team"]),
                f"{format_percent(upset_candidate['round_of_16_probability'])} R16",
            )
        )
    return items


def render_bracket_summary(simulation_summary, match_summary):
    items = build_bracket_summary_items(simulation_summary, match_summary)
    if not items:
        return
    item_html = "".join(
        textwrap.dedent(f"""
            <div class="wc-bracket-summary-item">
                <div class="wc-bracket-summary-label">{html.escape(label)}</div>
                <div class="wc-bracket-summary-value">{html.escape(value)}</div>
                <div class="wc-bracket-summary-sub">{html.escape(sub_value)}</div>
            </div>
        """).strip()
        for label, value, sub_value in items
    )
    st.markdown(
        f'<div class="wc-bracket-summary-grid">{item_html}</div>',
        unsafe_allow_html=True,
    )


def filter_group_matches(group_predictions):
    matches = add_outcome_summary(group_predictions)
    filter_columns = st.columns([1, 1, 1.4])
    with filter_columns[0]:
        groups = sorted(matches["group"].dropna().unique())
        selected_group = st.selectbox("Group", ["All", *groups], key="match_page_group")
    with filter_columns[1]:
        teams = sorted(set(matches["home_team"]).union(set(matches["away_team"])))
        option_map = team_options(teams)
        selected_team_label = st.selectbox("Team", ["All", *option_map.keys()], key="match_page_team")
        selected_team = option_map.get(selected_team_label, selected_team_label)
    with filter_columns[2]:
        search_query = st.text_input(
            "Search match",
            placeholder="Search team, venue, or result",
            key="match_page_search",
        ).strip().lower()

    if selected_group != "All":
        matches = matches[matches["group"] == selected_group]
    if selected_team != "All":
        matches = matches[
            (matches["home_team"] == selected_team)
            | (matches["away_team"] == selected_team)
        ]
    if search_query:
        searchable = (
            matches["home_team"].str.lower()
            + " "
            + matches["away_team"].str.lower()
            + " "
            + matches["location"].astype(str).str.lower()
            + " "
            + matches["predicted_result"].astype(str).str.lower()
            + " "
            + matches["home_team_result"].astype(str).str.lower()
        )
        matches = matches[searchable.str.contains(search_query, na=False)]

    return matches.sort_values(["date", "group", "home_team"])


def render_group_match_detail(group_predictions):
    matches = filter_group_matches(group_predictions)
    if matches.empty:
        st.info("No group-stage fixture matches the current filters.")
        return

    options = build_group_match_options(matches)
    selected_label = st.selectbox("Select fixture", list(options.keys()), key="group_match_select")
    match = matches.loc[options[selected_label]]

    st.subheader(f"{team_label(match['home_team'])} vs {team_label(match['away_team'])}")
    detail_columns = st.columns(5)
    detail_columns[0].metric("Group", match.get("group", "Unknown"))
    detail_columns[1].metric("Date", format_match_datetime(match.get("date", pd.NaT)))
    detail_columns[2].metric("Venue", match.get("location", "Unknown"))
    detail_columns[3].metric(
        "Prediction",
        get_match_winner_label(match["predicted_result"], match["home_team"], match["away_team"]),
    )
    detail_columns[4].metric("Elo diff", int(match.get("elo_difference", 0)))

    probability_chart = pd.Series(
        {
            f"{team_label(match['home_team'])} win": match["prob_home_win"],
            "Draw": match["prob_draw"],
            f"{team_label(match['away_team'])} win": match["prob_away_win"],
        }
    )
    st.subheader("Outcome Probabilities")
    st.bar_chart(probability_chart)

    fixture_display = build_fixture_display_table(pd.DataFrame([match]))
    st.dataframe(
        fixture_display,
        use_container_width=True,
        hide_index=True,
        column_config=fixture_probability_column_config(),
    )


def filter_knockout_matches(match_summary):
    matches = match_summary.copy()
    filter_columns = st.columns([1, 1, 1.4])
    with filter_columns[0]:
        rounds = sorted(matches["round"].dropna().unique())
        selected_round = st.selectbox("Round", ["All", *rounds], key="match_page_round")
    with filter_columns[1]:
        teams = sorted(set(matches["home_team"]).union(set(matches["away_team"])))
        option_map = team_options(teams)
        selected_team_label = st.selectbox(
            "Team",
            ["All", *option_map.keys()],
            key="match_page_knockout_team",
        )
        selected_team = option_map.get(selected_team_label, selected_team_label)
    with filter_columns[2]:
        search_query = st.text_input(
            "Search matchup",
            placeholder="Search team or round",
            key="match_page_knockout_search",
        ).strip().lower()

    if selected_round != "All":
        matches = matches[matches["round"] == selected_round]
    if selected_team != "All":
        matches = matches[
            (matches["home_team"] == selected_team)
            | (matches["away_team"] == selected_team)
        ]
    if search_query:
        searchable = (
            matches["home_team"].str.lower()
            + " "
            + matches["away_team"].str.lower()
            + " "
            + matches["round"].astype(str).str.lower()
        )
        matches = matches[searchable.str.contains(search_query, na=False)]

    return matches.sort_values("simulations", ascending=False)


def render_knockout_match_detail(match_summary):
    if match_summary is None:
        st.warning(
            f"Optional file missing: {MATCH_SIMULATION_SUMMARY_PATH}. "
            "Run `python3 src/simulate_world_cup_2026.py` to create it."
        )
        return

    matches = filter_knockout_matches(match_summary)
    if matches.empty:
        st.info("No simulated knockout matchup matches the current filters.")
        return

    options = build_knockout_match_options(matches)
    selected_label = st.selectbox("Select matchup", list(options.keys()), key="knockout_match_select")
    match = matches.loc[options[selected_label]]

    st.subheader(f"{team_label(match['home_team'])} vs {team_label(match['away_team'])}")
    detail_columns = st.columns(5)
    detail_columns[0].metric("Round", match["round"])
    detail_columns[1].metric("Simulations", int(match["simulations"]))
    detail_columns[2].metric(
        f"{match['home_team']} advance",
        format_percent(match["home_advancement_probability"]),
    )
    detail_columns[3].metric(
        f"{match['away_team']} advance",
        format_percent(match["away_advancement_probability"]),
    )
    detail_columns[4].metric("Draw handling", "Redistributed")

    advancement_chart = pd.Series(
        {
            f"{team_label(match['home_team'])} advance": match["home_advancement_probability"],
            f"{team_label(match['away_team'])} advance": match["away_advancement_probability"],
        }
    )
    st.subheader("Advancement Probabilities")
    st.bar_chart(advancement_chart)

    probability_columns = [
        "home_probability",
        "draw_probability",
        "away_probability",
        "home_advancement_probability",
        "away_advancement_probability",
    ]
    display = add_team_display_columns(pd.DataFrame([match]), ["home_team", "away_team"])
    display = format_probability_columns(
        display[
            [
                "round",
                "home_team_display",
                "away_team_display",
                "simulations",
                *probability_columns,
            ]
        ],
        probability_columns,
    ).rename(
        columns={
            "home_team_display": "home_team",
            "away_team_display": "away_team",
        }
    )
    st.dataframe(display, use_container_width=True, hide_index=True)


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


def render_team_page(group_predictions, group_tables, simulation_summary, match_summary):
    render_page_header(
        "Team Page",
        "Team-level fixtures, group position, advancement probabilities, and likely knockout matchups.",
    )

    selected_team = get_team_selection(
        simulation_summary["team"].dropna().unique(),
        label="Select team",
        key="team_page_selector",
    )
    team_row = get_team_summary_row(simulation_summary, selected_team)
    if team_row is None:
        st.error(f"No simulation summary found for {selected_team}.")
        return

    st.subheader(team_label(selected_team))
    metric_columns = st.columns(6)
    metric_columns[0].metric("Group", team_row.get("group", "Unknown"))
    metric_columns[1].metric("Round of 32", format_percent(team_row["round_of_32_probability"]))
    metric_columns[2].metric("Round of 16", format_percent(team_row["round_of_16_probability"]))
    metric_columns[3].metric("Quarter Final", format_percent(team_row["quarter_final_probability"]))
    metric_columns[4].metric("Final", format_percent(team_row["final_probability"]))
    metric_columns[5].metric("Champion", format_percent(team_row["champion_probability"]))

    chart_columns = st.columns([1.15, 1])
    with chart_columns[0]:
        st.subheader("Advancement Profile")
        st.bar_chart(build_team_advancement_series(team_row))
    with chart_columns[1]:
        st.subheader("Predicted Group Table")
        group_table = build_team_group_table(group_tables, selected_team)
        if group_table.empty:
            st.info("No group table row found for this team.")
        else:
            st.dataframe(group_table, use_container_width=True, hide_index=True)

    st.subheader("Group Fixtures")
    fixtures = build_team_fixtures(group_predictions, selected_team)
    if fixtures.empty:
        st.info("No group-stage fixtures found for this team.")
    else:
        st.dataframe(
            build_fixture_display_table(fixtures),
            use_container_width=True,
            hide_index=True,
            column_config=fixture_probability_column_config(),
        )

    st.subheader("Likely Knockout Matchups")
    matchup_display = build_team_knockout_matchups(match_summary, selected_team)
    if matchup_display is None:
        st.warning(
            f"Optional file missing: {MATCH_SIMULATION_SUMMARY_PATH}. "
            "Run `python3 src/simulate_world_cup_2026.py` to create it."
        )
    elif matchup_display.empty:
        st.info("No simulated knockout matchups found for this team.")
    else:
        st.caption("Sorted by how often the matchup appeared across simulations.")
        st.dataframe(matchup_display, use_container_width=True, hide_index=True)


def render_match_page(group_predictions, match_summary):
    render_page_header(
        "Match Page",
        "Inspect confirmed group fixtures and common simulated knockout matchups one match at a time.",
    )

    group_tab, knockout_tab = st.tabs(["Group-stage fixtures", "Simulated knockout matchups"])
    with group_tab:
        render_group_match_detail(group_predictions)
    with knockout_tab:
        st.caption("Knockout teams are simulation-derived, not confirmed fixtures.")
        render_knockout_match_detail(match_summary)


def render_bracket_page(
    fixtures,
    match_summary,
    simulation_summary,
    group_tables,
    group_predictions,
    annex_allocations,
):
    render_page_header(
        "Bracket Page",
        "Official knockout fixture slots and the most common simulated matchups by round.",
    )

    render_bracket_summary(simulation_summary, match_summary)

    summary = build_bracket_round_summary(fixtures, match_summary)
    st.subheader("Bracket Round Summary")
    st.dataframe(summary, use_container_width=True, hide_index=True)

    official_tab, simulated_tab = st.tabs(["Official fixture skeleton", "Simulation matchup view"])
    with official_tab:
        st.caption("Round of 32 placeholders are official fixture slots; later rounds remain to be announced.")
        render_official_bracket_grid(fixtures)
    with simulated_tab:
        st.caption("Single coherent most-likely path built from projected group finishers and deterministic knockout advancement.")
        render_simulated_bracket_grid(
            group_tables,
            group_predictions,
            simulation_summary,
            annex_allocations,
            fixtures,
        )


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
    fixtures = load_fixture_skeleton()
    annex_allocations = load_annex_allocations()
    metadata = load_metadata()
    selected_page = render_sidebar_nav()

    if selected_page == "Overview":
        render_overview(simulation_summary, metadata)
    elif selected_page == "Group Predictions":
        render_group_predictions(group_predictions)
    elif selected_page == "Group Tables":
        render_group_tables(group_tables, simulation_summary)
    elif selected_page == "Team Page":
        render_team_page(group_predictions, group_tables, simulation_summary, match_summary)
    elif selected_page == "Match Page":
        render_match_page(group_predictions, match_summary)
    elif selected_page == "Bracket Page":
        render_bracket_page(
            fixtures,
            match_summary,
            simulation_summary,
            group_tables,
            group_predictions,
            annex_allocations,
        )
    elif selected_page == "Simulation Results":
        render_simulation_results(simulation_summary)
    elif selected_page == "Knockout Summary":
        render_knockout_summary(match_summary)
    elif selected_page == "Methodology":
        render_methodology(metadata)


if __name__ == "__main__":
    main()
