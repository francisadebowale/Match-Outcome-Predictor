import html
import textwrap
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st

from src.simulate_world_cup_2026 import (
    build_match_probability_lookup,
    build_round_of_32_matches,
    build_team_elo_lookup,
    convert_to_advancement_probabilities,
    get_match_probabilities,
    run_single_simulation,
)


GROUP_PREDICTIONS_PATH = Path("results/predictions/world_cup_2026_group_predictions_v1.csv")
GROUP_TABLES_PATH = Path("results/predictions/world_cup_2026_group_tables_v1.csv")
SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_simulation_summary.csv")
MATCH_SIMULATION_SUMMARY_PATH = Path("results/simulations/world_cup_2026_match_simulation_summary.csv")
CHAMPION_HISTORY_PATH = Path("results/simulations/world_cup_2026_champion_probability_history.csv")
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


NAV_GROUPS = {
    "Overview": ["Overview"],
    "Tournament": [
        "Group Predictions",
        "Group Tables",
        "Bracket Page",
        "Knockout Summary",
        "Simulation Explorer",
    ],
    "Analysis": [
        "Team Page",
        "Team Comparison",
        "Match Page",
        "Tournament Storylines",
    ],
    "Simulations": [
        "Simulation Results",
        "Champion History",
    ],
    "Tools": [
        "Scenario Simulator",
        "Tournament Builder",
    ],
    "About": ["Methodology"],
}

NAV_ITEMS = [
    page
    for pages in NAV_GROUPS.values()
    for page in pages
]

NAV_PAGE_LABELS = {
    "Bracket Page": "Bracket",
    "Team Page": "Team",
    "Match Page": "Match",
}

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
        :root,
        [data-theme="light"],
        [data-baseweb-theme="light"] {
            --wc-bg: #ffffff;
            --wc-panel: #f8fafc;
            --wc-panel-soft: color-mix(in srgb, var(--wc-panel) 82%, var(--wc-bg));
            --wc-border: color-mix(in srgb, var(--wc-text) 16%, transparent);
            --wc-text: #0f172a;
            --wc-muted: color-mix(in srgb, var(--wc-text) 60%, var(--wc-bg));
            --wc-accent: var(--primary-color, #ef4444);
            --wc-accent-strong: #22c55e;
            --wc-panel-overlay: color-mix(in srgb, var(--wc-panel) 78%, transparent);
            --wc-card-bg: color-mix(in srgb, var(--wc-panel) 94%, var(--wc-bg));
            --wc-card-hover: color-mix(in srgb, var(--wc-panel) 86%, var(--wc-text));
            --wc-highlight-bg: color-mix(in srgb, var(--wc-accent) 16%, var(--wc-panel));
            --wc-chart-grid: color-mix(in srgb, var(--wc-text) 16%, transparent);
            --wc-bracket-line: color-mix(in srgb, var(--wc-text) 24%, transparent);
        }

        [data-theme="dark"],
        [data-baseweb-theme="dark"] {
            --wc-bg: #0b1120;
            --wc-panel: #111827;
            --wc-text: #e5e7eb;
        }

        @media (prefers-color-scheme: dark) {
            :root {
                --wc-bg: #0b1120;
                --wc-panel: #111827;
                --wc-text: #e5e7eb;
            }
        }

        html,
        body,
        #root,
        [data-testid="stAppViewContainer"],
        [data-testid="stMain"],
        [data-testid="stMainBlockContainer"],
        section.main,
        div.main {
            background: var(--wc-bg) !important;
            color: var(--wc-text) !important;
        }

        .stApp {
            background: var(--wc-bg) !important;
            color: var(--wc-text) !important;
        }

        [data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"] {
            background: var(--wc-bg) !important;
            color: var(--wc-text) !important;
        }

        [data-testid="stDecoration"] {
            display: none;
        }

        .block-container {
            padding-top: 1rem;
            padding-bottom: 3rem;
            max-width: 1420px;
        }

        h1, h2, h3 {
            color: var(--wc-text) !important;
            letter-spacing: 0;
        }

        p,
        label,
        span,
        div {
            color: inherit;
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
            margin-bottom: 0.1rem;
        }

        .wc-page-title {
            font-size: 1.9rem;
            font-weight: 750;
            line-height: 1.15;
            color: var(--wc-text);
            margin-bottom: 0.2rem;
        }

        .wc-page-caption {
            color: var(--wc-muted);
            font-size: 0.95rem;
            margin-bottom: 0.95rem;
        }

        .wc-top-nav-shell {
            background: var(--wc-bg);
            border-bottom: 1px solid var(--wc-border);
            margin: -0.05rem 0 0.95rem;
            padding: 0.1rem 0 0.72rem;
            position: sticky;
            top: 0;
            z-index: 50;
        }

        .wc-top-nav-title {
            color: var(--wc-text);
            font-size: 1.2rem;
            font-weight: 800;
            line-height: 1.2;
            margin-bottom: 0.48rem;
        }

        .wc-top-nav-bar {
            align-items: center;
            display: flex;
            flex-wrap: wrap;
            gap: 0.3rem;
            line-height: 1;
        }

        .wc-top-nav-separator {
            color: var(--wc-border);
            font-weight: 700;
            padding: 0 0.04rem;
        }

        .wc-top-nav-item {
            display: inline-flex;
            position: relative;
        }

        .wc-top-nav-link {
            border: 1.5px solid transparent;
            border-radius: 8px;
            color: var(--wc-muted) !important;
            display: inline-flex;
            font-size: 1rem;
            font-weight: 750;
            gap: 0.28rem;
            line-height: 1;
            padding: 0.62rem 0.88rem;
            text-decoration: none !important;
            white-space: nowrap;
        }

        .wc-top-nav-link:hover {
            background: color-mix(in srgb, var(--wc-text) 8%, transparent);
            border-color: color-mix(in srgb, var(--wc-text) 20%, transparent);
            color: var(--wc-text) !important;
        }

        .wc-top-nav-link-active {
            background: color-mix(in srgb, var(--wc-accent) 18%, var(--wc-panel));
            border-color: color-mix(in srgb, var(--wc-accent) 68%, var(--wc-border));
            box-shadow: inset 0 -2px 0 var(--wc-accent);
            color: var(--wc-accent) !important;
            font-weight: 850;
        }

        .wc-top-nav-caret {
            color: inherit;
            font-size: 0.72rem;
            transform: translateY(-0.03rem);
        }

        .wc-top-nav-dropdown {
            background: var(--wc-panel);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            box-shadow: 0 16px 32px color-mix(in srgb, #000 18%, transparent);
            display: none;
            left: 0;
            min-width: 13rem;
            padding: 0.42rem;
            position: absolute;
            top: 100%;
            z-index: 100;
        }

        .wc-top-nav-dropdown::before {
            content: "";
            height: 0.3rem;
            left: 0;
            position: absolute;
            right: 0;
            top: -0.3rem;
        }

        .wc-top-nav-item:hover .wc-top-nav-dropdown,
        .wc-top-nav-item:focus-within .wc-top-nav-dropdown {
            display: block;
        }

        .wc-top-nav-dropdown-link {
            border-radius: 6px;
            color: var(--wc-muted) !important;
            display: block;
            font-size: 0.92rem;
            font-weight: 750;
            padding: 0.58rem 0.68rem;
            text-decoration: none !important;
            white-space: nowrap;
        }

        .wc-top-nav-dropdown-link:hover,
        .wc-top-nav-dropdown-link-active {
            background: var(--wc-panel-soft);
            color: var(--wc-text) !important;
        }

        .wc-top-nav-dropdown-link-active {
            color: var(--wc-accent) !important;
            font-weight: 850;
        }

        @media (max-width: 760px) {
            .wc-top-nav-shell {
                position: static;
            }

            .wc-top-nav-bar,
            .wc-top-nav-subbar {
                flex-wrap: nowrap;
                overflow-x: auto;
                padding-bottom: 0.1rem;
            }

            .wc-top-nav-dropdown {
                left: 0;
                min-width: 12rem;
            }
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
            border-bottom: 1px solid var(--wc-border);
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
            background: var(--wc-card-bg);
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
            background: var(--wc-panel-soft);
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
            background: var(--wc-panel-overlay);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            min-width: 0;
            padding: 0.45rem;
        }

        .wc-bracket-round-title {
            color: var(--wc-muted);
            font-size: 0.7rem;
            font-weight: 800;
            letter-spacing: 0.055em;
            text-transform: uppercase;
            margin: 0 0 0.38rem;
        }

        .wc-bracket-card {
            background: var(--wc-card-bg);
            border: 1px solid var(--wc-border);
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
            background: var(--wc-card-hover);
            border-color: var(--wc-accent);
            transform: translateY(-1px);
        }

        .wc-bracket-card-highlight {
            background: var(--wc-highlight-bg);
            border-color: var(--wc-accent);
            box-shadow: 0 0 0 1px color-mix(in srgb, var(--wc-accent) 22%, transparent);
        }

        .wc-bracket-card-eliminated {
            border-color: color-mix(in srgb, #f97316 60%, var(--wc-border));
        }

        .wc-bracket-card::after {
            content: "";
            position: absolute;
            top: 50%;
            width: 0.48rem;
            border-top: 1px solid var(--wc-bracket-line);
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
            border-color: color-mix(in srgb, var(--wc-accent) 38%, var(--wc-border));
            margin-bottom: 0.5rem;
        }

        .wc-bracket-third-place {
            border-color: var(--wc-border);
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
            color: var(--wc-accent);
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

        .wc-bracket-path-note {
            color: var(--wc-muted);
            font-size: 0.78rem;
            font-weight: 700;
            margin: 0.35rem 0 0.15rem;
        }

        .wc-bracket-prob-row {
            align-items: center;
            display: grid;
            gap: 0.3rem;
            grid-template-columns: minmax(0, 1fr) 2.6rem;
            margin-top: 0.14rem;
        }

        .wc-bracket-bar {
            background: color-mix(in srgb, var(--wc-text) 14%, transparent);
            border-radius: 999px;
            height: 0.25rem;
            overflow: hidden;
        }

        .wc-bracket-bar-fill {
            background: var(--wc-accent);
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

        .wc-team-list-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.65rem 0 1rem;
        }

        .wc-team-list-card {
            background: var(--wc-panel);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.75rem;
        }

        .wc-team-list-title {
            color: var(--wc-muted);
            font-size: 0.68rem;
            font-weight: 850;
            letter-spacing: 0.04em;
            margin-bottom: 0.5rem;
            text-transform: uppercase;
        }

        .wc-team-pill {
            align-items: center;
            border-bottom: 1px solid var(--wc-border);
            color: var(--wc-text);
            display: flex;
            font-size: 0.86rem;
            font-weight: 700;
            justify-content: space-between;
            padding: 0.38rem 0;
        }

        .wc-team-pill:last-child {
            border-bottom: 0;
        }

        .wc-team-pill-seed {
            color: var(--wc-muted);
            font-size: 0.7rem;
            font-weight: 800;
        }

        .wc-result-chip-grid {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.55rem;
        }

        .wc-result-panel-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 0.65rem 0 1rem;
        }

        .wc-result-chip {
            background: var(--wc-panel-soft);
            border: 1px solid var(--wc-border);
            border-radius: 999px;
            color: var(--wc-text);
            font-size: 0.82rem;
            font-weight: 750;
            line-height: 1.2;
            padding: 0.38rem 0.58rem;
        }

        .wc-comparison-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin: 0.75rem 0 1rem;
        }

        .wc-comparison-card {
            background: var(--wc-card-bg);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.85rem;
        }

        .wc-comparison-team {
            color: var(--wc-text);
            font-size: 1.05rem;
            font-weight: 850;
            line-height: 1.2;
            margin-bottom: 0.45rem;
        }

        .wc-comparison-meta {
            color: var(--wc-muted);
            font-size: 0.76rem;
            font-weight: 750;
            letter-spacing: 0.03em;
            text-transform: uppercase;
        }

        .wc-comparison-kpis {
            display: grid;
            gap: 0.5rem;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            margin-top: 0.7rem;
        }

        .wc-comparison-kpi {
            background: var(--wc-panel-soft);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.55rem;
        }

        .wc-comparison-kpi-label {
            color: var(--wc-muted);
            font-size: 0.68rem;
            font-weight: 800;
            text-transform: uppercase;
        }

        .wc-comparison-kpi-value {
            color: var(--wc-text);
            font-size: 1rem;
            font-weight: 850;
            margin-top: 0.15rem;
        }

        .wc-compare-table {
            background: var(--wc-card-bg);
            border: 1px solid var(--wc-border);
            border-collapse: separate;
            border-radius: 8px;
            border-spacing: 0;
            overflow: hidden;
            width: 100%;
        }

        .wc-compare-table th,
        .wc-compare-table td {
            border-bottom: 1px solid var(--wc-border);
            color: var(--wc-text);
            padding: 0.55rem 0.65rem;
            text-align: left;
            vertical-align: middle;
        }

        .wc-compare-table tr:last-child td {
            border-bottom: 0;
        }

        .wc-compare-table th {
            color: var(--wc-muted);
            font-size: 0.72rem;
            font-weight: 850;
            letter-spacing: 0.035em;
            text-transform: uppercase;
        }

        .wc-compare-cell {
            display: grid;
            gap: 0.32rem;
            min-width: 9rem;
        }

        .wc-compare-cell-value {
            color: var(--wc-text);
            font-size: 0.82rem;
            font-weight: 850;
        }

        .wc-compare-track {
            background: color-mix(in srgb, var(--wc-text) 12%, transparent);
            border-radius: 999px;
            height: 0.42rem;
            overflow: hidden;
        }

        .wc-compare-fill {
            background: var(--wc-accent);
            border-radius: inherit;
            height: 100%;
        }

        .wc-history-chart {
            background: var(--wc-card-bg);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            margin-top: 0.65rem;
            overflow-x: auto;
            padding: 0.8rem;
        }

        .wc-history-chart svg {
            display: block;
            min-width: 720px;
            width: 100%;
        }

        .wc-history-grid-line {
            stroke: color-mix(in srgb, var(--wc-text) 13%, transparent);
            stroke-width: 1;
        }

        .wc-history-axis-label {
            fill: var(--wc-muted);
            font-size: 11px;
            font-weight: 700;
        }

        .wc-history-point {
            stroke: var(--wc-card-bg);
            stroke-width: 2;
        }

        .wc-history-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 0.45rem;
            margin-top: 0.7rem;
        }

        .wc-history-legend-item {
            align-items: center;
            background: var(--wc-panel-soft);
            border: 1px solid var(--wc-border);
            border-radius: 999px;
            color: var(--wc-text);
            display: inline-flex;
            font-size: 0.8rem;
            font-weight: 750;
            gap: 0.35rem;
            padding: 0.34rem 0.55rem;
        }

        .wc-history-legend-swatch {
            border-radius: 999px;
            display: inline-block;
            height: 0.56rem;
            width: 0.56rem;
        }

        .wc-builder-group-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            margin: 0.75rem 0 1rem;
        }

        .wc-builder-match {
            background: var(--wc-card-bg);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            margin-bottom: 0.65rem;
            padding: 0.65rem;
        }

        .wc-builder-match-meta {
            color: var(--wc-muted);
            font-size: 0.72rem;
            font-weight: 800;
            letter-spacing: 0.035em;
            margin-bottom: 0.35rem;
            text-transform: uppercase;
        }

        .wc-builder-disabled {
            color: var(--wc-muted);
            font-size: 0.8rem;
            font-weight: 750;
            margin-top: 0.35rem;
        }

        .wc-builder-summary-grid {
            display: grid;
            gap: 0.75rem;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            margin: 0.75rem 0 1rem;
        }

        .wc-chart-card {
            background: var(--wc-card-bg);
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            padding: 0.75rem;
        }

        .wc-chart-row {
            align-items: center;
            display: grid;
            gap: 0.55rem;
            grid-template-columns: minmax(8rem, 1fr) minmax(8rem, 2fr) 3.7rem;
            margin: 0.45rem 0;
        }

        .wc-chart-label {
            color: var(--wc-text);
            font-size: 0.82rem;
            font-weight: 750;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }

        .wc-chart-track {
            background: color-mix(in srgb, var(--wc-text) 12%, transparent);
            border-radius: 999px;
            height: 0.55rem;
            overflow: hidden;
        }

        .wc-chart-fill {
            background: var(--wc-accent);
            border-radius: 999px;
            height: 100%;
        }

        .wc-chart-value {
            color: var(--wc-muted);
            font-size: 0.76rem;
            font-weight: 800;
            text-align: right;
        }

        [data-testid="stVegaLiteChart"] {
            background: var(--wc-card-bg) !important;
            border: 1px solid var(--wc-border) !important;
            border-radius: 8px !important;
            color: var(--wc-text) !important;
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

            .wc-team-list-grid {
                grid-template-columns: 1fr;
            }

            .wc-result-panel-grid {
                grid-template-columns: 1fr;
            }

            .wc-comparison-grid {
                grid-template-columns: 1fr;
            }

            .wc-builder-group-grid,
            .wc-builder-summary-grid {
                grid-template-columns: 1fr;
            }

            .wc-chart-row {
                grid-template-columns: 1fr 3.7rem;
            }

            .wc-chart-track {
                grid-column: 1 / -1;
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
            background: var(--wc-panel) !important;
            border: 1px solid var(--wc-border) !important;
            border-radius: 8px;
            padding: 0.75rem 0.8rem;
        }

        [data-testid="stMetricLabel"] {
            color: var(--wc-muted) !important;
            font-size: 0.75rem;
        }

        [data-testid="stMetricValue"] {
            color: var(--wc-text) !important;
            font-size: 1.2rem;
        }

        [data-testid="stDataFrame"] {
            background: var(--wc-panel) !important;
            border: 1px solid var(--wc-border);
            border-radius: 8px;
            color: var(--wc-text) !important;
            overflow: hidden;
        }

        [data-testid="stDataFrame"] > div,
        [data-testid="stDataFrame"] [role="grid"],
        [data-testid="stDataFrame"] [role="row"],
        [data-testid="stDataFrame"] [role="columnheader"],
        [data-testid="stDataFrame"] [role="gridcell"],
        [data-testid="stDataFrame"] [class*="glide"],
        [data-testid="stDataFrame"] [class*="data-grid"],
        [data-testid="stDataFrame"] [class*="cell"],
        [data-testid="stDataFrame"] [class*="header"] {
            background-color: var(--wc-panel) !important;
            color: var(--wc-text) !important;
            border-color: var(--wc-border) !important;
        }

        [data-testid="stDataFrame"] div,
        [data-testid="stDataFrame"] span,
        [data-testid="stDataFrame"] label,
        [data-testid="stDataFrame"] svg,
        [data-testid="stDataFrame"] canvas {
            background-color: transparent !important;
            color: var(--wc-text) !important;
        }

        [data-testid="stDataFrame"] button {
            background: var(--wc-panel-soft) !important;
            border-color: var(--wc-border) !important;
            color: var(--wc-text) !important;
        }

        [data-testid="stDataFrame"] input {
            background: var(--wc-panel) !important;
            color: var(--wc-text) !important;
        }

        [data-testid="stDataFrame"] [style*="background-color: rgb(255, 255, 255)"],
        [data-testid="stDataFrame"] [style*="background: rgb(255, 255, 255)"],
        [data-testid="stDataFrame"] [style*="background-color:white"],
        [data-testid="stDataFrame"] [style*="background:white"] {
            background: var(--wc-panel) !important;
            background-color: var(--wc-panel) !important;
        }

        [data-testid="stTable"],
        [data-testid="stTable"] table,
        [data-testid="stTable"] thead,
        [data-testid="stTable"] tbody,
        [data-testid="stTable"] tr,
        [data-testid="stTable"] th,
        [data-testid="stTable"] td {
            background: var(--wc-panel) !important;
            color: var(--wc-text) !important;
            border-color: var(--wc-border) !important;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            background: var(--wc-panel) !important;
            border-color: var(--wc-border) !important;
            color: var(--wc-text) !important;
        }

        div[data-baseweb="select"] span,
        div[data-baseweb="input"] input,
        div[data-baseweb="popover"],
        div[data-baseweb="popover"] ul,
        div[data-baseweb="popover"] li,
        div[data-baseweb="menu"],
        div[data-baseweb="menu"] li {
            background: var(--wc-panel) !important;
            color: var(--wc-text) !important;
        }

        div[data-baseweb="tab-list"] {
            background: var(--wc-bg) !important;
            border-bottom-color: var(--wc-border) !important;
        }

        button[data-baseweb="tab"] {
            background: transparent !important;
            color: var(--wc-muted) !important;
        }

        button[data-baseweb="tab"][aria-selected="true"],
        button[data-baseweb="tab"][aria-selected="true"] p {
            color: var(--wc-accent) !important;
        }

        [data-testid="stRadio"] label,
        [data-testid="stRadio"] span,
        [data-testid="stRadio"] div {
            color: var(--wc-text) !important;
        }

        [data-testid="stRadio"] [aria-checked="true"] {
            color: var(--wc-accent) !important;
        }

        .stTextInput input {
            background: var(--wc-panel) !important;
            color: var(--wc-text) !important;
        }

        .stAlert {
            background: var(--wc-panel-soft) !important;
            border-color: var(--wc-border) !important;
            border-radius: 8px;
            color: var(--wc-text) !important;
        }

        hr {
            border-color: var(--wc-border) !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_nav_section_for_page(page):
    for section, pages in NAV_GROUPS.items():
        if page in pages:
            return section
    return "Overview"


def nav_href(page):
    return f"?page={quote(page)}"


def get_query_page():
    try:
        page = st.query_params.get("page")
    except Exception:
        return None
    if isinstance(page, list):
        page = page[0] if page else None
    return page if page in NAV_ITEMS else None


def render_top_nav():
    if "selected_page" not in st.session_state:
        st.session_state["selected_page"] = "Overview"
    if st.session_state["selected_page"] not in NAV_ITEMS:
        st.session_state["selected_page"] = "Overview"

    query_page = get_query_page()
    if query_page:
        st.session_state["selected_page"] = query_page

    selected_page = st.session_state["selected_page"]
    selected_section = get_nav_section_for_page(selected_page)
    section_items = []
    for index, (section, pages) in enumerate(NAV_GROUPS.items()):
        if index > 0:
            section_items.append('<span class="wc-top-nav-separator">|</span>')
        is_active_section = section == selected_section
        section_class = "wc-top-nav-link wc-top-nav-link-active" if is_active_section else "wc-top-nav-link"
        caret = '<span class="wc-top-nav-caret">▼</span>' if section != "Overview" else ""
        dropdown_items = []
        if section != "Overview":
            for page in pages:
                is_active_page = page == selected_page
                page_class = (
                    "wc-top-nav-dropdown-link wc-top-nav-dropdown-link-active"
                    if is_active_page
                    else "wc-top-nav-dropdown-link"
                )
                dropdown_items.append(
                    f'<a class="{page_class}" href="{html.escape(nav_href(page))}" target="_self">'
                    f'{html.escape(NAV_PAGE_LABELS.get(page, page))}</a>'
                )
        dropdown = (
            f'<div class="wc-top-nav-dropdown">{"".join(dropdown_items)}</div>'
            if dropdown_items
            else ""
        )
        section_items.append(
            f'<div class="wc-top-nav-item">'
            f'<a class="{section_class}" href="{html.escape(nav_href(pages[0]))}" target="_self">'
            f'{html.escape(section)}{caret}</a>'
            f'{dropdown}'
            f'</div>'
        )

    st.markdown(
        f"""
        <div class="wc-top-nav-shell">
            <div class="wc-top-nav-title">World Cup Predictor</div>
            <div class="wc-top-nav-bar">{"".join(section_items)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

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


def render_dark_bar_chart(data, label_column, value_column, value_title=None, height=320):
    chart_source = data[[label_column, value_column]].copy()
    chart_source[label_column] = chart_source[label_column].astype(str)
    chart_source[value_column] = pd.to_numeric(chart_source[value_column], errors="coerce")
    chart_source = chart_source.dropna(subset=[value_column])
    if chart_source.empty:
        st.info("No chart data available.")
        return

    rows = []
    for row in chart_source.itertuples(index=False):
        label = getattr(row, label_column)
        value = float(getattr(row, value_column))
        clipped_value = max(0.0, min(1.0, value))
        rows.append(
            "<div class='wc-chart-row'>"
            f"<div class='wc-chart-label' title='{html.escape(label)}'>{html.escape(label)}</div>"
            "<div class='wc-chart-track'>"
            f"<div class='wc-chart-fill' style='width: {clipped_value * 100:.1f}%'></div>"
            "</div>"
            f"<div class='wc-chart-value'>{format_percent(value)}</div>"
            "</div>"
        )

    st.markdown(
        f"""
        <div class="wc-chart-card" style="max-height: {int(height)}px; overflow-y: auto;">
            {''.join(rows)}
        </div>
        """,
        unsafe_allow_html=True,
    )


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
def load_champion_history():
    if not CHAMPION_HISTORY_PATH.exists():
        return None
    return pd.read_csv(CHAMPION_HISTORY_PATH)


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


@st.cache_data
def replay_single_simulation(simulation_id, base_seed, group_predictions, annex_allocations, fixtures):
    return run_single_simulation(
        simulation_id=simulation_id,
        base_seed=base_seed,
        group_predictions=group_predictions,
        annex_allocations=annex_allocations,
        fixtures=fixtures,
    )


def get_simulation_count(metadata):
    try:
        return int(metadata.get("number_of_simulations", 10000))
    except (TypeError, ValueError):
        return 10000


def get_base_seed(metadata):
    try:
        return int(metadata.get("base_seed", 2026))
    except (TypeError, ValueError):
        return 2026


def format_probability_columns(data, columns):
    formatted = data.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_percent)
    return formatted


def add_outcome_summary(data):
    data = data.copy()
    probability_columns = ["prob_home_win", "prob_draw", "prob_away_win"]
    missing_columns = [column for column in probability_columns if column not in data.columns]
    if missing_columns:
        raise ValueError(
            "Missing required probability columns: " + ", ".join(missing_columns)
        )
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
    )
    st.subheader(title)
    render_dark_bar_chart(
        chart_data,
        display_label_column,
        probability_column,
        value_title="Probability",
        height=max(240, min(520, 30 * len(chart_data))),
    )


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
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:.1%}"


def format_team_list(teams):
    if not teams:
        return "Unavailable"
    return ", ".join(team_label(team) for team in teams)


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
    if (
        "highest_probability_outcome" not in fixtures.columns
        or "highest_probability" not in fixtures.columns
    ):
        fixtures = add_outcome_summary(fixtures)

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


def official_bracket_card_html(row, selected_team=None):
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
    if selected_team and selected_team in {
        get_placeholder_label(row["Home Team"]),
        get_placeholder_label(row["Away Team"]),
    }:
        extra_class += " wc-bracket-card-highlight"
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
    if probability is None or pd.isna(probability):
        return textwrap.dedent("""
            <div class="wc-bracket-prob-row">
                <div class="wc-bracket-bar">
                    <div class="wc-bracket-bar-fill" style="width: 0%"></div>
                </div>
                <div class="wc-bracket-prob-value">N/A</div>
            </div>
        """).strip()
    bounded_probability = max(0, min(1, probability))
    return textwrap.dedent(f"""
        <div class="wc-bracket-prob-row">
            <div class="wc-bracket-bar">
                <div class="wc-bracket-bar-fill" style="width: {bounded_probability * 100:.1f}%"></div>
            </div>
            <div class="wc-bracket-prob-value">{bounded_probability:.0%}</div>
        </div>
    """).strip()


def simulated_bracket_card_html(row, selected_team=None):
    home_probability = row.get("home_advancement_probability")
    away_probability = row.get("away_advancement_probability")
    favored_team = row.get("winner")
    if not favored_team and home_probability is not None and away_probability is not None:
        favored_team = row["home_team"] if home_probability >= away_probability else row["away_team"]
    favored_probability = (
        max(home_probability, away_probability)
        if home_probability is not None and away_probability is not None
        else None
    )
    card_count = get_bracket_card_count(row)
    extra_class = " wc-bracket-third-place" if row.get("is_third_place") else ""
    if selected_team and selected_team in {row["home_team"], row["away_team"]}:
        extra_class += " wc-bracket-card-highlight"
        if row.get("winner") and row["winner"] != selected_team:
            extra_class += " wc-bracket-card-eliminated"
    meta_label = "Third-Place Match" if row.get("is_third_place") else card_count
    result_label = (
        f"{html.escape(team_label(favored_team))} {format_percent(favored_probability)}"
        if favored_team
        else "Result not simulated"
    )

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
            <div class="wc-bracket-prob">{result_label}</div>
        </div>
    """).strip()


def split_bracket_rows(rows):
    midpoint = (len(rows) + 1) // 2
    return rows[:midpoint], rows[midpoint:]


def bracket_round_html(round_name, rows, depth, card_builder, selected_team=None):
    cards = "".join(card_builder(row, selected_team=selected_team) for row in rows)
    if not cards:
        cards = '<div class="wc-bracket-card"><div class="wc-bracket-card-meta">No matchups</div></div>'
    return textwrap.dedent(f"""
        <div class="wc-bracket-round wc-bracket-depth-{depth}">
            <div class="wc-bracket-round-title">{html.escape(round_name)}</div>
            {cards}
        </div>
    """).strip()


def bracket_side_html(round_rows, side_classes, card_builder, reverse=False, selected_team=None):
    display_rows = list(reversed(round_rows)) if reverse else round_rows
    depth_by_round = {round_name: depth for depth, (round_name, _) in enumerate(round_rows)}
    columns = []
    for index, (round_name, rows) in enumerate(display_rows):
        columns.append(
            textwrap.dedent(f"""
                <div class="wc-bracket-column {side_classes[index]}">
                    {bracket_round_html(round_name, rows, depth_by_round[round_name], card_builder, selected_team)}
                </div>
            """).strip()
        )
    return "".join(columns)


def bracket_center_html(finals, card_builder, third_place=None, selected_team=None):
    cards = "".join(card_builder(row, selected_team=selected_team) for row in finals)
    if not cards:
        cards = '<div class="wc-bracket-card"><div class="wc-bracket-card-meta">No final matchups</div></div>'
    third_place_html = (
        card_builder(third_place, selected_team=selected_team)
        if third_place is not None
        else ""
    )
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


def add_advancement_probabilities(record):
    enriched = dict(record)
    home_adv_prob, away_adv_prob = convert_to_advancement_probabilities(
        enriched.get("home_probability", 0),
        enriched.get("draw_probability", 0),
        enriched.get("away_probability", 0),
    )
    enriched["home_advancement_probability"] = home_adv_prob
    enriched["away_advancement_probability"] = away_adv_prob
    return enriched


def get_final_record(simulation_result):
    return next(
        (
            record
            for record in simulation_result["knockout_records"]
            if record["round"] == "Finals"
        ),
        None,
    )


def get_runner_up(simulation_result):
    final_record = get_final_record(simulation_result)
    if final_record is None:
        return None
    return (
        final_record["away_team"]
        if final_record["winner"] == final_record["home_team"]
        else final_record["home_team"]
    )


def get_semi_final_losers(simulation_result):
    semi_records = [
        record
        for record in simulation_result["knockout_records"]
        if record["round"] == "Semi Finals"
    ]
    losers = []
    for record in semi_records[:2]:
        loser = (
            record["away_team"]
            if record["winner"] == record["home_team"]
            else record["home_team"]
        )
        losers.append(loser)
    return losers


def build_replay_third_place_match(simulation_result, group_predictions):
    semi_losers = get_semi_final_losers(simulation_result)
    if len(semi_losers) != 2:
        return None
    team_elos = build_team_elo_lookup(group_predictions)
    match_probability_lookup = build_match_probability_lookup(group_predictions)
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


def build_replay_bracket_records(simulation_result, group_predictions):
    records = [
        add_advancement_probabilities(record)
        for record in simulation_result["knockout_records"]
    ]
    third_place = build_replay_third_place_match(simulation_result, group_predictions)
    if third_place is not None:
        records.append(third_place)
    return records


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


def get_path_status(records, selected_team):
    if selected_team == "All":
        return "Select a team to highlight its path through the most-likely bracket."

    team_records = [
        record
        for record in records
        if selected_team in {record.get("home_team"), record.get("away_team")}
    ]
    if not team_records:
        return f"{team_label(selected_team)} does not appear in this most-likely bracket path."

    final_record = next(
        (record for record in records if record["round"] == "Finals"),
        None,
    )
    if final_record and final_record.get("winner") == selected_team:
        return f"{team_label(selected_team)} is champion in this most-likely path."

    elimination = next(
        (
            record
            for record in team_records
            if record.get("winner") != selected_team
            and record["round"] != "Third-Place Match"
        ),
        None,
    )
    if elimination is not None:
        return f"{team_label(selected_team)} path ends in {elimination['round']}."

    return f"{team_label(selected_team)} appears in the highlighted path."


def render_two_sided_bracket(
    left_round_rows,
    finals,
    right_round_rows,
    card_builder,
    third_place=None,
    selected_team=None,
):
    bracket_html = textwrap.dedent(f"""
        <div class="wc-bracket-mobile-note">Bracket scrolls horizontally on smaller screens.</div>
        <div class="wc-bracket-viewport">
            <div class="wc-bracket-board">
                {bracket_side_html(left_round_rows, ["wc-bracket-left", "wc-bracket-left", "wc-bracket-left-middle", "wc-bracket-left-middle"], card_builder, selected_team=selected_team)}
                {bracket_center_html(finals, card_builder, third_place, selected_team)}
                {bracket_side_html(right_round_rows, ["wc-bracket-right-middle", "wc-bracket-right-middle", "wc-bracket-right", "wc-bracket-right"], card_builder, reverse=True, selected_team=selected_team)}
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
    selected_team="All",
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

    st.markdown(
        f'<div class="wc-bracket-path-note">{get_path_status(path_records, selected_team)}</div>',
        unsafe_allow_html=True,
    )
    selected_team_for_highlight = None if selected_team == "All" else selected_team
    left_round_rows, finals, right_round_rows, third_place = split_records_for_bracket(path_records)
    render_two_sided_bracket(
        left_round_rows,
        finals,
        right_round_rows,
        simulated_bracket_card_html,
        third_place=third_place,
        selected_team=selected_team_for_highlight,
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
    render_dark_bar_chart(
        probability_chart.rename_axis("outcome").reset_index(name="probability"),
        "outcome",
        "probability",
        value_title="Probability",
        height=220,
    )

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
    render_dark_bar_chart(
        advancement_chart.rename_axis("outcome").reset_index(name="probability"),
        "outcome",
        "probability",
        value_title="Probability",
        height=180,
    )

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
        advancement_series = build_team_advancement_series(team_row)
        render_dark_bar_chart(
            advancement_series.rename_axis("stage").reset_index(name="probability"),
            "stage",
            "probability",
            value_title="Probability",
            height=260,
        )
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


def probability_cell_html(value):
    clipped_value = max(0.0, min(1.0, float(value))) if value is not None and not pd.isna(value) else 0
    return textwrap.dedent(f"""
        <div class="wc-compare-cell">
            <div class="wc-compare-cell-value">{format_percent(value)}</div>
            <div class="wc-compare-track">
                <div class="wc-compare-fill" style="width: {clipped_value * 100:.1f}%"></div>
            </div>
        </div>
    """).strip()


def render_team_comparison_summary_cards(team_a, team_b, row_a, row_b):
    card_html = []
    for team, row in [(team_a, row_a), (team_b, row_b)]:
        card_html.append(
            textwrap.dedent(f"""
                <div class="wc-comparison-card">
                    <div class="wc-comparison-team">{html.escape(team_label(team))}</div>
                    <div class="wc-comparison-meta">Group {html.escape(str(row.get("group", "Unknown")))}</div>
                    <div class="wc-comparison-kpis">
                        <div class="wc-comparison-kpi">
                            <div class="wc-comparison-kpi-label">Champion</div>
                            <div class="wc-comparison-kpi-value">{format_percent(row["champion_probability"])}</div>
                        </div>
                        <div class="wc-comparison-kpi">
                            <div class="wc-comparison-kpi-label">Final</div>
                            <div class="wc-comparison-kpi-value">{format_percent(row["final_probability"])}</div>
                        </div>
                    </div>
                </div>
            """).strip()
        )
    st.markdown(
        f'<div class="wc-comparison-grid">{"".join(card_html)}</div>',
        unsafe_allow_html=True,
    )


def render_team_probability_comparison(team_a, team_b, row_a, row_b):
    metrics = [
        ("Round of 32", "round_of_32_probability"),
        ("Round of 16", "round_of_16_probability"),
        ("Quarter-final", "quarter_final_probability"),
        ("Semi-final", "semi_final_probability"),
        ("Final", "final_probability"),
        ("Champion", "champion_probability"),
    ]
    rows = []
    for label, column in metrics:
        rows.append(
            "<tr>"
            f"<td><strong>{html.escape(label)}</strong></td>"
            f"<td>{probability_cell_html(row_a[column])}</td>"
            f"<td>{probability_cell_html(row_b[column])}</td>"
            "</tr>"
        )

    st.markdown(
        textwrap.dedent(f"""
            <table class="wc-compare-table">
                <thead>
                    <tr>
                        <th>Metric</th>
                        <th>{html.escape(team_label(team_a))}</th>
                        <th>{html.escape(team_label(team_b))}</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        """),
        unsafe_allow_html=True,
    )


def render_team_group_context(group_tables, selected_team):
    group_table = build_team_group_table(group_tables, selected_team)
    if group_table.empty:
        st.info("No group table context found for this team.")
        return
    st.dataframe(group_table, use_container_width=True, hide_index=True)


def render_team_fixture_context(group_predictions, selected_team):
    fixtures = build_team_fixtures(group_predictions, selected_team)
    if fixtures.empty:
        st.info("No group-stage fixtures found for this team.")
        return
    st.dataframe(
        build_fixture_display_table(fixtures),
        use_container_width=True,
        hide_index=True,
        column_config=fixture_probability_column_config(),
    )


def render_team_knockout_context(match_summary, selected_team):
    matchup_display = build_team_knockout_matchups(match_summary, selected_team, top_n=8)
    if matchup_display is None:
        st.info("Knockout opponent detail is unavailable from the current simulation summary outputs.")
    elif matchup_display.empty:
        st.info("No simulated knockout matchups found for this team.")
    else:
        st.dataframe(matchup_display, use_container_width=True, hide_index=True)


def render_team_comparison_page(group_predictions, group_tables, simulation_summary, match_summary):
    render_page_header(
        "Team Comparison",
        "Side-by-side team probabilities, group context, fixtures, and optional knockout outlook.",
    )

    teams = sorted(simulation_summary["team"].dropna().unique())
    option_map = team_options(teams)
    option_labels = list(option_map.keys())
    default_b_index = 1 if len(option_labels) > 1 else 0

    selector_columns = st.columns(2)
    with selector_columns[0]:
        team_a_label = st.selectbox("Team A", option_labels, key="comparison_team_a")
    with selector_columns[1]:
        team_b_label = st.selectbox(
            "Team B",
            option_labels,
            index=default_b_index,
            key="comparison_team_b",
        )

    team_a = option_map[team_a_label]
    team_b = option_map[team_b_label]
    if team_a == team_b:
        st.warning("Select two different teams to compare.")
        return

    row_a = get_team_summary_row(simulation_summary, team_a)
    row_b = get_team_summary_row(simulation_summary, team_b)
    if row_a is None or row_b is None:
        st.error("One of the selected teams is missing from the simulation summary.")
        return

    render_team_comparison_summary_cards(team_a, team_b, row_a, row_b)

    st.subheader("Advancement Probability Comparison")
    render_team_probability_comparison(team_a, team_b, row_a, row_b)

    st.subheader("Group Context")
    group_columns = st.columns(2)
    with group_columns[0]:
        st.markdown(f"**{team_label(team_a)}**")
        render_team_group_context(group_tables, team_a)
    with group_columns[1]:
        st.markdown(f"**{team_label(team_b)}**")
        render_team_group_context(group_tables, team_b)

    st.subheader("Group Fixture Context")
    fixture_columns = st.columns(2)
    with fixture_columns[0]:
        st.markdown(f"**{team_label(team_a)}**")
        render_team_fixture_context(group_predictions, team_a)
    with fixture_columns[1]:
        st.markdown(f"**{team_label(team_b)}**")
        render_team_fixture_context(group_predictions, team_b)

    st.subheader("Optional Knockout Outlook")
    knockout_columns = st.columns(2)
    with knockout_columns[0]:
        st.markdown(f"**{team_label(team_a)}**")
        render_team_knockout_context(match_summary, team_a)
    with knockout_columns[1]:
        st.markdown(f"**{team_label(team_b)}**")
        render_team_knockout_context(match_summary, team_b)


def build_group_difficulty_table(simulation_summary):
    group_source = simulation_summary.copy()
    aggregation_columns = {"round_of_16_probability": "mean"}
    if "champion_probability" in group_source.columns:
        aggregation_columns["champion_probability"] = "mean"

    difficulty = (
        group_source.groupby("group", dropna=False)
        .agg(aggregation_columns)
        .reset_index()
        .rename(
            columns={
                "round_of_16_probability": "avg_round_of_16_probability",
                "champion_probability": "avg_champion_probability",
            }
        )
        .sort_values("avg_round_of_16_probability", ascending=False)
    )
    difficulty["rank"] = range(1, len(difficulty) + 1)
    return difficulty[
        [
            "rank",
            "group",
            "avg_round_of_16_probability",
            "avg_champion_probability",
        ]
    ]


def get_most_likely_final_storyline(simulation_summary, match_summary):
    if match_summary is not None:
        final_rows = match_summary[match_summary["round"] == "Finals"]
        if not final_rows.empty:
            top_final = final_rows.sort_values("simulations", ascending=False).iloc[0]
            return (
                f"{team_label(top_final['home_team'])} vs {team_label(top_final['away_team'])}",
                f"{int(top_final['simulations'])} simulated finals",
                "Most common Finals matchup in match simulation summary.",
            )

    finalists = simulation_summary.sort_values("final_probability", ascending=False).head(2)
    if len(finalists) < 2:
        return ("Unavailable", "Optional match summary missing", "Not enough finalist data available.")

    return (
        f"{team_label(finalists.iloc[0]['team'])} vs {team_label(finalists.iloc[1]['team'])}",
        "Approximation",
        "Optional Finals matchup file missing; using top two final_probability teams.",
    )


def get_dark_horse_candidates(simulation_summary, candidate_count=5):
    ranked = simulation_summary.sort_values("champion_probability", ascending=False).copy()
    top_champion_teams = set(ranked.head(5)["team"])
    candidates = ranked[~ranked["team"].isin(top_champion_teams)].copy()
    if "group_winner_probability" in candidates.columns:
        modest_group_winner = candidates[candidates["group_winner_probability"] < 0.4]
        if not modest_group_winner.empty:
            candidates = modest_group_winner
    return candidates.sort_values("champion_probability", ascending=False).head(candidate_count)


def get_upset_candidates(simulation_summary, candidate_count=5):
    ranked = simulation_summary.sort_values("champion_probability", ascending=False).copy()
    top_champion_teams = set(ranked.head(10)["team"])
    candidates = ranked[~ranked["team"].isin(top_champion_teams)].copy()
    candidates["upset_signal"] = (
        candidates["quarter_final_probability"].fillna(0)
        + candidates["round_of_16_probability"].fillna(0)
        - candidates["champion_probability"].fillna(0)
    )
    return candidates.sort_values(
        ["quarter_final_probability", "round_of_16_probability", "upset_signal"],
        ascending=[False, False, False],
    ).head(candidate_count)


def get_most_volatile_team(simulation_summary):
    required_columns = {"team", "round_of_16_probability", "champion_probability"}
    if not required_columns.issubset(simulation_summary.columns):
        return None
    volatility = simulation_summary.copy()
    volatility["volatility_score"] = (
        volatility["round_of_16_probability"].fillna(0)
        - volatility["champion_probability"].fillna(0)
    )
    return volatility.sort_values("volatility_score", ascending=False).iloc[0]


def storyline_card_html(label, value, sub_value):
    return textwrap.dedent(f"""
        <div class="wc-insight">
            <div class="wc-insight-label">{html.escape(str(label))}</div>
            <div class="wc-insight-value">{html.escape(str(value))}</div>
            <div class="wc-bracket-summary-sub">{html.escape(str(sub_value))}</div>
        </div>
    """).strip()


def render_storyline_cards(items):
    for index in range(0, len(items), 3):
        card_html = "".join(
            storyline_card_html(label, value, sub_value)
            for label, value, sub_value in items[index:index + 3]
        )
        st.markdown(
            f'<div class="wc-insight-strip">{card_html}</div>',
            unsafe_allow_html=True,
        )


def build_candidate_table(candidates, include_group_winner=True):
    if candidates.empty:
        return pd.DataFrame()
    columns = [
        "team",
        "group",
        "group_winner_probability",
        "round_of_16_probability",
        "quarter_final_probability",
        "final_probability",
        "champion_probability",
    ]
    available_columns = [column for column in columns if column in candidates.columns]
    display = candidates[available_columns].copy()
    display["team"] = display["team"].map(team_label)
    probability_columns = [
        column
        for column in [
            "group_winner_probability",
            "round_of_16_probability",
            "quarter_final_probability",
            "final_probability",
            "champion_probability",
        ]
        if column in display.columns
    ]
    if not include_group_winner and "group_winner_probability" in display.columns:
        display = display.drop(columns=["group_winner_probability"])
        probability_columns = [
            column for column in probability_columns if column != "group_winner_probability"
        ]
    return format_probability_columns(display, probability_columns)


def render_tournament_storylines(
    simulation_summary,
    match_summary,
    group_tables,
    group_predictions,
):
    render_page_header(
        "Tournament Storylines",
        "High-level simulation insights derived from existing prediction and tournament probability files.",
    )

    champion_row = simulation_summary.sort_values("champion_probability", ascending=False).iloc[0]
    final_value, final_subtitle, final_note = get_most_likely_final_storyline(
        simulation_summary,
        match_summary,
    )
    strongest_group_row = simulation_summary.sort_values(
        "group_winner_probability",
        ascending=False,
    ).iloc[0]
    group_difficulty = build_group_difficulty_table(simulation_summary)
    hardest_group = group_difficulty.iloc[0]
    dark_horse_candidates = get_dark_horse_candidates(simulation_summary)
    dark_horse = dark_horse_candidates.iloc[0] if not dark_horse_candidates.empty else None
    upset_candidates = get_upset_candidates(simulation_summary)
    upset_candidate = upset_candidates.iloc[0] if not upset_candidates.empty else None
    volatile_team = get_most_volatile_team(simulation_summary)

    headline_items = [
        (
            "Most Likely Champion",
            team_label(champion_row["team"]),
            format_percent(champion_row["champion_probability"]),
        ),
        ("Most Likely Final", final_value, final_subtitle),
        (
            "Strongest Group-Stage Team",
            team_label(strongest_group_row["team"]),
            f"{format_percent(strongest_group_row['group_winner_probability'])} group winner",
        ),
    ]
    render_storyline_cards(headline_items)

    st.subheader("Group Difficulty")
    render_storyline_cards(
        [
            (
                "Hardest Group",
                f"Group {hardest_group['group']}",
                f"{format_percent(hardest_group['avg_round_of_16_probability'])} average R16 probability",
            )
        ]
    )
    difficulty_display = format_probability_columns(
        group_difficulty.rename(
            columns={
                "avg_round_of_16_probability": "avg_round_of_16",
                "avg_champion_probability": "avg_champion",
            }
        ),
        ["avg_round_of_16", "avg_champion"],
    )
    st.dataframe(difficulty_display, use_container_width=True, hide_index=True)

    st.subheader("Dark Horse Watch")
    if dark_horse is None:
        st.info("Dark horse candidates are unavailable from the current aggregate outputs.")
    else:
        render_storyline_cards(
            [
                (
                    "Main Dark Horse",
                    team_label(dark_horse["team"]),
                    f"{format_percent(dark_horse['champion_probability'])} champion probability",
                )
            ]
        )
        st.caption("Heuristic: exclude top five champion-probability teams, then prefer modest group-winner probability.")
        st.dataframe(
            build_candidate_table(dark_horse_candidates),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Upset Candidates")
    if upset_candidate is None:
        st.info("Upset candidates are unavailable from the current aggregate outputs.")
    else:
        render_storyline_cards(
            [
                (
                    "Biggest Upset Candidate",
                    team_label(upset_candidate["team"]),
                    f"{format_percent(upset_candidate['quarter_final_probability'])} quarter-final probability",
                )
            ]
        )
        st.caption("Heuristic: outside the top champion tier, ranked by quarter-final and Round of 16 probability.")
        st.dataframe(
            build_candidate_table(upset_candidates, include_group_winner=False),
            use_container_width=True,
            hide_index=True,
        )

    st.subheader("Volatility")
    if volatile_team is None:
        st.info("Volatility metric unavailable with current aggregate outputs.")
    else:
        render_storyline_cards(
            [
                (
                    "Most Volatile Team",
                    team_label(volatile_team["team"]),
                    (
                        f"{format_percent(volatile_team['round_of_16_probability'])} R16 vs "
                        f"{format_percent(volatile_team['champion_probability'])} champion"
                    ),
                )
            ]
        )
        st.caption(
            "Heuristic: highest gap between Round of 16 probability and champion probability. "
            "This captures teams likely to advance early but unlikely to win the tournament."
        )

    st.subheader("Method Notes")
    st.markdown(
        f"""
        <div class="wc-caveat-strip">
            <strong>Storylines are derived from existing aggregate outputs.</strong><br>
            Most likely champion uses champion probability. Most likely final uses the most common Finals matchup
            when `{MATCH_SIMULATION_SUMMARY_PATH}` is available; otherwise it approximates using the top two
            final probabilities. Hardest group is ranked by average Round of 16 probability. Dark horse, upset
            candidate, and volatility are labeled heuristics because they combine aggregate probabilities rather
            than simulation-level narratives.
        </div>
        """,
        unsafe_allow_html=True,
    )


def validate_champion_history(history):
    required_columns = {
        "snapshot_date",
        "team",
        "champion_probability",
        "input_fingerprint",
        "simulation_count",
    }
    missing_columns = sorted(required_columns - set(history.columns))
    if missing_columns:
        return missing_columns
    return []


def prepare_champion_history(history):
    prepared = history.copy()
    prepared["snapshot_date"] = pd.to_datetime(prepared["snapshot_date"], errors="coerce")
    prepared["champion_probability"] = pd.to_numeric(
        prepared["champion_probability"],
        errors="coerce",
    )
    prepared["simulation_count"] = pd.to_numeric(
        prepared["simulation_count"],
        errors="coerce",
    )
    prepared = prepared.dropna(subset=["snapshot_date", "team", "champion_probability"])
    return prepared.sort_values(["snapshot_date", "team"])


def get_latest_history_teams(history, team_count=10):
    latest_snapshot = history["snapshot_date"].max()
    latest_rows = history[history["snapshot_date"] == latest_snapshot]
    return (
        latest_rows.sort_values("champion_probability", ascending=False)
        .head(team_count)["team"]
        .tolist()
    )


def render_champion_history_chart(history, selected_teams):
    chart_data = history[history["team"].isin(selected_teams)].copy()
    if chart_data.empty:
        st.info("No history rows found for the selected teams.")
        return

    chart_data = (
        chart_data.sort_values("snapshot_date")
        .groupby(["snapshot_date", "team"], as_index=False)
        .tail(1)
    )
    dates = sorted(chart_data["snapshot_date"].dropna().unique())
    if not dates:
        st.info("No valid snapshot dates found in the history file.")
        return

    width = 900
    height = 320
    left_margin = 54
    right_margin = 20
    top_margin = 20
    bottom_margin = 44
    plot_width = width - left_margin - right_margin
    plot_height = height - top_margin - bottom_margin
    max_probability = min(1.0, max(0.05, chart_data["champion_probability"].max() * 1.12))
    min_timestamp = pd.Timestamp(min(dates)).timestamp()
    max_timestamp = pd.Timestamp(max(dates)).timestamp()

    def x_position(date_value):
        if min_timestamp == max_timestamp:
            return left_margin + plot_width / 2
        timestamp = pd.Timestamp(date_value).timestamp()
        return left_margin + ((timestamp - min_timestamp) / (max_timestamp - min_timestamp)) * plot_width

    def y_position(probability):
        return top_margin + plot_height - (float(probability) / max_probability) * plot_height

    palette = [
        "#ef4444",
        "#22c55e",
        "#38bdf8",
        "#f59e0b",
        "#a855f7",
        "#14b8a6",
        "#f97316",
        "#e879f9",
        "#84cc16",
        "#60a5fa",
    ]
    grid_lines = []
    for fraction in [0, 0.25, 0.5, 0.75, 1.0]:
        y = top_margin + plot_height - fraction * plot_height
        value = fraction * max_probability
        grid_lines.append(
            f'<line class="wc-history-grid-line" x1="{left_margin}" y1="{y:.1f}" '
            f'x2="{width - right_margin}" y2="{y:.1f}" />'
        )
        grid_lines.append(
            f'<text class="wc-history-axis-label" x="8" y="{y + 4:.1f}">{value:.0%}</text>'
        )

    team_shapes = []
    legend_items = []
    for index, team in enumerate(selected_teams):
        team_rows = chart_data[chart_data["team"] == team].sort_values("snapshot_date")
        if team_rows.empty:
            continue
        color = palette[index % len(palette)]
        points = [
            f"{x_position(row.snapshot_date):.1f},{y_position(row.champion_probability):.1f}"
            for row in team_rows.itertuples(index=False)
        ]
        if len(points) > 1:
            team_shapes.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2.5" '
                f'stroke-linecap="round" stroke-linejoin="round" points="{" ".join(points)}" />'
            )
        for row in team_rows.itertuples(index=False):
            x = x_position(row.snapshot_date)
            y = y_position(row.champion_probability)
            team_shapes.append(
                f'<circle class="wc-history-point" cx="{x:.1f}" cy="{y:.1f}" '
                f'r="4" fill="{color}"><title>{html.escape(team_label(team))}: '
                f'{format_percent(row.champion_probability)} on {pd.Timestamp(row.snapshot_date).date()}</title></circle>'
            )
        legend_items.append(
            f'<span class="wc-history-legend-item"><span class="wc-history-legend-swatch" '
            f'style="background:{color}"></span>{html.escape(team_label(team))}</span>'
        )

    first_date_label = pd.Timestamp(min(dates)).strftime("%Y-%m-%d")
    last_date_label = pd.Timestamp(max(dates)).strftime("%Y-%m-%d")
    x_axis_labels = (
        f'<text class="wc-history-axis-label" x="{left_margin}" y="{height - 12}">{first_date_label}</text>'
        f'<text class="wc-history-axis-label" x="{width - right_margin}" y="{height - 12}" text-anchor="end">{last_date_label}</text>'
    )

    st.markdown(
        textwrap.dedent(f"""
            <div class="wc-history-chart">
                <svg viewBox="0 0 {width} {height}" role="img" aria-label="Champion probability history">
                    {''.join(grid_lines)}
                    <line class="wc-history-grid-line" x1="{left_margin}" y1="{top_margin}" x2="{left_margin}" y2="{height - bottom_margin}" />
                    <line class="wc-history-grid-line" x1="{left_margin}" y1="{height - bottom_margin}" x2="{width - right_margin}" y2="{height - bottom_margin}" />
                    {''.join(team_shapes)}
                    {x_axis_labels}
                </svg>
                <div class="wc-history-legend">{''.join(legend_items)}</div>
            </div>
        """),
        unsafe_allow_html=True,
    )


def build_snapshot_metadata(history):
    metadata = (
        history.groupby("snapshot_date")
        .agg(
            input_fingerprint=("input_fingerprint", "first"),
            simulation_count=("simulation_count", "first"),
            teams_in_snapshot=("team", "nunique"),
        )
        .reset_index()
        .sort_values("snapshot_date", ascending=False)
    )
    metadata["snapshot_date"] = metadata["snapshot_date"].dt.strftime("%Y-%m-%d")
    metadata["simulation_count"] = metadata["simulation_count"].fillna(0).astype(int)
    return metadata


def render_champion_history(simulation_summary, champion_history):
    render_page_header(
        "Champion Probability History",
        "Track how each team’s title probability changes across future model or simulation snapshots.",
    )

    show_probability_bar_chart(
        simulation_summary,
        "team",
        "champion_probability",
        "Current Top 10 Champion Probabilities",
        top_n=10,
    )
    current_display = simulation_summary[
        ["team", "group", "final_probability", "champion_probability"]
    ].sort_values("champion_probability", ascending=False).head(10).copy()
    current_display["team"] = current_display["team"].map(team_label)
    current_display = format_probability_columns(
        current_display,
        ["final_probability", "champion_probability"],
    )
    st.dataframe(current_display, use_container_width=True, hide_index=True)

    st.subheader("History Chart")
    if champion_history is None:
        st.info(
            "Champion probability history is not available yet.\n\n"
            "This page will display trends once simulation snapshots are exported over time to:\n\n"
            f"`{CHAMPION_HISTORY_PATH}`\n\n"
            "No fake trends are shown."
        )
        st.subheader("Method Notes")
        st.markdown(
            """
            <div class="wc-caveat-strip">
                Champion history is based on saved simulation snapshots. This dashboard does not rerun
                simulations or create historical data. Until a history CSV exists, only current champion
                probabilities are shown.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    missing_columns = validate_champion_history(champion_history)
    if missing_columns:
        st.warning(
            "Champion history file exists, but it is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.info(f"Expected file shape: {CHAMPION_HISTORY_PATH}")
        return

    history = prepare_champion_history(champion_history)
    if history.empty:
        st.info("Champion history file exists, but no valid snapshot rows were found.")
        return

    latest_teams = get_latest_history_teams(history, team_count=10)
    option_map = team_options(history["team"].dropna().unique())
    default_labels = [
        label
        for label, team in option_map.items()
        if team in set(latest_teams[:5])
    ]
    selected_labels = st.multiselect(
        "Teams",
        list(option_map.keys()),
        default=default_labels,
        key="champion_history_teams",
    )
    selected_teams = [option_map[label] for label in selected_labels]
    if not selected_teams:
        st.info("Select at least one team to show history.")
    else:
        render_champion_history_chart(history, selected_teams)

    st.subheader("Snapshot Metadata")
    st.dataframe(
        build_snapshot_metadata(history),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Method Notes")
    st.markdown(
        f"""
        <div class="wc-caveat-strip">
            Champion history is based on saved simulation snapshots from `{CHAMPION_HISTORY_PATH}`.
            The dashboard reads that file only; it does not rerun simulations. Changes in this chart can reflect
            updates to model outputs, fixture inputs, team data, or simulation inputs over time.
        </div>
        """,
        unsafe_allow_html=True,
    )


def scenario_result_label(result, home_team, away_team):
    if result == "home_win":
        return f"{team_label(home_team)} win"
    if result == "away_win":
        return f"{team_label(away_team)} win"
    return "Draw"


def scenario_result_from_selection(selection, match):
    option_map = {
        "Default Prediction": match["predicted_result"],
        f"{match['home_team']} Win": "home_win",
        "Draw": "draw",
        f"{match['away_team']} Win": "away_win",
    }
    return option_map[selection]


def initialize_scenario_standings(teams, group):
    return {
        team: {
            "group": group,
            "team": team,
            "played": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "points": 0,
        }
        for team in teams
    }


def apply_scenario_match_result(standings, home_team, away_team, result):
    standings[home_team]["played"] += 1
    standings[away_team]["played"] += 1

    if result == "home_win":
        standings[home_team]["wins"] += 1
        standings[home_team]["points"] += 3
        standings[away_team]["losses"] += 1
    elif result == "away_win":
        standings[away_team]["wins"] += 1
        standings[away_team]["points"] += 3
        standings[home_team]["losses"] += 1
    else:
        standings[home_team]["draws"] += 1
        standings[away_team]["draws"] += 1
        standings[home_team]["points"] += 1
        standings[away_team]["points"] += 1


def build_scenario_group_table(group_matches, selected_results, group):
    teams = sorted(set(group_matches["home_team"]).union(group_matches["away_team"]))
    standings = initialize_scenario_standings(teams, group)
    for match in group_matches.itertuples(index=False):
        result = selected_results[getattr(match, "match_id")]
        apply_scenario_match_result(
            standings,
            getattr(match, "home_team"),
            getattr(match, "away_team"),
            result,
        )

    table = pd.DataFrame(standings.values())
    table = table.sort_values(
        ["points", "wins", "team"],
        ascending=[False, False, True],
    ).reset_index(drop=True)
    table.insert(0, "group_position", table.index + 1)
    return table


def build_scenario_comparison_table(scenario_table, baseline_table):
    baseline = baseline_table[
        ["team", "group_position", "points"]
    ].rename(
        columns={
            "group_position": "baseline_position",
            "points": "baseline_points",
        }
    )
    comparison = scenario_table.merge(baseline, on="team", how="left")
    comparison["movement_value"] = (
        comparison["baseline_position"] - comparison["group_position"]
    )
    comparison["movement"] = comparison["movement_value"].map(
        lambda value: "No change"
        if pd.isna(value) or value == 0
        else f"Up {int(value)}"
        if value > 0
        else f"Down {abs(int(value))}"
    )
    comparison["points_delta"] = comparison["points"] - comparison["baseline_points"].fillna(0)
    comparison = add_team_display_columns(comparison, ["team"])
    return comparison[
        [
            "group_position",
            "team_display",
            "played",
            "wins",
            "draws",
            "losses",
            "points",
            "baseline_position",
            "movement",
            "points_delta",
        ]
    ].rename(columns={"team_display": "team"})


def render_scenario_fixture_controls(group_matches):
    selected_results = {}
    for match in group_matches.sort_values(["date", "match_id"]).to_dict("records"):
        home_team = match["home_team"]
        away_team = match["away_team"]
        default_result = match["predicted_result"]
        options = [
            "Default Prediction",
            f"{home_team} Win",
            "Draw",
            f"{away_team} Win",
        ]
        st.markdown(
            f"**{team_label(home_team)} vs {team_label(away_team)}**  \n"
            f"Default: {scenario_result_label(default_result, home_team, away_team)}"
        )
        selection = st.radio(
            "Outcome",
            options,
            horizontal=True,
            label_visibility="collapsed",
            key=f"scenario_{match['group']}_{match['match_id']}",
        )
        selected_results[match["match_id"]] = scenario_result_from_selection(selection, match)
    return selected_results


def render_scenario_simulator(group_predictions, group_tables, simulation_summary):
    render_page_header(
        "Scenario Simulator",
        "Explore temporary group-stage what-if outcomes without changing official forecast files.",
    )
    st.markdown(
        """
        <div class="wc-caveat-strip">
            <strong>This is a local what-if scenario.</strong><br>
            Official model outputs and simulation results have not been changed. Knockout probabilities,
            champion probabilities, metadata, and prediction files are not recalculated or saved.
        </div>
        """,
        unsafe_allow_html=True,
    )

    groups = sorted(group_predictions["group"].dropna().unique())
    selected_group = st.selectbox("Group", groups, key="scenario_group")
    group_matches = group_predictions[group_predictions["group"] == selected_group].copy()
    if group_matches.empty:
        st.info("No fixtures found for the selected group.")
        return

    st.subheader(f"Group {selected_group} Fixtures")
    try:
        fixture_display = build_fixture_display_table(group_matches)
    except ValueError as error:
        st.error(f"Scenario Simulator cannot display fixtures: {error}")
        st.info(
            "Expected probability columns are `prob_home_win`, `prob_draw`, and `prob_away_win`."
        )
        return
    st.dataframe(
        fixture_display,
        use_container_width=True,
        hide_index=True,
        column_config=fixture_probability_column_config(),
    )

    st.subheader("Outcome Controls")
    selected_results = render_scenario_fixture_controls(group_matches)

    baseline_table = group_tables[group_tables["group"] == selected_group].copy()
    scenario_table = build_scenario_group_table(group_matches, selected_results, selected_group)

    st.subheader("Baseline vs Scenario")
    table_columns = st.columns(2)
    with table_columns[0]:
        st.markdown("**Baseline Table**")
        render_group_card(selected_group, baseline_table, simulation_summary)
    with table_columns[1]:
        st.markdown("**Scenario Table**")
        scenario_display = build_scenario_comparison_table(scenario_table, baseline_table)
        st.dataframe(scenario_display, use_container_width=True, hide_index=True)

    st.subheader("Scenario Notes")
    st.markdown(
        """
        <div class="wc-caveat-strip">
            Scenario standings use points, wins, then team name as deterministic ordering because this MVP
            does not simulate scorelines, goal difference, or goals scored. The scenario exists only in the
            current dashboard session and is not written to disk.
        </div>
        """,
        unsafe_allow_html=True,
    )


BUILDER_ROUNDS = [
    "Round of 32",
    "Round of 16",
    "Quarter Finals",
    "Semi Finals",
    "Finals",
    "Third-Place Match",
]


def builder_round_key(round_name):
    return round_name.lower().replace(" ", "_").replace("-", "_")


def builder_winner_key(round_name, match_id):
    return f"builder_winner_{builder_round_key(round_name)}_{match_id}"


def clear_builder_knockout_from(round_name):
    if round_name not in BUILDER_ROUNDS:
        return
    start_index = BUILDER_ROUNDS.index(round_name) + 1
    for downstream_round in BUILDER_ROUNDS[start_index:]:
        prefix = f"builder_winner_{builder_round_key(downstream_round)}_"
        for key in list(st.session_state.keys()):
            if key.startswith(prefix):
                del st.session_state[key]


def clear_builder_knockout_all():
    for round_name in BUILDER_ROUNDS:
        prefix = f"builder_winner_{builder_round_key(round_name)}_"
        for key in list(st.session_state.keys()):
            if key.startswith(prefix):
                del st.session_state[key]


def get_baseline_group_order(group_tables):
    return {
        group: table.sort_values("group_position")["team"].tolist()
        for group, table in group_tables.groupby("group")
    }


def get_group_teams_from_predictions(group_predictions):
    group_teams = {}
    for group, matches in group_predictions.groupby("group"):
        group_teams[group] = sorted(set(matches["home_team"]).union(matches["away_team"]))
    return group_teams


def set_builder_group_defaults(group_tables, group_predictions, mode="baseline"):
    baseline_order = get_baseline_group_order(group_tables)
    alphabetical_order = get_group_teams_from_predictions(group_predictions)
    source = baseline_order if mode == "baseline" else alphabetical_order
    for group, teams in source.items():
        for index, team in enumerate(teams[:4], start=1):
            key = builder_group_key(group, index)
            st.session_state[key] = team
            st.session_state[builder_group_previous_key(group, index)] = team
    clear_builder_knockout_all()


def builder_group_key(group, position):
    return f"builder_group_{group}_{position}"


def builder_group_previous_key(group, position):
    return f"{builder_group_key(group, position)}_previous"


def get_default_builder_order(group, teams, baseline_order):
    ordered = [team for team in baseline_order.get(group, []) if team in teams]
    ordered.extend(team for team in teams if team not in ordered)
    return ordered[:4]


def repair_builder_group_state(group, teams, baseline_order):
    default_order = get_default_builder_order(group, teams, baseline_order)
    used = set()
    repaired = []

    for position in range(1, 5):
        key = builder_group_key(group, position)
        current_team = st.session_state.get(key)
        if current_team not in teams or current_team in used:
            current_team = next(team for team in default_order if team not in used)
            st.session_state[key] = current_team
        used.add(current_team)
        repaired.append(current_team)
        st.session_state[builder_group_previous_key(group, position)] = current_team

    return repaired


def handle_builder_group_selection_change(group, position, teams):
    key = builder_group_key(group, position)
    previous_key = builder_group_previous_key(group, position)
    selected_team = st.session_state.get(key)
    previous_team = st.session_state.get(previous_key)

    if selected_team not in teams:
        return
    if previous_team not in teams:
        previous_team = selected_team

    for other_position in range(1, 5):
        if other_position == position:
            continue
        other_key = builder_group_key(group, other_position)
        if st.session_state.get(other_key) == selected_team:
            st.session_state[other_key] = previous_team
            st.session_state[builder_group_previous_key(group, other_position)] = previous_team
            break

    st.session_state[previous_key] = selected_team
    clear_builder_knockout_all()


def get_builder_group_order(group, group_teams, baseline_order):
    selected = []
    default_order = baseline_order.get(group, group_teams)
    for position in range(1, 5):
        key = builder_group_key(group, position)
        default_team = default_order[position - 1] if len(default_order) >= position else group_teams[position - 1]
        if key not in st.session_state or st.session_state[key] not in group_teams:
            st.session_state[key] = default_team
            st.session_state[builder_group_previous_key(group, position)] = default_team
        selected.append(st.session_state[key])
    return selected


def validate_builder_group_orders(group_orders):
    warnings = []
    for group, teams in group_orders.items():
        if len(teams) != 4 or any(not team for team in teams):
            warnings.append(f"Group {group} must have four placements.")
            continue
        duplicates = sorted({team for team in teams if teams.count(team) > 1})
        if duplicates:
            warnings.append(
                f"Group {group} has duplicate placements: "
                + ", ".join(team_label(team) for team in duplicates)
            )
    return warnings


def build_builder_qualifiers(group_orders, simulation_summary):
    probability_lookup = simulation_summary.set_index("team").to_dict("index")
    group_winners = {}
    group_runners_up = {}
    third_place_candidates = []

    for group, teams in sorted(group_orders.items()):
        group_winners[group] = teams[0]
        group_runners_up[group] = teams[1]
        third_team = teams[2]
        summary = probability_lookup.get(third_team, {})
        third_place_candidates.append(
            {
                "group": group,
                "team": third_team,
                "round_of_32_probability": summary.get("round_of_32_probability", 0),
                "champion_probability": summary.get("champion_probability", 0),
            }
        )

    best_thirds = sorted(
        third_place_candidates,
        key=lambda row: (
            -row["round_of_32_probability"],
            -row["champion_probability"],
            row["team"],
        ),
    )[:8]
    third_place_by_group = {row["group"]: row["team"] for row in best_thirds}

    return {
        "group_winners": group_winners,
        "group_runners_up": group_runners_up,
        "third_place_by_group": third_place_by_group,
        "third_place_candidates": third_place_candidates,
        "round_of_32_teams": (
            list(group_winners.values())
            + list(group_runners_up.values())
            + list(third_place_by_group.values())
        ),
    }


def get_builder_annex_row(qualifiers, annex_allocations):
    if annex_allocations is None:
        raise ValueError(f"Missing Annex C CSV: {ANNEX_C_PATH}")
    third_place_key = "".join(sorted(qualifiers["third_place_by_group"]))
    annex_match = annex_allocations[annex_allocations["qualified_groups"] == third_place_key]
    if annex_match.empty:
        raise ValueError(f"No Annex C row found for third-place groups: {third_place_key}")
    return third_place_key, annex_match.iloc[0].to_dict()


def is_valid_builder_team(team):
    if team is None or pd.isna(team):
        return False
    value = str(team).strip()
    return bool(value) and value.lower() not in {"to be announced", "nan", "none", "tba"}


def make_builder_match(round_name, match_id, home_team=None, away_team=None):
    key = builder_winner_key(round_name, match_id)
    current_winner = st.session_state.get(key)
    if current_winner not in {home_team, away_team}:
        current_winner = None
        if key in st.session_state:
            del st.session_state[key]
    return {
        "round": round_name,
        "match_id": match_id,
        "home_team": home_team,
        "away_team": away_team,
        "winner": current_winner,
        "home_advancement_probability": 1.0 if current_winner == home_team else 0.0,
        "away_advancement_probability": 1.0 if current_winner == away_team else 0.0,
    }


def build_next_builder_round(round_name, previous_matches, expected_matches):
    winners = [match.get("winner") for match in previous_matches]
    matches = []
    for index in range(expected_matches):
        home_team = winners[index * 2] if index * 2 < len(winners) else None
        away_team = winners[index * 2 + 1] if index * 2 + 1 < len(winners) else None
        matches.append(make_builder_match(round_name, index + 1, home_team, away_team))
    return matches


def build_builder_knockout_records(round_of_32_matches):
    round_32 = [
        make_builder_match(
            "Round of 32",
            match["match_id"],
            match.get("home_team"),
            match.get("away_team"),
        )
        for match in round_of_32_matches
    ]
    round_16 = build_next_builder_round("Round of 16", round_32, 8)
    quarters = build_next_builder_round("Quarter Finals", round_16, 4)
    semis = build_next_builder_round("Semi Finals", quarters, 2)
    finals = build_next_builder_round("Finals", semis, 1)

    semi_losers = []
    for match in semis:
        if match.get("winner") == match.get("home_team"):
            semi_losers.append(match.get("away_team"))
        elif match.get("winner") == match.get("away_team"):
            semi_losers.append(match.get("home_team"))
    third_place = []
    if len(semi_losers) == 2 and all(is_valid_builder_team(team) for team in semi_losers):
        third_place = [
            {
                **make_builder_match(
                    "Third-Place Match",
                    1,
                    semi_losers[0],
                    semi_losers[1],
                ),
                "is_third_place": True,
            }
        ]
    else:
        third_place = [
            {
                **make_builder_match("Third-Place Match", 1, None, None),
                "is_third_place": True,
            }
        ]

    return round_32 + round_16 + quarters + semis + finals + third_place


def builder_bracket_card_html(row, selected_team=None):
    winner = row.get("winner")
    home_team = row.get("home_team")
    away_team = row.get("away_team")
    extra_class = " wc-bracket-third-place" if row.get("is_third_place") else ""
    if selected_team and selected_team in {home_team, away_team}:
        extra_class += " wc-bracket-card-highlight"
    meta_label = "Third-Place Match" if row.get("is_third_place") else f"M{row.get('match_id')}"
    home_label = team_label(home_team) if is_valid_builder_team(home_team) else "To be announced"
    away_label = team_label(away_team) if is_valid_builder_team(away_team) else "To be announced"
    result = team_label(winner) if winner else "Winner not selected"
    return textwrap.dedent(f"""
        <div class="wc-bracket-card{extra_class}">
            <div class="wc-bracket-card-meta">{html.escape(meta_label)}</div>
            {bracket_team_row_html(home_label, home_team == winner)}
            {bracket_team_row_html(away_label, away_team == winner)}
            <div class="wc-bracket-advancing-label">Advancing</div>
            <div class="wc-bracket-prob">{html.escape(result)}</div>
        </div>
    """).strip()


def render_builder_winner_controls(records):
    records_by_round = {
        round_name: [record for record in records if record["round"] == round_name]
        for round_name in BUILDER_ROUNDS
    }
    for round_name in BUILDER_ROUNDS:
        matches = records_by_round.get(round_name, [])
        if not matches:
            continue
        with st.expander(round_name, expanded=round_name in {"Round of 32", "Finals"}):
            for match in matches:
                home_team = match.get("home_team")
                away_team = match.get("away_team")
                st.markdown(
                    f"""
                    <div class="wc-builder-match">
                        <div class="wc-builder-match-meta">Match {html.escape(str(match.get("match_id")))}</div>
                        <strong>{html.escape(team_label(home_team) if is_valid_builder_team(home_team) else "To be announced")}</strong>
                        vs
                        <strong>{html.escape(team_label(away_team) if is_valid_builder_team(away_team) else "To be announced")}</strong>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if not (is_valid_builder_team(home_team) and is_valid_builder_team(away_team)):
                    st.caption("Winner selection disabled until both teams are available.")
                    continue
                button_columns = st.columns(2)
                key = builder_winner_key(round_name, match["match_id"])
                with button_columns[0]:
                    if st.button(
                        f"Advance {team_label(home_team)}",
                        key=f"button_{key}_home",
                        type="primary" if st.session_state.get(key) == home_team else "secondary",
                    ):
                        if st.session_state.get(key) != home_team:
                            st.session_state[key] = home_team
                            clear_builder_knockout_from(round_name)
                            st.rerun()
                with button_columns[1]:
                    if st.button(
                        f"Advance {team_label(away_team)}",
                        key=f"button_{key}_away",
                        type="primary" if st.session_state.get(key) == away_team else "secondary",
                    ):
                        if st.session_state.get(key) != away_team:
                            st.session_state[key] = away_team
                            clear_builder_knockout_from(round_name)
                            st.rerun()


def render_builder_group_editor(group_tables, group_predictions):
    baseline_order = get_baseline_group_order(group_tables)
    group_teams = get_group_teams_from_predictions(group_predictions)

    control_columns = st.columns(3)
    with control_columns[0]:
        if st.button("Reset group placements to baseline predicted table"):
            set_builder_group_defaults(group_tables, group_predictions, mode="baseline")
            st.rerun()
    with control_columns[1]:
        if st.button("Load alphabetical group order"):
            set_builder_group_defaults(group_tables, group_predictions, mode="alphabetical")
            st.rerun()
    with control_columns[2]:
        if st.button("Reset everything"):
            set_builder_group_defaults(group_tables, group_predictions, mode="baseline")
            st.rerun()

    group_orders = {}
    groups = sorted(group_teams)
    for row_start in range(0, len(groups), 3):
        columns = st.columns(3)
        for column, group in zip(columns, groups[row_start:row_start + 3]):
            teams = group_teams[group]
            selected = repair_builder_group_state(group, teams, baseline_order)
            with column:
                st.markdown(f"**Group {group}**")
                for position in range(1, 5):
                    key = builder_group_key(group, position)
                    current_index = teams.index(st.session_state[key]) if st.session_state[key] in teams else 0
                    selected_team = st.selectbox(
                        f"{position}{'st' if position == 1 else 'nd' if position == 2 else 'rd' if position == 3 else 'th'}",
                        teams,
                        index=current_index,
                        key=key,
                        format_func=team_label,
                        on_change=handle_builder_group_selection_change,
                        args=(group, position, teams),
                    )
                    selected[position - 1] = selected_team
                group_orders[group] = selected
    return group_orders


def render_builder_third_place_summary(qualifiers, third_place_key):
    candidates = pd.DataFrame(qualifiers["third_place_candidates"]).copy()
    candidates["qualified"] = candidates["group"].isin(qualifiers["third_place_by_group"])
    candidates["team"] = candidates["team"].map(team_label)
    display = format_probability_columns(
        candidates.sort_values(
            ["qualified", "round_of_32_probability", "champion_probability", "team"],
            ascending=[False, False, False, True],
        ),
        ["round_of_32_probability", "champion_probability"],
    )
    st.subheader("Third-Place Qualifier Summary")
    st.caption(
        "Third-place ranking in this builder uses existing model probabilities as a deterministic helper because manual points/goals are not entered."
    )
    st.metric("Annex C key", third_place_key)
    st.dataframe(display, use_container_width=True, hide_index=True)


def render_builder_outcome_summary(records):
    final_record = next((record for record in records if record["round"] == "Finals"), None)
    semi_records = [record for record in records if record["round"] == "Semi Finals"]
    quarter_records = [record for record in records if record["round"] == "Quarter Finals"]
    third_record = next((record for record in records if record["round"] == "Third-Place Match"), None)

    champion = final_record.get("winner") if final_record else None
    runner_up = None
    if final_record and champion:
        runner_up = final_record["away_team"] if champion == final_record["home_team"] else final_record["home_team"]
    semi_finalists = [team for record in semi_records for team in [record.get("home_team"), record.get("away_team")] if is_valid_builder_team(team)]
    quarter_finalists = [team for record in quarter_records for team in [record.get("home_team"), record.get("away_team")] if is_valid_builder_team(team)]
    third_place_team = third_record.get("winner") if third_record else None

    items = [
        ("Champion", team_label(champion) if champion else "Not selected", "Final winner"),
        ("Runner-up", team_label(runner_up) if runner_up else "Not available", "Final loser"),
        ("Third Place", team_label(third_place_team) if third_place_team else "Not selected", "Optional match"),
        ("Semi-finalists", str(len(set(semi_finalists))), "Teams reaching semifinals"),
    ]
    render_storyline_cards(items)

    if quarter_finalists:
        st.markdown(
            "<div class='wc-result-panel-grid'>"
            + result_team_panel_html("Semi-finalists", sorted(set(semi_finalists)))
            + result_team_panel_html("Quarter-finalists", sorted(set(quarter_finalists)))
            + "</div>",
            unsafe_allow_html=True,
        )


def render_tournament_builder(
    group_predictions,
    group_tables,
    simulation_summary,
    annex_allocations,
    fixtures,
):
    render_page_header(
        "Tournament Builder",
        "Build a local 2026 knockout bracket from manual group placements and click winners through to a champion.",
    )
    st.markdown(
        """
        <div class="wc-caveat-strip">
            <strong>This is an interactive local bracket builder.</strong><br>
            It does not change model forecasts, simulations, prediction files, or official outputs. Manual
            placements use Annex C for third-place allocation, while winner picks exist only in this session.
        </div>
        """,
        unsafe_allow_html=True,
    )

    if fixtures is None:
        st.error(f"Missing fixture skeleton: {FIXTURES_PATH}")
        return
    if annex_allocations is None:
        st.error(f"Missing Annex C allocation table: {ANNEX_C_PATH}")
        return

    st.subheader("Group Placement Editor")
    group_orders = render_builder_group_editor(group_tables, group_predictions)
    placement_warnings = validate_builder_group_orders(group_orders)
    if placement_warnings:
        for warning in placement_warnings:
            st.warning(warning)
        st.info("Fix invalid group placements before generating the bracket.")
        return

    qualifiers = build_builder_qualifiers(group_orders, simulation_summary)
    try:
        third_place_key, annex_row = get_builder_annex_row(qualifiers, annex_allocations)
        round_of_32_matches = build_round_of_32_matches(qualifiers, annex_row, fixtures)
    except ValueError as error:
        st.error(str(error))
        return

    render_builder_third_place_summary(qualifiers, third_place_key)

    st.subheader("Generated Round of 32 Bracket")
    records = build_builder_knockout_records(round_of_32_matches)
    left_round_rows, finals, right_round_rows, third_place = split_records_for_bracket(records)
    render_two_sided_bracket(
        left_round_rows,
        finals,
        right_round_rows,
        builder_bracket_card_html,
        third_place=third_place,
    )

    st.subheader("Interactive Knockout Picks")
    control_columns = st.columns(2)
    with control_columns[0]:
        if st.button("Reset knockout selections"):
            clear_builder_knockout_all()
            st.rerun()
    with control_columns[1]:
        st.caption("Changing an earlier winner clears dependent downstream selections.")
    render_builder_winner_controls(records)

    st.subheader("Final Outcome Summary")
    render_builder_outcome_summary(build_builder_knockout_records(round_of_32_matches))


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

    option_map = team_options(simulation_summary["team"].dropna().unique())
    selected_team_label = st.selectbox(
        "Highlight team path",
        ["All", *option_map.keys()],
        key="bracket_highlight_team",
    )
    selected_team = option_map.get(selected_team_label, "All")

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
            selected_team=selected_team,
        )


def team_bucket_html(title, teams_by_label):
    rows = []
    for label, team in teams_by_label:
        rows.append(
            textwrap.dedent(f"""
                <div class="wc-team-pill">
                    <span>{html.escape(team_label(team))}</span>
                    <span class="wc-team-pill-seed">{html.escape(str(label))}</span>
                </div>
            """).strip()
        )
    return textwrap.dedent(f"""
        <div class="wc-team-list-card">
            <div class="wc-team-list-title">{html.escape(title)}</div>
            {''.join(rows)}
        </div>
    """).strip()


def team_chip_grid_html(teams):
    chips = "".join(
        f'<span class="wc-result-chip">{html.escape(team_label(team))}</span>'
        for team in teams
    )
    return f'<div class="wc-result-chip-grid">{chips}</div>'


def result_team_panel_html(title, teams):
    return textwrap.dedent(f"""
        <div class="wc-team-list-card">
            <div class="wc-team-list-title">{html.escape(title)}</div>
            {team_chip_grid_html(teams)}
        </div>
    """).strip()


def render_replay_summary(simulation_result, group_predictions):
    runner_up = get_runner_up(simulation_result)
    third_place = build_replay_third_place_match(simulation_result, group_predictions)
    if third_place is None:
        third_place_team = "Unavailable"
        third_place_sub = "Third-place match unavailable because semi-final loser data is missing."
        third_place_match = "Unavailable"
    else:
        third_place_team = team_label(third_place["winner"])
        third_place_sub = "Deterministic replay winner"
        third_place_match = (
            f"{team_label(third_place['home_team'])} vs "
            f"{team_label(third_place['away_team'])}"
        )
    summary_items = [
        ("Champion", team_label(simulation_result["champion"]), f"Simulation #{simulation_result['simulation_id']}"),
        ("Runner-up", team_label(runner_up) if runner_up else "Unavailable", "Final loser"),
        ("Third-place team", third_place_team, third_place_sub),
        ("Third-place match", third_place_match, "Semi-final losers"),
    ]
    item_html = "".join(
        textwrap.dedent(f"""
            <div class="wc-bracket-summary-item">
                <div class="wc-bracket-summary-label">{html.escape(label)}</div>
                <div class="wc-bracket-summary-value">{html.escape(value)}</div>
                <div class="wc-bracket-summary-sub">{html.escape(sub_value)}</div>
            </div>
        """).strip()
        for label, value, sub_value in summary_items
    )
    st.markdown(
        f'<div class="wc-bracket-summary-grid">{item_html}</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<div class='wc-result-panel-grid'>"
        + result_team_panel_html("Semi-finalists", simulation_result["semi_final_teams"])
        + result_team_panel_html("Quarter-finalists", simulation_result["quarter_final_teams"])
        + "</div>",
        unsafe_allow_html=True,
    )


def render_replay_qualified_teams(simulation_result):
    winners = [
        (f"1{group}", team)
        for group, team in sorted(simulation_result["group_winners"].items())
    ]
    runners_up = [
        (f"2{group}", team)
        for group, team in sorted(simulation_result["group_runners_up"].items())
    ]
    third_place_lookup = {
        row.group: row.team
        for row in simulation_result["group_table"].itertuples(index=False)
        if int(row.group_position) == 3
        and row.team in set(simulation_result["third_place_teams"])
    }
    third_place = [
        (f"3{group}", team)
        for group, team in sorted(third_place_lookup.items())
    ]

    st.markdown(
        "<div class='wc-team-list-grid'>"
        + team_bucket_html("Group Winners", winners)
        + team_bucket_html("Group Runners-up", runners_up)
        + team_bucket_html("Qualified Third-Place Teams", third_place)
        + "</div>",
        unsafe_allow_html=True,
    )


def render_replay_group_tables(simulation_result, simulation_summary):
    groups = sorted(simulation_result["group_table"]["group"].dropna().unique())
    selected_group = st.selectbox(
        "Simulation group table",
        ["All", *groups],
        key="simulation_explorer_group_table",
    )
    render_group_cards(simulation_result["group_table"], simulation_summary, selected_group)


def render_replay_bracket(simulation_result, group_predictions):
    records = build_replay_bracket_records(simulation_result, group_predictions)
    option_map = team_options(simulation_result["round_of_32_teams"])
    selected_team_label = st.selectbox(
        "Highlight replay path",
        ["All", *option_map.keys()],
        key="simulation_explorer_highlight_team",
    )
    selected_team = option_map.get(selected_team_label, "All")
    selected_team_for_highlight = None if selected_team == "All" else selected_team
    st.markdown(
        f'<div class="wc-bracket-path-note">{get_path_status(records, selected_team)}</div>',
        unsafe_allow_html=True,
    )
    left_round_rows, finals, right_round_rows, third_place = split_records_for_bracket(records)
    render_two_sided_bracket(
        left_round_rows,
        finals,
        right_round_rows,
        simulated_bracket_card_html,
        third_place=third_place,
        selected_team=selected_team_for_highlight,
    )


def build_replay_match_results(simulation_result, group_predictions):
    records = build_replay_bracket_records(simulation_result, group_predictions)
    display = pd.DataFrame(records)
    display = add_team_display_columns(display, ["home_team", "away_team"])
    display["winner_display"] = display["winner"].map(
        lambda team: team_label(team) if pd.notna(team) else "Not simulated"
    )
    display = format_probability_columns(
        display[
            [
                "round",
                "match_id",
                "home_team_display",
                "away_team_display",
                "winner_display",
                "home_advancement_probability",
                "away_advancement_probability",
            ]
        ],
        ["home_advancement_probability", "away_advancement_probability"],
    ).rename(
        columns={
            "home_team_display": "home_team",
            "away_team_display": "away_team",
            "winner_display": "winner",
        }
    )
    return display


def render_simulation_explorer(
    group_predictions,
    simulation_summary,
    annex_allocations,
    fixtures,
    metadata,
):
    render_page_header(
        "Simulation Explorer",
        "Replay one seeded tournament path from group tables through the knockout bracket.",
    )

    if annex_allocations is None or fixtures is None:
        st.error("Simulation Explorer requires Annex C allocations and the fixture skeleton.")
        st.info(
            f"Expected inputs: {ANNEX_C_PATH} and {FIXTURES_PATH}. "
            "Run the simulation data pipeline before using this page."
        )
        return

    simulation_count = get_simulation_count(metadata)
    base_seed = get_base_seed(metadata)
    default_simulation_id = min(789, max(0, simulation_count - 1))
    selected_simulation_id = st.number_input(
        "Simulation ID",
        min_value=0,
        max_value=max(0, simulation_count - 1),
        value=default_simulation_id,
        step=1,
    )
    selected_simulation_id = int(selected_simulation_id)
    st.caption(
        f"On-demand deterministic replay. Seed = {base_seed} + {selected_simulation_id}. "
        "No model retraining or full simulation regeneration is performed."
    )

    try:
        simulation_result = replay_single_simulation(
            selected_simulation_id,
            base_seed,
            group_predictions,
            annex_allocations,
            fixtures,
        )
    except Exception as error:
        st.error("Could not replay this simulation.")
        st.exception(error)
        return

    render_replay_summary(simulation_result, group_predictions)

    st.subheader("Qualified Teams")
    render_replay_qualified_teams(simulation_result)

    st.subheader("Group Tables")
    render_replay_group_tables(simulation_result, simulation_summary)

    st.subheader("Knockout Bracket")
    render_replay_bracket(simulation_result, group_predictions)

    st.subheader("Knockout Match Results")
    st.caption(
        "Group-stage sampled match records are not exported by the current replay function; "
        "the group tables above are the available group-stage replay output."
    )
    st.dataframe(
        build_replay_match_results(simulation_result, group_predictions),
        use_container_width=True,
        hide_index=True,
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
    render_dark_bar_chart(
        breakdown.rename_axis("stage").reset_index(name="probability"),
        "stage",
        "probability",
        value_title="Probability",
        height=280,
    )


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
    champion_history = load_champion_history()
    fixtures = load_fixture_skeleton()
    annex_allocations = load_annex_allocations()
    metadata = load_metadata()
    selected_page = render_top_nav()

    if selected_page == "Overview":
        render_overview(simulation_summary, metadata)
    elif selected_page == "Group Predictions":
        render_group_predictions(group_predictions)
    elif selected_page == "Group Tables":
        render_group_tables(group_tables, simulation_summary)
    elif selected_page == "Team Page":
        render_team_page(group_predictions, group_tables, simulation_summary, match_summary)
    elif selected_page == "Team Comparison":
        render_team_comparison_page(group_predictions, group_tables, simulation_summary, match_summary)
    elif selected_page == "Tournament Storylines":
        render_tournament_storylines(
            simulation_summary,
            match_summary,
            group_tables,
            group_predictions,
        )
    elif selected_page == "Champion History":
        render_champion_history(simulation_summary, champion_history)
    elif selected_page == "Scenario Simulator":
        render_scenario_simulator(group_predictions, group_tables, simulation_summary)
    elif selected_page == "Tournament Builder":
        render_tournament_builder(
            group_predictions,
            group_tables,
            simulation_summary,
            annex_allocations,
            fixtures,
        )
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
    elif selected_page == "Simulation Explorer":
        render_simulation_explorer(
            group_predictions,
            simulation_summary,
            annex_allocations,
            fixtures,
            metadata,
        )
    elif selected_page == "Simulation Results":
        render_simulation_results(simulation_summary)
    elif selected_page == "Knockout Summary":
        render_knockout_summary(match_summary)
    elif selected_page == "Methodology":
        render_methodology(metadata)


if __name__ == "__main__":
    main()
