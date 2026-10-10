import streamlit as st
import polars as pl
import numpy as np
import nflreadpy as nfl
import requests
import os
import joblib
import math
from datetime import datetime, timezone, date
from sklearn.ensemble import RandomForestClassifier

# ============================================================
# CONFIG
# ============================================================

st.set_page_config(
    page_title="NFL Game Predictor",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# CONSTANTS
# ============================================================

MODEL_DIR = "data/models"
MODEL_FILE_PATH = os.path.join(MODEL_DIR, "rf_v3_model.joblib")
PREDICTION_HISTORY_PATH = "data/processed/prediction_history.parquet"
INDIVIDUAL_BET_HISTORY_PATH = "data/processed/individual_bet_history.parquet"
ESPN_SCOREBOARD_URL = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"

V3_FEATURES = [
    "win_pct_diff",
    "ppg_diff",
    "defense_diff",
    "rolling_win_pct_diff_5",
    "rolling_ppg_diff_5",
    "rolling_defense_diff_5",
    "rolling_point_diff_diff_5",
    "off_epa_diff",
    "def_epa_diff",
    "rolling_off_epa_diff_5",
    "rolling_def_epa_diff_5",
    "rolling_pass_epa_diff_5",
    "rolling_rush_epa_diff_5",
]

FEATURE_LABELS = {
    "win_pct_diff": "Win Pct Difference",
    "ppg_diff": "Points/Game Difference",
    "defense_diff": "Points Allowed Defense Diff",
    "rolling_win_pct_diff_5": "Rolling 5-Game Win Pct Diff",
    "rolling_ppg_diff_5": "Rolling 5-Game PPG Diff",
    "rolling_defense_diff_5": "Rolling 5-Game PAPG Diff",
    "rolling_point_diff_diff_5": "Rolling 5-Game Net Point Diff",
    "off_epa_diff": "Offensive EPA/Play Diff",
    "def_epa_diff": "Defensive EPA/Play Allowed Diff",
    "rolling_off_epa_diff_5": "Rolling Offensive EPA Diff",
    "rolling_def_epa_diff_5": "Rolling Defensive EPA Diff",
    "rolling_pass_epa_diff_5": "Rolling Passing EPA Diff",
    "rolling_rush_epa_diff_5": "Rolling Rushing EPA Diff",
}

TEAM_NAMES = {
    "ARI": "Arizona Cardinals", "ATL": "Atlanta Falcons", "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills", "CAR": "Carolina Panthers", "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals", "CLE": "Cleveland Browns", "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos", "DET": "Detroit Lions", "GB": "Green Bay Packers",
    "HOU": "Houston Texans", "IND": "Indianapolis Colts", "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs", "LV": "Las Vegas Raiders", "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams", "MIA": "Miami Dolphins", "MIN": "Minnesota Vikings",
    "NE": "New England Patriots", "NO": "New Orleans Saints", "NYG": "New York Giants",
    "NYJ": "New York Jets", "PHI": "Philadelphia Eagles", "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers", "SEA": "Seattle Seahawks", "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans", "WAS": "Washington Commanders",
}

ESPN_TEAM_ALIASES = {
    "JAC": "JAX",
    "WSH": "WAS",
    "LA": "LAR",
}

# ============================================================
# GLOBAL STYLE
# ============================================================

st.markdown(
    """
    <style>
    /* ========================================================
       APP
    ======================================================== */
    .stApp {
        background: #0b1120;
        color: #e5e7eb;
    }
    .main .block-container {
        max-width: 1400px;
        padding-top: 1.8rem;
        padding-bottom: 4rem;
    }
    #MainMenu {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    header {
        background: transparent !important;
    }
    /* ========================================================
       SIDEBAR
    ======================================================== */
    section[data-testid="stSidebar"] {
        background: #080d19;
        border-right: 1px solid #1e293b;
    }
    section[data-testid="stSidebar"] > div {
        padding-top: 1.8rem;
    }
    section[data-testid="stSidebar"] .stMarkdown {
        color: #cbd5e1;
    }
    /* ========================================================
       HERO
    ======================================================== */
    .hero {
        position: relative;
        overflow: hidden;
        background:
            radial-gradient(
                circle at 85% 20%,
                rgba(37, 99, 235, 0.24),
                transparent 32%
            ),
            linear-gradient(
                135deg,
                #111827 0%,
                #0f172a 60%,
                #0b1120 100%
            );
        border: 1px solid #263244;
        border-radius: 18px;
        padding: 2.2rem 2.4rem;
        margin-bottom: 1.4rem;
    }
    .hero::after {
        content: "";
        position: absolute;
        right: -80px;
        bottom: -120px;
        width: 300px;
        height: 300px;
        border-radius: 50%;
        border: 1px solid rgba(59, 130, 246, 0.12);
    }
    .hero-kicker {
        color: #60a5fa;
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        color: #f8fafc;
        font-size: 2.65rem;
        line-height: 1;
        font-weight: 850;
        letter-spacing: -0.04em;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.92rem;
        margin-top: 0.8rem;
        max-width: 680px;
        line-height: 1.55;
    }
    .hero-badge {
        display: inline-block;
        margin-top: 1.25rem;
        padding: 0.4rem 0.7rem;
        background: rgba(30, 41, 59, 0.8);
        border: 1px solid #334155;
        border-radius: 7px;
        color: #cbd5e1;
        font-size: 0.68rem;
        font-weight: 750;
        letter-spacing: 0.04em;
    }
    /* ========================================================
       SECTION HEADER
    ======================================================== */
    .section-header {
        display: flex;
        justify-content: space-between;
        align-items: end;
        margin-top: 1.8rem;
        margin-bottom: 0.85rem;
    }
    .section-title {
        color: #f8fafc;
        font-size: 1.15rem;
        font-weight: 800;
        letter-spacing: -0.02em;
    }
    .section-subtitle {
        color: #64748b;
        font-size: 0.76rem;
        margin-top: 0.2rem;
    }
    /* ========================================================
       METRICS
    ======================================================== */
    .metric-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 1rem 1.1rem;
        min-height: 90px;
    }
    .metric-label {
        color: #64748b;
        font-size: 0.63rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .metric-value {
        color: #f8fafc;
        font-size: 1.45rem;
        font-weight: 850;
        margin-top: 0.3rem;
    }
    .metric-note {
        color: #475569;
        font-size: 0.68rem;
        margin-top: 0.15rem;
    }
    /* ========================================================
       GAME CARD
    ======================================================== */
    .game-card {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        margin-bottom: 0.9rem;
        overflow: hidden;
    }
    .game-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.75rem 1.1rem;
        border-bottom: 1px solid #1e293b;
        background: #0f172a;
    }
    .game-time {
        color: #64748b;
        font-size: 0.7rem;
        font-weight: 750;
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }
    .confidence-label {
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 750;
    }
    .confidence-high {
        color: #4ade80;
    }
    .confidence-medium {
        color: #fbbf24;
    }
    .confidence-low {
        color: #94a3b8;
    }
    /* ========================================================
       MATCHUP
    ======================================================== */
    .matchup {
        display: grid;
        grid-template-columns: 1fr 70px 1fr;
        align-items: center;
        padding: 1.4rem 1.3rem 1rem;
    }
    .team {
        display: flex;
        align-items: center;
        gap: 0.9rem;
    }
    .team-right {
        justify-content: flex-end;
        text-align: right;
    }
    .team-mark {
        display: flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 10px;
        background: #1e293b;
        border: 1px solid #334155;
        color: #f8fafc;
        font-size: 0.85rem;
        font-weight: 850;
    }
    .team-code {
        color: #f8fafc;
        font-size: 1.2rem;
        font-weight: 850;
        letter-spacing: -0.02em;
    }
    .team-name {
        color: #64748b;
        font-size: 0.68rem;
        margin-top: 0.15rem;
    }
    .team-record {
        color: #94a3b8;
        font-size: 0.7rem;
        font-weight: 700;
        margin-top: 0.25rem;
    }
    .versus {
        text-align: center;
        color: #475569;
        font-size: 0.7rem;
        font-weight: 800;
    }
    /* ========================================================
       PROBABILITY
    ======================================================== */
    .probability-section {
        padding: 0 1.3rem 1.15rem;
    }
    .probability-row {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-top: 0.6rem;
        margin-bottom: 0.25rem;
    }
    .probability-team {
        color: #64748b;
        font-size: 0.68rem;
        font-weight: 750;
    }
    .probability-value {
        color: #e2e8f0;
        font-size: 0.75rem;
        font-weight: 850;
    }
    .bar {
        width: 100%;
        height: 6px;
        background: #1e293b;
        border-radius: 999px;
        overflow: hidden;
    }
    .bar-fill-away {
        height: 100%;
        background: #475569;
        border-radius: 999px;
    }
    .bar-fill-home {
        height: 100%;
        background: #3b82f6;
        border-radius: 999px;
    }
    .bar-fill-winner {
        height: 100%;
        background: #22c55e;
        border-radius: 999px;
    }
    /* ========================================================
       CARD BOTTOM
    ======================================================== */
    .card-bottom {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        border-top: 1px solid #1e293b;
    }
    .card-stat {
        padding: 0.7rem 0.8rem;
        border-right: 1px solid #1e293b;
        text-align: center;
    }
    .card-stat:last-child {
        border-right: none;
    }
    .card-stat-label {
        color: #475569;
        font-size: 0.58rem;
        font-weight: 800;
        letter-spacing: 0.07em;
        text-transform: uppercase;
    }
    .card-stat-value {
        color: #cbd5e1;
        font-size: 0.76rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }
    /* ========================================================
       PREDICTION STRIP
    ======================================================== */
    .prediction-strip {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.85rem 1.1rem;
        background: #0f172a;
        border-top: 1px solid #1e293b;
    }
    .prediction-caption {
        color: #475569;
        font-size: 0.6rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
    }
    .prediction-name {
        color: #f8fafc;
        font-size: 0.85rem;
        font-weight: 850;
        margin-top: 0.1rem;
    }
    .prediction-probability {
        color: #4ade80;
        font-size: 1rem;
        font-weight: 900;
    }
    /* ========================================================
       EXPANDER & CONTROLS
    ======================================================== */
    div[data-testid="stExpander"] {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        margin-bottom: 0.9rem;
    }
    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
    }
    .sidebar-brand {
        color: #f8fafc;
        font-size: 1rem;
        font-weight: 850;
    }
    .sidebar-version {
        color: #475569;
        font-size: 0.67rem;
        margin-top: 0.2rem;
    }
    .sidebar-section {
        color: #64748b;
        font-size: 0.62rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-top: 1.3rem;
        margin-bottom: 0.5rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] {
        gap: 0.18rem;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label {
        padding: 0.42rem 0.55rem;
        border-radius: 0.55rem;
    }
    .model-panel {
        background: #111827;
        border: 1px solid #1e293b;
        border-radius: 14px;
        padding: 1.2rem;
        margin-top: 1.8rem;
    }
    .model-title {
        color: #f8fafc;
        font-size: 0.95rem;
        font-weight: 850;
    }
    .model-text {
        color: #64748b;
        font-size: 0.75rem;
        line-height: 1.6;
        margin-top: 0.55rem;
    }
    .stButton > button {
        border-radius: 8px;
        border: 1px solid #263244;
        background: #111827;
        color: #cbd5e1;
        font-weight: 700;
    }
    .stButton > button:hover {
        border-color: #3b82f6;
        color: white;
    }
    div[data-baseweb="select"] > div {
        background: #111827;
        border-color: #263244;
        color: #e2e8f0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# DATA & MODEL PERSISTENCE
# ============================================================

@st.cache_data
def load_historical_data():
    return pl.read_parquet("data/processed/model_data.parquet")

@st.cache_resource
def get_or_train_model():
    if os.path.exists(MODEL_FILE_PATH):
        try:
            return joblib.load(MODEL_FILE_PATH)
        except Exception:
            pass

    data = load_historical_data()
    historical = data.drop_nulls(subset=V3_FEATURES)
    X = historical.select(V3_FEATURES).to_numpy()
    y = historical.select("home_win").to_numpy().ravel()
    
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y)
    
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_FILE_PATH)
    return model

@st.cache_data(ttl=3600)
def load_2026_data():
    schedule = nfl.load_schedules(seasons=[2026])
    schedule = schedule.filter(pl.col("game_type") == "REG")
    pbp = nfl.load_pbp(seasons=[2026])
    pbp = pbp.filter(pl.col("season_type") == "REG")
    return schedule, pbp

model_data = load_historical_data()
final_rf = get_or_train_model()
schedule_2026, pbp_2026 = load_2026_data()

# ============================================================
# TEAM GAME FEATURES
# ============================================================

def build_team_games(games):
    home = games.select([
        "game_id",
        "season",
        "week",
        pl.col("home_team").alias("team"),
        pl.col("home_score").alias("points_for"),
        pl.col("away_score").alias("points_against"),
        (pl.col("result") > 0).cast(pl.Int8).alias("win"),
    ])
    away = games.select([
        "game_id",
        "season",
        "week",
        pl.col("away_team").alias("team"),
        pl.col("away_score").alias("points_for"),
        pl.col("home_score").alias("points_against"),
        (pl.col("result") < 0).cast(pl.Int8).alias("win"),
    ])
    return pl.concat([home, away]).sort(["team", "week"])

def build_current_team_features(schedule):
    completed = schedule.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    team_games = build_team_games(completed)
    season_stats = (
        team_games
        .group_by("team")
        .agg([
            pl.col("win").sum().alias("wins"),
            pl.len().alias("games"),
            pl.col("points_for").sum().alias("points_for"),
            pl.col("points_against").sum().alias("points_against"),
        ])
        .with_columns([
            (pl.col("wins") / pl.col("games")).alias("win_pct"),
            (pl.col("points_for") / pl.col("games")).alias("ppg"),
            (pl.col("points_against") / pl.col("games")).alias("papg"),
        ])
    )
    rolling = (
        team_games
        .sort(["team", "week"])
        .group_by("team")
        .agg([
            pl.col("win").sort_by("week").tail(5).mean().alias("rolling_win_pct_5"),
            pl.col("points_for").sort_by("week").tail(5).mean().alias("rolling_ppg_5"),
            pl.col("points_against").sort_by("week").tail(5).mean().alias("rolling_papg_5"),
            (pl.col("points_for") - pl.col("points_against")).sort_by("week").tail(5).mean().alias("rolling_point_diff_5"),
        ])
    )
    return season_stats.join(rolling, on="team", how="left")

current_team_features = build_current_team_features(schedule_2026)

# ============================================================
# EPA
# ============================================================

def build_current_epa_features(schedule, pbp):
    completed = schedule.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )
    completed_ids = completed.select("game_id").unique()
    pbp_completed = pbp.join(completed_ids, on="game_id", how="inner")
    plays = pbp_completed.filter(pl.col("posteam").is_not_null())
    offense = (
        plays
        .group_by(["game_id", "week", "posteam"])
        .agg([
            pl.col("epa").mean().alias("off_epa_per_play"),
            pl.when(pl.col("pass_attempt") == 1).then(pl.col("epa")).otherwise(None).mean().alias("pass_epa_per_play"),
            pl.when(pl.col("rush_attempt") == 1).then(pl.col("epa")).otherwise(None).mean().alias("rush_epa_per_play"),
        ])
        .rename({"posteam": "team"})
    )
    defense = (
        plays
        .group_by(["game_id", "week", "defteam"])
        .agg([
            pl.col("epa").mean().alias("def_epa_allowed_per_play"),
        ])
        .rename({"defteam": "team"})
    )
    combined = (
        offense
        .join(defense, on=["game_id", "week", "team"], how="inner")
        .sort(["team", "week"])
    )
    return (
        combined
        .group_by("team")
        .agg([
            pl.col("off_epa_per_play").mean().alias("off_epa_per_play"),
            pl.col("def_epa_allowed_per_play").mean().alias("def_epa_allowed_per_play"),
            pl.col("off_epa_per_play").sort_by("week").tail(5).mean().alias("rolling_off_epa_per_play_5"),
            pl.col("def_epa_allowed_per_play").sort_by("week").tail(5).mean().alias("rolling_def_epa_allowed_per_play_5"),
            pl.col("pass_epa_per_play").sort_by("week").tail(5).mean().alias("rolling_pass_epa_per_play_5"),
            pl.col("rush_epa_per_play").sort_by("week").tail(5).mean().alias("rolling_rush_epa_per_play_5"),
        ])
    )

current_epa = build_current_epa_features(schedule_2026, pbp_2026)

# ============================================================
# PREDICTION DATASET
# ============================================================

def build_prediction_dataset():
    upcoming = schedule_2026.filter(
        pl.col("home_score").is_null()
        | pl.col("away_score").is_null()
    )
    home = current_team_features.rename({
        "team": "home_team",
        "win_pct": "home_win_pct",
        "ppg": "home_ppg",
        "papg": "home_papg",
        "rolling_win_pct_5": "home_rolling_win_pct_5",
        "rolling_ppg_5": "home_rolling_ppg_5",
        "rolling_papg_5": "home_rolling_papg_5",
        "rolling_point_diff_5": "home_rolling_point_diff_5",
    })
    away = current_team_features.rename({
        "team": "away_team",
        "win_pct": "away_win_pct",
        "ppg": "away_ppg",
        "papg": "away_papg",
        "rolling_win_pct_5": "away_rolling_win_pct_5",
        "rolling_ppg_5": "away_rolling_ppg_5",
        "rolling_papg_5": "away_rolling_papg_5",
        "rolling_point_diff_5": "away_rolling_point_diff_5",
    })
    home_epa = current_epa.rename({
        "team": "home_team",
        "off_epa_per_play": "home_off_epa",
        "def_epa_allowed_per_play": "home_def_epa",
        "rolling_off_epa_per_play_5": "home_rolling_off_epa_5",
        "rolling_def_epa_allowed_per_play_5": "home_rolling_def_epa_5",
        "rolling_pass_epa_per_play_5": "home_rolling_pass_epa_5",
        "rolling_rush_epa_per_play_5": "home_rolling_rush_epa_5",
    })
    away_epa = current_epa.rename({
        "team": "away_team",
        "off_epa_per_play": "away_off_epa",
        "def_epa_allowed_per_play": "away_def_epa",
        "rolling_off_epa_per_play_5": "away_rolling_off_epa_5",
        "rolling_def_epa_allowed_per_play_5": "away_rolling_def_epa_5",
        "rolling_pass_epa_per_play_5": "away_rolling_pass_epa_5",
        "rolling_rush_epa_per_play_5": "away_rolling_rush_epa_5",
    })
    data = (
        upcoming
        .join(home, on="home_team", how="left")
        .join(away, on="away_team", how="left")
        .join(home_epa, on="home_team", how="left")
        .join(away_epa, on="away_team", how="left")
    )
    return data.with_columns([
        (pl.col("home_win_pct") - pl.col("away_win_pct")).alias("win_pct_diff"),
        (pl.col("home_ppg") - pl.col("away_ppg")).alias("ppg_diff"),
        (pl.col("away_papg") - pl.col("home_papg")).alias("defense_diff"),
        (pl.col("home_rolling_win_pct_5") - pl.col("away_rolling_win_pct_5")).alias("rolling_win_pct_diff_5"),
        (pl.col("home_rolling_ppg_5") - pl.col("away_rolling_ppg_5")).alias("rolling_ppg_diff_5"),
        (pl.col("away_rolling_papg_5") - pl.col("home_rolling_papg_5")).alias("rolling_defense_diff_5"),
        (pl.col("home_rolling_point_diff_5") - pl.col("away_rolling_point_diff_5")).alias("rolling_point_diff_diff_5"),
        (pl.col("home_off_epa") - pl.col("away_off_epa")).alias("off_epa_diff"),
        (pl.col("away_def_epa") - pl.col("home_def_epa")).alias("def_epa_diff"),
        (pl.col("home_rolling_off_epa_5") - pl.col("away_rolling_off_epa_5")).alias("rolling_off_epa_diff_5"),
        (pl.col("away_rolling_def_epa_5") - pl.col("home_rolling_def_epa_5")).alias("rolling_def_epa_diff_5"),
        (pl.col("home_rolling_pass_epa_5") - pl.col("away_rolling_pass_epa_5")).alias("rolling_pass_epa_diff_5"),
        (pl.col("home_rolling_rush_epa_5") - pl.col("away_rolling_rush_epa_5")).alias("rolling_rush_epa_diff_5"),
    ])

prediction_dataset = build_prediction_dataset()

# ============================================================
# PREDICT
# ============================================================

valid = prediction_dataset.drop_nulls(subset=V3_FEATURES)

if not valid.is_empty():
    X = valid.select(V3_FEATURES).to_numpy()
    probabilities = final_rf.predict_proba(X)[:, 1]

    metadata_cols = [
        "game_id",
        "week",
        "gameday",
        "gametime",
        "away_team",
        "home_team",
        "home_win_pct",
        "away_win_pct",
        "home_ppg",
        "away_ppg",
        "home_papg",
        "away_papg",
    ]
    # Deduplicate column names to prevent Polars DuplicateError
    select_cols = list(dict.fromkeys(metadata_cols + V3_FEATURES))

    predictions = (
        valid
        .select(select_cols)
        .with_columns([
            pl.Series("home_win_probability", probabilities)
        ])
        .with_columns([
            (1 - pl.col("home_win_probability")).alias("away_win_probability"),
            pl.when(pl.col("home_win_probability") >= 0.5)
            .then(pl.col("home_team"))
            .otherwise(pl.col("away_team"))
            .alias("predicted_winner"),
        ])
    )
else:
    predictions = pl.DataFrame()

# ============================================================
# PERSISTENT PREDICTION HISTORY
# ============================================================

def load_saved_prediction_history():
    if not os.path.exists(PREDICTION_HISTORY_PATH):
        return pl.DataFrame()
    try:
        return pl.read_parquet(PREDICTION_HISTORY_PATH)
    except (OSError, pl.exceptions.PolarsError) as exc:
        st.warning(f"Could not read saved prediction history: {exc}")
        return pl.DataFrame()

@st.cache_data(show_spinner=False)
def save_new_prediction_snapshots(current_predictions):
    if current_predictions.is_empty():
        return load_saved_prediction_history()

    os.makedirs(os.path.dirname(PREDICTION_HISTORY_PATH), exist_ok=True)
    history = load_saved_prediction_history()

    current = current_predictions.with_columns([
        pl.lit(2026).cast(pl.Int64).alias("season"),
        pl.lit(datetime.now(timezone.utc).isoformat()).alias("prediction_saved_at_utc"),
    ])

    if not history.is_empty():
        if "game_id" in history.columns and "game_id" in current.columns:
            existing_ids = history.select("game_id").drop_nulls().unique()
            new_rows = current.join(existing_ids, on="game_id", how="anti")
        else:
            key_cols = [c for c in ("season", "week", "away_team", "home_team") if c in history.columns and c in current.columns]
            if key_cols:
                existing_keys = history.select(key_cols).unique()
                new_rows = current.join(existing_keys, on=key_cols, how="anti")
            else:
                new_rows = current
    else:
        new_rows = current

    if not new_rows.is_empty():
        combined = (
            pl.concat([history, new_rows], how="diagonal_relaxed")
            if not history.is_empty()
            else new_rows
        )
        combined.write_parquet(PREDICTION_HISTORY_PATH)
        history = combined

    return history

prediction_history = save_new_prediction_snapshots(predictions)

# ============================================================
# ESPN MONEYLINES & VALUE CALCULATIONS
# ============================================================

def normalize_team_code(code):
    code = str(code or "").upper().strip()
    return ESPN_TEAM_ALIASES.get(code, code)

def parse_american_odds(value):
    if isinstance(value, dict):
        for key in ("current", "moneyLine", "moneyline", "odds", "value"):
            if key in value:
                parsed = parse_american_odds(value[key])
                if parsed is not None:
                    return parsed
        return None

    if value is None or isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        if value == 0:
            return None
        return float(value)

    raw = str(value).strip().replace("−", "-").replace("+", "")
    if raw.upper() in {"EVEN", "EV", "PK"}:
        return 100.0 if raw.upper() in {"EVEN", "EV"} else None

    try:
        number = float(raw)
    except (TypeError, ValueError):
        return None

    return number if number != 0 else None

def spread_to_moneyline(spread: float):
    if spread == 0:
        return -110.0, -110.0
    prob_home = 1.0 / (1.0 + math.exp(-0.155 * (-spread)))
    if prob_home >= 0.5:
        home_ml = - (prob_home / (1.0 - prob_home)) * 100.0
        away_ml = ((1.0 - prob_home) / prob_home) * 100.0
    else:
        home_ml = (prob_home / (1.0 - prob_home)) * 100.0
        away_ml = - ((1.0 - prob_home) / prob_home) * 100.0
    return round(away_ml), round(home_ml)

def extract_moneyline_pair(competition):
    markets = competition.get("odds") or []
    if isinstance(markets, dict):
        markets = [markets]

    for market in markets:
        if not isinstance(market, dict):
            continue

        provider = market.get("provider") or {}
        provider_name = (
            provider.get("name", "ESPN odds")
            if isinstance(provider, dict)
            else str(provider)
        )

        away_ml = market.get("awayMoneyLine") or market.get("awayMoneyline")
        home_ml = market.get("homeMoneyLine") or market.get("homeMoneyline")

        if away_ml is None or home_ml is None:
            a_obj = market.get("awayTeamOdds") or market.get("away_team_odds") or {}
            h_obj = market.get("homeTeamOdds") or market.get("home_team_odds") or {}
            if isinstance(a_obj, dict):
                away_ml = a_obj.get("moneyLine") or a_obj.get("moneyline") or a_obj.get("value")
            if isinstance(h_obj, dict):
                home_ml = h_obj.get("moneyLine") or h_obj.get("moneyline") or h_obj.get("value")

        if away_ml is None or home_ml is None:
            ml_block = market.get("moneyline") or market.get("moneyLine") or {}
            if isinstance(ml_block, dict):
                away_ml = ml_block.get("away") or ml_block.get("awayTeam")
                home_ml = ml_block.get("home") or ml_block.get("homeTeam")
                if isinstance(away_ml, dict):
                    away_ml = away_ml.get("odds") or away_ml.get("value")
                if isinstance(home_ml, dict):
                    home_ml = home_ml.get("odds") or home_ml.get("value")

        parsed_away = parse_american_odds(away_ml)
        parsed_home = parse_american_odds(home_ml)

        if parsed_away is not None and parsed_home is not None:
            return parsed_away, parsed_home, provider_name

        spread_val = market.get("spread")
        if spread_val is not None:
            try:
                spread_num = float(spread_val)
                est_away, est_home = spread_to_moneyline(spread_num)
                return est_away, est_home, f"{provider_name} (Spread Implied)"
            except (ValueError, TypeError):
                pass

    return None, None, None

@st.cache_data(ttl=300, show_spinner=False)
def fetch_espn_moneylines(season, week):
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params={
            "season": int(season),
            "seasontype": 2,
            "week": int(week),
            "region": "us",
            "lang": "en",
        },
        timeout=12,
        headers={"User-Agent": "Mozilla/5.0 NFL-Game-Predictor"},
    )
    response.raise_for_status()
    payload = response.json()

    rows = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors") or []
        home_team = None
        away_team = None

        for competitor in competitors:
            side = competitor.get("homeAway")
            team = competitor.get("team") or {}
            abbreviation = normalize_team_code(team.get("abbreviation"))
            if side == "home":
                home_team = abbreviation
            elif side == "away":
                away_team = abbreviation

        if not home_team or not away_team:
            continue

        away_ml, home_ml, provider = extract_moneyline_pair(competition)
        if away_ml is None or home_ml is None:
            event_away, event_home, event_provider = extract_moneyline_pair(event)
            away_ml = away_ml if away_ml is not None else event_away
            home_ml = home_ml if home_ml is not None else event_home
            provider = provider or event_provider

        rows.append({
            "away_team": away_team,
            "home_team": home_team,
            "away_moneyline": away_ml,
            "home_moneyline": home_ml,
            "odds_provider": provider or "ESPN",
            "odds_updated": event.get("date", ""),
            "odds_status": "Available" if away_ml is not None and home_ml is not None else "Odds unavailable",
        })

    schema = {
        "away_team": pl.String,
        "home_team": pl.String,
        "away_moneyline": pl.Float64,
        "home_moneyline": pl.Float64,
        "odds_provider": pl.String,
        "odds_updated": pl.String,
        "odds_status": pl.String,
    }
    if not rows:
        return pl.DataFrame(schema=schema)

    return pl.DataFrame(rows, schema=schema, strict=False).unique(
        subset=["away_team", "home_team"], keep="first"
    )

@st.cache_data(ttl=300, show_spinner=False)
def fetch_espn_results(season, week):
    response = requests.get(
        ESPN_SCOREBOARD_URL,
        params={
            "season": int(season),
            "seasontype": 2,
            "week": int(week),
            "region": "us",
            "lang": "en",
        },
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 NFL-Game-Predictor"},
    )
    response.raise_for_status()
    payload = response.json()

    rows = []
    for event in payload.get("events", []):
        competitions = event.get("competitions") or []
        if not competitions:
            continue

        competition = competitions[0]
        competitors = competition.get("competitors") or []
        away_team = home_team = None
        away_score = home_score = None

        for competitor in competitors:
            side = competitor.get("homeAway")
            team = competitor.get("team") or {}
            code = normalize_team_code(team.get("abbreviation"))
            score = competitor.get("score")
            try:
                score = float(score) if score is not None else None
            except (TypeError, ValueError):
                score = None

            if side == "away":
                away_team, away_score = code, score
            elif side == "home":
                home_team, home_score = code, score

        if not away_team or not home_team:
            continue

        status_obj = (competition.get("status") or event.get("status") or {})
        status_type = status_obj.get("type") or {}
        status_state = str(status_type.get("state") or "").lower()
        status_name = status_type.get("name") or status_type.get("description") or "Scheduled"
        is_final = bool(status_type.get("completed")) or status_state == "post"

        actual_winner = None
        if is_final and away_score is not None and home_score is not None:
            if away_score > home_score:
                actual_winner = away_team
            elif home_score > away_score:
                actual_winner = home_team
            else:
                actual_winner = "TIE"

        rows.append({
            "away_team": away_team,
            "home_team": home_team,
            "away_score": away_score,
            "home_score": home_score,
            "actual_winner": actual_winner,
            "game_status": "Final" if is_final else str(status_name),
            "is_final": is_final,
            "gameday_actual": event.get("date", ""),
        })

    schema = {
        "away_team": pl.String,
        "home_team": pl.String,
        "away_score": pl.Float64,
        "home_score": pl.Float64,
        "actual_winner": pl.String,
        "game_status": pl.String,
        "is_final": pl.Boolean,
        "gameday_actual": pl.String,
    }
    if not rows:
        return pl.DataFrame(schema=schema)
    return pl.DataFrame(rows, schema=schema, strict=False).unique(
        subset=["away_team", "home_team"], keep="first"
    )

def american_implied_probability(odds):
    if odds is None:
        return None
    odds = float(odds)
    if odds > 0:
        return 100.0 / (odds + 100.0)
    if odds < 0:
        return abs(odds) / (abs(odds) + 100.0)
    return None

def american_profit_per_unit(odds):
    if odds is None:
        return None
    odds = float(odds)
    if odds > 0:
        return odds / 100.0
    if odds < 0:
        return 100.0 / abs(odds)
    return None

def update_individual_bet_history(week_predictions, odds_frame, actual_results, season, week):
    try:
        history = pl.read_parquet(INDIVIDUAL_BET_HISTORY_PATH) if os.path.exists(INDIVIDUAL_BET_HISTORY_PATH) else pl.DataFrame()
    except (OSError, pl.exceptions.PolarsError) as exc:
        st.warning(f"Could not read individual-bet history: {exc}")
        history = pl.DataFrame()

    if not week_predictions.is_empty() and not odds_frame.is_empty() and not actual_results.is_empty():
        candidates = (
            week_predictions.filter(pl.col("week") == int(week))
            .join(odds_frame, on=["away_team", "home_team"], how="left")
            .join(actual_results.select(["away_team", "home_team", "game_status", "is_final"]),
                  on=["away_team", "home_team"], how="left")
        )
        existing_keys = set()
        if not history.is_empty():
            existing_keys = {
                (int(r["season"]), int(r["week"]), r["away_team"], r["home_team"])
                for r in history.iter_rows(named=True)
            }
        new_rows = []
        captured_at = datetime.now(timezone.utc).isoformat()
        for row in candidates.iter_rows(named=True):
            away_ml, home_ml = row.get("away_moneyline"), row.get("home_moneyline")
            status = str(row.get("game_status") or "").lower()
            if bool(row.get("is_final")) or status not in {"scheduled", "pre-game", "pregame", "warmup"}:
                continue
            if away_ml is None or home_ml is None:
                continue
            key = (int(season), int(week), row["away_team"], row["home_team"])
            if key in existing_keys:
                continue
            pick = row["home_team"] if float(row["home_win_probability"]) >= 0.5 else row["away_team"]
            price = home_ml if pick == row["home_team"] else away_ml
            new_rows.append({
                "season": int(season), "week": int(week), "away_team": row["away_team"],
                "home_team": row["home_team"], "model_pick": pick, "pick_moneyline": float(price),
                "odds_provider": row.get("odds_provider") or "ESPN", "odds_captured_at_utc": captured_at,
                "gameday": str(row.get("gameday") or ""), "actual_winner": None,
                "bet_result": "PENDING", "net_units": None, "settled_at_utc": None,
            })
            existing_keys.add(key)
        if new_rows:
            added = pl.DataFrame(new_rows, strict=False)
            history = pl.concat([history, added], how="diagonal_relaxed") if not history.is_empty() else added

    if not history.is_empty() and not actual_results.is_empty():
        result_map = {(r["away_team"], r["home_team"]): r for r in actual_results.iter_rows(named=True)}
        updated_rows = []
        now = datetime.now(timezone.utc).isoformat()
        for row in history.iter_rows(named=True):
            if int(row.get("season") or 0) == int(season) and int(row.get("week") or 0) == int(week):
                result = result_map.get((row.get("away_team"), row.get("home_team")))
                if result and result.get("is_final"):
                    winner = result.get("actual_winner")
                    if winner == "TIE":
                        row["bet_result"], row["net_units"] = "PUSH", 0.0
                    elif winner == row.get("model_pick"):
                        row["bet_result"] = "WON"
                        row["net_units"] = american_profit_per_unit(row.get("pick_moneyline"))
                    else:
                        row["bet_result"], row["net_units"] = "LOST", -1.0
                    row["actual_winner"] = winner
                    row["settled_at_utc"] = row.get("settled_at_utc") or now
            updated_rows.append(row)
        history = pl.DataFrame(updated_rows, strict=False)

    if not history.is_empty():
        os.makedirs(os.path.dirname(INDIVIDUAL_BET_HISTORY_PATH), exist_ok=True)
        history.write_parquet(INDIVIDUAL_BET_HISTORY_PATH)
    return history

def build_value_table(week_predictions, odds_frame):
    if week_predictions.is_empty():
        return pl.DataFrame()

    if odds_frame.is_empty():
        games = week_predictions.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("away_moneyline"),
            pl.lit(None, dtype=pl.Float64).alias("home_moneyline"),
            pl.lit("ESPN").alias("odds_provider"),
            pl.lit("Odds unavailable").alias("odds_status"),
        )
    else:
        games = week_predictions.join(
            odds_frame,
            on=["away_team", "home_team"],
            how="left",
        ).with_columns(
            pl.when(pl.col("away_moneyline").is_not_null() & pl.col("home_moneyline").is_not_null())
            .then(pl.lit("Available"))
            .otherwise(pl.lit("Odds unavailable"))
            .alias("odds_status")
        )

    rows = []
    for game in games.iter_rows(named=True):
        away_team, home_team = game["away_team"], game["home_team"]
        away_ml, home_ml = game.get("away_moneyline"), game.get("home_moneyline")
        complete_pair = away_ml is not None and home_ml is not None

        market_probs = {}
        if complete_pair:
            away_raw = american_implied_probability(away_ml)
            home_raw = american_implied_probability(home_ml)
            if away_raw is not None and home_raw is not None and away_raw + home_raw > 0:
                total_raw = away_raw + home_raw
                market_probs = {away_team: away_raw / total_raw, home_team: home_raw / total_raw}
            else:
                complete_pair = False

        for team, price, model_prob in (
            (away_team, away_ml, game["away_win_probability"]),
            (home_team, home_ml, game["home_win_probability"]),
        ):
            model_prob = float(model_prob)
            is_home_side = (team == home_team)

            if complete_pair and team in market_probs and price is not None:
                market_prob = float(market_probs[team])
                profit = american_profit_per_unit(price)
                ev = model_prob * profit - (1.0 - model_prob)
                edge = (model_prob - market_prob) * 100.0
                market_pct = market_prob * 100.0
                price_display = f"{float(price):+.0f}"
                status = "Available"
            else:
                market_pct = None
                edge = None
                ev = None
                price_display = "—"
                status = "Odds unavailable"

            row_dict = {
                "Matchup": f"{away_team} @ {home_team}",
                "Team": team,
                "Model win %": model_prob * 100.0,
                "No-vig market %": market_pct,
                "Edge (pp)": edge,
                "Moneyline": price_display,
                "EV / 1 unit": ev,
                "Provider": game.get("odds_provider") or "ESPN",
                "Odds status": status,
                "is_home": is_home_side,
                "raw_price": float(price) if price is not None else None,
            }

            for f in V3_FEATURES:
                row_dict[f] = game.get(f)

            rows.append(row_dict)

    if not rows:
        return pl.DataFrame()
    return pl.DataFrame(rows, strict=False).sort(
        ["Odds status", "EV / 1 unit"], descending=[False, True], nulls_last=True
    )

def get_current_nfl_week(schedule: pl.DataFrame, default_weeks: list[int]) -> int:
    """Calculates the current/next active week based on today's calendar date."""
    if not default_weeks:
        return 1

    if "gameday" in schedule.columns:
        today_str = date.today().isoformat()
        
        # Earliest week with matches today or in the future
        future_weeks = (
            schedule
            .filter(pl.col("gameday") >= today_str)
            .select("week")
            .drop_nulls()
            .sort("week")
            .to_series()
            .to_list()
        )
        if future_weeks:
            return future_weeks[0]

        # If regular season completed, default to final week
        past_weeks = (
            schedule
            .filter(pl.col("gameday") < today_str)
            .select("week")
            .drop_nulls()
            .sort("week", descending=True)
            .to_series()
            .to_list()
        )
        if past_weeks:
            return past_weeks[0]

    return default_weeks[0]

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.html(
        """
        <div class="sidebar-brand">NFL Predictor</div>
        <div class="sidebar-version">2026 Season · Random Forest V3</div>
        """
    )

    st.markdown("#### Navigate")
    st.caption("Choose what you want to do")

    nav_labels = {
        "Predictions": "Game Predictions",
        "Odds & Value": "Betting Value",
        "Weekly Parlay": "Parlay History",
        "Team Stats": "Team Statistics",
        "Model": "Model Performance",
    }
    nav_descriptions = {
        "Predictions": "See upcoming matchups, win probabilities, and the factors behind each pick.",
        "Odds & Value": "Compare model probabilities with available moneyline odds to estimate value.",
        "Weekly Parlay": "Review saved picks, final scores, and which legs won or lost.",
        "Team Stats": "Compare team form and performance across the season.",
        "Model": "Inspect feature importance and model evaluation metrics.",
    }

    page = st.radio(
        "Dashboard section",
        list(nav_labels.keys()),
        format_func=lambda option: nav_labels[option],
        label_visibility="collapsed",
        key="main_navigation",
    )
    st.caption(nav_descriptions[page])

    st.divider()
    st.markdown("#### Game week")
    st.caption("This selection applies to predictions, betting value, and parlay history.")

    weeks = sorted(
        schedule_2026
        .select("week")
        .drop_nulls()
        .unique()
        .to_series()
        .to_list()
    )

    default_week = get_current_nfl_week(schedule_2026, weeks)
    default_index = weeks.index(default_week) if default_week in weeks else 0

    selected_week = st.selectbox(
        "Select a week",
        weeks,
        index=default_index,
        format_func=lambda week: f"Week {week}",
        label_visibility="collapsed",
        help="Defaults to current game week based on today's calendar date.",
        key="selected_game_week",
    )

    st.divider()
    st.markdown("#### Data")
    if st.button(
        "Refresh data",
        use_container_width=True,
        help="Clear cached API responses and reload the dashboard data.",
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    with open(__file__, "rb") as current_script:
        st.download_button(
            "Download app.py",
            data=current_script.read(),
            file_name="nfl_predictor_app.py",
            mime="text/x-python",
            use_container_width=True,
            help="Download this complete Python file.",
        )

    training_min = model_data["season"].min()
    training_max = model_data["season"].max()
    st.html(
        f"""
        <div style="color:#64748b;font-size:0.72rem;line-height:1.6;margin-top:1rem;">
            <strong>Data sources</strong><br>
            Historical training: {training_min}–{training_max}<br>
            Current season: 2026 NFL regular season<br>
            Sources: NFLverse and ESPN
        </div>
        """
    )

# ============================================================
# HERO
# ============================================================

st.html(
    """
    <div class="hero">
        <div class="hero-kicker">
            NFL • 2026 SEASON
        </div>
        <div class="hero-title">
            NFL Game Predictor
        </div>
        <div class="hero-subtitle">
            Data-driven game probabilities powered by historical
            performance, recent form and advanced Expected Points
            Added metrics.
        </div>
        <div class="hero-badge">
            RANDOM FOREST • V3 • 13 FEATURES
        </div>
    </div>
    """
)

# ============================================================
# PREDICTIONS PAGE
# ============================================================

if page == "Predictions":
    week_predictions = (
        predictions
        .filter(pl.col("week") == selected_week)
        .sort("gameday")
        if not predictions.is_empty()
        else pl.DataFrame()
    )

    total_games = len(week_predictions)

    if total_games == 0:
        st.info("There are no upcoming games for this week.")
        st.stop()

    week_predictions = (
        week_predictions
        .with_columns(
            pl.max_horizontal([
                "home_win_probability",
                "away_win_probability",
            ])
            .alias("confidence")
        )
    )

    avg_confidence = (
        week_predictions
        .select(pl.col("confidence").mean())
        .item()
    )

    strongest = (
        week_predictions
        .sort("confidence", descending=True)
        .row(0, named=True)
    )

    high_confidence = (
        week_predictions
        .filter(pl.col("confidence") >= 0.70)
        .height
    )

    st.html(
        f"""
        <div class="section-header">
            <div>
                <div class="section-title">
                    Week {selected_week}
                </div>
                <div class="section-subtitle">
                    Model predictions for upcoming games
                </div>
            </div>
        </div>
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Games</div>
                <div class="metric-value">{total_games}</div>
                <div class="metric-note">Upcoming matchups</div>
            </div>
            """
        )
    with c2:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Avg. Win Probability</div>
                <div class="metric-value">{avg_confidence:.1%}</div>
                <div class="metric-note">Strongest side across games</div>
            </div>
            """
        )
    with c3:
        strongest_probability = max(
            strongest["home_win_probability"],
            strongest["away_win_probability"],
        )
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Strongest Pick</div>
                <div class="metric-value">{strongest["predicted_winner"]}</div>
                <div class="metric-note">{strongest_probability:.1%} win probability</div>
            </div>
            """
        )
    with c4:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">High Probability</div>
                <div class="metric-value">{high_confidence}</div>
                <div class="metric-note">Games ≥ 70%</div>
            </div>
            """
        )

    for row in week_predictions.iter_rows(named=True):
        home = row["home_team"]
        away = row["away_team"]
        home_prob = row["home_win_probability"] * 100
        away_prob = row["away_win_probability"] * 100
        winner = row["predicted_winner"]
        probability = max(home_prob, away_prob)

        if probability >= 70:
            probability_text = "HIGH"
            probability_class = "confidence-high"
        elif probability >= 60:
            probability_text = "MODERATE"
            probability_class = "confidence-medium"
        else:
            probability_text = "CLOSE"
            probability_class = "confidence-low"

        home_stats = (
            current_team_features
            .filter(pl.col("team") == home)
            .row(0, named=True)
        )
        away_stats = (
            current_team_features
            .filter(pl.col("team") == away)
            .row(0, named=True)
        )

        home_record = f"{int(home_stats['wins'])}-{int(home_stats['games'] - home_stats['wins'])}"
        away_record = f"{int(away_stats['wins'])}-{int(away_stats['games'] - away_stats['wins'])}"

        away_bar_class = "bar-fill-winner" if winner == away else "bar-fill-away"
        home_bar_class = "bar-fill-winner" if winner == home else "bar-fill-home"

        st.html(
            f"""
            <div class="game-card">
                <div class="game-top">
                    <div class="game-time">
                        {row['gameday']} • {row['gametime']}
                    </div>
                    <div class="confidence-label">
                        Win probability:
                        <span class="{probability_class}">
                            {probability_text}
                            {probability:.1f}%
                        </span>
                    </div>
                </div>
                <div class="matchup">
                    <div class="team">
                        <div class="team-mark">{away}</div>
                        <div>
                            <div class="team-code">{away}</div>
                            <div class="team-name">{TEAM_NAMES.get(away, away)}</div>
                            <div class="team-record">{away_record}</div>
                        </div>
                    </div>
                    <div class="versus">@</div>
                    <div class="team team-right">
                        <div>
                            <div class="team-code">{home}</div>
                            <div class="team-name">{TEAM_NAMES.get(home, home)}</div>
                            <div class="team-record">{home_record}</div>
                        </div>
                        <div class="team-mark">{home}</div>
                    </div>
                </div>
                <div class="probability-section">
                    <div class="probability-row">
                        <div class="probability-team">{away}</div>
                        <div class="probability-value">{away_prob:.1f}%</div>
                    </div>
                    <div class="bar">
                        <div class="{away_bar_class}" style="width:{away_prob}%"></div>
                    </div>
                    <div class="probability-row">
                        <div class="probability-team">{home}</div>
                        <div class="probability-value">{home_prob:.1f}%</div>
                    </div>
                    <div class="bar">
                        <div class="{home_bar_class}" style="width:{home_prob}%"></div>
                    </div>
                </div>
                <div class="card-bottom">
                    <div class="card-stat">
                        <div class="card-stat-label">Away PPG</div>
                        <div class="card-stat-value">{away_stats['ppg']:.1f}</div>
                    </div>
                    <div class="card-stat">
                        <div class="card-stat-label">Home PPG</div>
                        <div class="card-stat-value">{home_stats['ppg']:.1f}</div>
                    </div>
                    <div class="card-stat">
                        <div class="card-stat-label">EPA Edge</div>
                        <div class="card-stat-value">{row['off_epa_diff']:+.3f}</div>
                    </div>
                </div>
                <div class="prediction-strip">
                    <div>
                        <div class="prediction-caption">Model prediction</div>
                        <div class="prediction-name">{TEAM_NAMES.get(winner, winner)}</div>
                    </div>
                    <div class="prediction-probability">{probability:.1f}%</div>
                </div>
            </div>
            """
        )

        with st.expander(f"Matchup details • {away} @ {home}"):
            left, right = st.columns(2, gap="large")
            with left:
                st.markdown(f"### {away}")
                st.metric("Record", away_record)
                st.metric("Points / Game", f"{away_stats['ppg']:.1f}")
                st.metric("Points Allowed", f"{away_stats['papg']:.1f}")
                st.metric("Recent Point Diff.", f"{away_stats['rolling_point_diff_5']:+.1f}")
            with right:
                st.markdown(f"### {home}")
                st.metric("Record", home_record)
                st.metric("Points / Game", f"{home_stats['ppg']:.1f}")
                st.metric("Points Allowed", f"{home_stats['papg']:.1f}")
                st.metric("Recent Point Diff.", f"{home_stats['rolling_point_diff_5']:+.1f}")

            st.markdown("---")
            st.markdown("### Model factors")

            factor_data = pl.DataFrame({
                "Feature": [
                    "Season win percentage",
                    "Recent form",
                    "Offensive EPA",
                    "Defensive EPA",
                    "Passing EPA",
                    "Rushing EPA",
                ],
                "Home vs Away": [
                    f"{row['home_win_pct'] - row['away_win_pct']:+.3f}",
                    f"{row['rolling_point_diff_diff_5']:+.2f}",
                    f"{row['off_epa_diff']:+.3f}",
                    f"{row['def_epa_diff']:+.3f}",
                    f"{row['rolling_pass_epa_diff_5']:+.3f}",
                    f"{row['rolling_rush_epa_diff_5']:+.3f}",
                ],
            })

            st.dataframe(
                factor_data,
                hide_index=True,
                use_container_width=True,
            )

# ============================================================
# ODDS & VALUE PAGE (EXPLAINABLE & BEST PICK SPOTLIGHT)
# ============================================================

elif page == "Odds & Value":
    st.html(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Moneyline Value Finder</div>
                <div class="section-subtitle">
                    Explainable EV analysis comparing V3 model probabilities against market odds
                </div>
            </div>
        </div>
        """
    )

    st.markdown(
        "This window identifies betting value by comparing our Random Forest probabilities "
        "against bookmaker odds with the vig removed. It highlights the single **best pick of the week** "
        "and breaks down the exact statistical factors driving the edge."
    )

    controls_left, controls_right = st.columns(2)
    with controls_left:
        minimum_edge_pp = st.slider(
            "Minimum model edge (percentage points)",
            min_value=0.0,
            max_value=15.0,
            value=3.0,
            step=0.5,
            help="Model win probability minus the no-vig market probability.",
        )
    with controls_right:
        minimum_ev = st.slider(
            "Minimum expected value per unit",
            min_value=-0.10,
            max_value=0.50,
            value=0.03,
            step=0.01,
            format="%.2f",
            help="An EV of 0.03 means an estimated profit of 0.03 units per unit staked.",
        )

    week_predictions = (
        predictions
        .filter(pl.col("week") == selected_week)
        .sort("gameday")
        if not predictions.is_empty()
        else pl.DataFrame()
    )

    if week_predictions.is_empty():
        st.info("There are no upcoming matchups available for this week.")
    else:
        try:
            with st.spinner(f"Loading ESPN moneylines for Week {selected_week}..."):
                odds_frame = fetch_espn_moneylines(2026, int(selected_week))
        except requests.RequestException as exc:
            odds_frame = pl.DataFrame()
            st.warning(f"ESPN odds could not be reached: {exc}")
        except (ValueError, TypeError) as exc:
            odds_frame = pl.DataFrame()
            st.warning(f"Error parsing odds: {exc}")

        value_table = build_value_table(week_predictions, odds_frame)

        if value_table.is_empty():
            st.info("No matchups found for this week. Use Refresh data in the sidebar.")
        else:
            available_table = value_table.filter(pl.col("Odds status") == "Available")
            candidates = available_table.filter(
                (pl.col("Edge (pp)") >= minimum_edge_pp)
                & (pl.col("EV / 1 unit") >= minimum_ev)
            ).sort("EV / 1 unit", descending=True)

            best_pick = candidates.row(0, named=True) if not candidates.is_empty() else (
                available_table.sort("EV / 1 unit", descending=True).row(0, named=True) if not available_table.is_empty() else None
            )

            if best_pick:
                best_team = best_pick["Team"]
                best_matchup = best_pick["Matchup"]
                best_ev = best_pick["EV / 1 unit"]
                best_edge = best_pick["Edge (pp)"]
                best_ml = best_pick["Moneyline"]
                best_m_prob = best_pick["Model win %"]
                best_mkt_prob = best_pick["No-vig market %"]

                st.html(
                    f"""
                    <div style="
                        background: radial-gradient(circle at 90% 10%, rgba(34, 197, 94, 0.15), transparent 30%),
                                    linear-gradient(135deg, #111827 0%, #0f172a 100%);
                        border: 1px solid #15803d;
                        border-radius: 16px;
                        padding: 1.5rem 1.8rem;
                        margin-bottom: 1.4rem;
                    ">
                        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.6rem;">
                            <div style="color: #4ade80; font-size: 0.68rem; font-weight: 850; letter-spacing: 0.15em; text-transform: uppercase;">
                                ⭐ TOP VALUE PICK OF THE WEEK
                            </div>
                            <div style="background: rgba(34, 197, 94, 0.2); border: 1px solid #22c55e; border-radius: 6px; padding: 0.25rem 0.6rem; color: #4ade80; font-size: 0.75rem; font-weight: 800;">
                                EV: {best_ev:+.3f} units
                            </div>
                        </div>
                        <div style="display: flex; justify-content: space-between; align-items: flex-end; flex-wrap: wrap; gap: 1rem;">
                            <div>
                                <div style="color: #f8fafc; font-size: 2rem; font-weight: 850; letter-spacing: -0.03em;">
                                    {best_team} ({best_ml})
                                </div>
                                <div style="color: #94a3b8; font-size: 0.85rem; margin-top: 0.2rem;">
                                    {best_matchup} • {best_pick['Provider']}
                                </div>
                            </div>
                            <div style="display: flex; gap: 1.4rem;">
                                <div>
                                    <div style="color: #64748b; font-size: 0.62rem; font-weight: 800; text-transform: uppercase;">Model Win %</div>
                                    <div style="color: #f8fafc; font-size: 1.35rem; font-weight: 850;">{best_m_prob:.1f}%</div>
                                </div>
                                <div>
                                    <div style="color: #64748b; font-size: 0.62rem; font-weight: 800; text-transform: uppercase;">Market Implied</div>
                                    <div style="color: #94a3b8; font-size: 1.35rem; font-weight: 850;">{best_mkt_prob:.1f}%</div>
                                </div>
                                <div>
                                    <div style="color: #64748b; font-size: 0.62rem; font-weight: 800; text-transform: uppercase;">Model Edge</div>
                                    <div style="color: #4ade80; font-size: 1.35rem; font-weight: 850;">+{best_edge:.1f} pp</div>
                                </div>
                            </div>
                        </div>
                    </div>
                    """
                )

                with st.expander("🔍 Why does the model favor this pick? (Factor Explanations)", expanded=True):
                    is_home = best_pick["is_home"]
                    mult = 1.0 if is_home else -1.0

                    importances = final_rf.feature_importances_
                    explanations = []

                    for idx, feat in enumerate(V3_FEATURES):
                        val = best_pick.get(feat)
                        if val is not None:
                            oriented_val = float(val) * mult
                            imp = importances[idx]
                            weighted_impact = oriented_val * imp
                            explanations.append({
                                "Feature": FEATURE_LABELS.get(feat, feat),
                                "Team Advantage": oriented_val,
                                "Model Weight": imp,
                                "Impact Score": weighted_impact,
                            })

                    exp_df = pl.DataFrame(explanations).sort("Impact Score", descending=True)

                    st.markdown(f"**Key drivers supporting {best_team}:**")
                    col_top, col_table = st.columns([1, 2], gap="large")

                    with col_top:
                        top_driver = exp_df.row(0, named=True)
                        second_driver = exp_df.row(1, named=True)
                        st.markdown(
                            f"""
                            - **Primary Driver:** **{top_driver['Feature']}** ({top_driver['Team Advantage']:+.3f}). 
                            - **Secondary Driver:** **{second_driver['Feature']}** ({second_driver['Team Advantage']:+.3f}).
                            
                            *The Random Forest places the greatest probability weight on team EPA efficiency margins and rolling 5-game net differentials relative to market pricing.*
                            """
                        )

                    with col_table:
                        st.dataframe(
                            exp_df.select([
                                "Feature",
                                "Team Advantage",
                                "Impact Score"
                            ]),
                            hide_index=True,
                            use_container_width=True,
                            column_config={
                                "Team Advantage": st.column_config.NumberColumn(format="%+.3f"),
                                "Impact Score": st.column_config.NumberColumn(format="%+.4f"),
                            }
                        )

            metric_a, metric_b, metric_c = st.columns(3)
            with metric_a:
                st.metric("Games with odds", available_table["Matchup"].n_unique() if not available_table.is_empty() else 0)
            with metric_b:
                st.metric("Sides evaluated", available_table.height)
            with metric_c:
                st.metric("Potential value sides", candidates.height)

            missing_games = value_table.filter(pl.col("Odds status") != "Available")["Matchup"].n_unique()
            if missing_games:
                st.info(
                    f"Moneyline prices are unavailable or incomplete for {missing_games} matchup(s). "
                    "Those games remain visible below, but are excluded from value calculations. "
                    "Use Refresh data to check again later."
                )

            st.markdown("### Potential value candidates")
            display_cols = [
                "Matchup", "Team", "Model win %", "No-vig market %",
                "Edge (pp)", "Moneyline", "EV / 1 unit", "Provider", "Odds status"
            ]

            if candidates.is_empty():
                st.info(
                    "No sides meet both thresholds for this week. "
                    "You can adjust the filters, but avoid lowering them just to force a candidate."
                )
            else:
                st.dataframe(
                    candidates.select(display_cols),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Model win %": st.column_config.NumberColumn(format="%.1f%%"),
                        "No-vig market %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Edge (pp)": st.column_config.NumberColumn(format="%+.1f"),
                        "EV / 1 unit": st.column_config.NumberColumn(format="%+.3f"),
                    },
                )

            with st.expander("All matchups and moneyline comparisons", expanded=False):
                st.dataframe(
                    value_table.select(display_cols),
                    hide_index=True,
                    use_container_width=True,
                    column_config={
                        "Model win %": st.column_config.NumberColumn(format="%.1f%%"),
                        "No-vig market %": st.column_config.NumberColumn(format="%.1f%%"),
                        "Edge (pp)": st.column_config.NumberColumn(format="%+.1f"),
                        "EV / 1 unit": st.column_config.NumberColumn(format="%+.3f"),
                    },
                )

            st.caption(
                "Odds source: ESPN scoreboard endpoint. Prices may change and "
                "may be unavailable for some games. Model probabilities are not "
                "perfectly calibrated; positive estimated EV is not a guarantee "
                "of profit or a betting recommendation. Confirm the price with "
                "the sportsbook before making any decision."
            )

# ============================================================
# WEEKLY PARLAY PAGE
# ============================================================

elif page == "Weekly Parlay":
    st.html(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Weekly Parlay</div>
                <div class="section-subtitle">
                    See how an all-games winner parlay would have performed against actual results
                </div>
            </div>
        </div>
        """
    )

    with st.expander("How this backtest works", expanded=False):
        st.markdown(
            "1. **One pick per game:** the model selects the team with the higher win probability.\n"
            "2. **Results are checked against ESPN:** completed games are graded using their actual winner.\n"
            "3. **One wrong pick loses the parlay:** all legs must win for the full parlay to hit.\n"
            "4. **Pending games are not graded yet.** The app saves the first prediction it sees for each game "
            "to `data/processed/prediction_history.parquet`, so later model updates do not rewrite that pick. "
            "This is a winner-pick backtest, not a payout simulation, because historical pre-kickoff prices are not stored."
        )

    parlay_predictions = (
        prediction_history
        .filter(pl.col("week") == selected_week)
        .sort("gameday")
        if not prediction_history.is_empty()
        else pl.DataFrame()
    )

    individual_bet_history = pl.DataFrame()
    try:
        with st.spinner("Updating individual-bet ledger..."):
            current_week_odds = fetch_espn_moneylines(2026, int(selected_week))
            current_week_results = fetch_espn_results(2026, int(selected_week))
            individual_bet_history = update_individual_bet_history(
                predictions, current_week_odds, current_week_results, 2026, int(selected_week)
            )
    except requests.RequestException as exc:
        st.warning(f"Could not update the individual-bet ledger: {exc}")

    if not individual_bet_history.is_empty():
        settled_bets = individual_bet_history.filter(pl.col("net_units").is_not_null())
        cumulative_profit = float(settled_bets.select(pl.col("net_units").sum()).item() or 0.0) if not settled_bets.is_empty() else 0.0
        wins = settled_bets.filter(pl.col("bet_result") == "WON").height
        losses = settled_bets.filter(pl.col("bet_result") == "LOST").height
        pushes = settled_bets.filter(pl.col("bet_result") == "PUSH").height
        
        st.markdown("### Individual bets: 1 unit per game")
        st.caption(
            "A separate 1-unit moneyline bet on each saved model pick. Winning bets earn profit at the saved American odds; "
            "losses cost 1 unit and ties return the stake. Only prices captured before kickoff count."
        )
        b1, b2, b3, b4 = st.columns(4)
        b1.metric("Net profit", f"{cumulative_profit:+.2f} units")
        b2.metric("Settled bets", settled_bets.height)
        b3.metric("Wins / losses", f"{wins} / {losses}")
        b4.metric("Pushes", pushes)
        
        if not settled_bets.is_empty():
            weekly_profit = (
                settled_bets.group_by(["season", "week"])
                .agg(pl.col("net_units").sum().alias("Weekly net units"))
                .sort(["season", "week"])
                .with_columns(pl.col("Weekly net units").cum_sum().alias("Cumulative net units"))
            )
            st.markdown("**Cumulative profit over the season**")
            st.line_chart(weekly_profit.select("week", "Cumulative net units"), x="week", y="Cumulative net units")
            st.dataframe(weekly_profit, hide_index=True, use_container_width=True)
            st.download_button(
                "Download individual-bet ledger (CSV)",
                data=individual_bet_history.sort(["season", "week"]).write_csv(),
                file_name="nfl_individual_bet_history.csv",
                mime="text/csv",
            )
        st.caption(
            "Tracking begins when the app first captures complete pregame odds. It cannot reconstruct historical prices "
            "for games that had already started or finished."
        )
    else:
        st.info(
            "The individual-bet ledger will appear once complete moneyline odds are captured before kickoff. "
            "Past games without saved pregame prices are excluded rather than estimated."
        )

    if parlay_predictions.is_empty():
        try:
            with st.spinner(f"Loading games and results for Week {selected_week}..."):
                historical_results = fetch_espn_results(2026, int(selected_week))
        except requests.RequestException as exc:
            historical_results = pl.DataFrame()
            st.error(f"Could not retrieve ESPN results: {exc}")

        st.markdown(f"### Week {selected_week}: game results")
        st.caption(
            "No saved prediction snapshots are available for this week. Completed matchups are shown below, "
            "but picks cannot be scored retrospectively if the app did not save them before kickoff."
        )
        if historical_results.is_empty():
            st.info(
                "No games or results were returned for this week yet. If this week has already been played, "
                "try **Refresh data** in the sidebar."
            )
        else:
            historical_rows = []
            for row in historical_results.iter_rows(named=True):
                away_score = row.get("away_score")
                home_score = row.get("home_score")
                score = (
                    f"{int(away_score)}–{int(home_score)}"
                    if away_score is not None and home_score is not None
                    else "Not final"
                )
                historical_rows.append({
                    "Matchup": f"{row['away_team']} @ {row['home_team']}",
                    "Final score (away–home)": score,
                    "Actual winner": row.get("actual_winner") or "Pending",
                    "Game status": row.get("game_status") or "Unknown",
                    "Gameday": row.get("gameday_actual") or "",
                })
            historical_table = pl.DataFrame(historical_rows, strict=False)
            st.metric("Games this week", historical_table.height)
            st.dataframe(historical_table, hide_index=True, use_container_width=True)
            st.download_button(
                "Download game results as CSV",
                data=historical_table.write_csv(),
                file_name=f"nfl_week_{selected_week}_game_results.csv",
                mime="text/csv",
            )
    else:
        try:
            with st.spinner(f"Checking actual results for Week {selected_week}..."):
                actual_results = fetch_espn_results(2026, int(selected_week))
        except requests.RequestException as exc:
            actual_results = pl.DataFrame()
            st.error(f"Could not retrieve ESPN results: {exc}")

        if actual_results.is_empty():
            st.warning(
                "ESPN returned no results for this week. Use **Refresh data** in the sidebar "
                "to clear the cache and try again."
            )
        else:
            parlay = (
                parlay_predictions
                .join(actual_results, on=["away_team", "home_team"], how="left")
                .with_columns([
                    pl.max_horizontal([
                        "home_win_probability",
                        "away_win_probability",
                    ]).alias("model_confidence"),
                    pl.when(pl.col("home_win_probability") >= 0.5)
                    .then(pl.col("home_team"))
                    .otherwise(pl.col("away_team"))
                    .alias("predicted_winner"),
                ])
                .with_columns(
                    pl.when(pl.col("predicted_winner") == pl.col("home_team"))
                    .then(pl.col("home_win_probability"))
                    .otherwise(pl.col("away_win_probability"))
                    .mul(100)
                    .alias("Pick probability %")
                )
            )

            graded_rows = []
            for row in parlay.iter_rows(named=True):
                actual_winner = row.get("actual_winner")
                if actual_winner is None:
                    leg_result = "PENDING"
                elif actual_winner == "TIE":
                    leg_result = "PUSH / TIE"
                elif row["predicted_winner"] == actual_winner:
                    leg_result = "WON"
                else:
                    leg_result = "LOST"

                away_score = row.get("away_score")
                home_score = row.get("home_score")
                score_display = (
                    f"{int(away_score)}–{int(home_score)}"
                    if away_score is not None and home_score is not None
                    else "Not final"
                )
                graded_rows.append({
                    "Matchup": f"{row['away_team']} @ {row['home_team']}",
                    "Model pick": row["predicted_winner"],
                    "Pick probability %": row["Pick probability %"],
                    "Final score (away–home)": score_display,
                    "Actual winner": actual_winner or "Pending",
                    "Result": leg_result,
                    "Game status": row.get("game_status") or "Unknown",
                    "Gameday": row.get("gameday_actual") or str(row.get("gameday") or ""),
                })

            parlay_table = pl.DataFrame(graded_rows, strict=False)
            total_legs = parlay_table.height
            won_legs = parlay_table.filter(pl.col("Result") == "WON").height
            lost_legs = parlay_table.filter(pl.col("Result") == "LOST").height
            pushed_legs = parlay_table.filter(pl.col("Result") == "PUSH / TIE").height
            pending_legs = parlay_table.filter(pl.col("Result") == "PENDING").height
            graded_legs = won_legs + lost_legs
            accuracy = won_legs / graded_legs if graded_legs else 0.0

            st.markdown(f"### Week {selected_week} at a glance")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Games in parlay", total_legs)
            m2.metric("Correct picks", f"{won_legs}/{graded_legs}" if graded_legs else "—")
            m3.metric("Wrong picks", lost_legs)
            m4.metric("Still to play", pending_legs)

            if graded_legs:
                st.markdown("**Pick accuracy on decided games**")
                st.progress(accuracy, text=f"{accuracy:.1%} · {won_legs} correct out of {graded_legs} decisive results")

            st.markdown("### Parlay status")
            if pending_legs > 0:
                if lost_legs > 0:
                    st.error(
                        f"**Parlay already busted.** {lost_legs} pick(s) are incorrect. "
                        f"There are still {pending_legs} game(s) pending, but the full parlay can no longer hit."
                    )
                else:
                    st.info(
                        f"**In progress.** {pending_legs} game(s) have not been decided yet. "
                        "The parlay can only hit if every remaining pick wins."
                    )
            elif lost_legs > 0:
                st.error(
                    f"**Parlay missed.** {won_legs} picks were correct and {lost_legs} were wrong "
                    f"across {graded_legs} decisive results. One wrong leg is enough to lose an all-games parlay."
                )
            elif pushed_legs > 0:
                st.success(
                    f"**All decisive picks won.** {pushed_legs} game(s) ended tied. "
                    "A tie is usually treated as a void leg, but sportsbook rules determine the final payout."
                )
            elif total_legs > 0 and won_legs == total_legs:
                st.success(f"**Parlay hit!** All {total_legs} picks were correct.")

            st.markdown("### Game-by-game results")
            st.caption("Filter the table to quickly find missed picks or games that have not finished.")
            filter_choice = st.selectbox(
                "Show games",
                ["All games", "Wrong picks", "Correct picks", "Pending games", "Ties"],
                key="parlay_result_filter",
            )
            if filter_choice == "Wrong picks":
                visible_table = parlay_table.filter(pl.col("Result") == "LOST")
            elif filter_choice == "Correct picks":
                visible_table = parlay_table.filter(pl.col("Result") == "WON")
            elif filter_choice == "Pending games":
                visible_table = parlay_table.filter(pl.col("Result") == "PENDING")
            elif filter_choice == "Ties":
                visible_table = parlay_table.filter(pl.col("Result") == "PUSH / TIE")
            else:
                visible_table = parlay_table

            st.dataframe(
                visible_table,
                hide_index=True,
                use_container_width=True,
                column_config={
                    "Pick probability %": st.column_config.ProgressColumn(
                        "Pick confidence",
                        min_value=0,
                        max_value=100,
                        format="%.1f%%",
                    ),
                    "Matchup": st.column_config.TextColumn("Matchup", width="medium"),
                    "Model pick": st.column_config.TextColumn("Model pick", width="small"),
                    "Actual winner": st.column_config.TextColumn("Actual winner", width="small"),
                    "Result": st.column_config.TextColumn("Leg result", width="small"),
                },
            )
            if visible_table.is_empty():
                st.caption("No games match this filter.")

            st.download_button(
                "Download results as CSV",
                data=parlay_table.write_csv(),
                file_name=f"nfl_week_{selected_week}_parlay_backtest.csv",
                mime="text/csv",
                use_container_width=False,
            )

            st.caption(
                "This backtest evaluates winner predictions only; it does not calculate a real parlay payout. "
                "Historical moneylines from before kickoff would be needed to estimate returns. "
                "The number of legs varies by week because teams may have bye weeks."
            )

# ============================================================
# TEAM STATS PAGE
# ============================================================

elif page == "Team Stats":
    st.html(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Team Performance</div>
                <div class="section-subtitle">2026 regular-season statistics</div>
            </div>
        </div>
        """
    )

    stats = (
        current_team_features
        .with_columns([
            (
                pl.col("wins").cast(pl.Utf8)
                + pl.lit("-")
                + (pl.col("games") - pl.col("wins")).cast(pl.Utf8)
            ).alias("record"),
        ])
        .select([
            "team",
            "record",
            "win_pct",
            "ppg",
            "papg",
            "rolling_point_diff_5",
        ])
        .sort("win_pct", descending=True)
        .rename({
            "team": "Team",
            "record": "Record",
            "win_pct": "Win %",
            "ppg": "PPG",
            "papg": "PAPG",
            "rolling_point_diff_5": "Recent Point Diff.",
        })
    )

    st.dataframe(
        stats,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Win %": st.column_config.NumberColumn(format="%.1f%%"),
            "PPG": st.column_config.NumberColumn(format="%.1f"),
            "PAPG": st.column_config.NumberColumn(format="%.1f"),
            "Recent Point Diff.": st.column_config.NumberColumn(format="%+.1f"),
        },
    )

# ============================================================
# MODEL PAGE
# ============================================================

elif page == "Model":
    training_min = model_data["season"].min()
    training_max = model_data["season"].max()

    st.html(
        """
        <div class="section-header">
            <div>
                <div class="section-title">Model</div>
                <div class="section-subtitle">How the prediction engine works</div>
            </div>
        </div>
        """
    )

    a, b, c = st.columns(3)
    with a:
        st.html(
            """
            <div class="metric-card">
                <div class="metric-label">Algorithm</div>
                <div class="metric-value">Random Forest</div>
                <div class="metric-note">Ensemble classifier</div>
            </div>
            """
        )
    with b:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Features</div>
                <div class="metric-value">{len(V3_FEATURES)}</div>
                <div class="metric-note">V3 feature set</div>
            </div>
            """
        )
    with c:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">Training</div>
                <div class="metric-value">{training_min}–{training_max}</div>
                <div class="metric-note">Regular season games</div>
            </div>
            """
        )

    st.html(
        """
        <div class="model-panel">
            <div class="model-title">Feature groups</div>
            <div class="model-text">
                <b>Season performance</b><br>
                Win percentage, points scored and defensive
                performance relative to the opponent.
                <br><br>
                <b>Recent form</b><br>
                Five-game rolling measures capturing current
                team performance.
                <br><br>
                <b>EPA</b><br>
                Offensive, defensive, passing and rushing
                Expected Points Added metrics.
                <br><br>
                All matchup features are calculated using
                information available before the corresponding
                game, preventing future-game leakage.
            </div>
        </div>
        """
    )

    st.markdown("### V3 Feature Set")

    feature_table = pl.DataFrame({
        "Feature Group": [
            "Season", "Season", "Season",
            "Recent Form", "Recent Form", "Recent Form", "Recent Form",
            "EPA", "EPA", "EPA", "EPA", "EPA", "EPA",
        ],
        "Feature": [
            "Win percentage difference",
            "Points per game difference",
            "Defensive performance difference",
            "Rolling win percentage",
            "Rolling PPG",
            "Rolling defensive performance",
            "Rolling point differential",
            "Offensive EPA",
            "Defensive EPA",
            "Rolling offensive EPA",
            "Rolling defensive EPA",
            "Rolling passing EPA",
            "Rolling rushing EPA",
        ],
    })

    st.dataframe(
        feature_table,
        hide_index=True,
        use_container_width=True,
    )

# ============================================================
# FOOTER
# ============================================================

st.html(
    """
    <div style="
        text-align:center;
        color:#334155;
        font-size:0.65rem;
        margin-top:3rem;
        padding-top:1rem;
        border-top:1px solid #1e293b;
    ">
        NFL Game Predictor • Random Forest V3 • NFLverse
    </div>
    """
)