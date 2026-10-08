import streamlit as st
import polars as pl
import nflreadpy as nfl

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

TEAM_NAMES = {
    "ARI": "Arizona Cardinals",
    "ATL": "Atlanta Falcons",
    "BAL": "Baltimore Ravens",
    "BUF": "Buffalo Bills",
    "CAR": "Carolina Panthers",
    "CHI": "Chicago Bears",
    "CIN": "Cincinnati Bengals",
    "CLE": "Cleveland Browns",
    "DAL": "Dallas Cowboys",
    "DEN": "Denver Broncos",
    "DET": "Detroit Lions",
    "GB": "Green Bay Packers",
    "HOU": "Houston Texans",
    "IND": "Indianapolis Colts",
    "JAX": "Jacksonville Jaguars",
    "KC": "Kansas City Chiefs",
    "LV": "Las Vegas Raiders",
    "LAC": "Los Angeles Chargers",
    "LAR": "Los Angeles Rams",
    "MIA": "Miami Dolphins",
    "MIN": "Minnesota Vikings",
    "NE": "New England Patriots",
    "NO": "New Orleans Saints",
    "NYG": "New York Giants",
    "NYJ": "New York Jets",
    "PHI": "Philadelphia Eagles",
    "PIT": "Pittsburgh Steelers",
    "SF": "San Francisco 49ers",
    "SEA": "Seattle Seahawks",
    "TB": "Tampa Bay Buccaneers",
    "TEN": "Tennessee Titans",
    "WAS": "Washington Commanders",
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
       EXPANDER
    ======================================================== */

    div[data-testid="stExpander"] {
        background: #0f172a !important;
        border: 1px solid #1e293b !important;
        border-radius: 10px !important;
        margin-bottom: 0.9rem;
    }


    /* ========================================================
       DATAFRAME
    ======================================================== */

    div[data-testid="stDataFrame"] {
        border: 1px solid #1e293b;
        border-radius: 10px;
        overflow: hidden;
    }


    /* ========================================================
       SIDEBAR
    ======================================================== */

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


    /* ========================================================
       MODEL PANEL
    ======================================================== */

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


    /* ========================================================
       BUTTONS
    ======================================================== */

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


    /* ========================================================
       SELECTBOX
    ======================================================== */

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
# DATA
# ============================================================

@st.cache_data
def load_historical_data():
    return pl.read_parquet(
        "data/processed/model_data.parquet"
    )


@st.cache_resource
def train_model(data):

    historical = data.drop_nulls(
        subset=V3_FEATURES
    )

    X = (
        historical
        .select(V3_FEATURES)
        .to_numpy()
    )

    y = (
        historical
        .select("home_win")
        .to_numpy()
        .ravel()
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )

    model.fit(X, y)

    return model


@st.cache_data(ttl=3600)
def load_2026_data():

    schedule = nfl.load_schedules(
        seasons=[2026]
    )

    schedule = schedule.filter(
        pl.col("game_type") == "REG"
    )

    pbp = nfl.load_pbp(
        seasons=[2026]
    )

    pbp = pbp.filter(
        pl.col("season_type") == "REG"
    )

    return schedule, pbp


model_data = load_historical_data()

final_rf = train_model(
    model_data
)

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

        pl.col("home_score").alias(
            "points_for"
        ),

        pl.col("away_score").alias(
            "points_against"
        ),

        (
            pl.col("result") > 0
        )
        .cast(pl.Int8)
        .alias("win"),
    ])

    away = games.select([
        "game_id",
        "season",
        "week",

        pl.col("away_team").alias("team"),

        pl.col("away_score").alias(
            "points_for"
        ),

        pl.col("home_score").alias(
            "points_against"
        ),

        (
            pl.col("result") < 0
        )
        .cast(pl.Int8)
        .alias("win"),
    ])

    return (
        pl.concat([home, away])
        .sort([
            "team",
            "week",
        ])
    )


def build_current_team_features(schedule):

    completed = schedule.filter(
        pl.col("home_score").is_not_null()
        &
        pl.col("away_score").is_not_null()
    )

    team_games = build_team_games(
        completed
    )

    season_stats = (
        team_games
        .group_by("team")
        .agg([
            pl.col("win")
            .sum()
            .alias("wins"),

            pl.len()
            .alias("games"),

            pl.col("points_for")
            .sum()
            .alias("points_for"),

            pl.col("points_against")
            .sum()
            .alias("points_against"),
        ])
        .with_columns([
            (
                pl.col("wins")
                /
                pl.col("games")
            ).alias("win_pct"),

            (
                pl.col("points_for")
                /
                pl.col("games")
            ).alias("ppg"),

            (
                pl.col("points_against")
                /
                pl.col("games")
            ).alias("papg"),
        ])
    )

    rolling = (
        team_games
        .sort([
            "team",
            "week",
        ])
        .group_by("team")
        .agg([
            pl.col("win")
            .tail(5)
            .mean()
            .alias(
                "rolling_win_pct_5"
            ),

            pl.col("points_for")
            .tail(5)
            .mean()
            .alias(
                "rolling_ppg_5"
            ),

            pl.col("points_against")
            .tail(5)
            .mean()
            .alias(
                "rolling_papg_5"
            ),

            (
                (
                    pl.col("points_for")
                    -
                    pl.col("points_against")
                )
                .tail(5)
                .mean()
            )
            .alias(
                "rolling_point_diff_5"
            ),
        ])
    )

    return season_stats.join(
        rolling,
        on="team",
        how="left",
    )


current_team_features = build_current_team_features(
    schedule_2026
)


# ============================================================
# EPA
# ============================================================

def build_current_epa_features(
    schedule,
    pbp,
):

    completed = schedule.filter(
        pl.col("home_score").is_not_null()
        &
        pl.col("away_score").is_not_null()
    )

    completed_ids = (
        completed
        .select("game_id")
        .unique()
    )

    pbp_completed = (
        pbp
        .join(
            completed_ids,
            on="game_id",
            how="inner",
        )
    )

    plays = pbp_completed.filter(
        pl.col("posteam").is_not_null()
    )

    offense = (
        plays
        .group_by([
            "game_id",
            "week",
            "posteam",
        ])
        .agg([
            pl.col("epa")
            .mean()
            .alias(
                "off_epa_per_play"
            ),

            pl.when(
                pl.col("pass_attempt") == 1
            )
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias(
                "pass_epa_per_play"
            ),

            pl.when(
                pl.col("rush_attempt") == 1
            )
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias(
                "rush_epa_per_play"
            ),
        ])
        .rename({
            "posteam": "team"
        })
    )

    defense = (
        plays
        .group_by([
            "game_id",
            "week",
            "defteam",
        ])
        .agg([
            pl.col("epa")
            .mean()
            .alias(
                "def_epa_allowed_per_play"
            ),
        ])
        .rename({
            "defteam": "team"
        })
    )

    combined = (
        offense
        .join(
            defense,
            on=[
                "game_id",
                "week",
                "team",
            ],
            how="inner",
        )
        .sort([
            "team",
            "week",
        ])
    )

    return (
        combined
        .group_by("team")
        .agg([
            pl.col(
                "off_epa_per_play"
            )
            .mean()
            .alias(
                "off_epa_per_play"
            ),

            pl.col(
                "def_epa_allowed_per_play"
            )
            .mean()
            .alias(
                "def_epa_allowed_per_play"
            ),

            pl.col(
                "off_epa_per_play"
            )
            .tail(5)
            .mean()
            .alias(
                "rolling_off_epa_per_play_5"
            ),

            pl.col(
                "def_epa_allowed_per_play"
            )
            .tail(5)
            .mean()
            .alias(
                "rolling_def_epa_allowed_per_play_5"
            ),

            pl.col(
                "pass_epa_per_play"
            )
            .tail(5)
            .mean()
            .alias(
                "rolling_pass_epa_per_play_5"
            ),

            pl.col(
                "rush_epa_per_play"
            )
            .tail(5)
            .mean()
            .alias(
                "rolling_rush_epa_per_play_5"
            ),
        ])
    )


current_epa = build_current_epa_features(
    schedule_2026,
    pbp_2026,
)


# ============================================================
# PREDICTION DATASET
# ============================================================

def build_prediction_dataset():

    upcoming = schedule_2026.filter(
        pl.col("home_score").is_null()
        |
        pl.col("away_score").is_null()
    )

    home = current_team_features.rename({
        "team": "home_team",

        "win_pct": "home_win_pct",
        "ppg": "home_ppg",
        "papg": "home_papg",

        "rolling_win_pct_5":
            "home_rolling_win_pct_5",

        "rolling_ppg_5":
            "home_rolling_ppg_5",

        "rolling_papg_5":
            "home_rolling_papg_5",

        "rolling_point_diff_5":
            "home_rolling_point_diff_5",
    })

    away = current_team_features.rename({
        "team": "away_team",

        "win_pct": "away_win_pct",
        "ppg": "away_ppg",
        "papg": "away_papg",

        "rolling_win_pct_5":
            "away_rolling_win_pct_5",

        "rolling_ppg_5":
            "away_rolling_ppg_5",

        "rolling_papg_5":
            "away_rolling_papg_5",

        "rolling_point_diff_5":
            "away_rolling_point_diff_5",
    })

    home_epa = current_epa.rename({
        "team": "home_team",

        "off_epa_per_play":
            "home_off_epa",

        "def_epa_allowed_per_play":
            "home_def_epa",

        "rolling_off_epa_per_play_5":
            "home_rolling_off_epa_5",

        "rolling_def_epa_allowed_per_play_5":
            "home_rolling_def_epa_5",

        "rolling_pass_epa_per_play_5":
            "home_rolling_pass_epa_5",

        "rolling_rush_epa_per_play_5":
            "home_rolling_rush_epa_5",
    })

    away_epa = current_epa.rename({
        "team": "away_team",

        "off_epa_per_play":
            "away_off_epa",

        "def_epa_allowed_per_play":
            "away_def_epa",

        "rolling_off_epa_per_play_5":
            "away_rolling_off_epa_5",

        "rolling_def_epa_allowed_per_play_5":
            "away_rolling_def_epa_5",

        "rolling_pass_epa_per_play_5":
            "away_rolling_pass_epa_5",

        "rolling_rush_epa_per_play_5":
            "away_rolling_rush_epa_5",
    })

    data = (
        upcoming
        .join(
            home,
            on="home_team",
            how="left",
        )
        .join(
            away,
            on="away_team",
            how="left",
        )
        .join(
            home_epa,
            on="home_team",
            how="left",
        )
        .join(
            away_epa,
            on="away_team",
            how="left",
        )
    )

    return data.with_columns([

        (
            pl.col("home_win_pct")
            -
            pl.col("away_win_pct")
        ).alias("win_pct_diff"),

        (
            pl.col("home_ppg")
            -
            pl.col("away_ppg")
        ).alias("ppg_diff"),

        (
            pl.col("away_papg")
            -
            pl.col("home_papg")
        ).alias("defense_diff"),

        (
            pl.col("home_rolling_win_pct_5")
            -
            pl.col("away_rolling_win_pct_5")
        ).alias(
            "rolling_win_pct_diff_5"
        ),

        (
            pl.col("home_rolling_ppg_5")
            -
            pl.col("away_rolling_ppg_5")
        ).alias(
            "rolling_ppg_diff_5"
        ),

        (
            pl.col("away_rolling_papg_5")
            -
            pl.col("home_rolling_papg_5")
        ).alias(
            "rolling_defense_diff_5"
        ),

        (
            pl.col("home_rolling_point_diff_5")
            -
            pl.col("away_rolling_point_diff_5")
        ).alias(
            "rolling_point_diff_diff_5"
        ),

        (
            pl.col("home_off_epa")
            -
            pl.col("away_off_epa")
        ).alias("off_epa_diff"),

        (
            pl.col("away_def_epa")
            -
            pl.col("home_def_epa")
        ).alias("def_epa_diff"),

        (
            pl.col("home_rolling_off_epa_5")
            -
            pl.col("away_rolling_off_epa_5")
        ).alias(
            "rolling_off_epa_diff_5"
        ),

        (
            pl.col("away_rolling_def_epa_5")
            -
            pl.col("home_rolling_def_epa_5")
        ).alias(
            "rolling_def_epa_diff_5"
        ),

        (
            pl.col("home_rolling_pass_epa_5")
            -
            pl.col("away_rolling_pass_epa_5")
        ).alias(
            "rolling_pass_epa_diff_5"
        ),

        (
            pl.col("home_rolling_rush_epa_5")
            -
            pl.col("away_rolling_rush_epa_5")
        ).alias(
            "rolling_rush_epa_diff_5"
        ),
    ])


prediction_dataset = build_prediction_dataset()


# ============================================================
# PREDICT
# ============================================================

valid = prediction_dataset.drop_nulls(
    subset=V3_FEATURES
)

X = (
    valid
    .select(V3_FEATURES)
    .to_numpy()
)

probabilities = (
    final_rf
    .predict_proba(X)[:, 1]
)

predictions = (
    valid
    .select([
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

        "rolling_point_diff_diff_5",

        "off_epa_diff",
        "def_epa_diff",

        "rolling_off_epa_diff_5",
        "rolling_def_epa_diff_5",

        "rolling_pass_epa_diff_5",
        "rolling_rush_epa_diff_5",
    ])
    .with_columns([
        pl.Series(
            "home_win_probability",
            probabilities,
        )
    ])
    .with_columns([
        (
            1
            -
            pl.col(
                "home_win_probability"
            )
        ).alias(
            "away_win_probability"
        ),

        pl.when(
            pl.col(
                "home_win_probability"
            ) >= 0.5
        )
        .then(
            pl.col("home_team")
        )
        .otherwise(
            pl.col("away_team")
        )
        .alias(
            "predicted_winner"
        ),
    ])
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.html(
        """
        <div class="sidebar-brand">
            NFL Predictor
        </div>

        <div class="sidebar-version">
            2026 Season • Random Forest V3
        </div>
        """
    )

    st.html(
        """
        <div class="sidebar-section">
            Dashboard
        </div>
        """
    )

    page = st.radio(
        "View",
        [
            "Predictions",
            "Team Stats",
            "Model",
        ],
        label_visibility="collapsed",
    )

    st.html(
        """
        <div class="sidebar-section">
            Week
        </div>
        """
    )

    weeks = sorted(
        prediction_dataset
        .select("week")
        .unique()
        .to_series()
        .to_list()
    )

    selected_week = st.selectbox(
        "Week",
        weeks,
        label_visibility="collapsed",
    )

    st.html(
        """
        <div class="sidebar-section">
            Data
        </div>
        """
    )

    if st.button(
        "Refresh data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.rerun()

    training_min = model_data["season"].min()
    training_max = model_data["season"].max()

    st.html(
        f"""
        <div style="
            color:#475569;
            font-size:0.65rem;
            line-height:1.6;
            margin-top:1rem;
        ">
            Historical training data<br>
            {training_min}–{training_max}<br><br>

            Current season data<br>
            2026 NFL regular season<br><br>

            Source<br>
            NFLverse
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
        .filter(
            pl.col("week") == selected_week
        )
        .sort("gameday")
    )

    total_games = len(
        week_predictions
    )

    if total_games == 0:
        st.info(
            "There are no upcoming games for this week."
        )
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
        .select(
            pl.col("confidence").mean()
        )
        .item()
    )

    strongest = (
        week_predictions
        .sort(
            "confidence",
            descending=True,
        )
        .row(
            0,
            named=True,
        )
    )

    high_confidence = (
        week_predictions
        .filter(
            pl.col("confidence") >= 0.70
        )
        .height
    )

    # --------------------------------------------------------
    # WEEK HEADER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # SUMMARY CARDS
    # --------------------------------------------------------

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Games
                </div>

                <div class="metric-value">
                    {total_games}
                </div>

                <div class="metric-note">
                    Upcoming matchups
                </div>
            </div>
            """
        )

    with c2:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    Avg. Win Probability
                </div>

                <div class="metric-value">
                    {avg_confidence:.1%}
                </div>

                <div class="metric-note">
                    Strongest side across games
                </div>
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
                <div class="metric-label">
                    Strongest Pick
                </div>

                <div class="metric-value">
                    {strongest["predicted_winner"]}
                </div>

                <div class="metric-note">
                    {strongest_probability:.1%} win probability
                </div>
            </div>
            """
        )

    with c4:
        st.html(
            f"""
            <div class="metric-card">
                <div class="metric-label">
                    High Probability
                </div>

                <div class="metric-value">
                    {high_confidence}
                </div>

                <div class="metric-note">
                    Games ≥ 70%
                </div>
            </div>
            """
        )

    # --------------------------------------------------------
    # GAME CARDS
    # --------------------------------------------------------

    for row in week_predictions.iter_rows(
        named=True
    ):

        home = row["home_team"]
        away = row["away_team"]

        home_prob = (
            row["home_win_probability"] * 100
        )

        away_prob = (
            row["away_win_probability"] * 100
        )

        winner = row[
            "predicted_winner"
        ]

        probability = max(
            home_prob,
            away_prob,
        )

        if probability >= 70:
            probability_text = "HIGH"
            probability_class = (
                "confidence-high"
            )
        elif probability >= 60:
            probability_text = "MODERATE"
            probability_class = (
                "confidence-medium"
            )
        else:
            probability_text = "CLOSE"
            probability_class = (
                "confidence-low"
            )

        home_stats = (
            current_team_features
            .filter(
                pl.col("team") == home
            )
            .row(
                0,
                named=True,
            )
        )

        away_stats = (
            current_team_features
            .filter(
                pl.col("team") == away
            )
            .row(
                0,
                named=True,
            )
        )

        home_record = (
            f"{int(home_stats['wins'])}-"
            f"{int(home_stats['games'] - home_stats['wins'])}"
        )

        away_record = (
            f"{int(away_stats['wins'])}-"
            f"{int(away_stats['games'] - away_stats['wins'])}"
        )

        away_bar_class = (
            "bar-fill-winner"
            if winner == away
            else "bar-fill-away"
        )

        home_bar_class = (
            "bar-fill-winner"
            if winner == home
            else "bar-fill-home"
        )

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

                        <div class="team-mark">
                            {away}
                        </div>

                        <div>

                            <div class="team-code">
                                {away}
                            </div>

                            <div class="team-name">
                                {TEAM_NAMES.get(away, away)}
                            </div>

                            <div class="team-record">
                                {away_record}
                            </div>

                        </div>

                    </div>


                    <div class="versus">
                        @
                    </div>


                    <div class="team team-right">

                        <div>

                            <div class="team-code">
                                {home}
                            </div>

                            <div class="team-name">
                                {TEAM_NAMES.get(home, home)}
                            </div>

                            <div class="team-record">
                                {home_record}
                            </div>

                        </div>

                        <div class="team-mark">
                            {home}
                        </div>

                    </div>

                </div>


                <div class="probability-section">

                    <div class="probability-row">

                        <div class="probability-team">
                            {away}
                        </div>

                        <div class="probability-value">
                            {away_prob:.1f}%
                        </div>

                    </div>

                    <div class="bar">
                        <div
                            class="{away_bar_class}"
                            style="width:{away_prob}%"
                        ></div>
                    </div>


                    <div class="probability-row">

                        <div class="probability-team">
                            {home}
                        </div>

                        <div class="probability-value">
                            {home_prob:.1f}%
                        </div>

                    </div>

                    <div class="bar">
                        <div
                            class="{home_bar_class}"
                            style="width:{home_prob}%"
                        ></div>
                    </div>

                </div>


                <div class="card-bottom">

                    <div class="card-stat">

                        <div class="card-stat-label">
                            Away PPG
                        </div>

                        <div class="card-stat-value">
                            {away_stats['ppg']:.1f}
                        </div>

                    </div>


                    <div class="card-stat">

                        <div class="card-stat-label">
                            Home PPG
                        </div>

                        <div class="card-stat-value">
                            {home_stats['ppg']:.1f}
                        </div>

                    </div>


                    <div class="card-stat">

                        <div class="card-stat-label">
                            EPA Edge
                        </div>

                        <div class="card-stat-value">
                            {row['off_epa_diff']:+.3f}
                        </div>

                    </div>

                </div>


                <div class="prediction-strip">

                    <div>

                        <div class="prediction-caption">
                            Model prediction
                        </div>

                        <div class="prediction-name">
                            {TEAM_NAMES.get(winner, winner)}
                        </div>

                    </div>

                    <div class="prediction-probability">
                        {probability:.1f}%
                    </div>

                </div>

            </div>
            """
        )

        # ----------------------------------------------------
        # DETAILS
        # ----------------------------------------------------

        with st.expander(
            f"Matchup details • {away} @ {home}"
        ):

            left, right = st.columns(
                2,
                gap="large",
            )

            with left:

                st.markdown(
                    f"### {away}"
                )

                st.metric(
                    "Record",
                    away_record,
                )

                st.metric(
                    "Points / Game",
                    f"{away_stats['ppg']:.1f}",
                )

                st.metric(
                    "Points Allowed",
                    f"{away_stats['papg']:.1f}",
                )

                st.metric(
                    "Recent Point Diff.",
                    f"{away_stats['rolling_point_diff_5']:+.1f}",
                )

            with right:

                st.markdown(
                    f"### {home}"
                )

                st.metric(
                    "Record",
                    home_record,
                )

                st.metric(
                    "Points / Game",
                    f"{home_stats['ppg']:.1f}",
                )

                st.metric(
                    "Points Allowed",
                    f"{home_stats['papg']:.1f}",
                )

                st.metric(
                    "Recent Point Diff.",
                    f"{home_stats['rolling_point_diff_5']:+.1f}",
                )

            st.markdown("---")

            st.markdown(
                "### Model factors"
            )

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
                    (
                        f"{row['home_win_pct'] - row['away_win_pct']:+.3f}"
                    ),
                    (
                        f"{row['rolling_point_diff_diff_5']:+.2f}"
                    ),
                    (
                        f"{row['off_epa_diff']:+.3f}"
                    ),
                    (
                        f"{row['def_epa_diff']:+.3f}"
                    ),
                    (
                        f"{row['rolling_pass_epa_diff_5']:+.3f}"
                    ),
                    (
                        f"{row['rolling_rush_epa_diff_5']:+.3f}"
                    ),
                ],
            })

            st.dataframe(
                factor_data,
                hide_index=True,
                use_container_width=True,
            )


# ============================================================
# TEAM STATS PAGE
# ============================================================

elif page == "Team Stats":

    st.html(
        """
        <div class="section-header">

            <div>

                <div class="section-title">
                    Team Performance
                </div>

                <div class="section-subtitle">
                    2026 regular-season statistics
                </div>

            </div>

        </div>
        """
    )

    stats = (
        current_team_features
        .with_columns([
            (
                pl.col("wins").cast(pl.Utf8)
                +
                pl.lit("-")
                +
                (
                    pl.col("games")
                    -
                    pl.col("wins")
                ).cast(pl.Utf8)
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
        .sort(
            "win_pct",
            descending=True,
        )
        .rename({
            "team": "Team",
            "record": "Record",
            "win_pct": "Win %",
            "ppg": "PPG",
            "papg": "PAPG",
            "rolling_point_diff_5":
                "Recent Point Diff.",
        })
    )

    st.dataframe(
        stats,
        hide_index=True,
        use_container_width=True,
        column_config={
            "Win %": st.column_config.NumberColumn(
                format="%.1f%%"
            ),

            "PPG": st.column_config.NumberColumn(
                format="%.1f"
            ),

            "PAPG": st.column_config.NumberColumn(
                format="%.1f"
            ),

            "Recent Point Diff.": (
                st.column_config.NumberColumn(
                    format="%+.1f"
                )
            ),
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

                <div class="section-title">
                    Model
                </div>

                <div class="section-subtitle">
                    How the prediction engine works
                </div>

            </div>

        </div>
        """
    )

    a, b, c = st.columns(3)

    with a:

        st.html(
            """
            <div class="metric-card">

                <div class="metric-label">
                    Algorithm
                </div>

                <div class="metric-value">
                    Random Forest
                </div>

                <div class="metric-note">
                    Ensemble classifier
                </div>

            </div>
            """
        )

    with b:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Features
                </div>

                <div class="metric-value">
                    {len(V3_FEATURES)}
                </div>

                <div class="metric-note">
                    V3 feature set
                </div>

            </div>
            """
        )

    with c:

        st.html(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    Training
                </div>

                <div class="metric-value">
                    {training_min}–{training_max}
                </div>

                <div class="metric-note">
                    Regular season games
                </div>

            </div>
            """
        )

    st.html(
        """
        <div class="model-panel">

            <div class="model-title">
                Feature groups
            </div>

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

    st.markdown(
        "### V3 Feature Set"
    )

    feature_table = pl.DataFrame({

        "Feature Group": [
            "Season",
            "Season",
            "Season",

            "Recent Form",
            "Recent Form",
            "Recent Form",
            "Recent Form",

            "EPA",
            "EPA",
            "EPA",
            "EPA",
            "EPA",
            "EPA",
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
        NFL Game Predictor
        &nbsp;•&nbsp;
        Random Forest V3
        &nbsp;•&nbsp;
        NFLverse
    </div>
    """
)