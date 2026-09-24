import streamlit as st
import polars as pl
import nflreadpy as nfl

from sklearn.ensemble import RandomForestClassifier


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="NFL Game Predictor test",
    page_icon="🏈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CONFIGURATION
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

TARGET = "home_win"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        padding-top: 1rem;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 2rem;
    }

    .app-title {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0;
    }

    .app-subtitle {
        font-size: 1.1rem;
        color: #888;
        margin-bottom: 2rem;
    }

    .game-card {
        background: #111827;
        border: 1px solid #374151;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
    }

    .game-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        color: #9ca3af;
        font-size: 0.9rem;
        margin-bottom: 25px;
    }

    .matchup {
        display: grid;
        grid-template-columns: 1fr 80px 1fr;
        align-items: center;
        text-align: center;
        margin-bottom: 25px;
    }

    .team-name {
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .away-label,
    .home-label {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .at-symbol {
        font-size: 1.5rem;
        color: #6b7280;
    }

    .probability-row {
        display: flex;
        justify-content: space-between;
        margin-bottom: 6px;
        font-size: 0.95rem;
    }

    .probability-bar {
        height: 12px;
        width: 100%;
        background: #374151;
        border-radius: 999px;
        overflow: hidden;
        margin-bottom: 18px;
    }

    
    .probability-fill {
    height: 100%;
    border-radius: 999px;
    }

    .away-fill {
    background: #ef4444;
    }

    .home-fill {
    background: #3b82f6;
    }
    .prediction-box {
        background: #1f2937;
        border-radius: 10px;
        padding: 14px;
        text-align: center;
        margin-top: 20px;
    }

    .prediction-label {
        color: #9ca3af;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .prediction-team {
        font-size: 1.4rem;
        font-weight: 800;
        margin-top: 4px;
    }

    .confidence-high {
        color: #22c55e;
        font-weight: 700;
    }

    .confidence-medium {
        color: #f59e0b;
        font-weight: 700;
    }

    .confidence-low {
        color: #ef4444;
        font-weight: 700;
    }

    .section-title {
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    .factor-positive {
        color: #22c55e;
        font-weight: 700;
    }

    .factor-negative {
        color: #ef4444;
        font-weight: 700;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DATA LOADING
# ============================================================

@st.cache_data
def load_historical_data():
    return pl.read_parquet(
        "data/processed/model_data.parquet"
    )


@st.cache_resource
def train_model():

    model_data = load_historical_data()

    historical_v3 = model_data.drop_nulls(
        subset=V3_FEATURES
    )

    X_train = historical_v3.select(
        V3_FEATURES
    ).to_numpy()

    y_train = historical_v3.select(
        TARGET
    ).to_numpy().ravel()

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=10,
        random_state=42,
    )

    model.fit(X_train, y_train)

    return model


@st.cache_data(ttl=3600)
def load_2026_schedule():

    schedule = nfl.load_schedules(
        seasons=[2026]
    )

    return schedule.filter(
        pl.col("game_type") == "REG"
    )


@st.cache_data(ttl=3600)
def load_2026_pbp():

    pbp = nfl.load_pbp(
        seasons=[2026]
    )

    return pbp.filter(
        pl.col("season_type") == "REG"
    )


# ============================================================
# TEAM GAME DATA
# ============================================================

def build_team_games(games):

    home_games = games.select([
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",

        pl.col("home_team").alias("team"),

        pl.col("home_score").alias(
            "points_for"
        ),

        pl.col("away_score").alias(
            "points_against"
        ),

        (pl.col("result") > 0)
        .cast(pl.Int8)
        .alias("win"),
    ])

    away_games = games.select([
        "game_id",
        "season",
        "week",
        "home_team",
        "away_team",

        pl.col("away_team").alias("team"),

        pl.col("away_score").alias(
            "points_for"
        ),

        pl.col("home_score").alias(
            "points_against"
        ),

        (pl.col("result") < 0)
        .cast(pl.Int8)
        .alias("win"),
    ])

    return (
        pl.concat([
            home_games,
            away_games,
        ])
        .sort([
            "season",
            "team",
            "week",
        ])
    )


# ============================================================
# CURRENT TEAM FEATURES
# ============================================================

def build_current_team_features(games):

    team_games = build_team_games(games)

    current = (
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
                / pl.col("games")
            ).alias("win_pct"),

            (
                pl.col("points_for")
                / pl.col("games")
            ).alias("ppg"),

            (
                pl.col("points_against")
                / pl.col("games")
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
            .alias("rolling_win_pct_5"),

            pl.col("points_for")
            .tail(5)
            .mean()
            .alias("rolling_ppg_5"),

            pl.col("points_against")
            .tail(5)
            .mean()
            .alias("rolling_papg_5"),

            (
                (
                    pl.col("points_for")
                    - pl.col("points_against")
                )
                .tail(5)
                .mean()
            )
            .alias("rolling_point_diff_5"),
        ])
    )

    return current.join(
        rolling,
        on="team",
        how="left",
    )


# ============================================================
# EPA FEATURES
# ============================================================

def build_current_epa_features(
    pbp,
    completed_game_ids,
):

    completed_pbp = pbp.join(
        completed_game_ids,
        on="game_id",
        how="inner",
    )

    offensive_plays = completed_pbp.filter(
        pl.col("posteam").is_not_null()
    )

    offense = (
        offensive_plays
        .group_by([
            "game_id",
            "week",
            "posteam",
        ])
        .agg([
            pl.col("epa")
            .sum()
            .alias("off_epa"),

            pl.col("epa")
            .mean()
            .alias("off_epa_per_play"),

            pl.len()
            .alias("off_plays"),

            pl.when(
                pl.col("pass_attempt") == 1
            )
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("pass_epa_per_play"),

            pl.when(
                pl.col("rush_attempt") == 1
            )
            .then(pl.col("epa"))
            .otherwise(None)
            .mean()
            .alias("rush_epa_per_play"),
        ])
        .rename({
            "posteam": "team"
        })
    )

    defense = (
        offensive_plays
        .group_by([
            "game_id",
            "week",
            "defteam",
        ])
        .agg([
            pl.col("epa")
            .sum()
            .alias("def_epa_allowed"),

            pl.col("epa")
            .mean()
            .alias(
                "def_epa_allowed_per_play"
            ),

            pl.len()
            .alias("def_plays"),
        ])
        .rename({
            "defteam": "team"
        })
    )

    team_epa = (
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
        team_epa
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


# ============================================================
# BUILD UPCOMING GAMES
# ============================================================

def build_prediction_dataset(
    schedule,
    pbp,
):

    completed = schedule.filter(
        pl.col("home_score").is_not_null()
        & pl.col("away_score").is_not_null()
    )

    upcoming = schedule.filter(
        pl.col("home_score").is_null()
        | pl.col("away_score").is_null()
    )

    team_features = build_current_team_features(
        completed
    )

    home_features = team_features.rename({
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

    away_features = team_features.rename({
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

    predictions = (
        upcoming
        .join(
            home_features,
            on="home_team",
            how="left",
        )
        .join(
            away_features,
            on="away_team",
            how="left",
        )
    )

    predictions = predictions.with_columns([
        (
            pl.col("home_win_pct")
            - pl.col("away_win_pct")
        ).alias("win_pct_diff"),

        (
            pl.col("home_ppg")
            - pl.col("away_ppg")
        ).alias("ppg_diff"),

        (
            pl.col("away_papg")
            - pl.col("home_papg")
        ).alias("defense_diff"),

        (
            pl.col("home_rolling_win_pct_5")
            - pl.col("away_rolling_win_pct_5")
        ).alias(
            "rolling_win_pct_diff_5"
        ),

        (
            pl.col("home_rolling_ppg_5")
            - pl.col("away_rolling_ppg_5")
        ).alias(
            "rolling_ppg_diff_5"
        ),

        (
            pl.col("away_rolling_papg_5")
            - pl.col("home_rolling_papg_5")
        ).alias(
            "rolling_defense_diff_5"
        ),

        (
            pl.col("home_rolling_point_diff_5")
            - pl.col("away_rolling_point_diff_5")
        ).alias(
            "rolling_point_diff_diff_5"
        ),
    ])

    completed_ids = completed.select(
        "game_id"
    ).unique()

    epa_features = build_current_epa_features(
        pbp,
        completed_ids,
    )

    home_epa = epa_features.rename({
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

    away_epa = epa_features.rename({
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

    predictions = (
        predictions
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

    predictions = predictions.with_columns([
        (
            pl.col("home_off_epa")
            - pl.col("away_off_epa")
        ).alias("off_epa_diff"),

        (
            pl.col("away_def_epa")
            - pl.col("home_def_epa")
        ).alias("def_epa_diff"),

        (
            pl.col("home_rolling_off_epa_5")
            - pl.col("away_rolling_off_epa_5")
        ).alias(
            "rolling_off_epa_diff_5"
        ),

        (
            pl.col("away_rolling_def_epa_5")
            - pl.col("home_rolling_def_epa_5")
        ).alias(
            "rolling_def_epa_diff_5"
        ),

        (
            pl.col("home_rolling_pass_epa_5")
            - pl.col("away_rolling_pass_epa_5")
        ).alias(
            "rolling_pass_epa_diff_5"
        ),

        (
            pl.col("home_rolling_rush_epa_5")
            - pl.col("away_rolling_rush_epa_5")
        ).alias(
            "rolling_rush_epa_diff_5"
        ),
    ])

    return predictions


# ============================================================
# PREDICTIONS
# ============================================================

def generate_predictions(
    prediction_data,
    model,
):

    clean = prediction_data.drop_nulls(
        subset=V3_FEATURES
    )

    X = clean.select(
        V3_FEATURES
    ).to_numpy()

    probabilities = (
        model.predict_proba(X)[:, 1]
    )

    result = (
        clean
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

            "home_rolling_point_diff_5",
            "away_rolling_point_diff_5",

            "home_off_epa",
            "away_off_epa",

            "home_def_epa",
            "away_def_epa",

            *V3_FEATURES,
        ])
        .with_columns(
            pl.Series(
                "home_win_probability",
                probabilities,
            )
        )
        .with_columns([
            (
                1
                - pl.col(
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
            .alias("predicted_winner"),
        ])
        .sort([
            "gameday",
            "gametime",
        ])
    )

    return result


# ============================================================
# CONFIDENCE
# ============================================================

def get_confidence(probability):

    distance = abs(
        probability - 0.5
    )

    if distance >= 0.30:
        return "Very High", "confidence-high"

    if distance >= 0.20:
        return "High", "confidence-high"

    if distance >= 0.10:
        return "Medium", "confidence-medium"

    return "Low", "confidence-low"


# ============================================================
# FEATURE EXPLANATION
# ============================================================

def get_feature_factors(row):

    factors = []

    feature_labels = {
        "win_pct_diff":
            "Season win rate",

        "ppg_diff":
            "Points per game",

        "defense_diff":
            "Points allowed",

        "rolling_win_pct_diff_5":
            "Recent win rate",

        "rolling_ppg_diff_5":
            "Recent scoring",

        "rolling_defense_diff_5":
            "Recent defense",

        "rolling_point_diff_diff_5":
            "Recent point differential",

        "off_epa_diff":
            "Offensive EPA",

        "def_epa_diff":
            "Defensive EPA",

        "rolling_off_epa_diff_5":
            "Recent offensive EPA",

        "rolling_def_epa_diff_5":
            "Recent defensive EPA",

        "rolling_pass_epa_diff_5":
            "Passing EPA",

        "rolling_rush_epa_diff_5":
            "Rushing EPA",
    }

    for feature in V3_FEATURES:

        value = row[feature]

        if value is None:
            continue

        try:
            value = float(value)
        except (TypeError, ValueError):
            continue

        if abs(value) < 0.001:
            continue

        factors.append({
            "feature": feature,
            "label": feature_labels.get(
                feature,
                feature
            ),
            "value": value,
        })

    factors.sort(
        key=lambda x: abs(x["value"]),
        reverse=True,
    )

    return factors[:5]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("NFL Game Predictor")

    st.markdown(
        """
        **Model**

        Random Forest

        **Feature set**

        V3 — Team Form + EPA

        **Training data**

        2015–2025

        **Current season**

        2026
        """
    )

    st.divider()

    if st.button(
        "Refresh data",
        use_container_width=True,
    ):
        st.cache_data.clear()
        st.rerun()


# ============================================================
# LOAD EVERYTHING
# ============================================================

st.markdown(
    '<div class="app-title">NFL Game Predictor</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="app-subtitle">'
    'Machine-learning predictions for the 2026 NFL season'
    '</div>',
    unsafe_allow_html=True,
)


try:

    model = train_model()

    schedule = load_2026_schedule()

    pbp = load_2026_pbp()

    prediction_data = build_prediction_dataset(
        schedule,
        pbp,
    )

    predictions = generate_predictions(
        prediction_data,
        model,
    )

except Exception as e:

    st.error(
        "Something went wrong while building "
        "the prediction dataset."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SUMMARY METRICS
# ============================================================

completed_games = schedule.filter(
    pl.col("home_score").is_not_null()
    & pl.col("away_score").is_not_null()
)

upcoming_games = schedule.filter(
    pl.col("home_score").is_null()
    | pl.col("away_score").is_null()
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "2026 Games Played",
        len(completed_games),
    )

with col2:
    st.metric(
        "Upcoming Games",
        len(upcoming_games),
    )

with col3:
    st.metric(
        "Model",
        "Random Forest",
    )

with col4:
    st.metric(
        "Features",
        len(V3_FEATURES),
    )


# ============================================================
# WEEK FILTER
# ============================================================

available_weeks = (
    predictions
    .select("week")
    .unique()
    .sort("week")
    .to_series()
    .to_list()
)

if not available_weeks:
    st.warning(
        "No upcoming games are currently available."
    )
    st.stop()

selected_week = st.selectbox(
    "Select Week",
    available_weeks,
    format_func=lambda x: f"Week {x}",
)


week_predictions = predictions.filter(
    pl.col("week") == selected_week
)

# ============================================================
# PREDICTION CARDS
# ============================================================

st.markdown(
    f"""
    <div class="section-title">
        Week {selected_week} Predictions
    </div>
    """,
    unsafe_allow_html=True,
)


for row in week_predictions.iter_rows(named=True):

    home_probability = (
        float(row["home_win_probability"]) * 100
    )

    away_probability = (
        float(row["away_win_probability"]) * 100
    )

    predicted_winner = row["predicted_winner"]

    winning_probability = max(
        home_probability,
        away_probability,
    )

    confidence, confidence_class = get_confidence(
        winning_probability / 100
    )

    factors = get_feature_factors(row)

    # --------------------------------------------------------
    # GAME CARD
    # --------------------------------------------------------

    card_html = f"""
    <div class="game-card">

        <div class="game-header">

            <span>
                {row["gameday"]} • {row["gametime"]}
            </span>

            <span>
                Confidence:
                <b class="{confidence_class}">
                    {winning_probability:.1f}%
                    ({confidence})
                </b>
            </span>

        </div>


        <div class="matchup">

            <div>
                <div class="away-label">
                    Away
                </div>

                <div class="team-name">
                    {row["away_team"]}
                </div>
            </div>


            <div class="at-symbol">
                @
            </div>


            <div>
                <div class="home-label">
                    Home
                </div>

                <div class="team-name">
                    {row["home_team"]}
                </div>
            </div>

        </div>


        <div class="probability-row">
            <span>
                {row["away_team"]}
            </span>

            <strong>
                {away_probability:.1f}%
            </strong>
        </div>


        <div class="probability-bar">

            <div
                class="probability-fill away-fill"
                style="width: {away_probability}%"
            ></div>

        </div>


        <div class="probability-row">
            <span>
                {row["home_team"]}
            </span>

            <strong>
                {home_probability:.1f}%
            </strong>
        </div>


        <div class="probability-bar">

            <div
                class="probability-fill home-fill"
                style="width: {home_probability}%"
            ></div>

        </div>


        <div class="prediction-box">

            <div class="prediction-label">
                MODEL PREDICTION
            </div>

            <div class="prediction-team">
                {predicted_winner}
            </div>

        </div>

    </div>
    """

    st.html(card_html)


    # --------------------------------------------------------
    # GAME DETAILS
    # --------------------------------------------------------

    with st.expander(
        f"Why the model favors {predicted_winner}"
    ):

        detail_col1, detail_col2 = st.columns(2)


        with detail_col1:

            st.markdown(
                f"### {row['away_team']}"
            )

            st.metric(
                "Win Rate",
                f"{row['away_win_pct']:.1%}"
            )

            st.metric(
                "Points / Game",
                f"{row['away_ppg']:.1f}"
            )

            st.metric(
                "Points Allowed / Game",
                f"{row['away_papg']:.1f}"
            )

            st.metric(
                "Recent Point Differential",
                f"{row['away_rolling_point_diff_5']:.1f}"
            )


        with detail_col2:

            st.markdown(
                f"### {row['home_team']}"
            )

            st.metric(
                "Win Rate",
                f"{row['home_win_pct']:.1%}"
            )

            st.metric(
                "Points / Game",
                f"{row['home_ppg']:.1f}"
            )

            st.metric(
                "Points Allowed / Game",
                f"{row['home_papg']:.1f}"
            )

            st.metric(
                "Recent Point Differential",
                f"{row['home_rolling_point_diff_5']:.1f}"
            )


        st.markdown(
            "### Largest Feature Differences"
        )


        if factors:

            for factor in factors:

                value = factor["value"]

                if value > 0:

                    direction = (
                        f"{row['home_team']} "
                        f"+{value:.3f}"
                    )

                else:

                    direction = (
                        f"{row['away_team']} "
                        f"+{abs(value):.3f}"
                    )

                st.write(
                    f"**{factor['label']}** — "
                    f"{direction}"
                )

        else:

            st.write(
                "No feature differences available."
            )
# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "NFL Game Predictor • "
    "Random Forest trained on 2015–2025 "
    "regular-season data • "
    "V3 feature set"
)