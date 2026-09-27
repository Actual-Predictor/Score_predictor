import streamlit as st
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import Ridge
import os
import base64

# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="URC Rugby Predictor",
    page_icon="🏉",
    layout="wide"
)
#----------------------------------------------------------
# CALCULATED ALPHAS FROM calculate_aplhas
# ---------------------------------------------------------

home_alpha = 0.139421
away_alpha = 0.166147

#----------------------------------------------------------
# BACKGROUND IMAGE
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
BACKGROUND_IMAGE = os.path.join(BASE_DIR, "logos", "urcl.jpeg")

if os.path.exists(BACKGROUND_IMAGE):

    with open(BACKGROUND_IMAGE, "rb") as f:
        encoded_image = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(
                    rgba(0, 59, 92, 0.78),
                    rgba(0, 59, 92, 0.78)
                ),
                url("data:image/jpeg;base64,{encoded_image}");

            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

else:
    st.warning(f"Background image not found: {BACKGROUND_IMAGE}")

# ============================================================
# SETTINGS
# ============================================================

CURRENT_SEASON = "2026-27"

SEASON_WEIGHTS = {
    "2021-22": 0.50,
    "2022-23": 0.65,
    "2023-24": 0.80,
    "2024-25": 1.00,
    "2025-26": 1.50,
    CURRENT_SEASON: 3.00
}

CURRENT_RESULTS_FILE = os.path.join(BASE_DIR, "current_season_results.csv")

# ============================================================
# TITLE
# ============================================================

st.title("🏉 URC Rugby Predictor")

st.write(
    "Predict the expected score and match outcome using "
    "historical URC results and weighted current-season form."
)


# ============================================================
# LOAD HISTORICAL DATA
# ============================================================

@st.cache_data
def load_historical_data(file_modified_time):

    df = pd.read_csv(
        os.path.join(BASE_DIR, "urc_all_matches.csv")
    )

    df["date"] = pd.to_datetime(
        df["date"]
    )

    df = (
        df
        .sort_values("date")
        .reset_index(drop=True)
    )

    return df


historical_file = os.path.join(
    BASE_DIR,
    "urc_all_matches.csv"
)

historical_df = load_historical_data(
    os.path.getmtime(historical_file)
)

# ============================================================
# TEAM LOGOS
# ============================================================

LOGO_DIR = os.path.join(BASE_DIR, "logos")

TEAM_LOGOS = {
    "Benetton": "Benetton.png",
    "Bulls": "Bulls.png",
    "Cardiff": "Cardiff.png",
    "Connacht": "Connacht.png",
    "Dragons": "Dragons.png",
    "Edinburgh": "Edinburgh.png",
    "Glasgow Warriors": "Glasgow.png",
    "Leinster": "Leinster.png",
    "Lions": "Lions.png",
    "Munster": "Munster.png",
    "Ospreys": "Ospreys.png",
    "Scarlets": "Scarlets.png",
    "Sharks": "Sharks.png",
    "Stormers": "Stormers.png",
    "Ulster": "Ulster.png",
    "Zebre": "Zebre.png",
}


def get_logo(team):

    filename = TEAM_LOGOS.get(team)

    if filename:

        path = os.path.join(
            LOGO_DIR,
            filename
        )

        if os.path.exists(path):
            return path

    return None

def get_team_color(team):
    return TEAM_COLORS.get(team, "#003B5C")

TEAM_COLORS = {
    "Benetton": "#00843D",
    "Bulls": "#4B9CD3",
    "Cardiff": "#0072CE",
    "Connacht": "#006B3F",
    "Dragons": "#C8102E",
    "Edinburgh": "#7A1FA2",
    "Glasgow Warriors": "#003B5C",
    "Leinster": "#003DA5",
    "Lions": "#F9A800",
    "Munster": "#E2231A",
    "Ospreys": "#000000",
    "Scarlets": "#D71920",
    "Sharks": "#000000",
    "Stormers": "#0066A1",
    "Ulster": "#FFCD00",
    "Zebre": "#000000",
}


# ============================================================
# DASHBOARD STYLING
# ============================================================

st.markdown("""
<style>

    /* ========================================================
       MAIN WHITE DASHBOARD PANEL
       ======================================================== */

    [data-testid="stMainBlockContainer"] {
        background-color: rgba(255, 255, 255, 0.8);
        padding: 35px 45px 45px 45px;
        border-radius: 20px;
        margin-top: 25px;
        margin-bottom: 25px;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.20);
    }


    /* ========================================================
       MAIN TITLE
       ======================================================== */

    [data-testid="stMainBlockContainer"] h1 {
        color: #000000;
        font-weight: 800;
    }


    /* ========================================================
       MAIN HEADINGS
       ======================================================== */

    [data-testid="stMainBlockContainer"] h2,
    [data-testid="stMainBlockContainer"] h3,
    [data-testid="stMainBlockContainer"] h4 {
        color: #000000;
    }


    /* ========================================================
       NORMAL TEXT
       ======================================================== */

    [data-testid="stMainBlockContainer"] p {
        color: #000000;
    }


    /* ========================================================
       SELECTBOX LABELS
       ======================================================== */

    [data-testid="stMainBlockContainer"] label {
        color: #000000;
    }


    /* ========================================================
       SELECTBOX TEXT
       ======================================================== */

    [data-testid="stMainBlockContainer"] [data-baseweb="select"] {
        color: #000000;
    }


    /* ========================================================
       SELECTBOX SELECTED VALUE
       ======================================================== */

    [data-testid="stMainBlockContainer"] [data-baseweb="select"] * {
        color: #000000;
    }


    /* ========================================================
       BUTTON TEXT
       ======================================================== */

    [data-testid="stMainBlockContainer"] button {
        color: #000000;
    }


    /* ========================================================
       METRIC LABELS
       ======================================================== */

    [data-testid="stMainBlockContainer"] [data-testid="stMetricLabel"] {
        color: #000000;
    }


    /* ========================================================
       METRIC VALUES
       ======================================================== */

    [data-testid="stMainBlockContainer"] [data-testid="stMetricValue"] {
        color: #000000;
    }


    /* ========================================================
       METRIC DELTAS / SMALL TEXT
       ======================================================== */

    [data-testid="stMainBlockContainer"] [data-testid="stMetricDelta"] {
        color: #000000;
    }


    /* ========================================================
       CAPTION
       ======================================================== */

    [data-testid="stMainBlockContainer"] .stCaption {
        color: #000000;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    [data-testid="stMainBlockContainer"] hr {
        border-color: #D0D0D0;
    }


    /* ========================================================
       PREDICTION CARD
       ======================================================== */

    .prediction-card {
        border-radius: 18px;
        padding: 25px;
        margin-top: 10px;
        background-color: white;
        box-shadow: 0 4px 15px rgba(0,0,0,0.08);
    }


    /* ========================================================
       TEAM NAMES
       ======================================================== */

    .team-name {
        font-size: 24px;
        font-weight: 700;
        text-align: center;
    }


    /* ========================================================
       BIG SCORE
       ======================================================== */

    .big-score {
        font-size: 55px;
        font-weight: 800;
        text-align: center;
    }


    /* ========================================================
       VS
       ======================================================== */

    .versus {
        font-size: 30px;
        font-weight: 700;
        text-align: center;
        padding-top: 45px;
    }

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD CURRENT-SEASON RESULTS
# ============================================================

def load_current_results():

    try:

        current_df = pd.read_csv(
            CURRENT_RESULTS_FILE
        )

        if len(current_df) > 0:

            current_df["date"] = pd.to_datetime(
                current_df["date"]
            )

        return current_df

    except FileNotFoundError:

        return pd.DataFrame(
            columns=[
                "date",
                "season",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "playoff"
            ]
        )


current_df = load_current_results()


# ============================================================
# TEAM FORM FUNCTION
# ============================================================

def get_team_form(history, team, venue=None, n=5):

    matches = history[
        (history["home_team"] == team) |
        (history["away_team"] == team)
    ].copy()

    if venue == "home":

        matches = matches[
            matches["home_team"] == team
        ]

    elif venue == "away":

        matches = matches[
            matches["away_team"] == team
        ]

    matches = matches.sort_values("date").tail(n)

    # --------------------------------------------------------
    # NO HISTORY
    # --------------------------------------------------------

    if len(matches) == 0:

        return {
            "scored": 25,
            "conceded": 25,
            "win_rate": 0.5,
            "margin": 0
        }

    scored = []
    conceded = []
    wins = []
    margins = []

    for _, row in matches.iterrows():

        if row["home_team"] == team:

            scored.append(row["home_score"])
            conceded.append(row["away_score"])

            margins.append(
                row["home_score"] -
                row["away_score"]
            )

            wins.append(
                row["home_score"] >
                row["away_score"]
            )

        else:

            scored.append(row["away_score"])
            conceded.append(row["home_score"])

            margins.append(
                row["away_score"] -
                row["home_score"]
            )

            wins.append(
                row["away_score"] >
                row["home_score"]
            )

    return {
        "scored": np.mean(scored),
        "conceded": np.mean(conceded),
        "win_rate": np.mean(wins),
        "margin": np.mean(margins)
    }


# ============================================================
# TRAIN FINAL MODELS
# ============================================================

@st.cache_resource
def train_models(data):

    # --------------------------------------------------------
    # CREATE SEQUENTIAL FORM DATA
    # --------------------------------------------------------

    history = pd.DataFrame(
        columns=data.columns
    )

    ridge_rows = []

    for _, row in data.iterrows():

        home = row["home_team"]
        away = row["away_team"]

        home_form = get_team_form(
            history,
            home,
            venue="home",
            n=5
        )

        away_form = get_team_form(
            history,
            away,
            venue="away",
            n=5
        )

        # ----------------------------------------------------
        # SEASON WEIGHT
        # ----------------------------------------------------

        season_weight = SEASON_WEIGHTS.get(
            row["season"],
            1.0
        )

        ridge_rows.append({

            "home_score":
                row["home_score"],

            "away_score":
                row["away_score"],

            "home_scored":
                home_form["scored"],

            "home_conceded":
                home_form["conceded"],

            "home_win_rate":
                home_form["win_rate"],

            "home_margin":
                home_form["margin"],

            "away_scored":
                away_form["scored"],

            "away_conceded":
                away_form["conceded"],

            "away_win_rate":
                away_form["win_rate"],

            "away_margin":
                away_form["margin"],

            "attack_difference":
                home_form["scored"] -
                away_form["scored"],

            "defence_difference":
                away_form["conceded"] -
                home_form["conceded"],

            "win_rate_difference":
                home_form["win_rate"] -
                away_form["win_rate"],

            "margin_difference":
                home_form["margin"] -
                away_form["margin"],

            "weight":
                season_weight
        })

        # ----------------------------------------------------
        # ADD MATCH AFTER CALCULATING FORM
        # ----------------------------------------------------

        history = pd.concat(
            [
                history,
                row.to_frame().T
            ],
            ignore_index=True
        )

    ridge_train = pd.DataFrame(
        ridge_rows
    )

    # ========================================================
    # RIDGE MODEL
    # ========================================================

    features = [

        "home_scored",
        "home_conceded",
        "home_win_rate",
        "home_margin",

        "away_scored",
        "away_conceded",
        "away_win_rate",
        "away_margin",

        "attack_difference",
        "defence_difference",
        "win_rate_difference",
        "margin_difference"
    ]

    ridge_home = Ridge(alpha=10)

    ridge_away = Ridge(alpha=10)

    ridge_home.fit(
        ridge_train[features],
        ridge_train["home_score"],
        sample_weight=ridge_train["weight"]
    )

    ridge_away.fit(
        ridge_train[features],
        ridge_train["away_score"],
        sample_weight=ridge_train["weight"]
    )

    # ========================================================
    # ATTACK + DEFENCE MODEL
    # ========================================================

    long_rows = []

    for _, row in data.iterrows():

        season_weight = SEASON_WEIGHTS.get(
            row["season"],
            1.0
        )

        # Home scoring observation

        long_rows.append({

            "score":
                row["home_score"],

            "home":
                1,

            "team":
                row["home_team"],

            "opponent":
                row["away_team"],

            "weight":
                season_weight
        })

        # Away scoring observation

        long_rows.append({

            "score":
                row["away_score"],

            "home":
                0,

            "team":
                row["away_team"],

            "opponent":
                row["home_team"],

            "weight":
                season_weight
        })

    long_data = pd.DataFrame(
        long_rows
    )

    attack_defence = smf.glm(

        formula=
            "score ~ home + C(team) + C(opponent)",

        data=
            long_data,

        family=
            sm.families.Poisson(),

        freq_weights=
            long_data["weight"]

    ).fit()

    return (
        ridge_home,
        ridge_away,
        attack_defence,
        features
    )


# ============================================================
# COMBINE ALL DATA
# ============================================================

all_data = pd.concat(
    [
        historical_df,
        current_df
    ],
    ignore_index=True
)

all_data["date"] = pd.to_datetime(
    all_data["date"]
)

all_data = (
    all_data
    .sort_values("date")
    .reset_index(drop=True)
)


# ============================================================
# TRAIN MODELS
# ============================================================

with st.spinner(
    "Training weighted prediction model..."
):

    (
        ridge_home,
        ridge_away,
        attack_defence,
        features
    ) = train_models(all_data)

# SHOW ALPHAS IN SIDEBAR
st.sidebar.divider()
st.sidebar.subheader("📊 Model Dispersion")

st.sidebar.write(f"**Home α:** {home_alpha:.4f}")
st.sidebar.write(f"**Away α:** {away_alpha:.4f}")
# ============================================================
# TEAM LIST
# ============================================================

teams = sorted(
    set(all_data["home_team"]) |
    set(all_data["away_team"])
)


# ============================================================
# SIDEBAR — ADD CURRENT RESULT
# ============================================================

st.sidebar.header(
    "➕ Add Current Season Result"
)

st.sidebar.write(
    f"Season: **{CURRENT_SEASON}**"
)

result_date = st.sidebar.date_input(
    "Match date"
)

result_home = st.sidebar.selectbox(
    "Home team",
    teams,
    key="result_home"
)

result_away = st.sidebar.selectbox(
    "Away team",
    teams,
    index=1 if len(teams) > 1 else 0,
    key="result_away"
)

result_home_score = st.sidebar.number_input(
    "Home score",
    min_value=0,
    max_value=150,
    value=20,
    step=1
)

result_away_score = st.sidebar.number_input(
    "Away score",
    min_value=0,
    max_value=150,
    value=20,
    step=1
)


# ============================================================
# ADD RESULT BUTTON
# ============================================================

add_result = st.sidebar.button(
    "➕ Add Result",
    width="stretch"
)


if add_result:

    if result_home == result_away:

        st.sidebar.error(
            "Home and away teams must be different."
        )

    else:

        new_result = pd.DataFrame([{

            "date":
                pd.to_datetime(result_date),

            "season":
                CURRENT_SEASON,

            "home_team":
                result_home,

            "away_team":
                result_away,

            "home_score":
                result_home_score,

            "away_score":
                result_away_score,

            "playoff":
                False
        }])

        # ----------------------------------------------------
        # CHECK FOR DUPLICATE
        # ----------------------------------------------------

        duplicate = False

        if len(current_df) > 0:

            duplicate = (
                (
                    current_df["date"] ==
                    pd.to_datetime(result_date)
                )
                &
                (
                    current_df["home_team"] ==
                    result_home
                )
                &
                (
                    current_df["away_team"] ==
                    result_away
                )
            ).any()

        if duplicate:

            st.sidebar.warning(
                "This match has already been added."
            )

        else:

            current_df = pd.concat(
                [
                    current_df,
                    new_result
                ],
                ignore_index=True
            )

            current_df.to_csv(
                CURRENT_RESULTS_FILE,
                index=False
            )

            # Clear cached model so the new result
            # is immediately incorporated.

            st.cache_resource.clear()

            st.sidebar.success(
                "Result added successfully!"
            )

            st.rerun()


# ============================================================
# CURRENT SEASON RESULTS
# ============================================================

if len(current_df) > 0:

    st.sidebar.divider()

    st.sidebar.subheader(
        "📋 Current Season Results"
    )

    display_results = (
        current_df
        .sort_values("date", ascending=False)
        [
            [
                "date",
                "home_team",
                "home_score",
                "away_score",
                "away_team"
            ]
        ]
        .copy()
    )

    display_results["date"] = (
        display_results["date"]
        .dt.strftime("%d/%m/%Y")
    )

    st.sidebar.dataframe(
        display_results,
        hide_index=True,
        width="stretch"
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

st.sidebar.divider()

st.sidebar.subheader(
    "⚖️ Season Weights"
)

for season, weight in SEASON_WEIGHTS.items():

    st.sidebar.write(
        f"**{season}:** {weight:.2f}×"
    )


# ============================================================
# MAIN MATCH SELECTION
# ============================================================

st.subheader(
    "Select your match"
)

col1, col2 = st.columns(2)

with col1:

    home_team = st.selectbox(
        "🏠 Home Team",
        teams,
        index=
            teams.index("Bulls")
            if "Bulls" in teams
            else 0,

        key="prediction_home"
    )

with col2:

    away_team = st.selectbox(
        "✈️ Away Team",
        teams,

        index=
            teams.index("Stormers")
            if "Stormers" in teams
            else 1,

        key="prediction_away"
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================

predict_button = st.button(
    "🔮 Predict Match",
    width="stretch"
)


if predict_button:

    if home_team == away_team:

        st.error(
            "Please select two different teams."
        )

        st.stop()

    # ========================================================
    # CREATE HISTORICAL DATA
    # ========================================================

    history = all_data.copy()

    # ========================================================
    # RIDGE FEATURES
    # ========================================================

    home_form = get_team_form(
        history,
        home_team,
        venue="home",
        n=5
    )

    away_form = get_team_form(
        history,
        away_team,
        venue="away",
        n=5
    )

    input_data = pd.DataFrame([{

        "home_scored":
            home_form["scored"],

        "home_conceded":
            home_form["conceded"],

        "home_win_rate":
            home_form["win_rate"],

        "home_margin":
            home_form["margin"],

        "away_scored":
            away_form["scored"],

        "away_conceded":
            away_form["conceded"],

        "away_win_rate":
            away_form["win_rate"],

        "away_margin":
            away_form["margin"],

        "attack_difference":
            home_form["scored"] -
            away_form["scored"],

        "defence_difference":
            away_form["conceded"] -
            home_form["conceded"],

        "win_rate_difference":
            home_form["win_rate"] -
            away_form["win_rate"],

        "margin_difference":
            home_form["margin"] -
            away_form["margin"]
    }])


    # ========================================================
    # RIDGE PREDICTION
    # ========================================================

    ridge_home_pred = ridge_home.predict(
        input_data
    )[0]

    ridge_away_pred = ridge_away.predict(
        input_data
    )[0]


    # ========================================================
    # ATTACK + DEFENCE PREDICTION
    # ========================================================

    ad_home_data = pd.DataFrame([{

        "home":
            1,

        "team":
            home_team,

        "opponent":
            away_team
    }])

    ad_away_data = pd.DataFrame([{

        "home":
            0,

        "team":
            away_team,

        "opponent":
            home_team
    }])


    ad_home_pred = attack_defence.predict(
        ad_home_data
    )[0]

    ad_away_pred = attack_defence.predict(
        ad_away_data
    )[0]


    # ========================================================
    # FINAL SCORE MODEL
    #
    # 75% RIDGE
    # 25% ATTACK + DEFENCE
    # ========================================================

    expected_home = (

        0.75 *
        ridge_home_pred

        +

        0.25 *
        ad_home_pred
    )

    expected_away = (

        0.75 *
        ridge_away_pred

        +

        0.25 *
        ad_away_pred
    )

    simulations = 100000


    # ========================================================
    # NEGATIVE BINOMIAL DISPERSION
    # ========================================================




    # ========================================================
    # SIMULATION
    # ========================================================

    simulations = 10000

    home_n = 1 / home_alpha

    home_p = 1 / (
        1 +
        home_alpha *
        expected_home
    )

    away_n = 1 / away_alpha

    away_p = 1 / (
        1 +
        away_alpha *
        expected_away
    )


    home_scores = np.random.negative_binomial(
        home_n,
        home_p,
        simulations
    )

    away_scores = np.random.negative_binomial(
        away_n,
        away_p,
        simulations
    )


    # ========================================================
    # PROBABILITIES
    # ========================================================

    home_win_probability = np.mean(
        home_scores >
        away_scores
    )

    draw_probability = np.mean(
        home_scores ==
        away_scores
    )

    away_win_probability = np.mean(
        home_scores <
        away_scores
    )


    # ========================================================
    # MOST LIKELY SCORE
    # ========================================================

    score_pairs = list(
        zip(
            home_scores,
            away_scores
        )
    )

    score_counts = (
        pd.Series(score_pairs)
        .value_counts()
    )

    most_likely_score = (
        score_counts.index[0]
    )


    # ========================================================
    # EXPECTED MARGIN / TOTAL
    # ========================================================

    expected_margin = np.mean(
        home_scores -
        away_scores
    )

    expected_total = np.mean(
        home_scores +
        away_scores
    )


    # ========================================================
    # DETERMINE WINNER
    # ========================================================

    probabilities = {

        home_team:
            home_win_probability,

        "Draw":
            draw_probability,

        away_team:
            away_win_probability
    }

    predicted_result = max(
        probabilities,
        key=probabilities.get
    )


    # ========================================================
    # DISPLAY RESULT
    # ========================================================

    st.divider()

    st.subheader(
        "🏉 Prediction"
    )

    # ========================================================
    # TEAM COLOURS
    # ========================================================

    home_color = get_team_color(home_team)
    away_color = get_team_color(away_team)

    # ---------------------------------------------------------
    # SCORE + TEAM LOGOS
    # ---------------------------------------------------------

    home_color = get_team_color(home_team)
    away_color = get_team_color(away_team)

    score_col1, score_col2, score_col3 = st.columns([2, 1, 2])

    with score_col1:

        st.markdown(
            f"""
            <h2 style="
                text-align:center;
                color:{home_color};
                margin-bottom:5px;
            ">
                {home_team}
            </h2>
            """,
            unsafe_allow_html=True
        )

        home_logo = get_logo(home_team)

        if home_logo:
            logo_left, logo_center, logo_right = st.columns([1.3, 2, 0.7])

            with logo_center:
                st.image(home_logo, width=140)

        st.markdown(
            f"""
            <h1 style="
                text-align:center;
                color:{home_color};
                font-size:52px;
                margin-top:-20px;
            ">
                {round(expected_home)}
            </h1>
            """,
            unsafe_allow_html=True
        )

    with score_col2:

        st.markdown(
            """
            <div style="text-align:center; padding-top:100px;">
                <h2 style="color:#003B5C; font-size:28px;">
                    VS
                </h2>
            </div>
            """,
            unsafe_allow_html=True
        )

    with score_col3:

        st.markdown(
            f"""
            <h2 style="
                text-align:center;
                color:{away_color};
                margin-bottom:5px;
            ">
                {away_team}
            </h2>
            """,
            unsafe_allow_html=True
        )

        away_logo = get_logo(away_team)

        if away_logo:
            logo_left, logo_center, logo_right = st.columns([1.3, 2, 0.7])

            with logo_center:
                st.image(away_logo, width=140)

        st.markdown(
            f"""
            <h1 style="
                text-align:center;
                color:{away_color};
                font-size:52px;
                margin-top:-20px;
            ">
                {round(expected_away)}
            </h1>
            """,
            unsafe_allow_html=True
        )
    # ========================================================
    # WINNER
    # ========================================================

    st.success(
        f"Predicted result: **{predicted_result}**"
    )


    # ========================================================
    # PROBABILITY COLUMNS
    # ========================================================

    st.subheader(
        "Match probabilities"
    )

    p1, p2, p3 = st.columns(3)

    with p1:

        st.markdown(
            f"""
            <div style="
                border-top: 5px solid {home_color};
                padding-top: 10px;
            ">
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "🏠 Home Win",
            f"{home_win_probability * 100:.1f}%"
        )
    with p2:

        st.metric(
            "🤝 Draw",
            f"{draw_probability * 100:.1f}%"
        )

    with p3:

        st.markdown(
            f"""
            <div style="
                border-top: 5px solid {away_color};
                padding-top: 10px;
            ">
            """,
            unsafe_allow_html=True
        )

        st.metric(
            "✈️ Away Win",
            f"{away_win_probability * 100:.1f}%"
        )

    # ========================================================
    # OTHER INFORMATION
    # ========================================================

    st.subheader(
        "Simulation results"
    )

    info1, info2, info3 = st.columns(3)

    with info1:

        st.metric(
            "Most likely score",
            f"{most_likely_score[0]} – "
            f"{most_likely_score[1]}"
        )

    with info2:

        st.metric(
            "Expected margin",
            f"{expected_margin:+.1f}"
        )

    with info3:

        st.metric(
            "Expected total points",
            f"{expected_total:.1f}"
        )


    # ========================================================
    # CURRENT SEASON INFORMATION
    # ========================================================

    st.divider()

    current_match_count = len(
        current_df
    )

    historical_match_count = len(
        historical_df
    )

    info1, info2 = st.columns(2)

    with info1:

        st.metric(
            "Historical matches",
            historical_match_count
        )

    with info2:

        st.metric(
            "Current-season matches",
            current_match_count
        )


    st.caption(
        "The current season is weighted at "
        "3.00× the weighting of the 2024–25 season. "
        "Predictions use 10,000 Negative Binomial simulations."
    )


