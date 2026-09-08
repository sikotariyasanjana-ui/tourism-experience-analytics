# ============================================================
# TOURISM EXPERIENCE ANALYTICS
# Classification, Prediction & Recommendation System
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Tourism Experience Analytics",
    page_icon="🌍",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "DATA"
)


# ============================================================
# FILE SELECTION
# ============================================================

# Prefer the smaller deployment model.
DEPLOYMENT_CLASSIFIER = os.path.join(
    MODEL_DIR,
    "best_tourism_visitmode_deployment.joblib"
)

# Fallback to your existing 34.44 MB tuned model.
TUNED_CLASSIFIER = os.path.join(
    MODEL_DIR,
    "tourism_visitmode_tuned_model.pkl"
)

if os.path.exists(DEPLOYMENT_CLASSIFIER):
    CLASSIFIER_FILE = DEPLOYMENT_CLASSIFIER
elif os.path.exists(TUNED_CLASSIFIER):
    CLASSIFIER_FILE = TUNED_CLASSIFIER
else:
    CLASSIFIER_FILE = None


RECOMMENDATION_FILE = os.path.join(
    MODEL_DIR,
    "tourism_recommendation_tuned_model.pkl"
)

ATTRACTION_FILE = os.path.join(
    MODEL_DIR,
    "tourism_attraction_data.csv"
)

TRANSACTION_FILE = os.path.join(
    DATA_DIR,
    "Transaction.csv"
)

USER_FILE = os.path.join(
    DATA_DIR,
    "User.csv"
)

ITEM_FILE = os.path.join(
    DATA_DIR,
    "Item.csv"
)

MODE_FILE = os.path.join(
    DATA_DIR,
    "Mode.csv"
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_file(path):
    """Return True if file exists."""
    return path is not None and os.path.exists(path)


@st.cache_resource
def load_classifier(path):
    return joblib.load(path)


@st.cache_resource
def load_recommender(path):
    return joblib.load(path)


@st.cache_data
def load_csv(path):
    return pd.read_csv(
        path,
        encoding="latin1"
    )


# ============================================================
# TITLE
# ============================================================

st.title(
    "🌍 Tourism Experience Analytics"
)

st.subheader(
    "Classification, Prediction & Recommendation System"
)

st.write(
    """
    This application analyzes tourism data to predict a user's
    likely visit mode and provide personalized attraction
    recommendations based on previous tourism interactions.
    """
)

st.divider()


# ============================================================
# CHECK FILES
# ============================================================

missing_files = []

if not check_file(CLASSIFIER_FILE):
    missing_files.append(
        "VisitMode classification model"
    )

if not check_file(RECOMMENDATION_FILE):
    missing_files.append(
        "Recommendation model"
    )

if not check_file(ATTRACTION_FILE):
    missing_files.append(
        "Attraction data"
    )

if not check_file(TRANSACTION_FILE):
    missing_files.append(
        "Transaction.csv"
    )

if not check_file(USER_FILE):
    missing_files.append(
        "User.csv"
    )

if not check_file(ITEM_FILE):
    missing_files.append(
        "Item.csv"
    )

if not check_file(MODE_FILE):
    missing_files.append(
        "Mode.csv"
    )


if missing_files:

    st.error(
        "Required deployment files are missing."
    )

    st.write("Missing files:")

    for file_name in missing_files:
        st.write(f"- {file_name}")

    st.info(
        "Check that the required files are inside "
        "models/ and DATA/ folders."
    )

    st.stop()


# ============================================================
# LOAD MODELS AND DATA
# ============================================================

try:

    classification_model = load_classifier(
        CLASSIFIER_FILE
    )

    recommendation_model = load_recommender(
        RECOMMENDATION_FILE
    )

    transaction_df = load_csv(
        TRANSACTION_FILE
    )

    user_df = load_csv(
        USER_FILE
    )

    item_df = load_csv(
        ITEM_FILE
    )

    mode_df = load_csv(
        MODE_FILE
    )

    attraction_df = load_csv(
        ATTRACTION_FILE
    )

except Exception as e:

    st.error(
        "Error while loading models or datasets."
    )

    st.exception(e)

    st.stop()


# ============================================================
# SIDEBAR NAVIGATION
# IMPORTANT: UNIQUE KEY FIX
# ============================================================

st.sidebar.title(
    "📌 Navigation"
)

page = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "🤖 VisitMode Prediction",
        "🎯 Attraction Recommendation",
        "⭐ Rating Analysis",
        "📊 Tourism Analytics"
    ],
    key="main_navigation"
)


# ============================================================
# HOME
# ============================================================

if page == "🏠 Home":

    st.header(
        "Welcome to Tourism Experience Analytics"
    )

    st.write(
        """
        Our system combines machine learning and tourism
        analytics to understand user behavior, predict
        visit modes, and recommend suitable attractions.
        """
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Transactions",
            f"{len(transaction_df):,}"
        )

    with col2:
        st.metric(
            "Users",
            f"{user_df['UserId'].nunique():,}"
        )

    with col3:
        st.metric(
            "Attractions",
            f"{item_df['AttractionId'].nunique():,}"
        )

    with col4:

        if "Rating" in transaction_df.columns:

            ratings = pd.to_numeric(
                transaction_df["Rating"],
                errors="coerce"
            )

            st.metric(
                "Average Rating",
                f"{ratings.mean():.2f}"
            )

    st.divider()

    st.subheader(
        "Project Objectives"
    )

    st.write(
        """
        **1. Rating Prediction**

        Analyze tourist attraction ratings and user
        satisfaction.

        **2. VisitMode Classification**

        Predict the likely mode of visit such as Business,
        Family, Couples, Friends, etc.

        **3. Personalized Recommendation**

        Recommend attractions using users' historical
        attraction ratings.
        """
    )

    st.success(
        "Use the navigation menu on the left to explore the application."
    )


# ============================================================
# VISIT MODE PREDICTION
# ============================================================

elif page == "🤖 VisitMode Prediction":

    st.header(
        "🤖 Visit Mode Prediction"
    )

    st.write(
        """
        Select a user, visit period and attraction.
        The Random Forest classification model will predict
        the likely VisitMode.
        """
    )

    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    user_ids = (
        user_df["UserId"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "Select User",
        user_ids,
        key="prediction_user"
    )

    selected_user_data = (
        user_df[
            user_df["UserId"] == selected_user
        ]
        .iloc[0]
    )


    # --------------------------------------------------------
    # VISIT YEAR AND MONTH
    # --------------------------------------------------------

    col1, col2 = st.columns(2)

    with col1:

        visit_year = st.number_input(
            "Visit Year",
            min_value=2000,
            max_value=2100,
            value=2026,
            step=1,
            key="prediction_year"
        )

    with col2:

        visit_month = st.selectbox(
            "Visit Month",
            list(range(1, 13)),
            key="prediction_month"
        )


    # --------------------------------------------------------
    # ATTRACTION
    # --------------------------------------------------------

    attraction_options = (
        item_df[
            [
                "AttractionId",
                "Attraction",
                "AttractionCityId",
                "AttractionTypeId"
            ]
        ]
        .drop_duplicates(
            subset=["AttractionId"]
        )
        .copy()
    )

    attraction_options["Display"] = (
        attraction_options["AttractionId"]
        .astype(str)
        + " - "
        + attraction_options["Attraction"]
        .fillna("Unknown")
        .astype(str)
    )

    selected_attraction_name = st.selectbox(
        "Select Tourist Attraction",
        attraction_options["Display"].tolist(),
        key="prediction_attraction"
    )

    selected_attraction = (
        attraction_options[
            attraction_options["Display"]
            == selected_attraction_name
        ]
        .iloc[0]
    )


    # --------------------------------------------------------
    # USER PROFILE DISPLAY
    # --------------------------------------------------------

    st.subheader(
        "User Information"
    )

    p1, p2, p3, p4 = st.columns(4)

    with p1:
        st.metric(
            "Continent",
            selected_user_data["ContinentId"]
        )

    with p2:
        st.metric(
            "Region",
            selected_user_data["RegionId"]
        )

    with p3:
        st.metric(
            "Country",
            selected_user_data["CountryId"]
        )

    with p4:
        st.metric(
            "City",
            selected_user_data["CityId"]
        )


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    if st.button(
        "🔮 Predict Visit Mode",
        type="primary",
        key="predict_visitmode_button"
    ):

        input_data = pd.DataFrame({

            "VisitYear": [
                visit_year
            ],

            "VisitMonth": [
                visit_month
            ],

            "ContinentId": [
                selected_user_data["ContinentId"]
            ],

            "RegionId": [
                selected_user_data["RegionId"]
            ],

            "CountryId": [
                selected_user_data["CountryId"]
            ],

            "CityId": [
                selected_user_data["CityId"]
            ],

            "AttractionCityId": [
                selected_attraction["AttractionCityId"]
            ],

            "AttractionTypeId": [
                selected_attraction["AttractionTypeId"]
            ]
        })

        try:

            prediction = (
                classification_model
                .predict(input_data)
            )

            predicted_mode = prediction[0]

            st.success(
                f"### Predicted Visit Mode: {predicted_mode}"
            )


            # ------------------------------------------------
            # Probability
            # ------------------------------------------------

            if hasattr(
                classification_model,
                "predict_proba"
            ):

                probability = (
                    classification_model
                    .predict_proba(
                        input_data
                    )[0]
                )

                classes = (
                    classification_model
                    .classes_
                )

                probability_df = pd.DataFrame({
                    "Visit Mode": classes,
                    "Probability": probability
                })

                probability_df[
                    "Probability (%)"
                ] = (
                    probability_df[
                        "Probability"
                    ] * 100
                )

                probability_df = (
                    probability_df
                    .sort_values(
                        "Probability (%)",
                        ascending=False
                    )
                )

                st.subheader(
                    "Prediction Confidence"
                )

                st.dataframe(
                    probability_df[
                        [
                            "Visit Mode",
                            "Probability (%)"
                        ]
                    ].style.format({
                        "Probability (%)": "{:.2f}%"
                    }),
                    use_container_width=True
                )


            # ------------------------------------------------
            # Input data
            # ------------------------------------------------

            st.subheader(
                "Prediction Input"
            )

            st.dataframe(
                input_data,
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Prediction failed."
            )

            st.exception(e)


# ============================================================
# ATTRACTION RECOMMENDATION
# ============================================================

elif page == "🎯 Attraction Recommendation":

    st.header(
        "🎯 Personalized Attraction Recommendation"
    )

    st.write(
        """
        The recommendation system uses collaborative
        filtering based on historical user-attraction
        ratings.
        """
    )


    # --------------------------------------------------------
    # USER
    # --------------------------------------------------------

    user_ids = (
        user_df["UserId"]
        .dropna()
        .unique()
        .tolist()
    )

    selected_user = st.selectbox(
        "Select User",
        user_ids,
        key="recommendation_user"
    )


    # --------------------------------------------------------
    # NUMBER OF RECOMMENDATIONS
    # --------------------------------------------------------

    top_n = st.slider(
        "Number of Recommendations",
        min_value=5,
        max_value=20,
        value=10,
        key="recommendation_count"
    )


    # --------------------------------------------------------
    # BUTTON
    # --------------------------------------------------------

    if st.button(
        "🎯 Generate Recommendations",
        type="primary",
        key="recommendation_button"
    ):

        try:

            artifacts = recommendation_model

            user_to_index = (
                artifacts[
                    "user_to_index"
                ]
            )

            item_to_index = (
                artifacts[
                    "item_to_index"
                ]
            )

            index_to_item = (
                artifacts[
                    "index_to_item"
                ]
            )

            user_item_matrix = (
                artifacts[
                    "user_item_matrix"
                ]
            )

            similarity_matrix = (
                artifacts[
                    "item_similarity_matrix"
                ]
            )

            global_mean = (
                artifacts[
                    "global_mean_rating"
                ]
            )

            item_mean = (
                artifacts[
                    "item_mean_rating"
                ]
            )

            best_k = artifacts.get(
                "best_k",
                10
            )


            # ------------------------------------------------
            # UNKNOWN USER
            # ------------------------------------------------

            if selected_user not in user_to_index:

                recommendations = (
                    transaction_df
                    .groupby(
                        "AttractionId"
                    )
                    .agg(
                        AverageRating=(
                            "Rating",
                            "mean"
                        ),
                        NumberOfRatings=(
                            "Rating",
                            "count"
                        )
                    )
                    .sort_values(
                        [
                            "AverageRating",
                            "NumberOfRatings"
                        ],
                        ascending=False
                    )
                    .head(top_n)
                    .reset_index()
                )

                recommendations[
                    "PredictedRating"
                ] = recommendations[
                    "AverageRating"
                ]


            # ------------------------------------------------
            # KNOWN USER
            # ------------------------------------------------

            else:

                user_index = (
                    user_to_index[
                        selected_user
                    ]
                )

                user_ratings = (
                    user_item_matrix
                    .getrow(user_index)
                    .toarray()
                    .ravel()
                )

                rated_items = set(
                    np.where(
                        user_ratings > 0
                    )[0]
                )

                candidates = []


                for item_index in range(
                    len(item_to_index)
                ):

                    if item_index in rated_items:
                        continue

                    attraction_id = (
                        index_to_item[
                            item_index
                        ]
                    )

                    similarities = (
                        similarity_matrix[
                            item_index
                        ].copy()
                    )

                    similarities[
                        item_index
                    ] = 0

                    rated_indices = (
                        np.where(
                            user_ratings > 0
                        )[0]
                    )

                    if len(
                        rated_indices
                    ) == 0:

                        prediction = (
                            item_mean.get(
                                attraction_id,
                                global_mean
                            )
                        )

                    else:

                        sim_values = (
                            similarities[
                                rated_indices
                            ]
                        )

                        rating_values = (
                            user_ratings[
                                rated_indices
                            ]
                        )

                        positive_mask = (
                            sim_values > 0
                        )

                        sim_values = (
                            sim_values[
                                positive_mask
                            ]
                        )

                        rating_values = (
                            rating_values[
                                positive_mask
                            ]
                        )

                        if len(
                            sim_values
                        ) == 0:

                            prediction = (
                                item_mean.get(
                                    attraction_id,
                                    global_mean
                                )
                            )

                        else:

                            k = min(
                                best_k,
                                len(
                                    sim_values
                                )
                            )

                            top_indices = (
                                np.argsort(
                                    sim_values
                                )[-k:]
                            )

                            top_sim = (
                                sim_values[
                                    top_indices
                                ]
                            )

                            top_rating = (
                                rating_values[
                                    top_indices
                                ]
                            )

                            denominator = (
                                np.sum(
                                    top_sim
                                )
                            )

                            if denominator == 0:

                                prediction = (
                                    item_mean.get(
                                        attraction_id,
                                        global_mean
                                    )
                                )

                            else:

                                prediction = (
                                    np.sum(
                                        top_sim
                                        * top_rating
                                    )
                                    / denominator
                                )

                    candidates.append({
                        "AttractionId": attraction_id,
                        "PredictedRating": prediction
                    })


                recommendations = (
                    pd.DataFrame(
                        candidates
                    )
                    .sort_values(
                        "PredictedRating",
                        ascending=False
                    )
                    .head(top_n)
                    .reset_index(drop=True)
                )


            # ------------------------------------------------
            # ADD ATTRACTION INFORMATION
            # ------------------------------------------------

            display_recommendations = (
                recommendations
                .merge(
                    item_df[
                        [
                            "AttractionId",
                            "Attraction",
                            "AttractionAddress"
                        ]
                    ].drop_duplicates(
                        "AttractionId"
                    ),
                    on="AttractionId",
                    how="left"
                )
            )

            display_recommendations[
                "PredictedRating"
            ] = (
                display_recommendations[
                    "PredictedRating"
                ]
                .clip(
                    lower=1,
                    upper=5
                )
            )

            display_recommendations.insert(
                0,
                "Rank",
                range(
                    1,
                    len(
                        display_recommendations
                    ) + 1
                )
            )


            st.success(
                "✅ Recommendations generated successfully!"
            )

            st.dataframe(
                display_recommendations[
                    [
                        "Rank",
                        "Attraction",
                        "AttractionAddress",
                        "PredictedRating"
                    ]
                ].style.format({
                    "PredictedRating": "{:.2f}"
                }),
                use_container_width=True
            )

        except Exception as e:

            st.error(
                "Recommendation failed."
            )

            st.exception(e)


# ============================================================
# RATING ANALYSIS
# ============================================================

elif page == "⭐ Rating Analysis":

    st.header(
        "⭐ Tourism Rating Analysis"
    )

    if "Rating" not in transaction_df.columns:

        st.warning(
            "Rating column not found."
        )

    else:

        rating_data = pd.to_numeric(
            transaction_df["Rating"],
            errors="coerce"
        ).dropna()

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Average Rating",
                f"{rating_data.mean():.2f}"
            )

        with col2:
            st.metric(
                "Minimum Rating",
                f"{rating_data.min():.0f}"
            )

        with col3:
            st.metric(
                "Maximum Rating",
                f"{rating_data.max():.0f}"
            )


        # Rating distribution
        st.subheader(
            "Rating Distribution"
        )

        rating_counts = (
            rating_data
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            rating_counts
        )


        # Top rated attractions
        st.subheader(
            "Top Rated Attractions"
        )

        attraction_rating = (
            transaction_df[
                [
                    "AttractionId",
                    "Rating"
                ]
            ]
            .copy()
        )

        attraction_rating[
            "Rating"
        ] = pd.to_numeric(
            attraction_rating[
                "Rating"
            ],
            errors="coerce"
        )

        attraction_rating = (
            attraction_rating
            .dropna()
            .groupby(
                "AttractionId"
            )
            .agg(
                AverageRating=(
                    "Rating",
                    "mean"
                ),
                NumberOfRatings=(
                    "Rating",
                    "count"
                )
            )
            .reset_index()
        )

        # Require minimum reviews
        attraction_rating = (
            attraction_rating[
                attraction_rating[
                    "NumberOfRatings"
                ] >= 5
            ]
        )

        attraction_rating = (
            attraction_rating
            .sort_values(
                [
                    "AverageRating",
                    "NumberOfRatings"
                ],
                ascending=False
            )
            .head(10)
        )

        attraction_rating = (
            attraction_rating
            .merge(
                item_df[
                    [
                        "AttractionId",
                        "Attraction"
                    ]
                ].drop_duplicates(
                    "AttractionId"
                ),
                on="AttractionId",
                how="left"
            )
        )

        st.dataframe(
            attraction_rating[
                [
                    "Attraction",
                    "AverageRating",
                    "NumberOfRatings"
                ]
            ].style.format({
                "AverageRating": "{:.2f}"
            }),
            use_container_width=True
        )


# ============================================================
# TOURISM ANALYTICS
# ============================================================

elif page == "📊 Tourism Analytics":

    st.header(
        "📊 Tourism Analytics Dashboard"
    )


    # --------------------------------------------------------
    # Visits by Year
    # --------------------------------------------------------

    st.subheader(
        "Visits by Year"
    )

    if "VisitYear" in transaction_df.columns:

        year_counts = (
            transaction_df[
                "VisitYear"
            ]
            .value_counts()
            .sort_index()
        )

        st.line_chart(
            year_counts
        )


    # --------------------------------------------------------
    # Visits by Month
    # --------------------------------------------------------

    st.subheader(
        "Visits by Month"
    )

    if "VisitMonth" in transaction_df.columns:

        month_counts = (
            transaction_df[
                "VisitMonth"
            ]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            month_counts
        )


    # --------------------------------------------------------
    # VisitMode Distribution
    # --------------------------------------------------------

    st.subheader(
        "Visit Mode Distribution"
    )

    try:

        mode_counts = (
            transaction_df[
                "VisitMode"
            ]
            .value_counts()
            .reset_index()
        )

        mode_counts.columns = [
            "VisitModeId",
            "Visits"
        ]

        mode_lookup = (
            mode_df[
                [
                    "VisitModeId",
                    "VisitMode"
                ]
            ]
            .drop_duplicates(
                "VisitModeId"
            )
        )

        mode_counts = (
            mode_counts
            .merge(
                mode_lookup,
                on="VisitModeId",
                how="left"
            )
        )

        st.bar_chart(
            mode_counts.set_index(
                "VisitMode"
            )[
                "Visits"
            ]
        )

    except Exception:

        st.warning(
            "VisitMode analytics could not be displayed."
        )


    # --------------------------------------------------------
    # Top Attractions
    # --------------------------------------------------------

    st.subheader(
        "Top 10 Most Visited Attractions"
    )

    top_attractions = (
        transaction_df[
            "AttractionId"
        ]
        .value_counts()
        .head(10)
        .reset_index()
    )

    top_attractions.columns = [
        "AttractionId",
        "Visits"
    ]

    top_attractions = (
        top_attractions
        .merge(
            item_df[
                [
                    "AttractionId",
                    "Attraction"
                ]
            ].drop_duplicates(
                "AttractionId"
            ),
            on="AttractionId",
            how="left"
        )
    )

    st.dataframe(
        top_attractions[
            [
                "Attraction",
                "Visits"
            ]
        ],
        use_container_width=True
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Tourism Experience Analytics | "
    "Machine Learning • Classification • Recommendation"
)

st.caption(
    "Developed using Python, Pandas, Scikit-learn and Streamlit."
)