# 🌍 Tourism Experience Analytics

## Classification, Prediction and Recommendation System

A machine-learning-based Tourism Experience Analytics project developed to analyze tourist behavior, attraction ratings, visit patterns, and preferences.

The project combines data cleaning, preprocessing, exploratory data analysis, visualization, machine learning, recommendation systems, and Streamlit deployment.

---

## 📌 Project Overview

Tourism agencies and travel platforms generate data about users, visits, attractions, ratings, geographical locations, and visit modes.

This project uses that data to:

- Predict attraction ratings.
- Predict the likely user visit mode.
- Recommend personalized tourist attractions.
- Analyze tourism trends and popular attractions.
- Support customer segmentation and targeted marketing.

These objectives are defined in the project specification. 

---

## 🎯 Business Objectives

The main business objectives are:

1. **Personalized Recommendations**
   - Recommend attractions based on users' previous visits, preferences, and ratings.

2. **Tourism Analytics**
   - Identify popular attractions and regions.
   - Analyze tourism trends and visitor behavior.

3. **Customer Segmentation**
   - Predict visitor categories such as Business, Family, Couples, Friends, etc.

4. **Customer Retention**
   - Improve customer engagement and loyalty through personalized recommendations.

---

## 🚀 Machine Learning Objectives

The project contains three major machine-learning components.

### Model 1 - Rating Prediction

**Task:** Regression

**Target:** `Rating`

The model predicts the rating a user may give to a tourist attraction using user, visit, and attraction-related features.

Evaluation metrics:

- MAE
- MSE
- RMSE
- R² Score

---

### Model 2 - VisitMode Prediction

**Task:** Classification

**Target:** `VisitMode`

The model predicts the likely visit mode of a user.

Example categories include:

- Business
- Couples
- Family
- Friends
- Other available categories in the dataset

Model used:

**Random Forest Classifier**

Evaluation metrics:

- Accuracy
- Precision
- Recall
- F1-Score

Cross-validation:

**Stratified 5-Fold Cross-Validation**

Hyperparameter optimization:

**GridSearchCV**

---

### Model 3 - Attraction Recommendation

**Task:** Recommendation System

**Approach:** Item-Item Collaborative Filtering

The recommendation system uses historical user-attraction ratings to recommend attractions that a user may prefer.

Evaluation metrics:

- MAE
- RMSE

Hyperparameter tuning:

- Number of similar attractions (`K`)
- 5-Fold Cross-Validation

The project specification identifies collaborative filtering as a recommended approach based on users' ratings and preferences.

---

## 📊 Dataset

The project uses a Tourism Dataset consisting of multiple related tables.

### Transaction Data

Contains:

- `TransactionId`
- `UserId`
- `VisitYear`
- `VisitMonth`
- `VisitMode`
- `AttractionId`
- `Rating`

### User Data

Contains:

- `UserId`
- `ContinentId`
- `RegionId`
- `CountryId`
- `CityId`

### City Data

Contains:

- `CityId`
- `CityName`
- `CountryId`

### Item / Attraction Data

Contains:

- `AttractionId`
- `AttractionCityId`
- `AttractionTypeId`
- `Attraction`
- `AttractionAddress`

### Type Data

Contains:

- `AttractionTypeId`
- `AttractionType`

### Visit Mode Data

Contains:

- `VisitModeId`
- `VisitMode`

### Geographical Data

- Continent
- Country
- Region

---

## 🧹 Data Preprocessing

The project includes the following preprocessing activities:

- Handling missing values
- Missing-value imputation
- Data wrangling
- Outlier handling
- Categorical encoding
- Feature engineering
- Feature selection
- Feature correlation analysis
- Data transformation
- Data scaling
- Train-test splitting
- Imbalance checking where required

Text preprocessing techniques were also demonstrated where applicable.

---

## 📈 Exploratory Data Analysis

The project includes visualization and analysis of:

- Rating distribution
- Visit mode distribution
- Average rating by visit mode
- Visits by year
- Visits by month
- Average rating by month
- Most visited attractions
- Top-rated attractions
- Attraction type popularity
- Average rating by attraction type
- Country-wise tourism visits
- Country-wise average rating
- Correlation analysis
- Rating distribution by visit mode
- Average rating by year

These analyses are used to identify tourism trends, patterns, relationships, and business insights.

---

## 🤖 Model Validation and Tuning

### Model 1 - Regression

Cross-validation:

**5-Fold K-Fold Cross-Validation**

Hyperparameter tuning:

**GridSearchCV**

Important Random Forest parameters include:

- `n_estimators`
- `max_depth`
- `min_samples_leaf`
- `min_samples_split`

---

### Model 2 - Classification

Cross-validation:

**Stratified 5-Fold Cross-Validation**

Hyperparameter tuning:

**GridSearchCV**

Parameters include:

- `n_estimators`
- `max_depth`
- `min_samples_split`
- `min_samples_leaf`

The best model is selected using weighted F1-score.

---

### Model 3 - Recommendation

Cross-validation:

**5-Fold Cross-Validation**

The number of similar attractions (`K`) is tested using multiple values.

The best `K` is selected based on the lowest mean cross-validation RMSE.

---

## 💾 Saved Models

The project saves trained models for deployment.

Example:

```text
models/
├── best_tourism_visitmode_deployment.joblib
├── tourism_visitmode_tuned_model.pkl
├── tourism_recommendation_tuned_model.pkl
└── tourism_attraction_data.csv