- Match Outcome Predictor

This is a machine learning project about predicting international football match results, mainly around the World Cup.

The aim is to build a simple model that can predict whether a match will end as:

Home win
Draw
Away win

I am starting with Logistic Regression because it is simple, easy to understand, and good for explaining predictions. For this project, I want to focus more on improving the data and features instead of immediately using a complicated model.

- Why Logistic Regression?

Logistic Regression is a good starting point because it gives probability outputs and is easier to explain than models like Random Forest or XGBoost.

For example, if the model predicts that a team has a 65% chance of winning, I can look at the features and explain what helped or hurt that prediction.

Possible reasons could include:

Better Elo rating
Strong recent form
Home advantage
Better attacking or defensive stats
Betting odds suggesting the team is more likely to win

This makes the project easier to understand and explain.

- Data

The data I plan to use includes:

International football results from 1872 to 2026 from https://www.kaggle.com/datasets/martj42/international-football-results-from-1872-to-2017/data

Elo ratings from [eloratings.net](https://www.eloratings.net/)

World Cup results from 2014, 2018, and 2022
2026 World Cup qualifier results from https://www.football-data.co.uk/data.php

Match stats such as xG, shots, possession, goals scored, and goals conceded
Betting odds, including maximum and average market odds from https://www.football-data.co.uk/data.php

- Features

Some of the features I want to try are:

Elo difference between teams
Recent form
Goals scored
Goals conceded
Home advantage
xG
xGA
Shots
Possession
Betting odds
Tournament importance

The main idea is to see whether adding better football features improves the model.

- Model

The main model for this project will be:

- Multiclass Logistic Regression

I am keeping the model simple on purpose. This should make it easier to compare different feature sets and explain the results.

- Evaluation

I plan to evaluate the model using:

Accuracy
Log loss
Confusion matrix
Predicted probabilities

Accuracy will show how often the model gets the result right, but probability quality is also important because football predictions are uncertain.

- Future Work

Later on, I might compare Logistic Regression with other models such as:

Random Forest
XGBoost
LightGBM

For now, the focus is on building a clean and understandable first version.

