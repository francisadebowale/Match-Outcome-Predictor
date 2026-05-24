Recent Form Feature Experiment

Objective

Evaluate whether recent-form features improve the Logistic Regression match outcome predictor while keeping the model architecture and draw weight unchanged.

Baseline For Comparison

Model:
Logistic Regression

Draw weight:
1.6

Baseline features:
elo_difference
home_advantage
home_goals_for_avg
home_goals_against_avg
away_goals_for_avg
away_goals_against_avg

Baseline test metrics:
Accuracy: 0.581
Log Loss: 0.890
Macro Precision: 0.545
Macro Recall: 0.525
Macro F1: 0.530
Weighted Precision: 0.582
Weighted Recall: 0.581
Weighted F1: 0.577
ROC-AUC: 0.740

Experiment Features

Added:
recent_points_difference
home_recent_points_avg
away_recent_points_avg
home_recent_draw_rate
away_recent_draw_rate
recent_goal_difference_difference

Leakage Control

The new features use previous matches only. The feature builder shifts each team's history by one match before calculating rolling averages.

Experiment Test Metrics

Accuracy: 0.579
Log Loss: 0.890
Macro Precision: 0.541
Macro Recall: 0.523
Macro F1: 0.528
Weighted Precision: 0.579
Weighted Recall: 0.579
Weighted F1: 0.574
ROC-AUC: 0.738

Experiment Confusion Matrix

Actual away_win:
Predicted away_win: 606
Predicted draw: 338
Predicted home_win: 293

Actual draw:
Predicted away_win: 204
Predicted draw: 310
Predicted home_win: 463

Actual home_win:
Predicted away_win: 130
Predicted draw: 350
Predicted home_win: 1527

Draw Metrics

Baseline draw recall:
0.330

Experiment draw recall:
0.317

Experiment draw precision:
0.311

Feature Coefficient Notes

The strongest feature remained elo_difference.

Largest max absolute coefficients:
elo_difference: 0.788
home_advantage: 0.155
away_goals_against_avg: 0.103
home_goals_against_avg: 0.084
recent_goal_difference_difference: 0.073
recent_points_difference: 0.035
home_recent_draw_rate: 0.032

The new recent-form features had smaller coefficients than Elo and the existing goals-against features.

Correlation Notes

Some new features are highly correlated with existing or related features:

recent_points_difference and recent_goal_difference_difference:
0.813

away_goals_for_avg and away_recent_points_avg:
0.702

home_goals_for_avg and home_recent_points_avg:
0.686

This suggests some recent-form features overlap with information already captured by goal averages.

Analysis Questions

Did Macro F1 improve?
No. It moved from 0.530 to 0.528.

Did draw recall improve?
No. It moved from 0.330 to 0.317.

Did draw precision improve?
No meaningful improvement. The experiment draw precision was 0.311.

Did log loss improve?
No. It stayed effectively the same but slightly worse at full precision.

Which new features contributed most?
recent_goal_difference_difference contributed the most among the new features, followed by recent_points_difference and home_recent_draw_rate.

Are any features redundant or highly correlated?
Yes. recent_points_difference and recent_goal_difference_difference are highly correlated. Recent points averages also overlap with goals-for averages.

Recommendation

Do not adopt this full recent-form feature set as the final model.

Reason:
It did not meet the success criteria:
Log Loss was not below 0.890.
Macro F1 did not stay at or above 0.530.

The features are useful enough to keep in the processed dataset for future experiments, but they should not replace the current selected model feature set yet.

Next Recommendation

Test a smaller recent-form set instead of all six features together:
recent_goal_difference_difference
home_recent_draw_rate
recent_points_difference

Remove or avoid:
home_recent_points_avg
away_recent_points_avg

Reason:
The points-average features appear redundant with existing goal-average features and did not improve the main metrics in this run.

