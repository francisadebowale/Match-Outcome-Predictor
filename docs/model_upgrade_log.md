Model Upgrade Log

Purpose

This file tracks the baseline model and every meaningful upgrade made to the project.

Any time the model, features, data cleaning, leakage prevention, evaluation metrics, or prediction outputs change, update this file.

The project story is not a strict v1, v2, v3 pipeline anymore. The story is:

Start with a simple Logistic Regression baseline, then document each upgrade and explain whether it helped.

Current Model Setup

Model:
Multiclass Logistic Regression

Target classes:
home_win
draw
away_win

Train period:
2010 to 2018

Validation period:
2019 to 2021

Test period:
2022 onward

Current selected model:
Logistic Regression with draw class weight 1.6

Current selected features:
elo_difference
home_advantage
home_goals_for_avg
home_goals_against_avg
away_goals_for_avg
away_goals_against_avg

Baseline

Majority-class baseline:
Always predicts home_win.

Test metrics:
Accuracy: 0.475
Log Loss: 1.052
Macro Precision: 0.158
Macro Recall: 0.333
Macro F1: 0.215
Weighted F1: 0.306
ROC-AUC: 0.500

Why this matters:
This gives a simple floor. Any useful model should beat this.

Upgrade 1: Logistic Regression With Football Features

What changed:
Added Logistic Regression using Elo difference, home advantage, and backward-looking goal averages.

Why:
This tests whether basic team strength and recent scoring information improves prediction over a naive baseline.

Leakage control:
Recent goal averages are calculated only from matches before the match being predicted.

Test metrics:
Accuracy: 0.598
Log Loss: 0.870
Macro Precision: 0.460
Macro Recall: 0.499
Macro F1: 0.439
Weighted F1: 0.518
ROC-AUC: 0.740

Result:
Clear improvement over the majority-class baseline.

Issue found:
The model was almost draw-blind. It predicted only 10 draws in the test set and correctly predicted only 2 true draws.

Upgrade 2: Leakage-Safe Elo Snapshots

What changed:
Historical match features use previous year-end Elo snapshots instead of future Elo ratings.

For historical matches:
Use year-end Elo snapshots from before the match year.

For 2026 World Cup predictions:
Use pre-tournament 2026 Elo rankings because the tournament has not started.

Why:
This avoids leaking future team strength into past training examples.

Result:
The feature pipeline is more defensible and easier to explain.

Upgrade 3: Team Name Normalization

What changed:
Added a team name mapping file to normalize country names across sources.

Examples:
USA to United States
Korea Republic to South Korea
DR Congo to Dem. Rep. of Congo
Cabo Verde to Cape Verde
IR Iran to Iran

Why:
The project combines multiple external data sources, and team names are not always written the same way.

Result:
More rows match correctly between results, fixtures, and Elo ratings.

Upgrade 4: Balanced Logistic Regression

What changed:
Tested class_weight="balanced" in Logistic Regression.

Why:
The original model favored common outcomes and barely predicted draws.

Test metrics:
Accuracy: 0.558
Log Loss: 0.902
Macro Precision: 0.511
Macro Recall: 0.517
Macro F1: 0.513
Weighted F1: 0.559
ROC-AUC: 0.736

Result:
Better class balance and much better draw handling, but worse accuracy and log loss.

Conclusion:
Useful experiment, but not the best final compromise.

Upgrade 5: Draw Class Weight Tuning

What changed:
Tested draw class weights from 1.0 to 2.0.

Candidate results:

draw_weight, validation_log_loss, validation_accuracy, validation_macro_f1, test_log_loss, test_accuracy, test_macro_f1
1.0, 0.840, 0.614, 0.454, 0.870, 0.598, 0.439
1.1, 0.841, 0.618, 0.470, 0.870, 0.598, 0.452
1.2, 0.843, 0.617, 0.489, 0.873, 0.598, 0.476
1.3, 0.846, 0.615, 0.515, 0.876, 0.593, 0.494
1.4, 0.850, 0.607, 0.526, 0.880, 0.589, 0.509
1.5, 0.855, 0.598, 0.529, 0.884, 0.585, 0.521
1.6, 0.860, 0.594, 0.537, 0.890, 0.581, 0.530
1.7, 0.866, 0.586, 0.539, 0.896, 0.577, 0.535
1.8, 0.872, 0.579, 0.538, 0.902, 0.566, 0.531
1.9, 0.878, 0.574, 0.539, 0.908, 0.557, 0.527
2.0, 0.885, 0.572, 0.543, 0.915, 0.548, 0.522

Why 1.6 was selected:
1.0 has the best log loss and accuracy, but it is weak on class balance.
1.7 has slightly better macro F1, but accuracy and log loss drop more.
1.6 is a better balance between probability quality, accuracy, and macro F1.

Selected model test metrics:
Accuracy: 0.581
Log Loss: 0.890
Macro Precision: 0.545
Macro Recall: 0.525
Macro F1: 0.530
Weighted Precision: 0.582
Weighted Recall: 0.581
Weighted F1: 0.577
ROC-AUC: 0.740

Selected model confusion matrix:

Actual away_win:
Predicted away_win: 597
Predicted draw: 345
Predicted home_win: 295

Actual draw:
Predicted away_win: 199
Predicted draw: 322
Predicted home_win: 456

Actual home_win:
Predicted away_win: 129
Predicted draw: 345
Predicted home_win: 1533

Result:
The model is no longer draw-blind, but it still struggles to identify which specific matches are draws.

Main limitation:
Class weighting can only help so much. Better draw-related features are needed.

Upgrade 6: 2026 World Cup Group Prediction Dataset

What changed:
Built a prediction dataset from confirmed 2026 World Cup group fixtures.

The script creates:
data/processed/world_cup_2026_prediction_dataset_v1.csv
results/predictions/world_cup_2026_group_predictions_v1.csv
results/predictions/world_cup_2026_group_tables_v1.csv

Why:
The model can now generate fixture-level predictions and group tables for the confirmed World Cup group stage.

Current 2026 group prediction counts:
home_win: 36
away_win: 21
draw: 15

Group table points rule:
Win: 3 points
Draw: 1 point
Loss: 0 points

Limitation:
The group table does not use proper FIFA tie-breakers because the current model predicts result class only, not scoreline or goal difference.

Upgrade 7: Recent Form Feature Experiment

What changed:
Added recent-form features to the feature engineering pipeline and tested them with the same Logistic Regression model and draw class weight 1.6.

Features tested:
recent_points_difference
home_recent_points_avg
away_recent_points_avg
home_recent_draw_rate
away_recent_draw_rate
recent_goal_difference_difference

Why:
The model still struggles with draws. Recent form could help identify close matches and teams that draw often.

Leakage control:
Each recent-form feature uses previous matches only. The feature builder shifts each team's match history before calculating rolling averages.

Experiment test metrics:
Accuracy: 0.579
Log Loss: 0.890
Macro Precision: 0.541
Macro Recall: 0.523
Macro F1: 0.528
Weighted Precision: 0.579
Weighted Recall: 0.579
Weighted F1: 0.574
ROC-AUC: 0.738

Baseline selected model test metrics before this experiment:
Accuracy: 0.581
Log Loss: 0.890
Macro F1: 0.530
ROC-AUC: 0.740

Result:
The full recent-form feature set did not improve the model.

Draw recall moved from 0.330 to 0.317.
Macro F1 moved from 0.530 to 0.528.
Log loss stayed effectively the same but was slightly worse at full precision.

Feature coefficient finding:
recent_goal_difference_difference was the strongest new feature, but it was still much weaker than Elo difference.

Correlation finding:
recent_points_difference and recent_goal_difference_difference were highly correlated at 0.813.
Recent points averages also overlapped with goal-average features.

Decision:
Do not adopt the full recent-form feature set as the final model.

What stays:
The recent-form columns stay in the processed dataset for future experiments.

What does not stay:
The default selected model does not use the full recent-form feature set.

Next experiment:
Try a smaller recent-form set:
recent_goal_difference_difference
home_recent_draw_rate
recent_points_difference

Upgrade 8: Targeted Feature Group Experiments

What changed:
Tested three smaller feature groups while keeping the same Logistic Regression model, draw class weight 1.6, split, and scaling approach.

Experiment 1:
Reduced recent-form feature set.

Features:
recent_goal_difference_difference
home_recent_draw_rate
away_recent_draw_rate
recent_points_difference

Experiment 2:
Tournament strength and match context.

Features:
is_world_cup
is_world_cup_qualifier
is_continental_tournament
is_friendly
match_importance_weight

Experiment 3:
Draw and close-match behavior features.

Features:
home_recent_draw_rate
away_recent_draw_rate
draw_rate_difference
home_recent_close_match_rate
away_recent_close_match_rate
close_match_rate_difference
home_recent_low_scoring_rate
away_recent_low_scoring_rate

Also tested 10-match versions of the draw and close-match behavior features.

Leakage control:
All rolling features are shifted by one match per team before rolling averages are calculated.
Tournament context uses only the tournament label known before kickoff.

Outputs:
results/reports/targeted_feature_experiments.md
results/metrics/targeted_feature_experiment_metrics.csv
results/metrics/targeted_feature_experiment_confusion_matrices.csv
results/metrics/targeted_feature_experiment_coefficients.csv
results/metrics/targeted_feature_experiment_competition_metrics.csv

Baseline selected model test metrics:
Accuracy: 0.581
Log Loss: 0.890
Macro F1: 0.530
Draw F1: 0.324
ROC-AUC: 0.740

Best result by log loss:
Tournament context features.

Tournament context test metrics:
Accuracy: 0.582
Log Loss: 0.889
Macro F1: 0.529
Draw Precision: 0.319
Draw Recall: 0.318
Draw F1: 0.319
ROC-AUC: 0.739

Reduced recent-form test metrics:
Accuracy: 0.579
Log Loss: 0.890
Macro F1: 0.527
Draw F1: 0.315
ROC-AUC: 0.738

Draw and close-match 5-match test metrics:
Accuracy: 0.581
Log Loss: 0.890
Macro F1: 0.530
Draw F1: 0.320
ROC-AUC: 0.738

Draw and close-match 10-match test metrics:
Accuracy: 0.579
Log Loss: 0.891
Macro F1: 0.528
Draw F1: 0.318
ROC-AUC: 0.738

Coefficient finding:
Elo difference remains the dominant feature in every experiment.
The new features do not displace Elo difference or home advantage as the main model signals.

Decision:
Do not promote any targeted feature group to the production baseline.

Why:
Tournament context slightly improves log loss, but macro F1 and draw F1 are worse than the selected baseline.
The reduced recent-form and draw/close-match groups do not improve the main success criteria.

Upgrade 9: 2026 Host Home Advantage Fix

What changed:
Updated the 2026 World Cup prediction dataset generation logic so home advantage is assigned only when the listed home team is a host nation.

Host nations:
United States
Mexico
Canada

Logic:
home_advantage = 1 only if home_team is United States, Mexico, or Canada.
home_advantage = 0 otherwise.

Historical training data:
Unchanged.

Outputs regenerated:
data/processed/world_cup_2026_prediction_dataset_v1.csv
results/predictions/world_cup_2026_group_predictions_v1.csv
results/predictions/world_cup_2026_group_tables_v1.csv

Fixture counts:
Total fixtures: 72
home_advantage = 1: 6
home_advantage = 0: 66

Teams receiving home advantage:
Canada
Mexico
United States

Prediction impact:
Match outcome changes: 0
Probability changes: 0
Group standing changes: 0
Qualification changes: 0

Why there was no prediction change:
The existing generated dataset already matched the host-only behavior for the current official group fixtures.
The code was still simplified to enforce the requested rule directly.

Report:
results/reports/host_home_advantage_fix.md

Decision:
Promote this fix to the 2026 prediction pipeline because it makes the assumption explicit and prevents future fixture-ordering mistakes.

Upgrade 10: Calibration Experiment

What changed:
Tested probability calibration without changing the model family, features, draw weight, or train/validation/test split.

Methods tested:
Uncalibrated baseline
Sigmoid calibration
Isotonic calibration

Setup:
Train base Logistic Regression on 2010-2018.
Fit calibration on 2019-2021.
Evaluate on 2022-2025.

Outputs:
results/metrics/calibration_experiment_metrics.csv
results/metrics/calibration_experiment_confusion_matrices.csv
results/reports/calibration_experiment.md

Uncalibrated test metrics:
Accuracy: 0.581
Log Loss: 0.890
Macro F1: 0.530
Draw Precision: 0.318
Draw Recall: 0.330
Draw F1: 0.324
ROC-AUC: 0.740
Brier Score: 0.175

Sigmoid calibration test metrics:
Accuracy: 0.598
Log Loss: 0.880
Macro F1: 0.440
Draw Precision: 0.143
Draw Recall: 0.001
Draw F1: 0.002
ROC-AUC: 0.731
Brier Score: 0.172

Isotonic calibration test metrics:
Accuracy: 0.598
Log Loss: 0.904
Macro F1: 0.438
Draw Precision: 0.000
Draw Recall: 0.000
Draw F1: 0.000
ROC-AUC: 0.727
Brier Score: 0.173

Calibration finding:
Sigmoid calibration improves log loss and Brier score, but nearly eliminates draw predictions.
Isotonic calibration improves validation calibration but performs worse on test log loss and removes draw predictions entirely.

Decision:
Reject calibration for the production baseline.

Why:
The project needs useful class predictions and balanced performance, not only lower log loss.
Both calibration methods fail the Macro F1 and draw-recall requirements.

Current Weaknesses

The model still struggles with draws.

Draw recall improved, but many actual draws are still predicted as home wins or away wins.

Macro F1 is lower than weighted F1, which means performance is still stronger on the majority/easier classes.

Log loss is useful but not excellent, so the probabilities should be treated as estimates, not highly confident forecasts.

Next Upgrade Ideas

Recent form features from existing data:
home_recent_points_avg
away_recent_points_avg
recent_points_difference
home_recent_win_rate
away_recent_win_rate
home_recent_draw_rate
away_recent_draw_rate
home_recent_goal_difference_avg
away_recent_goal_difference_avg
recent_goal_difference_difference

Why this should help:
The current model knows basic team strength and recent goals, but not recent match outcomes or whether teams are drawing often.

Important rule:
All recent form features must be backward-looking only. Use previous matches only, never the current or future match.

Files To Update When Model Changes

src/features.py
src/train_model.py
src/build_2026_prediction_dataset.py
results/metrics/baseline_comparison.csv
results/reports/baseline_results.md
docs/model_upgrade_log.md

Upgrade 11: Tournament Simulation Pipeline

What changed:
Added a dashboard-ready 2026 World Cup simulation data layer without retraining the model.

Files added:
src/simulate_world_cup_2026.py
tests/test_world_cup_simulation.py

Annex C parsing:
Raw source:
data/raw/world_cup_2026/annex_c_third_place_allocation_wikitext.txt

Parsed outputs:
data/raw/world_cup_2026/annex_c_third_place_allocation.csv
data/raw/world_cup_2026/annex_c_third_place_allocation.json

Validation:
Annex C parsed successfully.
Rows parsed: 495
Each row has 8 qualified third-place groups.
Each row has 8 assigned third-place opponents.
No duplicate qualified_groups keys were found.

Simulation setup:
Number of simulations: 10000
Base seed: 2026
Simulation seed rule: base_seed + simulation_id

Input fingerprint:
The metadata output stores a SHA-256 fingerprint of:
data/raw/world_cup_2026/fixtures.csv
results/predictions/world_cup_2026_group_predictions_v1.csv
data/raw/world_cup_2026/annex_c_third_place_allocation.csv

Outputs:
results/simulations/world_cup_2026_simulation_summary.csv
results/simulations/world_cup_2026_match_simulation_summary.csv
results/simulations/world_cup_2026_metadata.json

Bracket method:
official_with_annex_c

Knockout draw handling:
Knockout matches cannot end in draws.
When a model probability row has home/draw/away probabilities, draw probability is redistributed proportionally between home and away advancement probabilities.

Tie-breaker assumptions:
The simulator uses outcome-level simulation, not score-level simulation.
Group and third-place ranking use:
points
wins
Elo rating
team name

Known limitations:
No simulated goals are created yet, so goal difference and goals scored are not used as tie-breakers.
Knockout pair probabilities use exact group-stage prediction rows when available and Elo fallback otherwise.
The full per-simulation match file was not written because it would be large; an aggregated match-level summary is produced for dashboard use.

Current top champion probabilities from the 10000 simulation run:
Spain: 0.2907
Argentina: 0.1723
France: 0.1507
England: 0.0679
Brazil: 0.0497

Decision:
Keep this as the dashboard and simulation data layer.
Do not treat it as a model upgrade because the trained model and selected features were unchanged.
