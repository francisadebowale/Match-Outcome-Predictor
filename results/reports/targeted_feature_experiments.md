Targeted Feature Experiments

Baseline selected model:
- Logistic Regression
- Draw class weight: 1.6
- Test accuracy: 0.581
- Test log loss: 0.890
- Test macro F1: 0.530
- Test draw F1: 0.324

Ranking:

| experiment | log loss | macro F1 | draw F1 | accuracy | recommendation |
|---|---:|---:|---:|---:|---|
| experiment_2_tournament_context | 0.889 | 0.529 | 0.319 | 0.582 | investigate further |
| baseline_selected_model | 0.890 | 0.530 | 0.324 | 0.581 | current baseline |
| experiment_3_draw_close_5_match | 0.890 | 0.530 | 0.320 | 0.581 | reject |
| experiment_1_reduced_recent_form | 0.890 | 0.527 | 0.315 | 0.579 | reject |
| experiment_3_draw_close_10_match | 0.891 | 0.528 | 0.318 | 0.579 | reject |

Validation metrics:

| experiment | accuracy | log loss | macro F1 | draw precision | draw recall | draw F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_selected_model | 0.594 | 0.860 | 0.537 | 0.308 | 0.320 | 0.314 | 0.757 |
| experiment_1_reduced_recent_form | 0.588 | 0.861 | 0.530 | 0.295 | 0.299 | 0.297 | 0.755 |
| experiment_2_tournament_context | 0.595 | 0.859 | 0.539 | 0.316 | 0.317 | 0.316 | 0.756 |
| experiment_3_draw_close_5_match | 0.588 | 0.861 | 0.529 | 0.291 | 0.291 | 0.291 | 0.755 |
| experiment_3_draw_close_10_match | 0.588 | 0.860 | 0.528 | 0.292 | 0.291 | 0.292 | 0.757 |

Test metrics:

| experiment | accuracy | log loss | macro F1 | draw precision | draw recall | draw F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| baseline_selected_model | 0.581 | 0.890 | 0.530 | 0.318 | 0.330 | 0.324 | 0.740 |
| experiment_1_reduced_recent_form | 0.579 | 0.890 | 0.527 | 0.311 | 0.318 | 0.315 | 0.738 |
| experiment_2_tournament_context | 0.582 | 0.889 | 0.529 | 0.319 | 0.318 | 0.319 | 0.739 |
| experiment_3_draw_close_5_match | 0.581 | 0.890 | 0.530 | 0.319 | 0.322 | 0.320 | 0.738 |
| experiment_3_draw_close_10_match | 0.579 | 0.891 | 0.528 | 0.313 | 0.323 | 0.318 | 0.738 |

Test confusion matrix summary:

baseline_selected_model:
predicted  away_win  draw  home_win
actual                             
away_win        597   345       295
draw            199   322       456
home_win        129   345      1533

experiment_1_reduced_recent_form:
predicted  away_win  draw  home_win
actual                             
away_win        603   339       295
draw            203   311       463
home_win        129   350      1528

experiment_2_tournament_context:
predicted  away_win  draw  home_win
actual                             
away_win        598   333       306
draw            205   311       461
home_win        130   331      1546

experiment_3_draw_close_10_match:
predicted  away_win  draw  home_win
actual                             
away_win        600   342       295
draw            204   316       457
home_win        127   350      1530

experiment_3_draw_close_5_match:
predicted  away_win  draw  home_win
actual                             
away_win        604   335       298
draw            200   315       462
home_win        133   339      1535

Top coefficient signals:

- baseline_selected_model: elo_difference (away_win -0.788), elo_difference (home_win 0.786), home_advantage (home_win 0.156), home_advantage (away_win -0.140), home_goals_against_avg (away_win 0.104)
- experiment_1_reduced_recent_form: elo_difference (home_win 0.787), elo_difference (away_win -0.786), home_advantage (home_win 0.155), home_advantage (away_win -0.140), away_goals_against_avg (draw -0.097)
- experiment_2_tournament_context: elo_difference (away_win -0.787), elo_difference (home_win 0.786), home_advantage (home_win 0.147), home_advantage (away_win -0.121), home_goals_against_avg (away_win 0.107)
- experiment_3_draw_close_10_match: elo_difference (away_win -0.788), elo_difference (home_win 0.787), home_advantage (home_win 0.157), home_advantage (away_win -0.140), home_goals_against_avg (away_win 0.103)
- experiment_3_draw_close_5_match: elo_difference (home_win 0.787), elo_difference (away_win -0.786), home_advantage (home_win 0.155), home_goals_against_avg (away_win 0.149), home_advantage (away_win -0.139)

Decision rule:
Promote only if test log loss improves and test macro F1 is stable or better.

Recommendation:
Do not promote any tested feature group to the production baseline.

Output files:
- results/metrics/targeted_feature_experiment_metrics.csv
- results/metrics/targeted_feature_experiment_confusion_matrices.csv
- results/metrics/targeted_feature_experiment_coefficients.csv
- results/metrics/targeted_feature_experiment_competition_metrics.csv
