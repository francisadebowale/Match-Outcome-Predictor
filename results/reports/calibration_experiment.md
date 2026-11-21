Calibration Experiment

Setup:
- Model: Logistic Regression
- Draw class weight: 1.6
- Features: elo_difference, home_advantage, home_goals_for_avg, home_goals_against_avg, away_goals_for_avg, away_goals_against_avg
- Train period: 2010-2018
- Calibration period: 2019-2021
- Test period: 2022-2025

Validation metrics:

| method | accuracy | log loss | macro precision | macro recall | macro F1 | weighted F1 | draw precision | draw recall | draw F1 | ROC-AUC | Brier | calibration error | recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| uncalibrated | 0.594 | 0.860 | 0.560 | 0.531 | 0.537 | 0.588 | 0.308 | 0.320 | 0.314 | 0.757 | 0.169 | 0.069 | current baseline |
| sigmoid | 0.618 | 0.849 | 0.408 | 0.518 | 0.456 | 0.537 | 0.000 | 0.000 | 0.000 | 0.751 | 0.166 | 0.030 | reject calibration |
| isotonic | 0.620 | 0.836 | 0.413 | 0.519 | 0.458 | 0.538 | 0.000 | 0.000 | 0.000 | 0.761 | 0.163 | 0.026 | reject calibration |

Test metrics:

| method | accuracy | log loss | macro precision | macro recall | macro F1 | weighted F1 | draw precision | draw recall | draw F1 | ROC-AUC | Brier | calibration error | recommendation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| uncalibrated | 0.581 | 0.890 | 0.545 | 0.525 | 0.530 | 0.577 | 0.318 | 0.330 | 0.324 | 0.740 | 0.175 | 0.067 | current baseline |
| sigmoid | 0.598 | 0.880 | 0.439 | 0.503 | 0.440 | 0.519 | 0.143 | 0.001 | 0.002 | 0.731 | 0.172 | 0.025 | reject calibration |
| isotonic | 0.598 | 0.904 | 0.392 | 0.499 | 0.438 | 0.517 | 0.000 | 0.000 | 0.000 | 0.727 | 0.173 | 0.026 | reject calibration |

Test confusion matrices:

isotonic:
predicted  away_win  draw  home_win
actual                             
away_win        772     0       465
draw            348     0       629
home_win        255     0      1752

sigmoid:
predicted  away_win  draw  home_win
actual                             
away_win        809     2       426
draw            381     1       595
home_win        288     4      1715

uncalibrated:
predicted  away_win  draw  home_win
actual                             
away_win        597   345       295
draw            199   322       456
home_win        129   345      1533

Decision:
Reject calibration for now. Neither calibrated method improves the required metrics.
