Baseline Results

This report compares the first Logistic Regression model against a simple majority-class baseline.

Dataset Split

Train:
2010 to 2018

Validation:
2019 to 2021

Test:
2022 onward

Baseline

The baseline always predicts the most common class from the training data.

In this dataset, the majority class is:

home_win

Test Results

Majority-class baseline:

Accuracy: 0.476
Log Loss: 1.052
Macro Precision: 0.159
Macro Recall: 0.333
Macro F1: 0.215
Weighted F1: 0.307
AUC-ROC: 0.500

Logistic Regression v1:

Accuracy: 0.598
Log Loss: 0.870
Macro Precision: 0.460
Macro Recall: 0.499
Macro F1: 0.439
Weighted F1: 0.518
AUC-ROC: 0.740

Balanced Logistic Regression v1:

Accuracy: 0.558
Log Loss: 0.902
Macro Precision: 0.511
Macro Recall: 0.517
Macro F1: 0.513
Weighted F1: 0.559
AUC-ROC: 0.736

Tuned Draw-Weight Logistic Regression v1:

Draw Weight: 1.6
Accuracy: 0.581
Log Loss: 0.890
Macro Precision: 0.545
Macro Recall: 0.525
Macro F1: 0.530
Weighted F1: 0.577
AUC-ROC: 0.740

Interpretation

The Logistic Regression model performs better than the majority-class baseline across every main metric.

The AUC-ROC score suggests the model has useful ranking ability. In other words, the features are helping the model separate stronger and weaker outcomes better than a naive baseline.

The weaker macro F1 score shows that the model still struggles across all classes equally, especially with harder outcomes such as draws and underdog wins.

Draw Blindness Experiment

The original Logistic Regression model had a major draw problem. In the test confusion matrix, it correctly predicted only 2 out of 977 true draws.

Adding class_weight="balanced" improved draw recall:

Original draw recall: 0.002
Balanced draw recall: 0.268
Tuned draw-weight recall: 0.330

This means the balanced model is much more willing to predict draws.

The tuned draw-weight model uses a moderate draw weight of 1.6. It fixes the worst draw blindness without pushing the model as far toward draws as the earlier 2.0 experiment.

However, this came with a tradeoff:

Original test accuracy: 0.598
Balanced test accuracy: 0.558
Tuned draw-weight test accuracy: 0.581

Original test log loss: 0.870
Balanced test log loss: 0.902
Tuned draw-weight test log loss: 0.890

The balanced and tuned models improve class balance and macro F1, but both are less accurate overall and have worse probability quality.

The tuned draw-weight model is now the better compromise between the original model and the fully balanced model. It improves draw handling while keeping accuracy and log loss closer to the original baseline.

Main Issue

The model is still too dependent on the most common outcomes. Accuracy is useful, but it does not tell the full story because home wins are more common than draws or away wins.

Next Experiment

The class balancing experiment improved draw prediction, but the tradeoff is too large to accept blindly.

Possible next change:

Add richer football features such as recent form, tournament strength, and betting odds. The current model is trying to solve the draw problem with class weights alone, but better features may help identify which matches are actually draw-prone.
