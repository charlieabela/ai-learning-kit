"""Modelling: linear regression, logistic regression and k-nearest neighbours.

A topic without graded tasks: its walkthrough and practice notebook check themselves, and the
practice exercises are checked with T.check(). Learners fit their first machine-learning models
with scikit-learn: linear regression (predicting a number), logistic regression and k-nearest
neighbours (predicting a category), with the data prepared for them by hand (features and
target, missing values, one-hot encoding, a training, validation and test split, scaling fitted
on the training set only). The kit supplies the Palmer penguins (as in exploring-data), the Auto MPG data,
readable helpers that show the calculations behind the models ("look inside"), and pictures of
fitted lines, probability curves, confusion matrices and decision regions. The concepts recalled
from exploring-data are descriptive statistics, correlation, visualisation and sampling;
probability is previewed where logistic regression returns one.
"""
from ...core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

TOPIC = "modelling"
TITLE = "Modelling: linear regression, logistic regression and k-nearest neighbours"
LEVEL = "introductory"
BUILDS_ON = ("exploring-data",)
PROBLEM = None
TASKS = {}
TOOLS = ".tools"
DATASETS = ()                       # the data ships with the kit
EXTRAS = ""                         # scikit-learn is a requirement of the kit from 0.10.0

STAGES = {
    "what_is_a_model": stage(
        "What a model is: features (inputs) and a target (output), learning from examples with known answers "
        "(supervised learning), and the difference between regression (predicting a number) and classification "
        "(predicting a category); both are predictions", FOCUS),
    "preparing": stage(
        "Preparing data for a model: choosing features and target, dealing with missing values, one-hot encoding of "
        "categorical variables (pd.get_dummies), splitting into training, validation and test sets "
        "(train_test_split with a fixed random_state, stratified by a category), and the rule that anything learned "
        "from data (a median for filling gaps, the mean and standard deviation for scaling) is learned from the "
        "training set only, so that no information leaks from the validation or test sets", FOCUS,
        note="Link the stratified split to stratified sampling and scaling to z-scores (mean and standard "
             "deviation) from the lectures."),
    "linear_regression": stage(
        "Linear regression with scikit-learn (fit, predict, coef_, intercept_): the straight line that makes the "
        "squared vertical distances to the points as small as possible (least squares), what the slope and the "
        "intercept mean in the units of the data, residuals, and multiple regression with several features",
        FOCUS, note="Explain ideas before formulas; give the formula only when asked or when the cell shows it."),
    "regression_metrics": stage(
        "Judging a regression model on the validation set: a baseline that always predicts the training mean, mean "
        "absolute error, root mean squared error and R² (the share of the variation explained; for one feature it "
        "equals r²)", FOCUS),
    "logistic_regression": stage(
        "Logistic regression for a yes/no target: a weighted sum turned into a probability between 0 and 1 by the "
        "S-shaped sigmoid curve, predict_proba versus predict, and the 0.5 threshold", FOCUS,
        note="The probability is P(category | features), a conditional probability; the lectures on probability "
             "come next, so connect without going further."),
    "classification_metrics": stage(
        "Judging a classifier: accuracy, the majority-class baseline, and the confusion matrix (which mistakes are "
        "made)", FOCUS),
    "knn": stage(
        "k-nearest neighbours: classify a new example by the majority among the k most similar training examples "
        "(Euclidean distance), why features must be scaled first (StandardScaler fitted on the training set), and "
        "choosing k on the validation set", FOCUS),
    "fitting_and_testing": stage(
        "Overfitting and underfitting read from training versus validation scores, choosing a model on the "
        "validation set, and using the test set once at the end", FOCUS),
    "look_inside": stage(
        "Helpers students may open: the least-squares line computed from r and the standard deviations, MAE, RMSE "
        "and R² computed step by step, the sigmoid, and k-nearest neighbours computed by hand", ON_REQUEST,
        how="T.line_by_hand(), T.scores_by_hand(), T.sigmoid(), T.knn_by_hand()",
        note="Explain the code step by step only when asked."),
    "pictures_data_checks": stage(
        "Loading the stored datasets, drawing fitted lines, probability curves, confusion matrices and decision "
        "regions, and checking the practice exercises", BLACK_BOX,
        how="T.penguins(), T.cars(), T.plot_line(), T.plot_predictions(), T.plot_probability(), "
            "T.plot_confusion(), T.plot_regions(), T.check()"),
    "beyond": stage(
        "How scikit-learn finds the coefficients (closed-form solutions, gradient descent, maximum likelihood), "
        "regularisation (including the small penalty LogisticRegression applies by default), cross-validation, "
        "confidence intervals and hypothesis tests for coefficients, precision, recall and ROC curves, decision "
        "trees and other models", OUT_OF_SCOPE,
        taught_in="probability, distributions and hypothesis testing, and machine learning, beyond this topic"),
}
