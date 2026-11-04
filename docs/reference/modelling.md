# Modelling: linear regression, logistic regression and k-nearest neighbours

`modelling` · level **introductory** · builds on [`exploring-data`](exploring-data.md) · pip extra `none`

Modelling: linear regression, logistic regression and k-nearest neighbours.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("modelling")
T = kit.tools()
```

Written for the **introductory** level: Learners who are new to programming and to data science, still getting used to Python, NumPy, pandas and Matplotlib.

## The three layers

### In focus: learners do these themselves

- What a model is: features (inputs) and a target (output), learning from examples with known answers (supervised learning), and the difference between regression (predicting a number) and classification (predicting a category); both are predictions
- Preparing data for a model: choosing features and target, dealing with missing values, one-hot encoding of categorical variables (pd.get_dummies), splitting into training, validation and test sets (train_test_split with a fixed random_state, stratified by a category), and the rule that anything learned from data (a median for filling gaps, the mean and standard deviation for scaling) is learned from the training set only, so that no information leaks from the validation or test sets
- Linear regression with scikit-learn (fit, predict, coef_, intercept_): the straight line that makes the squared vertical distances to the points as small as possible (least squares), what the slope and the intercept mean in the units of the data, residuals, and multiple regression with several features
- Judging a regression model on the validation set: a baseline that always predicts the training mean, mean absolute error, root mean squared error and R² (the share of the variation explained; for one feature it equals r²)
- Logistic regression for a yes/no target: a weighted sum turned into a probability between 0 and 1 by the S-shaped sigmoid curve, predict_proba versus predict, and the 0.5 threshold
- Judging a classifier: accuracy, the majority-class baseline, and the confusion matrix (which mistakes are made)
- k-nearest neighbours: classify a new example by the majority among the k most similar training examples (Euclidean distance), why features must be scaled first (StandardScaler fitted on the training set), and choosing k on the validation set
- Overfitting and underfitting read from training versus validation scores, choosing a model on the validation set, and using the test set once at the end

### On request: the kit does these, and learners may look inside

- Helpers students may open: the least-squares line computed from r and the standard deviations, MAE, RMSE and R² computed step by step, the sigmoid, and k-nearest neighbours computed by hand (with `T.line_by_hand(), T.scores_by_hand(), T.sigmoid(), T.knn_by_hand()`)

### Support: the kit does these for the learner

- Loading the stored datasets, drawing fitted lines, probability curves, confusion matrices and decision regions, and checking the practice exercises (with `T.penguins(), T.cars(), T.plot_line(), T.plot_predictions(), T.plot_probability(), T.plot_confusion(), T.plot_regions(), T.check()`)

### Left out of this topic

- How scikit-learn finds the coefficients (closed-form solutions, gradient descent, maximum likelihood), regularisation (including the small penalty LogisticRegression applies by default), cross-validation, confidence intervals and hypothesis tests for coefficients, precision, recall and ROC curves, decision trees and other models (taught in probability, distributions and hypothesis testing, and machine learning, beyond this topic)

## Helpers: `T = kit.tools()`

Modelling: data, look-inside helpers, pictures and practice checks for modelling.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### On request (look inside)

- **`T.line_by_hand(x, y, show: bool = True) -> tuple[float, float]`**: The least-squares line y = intercept + slope * x, computed from statistics met in exploring-data.
- **`T.scores_by_hand(y_true, y_pred, show: bool = True) -> dict`**: MAE, RMSE and R² computed step by step from the errors (residuals) of a regression model.
- **`T.sigmoid(z)`**: The S-shaped (logistic) curve 1 / (1 + e^(-z)), which turns any number into a value between 0 and 1.
- **`T.knn_by_hand(X_train: pd.DataFrame, y_train, new_point, k: int = 5, show: bool = True)`**: Classify one new example the way k-nearest neighbours does, showing the neighbours.

### Support

- **`T.cars() -> pd.DataFrame`**: The Auto MPG data: 398 car models sold in the United States from 1970 to 1982.
- **`T.plot_line(x, y, slope: float, intercept: float, residuals: bool = True, ax=None, title: str = '', xlabel: str | None = None, ylabel: str | None = None)`**: Scatter plot of the points with a fitted straight line; residuals drawn as thin vertical lines.
- **`T.plot_predictions(y_true, y_pred, ax=None, title: str = '', label: str = 'target')`**: Predicted against actual values: a perfect model puts every point on the dashed diagonal.
- **`T.plot_probability(model, data: pd.DataFrame, feature: str, columns, target, group: str | None = None, positive=1, ax=None, title: str = '', ylabel: str | None = None)`**: The probability that logistic regression gives as one feature changes, with the examples as dots.
- **`T.plot_confusion(y_true, y_pred, labels=None, ax=None, title: str = '')`**: The confusion matrix as a grid: rows are the actual categories, columns the predicted ones.
- **`T.plot_regions(model, X: pd.DataFrame, y, scaler=None, ax=None, title: str = '')`**: The decision regions of a classifier trained on two features, with the examples on top.
- **`T.check(exercise: str, answer=None, **more) -> None`**: Check a practice exercise, e.g. T.check("E2", slope, mae=mae, r2=r2). Says what is wrong and why.

### From other topics

Supplied ready-made here; each is taught in its own topic.

- `T.penguins`, from [`exploring-data`](exploring-data.md)

### Constants

`T.CARS_CITATION` (str), `T.COLOURS` (list), `T.DATA` (Path), `T.EFFICIENT_MPG` (int), `T.FEATURES_CLASS` (list), `T.FEATURES_MULTI` (list), `T.K_VALUES` (tuple), `T.PENGUINS_CITATION` (str), `T.ROOT` (Path), `T.SEED` (int)
