"""Modelling: data, look-inside helpers, pictures and practice checks for modelling.

Used in the modelling notebooks as T = kit.tools() after kit.start("modelling").
T.penguins() returns the Palmer penguins used in exploring-data (Horst, Hill and Gorman, 2020; CC0).
T.cars() returns the Auto MPG data (Quinlan, 1993; UCI Machine Learning Repository, CC BY 4.0):
398 car models from 1970 to 1982 with their fuel consumption. The models themselves come from
scikit-learn and are written in the notebooks. The functions marked "look inside" in the
notebooks (line_by_hand, scores_by_hand, sigmoid, knn_by_hand) repeat a calculation that
scikit-learn does, written to be read: kit.source(T.name) prints their code. The plot_ functions
only draw pictures.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from ..exploring_data.tools import PENGUINS_CITATION, penguins  # noqa: F401  (the same data as exploring-data)

ROOT = Path(__file__).parent
SEED = 1101
DATA = ROOT / "data"

CARS_CITATION = (
    "Quinlan, R. (1993). Auto MPG [Dataset]. UCI Machine Learning Repository, doi:10.24432/C5859H. "
    "Licence: CC BY 4.0. The kit's copy is the CSV of the seaborn-data repository (origin written as "
    "usa, europe, japan instead of 1, 2, 3).")

COLOURS = ["#D37B35", "#4279B7", "#5B8C3A", "#8E5BA8", "#B8463E"]


# --------------------------------------------------------------------------- data
def cars() -> pd.DataFrame:
    """The Auto MPG data: 398 car models sold in the United States from 1970 to 1982.

    One row per car model: mpg (miles per US gallon; higher means less fuel used), cylinders,
    displacement (engine size, cubic inches), horsepower, weight (pounds), acceleration (seconds
    from 0 to 60 miles per hour), model_year (70 means 1970), origin (usa, europe, japan) and
    name. Horsepower is missing (NaN) for six cars, as in the published data.
    """
    return pd.read_csv(DATA / "auto_mpg.csv")


# --------------------------------------------------------------------------- look inside
def line_by_hand(x, y, show: bool = True) -> tuple[float, float]:
    """The least-squares line y = intercept + slope * x, computed from statistics met in exploring-data.

    The slope is r * (standard deviation of y) / (standard deviation of x): the correlation r
    says how closely the points follow a line, and the ratio of the standard deviations converts
    from the units of x to the units of y. The line always passes through the point
    (mean of x, mean of y), so intercept = mean of y - slope * mean of x. This is the same line
    that LinearRegression finds for one feature.
    """
    pairs = pd.DataFrame({"x": np.asarray(x, dtype=float), "y": np.asarray(y, dtype=float)}).dropna()
    r = pairs["x"].corr(pairs["y"])
    sx, sy = pairs["x"].std(), pairs["y"].std()
    mx, my = pairs["x"].mean(), pairs["y"].mean()
    slope = r * sy / sx
    intercept = my - slope * mx
    if show:
        print(f"r = {r:.3f}; standard deviations: x {sx:.2f}, y {sy:.2f}; means: x {mx:.2f}, y {my:.2f}")
        print(f"slope = r * sy / sx = {r:.3f} * {sy:.2f} / {sx:.2f} = {slope:.4f}")
        print(f"intercept = mean y - slope * mean x = {my:.2f} - {slope:.4f} * {mx:.2f} = {intercept:.2f}")
    return float(slope), float(intercept)


def scores_by_hand(y_true, y_pred, show: bool = True) -> dict:
    """MAE, RMSE and R² computed step by step from the errors (residuals) of a regression model.

    error = actual - predicted, for every example.
    MAE  (mean absolute error): the average size of an error, in the units of the target.
    RMSE (root mean squared error): square the errors, average them, take the square root;
         also in the units of the target, but large errors count for more.
    R²   = 1 - (sum of squared errors) / (sum of squared differences from the mean of the actual
         values): the share of the variation in the target that the model explains. 1 is perfect;
         0 is no better than always predicting the mean; it can be negative.
    """
    actual = np.asarray(y_true, dtype=float)
    errors = actual - np.asarray(y_pred, dtype=float)
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    if show:
        print(f"{len(errors)} examples; mean of |error| = MAE = {mae:.3f}")
        print(f"square root of the mean of error² = RMSE = {rmse:.3f}")
        print(f"R² = 1 - {ss_res:.1f} / {ss_tot:.1f} = {r2:.3f}")
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def sigmoid(z):
    """The S-shaped (logistic) curve 1 / (1 + e^(-z)), which turns any number into a value between 0 and 1.

    Logistic regression first computes a weighted sum z of the features, exactly as linear
    regression does, and then passes it through this curve to obtain a probability. z = 0 gives
    0.5; large positive z gives values close to 1; large negative z gives values close to 0.
    """
    return 1 / (1 + np.exp(-np.asarray(z, dtype=float)))


def knn_by_hand(X_train: pd.DataFrame, y_train, new_point, k: int = 5, show: bool = True):
    """Classify one new example the way k-nearest neighbours does, showing the neighbours.

    1. Measure the (Euclidean) distance from the new example to every training example: the
       square root of the sum of the squared differences, feature by feature.
    2. Keep the k training examples with the smallest distances.
    3. Predict the category that appears most often among them (a majority vote).
    X_train and new_point must be on the same scale as the model's input (scaled, if the model
    was trained on scaled features).
    """
    X = np.asarray(X_train, dtype=float)
    point = np.asarray(new_point, dtype=float).reshape(-1)
    distances = np.sqrt(((X - point) ** 2).sum(axis=1))
    labels = np.asarray(y_train)
    order = np.argsort(distances, kind="stable")[:k]
    table = pd.DataFrame({"distance": distances[order].round(3), "category": labels[order]})
    votes = table["category"].value_counts()
    winner = votes.index[0]
    if show:
        print(f"The {k} nearest training examples:")
        print(table.to_string())
        print("Votes: " + ", ".join(f"{c} {n}" for c, n in votes.items()) + f"  ->  prediction: {winner}")
    return winner


# --------------------------------------------------------------------------- pictures
def _axes(ax, size=(7, 4.5)):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    return ax


def plot_line(x, y, slope: float, intercept: float, residuals: bool = True, ax=None, title: str = "",
              xlabel: str | None = None, ylabel: str | None = None):
    """Scatter plot of the points with a fitted straight line; residuals drawn as thin vertical lines.

    A residual is the vertical distance from a point to the line: the error of the prediction
    for that example. Least squares chooses the line that makes the sum of their squares smallest.
    """
    x = pd.Series(x)
    y = pd.Series(y)
    ax = _axes(ax)
    predicted = intercept + slope * x
    if residuals:
        ax.vlines(x, predicted, y, color="#B8463E", linewidth=0.7, alpha=0.6, label="residuals")
    ax.scatter(x, y, s=16, alpha=0.75, color=COLOURS[1], label="examples", zorder=3)
    xs = np.linspace(x.min(), x.max(), 2)
    ax.plot(xs, intercept + slope * xs, color=COLOURS[0], linewidth=2.2,
            label=f"y = {intercept:.1f} {'-' if slope < 0 else '+'} {abs(slope):.3g} × x", zorder=4)
    ax.set_xlabel(xlabel or x.name or "x")
    ax.set_ylabel(ylabel or y.name or "y")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_predictions(y_true, y_pred, ax=None, title: str = "", label: str = "target"):
    """Predicted against actual values: a perfect model puts every point on the dashed diagonal."""
    ax = _axes(ax, (5, 5))
    a, p = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    ax.scatter(a, p, s=18, alpha=0.75, color=COLOURS[1])
    lo, hi = min(a.min(), p.min()), max(a.max(), p.max())
    ax.plot([lo, hi], [lo, hi], color="#6e7781", linestyle="--", linewidth=1.2, label="predicted = actual")
    ax.set_xlabel(f"actual {label}")
    ax.set_ylabel(f"predicted {label}")
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_probability(model, data: pd.DataFrame, feature: str, columns, target, group: str | None = None,
                     positive=1, ax=None, title: str = "", ylabel: str | None = None):
    """The probability that logistic regression gives as one feature changes, with the examples as dots.

    model was fitted on a table with the given columns (a numerical feature and, optionally, the
    one-hot columns of `group`). For each group (for example each species) the feature is varied
    over the range seen in that group and the model's probability of the positive category is
    drawn as a curve. Each example is a dot at 1 (positive) or 0 (not), slightly spread out.
    """
    ax = _axes(ax)
    columns = list(columns)
    rng = np.random.default_rng(SEED)
    is_pos = (pd.Series(np.asarray(target)) == positive).astype(int).to_numpy()
    groups = [(None, data.index)] if group is None else [(g, idx) for g, idx in data.groupby(group).groups.items()]
    for (name, idx), colour in zip(groups, COLOURS):
        part = data.loc[idx, feature]
        grid = np.linspace(part.min(), part.max(), 200)
        frame = pd.DataFrame(0, index=range(len(grid)), columns=columns, dtype=float)
        frame[feature] = grid
        if name is not None and f"{group}_{name}" in columns:
            frame[f"{group}_{name}"] = 1
        prob = model.predict_proba(frame[columns])[:, list(model.classes_).index(positive)]
        ax.plot(grid, prob, color=colour, linewidth=2.2, label=str(name) if name is not None else "probability")
        positions = [data.index.get_loc(i) for i in idx]
        ax.scatter(part, is_pos[positions] + rng.uniform(-0.04, 0.04, len(part)), s=12, alpha=0.5, color=colour)
    ax.axhline(0.5, color="#6e7781", linestyle=":", linewidth=1)
    ax.set_xlabel(feature)
    ax.set_ylabel(ylabel or f"probability of {positive}")
    ax.set_ylim(-0.08, 1.08)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


def plot_confusion(y_true, y_pred, labels=None, ax=None, title: str = ""):
    """The confusion matrix as a grid: rows are the actual categories, columns the predicted ones.

    The diagonal counts correct predictions; every other square counts one kind of mistake.
    """
    actual, predicted = np.asarray(y_true), np.asarray(y_pred)
    labels = list(labels) if labels is not None else sorted(set(actual) | set(predicted), key=str)
    counts = np.array([[np.sum((actual == a) & (predicted == p)) for p in labels] for a in labels])
    ax = _axes(ax, (1.2 * len(labels) + 2.2, 1.1 * len(labels) + 1.4))
    ax.imshow(counts, cmap="Blues")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=12,
                    color="white" if counts[i, j] > counts.max() / 2 else "#1f2d3d")
    ax.set_xticks(range(len(labels)), [str(v) for v in labels])
    ax.set_yticks(range(len(labels)), [str(v) for v in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title or f"Confusion matrix: {np.trace(counts)} of {counts.sum()} correct")
    return ax


def plot_regions(model, X: pd.DataFrame, y, scaler=None, ax=None, title: str = ""):
    """The decision regions of a classifier trained on two features, with the examples on top.

    Every point of the plane is coloured by the category the model would predict there. X holds
    the two features in their original units; if the model was trained on scaled features, pass
    the fitted scaler, and the plane is scaled the same way before the model is asked.
    """
    from matplotlib.colors import ListedColormap
    ax = _axes(ax, (6, 4.8))
    a, b = X.columns[:2]
    xs = np.linspace(X[a].min(), X[a].max(), 250)
    ys = np.linspace(X[b].min(), X[b].max(), 250)
    gx, gy = np.meshgrid(xs, ys)
    grid = pd.DataFrame({a: gx.ravel(), b: gy.ravel()})
    if scaler is not None:
        grid = pd.DataFrame(scaler.transform(grid), columns=[a, b])
    if not hasattr(model, "feature_names_in_"):        # the model was trained on an array, not a table
        grid = grid.to_numpy()
    classes = list(model.classes_)
    z = np.array([classes.index(v) for v in model.predict(grid)]).reshape(gx.shape)
    colours = COLOURS[:len(classes)]
    ax.contourf(gx, gy, z, levels=np.arange(len(classes) + 1) - 0.5, cmap=ListedColormap(colours), alpha=0.22)
    labels = np.asarray(y)
    for c, colour in zip(classes, colours):
        mask = labels == c
        ax.scatter(X.loc[mask, a], X.loc[mask, b], s=16, color=colour, edgecolor="white", linewidth=0.4,
                   label=str(c))
    ax.set_xlabel(a)
    ax.set_ylabel(b)
    ax.legend(fontsize=8)
    if title:
        ax.set_title(title)
    return ax


# --------------------------------------------------------------------------- practice checks
FEATURES_MULTI = ["weight", "horsepower", "model_year", "origin"]
FEATURES_CLASS = ["weight", "horsepower", "model_year"]
EFFICIENT_MPG = 25
K_VALUES = (1, 3, 5, 9, 15, 25)


@lru_cache(maxsize=1)
def _expected() -> dict:
    """Every result of the practice notebook, computed with the steps the exercises describe."""
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.metrics import accuracy_score, mean_absolute_error, r2_score
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    c = cars()
    train_val, test = train_test_split(c, test_size=0.2, random_state=SEED)
    train, val = train_test_split(train_val, test_size=0.25, random_state=SEED)
    fill = float(train["horsepower"].median())
    train, val, test = (d.assign(horsepower=d["horsepower"].fillna(fill)) for d in (train, val, test))
    e = {"index": {"train": set(train.index), "val": set(val.index), "test": set(test.index)}, "fill": fill,
         "fill_all": float(c["horsepower"].median())}

    simple = LinearRegression().fit(train[["weight"]], train["mpg"])
    pred = simple.predict(val[["weight"]])
    e.update(slope=float(simple.coef_[0]), intercept=float(simple.intercept_),
             mae_simple=float(mean_absolute_error(val["mpg"], pred)), r2_simple=float(r2_score(val["mpg"], pred)),
             baseline_mae=float(mean_absolute_error(val["mpg"], np.full(len(val), train["mpg"].mean()))))

    def onehot(d):
        return pd.get_dummies(d[FEATURES_MULTI], columns=["origin"], drop_first=True, dtype=int)
    Xtr = onehot(train)
    multi = LinearRegression().fit(Xtr, train["mpg"])
    pred = multi.predict(onehot(val)[Xtr.columns])
    e.update(mae_multi=float(mean_absolute_error(val["mpg"], pred)), r2_multi=float(r2_score(val["mpg"], pred)),
             year_coef=float(dict(zip(Xtr.columns, multi.coef_))["model_year"]))

    ytr, yva, yte = ((d["mpg"] >= EFFICIENT_MPG).astype(int) for d in (train, val, test))
    logit = LogisticRegression(max_iter=1000).fit(train[FEATURES_CLASS], ytr)
    majority = int(ytr.mode()[0])
    e.update(acc_logit=float(accuracy_score(yva, logit.predict(val[FEATURES_CLASS]))),
             baseline_acc=float((yva == majority).mean()), majority=majority,
             share_efficient=float(ytr.mean()))

    scaler = StandardScaler().fit(train[FEATURES_CLASS])
    s_tr, s_va, s_te = (scaler.transform(d[FEATURES_CLASS]) for d in (train, val, test))
    scores = {k: float(KNeighborsClassifier(n_neighbors=k).fit(s_tr, ytr).score(s_va, yva)) for k in K_VALUES}
    best_k = max(K_VALUES, key=lambda k: (scores[k], k))   # ties: the larger k, the smoother model
    e.update(knn_scores=scores, best_k=best_k, acc_knn=scores[best_k])
    knn = KNeighborsClassifier(n_neighbors=best_k).fit(s_tr, ytr)
    e.update(test_logit=float(logit.score(test[FEATURES_CLASS], yte)), test_knn=float(knn.score(s_te, yte)))
    e["chosen"] = "logistic" if e["acc_logit"] >= e["acc_knn"] else "knn"
    return e


def _say(ok: bool, text: str) -> bool:
    print(("Correct: " if ok else "Not yet: ") + text)
    return ok


def _not_attempted(value, name) -> bool:
    if value is None:
        print(f"Not attempted yet: complete the cell above, so that {name} holds your answer, and run this check again.")
        return True
    return False


def _close(a, b, tol=0.01) -> bool:
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def _check_e1(train, val=None, test=None, fill_value=None):
    if _not_attempted(train, "train"):
        return
    e = _expected()
    sets = {"train": train, "val": val, "test": test}
    for name, d in sets.items():
        if d is None:
            continue
        if not isinstance(d, pd.DataFrame):
            _say(False, f"{name} should be a DataFrame (a part of cars), not {type(d).__name__}.")
            continue
        want = e["index"][name]
        if set(d.index) == want:
            _say(True, f"{name} holds the {len(want)} expected cars.")
        elif len(d) == len(want):
            _say(False, f"{name} has the right size ({len(want)}) but different cars: use random_state=SEED in both "
                        "calls of train_test_split, and split cars first into train_val and test (test_size=0.2), "
                        "then train_val into train and val (test_size=0.25).")
        else:
            _say(False, f"{name} has {len(d)} cars; expected {len(want)} (60% / 20% / 20% of 398).")
        if "horsepower" in d and d["horsepower"].isna().any():
            _say(False, f"{name} still has {int(d['horsepower'].isna().sum())} missing horsepower value(s); fill them "
                        "with fill_value.")
    if fill_value is not None:
        if _close(fill_value, e["fill"], 0.001):
            _say(True, f"fill_value = {e['fill']:.1f}, the median horsepower of the training set.")
        elif _close(fill_value, e["fill_all"], 0.001):
            _say(False, f"fill_value is {fill_value}, the median of all 398 cars. Compute it from train only "
                        f"(it is {e['fill']:.1f}): the validation and test cars must not influence any step of "
                        "the preparation.")
        else:
            _say(False, f"fill_value is {fill_value}; train['horsepower'].median() gives {e['fill']:.1f}.")


def _check_e2(slope, mae=None, r2=None, baseline=None):
    if _not_attempted(slope, "slope"):
        return
    e = _expected()
    _say(_close(slope, e["slope"], 1e-4),
         f"the slope is {e['slope']:.5f} mpg per pound, that is {1000 * e['slope']:.2f} mpg for every extra "
         "1,000 pounds." if _close(slope, e["slope"], 1e-4) else
         f"slope is {slope}; model.coef_[0] of LinearRegression fitted on train[['weight']] and train['mpg'] gives "
         f"{e['slope']:.5f}.")
    for name, got, want, what in (("the validation MAE", mae, e["mae_simple"],
                                   "mean_absolute_error(val['mpg'], predictions)"),
                                  ("the validation R²", r2, e["r2_simple"], "r2_score(val['mpg'], predictions)"),
                                  ("the baseline's validation MAE", baseline, e["baseline_mae"],
                                   "the MAE when every validation car is predicted as the mean mpg of train")):
        if got is None:
            continue
        ok = _close(got, want, 0.005)
        _say(ok, f"{name} is {want:.3f}." if ok else
             f"{name} is {got}; {what} gives {want:.3f}. Fit on train, measure on val.")


def _check_e3(mae, r2=None):
    if _not_attempted(mae, "mae_multi"):
        return
    e = _expected()
    ok = _close(mae, e["mae_multi"], 0.005)
    _say(ok, f"MAE = {e['mae_multi']:.3f} mpg on the validation set, down from {e['mae_simple']:.3f} with weight "
             "alone." if ok else
         f"mae_multi is {mae}; expected {e['mae_multi']:.3f}. Use the features {FEATURES_MULTI} with origin one-hot "
         "encoded, and make sure the validation table has the same columns, in the same order, as the training "
         "table.")
    if r2 is not None:
        ok = _close(r2, e["r2_multi"], 0.005)
        _say(ok, f"R² = {e['r2_multi']:.3f}." if ok else f"r2_multi is {r2}; expected {e['r2_multi']:.3f}.")


def _check_e4(acc, baseline=None):
    if _not_attempted(acc, "acc_logit"):
        return
    e = _expected()
    ok = _close(acc, e["acc_logit"], 0.006)
    _say(ok, f"the validation accuracy is {e['acc_logit']:.3f} ({round(e['acc_logit'] * 80)} of 80 cars)." if ok else
         f"acc_logit is {acc}; expected {e['acc_logit']:.3f}. The target is mpg >= {EFFICIENT_MPG} (1 or 0), the "
         f"features are {FEATURES_CLASS}, the model is LogisticRegression(max_iter=1000) fitted on train.")
    if baseline is not None:
        ok = _close(baseline, e["baseline_acc"], 0.006)
        _say(ok, f"the baseline accuracy is {e['baseline_acc']:.3f}: always predicting the most common category of "
                 f"the training set ({e['majority']})." if ok else
             f"baseline_acc is {baseline}; the most common category in train is {e['majority']}, and predicting it "
             f"for every validation car is right for {e['baseline_acc']:.3f} of them.")


def _check_e5(best_k, acc=None):
    if _not_attempted(best_k, "best_k"):
        return
    e = _expected()
    table = ", ".join(f"k={k}: {v:.3f}" for k, v in e["knn_scores"].items())
    ok = int(best_k) == e["best_k"]
    _say(ok, f"k = {e['best_k']} ({table})." if ok else
         f"best_k is {best_k}; with the features scaled by a StandardScaler fitted on train, the validation "
         f"accuracies are {table}. Take the k with the highest accuracy and, if several tie, the largest of "
         "them.")
    if acc is not None:
        ok = _close(acc, e["acc_knn"], 0.006)
        _say(ok, f"accuracy {e['acc_knn']:.3f} with k = {e['best_k']}." if ok else
             f"acc_knn is {acc}; expected {e['acc_knn']:.3f}.")


def _check_e6(test_acc, model=None):
    if _not_attempted(test_acc, "test_acc"):
        return
    e = _expected()
    chosen = e["chosen"]
    want = e["test_logit"] if chosen == "logistic" else e["test_knn"]
    if model is not None:
        name = "logistic regression" if chosen == "logistic" else "k-nearest neighbours"
        ok = str(model).lower().startswith(chosen[:3])
        _say(ok, f"{name} had the higher validation accuracy ({e['acc_logit']:.3f} for logistic regression, "
                 f"{e['acc_knn']:.3f} for k-nearest neighbours)." if ok else
             f"model is {model!r}; the model to choose is the one with the higher validation accuracy: {name} "
             f"({e['acc_logit']:.3f} for logistic regression, {e['acc_knn']:.3f} for k-nearest neighbours). Write "
             "'logistic' or 'knn'.")
    ok = _close(test_acc, want, 0.006)
    _say(ok, f"the test accuracy of the chosen model is {want:.3f}." if ok else
         f"test_acc is {test_acc}; the chosen model, trained on train only, scores {want:.3f} on test. Use the test "
         "set once, for the chosen model only.")


_CHECKS = {"E1": _check_e1, "E2": _check_e2, "E3": _check_e3, "E4": _check_e4, "E5": _check_e5, "E6": _check_e6}


def check(exercise: str, answer=None, **more) -> None:
    """Check a practice exercise, e.g. T.check("E2", slope, mae=mae, r2=r2). Says what is wrong and why."""
    fn = _CHECKS.get(str(exercise).upper())
    if fn is None:
        raise KeyError(f"No exercise {exercise!r}. Exercises: {', '.join(_CHECKS)}")
    fn(answer, **more)
