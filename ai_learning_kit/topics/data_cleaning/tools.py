"""Data cleaning: finding missing values, making formats consistent, removing duplicates and filling gaps.

Used in the data-cleaning notebooks as T = kit.tools() after kit.start("data-cleaning"). The data
are the ones the web-data topic collects (the fictional Campus Lets listings and textbook exchange,
and a month of daily weather from the Open-Meteo API); here the kit supplies them ready-collected,
so that the focus stays on cleaning. The functions marked "look inside" in the notebooks
(parse_dates and knn_impute) are written to be read: kit.source(T.name) prints their code.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# The data, collected as in the web-data topic (black box here: see that topic to learn how).
from ..web_data.tools import (LUQA, SITE, VICTORIA, get, listings_raw, show_page, textbooks_raw,  # noqa: F401
                              weather, where_from)

SEED = 1101


GAPS = {"luqa": [4, 11, 12, 13, 20, 26], "victoria": [2, 8, 9, 10, 17, 24]}     # rows (days - 1) made empty


def weather_with_gaps(place: str = "luqa") -> pd.DataFrame:
    """The same month with tmax and tmin removed on six days, as if the station had failed.

    The true values are known (weather(place) has them), so the error of each way of filling
    the gaps can be measured. Three of the missing days are consecutive.
    """
    df = weather(place)
    df.loc[GAPS[place], ["tmax", "tmin"]] = np.nan
    return df


# --------------------------------------------------------------------------- cleaning
DATE_FORMATS = ("%Y-%m-%d", "%d/%m/%Y", "%d %b %Y")      # 2026-10-01, 01/10/2026, 1 Oct 2026


def parse_dates(text: pd.Series, formats=DATE_FORMATS) -> pd.Series:
    """Turn text dates written in several formats into dates, trying each format in turn.

    Each pass reads only the values written in that format (the others become NaT), and
    fillna keeps what earlier passes found. A value that matches no format stays NaT.
    """
    dates = pd.to_datetime(text, format=formats[0], errors="coerce")
    for fmt in formats[1:]:
        dates = dates.fillna(pd.to_datetime(text, format=fmt, errors="coerce"))
    return dates


# --------------------------------------------------------------------------- imputation (look inside)
def knn_impute(df: pd.DataFrame, column: str, using: list, k: int = 3) -> pd.Series:
    """Fill the missing values of `column` with the average of the k most similar complete rows.

    Similarity is measured on the numeric columns in `using`: each is rescaled to the range
    0 to 1, so that no column dominates because of its units, and the distance between two
    rows is the usual straight-line (Euclidean) distance. Rows that are missing a value in
    `using` are left as they are.
    """
    scaled = df[using].astype(float)
    scaled = (scaled - scaled.min()) / (scaled.max() - scaled.min())
    result = df[column].astype(float).copy()
    donors = df.index[df[column].notna() & scaled.notna().all(axis=1)]
    for i in df.index[df[column].isna()]:
        if scaled.loc[i].isna().any():
            continue
        distance = np.sqrt(((scaled.loc[donors] - scaled.loc[i]) ** 2).sum(axis=1))
        nearest = distance.nsmallest(k).index
        result.loc[i] = df.loc[nearest, column].astype(float).mean()
    return result


def plot_filled(dates, true, gappy, filled: dict, title: str = ""):
    """Plot the true values, the recorded values with gaps, and the values each method put in the gaps."""
    import matplotlib.pyplot as plt
    dates, true, gappy = pd.Series(dates).reset_index(drop=True), pd.Series(true).reset_index(drop=True), \
        pd.Series(gappy).reset_index(drop=True)
    gap = gappy.isna()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(dates, true, color="#C9D1D9", linewidth=5, label="true values")
    ax.plot(dates, gappy, color="#1f2d3d", linewidth=1.3, label="recorded, with gaps")
    markers = ["o", "s", "^", "D"]
    for (name, series), colour, marker in zip(filled.items(), ["#D37B35", "#4279B7", "#5B8C3A", "#8E5BA8"], markers):
        series = pd.Series(series).reset_index(drop=True)
        ax.plot(dates[gap], series[gap], marker, color=colour, label=name)
    ax.set_title(title)
    ax.set_ylabel("°C")
    ax.legend(fontsize=8, ncol=3)
    fig.autofmt_xdate()
    plt.show()


# --------------------------------------------------------------------------- practice checks
def _say(ok: bool, text: str) -> bool:
    print(("Correct: " if ok else "Not yet: ") + text)
    return ok


def _not_attempted(value, name) -> bool:
    if value is None:
        print(f"Not attempted yet: complete the cell above, so that {name} holds your answer, and run this check again.")
        return True
    return False


def _ref_clean_books() -> pd.DataFrame:
    b = textbooks_raw()
    b = b.replace({"price": {"Make an offer": None}, "rating": {"No rating yet": None}})
    b["course"] = b["course"].str.upper().str.replace(" ", "").str.replace("-", "")
    b["condition"] = b["condition"].str.lower().replace({"ln": "like new"})
    b["price"] = pd.to_numeric(b["price"].str.lower().replace({"free": "0"})
                               .str.replace("€", "").str.replace("eur", "").str.strip(), errors="coerce")
    b["rating"] = pd.to_numeric(b["rating"].str.split("/").str[0], errors="coerce")
    b["posted"] = parse_dates(b["posted"])
    return b


def _check_series_equal(student, expected, what) -> bool:
    try:
        s = pd.Series(student).astype(float)
    except Exception:
        return _say(False, f"{what} should be numbers.")
    e = pd.Series(expected).astype(float)
    if len(s) != len(e):
        return _say(False, f"{what} has {len(s)} values; expected {len(e)}.")
    diff = (s.reset_index(drop=True) - e.reset_index(drop=True)).abs()
    wrong = diff[diff > 1e-6].index.tolist()
    return _say(not wrong, f"{what} matches." if not wrong else f"{what} differs in {len(wrong)} row(s), first at position {wrong[0]}.")


def check(exercise: str, answer=None, **more) -> None:
    """Check a practice exercise, e.g. T.check("C1", missing_counts). Says what is wrong and why."""
    fn = _CHECKS.get(str(exercise).upper())
    if fn is None:
        raise KeyError(f"No exercise {exercise!r}. Exercises: {', '.join(_CHECKS)}")
    fn(answer, **more)


def _check_c1(missing_counts):
    if _not_attempted(missing_counts, "missing_counts"):
        return
    b = textbooks_raw().replace({"price": {"Make an offer": None}, "rating": {"No rating yet": None}})
    exp = b.isna().sum()
    got = pd.Series(missing_counts)
    wrong = [c for c in exp.index if int(got.get(c, -1)) != int(exp[c])]
    _say(not wrong, "missing values per column: " + ", ".join(f"{c} {int(exp[c])}" for c in exp.index) if not wrong else
         f"{wrong[0]} has {int(exp[wrong[0]])} missing values once text such as \"Make an offer\" and "
         f"\"No rating yet\" is turned into a missing value; you have {got.get(wrong[0])}.")


def _check_c2(books):
    if _not_attempted(books, "books"):
        return
    ref = _ref_clean_books()
    courses = sorted(set(books["course"].dropna()))
    _say(courses == sorted(set(ref["course"])), f"course codes: {courses}." if courses == sorted(set(ref["course"])) else
         f"course codes still differ: {courses}. Upper case, then remove spaces and hyphens.")
    conds = sorted(set(books["condition"].dropna()))
    exp = sorted(set(ref["condition"].dropna()))
    _say(conds == exp, f"conditions: {conds}." if conds == exp else f"conditions: {conds}; expected {exp} (\"LN\" means like new).")


def _check_c3(books):
    if _not_attempted(books, "books"):
        return
    ref = _ref_clean_books()
    for col in ("price", "rating"):
        if not pd.api.types.is_numeric_dtype(books[col]):
            _say(False, f"{col} is still text ({books[col].dtype}); use pd.to_numeric(..., errors=\"coerce\").")
        else:
            _check_series_equal(books[col].fillna(-1), ref[col].fillna(-1), col)


def _check_c4(books, removed=None):
    if _not_attempted(books, "books"):
        return
    _say(len(books) == 19, "19 books remain after removing the repeated listing." if len(books) == 19 else
         f"{len(books)} books remain; one book is listed twice once course and condition are cleaned: look for rows "
         "that agree on title, course, price and condition.")
    if removed is not None:
        _say(removed == 1, "one duplicate removed." if removed == 1 else f"removed is {removed}; expected 1.")


def _check_c5(books):
    if _not_attempted(books, "books"):
        return
    ref = _ref_clean_books().drop_duplicates(subset=["title", "course", "price", "condition"]).copy()
    medians = ref.groupby("condition")["price"].median()
    if "price_imputed" not in books.columns:
        return _say(False, "add a column price_imputed (True where the price was missing) before filling the prices.")
    _say(int(books["price_imputed"].sum()) == 3, "price_imputed marks the 3 books without a price." if int(books["price_imputed"].sum()) == 3
         else f"price_imputed marks {int(books['price_imputed'].sum())} books; 3 had no price.")
    filled = books.loc[books["price_imputed"].astype(bool)]
    wrong = [(r["title"], r["price"], medians.get(r["condition"])) for _, r in filled.iterrows()
             if pd.isna(r["price"]) or abs(r["price"] - medians.get(r["condition"], np.nan)) > 1e-6]
    _say(not wrong, "each missing price is the median price of books in the same condition." if not wrong else
         f"{wrong[0][0]} was filled with {wrong[0][1]}; the median for its condition is {wrong[0][2]}.")


def _check_c6(mae_ffill, mae_interpolate=None):
    if _not_attempted(mae_ffill, "mae_ffill"):
        return
    true = weather("victoria")["tmax"]
    gappy = weather_with_gaps("victoria")["tmax"]
    mask = gappy.isna()
    exp_f = float((gappy.ffill()[mask] - true[mask]).abs().mean())
    exp_i = float((gappy.interpolate()[mask] - true[mask]).abs().mean())
    _say(abs(mae_ffill - exp_f) < 1e-6, f"forward fill: mean absolute error {exp_f:.2f} °C." if abs(mae_ffill - exp_f) < 1e-6 else
         f"mae_ffill is {mae_ffill:.3f}; measure the error only on the missing days, against weather('victoria').")
    if mae_interpolate is not None:
        _say(abs(mae_interpolate - exp_i) < 1e-6, f"interpolation: mean absolute error {exp_i:.2f} °C." if abs(mae_interpolate - exp_i) < 1e-6
             else f"mae_interpolate is {mae_interpolate:.3f}; expected {exp_i:.3f}.")


_CHECKS = {"C1": _check_c1, "C2": _check_c2, "C3": _check_c3, "C4": _check_c4, "C5": _check_c5, "C6": _check_c6}
