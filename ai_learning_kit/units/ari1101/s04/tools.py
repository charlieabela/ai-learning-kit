"""ARI1101 Session 04 (Explore): data, look-inside helpers and practice checks for exploring data.

Used in the Session 04 notebooks as T = kit.tools() after kit.start("ARI1101", "S04").
T.penguins() returns the Palmer penguins (Horst, Hill and Gorman, 2020; data CC0, collected by
Dr Kristen Gorman and the Palmer Station Long Term Ecological Research programme, Antarctica).
T.weather_year() returns a year of daily weather for Malta from the Open-Meteo archive API
(data CC BY 4.0, ERA5 from the Copernicus Climate Change Service), stored in the kit as the
API returned it. The functions marked "look inside" in the notebooks (iqr_outliers,
pearson_by_hand, scatter_with_trend, corr_heatmap, sample_means) are written to be read:
kit.source(T.name) prints their code.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...._version import __version__

ROOT = Path(__file__).parent
SEED = 1101
DATA = ROOT / "data"
SAVED = ROOT / "saved"

PENGUINS_CITATION = (
    "Horst, A. M., Hill, A. P. and Gorman, K. B. (2020). palmerpenguins: Palmer Archipelago (Antarctica) "
    "penguin data. R package version 0.1.0, doi:10.5281/zenodo.3960218. Data collected by Gorman, K. B., "
    "Williams, T. D. and Fraser, W. R. (2014). PLoS ONE 9(3): e90081, and the Palmer Station LTER. Licence: CC0.")

ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
LUQA = {"latitude": 35.857, "longitude": 14.477}          # Malta International Airport, Luqa
YEAR_VARIABLES = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max,sunshine_duration"
HEADERS = {"User-Agent": f"ai-learning-kit/{__version__} (ARI1101 teaching notebook; "
                         "https://github.com/charlieabela/ai-learning-kit)"}

# The API request whose reply the kit stores; dev.saved_copies compares it with the live service in CI.
SAVED_REQUESTS = {
    "archive-luqa-2025.json": (ARCHIVE_API, {**LUQA, "start_date": "2025-01-01", "end_date": "2025-12-31",
                                             "daily": YEAR_VARIABLES, "timezone": "Europe/Malta", "models": "era5"}),
}


def _index() -> dict:
    path = SAVED / "saved.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


# --------------------------------------------------------------------------- data
def penguins() -> pd.DataFrame:
    """The Palmer penguins: 344 penguins of three species measured on three islands, 2007 to 2009.

    One row per penguin: species, island, bill length and depth (mm), flipper length (mm),
    body mass (g), sex and year. Missing values are NaN, as in the published file.
    """
    return pd.read_csv(DATA / "penguins.csv")


def weather_year() -> pd.DataFrame:
    """Daily weather for Malta in 2025 (Open-Meteo archive, ERA5 grid square nearest to Luqa).

    Columns: date, tmax and tmin (°C), rain (mm), wind (maximum, km/h) and sunshine (hours).
    The API returns sunshine in seconds; it is divided by 3600 here.
    """
    data = json.loads((SAVED / "archive-luqa-2025.json").read_text(encoding="utf-8"))
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={"time": "date", "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
                            "precipitation_sum": "rain", "wind_speed_10m_max": "wind",
                            "sunshine_duration": "sunshine"})
    df["sunshine"] = (df["sunshine"] / 3600).round(2)
    return df


def anscombe() -> pd.DataFrame:
    """Anscombe's quartet (Anscombe, 1973): four small datasets with almost the same statistics.

    Columns: dataset ("I" to "IV"), x and y. Each dataset has 11 points.
    """
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    ys = {"I": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
          "II": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
          "III": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
          "IV": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]}
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    rows = []
    for name, y in ys.items():
        for xi, yi in zip(x4 if name == "IV" else x, y):
            rows.append({"dataset": name, "x": xi, "y": yi})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- look inside
def iqr_outliers(values: pd.Series, k: float = 1.5) -> pd.Series:
    """Mark the values that lie unusually far from the middle half of the data (the IQR rule).

    Q1 and Q3 are the first and third quartiles, and the interquartile range is IQR = Q3 - Q1.
    A value below Q1 - k * IQR or above Q3 + k * IQR is marked as a possible outlier (k = 1.5
    is the usual choice, and the one a boxplot uses for its whiskers). Missing values are not
    marked. Returns True/False for every value.
    """
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    iqr = q3 - q1
    low, high = q1 - k * iqr, q3 + k * iqr
    return (values < low) | (values > high)


def pearson_by_hand(x, y, show: bool = True) -> float:
    """Pearson's correlation coefficient r, computed step by step as the lectures define it.

    1. Subtract the mean from every value: how far each point is from the centre.
    2. Multiply the two deviations of each point and add the products: positive when x and y
       are above (or below) their means together, negative when one is above and the other below.
    3. Divide by the square root of (sum of squared x deviations) * (sum of squared y deviations),
       which scales the result to lie between -1 and +1.
    Pairs where either value is missing are left out.
    """
    pairs = pd.DataFrame({"x": x, "y": y}).dropna()
    dx = pairs["x"] - pairs["x"].mean()
    dy = pairs["y"] - pairs["y"].mean()
    top = (dx * dy).sum()
    bottom = np.sqrt((dx ** 2).sum() * (dy ** 2).sum())
    r = top / bottom
    if show:
        print(f"{len(pairs)} pairs; sum of products of deviations = {top:.1f}; "
              f"scaling term = {bottom:.1f}; r = {top:.1f} / {bottom:.1f} = {r:.3f}")
    return float(r)


def scatter_with_trend(df: pd.DataFrame, x: str, y: str, group: str | None = None, ax=None, title: str = ""):
    """Scatter plot of y against x with a straight trend line, for all points or for each group.

    The trend line is the least-squares line from np.polyfit(x, y, 1): the straight line that
    is closest to the points. With group (for example "species"), each group gets its own
    colour and its own line, and a dashed grey line shows the trend of all points together.
    """
    import matplotlib.pyplot as plt
    data = df[[x, y] + ([group] if group else [])].dropna()
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))
    colours = ["#D37B35", "#4279B7", "#5B8C3A", "#8E5BA8", "#B8463E"]
    groups = [(None, data)] if group is None else list(data.groupby(group))
    for (name, part), colour in zip(groups, colours):
        ax.scatter(part[x], part[y], s=18, alpha=0.7, color=colour, label=name)
        slope, intercept = np.polyfit(part[x], part[y], 1)
        xs = np.linspace(part[x].min(), part[x].max(), 2)
        ax.plot(xs, slope * xs + intercept, color=colour, linewidth=2)
    if group is not None:
        slope, intercept = np.polyfit(data[x], data[y], 1)
        xs = np.linspace(data[x].min(), data[x].max(), 2)
        ax.plot(xs, slope * xs + intercept, color="#6e7781", linewidth=2, linestyle="--", label="all together")
        ax.legend(fontsize=8)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title:
        ax.set_title(title)
    return ax


def corr_heatmap(df: pd.DataFrame, method: str = "pearson", title: str = ""):
    """Draw the correlation matrix of the numeric columns as a grid of coloured squares.

    Each square holds r for one pair of columns: dark red for strong positive, dark blue
    for strong negative, pale for close to zero. The diagonal is always 1 (a column with itself).
    """
    import matplotlib.pyplot as plt
    corr = df.select_dtypes("number").corr(method=method)
    fig, ax = plt.subplots(figsize=(1.1 * len(corr) + 2, 0.9 * len(corr) + 1.5))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=40, ha="right")
    ax.set_yticks(range(len(corr)), corr.columns)
    for i in range(len(corr)):
        for j in range(len(corr)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 else "#1f2d3d")
    fig.colorbar(im, ax=ax, shrink=0.8, label=f"{method} r")
    ax.set_title(title or f"Correlation matrix ({method})")
    plt.tight_layout()
    plt.show()


def sample_means(values: pd.Series, n: int, repeats: int = 1000, seed: int = SEED) -> pd.Series:
    """Draw `repeats` random samples of n values (without replacement) and return the mean of each.

    Every sample gives a slightly different mean. The spread of these means shows how much a
    statistic computed from one sample of size n can differ from the value for all the data.
    """
    rng = np.random.default_rng(seed)
    data = values.dropna().to_numpy()
    return pd.Series([rng.choice(data, size=n, replace=False).mean() for _ in range(repeats)], name=f"mean of {n}")


# --------------------------------------------------------------------------- practice checks
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


def _check_e1(n_days, missing=None):
    if _not_attempted(n_days, "n_days"):
        return
    w = weather_year()
    _say(n_days == len(w), f"the table has {len(w)} days, one row per day of 2025." if n_days == len(w)
         else f"n_days is {n_days}; len(weather) or weather.shape[0] gives the number of rows.")
    if missing is not None:
        m = int(w.isna().sum().sum())
        _say(int(missing) == m, f"{m} missing values." if int(missing) == m else
             f"missing is {missing}; weather.isna().sum().sum() counts every missing value in the table.")


def _check_e2(stats):
    if _not_attempted(stats, "stats"):
        return
    w = weather_year()
    expected = {"tmax_mean": w["tmax"].mean(), "tmax_median": w["tmax"].median(), "tmax_std": w["tmax"].std(),
                "rain_mean": w["rain"].mean(), "rain_median": w["rain"].median()}
    for key, value in expected.items():
        got = stats.get(key) if isinstance(stats, dict) else None
        if got is None:
            _say(False, f"stats has no {key!r}.")
        elif _close(got, value, 0.002 if key.endswith("std") else 0.01):
            _say(True, f"{key} = {value:.2f}.")
        else:
            hint = (" pandas' .std() divides by n - 1 (the sample standard deviation); np.std divides by n."
                    if key.endswith("std") else "")
            _say(False, f"{key} is {got}; expected {value:.2f}.{hint}")


def _check_e3(n_outliers, threshold=None):
    if _not_attempted(n_outliers, "n_outliers"):
        return
    w = weather_year()
    wet = w.loc[w["rain"] > 0, "rain"]
    mask = iqr_outliers(wet)
    q3, iqr = wet.quantile(0.75), wet.quantile(0.75) - wet.quantile(0.25)
    upper = q3 + 1.5 * iqr
    _say(int(n_outliers) == int(mask.sum()), f"{int(mask.sum())} wet days lie above Q3 + 1.5 × IQR." if
         int(n_outliers) == int(mask.sum()) else
         f"n_outliers is {n_outliers}; among the {len(wet)} days with rain above 0 mm, {int(mask.sum())} lie above "
         f"Q3 + 1.5 × IQR = {upper:.2f} mm. Did you use only the wet days, and 1.5 × IQR?")
    if threshold is not None:
        _say(_close(threshold, upper), f"the upper limit is {upper:.2f} mm." if _close(threshold, upper) else
             f"threshold is {threshold}; Q3 = {q3:.2f}, IQR = {iqr:.2f}, so Q3 + 1.5 × IQR = {upper:.2f} mm.")


def _check_e4(monthly, warmest=None):
    if _not_attempted(monthly, "monthly"):
        return
    w = weather_year()
    exp = w.groupby(w["date"].dt.month)["tmax"].mean()
    got = pd.Series(monthly)
    ok = len(got) == 12 and np.allclose(got.to_numpy(dtype=float), exp.to_numpy(), atol=0.01)
    _say(ok, "the mean tmax of every month matches." if ok else
         "monthly should hold 12 values, the mean tmax of each month: weather.groupby(weather['date'].dt.month)['tmax'].mean().")
    if warmest is not None:
        best = int(exp.idxmax())
        _say(int(warmest) == best, f"month {best} was the warmest." if int(warmest) == best else
             f"month {warmest} is not the warmest; idxmax() on the monthly means gives the month with the largest value.")


def _check_e5(r_temp, r_sun_rain=None, method="pearson"):
    if _not_attempted(r_temp, "r_temp"):
        return
    w = weather_year()
    e1 = w["tmax"].corr(w["tmin"])
    _say(_close(r_temp, e1, 0.005), f"r(tmax, tmin) = {e1:.3f}." if _close(r_temp, e1, 0.005) else
         f"r_temp is {r_temp}; weather['tmax'].corr(weather['tmin']) gives {e1:.3f}.")
    if r_sun_rain is not None:
        e2 = w["sunshine"].corr(w["rain"])
        _say(_close(r_sun_rain, e2, 0.005), f"r(sunshine, rain) = {e2:.3f}." if _close(r_sun_rain, e2, 0.005) else
             f"r_sun_rain is {r_sun_rain}; expected {e2:.3f}.")


_CHECKS = {"E1": _check_e1, "E2": _check_e2, "E3": _check_e3, "E4": _check_e4, "E5": _check_e5}


def check(exercise: str, answer=None, **more) -> None:
    """Check a practice exercise, e.g. T.check("E1", n_days, missing=missing). Says what is wrong and why."""
    fn = _CHECKS.get(str(exercise).upper())
    if fn is None:
        raise KeyError(f"No exercise {exercise!r}. Exercises: {', '.join(_CHECKS)}")
    fn(answer, **more)
