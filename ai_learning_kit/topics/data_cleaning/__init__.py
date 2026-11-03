"""Data cleaning: finding missing values, making formats consistent, removing duplicates and filling gaps.

The topic's notebooks (a walkthrough and a practice notebook) check themselves: the practice exercises are
checked with T.check(). The data are those the web-data topic collects (the fictional Campus Lets listings and
textbook exchange, and a month of daily weather from the Open-Meteo API); the kit supplies them
ready-collected, so that learners work on cleaning only. Steps that are hard for learners new to programming
are done by helpers that they may open ("look inside") but need not.
"""
from ...core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

TOPIC = "data-cleaning"
TITLE = "Cleaning data: missing values, consistent formats, duplicates and filling gaps"
LEVEL = "introductory"
BUILDS_ON = ("web-data",)
PROBLEM = None
TASKS = {}
TOOLS = ".tools"
DATASETS = ()
EXTRAS = ""

STAGES = {
    "missing": stage("Finding missing values, including text that stands for 'missing' (\"Price on request\", "
                     "\"Not stated\") and elements absent from the page", FOCUS),
    "formats": stage("Making formats consistent: text case and spelling, numbers stored as text with symbols and "
                     "units, mixed date formats, then removing duplicates", FOCUS),
    "imputation": stage("Dropping or filling missing values: mean, median, mode, a separate category, the median of a "
                        "group, forward fill and interpolation for daily data, an indicator column, and measuring "
                        "the error of a filling method when the true values are known", FOCUS),
    "knn": stage("Filling a value from the most similar complete rows (k-nearest-neighbour imputation)", ON_REQUEST,
                 how="T.knn_impute()",
                 note="an optional extension; explain the idea (similar rows, rescaling, averaging) only when asked"),
    "data": stage("The listings, the textbook exchange and the daily weather, collected as in the web-data topic",
                  BLACK_BOX, how="T.listings_raw(), T.textbooks_raw(), T.weather(), T.weather_with_gaps()",
                  taught_in="the web-data topic",
                  note="say only that the kit has already collected the data; how is the web-data topic's focus"),
    "pages_display": stage("Plotting filled gaps, and the practice checks", BLACK_BOX, how="T.plot_filled(), T.check()"),
    "advanced_imputation": stage("Model-based and multiple imputation, outlier detection", OUT_OF_SCOPE,
                                 taught_in="the exploring-data and modelling topics, and beyond"),
}
