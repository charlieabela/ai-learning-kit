"""ARI1101 Session 04: exploring data (the Explore phase of OSEMN).

A session without graded tasks: its walkthrough and practice notebook check themselves, and
the practice exercises are checked with T.check(). The kit supplies the Palmer penguins, a year
of daily weather for Malta from the Open-Meteo archive API (stored as the API returned it),
Anscombe's quartet, and helpers for the harder steps, which students may open ("look inside")
but need not. The concepts come from the unit's lectures on descriptive statistics,
correlation and visualisation; sampling, the next lecture topic, is previewed only.
"""
from ....core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

UNIT = "ARI1101"
SESSION = "S04"
TITLE = "Exploring data: descriptive statistics, distributions, correlation and visualisation"
PROBLEM = None
TASKS = {}
TOOLS = ".tools"
DATASETS = ()                       # the data ships with the kit
EXTRAS = ""

STAGES = {
    "first_look": stage("A first look: shape, columns, data types, missing values, and the kind of each variable "
                        "(categorical nominal or ordinal, numerical discrete or continuous)", FOCUS),
    "descriptive": stage("Descriptive statistics of one variable: mean, median, mode, range, variance, standard "
                         "deviation, quartiles, interquartile range, and skewness read from mean versus median",
                         FOCUS, note="pandas .std() and .var() divide by n - 1 (sample statistics); np.std divides "
                                     "by n. Connect each statistic to the lecture definition in plain words."),
    "distributions": stage("Distributions: histograms (and the effect of the number of bins), boxplots, and the "
                           "1.5 × IQR rule for possible outliers", FOCUS),
    "categorical": stage("Categorical variables: value_counts, bar charts, crosstab, and comparing groups with "
                         "groupby and side-by-side boxplots", FOCUS),
    "correlation": stage("Relationships between two numerical variables: scatter plots, Pearson's r and "
                         "Spearman's rank correlation, correlation is not causation, why to plot before trusting r "
                         "(Anscombe's quartet), and a relationship that reverses within groups (Simpson's paradox)",
                         FOCUS),
    "charts": stage("Choosing a chart for a question and a data type; labelled axes with units and honest scales",
                    FOCUS),
    "look_inside": stage("Helpers students may open: the IQR rule, Pearson's r computed step by step, scatter "
                         "plots with trend lines per group, the correlation heatmap, repeated sample means",
                         ON_REQUEST, how="T.iqr_outliers(), T.pearson_by_hand(), T.scatter_with_trend(), "
                                         "T.corr_heatmap(), T.sample_means()",
                         note="explain the code step by step only when asked; np.polyfit is the least-squares line, "
                              "say only that it is the straight line closest to the points (regression comes later)"),
    "sampling": stage("A preview of sampling: the penguins are a sample, and a statistic varies from sample to "
                      "sample", ON_REQUEST, how="T.sample_means()", taught_in="the next ARI1101 lecture (sampling "
                                                                           "techniques)"),
    "data_and_checks": stage("Loading the stored datasets and checking the practice exercises", BLACK_BOX,
                             how="T.penguins(), T.weather_year(), T.anscombe(), T.check()"),
    "inference": stage("Hypothesis tests, confidence intervals, p-values for correlations, regression modelling",
                       OUT_OF_SCOPE, taught_in="later ARI1101 sessions (Model) and later units"),
}
