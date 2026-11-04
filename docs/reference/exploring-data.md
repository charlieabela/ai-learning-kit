# Exploring data: descriptive statistics, distributions, correlation and visualisation

`exploring-data` · level **introductory** · builds on [`data-cleaning`](data-cleaning.md) · pip extra `none`

Exploring data: descriptive statistics, distributions, correlation and visualisation.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("exploring-data")
T = kit.tools()
```

Written for the **introductory** level: Learners who are new to programming and to data science, still getting used to Python, NumPy, pandas and Matplotlib.

## The three layers

### In focus: learners do these themselves

- A first look: shape, columns, data types, missing values, and the kind of each variable (categorical nominal or ordinal, numerical discrete or continuous)
- Descriptive statistics of one variable: mean, median, mode, range, variance, standard deviation, quartiles, interquartile range, and skewness read from mean versus median
- Distributions: histograms (and the effect of the number of bins), boxplots, and the 1.5 × IQR rule for possible outliers
- Categorical variables: value_counts, bar charts, crosstab, and comparing groups with groupby and side-by-side boxplots
- Relationships between two numerical variables: scatter plots, Pearson's r and Spearman's rank correlation, correlation is not causation, why to plot before trusting r (Anscombe's quartet), and a relationship that reverses within groups (Simpson's paradox)
- Choosing a chart for a question and a data type; labelled axes with units and honest scales

### On request: the kit does these, and learners may look inside

- Helpers students may open: the IQR rule, Pearson's r computed step by step, scatter plots with trend lines per group, the correlation heatmap, repeated sample means (with `T.iqr_outliers(), T.pearson_by_hand(), T.scatter_with_trend(), T.corr_heatmap(), T.sample_means()`)
- A preview of sampling: the penguins are a sample, and a statistic varies from sample to sample (with `T.sample_means()`; taught in sampling techniques, beyond this topic)

### Support: the kit does these for the learner

- Loading the stored datasets and checking the practice exercises (with `T.penguins(), T.weather_year(), T.anscombe(), T.check()`)

### Left out of this topic

- Hypothesis tests, confidence intervals, p-values for correlations, regression modelling (taught in the modelling topic, and beyond)

## Helpers: `T = kit.tools()`

Exploring data: data, look-inside helpers and practice checks for exploring data.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### On request (look inside)

- **`T.iqr_outliers(values: pd.Series, k: float = 1.5) -> pd.Series`**: Mark the values that lie unusually far from the middle half of the data (the IQR rule).
- **`T.pearson_by_hand(x, y, show: bool = True) -> float`**: Pearson's correlation coefficient r, computed step by step as the lectures define it.
- **`T.scatter_with_trend(df: pd.DataFrame, x: str, y: str, group: str | None = None, ax=None, title: str = '')`**: Scatter plot of y against x with a straight trend line, for all points or for each group.
- **`T.corr_heatmap(df: pd.DataFrame, method: str = 'pearson', title: str = '')`**: Draw the correlation matrix of the numeric columns as a grid of coloured squares.
- **`T.sample_means(values: pd.Series, n: int, repeats: int = 1000, seed: int = 1101) -> pd.Series`**: Draw `repeats` random samples of n values (without replacement) and return the mean of each.

### Support

- **`T.penguins() -> pd.DataFrame`**: The Palmer penguins: 344 penguins of three species measured on three islands, 2007 to 2009.
- **`T.weather_year() -> pd.DataFrame`**: Daily weather for Malta in 2025 (Open-Meteo archive, ERA5 grid square nearest to Luqa).
- **`T.anscombe() -> pd.DataFrame`**: Anscombe's quartet (Anscombe, 1973): four small datasets with almost the same statistics.
- **`T.check(exercise: str, answer=None, **more) -> None`**: Check a practice exercise, e.g. T.check("E1", n_days, missing=missing). Says what is wrong and why.

### Constants

`T.ARCHIVE_API` (str), `T.DATA` (Path), `T.HEADERS` (dict), `T.LUQA` (dict), `T.PENGUINS_CITATION` (str), `T.ROOT` (Path), `T.SAVED` (Path), `T.SAVED_REQUESTS` (dict), `T.SEED` (int), `T.YEAR_VARIABLES` (str)
