# Cleaning data: missing values, consistent formats, duplicates and filling gaps

`data-cleaning` · level **introductory** · builds on [`web-data`](web-data.md) · pip extra `none`

Data cleaning: finding missing values, making formats consistent, removing duplicates and filling gaps.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("data-cleaning")
T = kit.tools()
```

Written for the **introductory** level: Learners who are new to programming and to data science, still getting used to Python, NumPy, pandas and Matplotlib.

## The three layers

### In focus: learners do these themselves

- Finding missing values, including text that stands for 'missing' ("Price on request", "Not stated") and elements absent from the page
- Making formats consistent: text case and spelling, numbers stored as text with symbols and units, mixed date formats, then removing duplicates
- Dropping or filling missing values: mean, median, mode, a separate category, the median of a group, forward fill and interpolation for daily data, an indicator column, and measuring the error of a filling method when the true values are known

### On request: the kit does these, and learners may look inside

- Filling a value from the most similar complete rows (k-nearest-neighbour imputation) (with `T.knn_impute()`)

### Support: the kit does these for the learner

- The listings, the textbook exchange and the daily weather, collected as in the web-data topic (with `T.listings_raw(), T.textbooks_raw(), T.weather(), T.weather_with_gaps()`; taught in the web-data topic)
- Plotting filled gaps, and the practice checks (with `T.plot_filled(), T.check()`)

### Left out of this topic

- Model-based and multiple imputation, outlier detection (taught in the exploring-data and modelling topics, and beyond)

## Helpers: `T = kit.tools()`

Data cleaning: finding missing values, making formats consistent, removing duplicates and filling gaps.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### On request (look inside)

- **`T.knn_impute(df: pd.DataFrame, column: str, using: list, k: int = 3) -> pd.Series`**: Fill the missing values of `column` with the average of the k most similar complete rows.

### Support

- **`T.weather_with_gaps(place: str = 'luqa') -> pd.DataFrame`**: The same month with tmax and tmin removed on six days, as if the station had failed.
- **`T.plot_filled(dates, true, gappy, filled: dict, title: str = '')`**: Plot the true values, the recorded values with gaps, and the values each method put in the gaps.
- **`T.check(exercise: str, answer=None, **more) -> None`**: Check a practice exercise, e.g. T.check("C1", missing_counts). Says what is wrong and why.

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.parse_dates(text: pd.Series, formats=('%Y-%m-%d', '%d/%m/%Y', '%d %b %Y')) -> pd.Series`**: Turn text dates written in several formats into dates, trying each format in turn.

### From other topics

Supplied ready-made here; each is taught in its own topic.

- `T.get`, from [`web-data`](web-data.md)
- `T.listings_raw`, from [`web-data`](web-data.md)
- `T.show_page`, from [`web-data`](web-data.md)
- `T.textbooks_raw`, from [`web-data`](web-data.md)
- `T.weather`, from [`web-data`](web-data.md)
- `T.where_from`, from [`web-data`](web-data.md)

### Constants

`T.DATE_FORMATS` (tuple), `T.GAPS` (dict), `T.LUQA` (dict), `T.SEED` (int), `T.SITE` (str), `T.VICTORIA` (dict)
