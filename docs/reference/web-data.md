# Obtaining data from web pages and web APIs

`web-data` · level **introductory** · builds on [`data-formats`](data-formats.md) · pip extra `none`

Web data: obtaining data from web pages and web APIs.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("web-data")
T = kit.tools()
```

Written for the **introductory** level: Learners who are new to programming and to data science, still getting used to Python, NumPy, pandas and Matplotlib.

## The three layers

### In focus: learners do these themselves

- Requesting a web page or an API: the address, the parameters, the status code, the content type and the text of the response (requests.get, here through T.get)
- Reading HTML with Beautiful Soup: elements, classes and attributes; find, find_all and get_text; a list of dictionaries turned into a DataFrame
- Reading a web API's JSON reply: response.json(), dictionaries and lists, choosing the right result, building a DataFrame
- Scraping responsibly: robots.txt, waiting between requests, identifying the program, terms of use, personal data, and preferring an API when one exists

### On request: the kit does these, and learners may look inside

- Following the 'Next page' links automatically, checking robots.txt and waiting between pages (with `T.scrape_listings(), T.allowed(), T.crawl_delay()`)
- Saved copies of pages and API responses, used when the network fails (with `T.get(), T.where_from()`)

### Support: the kit does these for the learner

- Showing a page as a browser would, and the practice checks (with `T.show_page(), T.check()`)

### Left out of this topic

- Pages built by JavaScript, logging in, browser automation (Selenium, Playwright), API keys and OAuth (taught in beyond this topic, when needed)

## Helpers: `T = kit.tools()`

Web data: helpers for reading web pages and web APIs.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### On request (look inside)

- **`T.get(url, params=None, timeout=10)`**: Send a GET request, as requests.get(url, params=params) does, and return the response.
- **`T.where_from(response) -> str`**: Say whether a response came from the live site or service, or from the kit's saved copy.
- **`T.allowed(url, user_agent: str = '*') -> bool`**: Does the website's robots.txt allow a program to read this address?
- **`T.crawl_delay(user_agent: str = '*') -> float`**: The number of seconds robots.txt asks programs to wait between requests (0 when it says nothing).
- **`T.scrape_listings(start: str = 'listings-1.html', delay: float | None = None, verbose: bool = True) -> pd.DataFrame`**: Read every page of the Campus Lets listings by following the "Next page" links.

### Support

- **`T.show_page(page, height: int = 360)`**: Show a web page roughly as a browser would, inside the notebook (page: a response or an address).
- **`T.check(exercise: str, answer=None, **more) -> None`**: Check a practice exercise, e.g. T.check("S1", status, n_books=n_books). Says what is wrong and why.

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.parse_listings(html: str) -> list`**: The listings on one Campus Lets page, as a list of dictionaries (one per listing).
- **`T.listings_raw() -> pd.DataFrame`**: The 36 listings exactly as the scraping walkthrough collects them: every value is text, or None.
- **`T.textbooks_raw() -> pd.DataFrame`**: The 20 books of the textbook exchange as the scraping practice collects them (text, or None).
- **`T.weather_frame(data: dict) -> pd.DataFrame`**: The daily part of an Open-Meteo archive response as a DataFrame with short column names.
- **`T.victoria() -> dict`**: Victoria (Gozo) as the geocoding API describes it: the result whose country code is MT.
- **`T.weather(place: str = 'luqa') -> pd.DataFrame`**: A month of daily weather from the archive API: "luqa" (October 2025) or "victoria" (November 2025).

### Constants

`T.ARCHIVE_API` (str), `T.GEOCODING_API` (str), `T.HEADERS` (dict), `T.LUQA` (dict), `T.ROOT` (Path), `T.SAVED` (Path), `T.SAVED_REQUESTS` (dict), `T.SEED` (int), `T.SITE` (str), `T.SITE_FILES` (Path), `T.VICTORIA` (dict), `T.WEATHER_VARIABLES` (str)
