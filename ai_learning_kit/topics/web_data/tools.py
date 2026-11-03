"""Web data: helpers for reading web pages and web APIs.

Used in the web-data notebooks as T = kit.tools() after kit.start("web-data").
T.get() sends a web request as requests.get() does; when a site or service cannot be reached,
it returns the kit's saved copy of the same response, so every notebook also runs offline.
The Campus Lets website is fictional and was made for teaching (T.SITE); the weather
data comes from the Open-Meteo API (open-meteo.com; data licensed CC BY 4.0, ERA5 from the
Copernicus Climate Change Service). The functions marked "look inside" in the notebooks
(scrape_listings, allowed and get itself) are written to be read: kit.source(T.name)
prints their code.
"""
from __future__ import annotations

import http.client
import json
import os
import time
import urllib.robotparser
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import pandas as pd
import requests
from bs4 import BeautifulSoup
from requests.structures import CaseInsensitiveDict

from ..._version import __version__

ROOT = Path(__file__).parent
SEED = 1101

# The fictional course website (published with GitHub Pages from the kit's repository).
SITE = os.environ.get("AILK_CAMPUS_LETS_SITE", "https://charlieabela.github.io/ai-learning-kit/sites/campus-lets/")
SITE_FILES = ROOT / "site"                 # the kit's copy of every page of the website
SAVED = ROOT / "saved"                     # the kit's copies of the API responses used in the notebooks

# The Open-Meteo APIs: free, no key needed, for non-commercial use.
ARCHIVE_API = "https://archive-api.open-meteo.com/v1/archive"
GEOCODING_API = "https://geocoding-api.open-meteo.com/v1/search"
WEATHER_VARIABLES = "temperature_2m_max,temperature_2m_min,precipitation_sum,wind_speed_10m_max"
LUQA = {"latitude": 35.857, "longitude": 14.477}          # Malta International Airport, Luqa
VICTORIA = {"latitude": 36.04473, "longitude": 14.24118}  # Victoria, Gozo, as the geocoding API gives it

HEADERS = {"User-Agent": f"ai-learning-kit/{__version__} (teaching notebook; "
                         "https://github.com/charlieabela/ai-learning-kit)"}


def _weather_params(place: dict, start: str, end: str) -> dict:
    return {"latitude": place["latitude"], "longitude": place["longitude"], "start_date": start, "end_date": end,
            "daily": WEATHER_VARIABLES, "timezone": "Europe/Malta", "models": "era5"}


# Every API request the notebooks make, with the file holding the kit's copy of its response.
# dev.saved_copies compares each with the live service (in CI) and can capture it again.
SAVED_REQUESTS = {
    "archive-luqa-2025-10.json": (ARCHIVE_API, _weather_params(LUQA, "2025-10-01", "2025-10-31")),
    "archive-luqa-dates-reversed.json": (ARCHIVE_API, _weather_params(LUQA, "2025-10-31", "2025-10-01")),
    "geocoding-victoria.json": (GEOCODING_API, {"name": "Victoria", "count": 10, "language": "en", "format": "json"}),
    "archive-victoria-2025-11.json": (ARCHIVE_API, _weather_params(VICTORIA, "2025-11-01", "2025-11-30")),
}


# --------------------------------------------------------------------------- web requests
def get(url, params=None, timeout=10):
    """Send a GET request, as requests.get(url, params=params) does, and return the response.

    The request names the notebook in its User-Agent header and waits at most `timeout`
    seconds. When the site or service cannot be reached (no internet, a blocked network, a
    site that is down or not published yet), and the kit holds a saved copy of the same
    request, the saved copy is returned instead, so the notebook still runs:
    where_from(response) says which you got. A request the kit has no copy of is sent as it is.
    """
    key = _key(url, params)
    saved = _saved(key)
    if saved is not None and os.environ.get("AILK_OFFLINE", "").lower() in ("1", "true", "yes"):
        return saved
    try:
        response = requests.get(url, params=params, headers=HEADERS, timeout=timeout)
    except requests.exceptions.RequestException as error:
        if saved is not None:
            return saved
        raise ConnectionError(f"Could not reach {url} ({type(error).__name__}). Check the address and your "
                              "internet connection; the kit has no saved copy of this request.") from None
    if saved is not None and response.status_code != saved.status_code:
        return saved            # for example, the course website is not published yet, or the API is busy
    response.from_saved_copy = False
    return response


def where_from(response) -> str:
    """Say whether a response came from the live site or service, or from the kit's saved copy."""
    if getattr(response, "from_saved_copy", False):
        return "the kit's saved copy (the live site could not be reached)"
    return "the live site"


def _key(url, params=None) -> str:
    """One name for a request, whatever the order of its parameters."""
    full = requests.Request("GET", url, params=params).prepare().url
    for base in {SITE, "https://charlieabela.github.io/ai-learning-kit/sites/campus-lets/"}:
        if full.startswith(base):
            return "site:" + full[len(base):].split("#")[0]
    parts = urlsplit(full)
    query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((parts.scheme, parts.netloc.lower(), parts.path, query, ""))


def _index() -> dict:
    path = SAVED / "saved.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def _response(content: bytes, status: int, content_type: str, url: str):
    r = requests.models.Response()
    r.status_code = status
    r.reason = http.client.responses.get(status, "")
    r._content = content
    r.headers = CaseInsensitiveDict({"Content-Type": content_type})
    r.url = url
    r.encoding = "utf-8"
    r.from_saved_copy = True
    return r


def _saved(key: str):
    """The kit's saved copy of a request as a requests.Response, or None."""
    if key.startswith("site:"):
        name = key[5:] or "index.html"
        path = SITE_FILES / name
        if "/" in name or not path.is_file():         # the kit holds the whole site, so any other page is missing
            return _response(b"<!DOCTYPE html><html><body><h1>404 Not Found</h1></body></html>\n", 404,
                             "text/html; charset=utf-8", SITE + name)
        kind = {".txt": "text/plain", ".css": "text/css"}.get(path.suffix, "text/html") + "; charset=utf-8"
        return _response(path.read_bytes(), 200, kind, SITE + name)
    for name, (url, params) in SAVED_REQUESTS.items():
        if _key(url, params) == key:
            meta = _index().get(name)
            if meta is None or not (SAVED / name).is_file():
                return None
            full = requests.Request("GET", url, params=params).prepare().url
            return _response((SAVED / name).read_bytes(), meta["status"], meta["content_type"], full)
    return None


def show_page(page, height: int = 360):
    """Show a web page roughly as a browser would, inside the notebook (page: a response or an address)."""
    from IPython.display import HTML, display
    if isinstance(page, str):
        page = get(page)
    soup = BeautifulSoup(page.text, "html.parser")
    main = soup.find("main") or soup.body or soup
    css = get(urljoin(SITE, "style.css")).text
    css = "\n".join(".ailk-page " + line if line.strip() and not line.startswith(" ") else line for line in css.splitlines())
    box = (f'<div class="ailk-page" style="border:1px solid #d0d7de;border-radius:6px;padding:8px 14px;'
           f'max-height:{height}px;overflow:auto;background:#fff;color:#1f2d3d">'
           f"<style>{css}</style>{main}</div>")
    display(HTML(box))


# --------------------------------------------------------------------------- scraping (look inside)
def allowed(url, user_agent: str = "*") -> bool:
    """Does the website's robots.txt allow a program to read this address?

    robots.txt is a plain-text file in which a website states which of its pages programs may
    read. Python's urllib.robotparser reads the rules and answers for one address.
    """
    rules = urllib.robotparser.RobotFileParser()
    rules.parse(get(urljoin(SITE, "robots.txt")).text.splitlines())
    return rules.can_fetch(user_agent, urljoin(SITE, url))


def crawl_delay(user_agent: str = "*") -> float:
    """The number of seconds robots.txt asks programs to wait between requests (0 when it says nothing)."""
    rules = urllib.robotparser.RobotFileParser()
    rules.parse(get(urljoin(SITE, "robots.txt")).text.splitlines())
    return float(rules.crawl_delay(user_agent) or 0)


def _text(article, class_name):
    """The text of the first element with this class inside an article, or None when it is absent."""
    element = article.find(class_=class_name)
    return element.get_text(strip=True) if element is not None else None


def parse_listings(html: str) -> list:
    """The listings on one Campus Lets page, as a list of dictionaries (one per listing)."""
    soup = BeautifulSoup(html, "html.parser")
    rows = []
    for article in soup.find_all("article", class_="listing"):
        rows.append({
            "id": article["id"],
            "title": _text(article, "title"),
            "locality": _text(article, "locality"),
            "price": _text(article, "price"),
            "bedrooms": _text(article, "bedrooms"),
            "size": _text(article, "size"),
            "furnished": _text(article, "furnished"),
            "listed": _text(article, "listed"),
        })
    return rows


def scrape_listings(start: str = "listings-1.html", delay: float | None = None, verbose: bool = True) -> pd.DataFrame:
    """Read every page of the Campus Lets listings by following the "Next page" links.

    For each page it checks robots.txt, downloads the page, keeps its listings, finds the link
    to the next page and waits `delay` seconds (by default the delay robots.txt asks for)
    before the next request. It stops when a page has no "Next page" link.
    """
    if delay is None:
        delay = crawl_delay()
    url, rows = urljoin(SITE, start), []
    while url is not None:
        if not allowed(url):
            print(f"robots.txt does not allow {url}; stopping.")
            break
        page = get(url)
        page.raise_for_status()
        found = parse_listings(page.text)
        rows.extend(found)
        if verbose:
            print(f"{url.rsplit('/', 1)[-1]}: {len(found)} listings")
        link = BeautifulSoup(page.text, "html.parser").find("a", class_="next")
        url = urljoin(url, link["href"]) if link is not None else None
        if url is not None:
            time.sleep(delay)
    return pd.DataFrame(rows)


def listings_raw() -> pd.DataFrame:
    """The 36 listings exactly as the scraping walkthrough collects them: every value is text, or None."""
    rows = []
    for n in (1, 2, 3):
        rows.extend(parse_listings((SITE_FILES / f"listings-{n}.html").read_text(encoding="utf-8")))
    return pd.DataFrame(rows)


def textbooks_raw() -> pd.DataFrame:
    """The 20 books of the textbook exchange as the scraping practice collects them (text, or None)."""
    rows = []
    for n in (1, 2):
        soup = BeautifulSoup((SITE_FILES / f"textbooks-{n}.html").read_text(encoding="utf-8"), "html.parser")
        for article in soup.find_all("article", class_="book"):
            rows.append({key: _text(article, key) for key in ("title", "course", "price", "condition", "rating", "posted")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- weather data from the API
def weather_frame(data: dict) -> pd.DataFrame:
    """The daily part of an Open-Meteo archive response as a DataFrame with short column names."""
    df = pd.DataFrame(data["daily"])
    df["time"] = pd.to_datetime(df["time"])
    return df.rename(columns={"time": "date", "temperature_2m_max": "tmax", "temperature_2m_min": "tmin",
                              "precipitation_sum": "rain", "wind_speed_10m_max": "wind"})


def victoria() -> dict:
    """Victoria (Gozo) as the geocoding API describes it: the result whose country code is MT."""
    results = get(GEOCODING_API, params=SAVED_REQUESTS["geocoding-victoria.json"][1]).json()["results"]
    return next(r for r in results if r.get("country_code") == "MT")


def weather(place: str = "luqa") -> pd.DataFrame:
    """A month of daily weather from the archive API: "luqa" (October 2025) or "victoria" (November 2025)."""
    if place == "luqa":
        url, params = SAVED_REQUESTS["archive-luqa-2025-10.json"]
    elif place == "victoria":
        url, params = SAVED_REQUESTS["archive-victoria-2025-11.json"]
    else:
        raise ValueError('place is "luqa" or "victoria"')
    return weather_frame(get(url, params=params).json())


# --------------------------------------------------------------------------- practice checks
def _say(ok: bool, text: str) -> bool:
    print(("Correct: " if ok else "Not yet: ") + text)
    return ok


def _not_attempted(value, name) -> bool:
    if value is None:
        print(f"Not attempted yet: complete the cell above, so that {name} holds your answer, and run this check again.")
        return True
    return False


def _ref_books() -> pd.DataFrame:
    return textbooks_raw()


def check(exercise: str, answer=None, **more) -> None:
    """Check a practice exercise, e.g. T.check("S1", status, n_books=n_books). Says what is wrong and why."""
    fn = _CHECKS.get(str(exercise).upper())
    if fn is None:
        raise KeyError(f"No exercise {exercise!r}. Exercises: {', '.join(_CHECKS)}")
    fn(answer, **more)


def _check_s1(status, n_books=None):
    if _not_attempted(status, "status"):
        return
    _say(status == 200, "status is 200 (OK)." if status == 200 else
         f"status is {status}; 200 means the page was returned. Print page.status_code to see the code.")
    if _not_attempted(n_books, "n_books"):
        return
    _say(n_books == 10, "page 1 lists 10 books." if n_books == 10 else
         f"n_books is {n_books}, but page 1 has 10 <article class=\"book\"> elements. "
         "find_all(\"article\", class_=\"book\") finds them; len() counts them.")


def _check_s2(first_book):
    if _not_attempted(first_book, "first_book"):
        return
    expected = _ref_books().iloc[0].to_dict()
    if not isinstance(first_book, dict):
        return _say(False, "first_book should be a dictionary, e.g. {\"title\": ..., \"course\": ...}.")
    missing = [k for k in expected if k not in first_book]
    if missing:
        return _say(False, f"first_book has no key(s) {', '.join(missing)}. Use the names title, course, price, "
                           "condition, rating and posted.")
    wrong = {k: first_book[k] for k in expected if str(first_book[k]).strip() != str(expected[k])}
    if wrong:
        k, v = next(iter(wrong.items()))
        return _say(False, f"{k} is {v!r}, but the page shows {expected[k]!r}. Did you take the text of the element "
                           f"with class \"{k}\" inside the first article, with get_text(strip=True)?")
    _say(True, "every field of the first book matches the page.")


def _check_s3(books):
    if _not_attempted(books, "books"):
        return
    expected = _ref_books()
    if not isinstance(books, pd.DataFrame):
        return _say(False, "books should be a DataFrame: pd.DataFrame(rows), where rows is your list of dictionaries.")
    if len(books) != 20:
        hint = " Did you read both pages?" if len(books) == 10 else ""
        return _say(False, f"books has {len(books)} rows; the two pages hold 20 books.{hint}")
    cols = list(expected.columns)
    if sorted(books.columns) != sorted(cols):
        return _say(False, f"books has the columns {list(books.columns)}; expected {cols}.")
    absent = {c: int(books[c].isna().sum()) for c in cols}
    exp_absent = {c: int(expected[c].isna().sum()) for c in cols}
    if absent != exp_absent:
        c = next(c for c in cols if absent[c] != exp_absent[c])
        return _say(False, f"column {c} has {absent[c]} empty values; the pages leave it out for {exp_absent[c]} book(s). "
                           "When an element is absent, find() returns None: keep None rather than skipping the book.")
    same = books[cols].astype(object).where(books[cols].notna(), None).values.tolist() == \
        expected.astype(object).where(expected.notna(), None).values.tolist()
    _say(same, "all 20 books match the pages, with None where an element is absent." if same else
         "the values differ from the pages in at least one row; compare your table with the pages in the browser.")


def _check_s4(answers):
    if _not_attempted(answers, "robots_answers"):
        return
    expected = {"textbooks-2.html": True, "admin/sellers.html": False, "listings-3.html": True}
    for path, ok in expected.items():
        given = answers.get(path) if isinstance(answers, dict) else None
        _say(given == ok, f"{path}: {'allowed' if ok else 'not allowed'}." if given == ok else
             f"{path}: you said {given}; robots.txt {'does not mention it, so it is allowed' if ok else 'disallows everything under admin/'}.")


def _check_s5(place):
    if _not_attempted(place, "malta"):
        return
    ref = victoria()
    if not isinstance(place, dict):
        return _say(False, "malta should be the dictionary of the result whose country_code is \"MT\".")
    ok = place.get("country_code") == "MT" and place.get("latitude") == ref["latitude"] and place.get("longitude") == ref["longitude"]
    _say(ok, f"Victoria in Gozo is at latitude {ref['latitude']}, longitude {ref['longitude']}." if ok else
         f"this result is in {place.get('country', '?')} ({place.get('country_code', '?')}). The first result is not "
         "necessarily the one you want: look for country_code == \"MT\".")


def _check_s6(df, hottest=None, total_rain=None):
    if _not_attempted(df, "gozo"):
        return
    ref = weather("victoria")
    if not isinstance(df, pd.DataFrame) or len(df) != len(ref):
        return _say(False, f"gozo should be a DataFrame with one row per day of November 2025 ({len(ref)} rows).")
    _say(True, f"gozo has {len(ref)} rows, one per day.")
    if hottest is not None:
        day = pd.Timestamp(ref.loc[ref["tmax"].idxmax(), "date"])
        ok = pd.Timestamp(hottest) == day
        _say(ok, f"the warmest day was {day.date()}." if ok else
             f"{pd.Timestamp(hottest).date()} is not the day with the highest temperature_2m_max; use idxmax().")
    if total_rain is not None:
        rain = round(float(ref["rain"].sum()), 1)
        ok = abs(float(total_rain) - rain) < 0.05
        _say(ok, f"total rainfall {rain} mm." if ok else f"total_rain is {total_rain}; the daily precipitation_sum adds up to {rain} mm.")


_CHECKS = {"S1": _check_s1, "S2": _check_s2, "S3": _check_s3, "S4": _check_s4, "S5": _check_s5, "S6": _check_s6}
