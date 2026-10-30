"""ARI1101 Session 03: obtaining data from web pages and web APIs, and cleaning it (Obtain and Scrub).

A session without graded tasks: its four notebooks (two walkthroughs, two practice notebooks)
check themselves, and the practice exercises are checked with T.check(). The kit supplies the
helpers (T = kit.tools()), the fictional Campus Lets website the notebooks scrape, and saved
copies of every web page and API response the notebooks use, so the notebooks also run offline.
Steps that are hard for first-year students are done by helpers that students may open
("look inside") but need not.
"""
from ....core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

UNIT = "ARI1101"
SESSION = "S03"
TITLE = "Obtaining data from the web, and cleaning it"
PROBLEM = None                      # no case problem: a fictional website and the Open-Meteo API
TASKS = {}                          # no answer book: the notebooks check themselves
TOOLS = ".tools"                    # T = kit.tools()
DATASETS = ()                       # nothing to download in advance: pages and API responses ship with the kit
EXTRAS = ""                         # requests and Beautiful Soup are part of the kit's own requirements

STAGES = {
    "http": stage("Requesting a web page or an API: the address, the parameters, the status code, the content type "
                  "and the text of the response (requests.get, here through T.get)", FOCUS,
                  note="T.get sends the same request as requests.get and returns the same response object; when the "
                       "site cannot be reached it returns the kit's saved copy. Say this when asked about T.get."),
    "html": stage("Reading HTML with Beautiful Soup: elements, classes and attributes; find, find_all and "
                  "get_text; a list of dictionaries turned into a DataFrame", FOCUS),
    "api_json": stage("Reading a web API's JSON reply: response.json(), dictionaries and lists, choosing the right "
                      "result, building a DataFrame", FOCUS),
    "responsible": stage("Scraping responsibly: robots.txt, waiting between requests, identifying the program, "
                         "terms of use, personal data, and preferring an API when one exists", FOCUS),
    "crawl": stage("Following the 'Next page' links automatically, checking robots.txt and waiting between pages",
                   ON_REQUEST, how="T.scrape_listings(), T.allowed(), T.crawl_delay()",
                   note="students loop over a known list of pages themselves; the helper does the general case. "
                        "Explain its code only when the student asks, step by step."),
    "saved_copies": stage("Saved copies of pages and API responses, used when the network fails", ON_REQUEST,
                          how="T.get(), T.where_from()",
                          note="a convenience so that the notebooks run offline, not a technique students must learn"),
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
    "pages_display": stage("Showing a page as a browser would, plotting filled gaps, and the practice checks",
                           BLACK_BOX, how="T.show_page(), T.plot_filled(), T.check()"),
    "dynamic": stage("Pages built by JavaScript, logging in, browser automation (Selenium, Playwright), API keys "
                     "and OAuth", OUT_OF_SCOPE, taught_in="later units, when needed"),
    "advanced_imputation": stage("Model-based and multiple imputation, outlier detection", OUT_OF_SCOPE,
                                 taught_in="later ARI1101 sessions (Explore and Model) and later units"),
}
