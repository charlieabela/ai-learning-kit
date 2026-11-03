"""web-data (web pages and web APIs with saved copies) and data-cleaning (which reuses its data). No network needed."""
import json

import numpy as np
import pandas as pd
import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry
from ai_learning_kit.core.session import ON_REQUEST, scope_text


@pytest.fixture(autouse=True)
def _offline(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AILK_OFFLINE", "1")          # every request the kit has a copy of is served from it
    kit._state.update(topic=None, problem=None, book=None, folder=None, tools=None)


@pytest.fixture()
def W():
    kit.start("web-data")
    return kit.tools()


@pytest.fixture()
def C():
    kit.start("data-cleaning")
    return kit.tools()


@pytest.mark.parametrize("name, builds_on, own, not_own", [
    ("web-data", ("data-formats",), "scrape_listings", "knn_impute"),
    ("data-cleaning", ("web-data",), "knn_impute", "scrape_listings"),
])
def test_topic_contract_and_scope(name, builds_on, own, not_own):
    mod = registry.topic(name)
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.SAVES_WORK is False
    assert mod.DATASETS == () and mod.EXTRAS == "" and mod.LEVEL == "introductory" and mod.BUILDS_ON == builds_on
    assert {st["visibility"] for st in mod.STAGES.values()} >= {"focus", ON_REQUEST, "black_box", "out_of_scope"}
    text = scope_text(mod)
    assert "Available on request" in text and "kit.source(T.<name>)" in text
    ctx = registry.tutor_context(name)
    json.dumps(ctx)
    # each topic lists only its own helpers, so a tutor keeps to the topic's focus
    assert own in ctx["tools"]["functions"] and not_own not in ctx["tools"]["functions"]


def test_published_site_is_the_kits_copy():
    from pathlib import Path
    from ai_learning_kit.topics.web_data import tools
    docs = Path(__file__).resolve().parents[1] / "docs" / "sites" / "campus-lets"
    names = sorted(p.name for p in tools.SITE_FILES.iterdir())
    assert names == sorted(p.name for p in docs.iterdir())
    for name in names:
        assert (tools.SITE_FILES / name).read_bytes() == (docs / name).read_bytes(), name


def test_get_returns_a_response_from_the_saved_copy(W):
    page = W.get(W.SITE + "listings-2.html")
    assert page.status_code == 200 and page.headers["Content-Type"].startswith("text/html")
    assert "Page 2 of 3" in page.text and page.from_saved_copy
    assert "saved copy" in W.where_from(page)
    assert W._key("https://x.org/a", {"b": 1, "a": 2}) == W._key("https://x.org/a?a=2&b=1")


def test_robots_and_crawl(W, capsys):
    assert W.allowed("textbooks-2.html") and not W.allowed("admin/sellers.html")
    assert W.crawl_delay() == 1.0
    df = W.scrape_listings(delay=0)
    assert df.shape == (36, 8)
    assert df.isna().sum().to_dict() == {"id": 0, "title": 0, "locality": 0, "price": 0, "bedrooms": 2, "size": 5,
                                          "furnished": 0, "listed": 0}
    assert df.equals(W.listings_raw())
    books = W.textbooks_raw()
    assert books.shape == (20, 6) and books["condition"].isna().sum() == 2 and books["rating"].isna().sum() == 2


def test_cleaning_the_listings_finds_two_reposted_listings(C):
    df = C.listings_raw()
    df["locality"] = (df["locality"].str.strip().str.lower()
                      .replace({"gzira": "gżira", "san gwann": "san ġwann", "pieta": "pietà"}).str.title())
    df["furnished"] = df["furnished"].str.lower().replace({"not stated": None})
    df["price"] = pd.to_numeric(df["price"].str.replace("€", "").str.replace(",", "").str.replace("/ month", "")
                                .str.replace("EUR", "").str.strip(), errors="coerce")
    assert df["price"].isna().sum() == 4
    assert sorted(df["locality"].unique()) == ["Birkirkara", "Gżira", "Msida", "Pietà", "San Ġwann", "Sliema",
                                               "Swieqi", "Ta' Xbiex"]
    subset = ["title", "locality", "price", "bedrooms", "size", "furnished"]
    assert df.duplicated(subset=subset).sum() == 2
    dates = C.parse_dates(df["listed"])
    assert dates.notna().all() and dates.min() == pd.Timestamp("2026-09-28") and dates.max() == pd.Timestamp("2026-10-17")


def test_knn_impute_on_a_small_table(C):
    df = pd.DataFrame({"size": [30, 32, 80, 85, 82], "bedrooms": [0, 0, 2, 2, np.nan]})
    filled = C.knn_impute(df, "bedrooms", using=["size"], k=2)
    assert filled.iloc[4] == 2 and filled.iloc[:4].tolist() == [0, 0, 2, 2]


def test_checks_say_not_attempted_and_accept_the_reference(W, capsys):
    W.check("S1", None)
    assert "Not attempted yet" in capsys.readouterr().out
    W.check("S1", 200, n_books=10)
    W.check("S3", W.textbooks_raw())
    W.check("S4", {"textbooks-2.html": True, "admin/sellers.html": False, "listings-3.html": True})
    out = capsys.readouterr().out
    assert "Not yet" not in out and out.count("Correct") == 6
    W.check("S1", 404, n_books=12)
    assert capsys.readouterr().out.count("Not yet") == 2


def test_cleaning_checks_accept_the_reference_and_explain_mistakes(C, capsys):
    C.check("C1", None)
    assert "Not attempted yet" in capsys.readouterr().out
    raw = C.textbooks_raw().replace({"price": {"Make an offer": None}, "rating": {"No rating yet": None}})
    C.check("C1", raw.isna().sum().to_dict())
    clean = C._ref_clean_books()
    C.check("C2", clean)
    C.check("C3", clean)
    true, gappy = C.weather("victoria")["tmax"], C.weather_with_gaps("victoria")["tmax"]
    gap = gappy.isna()
    C.check("C6", float((gappy.ffill()[gap] - true[gap]).abs().mean()),
            mae_interpolate=float((gappy.interpolate()[gap] - true[gap]).abs().mean()))
    out = capsys.readouterr().out
    assert "Not yet" not in out and "DAT101" in out
    C.check("C4", clean)                       # the duplicate is still there
    assert "20 books remain" in capsys.readouterr().out
    with pytest.raises(KeyError, match="C1"):
        C.check("S1", 200)                     # the scraping exercises belong to web-data
