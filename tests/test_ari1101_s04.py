"""ARI1101 S04: exploring data. The data ships with the kit; no network needed."""
import json

import numpy as np
import pandas as pd
import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(session=None, problem=None, book=None, folder=None, tools=None)


@pytest.fixture()
def T():
    kit.start("ARI1101", "S04")
    return kit.tools()


def test_session_contract():
    mod = registry.session("ARI1101", "S04")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.DATASETS == () and mod.EXTRAS == ""
    ctx = registry.tutor_context("ARI1101", "S04")
    json.dumps(ctx)
    assert "Available on request" in ctx["scope"] and "pearson_by_hand" in ctx["tools"]["functions"]


def test_penguins_as_published(T):
    p = T.penguins()
    assert p.shape == (344, 8) and p["sex"].isna().sum() == 11 and p["bill_length_mm"].isna().sum() == 2
    q = p.dropna(subset=["bill_length_mm"])
    assert round(q["body_mass_g"].mean(), 1) == 4201.8 and q["body_mass_g"].median() == 4050
    assert round(q["bill_length_mm"].corr(q["bill_depth_mm"]), 2) == -0.24          # Simpson's paradox overall ...
    within = q.groupby("species")[["bill_length_mm", "bill_depth_mm"]].apply(
        lambda g: g["bill_length_mm"].corr(g["bill_depth_mm"]))
    assert (within > 0.3).all()                                                     # ... positive within species


def test_weather_year_from_the_saved_reply(T):
    w = T.weather_year()
    assert w.shape == (365, 6) and w.isna().sum().sum() == 0
    assert w["sunshine"].max() <= 24 and round(w["tmax"].corr(w["tmin"]), 2) == 0.98
    assert T.iqr_outliers(w.loc[w["rain"] > 0, "rain"]).sum() == 10


def test_helpers_agree_with_pandas(T):
    a = T.anscombe()
    r = a.groupby("dataset")[["x", "y"]].apply(lambda g: g["x"].corr(g["y"]))
    assert len(a) == 44 and np.allclose(r, 0.816, atol=0.002)
    x, y = pd.Series([1, 2, 3, 4, 5.0]), pd.Series([2, 1, 4, 3, 5.0])
    assert abs(T.pearson_by_hand(x, y, show=False) - x.corr(y)) < 1e-12
    means = T.sample_means(pd.Series(range(100)), n=10, repeats=50)
    assert len(means) == 50 and means.between(0, 99).all()
    assert T.iqr_outliers(pd.Series([1, 2, 3, 4, 100])).tolist() == [False, False, False, False, True]


def test_checks(T, capsys):
    w = T.weather_year()
    T.check("E1", None)
    assert "Not attempted yet" in capsys.readouterr().out
    T.check("E1", len(w), missing=0)
    T.check("E2", {"tmax_mean": w["tmax"].mean(), "tmax_median": w["tmax"].median(), "tmax_std": w["tmax"].std(),
                   "rain_mean": w["rain"].mean(), "rain_median": w["rain"].median()})
    T.check("E4", w.groupby(w["date"].dt.month)["tmax"].mean(), warmest=7)
    T.check("E5", w["tmax"].corr(w["tmin"]), r_sun_rain=w["sunshine"].corr(w["rain"]))
    out = capsys.readouterr().out
    assert "Not yet" not in out and out.count("Correct") == 11
    T.check("E2", {"tmax_mean": 20.74, "tmax_median": 20.0, "tmax_std": float(np.std(w["tmax"])),
                   "rain_mean": 1.1, "rain_median": 0.0})
    assert "n - 1" in capsys.readouterr().out
