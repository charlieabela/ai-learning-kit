"""ARI1101 S05: modelling. The data ships with the kit; no network needed."""
import json
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry

matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(session=None, problem=None, book=None, folder=None, tools=None)


@pytest.fixture()
def T():
    kit.start("ARI1101", "S05")
    return kit.tools()


def test_session_contract():
    mod = registry.session("ARI1101", "S05")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.DATASETS == () and mod.EXTRAS == ""
    ctx = registry.tutor_context("ARI1101", "S05")
    json.dumps(ctx)
    assert "Available on request" in ctx["scope"] and "line_by_hand" in ctx["tools"]["functions"]


def test_cars_as_published(T):
    c = T.cars()
    assert c.shape == (398, 9) and c["horsepower"].isna().sum() == 6 and c.isna().sum().sum() == 6
    assert set(c["origin"]) == {"usa", "europe", "japan"} and c["model_year"].between(70, 82).all()
    assert round(c["weight"].corr(c["mpg"]), 2) == -0.83
    assert T.penguins().shape == (344, 8)


def test_look_inside_helpers_agree_with_scikit_learn(T):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from sklearn.neighbors import KNeighborsClassifier
    p = T.penguins().dropna()
    x, y = p["flipper_length_mm"], p["body_mass_g"]
    slope, intercept = T.line_by_hand(x, y, show=False)
    model = LinearRegression().fit(p[["flipper_length_mm"]], y)
    assert abs(slope - model.coef_[0]) < 1e-8 and abs(intercept - model.intercept_) < 1e-6
    pred = model.predict(p[["flipper_length_mm"]])
    s = T.scores_by_hand(y, pred, show=False)
    assert abs(s["MAE"] - mean_absolute_error(y, pred)) < 1e-9
    assert abs(s["RMSE"] - np.sqrt(mean_squared_error(y, pred))) < 1e-9 and abs(s["R2"] - r2_score(y, pred)) < 1e-12
    assert T.sigmoid(0) == 0.5 and np.allclose(T.sigmoid([-50, 50]), [0, 1])
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(40, 3)), columns=list("abc"))
    labels = np.where(X["a"] + X["b"] > 0, "yes", "no")
    point = X.iloc[[3]] + 0.1
    knn = KNeighborsClassifier(n_neighbors=5).fit(X, labels)
    assert T.knn_by_hand(X, labels, point, k=5, show=False) == knn.predict(point)[0]


def test_practice_results(T):
    e = T._expected()
    assert [len(e["index"][s]) for s in ("train", "val", "test")] == [238, 80, 80]
    assert e["fill"] == 95.0 and e["fill_all"] == 93.5
    assert round(e["slope"] * 1000, 2) == -7.83 and round(e["r2_simple"], 2) == 0.62
    assert round(e["mae_multi"], 2) == 2.67 and round(e["r2_multi"], 2) == 0.80
    assert e["acc_logit"] == 0.875 and e["baseline_acc"] == 0.5625
    assert e["best_k"] == 5 and e["acc_knn"] == 0.85 and e["chosen"] == "logistic" and e["test_logit"] == 0.8375


def test_checks_say_correct_and_explain_mistakes(T, capsys):
    e = T._expected()
    c = T.cars()
    train, val, test = (c.loc[sorted(e["index"][s])].fillna({"horsepower": e["fill"]}) for s in ("train", "val", "test"))
    T.check("E1", train, val=val, test=test, fill_value=95.0)
    T.check("E2", e["slope"], mae=e["mae_simple"], r2=e["r2_simple"], baseline=e["baseline_mae"])
    T.check("E3", e["mae_multi"], r2=e["r2_multi"])
    T.check("E4", e["acc_logit"], baseline=e["baseline_acc"])
    T.check("E5", 5, acc=0.85)
    T.check("E6", 0.8375, model="logistic")
    out = capsys.readouterr().out
    assert "Not yet" not in out and out.count("Correct") == 16
    T.check("E1", train, fill_value=93.5)
    T.check("E5", 1)
    T.check("E6", 0.8625, model="knn")
    out = capsys.readouterr().out
    assert "median of all 398 cars" in out and "largest of" in out and out.count("Not yet") == 4
    T.check("E2", None)
    assert "Not attempted" in capsys.readouterr().out
    with pytest.raises(KeyError):
        T.check("E9", 1)


def test_pictures_draw_without_warnings(T):
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    p = T.penguins().dropna().reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        T.plot_line(p["flipper_length_mm"], p["body_mass_g"], 49.4, -5693)
        T.plot_predictions(p["body_mass_g"], p["body_mass_g"] + 10)
        X = pd.get_dummies(p[["body_mass_g", "species"]], columns=["species"], drop_first=True, dtype=int)
        y = (p["sex"] == "male").astype(int)
        T.plot_probability(LogisticRegression(max_iter=1000).fit(X, y), p, "body_mass_g", X.columns, y,
                           group="species")
        T.plot_confusion(y, y)
        pair = p[["bill_length_mm", "bill_depth_mm"]]
        scaler = StandardScaler().fit(pair)
        knn = KNeighborsClassifier(n_neighbors=5).fit(scaler.transform(pair), p["species"])
        T.plot_regions(knn, pair, p["species"], scaler=scaler)
        knn2 = KNeighborsClassifier(n_neighbors=5).fit(pair, p["species"])
        T.plot_regions(knn2, pair, p["species"])
    plt.close("all")
