"""ICS5111 S02: a session without graded tasks, with helpers, Cora from LINQS and reference results."""
import json
import os
import numpy as np
import pytest

import ai_learning_kit as kit
from ai_learning_kit import datasets
from ai_learning_kit.core import registry


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(session=None, problem=None, book=None, folder=None, tools=None)


@pytest.fixture()
def cora_available():
    """Cora from wherever the kit finds it; skipped when it cannot, unless AILK_REQUIRE_DATA=true (CI with a
    course copy configured), when the test fails instead."""
    try:
        return datasets.fetch("cora", quiet=True)
    except datasets.DatasetUnavailable as e:
        if os.environ.get("AILK_REQUIRE_DATA", "").lower() == "true":
            raise
        pytest.skip(f"Cora is not available here: {str(e).splitlines()[0]}")


def test_session_contract_without_tasks():
    mod = registry.session("ICS5111", "S02")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.SAVES_WORK is False
    assert mod.DATASETS == ("cora",) and mod.EXTRAS == "graphs"
    assert set(mod.NOTEBOOKS) == {"prepare", "explore", "practice"}
    ctx = registry.tutor_context("ICS5111", "S02")
    json.dumps(ctx)
    assert ctx["problem"] is None and ctx["tasks"] == {} and "load_cora" in ctx["tools"]["functions"]
    assert ctx["datasets"][0]["owner"].startswith("LINQS") and "SCOPE OF THIS SESSION" in ctx["scope"]


def test_start_needs_no_name_and_gives_the_helpers(cora_available, capsys):
    kit.start("ICS5111", "S02")
    out = capsys.readouterr().out
    assert "Graph embeddings" in out and "Cora citation network (LINQS), checked" in out and "T = kit.tools()" in out
    T = kit.tools()
    assert T.SEED == 5111 and T.VERSION == "2026-10-27"
    with pytest.raises(RuntimeError, match="no graded tasks"):
        kit.answer("T1.1", value=1)
    with pytest.raises(RuntimeError, match="no case problem"):
        kit.extract.policy_rules()


def test_tools_before_start_says_what_to_do():
    with pytest.raises(RuntimeError, match="kit.start"):
        kit.tools()


def test_cora_and_the_fixed_split(cora_available):
    kit.start("ICS5111", "S02")
    G, X, y, ids, classes, split = kit.tools().load_cora()
    assert (G.number_of_nodes(), G.number_of_edges(), X.shape) == (2708, 5278, (2708, 1433))
    assert len(classes) == 7 and [len(split[k]) for k in ("train", "val", "test")] == [541, 541, 1626]


def test_reference_embedding_is_reproduced(cora_available):
    """The walkthrough's check: training with the slide settings gives the frozen reference vectors."""
    kit.start("ICS5111", "S02")
    T = kit.tools()
    G, *_ = T.load_cora()
    ref = json.loads((T.ROOT / "reference/results.json").read_text())
    config = dict(n_walks=3, length=16, window=2, dim=16, negatives=3, epochs=2, lr=.03, p=1., q=1.)
    _, Z, training = T.embed(G, **config)
    assert training["pairs"] == ref["training"]["pairs"]
    assert np.allclose(training["loss_per_positive_pair"], ref["training"]["loss_per_positive_pair"])
    assert np.allclose(Z, np.load(T.ROOT / "reference/cora-s02-baseline.npz")["embeddings"])


def test_small_graphs_come_from_networkx():
    from ai_learning_kit.units.ics5111.s02 import tools as T
    g, club = T.karate()
    assert g.number_of_nodes() == 34 and set(club) == {0, 1}
    assert T.lesmis().number_of_nodes() == 77
