"""The paths-and-connectivity topic. The toy graphs and the StudyHub forum are generated without
downloads; the SciFact kNN test is skipped when the corpus cannot be fetched (unless AILK_REQUIRE_DATA=true)."""
import json
import os

import networkx as nx
import pandas as pd
import pytest

import ai_learning_kit as kit
from ai_learning_kit import datasets
from ai_learning_kit.core import registry


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(topic=None, problem=None, book=None, folder=None, tools=None)


def _tools():
    from ai_learning_kit.topics.paths_and_connectivity import tools
    return tools


def test_topic_contract():
    mod = registry.topic("paths-and-connectivity")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.DATASETS == ("scifact",) and mod.EXTRAS == "graphs"
    ctx = registry.tutor_context("paths-and-connectivity")
    json.dumps(ctx)
    assert "explain_unreachable" in ctx["tools"]["functions"] and "three stages" in ctx["scope"] and mod.LEVEL == "intermediate"


def test_paths_toy_three_meanings():
    T = _tools()
    G = T.add_costs(T.paths_toy())
    assert G.number_of_nodes() == 8 and G.number_of_edges() == 12
    hops = T.describe_path(G, "Amy", "Hana")
    km = T.describe_path(G, "Amy", "Hana", "km")
    strong = T.describe_path(G, "Amy", "Hana", "cost")
    wrong = T.describe_path(G, "Amy", "Hana", "msgs")
    assert hops["route"] == "Amy > Dylan > Hana" and hops["edges"] == 2
    assert km["route"] == "Amy > Ben > Clara > Gianni > Hana" and km["total"] == 10
    assert strong["route"] == "Amy > Ella > Fabio > Hana" and abs(strong["total"] - 0.2) < 1e-9
    assert wrong["route"] == hops["route"] and "WRONG" in wrong["definition"]
    assert T.all_shortest(G, "Ben", "Fabio") == [["Ben", "Clara", "Fabio"], ["Ben", "Ella", "Fabio"]]
    assert T.frequency_to_cost(4) == 0.25


def test_reply_toy_reachability():
    T = _tools()
    R = T.reply_toy()
    assert T.reachable_from(R, "Amy") == {"Ben", "Clara", "Dylan", "Ella", "Hana"}
    assert T.can_reach(R, "Amy") == {"Ben", "Clara", "Fabio"}
    ex = T.explain_unreachable(R, "Amy").set_index("user").reason.to_dict()
    assert ex == {"Kurt": "isolated", "Fabio": "never_replied", "Gianni": "never_replied",
                  "Ivan": "other_component", "Julia": "other_component"}
    assert nx.number_weakly_connected_components(R) == 3 and nx.number_strongly_connected_components(R) == 7
    # weak connectivity does not give a directed path
    assert nx.has_path(R.to_undirected(), "Amy", "Fabio") and not nx.has_path(R, "Amy", "Fabio")


def test_forum_matches_reference():
    T = _tools()
    ref = T.reference()["forum"]
    users, replies = T.forum()
    assert len(users) == ref["users"] == 300 and len(replies) == ref["rows"]
    start = T.default_start()
    assert start == ref["start"]
    summary = T.reference_summary(T.forum_graph(include_users=False), users.user, start)
    assert summary == ref["summary"]
    ex = T.explain_unreachable(T.forum_graph(), start, users.user)
    assert ex.reason.value_counts().to_dict() == ref["unreachable_counts"]
    assert set(ex.reason) == set(T.REASONS)


def test_views_and_structure():
    T = _tools()
    F = T.forum_graph()
    tab = T.structure_table(F)
    assert list(tab.view) == ["directed", "undirected", "reciprocal", "strong (>= 3 replies)"]
    und, rec = tab.iloc[1], tab.iloc[2]
    assert und.edges >= rec.edges and rec.isolated >= und.isolated
    assert und.avg_clustering > 2 * T.random_baseline_clustering(T.view(F, "undirected"), trials=2)


def test_tail_fit_lognormal_is_not_a_power_law():
    import numpy as np
    T = _tools()
    rng = np.random.default_rng(0)
    vals = np.round(rng.lognormal(2.0, 0.8, 3000)).astype(int)
    fit = T.tail_fit(vals[vals > 0], 5)
    assert fit["verdict"] == "log-normal favoured"
    pl = np.floor(5 * (1 - rng.random(3000)) ** (-1 / 1.5)).astype(int)
    assert T.tail_fit(pl, 5)["verdict"] in ("power law favoured", "cannot tell them apart")


def test_checks_pass_on_reference(capsys):
    T = _tools()
    users, _ = T.forum()
    start = T.default_start()
    s = T.reference_summary(T.forum_graph(include_users=False), users.user, start)
    s["path_definition"] = "u -> v: v replied to u; paths follow the edges, from author to replier"
    T.check("P1", s)
    F = T.forum_graph()
    rows = []
    for t in sorted(T.reachable_from(F, start))[:20]:
        h = nx.shortest_path(F, start, t)
        c = nx.shortest_path(F, start, t, weight="inv_replies")
        rows.append({"target": t, "hops_edges": len(h) - 1, "strength_edges": len(c) - 1,
                     "strength_cost": nx.path_weight(F, c, "inv_replies"), "same_route": h == c,
                     "path_definition": "hops vs 1/replies"})
    T.check("P2", pd.DataFrame(rows))
    T.check("P3", T.explain_unreachable(F, start, users.user))
    out = capsys.readouterr().out
    assert "FAIL" not in out and out.count("PASS") >= 10
    T.check("P1", None)
    assert "Not attempted" in capsys.readouterr().out


def test_quiz():
    T = _tools()
    T.quiz("Q1", ["Ella", "Ben", "Dylan"])
    T.quiz("q5", "No")


@pytest.fixture()
def scifact_available():
    try:
        return datasets.fetch("scifact", quiet=True)
    except datasets.DatasetUnavailable as e:
        if os.environ.get("AILK_REQUIRE_DATA", "").lower() == "true":
            raise
        pytest.skip(f"SciFact is not available here: {str(e).splitlines()[0]}")


def test_scifact_indegree(scifact_available):
    T = _tools()
    _, ind = T.scifact_knn_indegree(5)
    ref = T.reference()["scifact_knn5_indegree"]
    assert max(ind.values()) == ref["max"] and sum(1 for v in ind.values() if v == 0) == ref["zero"]
