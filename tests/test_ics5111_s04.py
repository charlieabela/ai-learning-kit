"""ICS5111 S04: knowledge graphs. Download-free tests of the helpers (RDF example, reasoning, Neo4j fallback,
ranking metrics, comparisons); Hetionet tests are skipped when the files cannot be fetched."""
import json
import os
import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")
pytest.importorskip("rdflib")
pytest.importorskip("owlrl")

import ai_learning_kit as kit
from ai_learning_kit.core import registry


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    kit._state.update(session=None, problem=None, book=None, folder=None, tools=None)


def _tools():
    from ai_learning_kit.units.ics5111.s04 import tools
    return tools


def test_session_contract():
    mod = registry.session("ICS5111", "S04")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.EXTRAS == "advanced"
    assert mod.DATASETS == ("hetionet_nodes", "hetionet_edges")
    ctx = registry.tutor_context("ICS5111", "S04")
    json.dumps(ctx)
    assert "filtered_ranks" in ctx["tools"]["functions"] and "Available on request" in ctx["scope"]


def test_rdf_rule_and_reasoning():
    T = _tools()
    g = T.ddi_rdf()
    rule = T.sparql(g, T.DDI_RULE_SPARQL)
    assert len(rule) == 15 and set(rule.enzyme) == {"CYP3A4", "CYP2C9", "CYP2C19", "CYP2D6"}
    assert T.drugs_typed(g) == []
    T.reason(g)
    assert len(T.drugs_typed(g)) == 17
    pairs, selfp = T.interacting_pairs(g)
    assert pairs == [("aspirin", "warfarin")] and selfp == []
    juice = T.ddi_rdf("ex:grapefruit_juice ex:inhibits ex:CYP3A4 .")
    T.reason(juice)
    assert "grapefruit_juice" in T.drugs_typed(juice)      # a domain axiom infers, it does not reject


def test_property_graph_holds_the_same_facts():
    T = _tools()
    p = T.ddi_cypher_params()
    assert len(p["drugs"]) == 17 and p["enzymes"] == ["CYP2C19", "CYP2C9", "CYP2D6", "CYP3A4"]
    rule = {(a, b) for a, e in p["inhibits"] for b, e2 in p["metabolised"] if e == e2}
    assert len(rule) == 15


def test_saved_results_without_a_database(capsys):
    T = _tools()
    db = T.connect()
    assert not db.connected and "No Neo4j credentials" in capsys.readouterr().out
    df = db.run("ddi_rule")
    assert len(df) == 15 and "saved" in db.last_source
    with pytest.raises(RuntimeError, match="No Neo4j database is connected"):
        db.run("MATCH (n) RETURN count(n)")
    for name in T.QUERIES:
        assert (T.SAVED / f"{name}.json").exists(), name


def test_credentials_file_is_read(tmp_path):
    T = _tools()
    (tmp_path / "Neo4j-abc-Created.txt").write_text("NEO4J_URI=neo4j+s://x.example\nNEO4J_USERNAME=neo4j\nNEO4J_PASSWORD=secret\n")
    assert T._credentials(tmp_path) == ("neo4j+s://x.example", "neo4j", "secret")


def test_rank_metrics_and_random_baseline():
    T = _tools()
    m = T.rank_metrics([1, 3, 2, 10])
    assert round(m["MRR"], 3) == 0.483 and m["Hits@3"] == 0.75
    assert abs(T.random_mrr(1) - 1) < 1e-12 and T.random_mrr(100) < 0.06


def test_filtered_ranks_on_a_tiny_graph():
    import torch
    T = _tools()
    tri = np.array([[0, 0, 1], [0, 0, 2], [3, 0, 1], [3, 0, 4], [5, 1, 0]])
    torch.manual_seed(0)
    m = T.make_kge("TransE", 6, 2, dim=4)
    T.train_kge(m, tri, epochs=5, batch=4)
    r = T.filtered_ranks(m, tri[:2], tri, 6)
    assert r.shape == (4,) and (r >= 1).all() and (r <= 6).all()
    # filtering: with every other answer known, the rank can only fall
    raw = T.filtered_ranks(m, tri[:2], tri[:2], 6)
    assert (r <= raw + 1e-9).all()


def test_transe_hand_step_matches_the_notes():
    T = _tools()
    h = T.transe_hand_step()
    assert round(h["d_pos"], 3) == 0.671 and round(h["d_neg"], 3) == 1.004 and round(h["loss"], 3) == 0.667


def test_comparisons():
    import networkx as nx
    T = _tools()
    g = nx.karate_club_graph()
    pr = T.nx_pagerank(g)
    gf = T.gds_formula_pagerank(g)
    c = T.compare_rankings(pr, gf, k=5)
    assert c["spearman"] > 0.95 and abs(pr.sum() - 1) < 1e-6 and gf.sum() > 10
    p1, p2 = T.nx_louvain(g, seed=1), T.nx_louvain(g, seed=2)
    cp = T.compare_partitions(g, p1, p2)
    assert 0.3 < cp["modularity_1"] < 0.5 and -0.1 <= cp["ARI"] <= 1


def test_gini():
    T = _tools()
    assert T.gini([1, 1, 1, 1]) == 0 and T.gini([0, 0, 0, 10]) > 0.7


def test_reference_results_are_packaged():
    T = _tools()
    ref = T.reference()
    assert ref["seed"] == 5111 and ref["ddi"]["rule_rows"] == 15
