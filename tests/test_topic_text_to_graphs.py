"""The text-to-graphs topic. Download-free tests on the miniature: every number below is printed by the topic's
walkthrough or practice notebook. SciFact tests run when the dataset is available."""
import json
import math

import networkx as nx
import numpy as np
import pandas as pd
import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry
from ai_learning_kit.topics.text_to_graphs import tools as T


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(topic=None, problem=None, book=None, folder=None, tools=None)


def _terms_graph():
    return T.project(T.bipartite_graph(T.miniature_tokens()), "term")


def test_topic_contract():
    mod = registry.topic("text-to-graphs")
    assert mod.TASKS == {} and mod.EXTRAS == "graphs" and mod.DATASETS == ("scifact",) and mod.LEVEL == "intermediate"
    ctx = registry.tutor_context("text-to-graphs")
    json.dumps(ctx)
    assert "dijkstra" in ctx["tools"]["functions"] and "In focus" in ctx["scope"]


def test_bipartite_graph_and_projections():
    B = T.bipartite_graph(T.miniature_tokens())
    docs = [n for n, s in B.nodes(data="side") if s == "doc"]
    assert (len(docs), B.number_of_nodes() - len(docs), B.number_of_edges()) == (6, 14, 24)
    terms, docs_graph = T.project(B, "term"), T.project(B, "doc")
    assert (terms.number_of_nodes(), terms.number_of_edges()) == (14, 30)
    assert sorted(docs_graph.edges(data="weight")) == [("D1", "D2", 2), ("D2", "D3", 2), ("D2", "D5", 1),
                                                       ("D3", "D5", 1), ("D4", "D5", 2), ("D4", "D6", 3), ("D5", "D6", 1)]
    same = T.cooccurrence_graph(T.miniature_tokens(), terms=sorted(terms.nodes()))
    assert {frozenset(e) for e in terms.edges()} == {frozenset(e) for e in same.edges()}


def test_bfs_and_dfs_orders_on_the_miniature():
    g = _terms_graph()
    assert T.bfs_order(g, "vitamin") == ["vitamin", "deficiency", "infection", "risk", "supplement", "trial", "mortality",
                                         "placebo", "cell", "therapy", "tumour", "apoptosis", "mutation", "p53"]
    assert T.dfs_order(g, "vitamin") == ["vitamin", "deficiency", "infection", "risk", "supplement", "mortality",
                                         "placebo", "trial", "cell", "apoptosis", "mutation", "p53", "tumour", "therapy"]
    hops = T.bfs_hops(g, "vitamin")
    assert sorted(t for t, h in hops.items() if h == 3) == ["apoptosis", "mutation", "p53"]


def test_count_threshold_breaks_the_concept_graph_apart():
    tok = T.miniature_tokens()
    g = T.cooccurrence_graph(tok, terms=sorted(_terms_graph().nodes()), min_weight=2)
    assert T.summary(g) == {"nodes": 14, "edges": 6, "density": 0.065934, "mean_degree": 0.857, "isolated": 6,
                            "components": 9, "largest_component_share": 0.2857}
    assert [sorted(c) for c in T.weak_components(g)][:3] == [["cell", "mutation", "p53", "tumour"],
                                                            ["infection", "vitamin"], ["supplement", "trial"]]


def test_tfidf_similarities_and_knn_components():
    X, ids, _ = T.tfidf(T.miniature_tokens(), min_df=1)
    S = T.similarity_matrix(X).round(3)
    assert (S[0, 1], S[1, 2], S[1, 4], S[2, 4], S[3, 4], S[3, 5], S[4, 5]) == (0.465, 0.411, 0.187, 0.166, 0.449,
                                                                              0.687, 0.176)
    k1 = T.knn_graph(X, ids, k=1)
    assert sorted(k1.edges()) == [("D1", "D2"), ("D2", "D1"), ("D3", "D2"), ("D4", "D6"), ("D5", "D4"), ("D6", "D4")]
    assert [sorted(c) for c in T.weak_components(k1)] == [["D1", "D2", "D3"], ["D4", "D5", "D6"]]
    assert [sorted(c) for c in T.strong_components(k1)] == [["D1", "D2"], ["D4", "D6"], ["D3"], ["D5"]]
    assert T.summary(T.knn_graph(X, ids, k=2)) == {"nodes": 6, "edges": 11, "density": 0.366667, "mean_degree": 1.833,
                                                   "isolated": 0, "weak_components": 1, "strong_components": 1,
                                                   "largest_weak_share": 1.0, "largest_strong_share": 1.0}


def test_shortest_paths_and_distance_conversions():
    gs = T.miniature_similarity_graph()
    assert sorted(gs.edges(data="similarity")) == [("D1", "D2", 0.465), ("D2", "D3", 0.411), ("D2", "D5", 0.187),
                                                   ("D3", "D5", 0.166), ("D4", "D5", 0.449), ("D4", "D6", 0.687),
                                                   ("D5", "D6", 0.176)]
    raw = gs.copy()
    for _, _, d in raw.edges(data=True):
        d["length"] = d["similarity"]
    dist, prev = T.dijkstra(raw, "D3", weight="length")
    assert T.path_to(prev, "D2") == ["D3", "D5", "D2"] and round(dist["D2"], 3) == 0.353
    expected = {"1-s": (["D1", "D2", "D5", "D6"], 2.172), "-log": (["D1", "D2", "D5", "D4", "D6"], 3.619),
                "1/s": (["D1", "D2", "D5", "D4", "D6"], 11.181)}
    for method, (path, total) in expected.items():
        dist, prev = T.dijkstra(T.add_distances(gs.copy(), method), "D1")
        assert T.path_to(prev, "D6") == path and round(dist["D6"], 3) == total
    with pytest.raises(ValueError):
        g = nx.Graph([("a", "b")]); g["a"]["b"]["distance"] = -1
        T.dijkstra(g, "a")


def test_lift_and_concepts_on_a_toy_corpus():
    tok = {"1": ["a", "b"], "2": ["a", "b"], "3": ["a", "c"], "4": ["c"]}
    assert T.lift(tok, "a", "b") == {"N": 4, "df(a)": 3, "df(b)": 2, "observed": 2, "expected": 1.5, "lift": 1.333}
    assert T.concepts(tok, 2) == ["a", "b"]          # ties in alphabetical order
    g = T.cooccurrence_graph(tok, top_n=3, min_lift=1.2)
    assert sorted(g.edges()) == [("a", "b")]


def test_tokenize_variants():
    text = "The anti-inflammatory effects of Vitamin D were studied in 2 trials of IL-6 and p53 in 2019."
    assert T.tokenize(text) == ["anti-inflammatory", "effects", "vitamin", "studied", "trials", "il-6", "p53"]
    assert "the" in T.tokenize(text, "keep_stopwords")
    assert {"anti", "inflammatory"} <= set(T.tokenize(text, "split_hyphens"))
    pytest.importorskip("nltk")
    assert "studi" in T.tokenize(text, "porter")


def test_tokens_and_tfidf_from_files(tmp_path):
    (tmp_path / "t.json").write_text(json.dumps({"1": ["a", "b"], "2": "b c"}))
    assert T.tokens_from_file(tmp_path / "t.json") == {"1": ["a", "b"], "2": ["b", "c"]}
    (tmp_path / "t.jsonl").write_text('{"_id": 1, "tokens": ["a"]}\n{"doc_id": "2", "tokens": "b c"}\n')
    assert T.tokens_from_file(tmp_path / "t.jsonl") == {"1": ["a"], "2": ["b", "c"]}
    pd.DataFrame({"doc_id": ["1"], "tokens": ["a b"]}).to_csv(tmp_path / "t.tsv", sep="\t", index=False)
    assert T.tokens_from_file(tmp_path / "t.tsv") == {"1": ["a", "b"]}
    (tmp_path / "w.json").write_text(json.dumps({"1": {"a": 3, "b": 4}, "2": {"b": 1}}))
    X, ids, vocab = T.tfidf_from_file(tmp_path / "w.json")
    assert ids == ["1", "2"] and vocab == ["a", "b"]
    assert np.allclose(T.similarity_matrix(X), [[1, 0.8], [0.8, 1]])


def test_mutual_symmetric_and_threshold_graphs():
    X, ids, _ = T.tfidf(T.miniature_tokens(), min_df=1)
    assert sorted(T.knn_graph(X, ids, k=1, mutual=True).edges()) == [("D1", "D2"), ("D4", "D6")]
    assert T.knn_graph(X, ids, k=1, symmetric=True).number_of_edges() == 4
    th = T.threshold_graph(X, ids, min_sim=0.4)
    assert sorted(tuple(sorted(e)) for e in th.edges()) == [("D1", "D2"), ("D2", "D3"), ("D4", "D5"), ("D4", "D6")]


def _answers():
    from collections import deque
    import heapq

    def bfs(graph, source):
        hops, queue = {source: 0}, deque([source])
        while queue:
            u = queue.popleft()
            for v in graph.neighbors(u):
                if v not in hops:
                    hops[v] = hops[u] + 1
                    queue.append(v)
        return hops

    def dfs(graph, source):
        visited, order, stack = set(), [], [source]
        while stack:
            u = stack.pop()
            if u in visited:
                continue
            visited.add(u); order.append(u)
            for v in sorted(graph.neighbors(u), reverse=True):
                if v not in visited:
                    stack.append(v)
        return order

    def weak_components(graph):
        g = graph.to_undirected() if graph.is_directed() else graph
        assigned, comps = set(), []
        for s in g.nodes():
            if s not in assigned:
                comp = set(bfs(g, s)); assigned |= comp; comps.append(comp)
        return comps

    def strong_components(graph):
        rev = graph.reverse(copy=False)
        assigned, comps = set(), []
        for s in graph.nodes():
            if s not in assigned:
                comp = set(bfs(graph, s)) & set(bfs(rev, s)); assigned |= comp; comps.append(comp)
        return comps

    def dijkstra(graph, source, weight="distance"):
        dist, prev, final = {source: 0.0}, {source: None}, set()
        heap = [(0.0, str(source), source)]
        while heap:
            d, _, u = heapq.heappop(heap)
            if u in final:
                continue
            final.add(u)
            for v in graph.neighbors(u):
                nd = d + graph[u][v][weight]
                if v not in dist or nd < dist[v]:
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, str(v), v))
        return dist, prev

    return bfs, dfs, weak_components, strong_components, dijkstra


def test_checks_pass_the_worked_answers(capsys):
    bfs, dfs, weak, strong, dij = _answers()
    T.check("E1", bfs); T.check("E2", dfs); T.check("E3", weak, strong=strong); T.check("E4", dij)
    out = capsys.readouterr().out.splitlines()
    assert out and all(line.startswith("PASS") for line in out), out
    assert "PASS  miniature term graph: hop counts from 'vitamin' are right (14 nodes reached)." in out
    assert "PASS  directed graph: 3 strong component(s)." in out
    assert "PASS  miniature kNN, k = 1: 4 strong component(s)." in out
    assert len(out) == 4 + 4 + 6 + 2


def test_checks_say_not_attempted_and_catch_mistakes(capsys):
    def bfs(graph, source):
        return None
    T.check("E1", bfs)
    assert capsys.readouterr().out.strip() == "Not attempted yet: bfs returns None."
    T.check("E5", None, kind="knn")
    assert capsys.readouterr().out.startswith("Not attempted yet: complete the cell above")

    def wrong_bfs(graph, source):            # counts a node when it is taken from the queue, not when found
        return {n: 0 for n in nx.descendants(graph, source) | {source}}
    T.check("E1", wrong_bfs)
    assert "FAIL" in capsys.readouterr().out

    def bad_dfs(graph, source):              # a breadth-first order is not a valid depth-first order
        return T.bfs_order(graph, source)
    T.check("E2", bad_dfs)
    assert "FAIL  miniature term graph" in capsys.readouterr().out


def test_e5_checks_the_sweep_tables(capsys):
    tok = T.miniature_tokens()
    rows = [{"min_lift": L, **T.summary(T.cooccurrence_graph(tok, terms=sorted(_terms_graph()), min_lift=L))}
            for L in (1, 1.5, 2, 3, 5)]
    T.check("E5", pd.DataFrame(rows), kind="cooccurrence")
    X, ids, _ = T.tfidf(tok, min_df=1)
    rows = []
    for k in (1, 2, 3, 4, 5):
        s = T.summary(T.knn_graph(X, ids, k=k))
        rows.append({c: s[c] for c in ("edges", "density", "weak_components", "strong_components")} | {"k": k})
    T.check("E5", pd.DataFrame(rows), kind="knn")
    out = capsys.readouterr().out.splitlines()
    assert len(out) == 8 and all(l.startswith("PASS") for l in out), out
    T.check("E5", pd.DataFrame({"min_lift": [1, 2], "nodes": [5, 5], "edges": [3, 4], "density": [0.3, 0.4],
                                "components": [1, 1]}), kind="cooccurrence")
    out = capsys.readouterr().out
    assert "FAIL  only 2 setting(s)" in out and "FAIL  edges should not grow" in out


def _scifact_or_skip():
    from ai_learning_kit import datasets
    try:
        datasets.source("scifact")
    except KeyError:
        pytest.skip("SciFact is not registered yet")
    if not datasets.available("scifact"):
        try:
            datasets.fetch("scifact", quiet=True)
        except Exception as e:
            pytest.skip(f"SciFact cannot be fetched here: {e}")


def test_scifact_reproduces_the_numbers_on_the_slides():
    _scifact_or_skip()
    tok = T.corpus_tokens()
    assert len(tok) == 5183 and len(T.document_frequency(tok)) == 44146
    assert T.concepts(tok, 3) == ["results", "cells", "cell"]
    assert T.lift(tok, "confidence", "interval")["lift"] == 16.397
    g = T.cooccurrence_graph(tok, top_n=300, min_weight=5, min_lift=3)
    assert T.summary(g) == {"nodes": 300, "edges": 449, "density": 0.010011, "mean_degree": 2.993, "isolated": 142,
                            "components": 156, "largest_component_share": 0.32}
    X, ids, vocab = T.tfidf(tok, min_df=2)
    assert X.shape == (5183, 19292) and X.nnz == 416123
    s = T.summary(T.knn_graph(X, ids, k=5))
    assert (s["edges"], s["weak_components"], s["strong_components"], s["largest_strong_share"]) == (25915, 1, 326, 0.9093)


def test_setup_checks_and_self_check(tmp_path, monkeypatch, capsys):
    from ai_learning_kit.core import environment
    monkeypatch.chdir(tmp_path)
    for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD"):
        monkeypatch.delenv(k, raising=False)
    results = T.check_environment()
    assert [r[1] for r in results] == ["Python", "NetworkX", "scikit-learn", "SciPy", "NLTK", "Neo4j driver"]
    pytest.importorskip("neo4j")
    assert T.check_neo4j()[-1][0] == "SKIP"                         # no credentials yet
    assert T.check_neo4j("missing.txt")[-1][0] == "FAIL"
    (tmp_path / "creds.txt").write_text("NEO4J_URI=neo4j+s://x.example\nNEO4J_USERNAME=neo4j\nNEO4J_PASSWORD=s3cret\n")
    assert environment.neo4j_credentials("creds.txt") == ("neo4j+s://x.example", "neo4j", "s3cret")
    capsys.readouterr()
    T.self_check("Q1")
    assert "Not attempted" in capsys.readouterr().out
    for q, a in (("Q1", 10), ("Q2", 10), ("Q3", "E"), ("Q4", "b"), ("Q5", "c")):
        T.self_check(q, a)
    out = capsys.readouterr().out
    assert out.count("PASS") == 5 and "s3cret" not in out
    T.self_check("Q3", "D")
    assert "Not yet" in capsys.readouterr().out


def test_draw_bipartite_labels_both_sides():
    import matplotlib
    matplotlib.use("Agg")
    ax = T.draw_bipartite(T.bipartite_graph({"d1": ["vitamin", "trial"], "d2": ["trial"]}))
    assert sorted(t.get_text() for t in ax.texts) == ["d1", "d2", "trial", "vitamin"]
