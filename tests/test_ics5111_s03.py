"""ICS5111 S03: graph neural networks. Download-free tests of the helpers on the five-paper graph,
small classic graphs and a synthetic graph; Cora tests are skipped when the data cannot be fetched."""
import json
import numpy as np
import pytest

torch = pytest.importorskip("torch")   # the session's helpers need the 'advanced' extra

import ai_learning_kit as kit
from ai_learning_kit.core import registry


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(session=None, problem=None, book=None, folder=None, tools=None)


def _tools():
    from ai_learning_kit.units.ics5111.s03 import tools
    return tools


def test_session_contract():
    mod = registry.session("ICS5111", "S03")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.DATASETS == ("cora",) and mod.EXTRAS == "advanced"
    ctx = registry.tutor_context("ICS5111", "S03")
    json.dumps(ctx)
    assert "GCN" in ctx["tools"]["functions"] and "three stages" in ctx["scope"]


def test_primer_matches_the_notes():
    T = _tools()
    x, W, b, y = T.primer_tensors()
    p = torch.softmax(W @ x + b, 0)
    assert np.allclose(p.numpy(), [0.649, 0.059, 0.292], atol=1e-3)
    assert abs(float(-torch.log(p[y])) - 0.432) < 1e-3


def test_five_papers_hand_calculations():
    T = _tools()
    g, X, names = T.five_papers()
    A = T.adjacency(g, names)
    r = T.mean_aggregate(A, X, 2)
    assert np.allclose(r[1][0], [0.75, 0.5]) and np.allclose(r[1][4], [2 / 3, 2 / 3])
    Ah = T.gcn_adjacency(A)
    assert np.isclose(Ah[0, 2], 1 / np.sqrt(8)) and np.allclose(Ah, Ah.T)
    H = np.maximum(0, Ah @ X @ np.array([[1, -1], [0.5, 1]]))
    assert np.allclose(H[1], H[3])                       # B and D become identical under the GCN
    demo = T.position_layer_demo()
    a, b = demo["A,B,C,D,E"], demo["E,D,C,B,A"]
    assert not np.isclose(a.loc["C", "position layer"], b.loc["C", "position layer"])
    assert (a["neighbour sum"].sort_index() == b["neighbour sum"].sort_index()).all()


def test_sparse_and_dense_propagation_agree():
    T = _tools()
    g, X, names = T.five_papers()
    A = T.adjacency(g, names)
    P = T.propagation_matrices(A)
    x = torch.tensor(X, dtype=torch.float32)
    assert np.allclose(T.propagate(P["gcn"], x).numpy(), T.gcn_adjacency(A) @ X, atol=1e-6)
    assert np.allclose(T.propagate(P["mean"], x).numpy(), T.mean_adjacency(A, self_loops=False) @ X, atol=1e-6)


def test_weisfeiler_lehman_limit():
    import networkx as nx
    T = _tools()
    c6, tri = nx.cycle_graph(6), nx.disjoint_union(nx.complete_graph(3), nx.complete_graph(3))
    assert T.wl_same(c6, tri) and not T.wl_same(nx.path_graph(4), nx.star_graph(3))
    assert np.allclose(T.sum_gnn_vector(c6), T.sum_gnn_vector(tri))


def test_models_train_on_a_small_synthetic_graph():
    import networkx as nx
    T = _tools()
    g, x, y, split = T.synthetic_graph(0.9, n=200, seed=1)
    P = T.propagation_matrices(nx.to_scipy_sparse_array(g, nodelist=range(len(g))))
    for model, a in ((T.GCN(x.shape[1], 4), P["gcn"]), (T.SAGE(x.shape[1], 4), P["mean"]), (T.MLP(x.shape[1], 4), None)):
        r = T.train_node_classifier(model, x, a, y, split, epochs=60)
        assert 0.25 < r["test_acc"] <= 1 and len(r["history"]["loss"]) == 60


def test_reference_results_are_packaged():
    T = _tools()
    ref = T.reference()
    assert ref["seed"] == 5111 and 0.75 < ref["node_classification"]["gcn"]["mean"] < 0.85
