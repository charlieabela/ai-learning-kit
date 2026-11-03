"""retrieval-augmented-generation: Download-free tests of the helpers on the formulary and the MedQuAD subset."""
import json
import numpy as np
import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(topic=None, problem=None, book=None, folder=None, tools=None)


def _tools():
    from ai_learning_kit.topics.retrieval_augmented_generation import tools
    return tools


def test_topic_contract():
    mod = registry.topic("retrieval-augmented-generation")
    assert mod.PROBLEM is None and mod.TASKS == {} and mod.EXTRAS == "advanced"
    ctx = registry.tutor_context("retrieval-augmented-generation")
    json.dumps(ctx)
    assert "BM25" in ctx["tools"]["functions"] and "three stages" in ctx["scope"]


def test_bm25_by_hand_matches_the_notes():
    T = _tools()
    h = T.bm25_hand()
    assert h["scores"] == [0.984, 0.431, 0.492] and h["idf"]["statin"] == 0.47


def test_stemming_joins_word_forms():
    T = _tools()
    assert {T.stem(w) for w in ("inhibits", "inhibiting", "inhibitor", "inhibition")} == {"inhibit"}
    assert T.stem("process") == T.stem("processes") == "process"


def test_formulary_questions_and_typed_graph():
    T = _tools()
    q = T.two_hop_questions()
    assert len(q) == 13 and q.iloc[0].gold == ["simvastatin", "clarithromycin", "fluconazole", "itraconazole"]
    tri = T.kg_retrieve(T.typed_graph(), "simvastatin")
    assert ("simvastatin", "METABOLISED_BY", "CYP3A4") in tri and ("fluconazole", "INHIBITS", "CYP3A4") in tri


def test_entity_graph_shows_the_negation_error():
    T = _tools()
    G, ment = T.entity_graph(dict(zip(T.formulary().id, T.formulary().text)))
    assert G.has_edge("azithromycin", "CYP3A4")          # co-occurrence ignores 'does not inhibit'
    ranked, pr = T.ppr_rank(G, ment, ["simvastatin"])
    assert ranked[0] == "simvastatin" and "clarithromycin" in ranked[:3]


def test_fusion_and_metrics():
    T = _tools()
    assert T.rrf([["a", "b", "c"], ["b", "a", "d"]])[:2] in (["a", "b"], ["b", "a"])
    assert T.mrr(["x", "a"], ["a"]) == 0.5 and T.context_recall(["a", "b"], ["a", "c"], 2) == 0.5
    assert abs(T.ndcg(["a", "x"], ["a"], 10) - 1) < 1e-12


def test_extractive_answer_is_grounded_and_a_bad_one_is_not():
    T = _tools()
    ev = ["Clarithromycin is a macrolide. It is a strong inhibitor of CYP3A4.",
          "Itraconazole is an azole antifungal and a strong inhibitor of CYP3A4."]
    a = T.extractive_answer("Which medicines inhibit CYP3A4?", ev)
    _, m = T.grounding(a, ev)
    assert m["faithfulness"] == 1.0 and m["citation_precision"] == 1.0
    _, bad = T.grounding("Simvastatin is safe with every antibiotic [1].", ev)
    assert bad["faithfulness"] == 0.0


def test_chunking():
    T = _tools()
    text = " ".join(f"w{i}" for i in range(250))
    ch = T.chunk(text, size=100, overlap=20)
    assert len(ch) == 3 and ch[1].split()[0] == "w80"
    assert T.chunk("One. Two three. Four.", size=3, by="sentences") == ["One. Two three.", "Four."]


def test_medquad_subset_and_lsa():
    T = _tools()
    mq = T.medquad()
    assert len(mq) == 2795 and mq.hetionet_disease.notna().sum() == 795
    qs = T.medquad_questions(mq)
    assert len(qs) == 526 and all(len(g) >= 1 for g in qs.gold)
    lsa = T.LSA(T.formulary().text, dim=5)
    s = lsa.scores("statin cholesterol")
    assert s.shape == (20,) and np.isfinite(s).all()


def test_reference_results_are_packaged():
    T = _tools()
    ref = T.reference()
    assert ref["seed"] == 5111 and ref["formulary"]["questions"] == 13
