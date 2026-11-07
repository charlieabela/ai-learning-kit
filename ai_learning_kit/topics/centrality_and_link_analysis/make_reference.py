"""Recompute the reference results of the centrality-and-link-analysis topic (reference/results.json).

Run from a checkout: python -m ai_learning_kit.topics.centrality_and_link_analysis.make_reference
Every number quoted on the slides and in the student notes comes from this file.
"""
import json

import networkx as nx
import numpy as np
import pandas as pd

from . import tools as T


def _r(d, nd=4):
    return {str(k): (round(float(v), nd) if isinstance(v, (float, np.floating)) else
                     int(v) if isinstance(v, (np.integer,)) else v) for k, v in d.items()}


def _top(series, k=6):
    s = pd.Series(series, dtype=float)
    return [[n, round(float(s[n]), 4)] for n in sorted(s.index, key=lambda n: (-s[n], str(n)))[:k]]


def run() -> dict:
    out = {"version": T.VERSION}
    # ----- trio: two steps of matrix multiplication
    P, dang = T.transition_matrix(T.trio())
    out["trio"] = {"P": P.round(4).to_dict(orient="index"), "dangling": dang,
                   "flow_from_Amy": T.flow({"Amy": 1}, P, steps=6).drop(columns="total").to_dict(orient="index"),
                   "pagerank_unweighted_085": _r(nx.pagerank(T.trio(), weight=None, tol=1e-12, max_iter=10000)),
                   "stationary_no_teleport": _r(T.pagerank_power(T.trio(), alpha=1.0, weight=None, steps=5000)[0])}
    # ----- mention_toy
    G = T.mention_toy()
    U = G.to_undirected()
    out["toy"] = {"nodes": G.number_of_nodes(), "edges": G.number_of_edges(),
                  "mentions_total": int(sum(d["mentions"] for *_, d in G.edges(data=True))),
                  "degree_table": T.degree_table(G).to_dict(orient="index"),
                  "undirected_degree": dict(U.degree()),
                  "rank_table": T.rank_table(G).to_dict(orient="index")}
    out["toy"]["betweenness_unnormalised"] = {
        "directed": _r(nx.betweenness_centrality(G, normalized=False)),
        "undirected": _r(nx.betweenness_centrality(U, normalized=False)),
        "directed_weighted_by_mentions_WRONG": _r(nx.betweenness_centrality(G, normalized=False, weight="mentions"))}
    for v in ("Gianni", "Ben", "Hana"):
        for name, H in (("directed", G), ("undirected", U)):
            bh = T.betweenness_by_hand(H, v)
            out["toy"][f"by_hand_{v}_{name}"] = {"pairs": len(bh), "sum": round(float(bh.share.sum()), 4),
                                                 "rows": bh.head(12).to_dict(orient="records")}
    out["toy"]["closeness_directed_out"] = T.closeness_table(G, "out").to_dict(orient="index")
    out["toy"]["closeness_undirected"] = T.closeness_table(U).to_dict(orient="index")
    # PageRank
    pr_w, h_w = T.pagerank_power(G, weight="mentions")
    pr_u, _ = T.pagerank_power(G, weight=None)
    pr_leak, h_leak = T.pagerank_power(G, dangling="leak", steps=60)
    pr_1, h_1 = T.pagerank_power(G, alpha=1.0, steps=5000)
    out["toy"]["pagerank"] = {
        "weighted_085": _r(pr_w), "unweighted_085": _r(pr_u), "steps_to_converge_1e-10": int(len(h_w) - 1),
        "history_first_steps": h_w.head(6).round(4).to_dict(orient="index"),
        "leak_totals": [round(float(x), 4) for x in h_leak.total.iloc[[0, 1, 2, 3, 5, 10, 20]]],
        "leak_renormalised": _r(pr_leak / pr_leak.sum()),
        "no_teleport": _r(pr_1), "no_teleport_steps": int(len(h_1) - 1),
        "damping_sweep": {str(a): _top(T.pagerank_power(G, alpha=a)[0], 5) for a in (0.5, 0.85, 0.95)},
        "networkx_weighted": _r(nx.pagerank(G, weight="mentions", tol=1e-12, max_iter=10000))}
    P, dang = T.transition_matrix(G, weight="mentions")
    out["toy"]["transition_rows"] = {u: {v: round(float(p), 4) for v, p in P.loc[u].items() if p} for u in
                                     ("Amy", "Ben", "Gianni", "Hana")}
    out["toy"]["dangling"] = dang
    out["toy"]["hits"] = T.hits_table(G).to_dict(orient="index")
    rt = T.rank_table(G)
    pairs = [("in_degree", "pagerank"), ("in_degree", "betweenness"), ("pagerank", "betweenness"),
             ("pagerank", "authority"), ("betweenness", "betweenness_undirected"), ("in_degree", "in_strength")]
    out["toy"]["comparisons_top5"] = {f"{a}~{b}": T.compare_rankings(rt[a], rt[b], k=5) for a, b in pairs}
    # perturbations
    base = nx.pagerank(G, weight="mentions", tol=1e-12, max_iter=10000)
    perts = {"add Hana->Ben": dict(add=("Hana", "Ben", 1)), "remove Ella->Gianni": dict(remove=("Ella", "Gianni")),
             "add Mark->Ben": dict(add=("Mark", "Ben", 1)), "Amy->Ben 5 -> 1": dict(remove=("Amy", "Ben"), add=("Amy", "Ben", 1))}
    out["toy"]["perturbations"] = {}
    for name, kw in perts.items():
        H = T.perturb(G, **kw)
        sh = T.rank_shift(base, nx.pagerank(H, weight="mentions", tol=1e-12, max_iter=10000))
        btw = T.rank_shift(nx.betweenness_centrality(G), nx.betweenness_centrality(H))
        out["toy"]["perturbations"][name] = {
            "pagerank": sh[["before", "after", "rank_before", "rank_after", "moved_up"]].to_dict(orient="index"),
            "betweenness_top3_after": _top(nx.betweenness_centrality(H), 3),
            "pagerank_changed": int((sh.moved_up != 0).sum()), "btw_changed": int((btw.moved_up != 0).sum())}
    # ----- StudyHub as an attention graph
    A = T.attention_graph()
    users, _ = T.forum()
    role = users.set_index("user").role.to_dict()
    R = T.reference_ranking()
    out["forum"] = {"users": A.number_of_nodes(), "edges": A.number_of_edges(),
                    "dangling": int(sum(1 for n in A if A.out_degree(n) == 0)),
                    "isolated": int(nx.number_of_isolates(A)),
                    "top": {c: [[n, v, role[n]] for n, v in _top(R[c], 8)] for c in R.columns},
                    "bot": _r(R.loc["studyhub_announcements"].to_dict()),
                    "tutors": {u: _r(R.loc[u].to_dict()) for u in users[users.role == "tutor"].user}}
    U = A.to_undirected()
    out["forum"]["betweenness_undirected_top"] = [[n, v, role[n]] for n, v in _top(nx.betweenness_centrality(U), 6)]
    h = T.hits_table(A, weight="replies")
    out["forum"]["hits_top_authority"] = [[n, float(h.loc[n, "authority"]), role[n]] for n in h.sort_values("authority", ascending=False).index[:5]]
    out["forum"]["hits_top_hub"] = [[n, float(h.loc[n, "hub"]), role[n]] for n in h.sort_values("hub", ascending=False).index[:5]]
    pairs = [("in_degree", "in_strength"), ("in_degree", "pagerank"), ("in_strength", "pagerank"),
             ("in_degree", "betweenness"), ("pagerank", "betweenness"), ("out_degree", "betweenness")]
    out["forum"]["comparisons_top10"] = {f"{a}~{b}": T.compare_rankings(R[a], R[b], k=10) for a, b in pairs}
    clo = T.closeness_table(U)
    out["forum"]["closeness_undirected"] = {"raw_top": _top(clo.raw, 5), "harmonic_top": _top(clo.harmonic, 5),
                                            "raw_equal_1": int((clo.raw == 1).sum())}
    # perturbation used in the worked answers: the default tutor replies three times to a user nobody replied to
    tutor = T.default_start()
    cand = sorted(n for n in A if A.in_degree(n) == 0 and A.out_degree(n) > 0)
    target = cand[0]
    base = pd.Series(nx.pagerank(A, weight="replies", tol=1e-12, max_iter=10000))
    B = T.perturb(A, add=(tutor, target, 3), weight="replies")
    sh = T.rank_shift(base, nx.pagerank(B, weight="replies", tol=1e-12, max_iter=10000))
    out["forum"]["perturbation"] = {"add": [tutor, target, 3], "target_before_after": sh.loc[target, ["rank_before", "rank_after"]].astype(int).tolist(),
                                    "changed": int((sh.moved_up != 0).sum()), "max_move": int(sh.moved_up.abs().max()),
                                    "top10_after": sh.head(10).index.tolist(), "zero_in_degree_users": len(cand)}
    from ..paths_and_connectivity import tools as S2
    ex = S2.explain_unreachable(S2.forum_graph(), tutor, users.user)
    group = sorted(ex[ex.reason == "other_component"].user)
    prr = R.pagerank.rank(ascending=False, method="min")
    out["forum"]["separate_group"] = {u: {"pagerank_rank": int(prr[u]), "in_degree": int(A.in_degree(u))} for u in group}
    out["forum"]["perturbation"]["target_replies_to"] = sorted(A.successors(target))
    out["forum"]["perturbation"]["swap_10_11"] = sh.iloc[9:11][["rank_before", "rank_after"]].astype(int).to_dict(orient="index")
    Bb = A.copy(); Bb.remove_node("studyhub_announcements"); Bb.add_node("studyhub_announcements")
    sh2 = T.rank_shift(base, nx.pagerank(Bb, weight="replies", tol=1e-12, max_iter=10000))
    out["forum"]["without_bot_edges_top5"] = sh2.head(5).index.tolist()
    return out


if __name__ == "__main__":
    res = run()
    (T.ROOT / "reference").mkdir(exist_ok=True)
    (T.ROOT / "reference" / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "toy"}, indent=1, default=str)[:5000])
