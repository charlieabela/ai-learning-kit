"""Centrality and link analysis: readable helpers.

Used in the topic's notebooks as T = kit.tools() after kit.start("centrality-and-link-analysis").

What is here, in the order the notebooks use it:
- small graphs that can be checked by hand:
  trio()          : three users, for a two-step matrix multiplication by hand;
  mention_toy()   : thirteen users of a course forum (invented), directed and weighted: u -> v means u mentioned v,
                    with the number of mentions on the edge; one user is never the one who mentions (a dangling
                    node), one user mentions another five times, the user with the most links is not the bridge
                    between the two groups, and two users mention only each other;
- StudyHub (the fictional forum of paths-and-connectivity, 300 users) as an attention graph: attention_graph()
  has an edge u -> v when u replied to v's posts, weighted by the number of replies (the reverse of the flow graph
  of paths-and-connectivity);
- degree and strength: degree_table();
- betweenness: betweenness_by_hand() (the shortest paths through one node, pair by pair);
- closeness on disconnected data: closeness_table() (raw, Wasserman-Faust scaled, harmonic);
- PageRank as probability flow: transition_matrix(), flow(), pagerank_power() (with or without teleportation,
  with dangling nodes handled or leaking);
- HITS: hits_table();
- comparing rankings: rank_table(), compare_rankings(), perturb(), rank_shift();
- the setup self-check quiz() and the practice checks check("P1") ... check("P4").

Every ranking is reported with its definition: what an edge means, which direction it runs, whether its weight
is used, and which view of the graph was measured.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from ..paths_and_connectivity import tools as S2

ROOT = Path(__file__).parent              # ROOT / "reference" holds the frozen reference results
SEED = 2205
VERSION = "2026-11-06"
NAVY, BLUE, ORANGE, GREEN, GREY, PURPLE = "#1F3864", "#4472C4", "#ED7D31", "#70AD47", "#9CA9B6", "#7E57C2"
draw = S2.draw                              # the drawing helper of paths-and-connectivity
forum = S2.forum
default_start = S2.default_start


# =========================================================================== small graphs
def trio() -> nx.DiGraph:
    """Three users: Amy mentions Ben and Clara, Ben mentions Amy, Clara mentions Ben (one mention each)."""
    G = nx.DiGraph()
    G.add_weighted_edges_from([("Amy", "Ben", 1), ("Amy", "Clara", 1), ("Ben", "Amy", 1), ("Clara", "Ben", 1)],
                              weight="mentions")
    return G


TRIO_POS = {"Amy": (0, 0), "Ben": (2, 0.9), "Clara": (2, -0.9)}

_MENTION_EDGES = [  # (u, v, mentions): u mentioned v in posts, that many times
    ("Amy", "Ben", 5), ("Ben", "Amy", 1), ("Clara", "Ben", 1), ("Ben", "Clara", 1), ("Dylan", "Ben", 2),
    ("Ella", "Ben", 1), ("Fabio", "Ben", 1), ("Ben", "Fabio", 1), ("Amy", "Clara", 1), ("Clara", "Dylan", 1),
    ("Dylan", "Ella", 1), ("Ella", "Fabio", 1), ("Fabio", "Amy", 1),
    ("Ella", "Gianni", 1), ("Gianni", "Ella", 1), ("Fabio", "Gianni", 1),
    ("Gianni", "Hana", 3), ("Gianni", "Ivan", 1), ("Ivan", "Hana", 1), ("Ivan", "Julia", 1), ("Julia", "Hana", 2),
    ("Julia", "Kurt", 1), ("Kurt", "Hana", 1),
    ("Lea", "Mark", 2), ("Mark", "Lea", 1),
]


def mention_toy() -> nx.DiGraph:
    """Thirteen forum users (invented). Edge u -> v: u mentioned v; attribute mentions = how many times."""
    G = nx.DiGraph()
    G.add_weighted_edges_from(_MENTION_EDGES, weight="mentions")
    return G


MENTION_POS = {"Amy": (0.0, 1.6), "Ben": (1.3, 0.6), "Clara": (0.0, -0.4), "Dylan": (1.0, -1.6), "Ella": (2.6, -0.9),
               "Fabio": (2.8, 1.5), "Gianni": (4.4, 0.3), "Hana": (6.4, 0.3), "Ivan": (5.4, 1.8), "Julia": (7.6, 1.7),
               "Kurt": (8.0, -0.7), "Lea": (5.4, -1.7), "Mark": (6.9, -1.7)}


def draw_mention_toy(G=None, scores=None, ax=None, title=None, labels: bool = True, highlight=None, colors=None):
    """Draw mention_toy. labels: write the number of mentions on each edge. scores: a dict or Series; the score
    is written under each name, and the colour shows its third of the range (orange: top, blue: middle, grey:
    bottom). highlight: a route (list of nodes) in orange."""
    G = mention_toy() if G is None else G
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 5.6))
    pos = {n: (1.25 * x, y) for n, (x, y) in MENTION_POS.items() if n in G}
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["label"] = str(d.get("mentions", "")) if labels else ""
    if scores is not None:
        s = pd.Series(scores, dtype=float)
        rel = (s - s.min()) / ((s.max() - s.min()) or 1)
        H = nx.relabel_nodes(H, {n: f"{n}\n{s[n]:.3g}" for n in H if n in s.index})
        pos = {f"{n}\n{s[n]:.3g}" if n in s.index else n: p for n, p in pos.items()}
        if colors is None:
            colors = {f"{n}\n{s[n]:.3g}": (ORANGE if rel[n] > 0.66 else BLUE if rel[n] > 0.33 else GREY) for n in s.index}
        if highlight:
            highlight = [f"{n}\n{s[n]:.3g}" for n in highlight]
    return draw(H, pos=pos, ax=ax, edge_labels="label" if labels else None, colors=colors, highlight=highlight,
                title=title, boxes=True, font_size=11)


# =========================================================================== StudyHub as an attention graph
def attention_graph(seed: int = SEED, include_users: bool = True) -> nx.DiGraph:
    """StudyHub as an attention graph: u -> v when u replied to v's posts; attribute replies = how many times.
    It is the flow graph of paths-and-connectivity (author -> replier) with every edge reversed; self-loops removed,
    and every registered user is a node (include_users=True)."""
    F = S2.forum_graph(seed, include_users=include_users)
    A = nx.DiGraph()
    A.add_nodes_from(F.nodes(data=True))
    for u, v, d in F.edges(data=True):
        A.add_edge(v, u, replies=d["replies"])
    return A


# =========================================================================== degree and strength
def degree_table(G, weight: str = "mentions") -> pd.DataFrame:
    """In-degree and out-degree (how many different users), and in- and out-strength (how many mentions/replies)."""
    t = pd.DataFrame({"in_degree": dict(G.in_degree()), "out_degree": dict(G.out_degree()),
                      "in_strength": dict(G.in_degree(weight=weight)), "out_strength": dict(G.out_degree(weight=weight))})
    return t.sort_values(["in_degree", "in_strength"], ascending=False)


# =========================================================================== betweenness, by hand
def betweenness_by_hand(G, v, weight=None) -> pd.DataFrame:
    """Betweenness written out: for every ordered pair (s, t) with s, t != v and a path from s to t, the number of
    shortest paths from s to t, how many pass through v, and the share. Betweenness (unnormalised) is the sum of
    the shares. For an undirected graph each unordered pair is listed once."""
    rows = []
    nodes = sorted(G, key=str)
    for i, s in enumerate(nodes):
        for t in nodes:
            if s == t or v in (s, t):
                continue
            if not G.is_directed() and t < s:
                continue
            if not nx.has_path(G, s, t):
                continue
            paths = list(nx.all_shortest_paths(G, s, t, weight=weight))
            through = sum(1 for p in paths if v in p[1:-1])
            if through:
                rows.append({"source": s, "target": t, "shortest_paths": len(paths), "through": through,
                             "share": round(through / len(paths), 4),
                             "example": " > ".join(next(p for p in paths if v in p[1:-1]))})
    return pd.DataFrame(rows, columns=["source", "target", "shortest_paths", "through", "share", "example"])


# =========================================================================== closeness on disconnected data
def closeness_table(G, direction: str = "out") -> pd.DataFrame:
    """Three versions of closeness for every node. For a directed graph, direction="out" measures distances from
    the node to the others (how quickly its mentions reach them), "in" from the others to the node.
    reachable: how many others are reachable; raw: reachable / (sum of their distances), the classic formula
    applied inside what is reachable; scaled: raw x reachable / (n - 1) (Wasserman and Faust); harmonic: the mean
    of 1 / distance over all n - 1 others, with 1 / infinity = 0."""
    H = G.reverse(copy=True) if (G.is_directed() and direction == "in") else G
    n = H.number_of_nodes()
    rows = {}
    for u in H:
        d = nx.single_source_shortest_path_length(H, u)
        d.pop(u)
        r, s = len(d), sum(d.values())
        raw = r / s if s else 0.0
        rows[u] = {"reachable": r, "sum_distance": s, "raw": round(raw, 4), "scaled": round(raw * r / (n - 1), 4),
                   "harmonic": round(sum(1 / x for x in d.values()) / (n - 1), 4)}
    return pd.DataFrame(rows).T.astype({"reachable": int, "sum_distance": int}).sort_values("harmonic", ascending=False)


# =========================================================================== PageRank as probability flow
def transition_matrix(G, weight=None, nodelist=None):
    """Row-stochastic transition matrix of a random walk that follows out-edges: P[u, v] = w(u, v) / out-strength(u).
    A node with no out-edges (dangling) has a row of zeros. Returns (P as a DataFrame, list of dangling nodes)."""
    nodes = list(nodelist) if nodelist is not None else sorted(G, key=str)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=weight)
    out = A.sum(axis=1)
    P = np.divide(A, out[:, None], out=np.zeros_like(A), where=out[:, None] > 0)
    return pd.DataFrame(P, index=nodes, columns=nodes), [n for n, o in zip(nodes, out) if o == 0]


def flow(x0, P, steps: int = 1) -> pd.DataFrame:
    """Repeated matrix multiplication x_{t+1} = x_t P, starting from x0 (a dict or Series over P's nodes).
    Returns one row per step (step 0 is x0) and a 'total' column: the probability still in the walk."""
    P = P if isinstance(P, pd.DataFrame) else pd.DataFrame(P)
    x = pd.Series(x0, index=P.index, dtype=float).fillna(0.0).values
    rows = [x]
    for _ in range(steps):
        x = x @ P.values
        rows.append(x)
    out = pd.DataFrame(rows, columns=P.index).round(4)
    out["total"] = out.sum(axis=1).round(4)
    out.index.name = "step"
    return out


def pagerank_power(G, alpha: float = 0.85, weight="mentions", dangling: str = "uniform", steps: int = 1000,
                   tol: float = 1e-10, start=None):
    """PageRank by power iteration, written out. Each step: with probability alpha the walker follows an out-edge
    (chosen in proportion to its weight), with probability 1 - alpha it teleports to a node chosen uniformly.
    dangling="uniform": a walker on a node with no out-edges teleports (networkx's default); dangling="leak": its
    probability is lost, so the total falls below 1. Returns (scores as a Series, history as a DataFrame)."""
    P, dang = transition_matrix(G, weight=weight)
    nodes = list(P.index)
    n = len(nodes)
    x = np.full(n, 1 / n) if start is None else pd.Series(start, index=nodes, dtype=float).fillna(0).values
    is_d = np.array([u in dang for u in nodes])
    hist = [x]
    for _ in range(steps):
        new = alpha * (x @ P.values) + (1 - alpha) * x.sum() / n
        if dangling == "uniform":
            new = new + alpha * x[is_d].sum() / n
        done = np.abs(new - x).sum() < tol
        x = new
        hist.append(x)
        if done:
            break
    h = pd.DataFrame(hist, columns=nodes)
    h["total"] = h.sum(axis=1)
    h.index.name = "step"
    return pd.Series(x, index=nodes).sort_values(ascending=False), h


# =========================================================================== HITS
def hits_table(G, weight=None) -> pd.DataFrame:
    """HITS hub and authority scores (each sums to 1). A good hub points to good authorities; a good authority is
    pointed to by good hubs. With weight, the iteration uses the weighted adjacency matrix."""
    nodes = sorted(G, key=str)
    A = nx.to_numpy_array(G, nodelist=nodes, weight=weight)
    h = np.ones(len(nodes))
    for _ in range(1000):
        a = A.T @ h
        a = a / a.sum() if a.sum() else a
        h_new = A @ a
        h_new = h_new / h_new.sum() if h_new.sum() else h_new
        if np.abs(h_new - h).sum() < 1e-12:
            h = h_new
            break
        h = h_new
    return pd.DataFrame({"hub": h, "authority": a}, index=nodes).round(4).sort_values("authority", ascending=False)


# =========================================================================== comparing rankings
def rank_table(G, weight: str = "mentions", alpha: float = 0.85) -> pd.DataFrame:
    """Several definitions of importance side by side (directed graph G; weight names the edge weight):
    in_degree, in_strength, out_degree, betweenness (directed, unweighted, normalised), betweenness_undirected,
    harmonic closeness (directed, outgoing distances), pagerank (weighted, alpha), hub and authority (HITS)."""
    U = G.to_undirected()
    pr = nx.pagerank(G, alpha=alpha, weight=weight)
    hits = hits_table(G)
    clo = closeness_table(G, "out")
    t = pd.DataFrame({"in_degree": dict(G.in_degree()), "in_strength": dict(G.in_degree(weight=weight)),
                      "out_degree": dict(G.out_degree()),
                      "betweenness": nx.betweenness_centrality(G, normalized=True),
                      "betweenness_undirected": nx.betweenness_centrality(U, normalized=True),
                      "harmonic": clo.harmonic, "pagerank": pr,
                      "hub": hits.hub, "authority": hits.authority})
    return t.round(4).sort_values("pagerank", ascending=False)


def ranks(scores, ascending: bool = False) -> pd.Series:
    """Rank positions (1 = top); ties share the best position."""
    return pd.Series(scores, dtype=float).rank(ascending=ascending, method="min").astype(int)


def compare_rankings(a, b, k: int = 5) -> dict:
    """How far two rankings agree: Spearman and Kendall rank correlations over all nodes, and the overlap of their
    top-k sets (ties at the k-th place broken by node name)."""
    from scipy import stats
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    idx = a.index.intersection(b.index)
    a, b = a[idx], b[idx]
    top = lambda s: set(sorted(s.index, key=lambda n: (-s[n], str(n)))[:k])
    ta, tb = top(a), top(b)
    return {"spearman": round(float(stats.spearmanr(a, b).statistic), 4),
            "kendall": round(float(stats.kendalltau(a, b).statistic), 4),
            f"top{k}_overlap": len(ta & tb), "only_first": sorted(ta - tb, key=str), "only_second": sorted(tb - ta, key=str)}


def perturb(G, add=None, remove=None, weight: str = "mentions"):
    """A copy of G with one edge added (add=(u, v) or (u, v, w)) and/or one removed (remove=(u, v))."""
    H = G.copy()
    if remove is not None:
        H.remove_edge(*remove[:2])
    if add is not None:
        u, v = add[:2]
        H.add_edge(u, v, **{weight: add[2] if len(add) > 2 else 1})
    return H


def rank_shift(before, after, k: int = None) -> pd.DataFrame:
    """Scores and rank positions before and after a change, with the change in rank (positive = moved up).
    Scores are rounded to 6 decimals before ranking, so that numerical noise does not separate users who tie."""
    b, a = pd.Series(before, dtype=float).round(6), pd.Series(after, dtype=float).round(6)
    t = pd.DataFrame({"before": b.round(4), "after": a.round(4), "rank_before": ranks(b), "rank_after": ranks(a)})
    t["moved_up"] = t.rank_before - t.rank_after
    t = t.sort_values("rank_after")
    return t.head(k) if k else t


# =========================================================================== setup notebook self-check
_QUIZ = {
    "Q1": ("Ben", "Ben is mentioned by five different users (Amy, Clara, Dylan, Ella and Fabio): the largest in-degree.",
           "Section 2 of this notebook"),
    "Q2": ("Gianni", "Every chain of mentions from the left-hand group to the right-hand group passes through Gianni. "
                     "Ben has more links, but they all stay inside his own group.", "Section 2 of this notebook"),
    "Q3": ("Hana", "Hana is mentioned by four users (seven mentions in all) but never mentions anyone: she has no out-edges.",
           "Section 2 of this notebook"),
    "Q4": (0.5, "After one step, Amy's probability is split equally between Ben and Clara: 1/2 each.",
           "Section 3 of this notebook"),
    "Q5": (0.5, "Step two: Ben passes his 1/2 to Amy, and Clara passes her 1/2 to Ben. Amy now holds 1/2, Ben 1/2, "
                "Clara 0.", "Section 3 of this notebook"),
    "Q6": (1.0, "Every row of the transition matrix sums to 1, so the total probability stays 1 at every step: none is "
                "created and none is lost (no user in trio is a dead end).", "Section 3 of this notebook"),
}


def quiz(qid: str, answer=None) -> None:
    """Self-check: compare your prediction with the answer, and say where to revisit the topic."""
    qid = str(qid).upper()
    if qid not in _QUIZ:
        raise KeyError(f"No question {qid!r}; the questions are {', '.join(_QUIZ)}")
    right, why, where = _QUIZ[qid]
    if answer is None:
        print(f"No prediction recorded. The answer is {right!r}. {why}")
    else:
        if isinstance(right, float):
            try:
                ok = abs(float(answer) - right) < 1e-6
            except (TypeError, ValueError):
                ok = False
        else:
            ok = str(answer).strip().lower() == str(right).lower()
        print(("Correct. " if ok else f"Not quite: you answered {answer!r}; the answer is {right!r}. ") + why)
    print(f"Revisit: {where}.")


# =========================================================================== practice checks
_say, _not_attempted = S2._say, S2._not_attempted

P1_COLUMNS = ["in_degree", "in_strength", "out_degree", "pagerank", "betweenness"]


def reference_ranking(alpha: float = 0.85) -> pd.DataFrame:
    """The P1 table computed by the kit on the attention graph (every registered user, self-loops removed):
    in_degree, in_strength (replies received), out_degree, pagerank (weighted by replies, alpha, dangling nodes
    teleport) and betweenness (directed, unweighted, normalised)."""
    A = attention_graph()
    return pd.DataFrame({"in_degree": dict(A.in_degree()), "in_strength": dict(A.in_degree(weight="replies")),
                         "out_degree": dict(A.out_degree()),
                         "pagerank": nx.pagerank(A, alpha=alpha, weight="replies", tol=1e-12, max_iter=10000),
                         "betweenness": nx.betweenness_centrality(A, normalized=True)})


_REF = {}


def _ref_ranking():
    if "r" not in _REF:
        _REF["r"] = reference_ranking()
    return _REF["r"]


def check(exercise: str, answer=None, **more):
    """Check a practice exercise, e.g. T.check("P1", table). Prints one PASS or FAIL line per fact."""
    fn = {"P1": _check_p1, "P2": _check_p2, "P3": _check_p3, "P4": _check_p4}.get(str(exercise).upper())
    if fn is None:
        raise KeyError(f"No exercise {exercise!r}. Exercises: P1, P2, P3, P4")
    fn(answer, **more)


def _check_p1(table, definition=None):
    if _not_attempted(table, "table"):
        return
    if not isinstance(table, pd.DataFrame):
        return _say(False, "the answer should be a pandas DataFrame indexed by user.")
    ref = _ref_ranking()
    _say(len(table) == len(ref), f"{len(table)} users" + ("." if len(table) == len(ref) else
         f" (expected {len(ref)}: every registered user, including those with no edges)"))
    hints = {"in_degree": "in the attention graph u -> v means u replied to v: in-degree counts the different users who "
                          "replied to v (check that you reversed the flow graph's edges, author -> replier)",
             "in_strength": "in-strength adds the replies on the incoming edges (weight='replies')",
             "out_degree": "out-degree counts the different users v replied to",
             "pagerank": "use the reply counts as weights, alpha = 0.85, and keep every user as a node",
             "betweenness": "directed, unweighted, normalised betweenness on the attention graph"}
    for c in P1_COLUMNS:
        if c not in table.columns:
            _say(False, f"column {c} is missing.")
            continue
        got = pd.Series(table[c], dtype=float).reindex(ref.index)
        ok = got.notna().all() and np.allclose(got.values, ref[c].values, atol=1e-4)
        _say(ok, f"{c} agrees with the kit for every user." if ok else f"{c} differs; hint: {hints[c]}")
    has_def = bool(str(definition or "").strip())
    _say(has_def, "the table comes with its definition." if has_def else
         "pass definition='...' saying what an edge means, its direction and its weight.")


def _check_p2(scores, alpha: float = 0.85):
    if _not_attempted(scores, "my_scores"):
        return
    s = pd.Series(scores, dtype=float)
    ref = nx.pagerank(attention_graph(), alpha=alpha, weight="replies", tol=1e-12, max_iter=10000)
    ref = pd.Series(ref)
    _say(abs(s.sum() - 1) < 1e-6, f"the scores sum to {s.sum():.6f}" + ("." if abs(s.sum() - 1) < 1e-6 else
         ": probability is leaking; a walker on a node with no out-edges must teleport (spread its share over all users)"))
    s = s.reindex(ref.index)
    ok = s.notna().all() and np.allclose(s.values, ref.values, atol=1e-5)
    _say(ok, "your power iteration agrees with networkx's PageRank to 1e-5." if ok else
         f"largest difference {np.nanmax(np.abs(s.values - ref.values)):.2e}; check the weights (replies), alpha, the "
         "teleportation term (1 - alpha) / n and the dangling term, and iterate until the change is tiny")


def _check_p3(table):
    if _not_attempted(table, "comparison"):
        return
    need = {"first", "second", "spearman", "top10_overlap"}
    if not isinstance(table, pd.DataFrame) or not need <= set(table.columns):
        return _say(False, f"the comparison should be a DataFrame with columns {sorted(need)}.")
    ref = _ref_ranking()
    bad = []
    for r in table.itertuples(index=False):
        if r.first not in ref.columns or r.second not in ref.columns:
            bad.append(f"unknown measure in ({r.first}, {r.second})")
            continue
        c = compare_rankings(ref[r.first], ref[r.second], k=10)
        if abs(r.spearman - c["spearman"]) > 1e-3:
            bad.append(f"{r.first} vs {r.second}: Spearman {c['spearman']}, not {r.spearman}")
        if int(r.top10_overlap) != c["top10_overlap"]:
            bad.append(f"{r.first} vs {r.second}: top-10 overlap {c['top10_overlap']}, not {r.top10_overlap} "
                       "(break ties at the tenth place by user name)")
    _say(not bad, f"{len(table)} comparisons agree with the kit." if not bad else "; ".join(bad[:3]))


def _check_p4(shift, add=None, remove=None, alpha: float = 0.85):
    if _not_attempted(shift, "shift"):
        return
    if not isinstance(shift, pd.DataFrame) or not {"rank_before", "rank_after"} <= set(shift.columns):
        return _say(False, "the answer should be a DataFrame indexed by user with rank_before and rank_after.")
    if add is None and remove is None:
        return _say(False, "pass the edge you changed, e.g. T.check('P4', shift, add=('u', 'v', 3)).")
    A = attention_graph()
    B = perturb(A, add=add, remove=remove, weight="replies")
    pr = lambda H: nx.pagerank(H, alpha=alpha, weight="replies", tol=1e-12, max_iter=10000)
    ref = rank_shift(pr(A), pr(B))
    got = shift.reindex(ref.index)
    need = {"before", "after", "rank_before", "rank_after"}
    if not need <= set(shift.columns) or got[list(need)].isna().any().any():
        return _say(False, "use T.rank_shift(before, after) on the PageRank of every registered user, before and after.")
    top = ref.moved_up.abs().idxmax()
    # scores are compared to 1e-4; users whose scores differ only in the sixth decimal place may swap places
    close = (np.allclose(got.before, ref.before, atol=1e-4) and np.allclose(got.after, ref.after, atol=1e-4) and
             abs(int(got.loc[top, "rank_after"]) - int(ref.loc[top, "rank_after"])) <= 2)
    _say(close, "scores and ranks before and after agree with the kit." if close else
         "the scores differ: compute weighted PageRank (replies, alpha 0.85) on the whole graph before and after the change")
    if close:
        print(f"The largest move is {top}: {int(ref.loc[top, 'rank_before'])} -> {int(ref.loc[top, 'rank_after'])}. "
              f"Moves of one or two places among users with almost equal scores are expected and not meaningful.")


# =========================================================================== reference results
def reference() -> dict:
    """The frozen results of the run used on the slides and in the notes (reference/results.json)."""
    return json.loads((ROOT / "reference" / "results.json").read_text(encoding="utf-8"))
