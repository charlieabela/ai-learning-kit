"""Paths, connectivity and structure: readable helpers.

Used in the topic's notebooks as T = kit.tools() after kit.start("paths-and-connectivity").

What is here, in the order the notebooks use it:
- two small graphs that can be checked by hand:
  paths_toy()  : eight students, undirected; every link carries km (distance between homes) and
                 msgs (messages per week, an interaction frequency);
  reply_toy()  : the same students plus three more on a forum, directed: u -> v means v replied to u,
                 so information that u posted reached v; one student never interacted;
- StudyHub, a fictional course forum of 300 users generated with a fixed seed: forum() returns the users
  table and the replies table; forum_graph() builds the directed graph;
- paths: path_cost(), shortest_route(), all_shortest(), frequency_to_cost(), describe_path();
- reachability: reachable_from(), can_reach(), explain_unreachable();
- graph views and structure: view(), structure_table(), random_baseline_clustering();
- degree distributions: degree_counts(), ccdf(), plot_distribution(), tail_fit() (optional depth);
- SciFact's kNN document graph from the text-to-graphs topic: scifact_knn_indegree();
- the setup self-check quiz() and the practice checks check("P1") ... check("P3").

Edge semantics are stated with every result: a path is only meaningful together with the definition of
what an edge is and what its cost means (describe_path() returns both).
"""
from __future__ import annotations

import heapq
import json
import math
from collections import Counter, deque
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from ..text_to_graphs import tools as TG

ROOT = Path(__file__).parent              # ROOT / "reference" holds the frozen reference results
SEED = 2205
VERSION = "2026-11-02"
NAVY, BLUE, ORANGE, GREY, GREEN = "#1F3864", "#4472C4", "#ED7D31", "#9CA9B6", "#70AD47"
BLUE, ORANGE, GREY, GREEN, PURPLE = "#2F5D8C", "#D37B35", "#9CA9B6", "#2E9D6B", "#7E57C2"


def _layout(G, seed=SEED):
    return nx.spring_layout(G.to_undirected() if G.is_directed() else G, seed=seed, weight=None)


def draw(G, pos=None, ax=None, labels=True, edge_labels=None, highlight=None, node_size=900, title=None,
         node_color=BLUE, font_size=10, colors=None, boxes=False):
    """Draw a small graph. Directed edges get arrowheads, and a pair of opposite edges is drawn as two curves so
    both arrows show. edge_labels: an edge attribute to print on edges. highlight: a list of nodes forming a path.
    colors: {node: colour}. boxes=True draws each node as a labelled box (better for names) instead of a circle."""
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    pos = pos or _layout(G)
    nc = [colors.get(n, node_color) for n in G.nodes()] if colors else node_color
    if boxes:
        for n in G.nodes():
            x, y = pos[n]
            ax.text(x, y, str(n), ha="center", va="center", color="white", fontsize=font_size, fontweight="bold",
                    zorder=5, bbox=dict(boxstyle="round,pad=0.35", fc=(colors or {}).get(n, node_color), ec="white", lw=1.5))
        ax.scatter([p[0] for p in pos.values()], [p[1] for p in pos.values()], s=1, alpha=0)
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=nc, node_size=node_size, edgecolors="white", linewidths=1.5)
    if labels and not boxes:
        nx.draw_networkx_labels(G, pos, ax=ax, font_color="white", font_size=font_size, font_weight="bold")
    hl = set()
    if highlight:
        hl = set(zip(highlight[:-1], highlight[1:])) | (set() if G.is_directed() else set(zip(highlight[1:], highlight[:-1])))
    if G.is_directed():
        straight = [(u, v) for u, v in G.edges() if not G.has_edge(v, u)]
        curved = [(u, v) for u, v in G.edges() if G.has_edge(v, u)]
        for group, rad in ((straight, 0.0), (curved, 0.18)):
            if not group:
                continue
            cols = [ORANGE if e in hl else GREY for e in group]
            nx.draw_networkx_edges(G, pos, edgelist=group, ax=ax, arrows=True, arrowstyle="-|>", arrowsize=18,
                                   width=[3.5 if e in hl else 1.8 for e in group], edge_color=cols,
                                   connectionstyle=f"arc3,rad={rad}", node_size=node_size,
                                   min_source_margin=8, min_target_margin=12)
    else:
        edges = list(G.edges())
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax, width=[3.5 if e in hl else 1.8 for e in edges],
                               edge_color=[ORANGE if e in hl else GREY for e in edges])
    if edge_labels:
        lab = {(u, v): (f"{d[edge_labels]:.3g}" if isinstance(d.get(edge_labels), float) else d.get(edge_labels))
               for u, v, d in G.edges(data=True) if edge_labels in d}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=lab, ax=ax, font_size=font_size - 1,
                                     connectionstyle="arc3,rad=0.18" if G.is_directed() else "arc3",
                                     bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    if title:
        ax.set_title(title)
    ax.axis("off")
    return ax


# =========================================================================== the two hand-checkable graphs
_PATH_EDGES = [  # (u, v, km between their homes, messages per week)
    ("Amy", "Dylan", 9, 1), ("Dylan", "Hana", 8, 2),
    ("Amy", "Ben", 2, 3), ("Ben", "Clara", 3, 2), ("Clara", "Gianni", 2, 4), ("Gianni", "Hana", 3, 3),
    ("Amy", "Ella", 6, 20), ("Ella", "Fabio", 5, 15), ("Fabio", "Hana", 7, 12),
    ("Ben", "Ella", 5, 6), ("Clara", "Fabio", 4, 5), ("Dylan", "Gianni", 6, 2),
]


def paths_toy() -> nx.Graph:
    """Eight students (invented). An edge means the two know each other. Each edge carries:
    km   : the distance between their homes, a cost (a printed handout passed hand to hand travels it);
    msgs : messages they exchange per week, an interaction frequency (a strength, not a cost)."""
    G = nx.Graph()
    for u, v, km, m in _PATH_EDGES:
        G.add_edge(u, v, km=km, msgs=m)
    return G


PATHS_POS = {"Amy": (-0.8, 0), "Ben": (1.2, 0.6), "Clara": (3.0, 1.0), "Gianni": (4.8, 0.6), "Hana": (6.8, 0),
             "Dylan": (3.0, 3.2), "Ella": (1.4, -1.8), "Fabio": (4.6, -1.8)}

_REPLY_EDGES = [  # (u, v): v replied to u, so what u posted reached v
    ("Amy", "Ben"), ("Ben", "Clara"), ("Clara", "Amy"), ("Clara", "Dylan"), ("Dylan", "Ella"), ("Ella", "Dylan"),
    ("Fabio", "Amy"), ("Ella", "Hana"), ("Gianni", "Hana"), ("Ivan", "Julia"), ("Julia", "Ivan"),
]


def reply_toy() -> nx.DiGraph:
    """Eleven forum users (invented). u -> v means v replied to u at least once, so information u posted reached
    v. Kurt registered but never posted or replied: he is a node with no edges."""
    G = nx.DiGraph()
    G.add_nodes_from(["Amy", "Ben", "Clara", "Dylan", "Ella", "Fabio", "Gianni", "Hana", "Ivan", "Julia", "Kurt"])
    G.add_edges_from(_REPLY_EDGES)
    return G


REPLY_POS = {"Amy": (0, 1), "Ben": (1.2, 2), "Clara": (2.2, 1), "Dylan": (3.6, 1), "Ella": (4.8, 1.8),
             "Fabio": (-1.3, 0.2), "Hana": (6.0, 1), "Gianni": (5.0, -0.2), "Ivan": (1.0, -0.6), "Julia": (2.4, -0.6),
             "Kurt": (-1.3, -0.8)}


def draw_paths_toy(G=None, show="both", highlight=None, ax=None, title=None):
    """Draw paths_toy with its edge labels. show: "both" (km and messages per week), "km", "msgs", "cost" (1/msgs)
    or None. highlight: a route (list of nodes) drawn in orange."""
    G = add_costs(paths_toy()) if G is None else G
    H = G.copy()
    for u, v, d in H.edges(data=True):
        d["label"] = {"both": f"{d['km']} km | {d['msgs']}/wk", "km": f"{d['km']} km", "msgs": f"{d['msgs']}/wk",
                      "cost": f"{1 / d['msgs']:.3g}"}.get(show, "")
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))
    pos = {n: (1.4 * x, y) for n, (x, y) in PATHS_POS.items()}      # wider, so that labels leave the lines visible
    return draw(H, pos=pos, ax=ax, edge_labels="label" if show else None, highlight=highlight, title=title,
                boxes=True, font_size=11)


def draw_reply_toy(R=None, start=None, ax=None, title=None):
    """Draw reply_toy (u -> v: v replied to u). With start, the start is orange, the users it reaches are blue and
    the users it cannot reach are grey."""
    R = reply_toy() if R is None else R
    colors = None
    if start is not None:
        reach = reachable_from(R, start)
        colors = {n: ORANGE if n == start else BLUE if n in reach else GREY for n in R.nodes()}
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 4.5))
    return draw(R, pos=REPLY_POS, ax=ax, colors=colors, title=title, boxes=True, font_size=11)


# =========================================================================== breadth-first search, step by step
def bfs_trace(G, source, reverse: bool = False) -> pd.DataFrame:
    """Breadth-first search written out step by step: at each step the front of the queue is taken out, its
    neighbours that have not been seen yet are discovered (in alphabetical order) and join the back of the queue.
    Directed graphs are followed along the edges (reverse=True follows them backwards: who can reach source).
    Returns one row per step: the node taken out, its hop count, the nodes it discovered and the queue after."""
    from collections import deque
    hops, queue, rows = {source: 0}, deque([source]), []
    while queue:
        u = queue.popleft()
        if G.is_directed():
            nbrs = G.predecessors(u) if reverse else G.successors(u)
        else:
            nbrs = G.neighbors(u)
        new = [v for v in sorted(nbrs, key=str) if v not in hops]
        for v in new:
            hops[v] = hops[u] + 1
            queue.append(v)
        rows.append({"step": len(rows) + 1, "visit": u, "hops": hops[u], "discovered": ", ".join(map(str, new)) or "-",
                     "queue_after": ", ".join(map(str, queue)) or "(empty)"})
    return pd.DataFrame(rows)


bfs_hops = TG.bfs_hops                       # from text-to-graphs: {node: hop count from source}
load_scifact = TG.load_scifact               # from text-to-graphs: the SciFact abstracts (doc_id, title, text)


# =========================================================================== paths and their meaning
def frequency_to_cost(f, method: str = "1/f"):
    """Turn an interaction frequency (a strength: more is closer) into a cost (more is farther).
    "1/f"   : one over the frequency (20 messages a week cost 0.05, one message costs 1);
    "-log"  : -ln(f / f_max) for a given f_max (pass method=("-log", f_max)); the strongest link costs 0."""
    if isinstance(method, tuple) and method[0] == "-log":
        return -math.log(f / method[1])
    if method == "1/f":
        if f <= 0:
            raise ValueError("a frequency of 0 means no interaction: such pairs should not be edges")
        return 1.0 / f
    raise ValueError('method must be "1/f" or ("-log", f_max)')


def add_costs(G, source: str = "msgs", target: str = "cost", method="1/f"):
    """Add a cost attribute computed from a frequency attribute with frequency_to_cost(). Returns G."""
    for u, v, d in G.edges(data=True):
        d[target] = frequency_to_cost(d[source], method)
    return G


def path_cost(G, path, weight):
    """The sum of an edge attribute along a path (weight=None counts edges)."""
    if weight is None:
        return len(path) - 1
    return sum(G[u][v][weight] for u, v in zip(path[:-1], path[1:]))


def shortest_route(G, source, target, weight=None):
    """One shortest path (fewest edges when weight is None, smallest total weight otherwise), with ties broken
    by the alphabetical order of the paths, so that every notebook gets the same answer."""
    if weight is None:
        paths = list(nx.all_shortest_paths(G, source, target))
    else:
        paths = list(nx.all_shortest_paths(G, source, target, weight=weight))
    return sorted(paths)[0]


def all_shortest(G, source, target, weight=None):
    """Every shortest path between source and target (there can be several: a tie)."""
    return sorted(nx.all_shortest_paths(G, source, target, weight=weight))


DEFINITIONS = {
    None: "fewest edges: every link costs 1, whatever it carries",
    "km": "smallest total distance: each link costs its km, the distance between the two homes",
    "cost": "strongest chain: each link costs 1/frequency, so frequent interaction makes a link short",
    "msgs": "WRONG for routing: each link costs its frequency, so the weakest links look shortest",
    "inv_replies": "strongest chain: each link costs 1/replies, so users who reply often are close",
}


def describe_path(G, source, target, weight=None, definition=None):
    """A shortest path together with what it means: the route, its length in edges, its total cost and the
    definition of an edge cost. Keep the definition with the result."""
    route = shortest_route(G, source, target, weight)
    return {"source": source, "target": target, "route": " > ".join(map(str, route)), "edges": len(route) - 1,
            "total": round(path_cost(G, route, weight), 4) if weight else len(route) - 1,
            "weight": weight or "hops", "definition": definition or DEFINITIONS.get(weight, f"each link costs its {weight}"),
            "ties": len(all_shortest(G, source, target, weight))}


# =========================================================================== reachability
def reachable_from(G, source) -> set:
    """Every node that can be reached from source along the edges (directions respected), source excluded."""
    return set(nx.descendants(G, source))


def can_reach(G, target) -> set:
    """Every node from which target can be reached (the nodes whose information can arrive at target)."""
    return set(nx.ancestors(G, target))


REASONS = {
    "isolated": "never interacted: the user has no edge at all (only the users table knows they exist)",
    "other_component": "in a different weakly connected component: no link joins them to the start, in any direction",
    "never_replied": "never replied to anyone: no edge points into this user, so nothing can arrive",
    "upstream_only": "replied only to users the start cannot reach: every edge into this user comes from unreachable users",
}


def explain_unreachable(G, source, users=None) -> pd.DataFrame:
    """Every user that cannot be reached from source, with the reason (see REASONS). users: the full list of
    users (for example the users table's user column); users missing from G are reported as isolated."""
    nodes = list(users) if users is not None else list(G.nodes())
    reach = reachable_from(G, source) | {source}
    weak = {n: i for i, comp in enumerate(nx.weakly_connected_components(G) if G.is_directed() else nx.connected_components(G)) for n in comp}
    rows = []
    for u in nodes:
        if u in reach:
            continue
        if u not in G or G.degree(u) == 0:
            reason = "isolated"
        elif weak[u] != weak[source]:
            reason = "other_component"
        elif G.in_degree(u) == 0:
            reason = "never_replied"
        else:
            reason = "upstream_only"
        rows.append({"user": u, "reason": reason})
    return pd.DataFrame(rows, columns=["user", "reason"]).sort_values(["reason", "user"]).reset_index(drop=True)


# =========================================================================== StudyHub: the fictional forum
_FIRST = ["luca", "maria", "jake", "sara", "matthew", "nicole", "kurt", "elena", "daniel", "julia", "andrea", "karl",
          "rebecca", "neil", "chiara", "liam", "martina", "gabriel", "francesca", "ryan", "michela", "aidan", "bettina",
          "clayton", "denise", "edward", "fiona", "glen", "isabel", "jacob", "kristina", "leon", "miriam", "nathan",
          "oriana", "paul", "roberta", "sean", "tiziana", "warren"]
_LAST = ["borg", "camilleri", "vella", "farrugia", "zammit", "galea", "micallef", "grech", "attard", "spiteri",
         "azzopardi", "mifsud", "agius", "cassar", "debono", "pace", "fenech", "caruana", "sammut", "abela"]


def _handles(n, rng):
    seen, out = set(), []
    while len(out) < n:
        h = f"{rng.choice(_FIRST)}_{rng.choice(_LAST)}"
        if h in seen:
            h = f"{h}{rng.integers(2, 99)}"
        if h not in seen:
            seen.add(h)
            out.append(h)
    return out


def _generate(seed=SEED):
    """Thread-based generation. Each thread has an author and repliers; a replier replies to the author and,
    sometimes, to an earlier replier in the same thread. An edge author -> replier counts the replies."""
    rng = np.random.default_rng(seed)
    n_core, n_new, n_group, n_eras, n_inactive = 251, 24, 6, 10, 8
    handles = _handles(n_core + n_new + n_group + n_eras + n_inactive, rng)
    core = handles[:n_core]
    tutors = core[:3]
    newcomers = handles[n_core:n_core + n_new]
    group = handles[n_core + n_new:n_core + n_new + n_group]
    eras = handles[n_core + n_new + n_group:n_core + n_new + n_group + n_eras]
    inactive = handles[n_core + n_new + n_group + n_eras:]
    bot = "studyhub_announcements"
    topic = {u: int(rng.integers(0, 4)) for u in core}
    attract = {u: float(rng.lognormal(0.0, 1.15)) for u in core}        # how much a user's posts attract replies
    for t in tutors:
        attract[t] *= 2.5
    replyrate = {u: float(rng.lognormal(0.0, 0.9)) for u in core}       # how readily a user replies
    for t in tutors:
        replyrate[t] *= 6
    never_reply = set(rng.choice(core[3:], size=14, replace=False))      # core members who only post
    for u in never_reply:
        replyrate[u] = 0.0
    counts = Counter()

    def pick(cands, weights, k):
        w = np.asarray(weights, float)
        if w.sum() == 0:
            return []
        k = min(k, int((w > 0).sum()))
        return list(rng.choice(cands, size=k, replace=False, p=w / w.sum()))

    def thread(author, pool, weights, n_rep, chain=0.45, self_bump=0.03):
        repliers = pick(pool, weights, n_rep)
        prev = []
        for r in repliers:
            if r == author:
                continue
            counts[(author, r)] += 1
            if prev and rng.random() < chain:
                q = prev[int(rng.integers(0, len(prev)))]
                if q != r:
                    counts[(q, r)] += 1
            prev.append(r)
        if repliers and rng.random() < self_bump:
            counts[(author, author)] += 1        # the author replying in their own thread: a self-loop
    # core threads: authors weighted by attractiveness, repliers mostly from the same topic
    for _ in range(1150):
        a = core[int(rng.choice(len(core), p=np.array([attract[u] for u in core]) / sum(attract.values())))]
        w = [replyrate[u] * (1.0 if topic[u] == topic[a] else 0.12) * (0 if u == a else 1) for u in core]
        thread(a, core, w, int(rng.integers(1, 6)))
    # announcements: many members reply to the bot; the bot never replies
    for _ in range(14):
        thread(bot, core, [replyrate[u] for u in core], int(rng.integers(8, 20)), chain=0.1, self_bump=0)
    # newcomers ask one or two questions, answered by core members, and never reply themselves
    for u in newcomers:
        for _ in range(int(rng.integers(1, 3))):
            thread(u, core, [replyrate[x] for x in core], int(rng.integers(1, 4)), chain=0.2, self_bump=0)
    # a study group replies only within the group; its questions are answered by core members
    for _ in range(18):
        a = group[int(rng.integers(0, len(group)))]
        thread(a, group, [0 if x == a else 1 for x in group], int(rng.integers(1, 4)), chain=0.3, self_bump=0)
    for u in group:
        thread(u, core, [replyrate[x] for x in core], int(rng.integers(1, 3)), chain=0.2, self_bump=0)
    # an Erasmus sub-forum that only talks to itself
    for _ in range(22):
        a = eras[int(rng.integers(0, len(eras)))]
        thread(a, eras, [0 if x == a else 1 for x in eras], int(rng.integers(1, 4)), chain=0.3, self_bump=0)
    users = pd.DataFrame({"user": tutors + core[3:] + newcomers + group + eras + inactive + [bot],
                          "role": ["tutor"] * 3 + ["member"] * (len(core) - 3 + n_new + n_group + n_eras + n_inactive) + ["bot"]})
    users = users.sort_values("user").reset_index(drop=True)
    replies = pd.DataFrame([{"author": str(a), "replier": str(r), "replies": c} for (a, r), c in counts.items()])
    replies = replies.sort_values(["author", "replier"]).reset_index(drop=True)
    truth = {"tutors": tutors, "newcomers": newcomers, "group": group, "erasmus": eras, "inactive": inactive,
             "bot": bot, "never_reply_core": sorted(map(str, never_reply))}
    return users, replies, truth


_CACHE = {}


def forum(seed: int = SEED):
    """StudyHub, a fictional course forum (invented users; generated with a fixed seed).
    Returns (users, replies): users has one row per registered user (user, role: member, tutor or bot);
    replies has one row per pair (author, replier, replies): replier replied `replies` times to posts of author.
    Some rows have author == replier (users replying in their own threads)."""
    if seed not in _CACHE:
        _CACHE[seed] = _generate(seed)
    users, replies, _ = _CACHE[seed]
    return users.copy(), replies.copy()


def forum_graph(seed: int = SEED, include_users: bool = True, drop_self_loops: bool = True) -> nx.DiGraph:
    """The forum as a directed graph: author -> replier means the replier replied to the author's posts, so
    information the author posted reached the replier. Edge attributes: replies (an interaction frequency) and
    inv_replies = 1 / replies (a cost for strongest-chain paths). include_users adds users with no edges."""
    users, replies = forum(seed)
    G = nx.DiGraph()
    if include_users:
        G.add_nodes_from(users.user)
    for a, r, c in replies.itertuples(index=False):
        if drop_self_loops and a == r:
            continue
        G.add_edge(a, r, replies=int(c), inv_replies=1.0 / c)
    return G


def default_start(seed: int = SEED) -> str:
    """The suggested starting user for the practice notebook: the tutor who replied most."""
    users, replies = forum(seed)
    tutors = set(users.user[users.role == "tutor"])
    r = replies[replies.replier.isin(tutors) & (replies.author != replies.replier)].groupby("replier").replies.sum()
    return str(r.sort_values(ascending=False).index[0])


# =========================================================================== graph views and structure
def view(G: nx.DiGraph, kind: str = "directed", min_replies: int = 1):
    """A view of a directed, weighted interaction graph, each answering a different question.
    "directed"   : as it is (u -> v: v replied to u);
    "undirected" : an edge when either replied to the other (direction dropped; weights added);
    "reciprocal" : an edge only when both replied to each other (a two-way relationship);
    "strong"     : directed, keeping only pairs with at least min_replies replies.
    Every view keeps all nodes, so that isolated users still count."""
    if kind == "directed":
        return G.copy()
    if kind == "strong":
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes())
        H.add_edges_from((u, v, d) for u, v, d in G.edges(data=True) if d.get("replies", 1) >= min_replies)
        return H
    H = nx.Graph()
    H.add_nodes_from(G.nodes())
    for u, v, d in G.edges(data=True):
        w = d.get("replies", 1)
        if kind == "undirected":
            if H.has_edge(u, v):
                H[u][v]["replies"] += w
            else:
                H.add_edge(u, v, replies=w)
        elif kind == "reciprocal":
            if G.has_edge(v, u) and not H.has_edge(u, v):
                H.add_edge(u, v, replies=w + G[v][u].get("replies", 1))
        else:
            raise ValueError('kind must be "directed", "undirected", "reciprocal" or "strong"')
    return H


def structure_row(H, name: str) -> dict:
    """Nodes, edges, density, mean degree, isolated nodes, components and clustering of one view.
    Clustering is computed on the undirected version, and only reported for undirected views."""
    n, m = H.number_of_nodes(), H.number_of_edges()
    row = {"view": name, "nodes": n, "edges": m, "density": round(nx.density(H), 5),
           "mean_degree": round((m if H.is_directed() else 2 * m) / n, 3), "isolated": sum(1 for _ in nx.isolates(H))}
    if H.is_directed():
        row["weak_components"] = nx.number_weakly_connected_components(H)
        row["strong_components"] = nx.number_strongly_connected_components(H)
        row["avg_clustering"] = row["transitivity"] = None
    else:
        row["weak_components"] = nx.number_connected_components(H)
        row["strong_components"] = None
        row["avg_clustering"] = round(nx.average_clustering(H), 4)
        row["transitivity"] = round(nx.transitivity(H), 4)
    return row


def structure_table(G: nx.DiGraph, min_replies: int = 3) -> pd.DataFrame:
    """structure_row() for the four views of G (the strong view keeps pairs with at least min_replies replies)."""
    rows = [structure_row(view(G, "directed"), "directed"), structure_row(view(G, "undirected"), "undirected"),
            structure_row(view(G, "reciprocal"), "reciprocal"),
            structure_row(view(G, "strong", min_replies), f"strong (>= {min_replies} replies)")]
    return pd.DataFrame(rows)


def random_baseline_clustering(H: nx.Graph, trials: int = 5, seed: int = SEED) -> float:
    """Average clustering of random graphs with the same number of nodes and edges (G(n, m)): what clustering
    would be if links were placed at random. For a random graph it is about the density."""
    n, m = H.number_of_nodes(), H.number_of_edges()
    vals = [nx.average_clustering(nx.gnm_random_graph(n, m, seed=seed + i)) for i in range(trials)]
    return round(float(np.mean(vals)), 4)


# =========================================================================== degree distributions
def degree_counts(G, kind: str = "out") -> pd.Series:
    """How many nodes have each degree. kind: "out", "in" or "total" (directed graphs), or "total" for undirected."""
    if G.is_directed():
        deg = dict(G.out_degree() if kind == "out" else G.in_degree() if kind == "in" else G.degree())
    else:
        deg = dict(G.degree())
    return pd.Series(Counter(deg.values())).sort_index()


def ccdf(values) -> pd.DataFrame:
    """The complementary cumulative distribution: for each value x, the share of nodes with degree >= x."""
    v = np.sort(np.asarray(list(values)))
    xs = np.unique(v)
    share = [(v >= x).mean() for x in xs]
    return pd.DataFrame({"degree": xs, "share_at_least": share})


def plot_distribution(values, title="", axes=None, color=BLUE):
    """Three views of the same degrees: a histogram on linear axes, the same histogram on log-log axes, and the
    complementary cumulative distribution on log-log axes."""
    values = np.asarray(list(values))
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(15, 4))
    cnt = Counter(values)
    xs = np.array(sorted(cnt)); ys = np.array([cnt[x] for x in xs])
    axes[0].bar(xs, ys, color=color, width=0.9)
    axes[0].set_xlabel("degree"); axes[0].set_ylabel("number of nodes"); axes[0].set_title("linear axes")
    pos = xs > 0
    axes[1].scatter(xs[pos], ys[pos], color=color, s=18)
    axes[1].set_xscale("log"); axes[1].set_yscale("log")
    axes[1].set_xlabel("degree"); axes[1].set_ylabel("number of nodes"); axes[1].set_title("log-log histogram")
    c = ccdf(values[values > 0])
    axes[2].plot(c.degree, c.share_at_least, "o-", color=color, ms=3)
    axes[2].set_xscale("log"); axes[2].set_yscale("log")
    axes[2].set_xlabel("degree x"); axes[2].set_ylabel("share with degree >= x"); axes[2].set_title("CCDF, log-log")
    if title:
        axes[0].figure.suptitle(title)
    return axes


def tail_fit(values, xmin: int) -> dict:
    """Optional depth. Fits two models to the tail (values >= xmin) and compares them, following the approach of
    Clauset, Shalizi and Newman (2009) in a simplified, continuous approximation:
    - a power law, p(x) ~ x^-alpha, with the maximum-likelihood estimate alpha = 1 + n / sum ln(x / (xmin - 0.5));
    - a log-normal truncated at xmin, fitted by maximum likelihood.
    R is the log-likelihood ratio (positive favours the power law); p is Vuong's two-sided p-value: a large p
    means the data cannot tell the two apart. A straight-looking log-log plot is not, by itself, evidence."""
    from scipy import optimize, stats
    x = np.asarray([v for v in values if v >= xmin], float)
    n = len(x)
    if n < 10:
        raise ValueError("fewer than 10 values in the tail: choose a smaller xmin")
    lo = xmin - 0.5
    alpha = 1 + n / np.sum(np.log(x / lo))
    ll_pl = np.log((alpha - 1) / lo) - alpha * np.log(x / lo)

    def nll(theta):
        mu, s = theta[0], abs(theta[1]) + 1e-9
        tail = stats.norm.sf((np.log(lo) - mu) / s)
        return -np.sum(stats.norm.logpdf((np.log(x) - mu) / s) - np.log(s * x) - np.log(max(tail, 1e-300)))
    res = optimize.minimize(nll, [np.log(x).mean(), np.log(x).std() + 0.1], method="Nelder-Mead")
    mu, s = res.x[0], abs(res.x[1]) + 1e-9
    tailp = stats.norm.sf((np.log(lo) - mu) / s)
    ll_ln = stats.norm.logpdf((np.log(x) - mu) / s) - np.log(s * x) - np.log(tailp)
    d = ll_pl - ll_ln
    R = float(d.sum())
    sd = float(d.std(ddof=1))
    p = float(2 * stats.norm.sf(abs(R) / (sd * np.sqrt(n)))) if sd > 0 else 1.0
    return {"xmin": xmin, "n_tail": n, "alpha": round(float(alpha), 3), "lognormal_mu": round(float(mu), 3),
            "lognormal_sigma": round(float(s), 3), "R": round(R, 3), "p": round(p, 4),
            "verdict": ("power law favoured" if R > 0 else "log-normal favoured") if p < 0.1 else "cannot tell them apart"}


def scifact_knn_indegree(k: int = 5):
    """In-degrees of SciFact's directed kNN document graph from text-to-graphs (TF-IDF cosine, reference pipeline):
    how many abstracts list each abstract among their k nearest neighbours. Returns (graph, in-degree dict)."""
    tok = TG.corpus_tokens()
    X, ids, _ = TG.tfidf(tok, min_df=2)
    G = TG.knn_graph(X, ids, k=k)
    return G, dict(G.in_degree())


# =========================================================================== setup notebook self-check
_QUIZ = {
    "Q1": (["Ben", "Dylan", "Ella"], "Amy's neighbours in the paths graph are Ben, Dylan and Ella: they form layer 1.",
           "Section 3 of this notebook; breadth-first search in the text-to-graphs walkthrough"),
    "Q2": (["Clara", "Fabio", "Gianni", "Hana"], "Layer 2 holds the nodes first found from layer 1: Clara (from Ben), "
           "Gianni and Hana (from Dylan), Fabio (from Ella).", "Section 3 of this notebook"),
    "Q3": (3, "In reply_toy there are three weakly connected components: the eight users joined by replies, Ivan and "
              "Julia, and Kurt on his own.", "Section 4 of this notebook; weak and strong components in the text-to-graphs walkthrough"),
    "Q4": (7, "Strong components: {Amy, Ben, Clara}, {Dylan, Ella}, {Ivan, Julia}, and Fabio, Gianni, Hana and Kurt "
              "each on their own.", "Section 4 of this notebook"),
    "Q5": ("no", "Fabio and Amy are in the same weak component, but no edge leads into Fabio: nobody's posts ever reached "
                 "him, because he never replied. Weak connectivity ignores direction; reaching needs it.",
           "Section 4 of this notebook"),
    "Q6": (3, "Row Amy of the adjacency matrix has a 1 in Ben's column only, so Amy's out-degree is 1; column Amy has "
              "ones in the rows of Clara and Fabio, so Amy's in-degree is 2. Total degree: 3.",
           "Section 2 of this notebook; adjacency matrices in the text-to-graphs setup notebook"),
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
        if isinstance(right, list):
            ok = sorted(map(str, answer)) == sorted(right) if not isinstance(answer, str) else False
        else:
            ok = str(answer).strip().lower() == str(right).lower()
        print(("Correct. " if ok else f"Not quite: you answered {answer!r}; the answer is {right!r}. ") + why)
    print(f"Revisit: {where}.")


# =========================================================================== practice checks
def _say(ok: bool, text: str) -> bool:
    print(("PASS  " if ok else "FAIL  ") + text)
    return ok


def _not_attempted(value, name) -> bool:
    if value is None:
        print(f"Not attempted yet: complete the cell above so that {name} holds your answer, then run this check again.")
        return True
    return False


def reference_summary(G: nx.DiGraph, users, start) -> dict:
    """The connectivity summary the practice notebook asks for (P1), computed by the kit."""
    users = list(users)
    H = G.copy()
    H.add_nodes_from(users)
    H.remove_edges_from(list(nx.selfloop_edges(H)))
    n = H.number_of_nodes()
    weak = sorted(nx.weakly_connected_components(H), key=len, reverse=True)
    strong = sorted(nx.strongly_connected_components(H), key=len, reverse=True)
    mutual = sum(1 for u, v in H.edges() if H.has_edge(v, u))
    return {"users": n, "edges": H.number_of_edges(), "density": round(nx.density(H), 5),
            "isolated": sum(1 for _ in nx.isolates(H)),
            "weak_components": len(weak), "largest_weak_share": round(len(weak[0]) / n, 4),
            "strong_components": len(strong), "largest_strong_share": round(len(strong[0]) / n, 4),
            "reciprocity": round(mutual / H.number_of_edges(), 4),
            "reachable_from_start": len(nx.descendants(H, start)),
            "start": start}


def check(exercise: str, answer=None, **more):
    """Check a practice exercise, e.g. T.check("P1", summary). Prints one PASS or FAIL line per fact."""
    fn = {"P1": _check_p1, "P2": _check_p2, "P3": _check_p3}.get(str(exercise).upper())
    if fn is None:
        raise KeyError(f"No exercise {exercise!r}. Exercises: P1, P2, P3")
    fn(answer, **more)


def _check_p1(summary, start=None):
    if _not_attempted(summary, "summary"):
        return
    users, _ = forum()
    start = start or summary.get("start") or default_start()
    ref = reference_summary(forum_graph(include_users=False), users.user, start)
    hints = {"users": "count every registered user: add the users table's users as nodes, not only those in replies",
             "edges": "remove self-loops (author == replier) before counting edges",
             "density": "density of a directed graph is edges / (n (n - 1)), with n including isolated users",
             "isolated": "isolated users have no edges; they exist only in the users table",
             "weak_components": "weak components ignore direction; isolated users are components of their own",
             "strong_components": "a strong component needs paths both ways; every user that is not on a cycle is its own",
             "reciprocity": "reciprocity = edges whose reverse also exists / all edges (self-loops removed)",
             "reachable_from_start": "count the users reachable from the start along the edges (nx.descendants)"}
    for key, ref_value in ref.items():
        if key == "start":
            continue
        got = summary.get(key)
        if got is None:
            _say(False, f"{key}: missing from your summary.")
            continue
        ok = abs(float(got) - float(ref_value)) <= (1e-3 if isinstance(ref_value, float) else 0)
        _say(ok, f"{key} = {got}" + ("" if ok else f" (expected {ref_value}); hint: {hints.get(key, '')}"))
    has_def = bool(str(summary.get("path_definition", "")).strip())
    _say(has_def, "the summary states its path definition (what an edge means and which direction a path follows)."
         if has_def else "add a 'path_definition' entry: what an edge means and which direction a path follows.")


def _check_p2(table, start=None):
    if _not_attempted(table, "the comparison table"):
        return
    if not isinstance(table, pd.DataFrame) or table.empty:
        return _say(False, "the comparison should be a non-empty pandas DataFrame, one row per target user.")
    need = {"target", "hops_edges", "strength_edges", "strength_cost", "same_route", "path_definition"}
    missing = need - set(table.columns)
    _say(not missing, "all columns present." if not missing else f"missing columns: {sorted(missing)}")
    if missing:
        return
    start = start or default_start()
    G = forum_graph()
    bad = []
    for r in table.itertuples(index=False):
        try:
            h = nx.shortest_path_length(G, start, r.target)
            c = nx.shortest_path_length(G, start, r.target, weight="inv_replies")
        except nx.NetworkXNoPath:
            bad.append(f"{r.target} is not reachable from {start}; compare only reachable users")
            continue
        if r.hops_edges != h:
            bad.append(f"{r.target}: fewest-edges path has {h} edges, not {r.hops_edges}")
        if abs(r.strength_cost - c) > 1e-6:
            bad.append(f"{r.target}: strongest-chain cost is {c:.4f}, not {r.strength_cost}")
        if r.strength_edges < h:
            bad.append(f"{r.target}: a strongest chain cannot have fewer edges than the fewest-edges path")
    _say(not bad, f"{len(table)} rows agree with the kit's paths." if not bad else "; ".join(bad[:3]))
    defs = table.path_definition.astype(str).str.strip().str.len().gt(0).all()
    _say(bool(defs), "every row keeps its path definition." if defs else "fill path_definition in every row.")


def _check_p3(table, start=None):
    if _not_attempted(table, "the unreachable table"):
        return
    if not isinstance(table, pd.DataFrame) or not {"user", "reason"} <= set(table.columns):
        return _say(False, "the answer should be a DataFrame with columns user and reason.")
    start = start or default_start()
    users, _ = forum()
    ref = explain_unreachable(forum_graph(), start, users.user)
    got_users, ref_users = set(table.user), set(ref.user)
    missed = sorted(ref_users - got_users)
    extra = sorted(got_users - ref_users)
    _say(not missed and not extra, f"{len(ref_users)} unreachable users found." if not missed and not extra else
         (f"missed {len(missed)} (for example {missed[:3]}; isolated users are only in the users table)" if missed else "")
         + (f"; {len(extra)} listed that are reachable (for example {extra[:3]})" if extra else ""))
    merged = ref.merge(table, on="user", how="inner", suffixes=("_ref", ""))
    wrong = merged[merged.reason != merged.reason_ref]
    _say(wrong.empty, "every reason matches." if wrong.empty else
         f"{len(wrong)} reasons differ, for example {wrong.user.iloc[0]}: expected {wrong.reason_ref.iloc[0]} "
         f"({REASONS[wrong.reason_ref.iloc[0]]}).")
    counts = ref.reason.value_counts().to_dict()
    print("Reference counts by reason:", counts)


# =========================================================================== reference results
def reference() -> dict:
    """The frozen results of the run used on the slides and in the notes (reference/results.json)."""
    return json.loads((ROOT / "reference" / "results.json").read_text(encoding="utf-8"))
