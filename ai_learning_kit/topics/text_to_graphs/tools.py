"""Text to graphs: helpers returned by T = kit.tools() after kit.start("text-to-graphs").

- A miniature of six short abstracts, each reduced to four index terms, so that every number in the
  notebooks can be checked by hand (miniature(), miniature_tokens(), miniature_similarity_graph()).
- SciFact (Wadden et al., 2020), 5,183 biomedical abstracts, read through ai_learning_kit.datasets, and
  the reference preprocessing pipeline with its variants (corpus_tokens()), or a learner's own
  preprocessing output (tokens_from_file(), tfidf_from_file()).
- Graph construction: the bipartite term-document graph and its projections, the concept co-occurrence
  graph with count and lift thresholds, and k-nearest-neighbour, mutual and threshold document graphs from
  TF-IDF cosine similarity.
- Traversal and paths written out in a few readable lines (bfs_hops, bfs_order, dfs_order, components,
  dijkstra), so that kit.source(T.<name>) shows the whole algorithm, and the conversions from similarity
  to distance.
- summary(), sweeps over construction settings, drawing, and the practice checks (check("E1") ...).

Released helpers never change; a change is a new function.
"""
from __future__ import annotations

import heapq
import json
import math
import re
import tarfile
from collections import Counter, deque
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from scipy import sparse

from ... import datasets

ROOT = Path(__file__).parent
SEED = 42

# =============================================================================================
# The miniature: six abstracts in the style of SciFact, each reduced to four index terms
# =============================================================================================
MINIATURE = [
    ("D1", "Vitamin D deficiency and the risk of respiratory infection", "vitamin deficiency infection risk"),
    ("D2", "Vitamin D supplements against infection: a randomised trial", "vitamin supplement infection trial"),
    ("D3", "Supplement trials, placebo groups and mortality", "supplement trial placebo mortality"),
    ("D4", "p53 mutations in tumour cells", "tumour cell mutation p53"),
    ("D5", "A trial of a new therapy for tumour cells", "tumour cell therapy trial"),
    ("D6", "p53, mutation and apoptosis in the cell", "mutation p53 cell apoptosis"),
]

_TERM_POSITIONS = {   # vitamin D and infection on the left, tumours on the right, trial in the middle
    "deficiency": (-3.0, 1.2), "risk": (-3.6, 0.0), "vitamin": (-2.2, 0.4), "infection": (-2.6, -1.0),
    "supplement": (-1.0, -0.6), "placebo": (-1.6, -1.8), "mortality": (-0.4, -1.8), "trial": (0.2, 0.2),
    "therapy": (1.2, 1.3), "tumour": (1.6, 0.2), "cell": (2.6, -0.4), "mutation": (3.4, 0.8),
    "p53": (3.6, -0.8), "apoptosis": (2.8, -1.7),
}
_DOC_POSITIONS = {"D1": (-2.0, 1.0), "D2": (-1.0, 0.0), "D3": (-2.0, -1.0),
                  "D4": (2.0, 1.0), "D5": (1.0, 0.0), "D6": (2.0, -1.0)}


def miniature() -> pd.DataFrame:
    """The six miniature abstracts: doc_id, title and their four index terms."""
    return pd.DataFrame(MINIATURE, columns=["doc_id", "title", "terms"])


def miniature_tokens() -> dict:
    """{doc_id: [terms]} for the miniature."""
    return {d: terms.split() for d, _, terms in MINIATURE}


def miniature_positions(kind: str = "terms") -> dict:
    """Fixed positions for drawing the miniature: kind "terms" or "docs"."""
    return dict(_TERM_POSITIONS if kind == "terms" else _DOC_POSITIONS)


def miniature_similarity_graph() -> nx.Graph:
    """The miniature's documents linked when their TF-IDF cosine similarity is positive (rounded to 3 places)."""
    X, ids, _ = tfidf(miniature_tokens(), min_df=1)
    S = similarity_matrix(X)
    g = nx.Graph()
    g.add_nodes_from(ids)
    for i, j in combinations(range(len(ids)), 2):
        if S[i, j] > 0:
            g.add_edge(ids[i], ids[j], similarity=round(float(S[i, j]), 3))
    return g


# =============================================================================================
# SciFact and preprocessing
# =============================================================================================
_TOKEN = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_LETTER = re.compile(r"[a-z]")
VARIANTS = ("reference", "keep_stopwords", "split_hyphens", "porter")


def _scifact_members(names=("corpus.jsonl",)):
    out = {}
    with tarfile.open(datasets.fetch("scifact", quiet=True)) as t:
        for m in t.getmembers():
            base = m.name.rsplit("/", 1)[-1]
            if base in names:
                out[base] = t.extractfile(m).read().decode("utf-8")
    return out


def load_scifact() -> pd.DataFrame:
    """SciFact's corpus: one row per abstract (doc_id as text, title, abstract as one string)."""
    text = _scifact_members()["corpus.jsonl"]
    rows = []
    for line in text.splitlines():
        if line.strip():
            r = json.loads(line)
            rows.append({"doc_id": str(r["doc_id"]), "title": r["title"].strip(),
                         "abstract": " ".join(s.strip() for s in r["abstract"])})
    return pd.DataFrame(rows)


def load_claims() -> pd.DataFrame:
    """SciFact's claims (train and dev, with their evidence document ids)."""
    members = _scifact_members(("claims_train.jsonl", "claims_dev.jsonl"))
    rows = []
    for split in ("train", "dev"):
        for line in members.get(f"claims_{split}.jsonl", "").splitlines():
            if line.strip():
                r = json.loads(line)
                rows.append({"id": r["id"], "split": split, "claim": r["claim"],
                             "cited_doc_ids": [str(d) for d in r.get("cited_doc_ids", [])]})
    return pd.DataFrame(rows)


def tokenize(text: str, variant: str = "reference") -> list:
    """One text as tokens. reference: lower case, terms of at least two characters with at least one letter
    (so numbers such as 1 or 2019 are dropped, but il-6 and p53 are kept), hyphenated terms kept whole,
    scikit-learn's English stop words removed, no stemming. The variants change one choice each:
    keep_stopwords, split_hyphens (anti-inflammatory becomes anti, inflammatory) or porter (adds the Porter
    stemmer)."""
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {', '.join(VARIANTS)}")
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    tokens = _TOKEN.findall(text.lower())
    if variant == "split_hyphens":
        tokens = [p for t in tokens for p in t.split("-") if p]
    tokens = [t for t in tokens if len(t) > 1 and _LETTER.search(t)]
    if variant != "keep_stopwords":
        tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS]
    if variant == "porter":
        stem = _porter()
        tokens = [stem(t) for t in tokens]
    return tokens


def _porter():
    try:
        from nltk.stem import PorterStemmer
    except ImportError:
        raise ImportError("The porter variant needs NLTK: install ai-learning-kit with the graphs extra.") from None
    return PorterStemmer().stem


_TOKENS_CACHE: dict = {}


def corpus_tokens(variant: str = "reference") -> dict:
    """{doc_id: [tokens]} for SciFact (title and abstract) from the reference pipeline or one of its variants."""
    if variant not in _TOKENS_CACHE:
        docs = load_scifact()
        _TOKENS_CACHE[variant] = {d: tokenize(f"{t} {a}", variant)
                                  for d, t, a in zip(docs.doc_id, docs.title, docs.abstract)}
    return {d: list(t) for d, t in _TOKENS_CACHE[variant].items()}


def tokens_from_file(path) -> dict:
    """Your own preprocessing output as {doc_id: [tokens]}. Accepted shapes:
    .json   {doc_id: [tokens]} (or {doc_id: "tokens separated by spaces"})
    .jsonl  one object per line with doc_id (or _id) and tokens (a list or a string)
    .csv / .tsv  columns doc_id and tokens, the tokens separated by spaces
    .pkl / .pickle  a dict {doc_id: [tokens]}"""
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".json":
        raw = json.loads(p.read_text(encoding="utf-8"))
    elif suffix == ".jsonl":
        raw = {}
        for line in p.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                raw[r.get("doc_id", r.get("_id"))] = r["tokens"]
    elif suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, sep="\t" if suffix == ".tsv" else ",", dtype=str).fillna("")
        raw = dict(zip(df["doc_id"], df["tokens"]))
    elif suffix in (".pkl", ".pickle"):
        import pickle
        with open(p, "rb") as f:
            raw = pickle.load(f)
    else:
        raise ValueError(f"{p.name}: use .json, .jsonl, .csv, .tsv or .pkl (see help(T.tokens_from_file))")
    return {str(k): (v.split() if isinstance(v, str) else [str(t) for t in v]) for k, v in raw.items()}


def document_frequency(tok: dict) -> Counter:
    """How many documents contain each term."""
    return Counter(t for tokens in tok.values() for t in set(tokens))


def concepts(tok: dict, top_n: int = 300) -> list:
    """The top_n terms that occur in the most documents (ties in alphabetical order)."""
    df = document_frequency(tok)
    return [t for t, _ in sorted(df.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]


def tfidf(tok: dict, min_df: int = 2):
    """TF-IDF vectors as in scikit-learn's TfidfVectorizer (smoothed idf, l2-normalised rows) for documents
    already split into tokens. Returns (X sparse matrix, ids, vocabulary), rows in the order of tok."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    ids = list(tok)
    vec = TfidfVectorizer(analyzer=lambda doc: doc, min_df=min_df)
    X = vec.fit_transform([tok[i] for i in ids])
    return X.tocsr(), ids, list(vec.get_feature_names_out())


def tfidf_from_file(path):
    """Your own TF-IDF weights, a JSON file {doc_id: {term: weight}}, as (X, ids, vocabulary) with each row
    scaled to length 1, so that dot products are cosine similarities."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = [str(d) for d in raw]
    vocab = sorted({t for w in raw.values() for t in w})
    col = {t: j for j, t in enumerate(vocab)}
    rows, cols, vals = [], [], []
    for i, d in enumerate(raw):
        for t, w in raw[d].items():
            rows.append(i); cols.append(col[t]); vals.append(float(w))
    X = sparse.csr_matrix((vals, (rows, cols)), shape=(len(ids), len(vocab)))
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    return sparse.diags(1.0 / norms) @ X, ids, vocab


def similarity_matrix(X) -> np.ndarray:
    """Cosine similarity of every pair of rows of an l2-normalised matrix, as a dense array."""
    S = X @ X.T
    return S.toarray() if sparse.issparse(S) else np.asarray(S)


# =============================================================================================
# Building graphs
# =============================================================================================
def bipartite_graph(tok: dict) -> nx.Graph:
    """Documents and terms as two kinds of node (attribute side = "doc" or "term"); an edge when a document
    contains a term."""
    g = nx.Graph()
    for d, tokens in tok.items():
        g.add_node(d, side="doc")
        for t in tokens:
            g.add_node(t, side="term")
            g.add_edge(d, t)
    return g


def project(B: nx.Graph, side: str = "term") -> nx.Graph:
    """Keep one kind of node (side "term" or "doc") and link two that share a neighbour on the other side,
    with weight = the number of shared neighbours."""
    keep = [n for n, s in B.nodes(data="side") if s == side]
    g = nx.Graph()
    g.add_nodes_from(keep)
    keep_set = set(keep)
    counts = Counter()
    for n, s in B.nodes(data="side"):
        if s != side:
            nbrs = sorted(v for v in B.neighbors(n) if v in keep_set)
            counts.update(combinations(nbrs, 2))
    for (u, v), w in counts.items():
        g.add_edge(u, v, weight=w)
    return g


def lift(tok: dict, a: str, b: str) -> dict:
    """Observed co-occurrences of two terms against those expected if they were unrelated:
    lift = observed / (df(a) x df(b) / N)."""
    n = len(tok)
    sets = [set(t) for t in tok.values()]
    da, db = sum(a in s for s in sets), sum(b in s for s in sets)
    obs = sum(a in s and b in s for s in sets)
    expected = da * db / n if n else 0.0
    return {"N": n, f"df({a})": da, f"df({b})": db, "observed": obs, "expected": round(expected, 2),
            "lift": round(obs / expected, 3) if expected else float("nan")}


def cooccurrence_graph(tok: dict, top_n: int | None = 300, terms=None, min_weight: int = 1,
                       min_lift: float | None = None, window: int | None = None) -> nx.Graph:
    """The concept co-occurrence graph: the chosen terms (the top_n by document frequency, or the terms
    given), linked when they occur in the same document, with weight = the number of such documents and
    lift = weight / (df(u) x df(v) / N). Edges need weight >= min_weight and, if given, lift >= min_lift.
    window=w counts two terms only when they occur within w consecutive tokens of each other."""
    chosen = list(terms) if terms is not None else concepts(tok, top_n)
    keep = set(chosen)
    n = len(tok)
    df = Counter()
    pairs = Counter()
    for tokens in tok.values():
        present = keep.intersection(tokens)
        df.update(present)
        if window is None:
            pairs.update(combinations(sorted(present), 2))
        else:
            seen = set()
            kept = [(i, t) for i, t in enumerate(tokens) if t in keep]
            for x, (i, t) in enumerate(kept):
                for j, u in kept[x + 1:]:
                    if j - i >= window:
                        break
                    if u != t:
                        seen.add(tuple(sorted((t, u))))
            pairs.update(seen)
    g = nx.Graph()
    g.add_nodes_from(chosen)
    for (u, v), w in pairs.items():
        lv = w / (df[u] * df[v] / n)
        if w >= min_weight and (min_lift is None or lv >= min_lift):
            g.add_edge(u, v, weight=w, lift=round(lv, 4))
    return g


def _similarity_rows(X, block: int = 512):
    """Cosine similarities row block by row block, so that a large corpus never needs a full dense matrix."""
    Xt = X.T.tocsc() if sparse.issparse(X) else np.asarray(X).T
    for start in range(0, X.shape[0], block):
        S = X[start:start + block] @ Xt
        yield start, (S.toarray() if sparse.issparse(S) else np.asarray(S))


def knn_graph(X, ids, k: int = 5, symmetric: bool = False, mutual: bool = False) -> nx.Graph:
    """Each document points to its k most similar documents (cosine similarity of the rows of X; only
    positive similarities; ties broken by position). Edge attribute similarity.
    symmetric=True: undirected, an edge when either document lists the other.
    mutual=True: undirected, an edge only when both list each other."""
    g = nx.DiGraph()
    g.add_nodes_from(ids)
    for start, S in _similarity_rows(X):
        for r, row in enumerate(S):
            i = start + r
            row[i] = -1.0
            for j in np.argsort(-row, kind="stable")[:k]:
                if row[j] > 0:
                    g.add_edge(ids[i], ids[j], similarity=float(row[j]))
    if not (symmetric or mutual):
        return g
    u = nx.Graph()
    u.add_nodes_from(ids)
    for a, b, s in g.edges(data="similarity"):
        if mutual and not g.has_edge(b, a):
            continue
        u.add_edge(a, b, similarity=s)
    return u


def threshold_graph(X, ids, min_sim: float = 0.3) -> nx.Graph:
    """Documents linked when their cosine similarity is at least min_sim (undirected)."""
    g = nx.Graph()
    g.add_nodes_from(ids)
    for start, S in _similarity_rows(X):
        for r, row in enumerate(S):
            i = start + r
            for j in np.nonzero(row >= min_sim)[0]:
                if j > i:
                    g.add_edge(ids[i], ids[j], similarity=float(row[j]))
    return g


DISTANCES = {"1-s": lambda s: 1.0 - s, "-log": lambda s: -math.log(s), "1/s": lambda s: 1.0 / s}


def add_distances(G, method: str = "1-s"):
    """Turn each edge's similarity into a distance, stored as the edge attribute distance:
    "1-s" (1 - s), "-log" (-ln s) or "1/s". Returns the same graph."""
    if method not in DISTANCES:
        raise ValueError(f"method must be one of {', '.join(DISTANCES)}")
    f = DISTANCES[method]
    for _, _, d in G.edges(data=True):
        d["distance"] = f(d["similarity"])
    return G


# =============================================================================================
# Exploring graphs: traversal, components, shortest paths
# =============================================================================================
def _neighbours(G, u):
    """The nodes an edge leads to from u (successors in a directed graph), in sorted order."""
    return sorted(G.neighbors(u), key=str)


def bfs_hops(G, source) -> dict:
    """Breadth-first search: {node: number of edges on a shortest path from source}, for reachable nodes."""
    hops, queue = {source: 0}, deque([source])
    while queue:
        u = queue.popleft()
        for v in _neighbours(G, u):
            if v not in hops:
                hops[v] = hops[u] + 1
                queue.append(v)
    return hops


def bfs_order(G, source) -> list:
    """The order in which breadth-first search discovers the nodes reachable from source."""
    return list(bfs_hops(G, source))


def bfs_tree(G, source) -> dict:
    """{node: the node from which breadth-first search first reached it} (source: None), for path_to()."""
    prev, queue = {source: None}, deque([source])
    while queue:
        u = queue.popleft()
        for v in _neighbours(G, u):
            if v not in prev:
                prev[v] = u
                queue.append(v)
    return prev


def dfs_order(G, source) -> list:
    """The order in which depth-first search visits the nodes reachable from source (neighbours in sorted
    order, always continuing from the node reached last)."""
    order, visited, stack = [], set(), [source]
    while stack:
        u = stack.pop()
        if u in visited:
            continue
        visited.add(u)
        order.append(u)
        stack.extend(v for v in reversed(_neighbours(G, u)) if v not in visited)
    return order


def _sorted_components(comps) -> list:
    return sorted((set(c) for c in comps), key=lambda c: (-len(c), sorted(map(str, c))))


def weak_components(G) -> list:
    """Connected components (for a directed graph, ignoring direction), largest first."""
    g = G.to_undirected(as_view=True) if G.is_directed() else G
    assigned, comps = set(), []
    for s in sorted(g.nodes(), key=str):
        if s not in assigned:
            comp = set(bfs_hops(g, s))
            assigned |= comp
            comps.append(comp)
    return _sorted_components(comps)


def strong_components(G) -> list:
    """Strongly connected components of a directed graph, largest first: the nodes reachable from s
    intersected with the nodes that can reach s, repeated for every node not yet assigned."""
    if not G.is_directed():
        return weak_components(G)
    reverse = G.reverse(copy=False)
    assigned, comps = set(), []
    for s in sorted(G.nodes(), key=str):
        if s not in assigned:
            comp = set(bfs_hops(G, s)) & set(bfs_hops(reverse, s))
            assigned |= comp
            comps.append(comp)
    return _sorted_components(comps)


def dijkstra(G, source, weight: str = "distance"):
    """Dijkstra's algorithm with a priority queue. Returns (dist, prev): dist[v] is the length of a shortest path
    from source, prev[v] the node before v on it. Every edge weight must be zero or positive."""
    dist, prev = {source: 0.0}, {source: None}
    done = set()
    heap = [(0.0, str(source), source)]
    while heap:
        d, _, u = heapq.heappop(heap)
        if u in done:
            continue
        done.add(u)
        for v in _neighbours(G, u):
            w = G[u][v][weight]
            if w < 0:
                raise ValueError(f"edge {u}-{v} has a negative {weight}; Dijkstra's algorithm needs non-negative weights")
            nd = d + w
            if v not in dist or nd < dist[v] - 1e-12:
                dist[v], prev[v] = nd, u
                heapq.heappush(heap, (nd, str(v), v))
    return {k: round(v, 10) for k, v in dist.items()}, prev


def path_to(prev: dict, target) -> list:
    """The path from the source to target, read back from a prev dictionary ([] if target was not reached)."""
    if target not in prev:
        return []
    path = [target]
    while prev[path[-1]] is not None:
        path.append(prev[path[-1]])
    return path[::-1]


# =============================================================================================
# Summaries, sweeps and hubs
# =============================================================================================
def summary(G) -> dict:
    """Size, density, mean degree, isolated nodes and components of a graph (weak and strong for a directed
    graph)."""
    n, m = G.number_of_nodes(), G.number_of_edges()
    pairs = n * (n - 1) if G.is_directed() else n * (n - 1) / 2
    out = {"nodes": n, "edges": m, "density": round(m / pairs, 6) if pairs else 0.0,
           "mean_degree": round((m if G.is_directed() else 2 * m) / n, 3) if n else 0.0,
           "isolated": sum(1 for v in G.nodes() if G.degree(v) == 0)}
    if G.is_directed():
        weak, strong = weak_components(G), strong_components(G)
        out.update(weak_components=len(weak), strong_components=len(strong),
                   largest_weak_share=round(len(weak[0]) / n, 4) if n else 0.0,
                   largest_strong_share=round(len(strong[0]) / n, 4) if n else 0.0)
    else:
        comps = weak_components(G)
        out.update(components=len(comps), largest_component_share=round(len(comps[0]) / n, 4) if n else 0.0)
    return out


def sweep_cooccurrence(tok: dict, thresholds=(1, 2, 5, 10, 20, 50, 100), top_n: int = 300, by: str = "weight",
                       min_weight: int = 1) -> pd.DataFrame:
    """summary() of the concept graph for each threshold: by="weight" varies the count threshold, by="lift" the
    lift threshold (with edges also needing min_weight shared documents)."""
    if by not in ("weight", "lift"):
        raise ValueError('by must be "weight" or "lift"')
    rows = []
    for t in thresholds:
        g = (cooccurrence_graph(tok, top_n=top_n, min_weight=t) if by == "weight"
             else cooccurrence_graph(tok, top_n=top_n, min_weight=min_weight, min_lift=t))
        rows.append({("min_weight" if by == "weight" else "min_lift"): t, **summary(g)})
    return pd.DataFrame(rows)


def sweep_knn(X, ids, ks=(1, 2, 3, 5, 10)) -> pd.DataFrame:
    """For each k: the directed kNN graph's size and components, and the mutual graph's edges, components and
    isolated documents."""
    rows = []
    for k in ks:
        s = summary(knn_graph(X, ids, k=k))
        m = summary(knn_graph(X, ids, k=k, mutual=True))
        rows.append({"k": k, "edges": s["edges"], "density": s["density"], "weak_components": s["weak_components"],
                     "strong_components": s["strong_components"], "largest_weak_share": s["largest_weak_share"],
                     "largest_strong_share": s["largest_strong_share"], "mutual_edges": m["edges"],
                     "mutual_components": m["components"], "mutual_isolated": m["isolated"]})
    return pd.DataFrame(rows)


def hubs(G, n: int = 10) -> pd.DataFrame:
    """The n nodes with the most edges (ties in alphabetical order)."""
    ranked = sorted(G.degree(), key=lambda kv: (-kv[1], str(kv[0])))[:n]
    return pd.DataFrame(ranked, columns=["node", "degree"])


# =============================================================================================
# Drawing
# =============================================================================================
def draw(G, pos=None, ax=None, title: str = "", boxes: bool = False, font_size: int = 10, edge_labels=None,
         node_color="#cfe3f5"):
    """Draw a graph with labelled nodes; boxes=True writes the labels in boxes instead of circles,
    edge_labels="weight" (or another attribute) labels the edges. Arrows are drawn for a directed graph."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 4))
    pos = pos or nx.spring_layout(G, seed=SEED)
    kw = dict(ax=ax, edge_color="#7a8899")
    if G.is_directed():
        kw.update(arrows=True, arrowsize=14, connectionstyle="arc3,rad=0.08", node_size=900)
    nx.draw_networkx_edges(G, pos, **kw)
    if boxes:
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size,
                                bbox=dict(boxstyle="round,pad=0.3", fc=node_color, ec="#5a6b7d"))
    else:
        nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_color, edgecolors="#5a6b7d", node_size=900)
        nx.draw_networkx_labels(G, pos, ax=ax, font_size=font_size)
    if edge_labels:
        labels = {(u, v): d[edge_labels] for u, v, d in G.edges(data=True) if edge_labels in d}
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, ax=ax, font_size=font_size - 1)
    ax.set_title(title)
    ax.axis("off")
    return ax


def draw_bipartite(B, title: str = "", ax=None):
    """Draw a bipartite term-document graph: documents on the left, terms on the right."""
    import matplotlib.pyplot as plt
    docs = sorted(n for n, s in B.nodes(data="side") if s == "doc")
    terms = sorted(n for n, s in B.nodes(data="side") if s == "term")
    pos = {d: (0.0, -i * (len(terms) - 1) / max(len(docs) - 1, 1)) for i, d in enumerate(docs)}
    pos.update({t: (1.0, -i) for i, t in enumerate(terms)})
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))
    nx.draw_networkx_edges(B, pos, ax=ax, edge_color="#9aa7b4")
    nx.draw_networkx_nodes(B, pos, nodelist=docs, ax=ax, node_color="#f6d7a7", edgecolors="#8a6d3b", node_size=700)
    nx.draw_networkx_labels(B, {d: pos[d] for d in docs}, labels={d: d for d in docs}, ax=ax, font_size=10)
    nx.draw_networkx_labels(B, {t: (pos[t][0] + 0.04, pos[t][1]) for t in terms}, labels={t: t for t in terms},
                            ax=ax, font_size=10,
                            horizontalalignment="left",
                            bbox=dict(boxstyle="round,pad=0.25", fc="#cfe3f5", ec="#5a6b7d"))
    ax.set_xlim(-0.2, 1.5)
    ax.set_title(title)
    ax.axis("off")
    return ax


# =============================================================================================
# Practice checks
# =============================================================================================
def _test_graphs():
    terms = project(bipartite_graph(miniature_tokens()), "term")
    two = nx.Graph([("a", "b"), ("b", "c"), ("c", "d"), ("e", "f")])
    directed = nx.DiGraph([("a", "b"), ("b", "a"), ("b", "c"), ("c", "d"), ("d", "c"), ("d", "e")])
    X, ids, _ = tfidf(miniature_tokens(), min_df=1)
    k1 = knn_graph(X, ids, k=1)
    return [("miniature term graph", terms, "vitamin"), ("two-part undirected graph", two, "a"),
            ("directed graph", directed, "a"), ("miniature kNN, k = 1", k1, "D3")]


def _line(ok: bool, text: str) -> None:
    print(("PASS  " if ok else "FAIL  ") + text)


def _not_attempted(fn, value) -> bool:
    if value is None:
        name = getattr(fn, "__name__", "your function")
        print(f"Not attempted yet: {name} returns None.")
        return True
    return False


def _valid_dfs(G, source, order) -> bool:
    """True if order is a valid depth-first order of the nodes reachable from source."""
    reach = set(bfs_hops(G, source))
    if not order or order[0] != source or len(order) != len(set(order)) or set(order) != reach:
        return False
    visited, path = {source}, [source]
    for v in order[1:]:
        while path and v not in set(G.neighbors(path[-1])):
            if any(w not in visited for w in G.neighbors(path[-1])):
                return False
            path.pop()
        if not path:
            return False
        visited.add(v)
        path.append(v)
    return True


def _check_e1(bfs):
    for label, g, s in _test_graphs():
        got = bfs(g, s)
        if _not_attempted(bfs, got):
            return
        want = bfs_hops(g, s)
        ok = isinstance(got, dict) and {k: int(v) for k, v in got.items()} == want
        hint = ""
        if not ok and isinstance(got, dict):
            wrong = sorted((k for k in want if got.get(k) != want[k]), key=str)[:3]
            hint = (f" Wrong or missing: {wrong}. Set a node's hop count when it is first discovered, and use "
                    "graph.neighbors(u).") if wrong else " It contains nodes that cannot be reached from the source."
        _line(ok, f"{label}: hop counts from {s!r} are right ({len(want)} nodes reached)." if ok
              else f"{label}: hop counts from {s!r} differ from the expected ones.{hint}")


def _check_e2(dfs):
    for label, g, s in _test_graphs():
        got = dfs(g, s)
        if _not_attempted(dfs, got):
            return
        n = len(bfs_hops(g, s))
        ok = isinstance(got, list) and _valid_dfs(g, s, got)
        _line(ok, f"{label}: {n} nodes in a valid depth-first order from {s!r}." if ok else
              f"{label}: not a valid depth-first order from {s!r}. Every node reachable from the source must appear "
              "once, and the search must continue from the most recently visited node that still has an unvisited "
              "neighbour.")


def _as_partition(comps):
    return sorted(sorted(map(str, c)) for c in comps)


def _check_e3(weak, strong=None):
    graphs = _test_graphs()
    for label, g, _ in graphs:
        got = weak(g)
        if _not_attempted(weak, got):
            return
        want = weak_components(g)
        ok = _as_partition(got) == _as_partition(want)
        _line(ok, f"{label}: {len(want)} weak component(s)." if ok else
              f"{label}: expected {len(want)} weak component(s), got {len(got)}. Ignore direction, and start a new "
              "search from every node that is not yet in a component.")
    if strong is None:
        return
    for label, g, _ in graphs:
        if not g.is_directed():
            continue
        got = strong(g)
        if _not_attempted(strong, got):
            return
        want = strong_components(g)
        ok = _as_partition(got) == _as_partition(want)
        _line(ok, f"{label}: {len(want)} strong component(s)." if ok else
              f"{label}: expected {len(want)} strong component(s), got {len(got)}. A node's strong component is what it "
              "reaches along the arrows intersected with what reaches it (a search on graph.reverse()).")


def _check_e4(dijkstra_fn):
    for method, s in (("-log", "D1"), ("1-s", "D3")):
        g = add_distances(miniature_similarity_graph(), method)
        got = dijkstra_fn(g, s)
        if _not_attempted(dijkstra_fn, got):
            return
        want, _ = dijkstra(g, s)
        try:
            dist, prev = got
            ok = set(dist) == set(want) and all(abs(float(dist[v]) - want[v]) < 1e-6 for v in want)
            ok = ok and prev.get(s) is None and all(prev.get(v) is not None for v in want if v != s)
        except Exception:
            ok = False
        _line(ok, f"miniature, {method} distances, from {s}: all shortest distances right." if ok else
              f"miniature, {method} distances, from {s}: some distances are wrong. Pop the smallest (distance, node) "
              "pair, skip nodes already final, and relax each edge with graph[u][v][weight]; return (dist, prev).")


_E5 = {"cooccurrence": ("min_lift", ["components", "density", "edges", "min_lift", "nodes"], -1),
       "knn": ("k", ["density", "edges", "k", "strong_components", "weak_components"], +1)}


def _check_e5(table, kind="cooccurrence"):
    if table is None:
        print("Not attempted yet: complete the cell above so that the sweep table is defined and returns a value, "
              "then run this check again.")
        return
    setting, columns, direction = _E5[kind]
    missing = [c for c in columns if c not in getattr(table, "columns", [])]
    _line(not missing, f"columns {columns} present." if not missing else f"missing columns: {missing}.")
    if missing:
        return
    t = table.sort_values(setting)
    _line(len(t) >= 3, f"{len(t)} settings (at least three are needed to see a trend)." if len(t) >= 3 else
          f"only {len(t)} setting(s): use at least five values of {setting}.")
    diffs = np.diff(t["edges"].to_numpy(dtype=float)) * direction
    ok = len(t) >= 2 and bool(np.all(diffs >= 0))
    _line(ok, "edges change in the expected direction as the setting grows." if ok else
          ("edges should not grow as the lift threshold grows." if kind == "cooccurrence"
           else "edges should grow as k grows."))
    if "nodes" in t:
        n = t["nodes"].to_numpy(dtype=float)
    else:   # infer the number of nodes from density and edges; it must be the same in every row
        d = t["density"].to_numpy(dtype=float)
        n = np.array([(1 + math.sqrt(1 + 4 * m / x)) / 2 if x > 0 else float("nan")
                      for m, x in zip(t["edges"].to_numpy(dtype=float), d)])
        n = np.full_like(n, np.nanmedian(n).round())
    pairs = n * (n - 1) if kind == "knn" else n * (n - 1) / 2
    expect = t["edges"].to_numpy(dtype=float) / pairs
    ok = bool(np.allclose(t["density"].to_numpy(dtype=float), expect, rtol=1e-3, atol=1e-6))
    _line(ok, "density agrees with the number of nodes and edges." if ok else
          "density does not agree with nodes and edges: density = edges / (n(n-1)/2) for an undirected graph and "
          "edges / (n(n-1)) for a directed one.")


# =============================================================================================
# Setup: the environment, the database, and the self-check of the setup notebook
# =============================================================================================
def check_environment() -> list:
    """One PASS or FAIL line each for Python, NetworkX, scikit-learn, SciPy, NLTK and the Neo4j driver."""
    from ...core import environment
    return environment.environment(("NetworkX", "scikit-learn", "SciPy", "NLTK", "Neo4j driver"))


def check_neo4j(credentials=None) -> list:
    """Check the Neo4j driver and, with a credentials file next to the notebook (or NEO4J_* environment
    variables), that your database answers: PASS, SKIP (no credentials yet) or FAIL with a hint.
    The password is read from the file and never printed."""
    from ...core import environment
    return environment.neo4j(credentials)


# The five-node graph of the setup notebook (the notebook writes the same edges out).
TOY_EDGES = [("A", "B"), ("A", "C"), ("B", "C"), ("C", "D"), ("D", "E")]
TOY_ARCS = [("A", "B"), ("B", "C"), ("C", "A"), ("C", "D"), ("D", "E")]

_SELF_CHECK = {
    "Q1": ("the sum of the degrees in the undirected toy graph", lambda: 2 * len(TOY_EDGES),
           "Every edge has two ends and adds 1 to the degree of each, so the degrees add up to twice the number "
           "of edges (the handshake lemma)."),
    "Q2": ("the number of 1s in the undirected toy graph's adjacency matrix", lambda: 2 * len(TOY_EDGES),
           "An undirected edge u-v is written twice, at (u, v) and at (v, u): the matrix is symmetric."),
    "Q3": ("the node of the directed toy graph with no outgoing arc", lambda: "E",
           "E has an arc coming in (from D) but none going out: its out-degree is 0, its in-degree 1."),
    "Q4": ("what the number of edges of one abstract's star counts (a, b or c)", lambda: "b",
           "An edge joins the abstract to each distinct term it contains once preprocessed, so the star has "
           "as many edges as distinct terms; repeated words add no edge (a), and other abstracts are not in "
           "the star (c)."),
    "Q5": ("how much of SciFact's term-document matrix is filled (a above 10%, b between 1% and 10%, c below 1%)",
           lambda: "c",
           "Each abstract uses about 85 of the 44,146 distinct terms (0.19% of a row), so well under 1% of the matrix is "
           "filled: the matrix is sparse, which is why graphs and sparse matrices store only the non-zeros."),
}


def self_check(question: str, answer=None) -> None:
    """The setup notebook's predict-then-check questions: self_check("Q1", 10). Says whether the prediction
    was right and why."""
    key = str(question).upper()
    if key not in _SELF_CHECK:
        raise KeyError(f"No question {question!r}; the questions are {', '.join(_SELF_CHECK)}.")
    what, expected, why = _SELF_CHECK[key]
    if answer is None:
        print(f"Not attempted yet: write your prediction of {what} in the cell, then run it again.")
        return
    right = expected()
    ok = str(answer).strip().lower() == str(right).lower()
    print(("PASS  " if ok else "Not yet: ") + f"{what} is {right}. {why}")


def check(exercise: str, *args, **kwargs) -> None:
    """Check a practice exercise: one PASS or FAIL line per fact, with a hint when something fails.
    check("E1", bfs), check("E2", dfs), check("E3", weak_components, strong=strong_components),
    check("E4", dijkstra), check("E5", table, kind="cooccurrence" or "knn")."""
    checks = {"E1": _check_e1, "E2": _check_e2, "E3": _check_e3, "E4": _check_e4, "E5": _check_e5}
    key = str(exercise).upper()
    if key not in checks:
        raise KeyError(f"No exercise {exercise!r}; the exercises are {', '.join(checks)}.")
    checks[key](*args, **kwargs)
