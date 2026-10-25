"""Small research datasets stored in the kit, so they work offline and identically everywhere.

    from ai_learning_kit import datasets
    G = datasets.small("karate")          # a networkx graph built from the stored copy
    datasets.small_catalog()              # names, titles, sizes and citations

Each dataset is a JSON file in small_data/ with its title, source, citation and licence note,
its nodes (with attributes) and its edges (with attributes). The stored copies of the classic
graphs were written from the copies networkx distributes (see _build()), so they match what
nx.karate_club_graph() and friends return, but they no longer depend on the installed networkx
version. Synthetic datasets are stored the same way, with "synthetic": true and the generating
code and seed in "source".
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

FOLDER = Path(__file__).parent / "small_data"

# What --build writes: name -> (networkx function, title, citation, notes)
CLASSICS = {
    "karate": ("karate_club_graph", "Zachary's karate club",
               "Zachary, W. W. (1977). An information flow model for conflict and fission in small groups. "
               "Journal of Anthropological Research, 33(4), 452-473.",
               "34 members, 78 friendships; node attribute 'club' is the faction each member joined after the split "
               "('Mr. Hi' or 'Officer'); edge attribute 'weight' is the number of contexts in which the two interacted."),
    "lesmis": ("les_miserables_graph", "Les Miserables co-appearance network",
               "Knuth, D. E. (1993). The Stanford GraphBase: A Platform for Combinatorial Computing. Addison-Wesley.",
               "77 characters; an edge joins two characters who appear in the same chapter; 'weight' counts the chapters."),
    "florentine": ("florentine_families_graph", "Florentine families (marriage ties)",
                   "Breiger, R. L. and Pattison, P. E. (1986). Cumulated social roles: the duality of persons and their "
                   "algebras. Social Networks, 8(3), 215-256.",
                   "15 Renaissance Florentine families joined by marriage ties (the Pucci family, with no ties, is "
                   "left out, as in networkx)."),
    "davis": ("davis_southern_women_graph", "Davis Southern women (event attendance)",
              "Davis, A., Gardner, B. B. and Gardner, M. R. (1941). Deep South: A Social Anthropological Study of "
              "Caste and Class. University of Chicago Press.",
              "A bipartite graph: 18 women and the 14 social events they attended; node attribute 'bipartite' is 0 for "
              "women and 1 for events."),
}
LICENCE = ("Data from the cited publication, as distributed with networkx (BSD-3-Clause); stored here so that "
           "it is available offline and does not change with the installed networkx version.")


@lru_cache(maxsize=None)
def _record(name: str) -> dict:
    path = FOLDER / f"{name}.json"
    if not path.is_file():
        available = ", ".join(sorted(p.stem for p in FOLDER.glob("*.json")))
        raise KeyError(f"Unknown small dataset {name!r}. Available: {available}")
    return json.loads(path.read_text(encoding="utf-8"))


def catalog() -> list:
    """Every small dataset stored in the kit: name, title, size, citation, licence note."""
    out = []
    for p in sorted(FOLDER.glob("*.json")):
        r = _record(p.stem)
        out.append({"name": p.stem, "title": r["title"], "nodes": len(r["nodes"]), "edges": len(r["edges"]),
                    "directed": r.get("directed", False), "citation": r.get("citation", ""),
                    "licence": r.get("licence", ""), "notes": r.get("notes", ""),
                    "synthetic": r.get("synthetic", False)})
    return out


def small(name: str):
    """The small dataset called name, as a networkx graph (needs networkx: the kit's "graphs" extra)."""
    r = _record(str(name).strip().lower())
    try:
        import networkx as nx
    except ImportError as e:
        raise ImportError("datasets.small() builds a networkx graph: install the kit with the 'graphs' extra.") from e
    g = nx.DiGraph() if r.get("directed") else nx.Graph()
    g.graph.update(r.get("graph", {}))
    for node, attrs in r["nodes"]:
        g.add_node(node, **attrs)
    for u, v, attrs in r["edges"]:
        g.add_edge(u, v, **attrs)
    return g


def _build() -> None:
    """Write small_data/ from the copies networkx distributes (run once by a maintainer)."""
    import networkx as nx
    FOLDER.mkdir(exist_ok=True)
    for name, (func, title, citation, notes) in CLASSICS.items():
        g = getattr(nx, func)()
        record = {"title": title, "citation": citation, "licence": LICENCE, "notes": notes,
                  "source": f"networkx {nx.__version__}, nx.{func}()", "directed": g.is_directed(),
                  "graph": {k: v for k, v in g.graph.items() if isinstance(v, (str, int, float, bool))},
                  "nodes": [[n, dict(a)] for n, a in g.nodes(data=True)],
                  "edges": [[u, v, dict(a)] for u, v, a in g.edges(data=True)]}
        (FOLDER / f"{name}.json").write_text(json.dumps(record, ensure_ascii=False, indent=0) + "\n", encoding="utf-8")
        print(f"{name}: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges")

