"""Recompute the reference results of the paths-and-connectivity topic (reference/results.json).

Run from a checkout: python -m ai_learning_kit.topics.paths_and_connectivity.make_reference
Every number quoted on the slides and in the student notes comes from this file.
"""
import json

import networkx as nx
import numpy as np

from . import tools as T


def run() -> dict:
    out = {"version": T.VERSION}
    # ----- paths_toy
    G = T.add_costs(T.paths_toy())
    out["paths_toy"] = {w or "hops": T.describe_path(G, "Amy", "Hana", w) for w in (None, "km", "cost", "msgs")}
    out["paths_toy_ties"] = {"Ben-Fabio": T.all_shortest(G, "Ben", "Fabio")}
    out["paths_toy_bfs_layers_from_Amy"] = {}
    for n, h in nx.single_source_shortest_path_length(G, "Amy").items():
        out["paths_toy_bfs_layers_from_Amy"].setdefault(str(h), []).append(n)
    for k in out["paths_toy_bfs_layers_from_Amy"]:
        out["paths_toy_bfs_layers_from_Amy"][k].sort()
    # every route Amy -> Hana under the three costs, for the comparison table
    rows = []
    for p in nx.all_simple_paths(G, "Amy", "Hana"):
        rows.append({"route": " > ".join(p), "edges": len(p) - 1, "km": T.path_cost(G, p, "km"),
                     "cost": round(T.path_cost(G, p, "cost"), 4), "msgs_sum": T.path_cost(G, p, "msgs")})
    out["paths_toy_all_routes"] = sorted(rows, key=lambda r: (r["edges"], r["km"]))
    # ----- reply_toy
    R = T.reply_toy()
    out["reply_toy"] = {"reachable_from_Amy": sorted(T.reachable_from(R, "Amy")),
                        "can_reach_Amy": sorted(T.can_reach(R, "Amy")),
                        "unreachable": T.explain_unreachable(R, "Amy").to_dict(orient="records"),
                        "weak_components": [sorted(c) for c in nx.weakly_connected_components(R)],
                        "strong_components": [sorted(c) for c in nx.strongly_connected_components(R)],
                        "in_degree": dict(R.in_degree()), "out_degree": dict(R.out_degree())}
    # ----- the forum
    users, replies = T.forum()
    F = T.forum_graph()
    start = T.default_start()
    out["forum"] = {"users": len(users), "rows": len(replies), "self_loop_rows": int((replies.author == replies.replier).sum()),
                    "roles": users.role.value_counts().to_dict(),
                    "replies_total": int(replies.replies.sum()),
                    "replies_per_pair": {k: int(v) for k, v in replies.replies.value_counts().sort_index().items()},
                    "start": start,
                    "summary": T.reference_summary(T.forum_graph(include_users=False), users.user, start)}
    ex = T.explain_unreachable(F, start, users.user)
    out["forum"]["unreachable_counts"] = ex.reason.value_counts().to_dict()
    out["forum"]["unreachable_examples"] = {r: ex[ex.reason == r].user.head(3).tolist() for r in T.REASONS}
    out["forum"]["can_reach_start"] = len(T.can_reach(F, start))
    reach = sorted(T.reachable_from(F, start))
    cmp_rows = []
    for t in reach:
        h = nx.shortest_path(F, start, t)
        c = nx.shortest_path(F, start, t, weight="inv_replies")
        cmp_rows.append({"target": t, "hops_edges": len(h) - 1, "strength_edges": len(c) - 1,
                         "strength_cost": round(nx.path_weight(F, c, "inv_replies"), 4)})
    hops = np.array([r["hops_edges"] for r in cmp_rows]); st = np.array([r["strength_edges"] for r in cmp_rows])
    out["forum"]["path_comparison"] = {"reachable": len(reach), "longer_strength_route": int((st > hops).sum()),
                                       "hops_distribution": {int(k): int(v) for k, v in zip(*np.unique(hops, return_counts=True))},
                                       "strength_edges_distribution": {int(k): int(v) for k, v in zip(*np.unique(st, return_counts=True))},
                                       "max_extra_edges": int((st - hops).max())}
    ex_row = max(cmp_rows, key=lambda r: (r["strength_edges"] - r["hops_edges"], r["target"]))
    t = ex_row["target"]
    out["forum"]["example_pair"] = {"target": t, "hops_route": nx.shortest_path(F, start, t),
                                    "strength_route": nx.shortest_path(F, start, t, weight="inv_replies"),
                                    "hops_route_replies": [F[u][v]["replies"] for u, v in zip(nx.shortest_path(F, start, t)[:-1], nx.shortest_path(F, start, t)[1:])],
                                    "strength_route_replies": [F[u][v]["replies"] for u, v in zip(nx.shortest_path(F, start, t, weight="inv_replies")[:-1], nx.shortest_path(F, start, t, weight="inv_replies")[1:])]}
    # structure on views
    out["forum"]["structure"] = [{k: (None if isinstance(v, float) and v != v else v) for k, v in row.items()}
                                 for row in T.structure_table(F).to_dict(orient="records")]
    out["forum"]["random_baseline_clustering"] = T.random_baseline_clustering(T.view(F, "undirected"))
    od, idg = dict(F.out_degree()), dict(F.in_degree())
    U = T.view(F, "undirected")
    out["forum"]["top_out"] = sorted(od.items(), key=lambda x: (-x[1], x[0]))[:6]
    out["forum"]["top_in"] = sorted(idg.items(), key=lambda x: (-x[1], x[0]))[:6]
    out["forum"]["top_undirected"] = sorted(dict(U.degree()).items(), key=lambda x: (-x[1], x[0]))[:6]
    out["forum"]["bot"] = {"out": od["studyhub_announcements"], "in": idg["studyhub_announcements"],
                           "undirected": U.degree("studyhub_announcements"),
                           "clustering_undirected": round(nx.clustering(U, "studyhub_announcements"), 4)}
    vals = [v for v in od.values() if v > 0]
    out["forum"]["out_degree"] = {"mean": round(float(np.mean(list(od.values()))), 3), "median": float(np.median(list(od.values()))),
                                  "max": int(max(od.values())), "zero": sum(1 for v in od.values() if v == 0),
                                  "share_of_edges_top10pct": round(sum(sorted(od.values(), reverse=True)[:len(od) // 10]) / F.number_of_edges(), 4),
                                  "tail_fits": [T.tail_fit(vals, xm) for xm in (3, 8)]}
    _, ind = T.scifact_knn_indegree(5)
    iv = list(ind.values())
    out["scifact_knn5_indegree"] = {"mean": round(float(np.mean(iv)), 3), "median": float(np.median(iv)), "max": int(max(iv)),
                                    "zero": sum(1 for v in iv if v == 0), "top": sorted(ind.items(), key=lambda x: (-x[1], x[0]))[:5],
                                    "share_at_least_20": round(sum(1 for v in iv if v >= 20) / len(iv), 4),
                                    "tail_fits": [T.tail_fit([v for v in iv if v > 0], xm) for xm in (5, 12, 20)]}
    return out


if __name__ == "__main__":
    res = run()
    (T.ROOT / "reference").mkdir(exist_ok=True)
    (T.ROOT / "reference" / "results.json").write_text(json.dumps(res, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, indent=1, default=str)[:6000])
