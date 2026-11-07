"""Paths, connectivity and structure: what a shortest path, a reachable set and a structural measure mean.

A topic without graded tasks: its notebooks check themselves with prediction-then-check cells, and the
practice notebook compares the student's connectivity summary, path comparison and list of unreachable users
with the kit's. The kit supplies the helpers (T = kit.tools()), two hand-checkable graphs (paths_toy with a
distance and an interaction frequency on every link; reply_toy, directed and disconnected), StudyHub (a
fictional course forum of 300 users generated with a fixed seed) and, from the text-to-graphs topic, SciFact's
kNN document graph for a real degree distribution.

Mathematical depth is offered on request: every concept has the idea, the mechanics (the rule or formula with
the toy's numbers) and, optionally, going deeper (why it works, costs and limits). Notebook cells headed
'Optional' are the last stage.
"""
from ...core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

TOPIC = "paths-and-connectivity"
TITLE = "Paths, connectivity and structure"
LEVEL = "intermediate"
BUILDS_ON = ("text-to-graphs",)       # graph representations, BFS, components and Dijkstra are assumed
PROBLEM = None                      # no case problem: two toy graphs, the StudyHub forum and SciFact
TASKS = {}                          # no answer book: the notebooks check themselves
TOOLS = ".tools"                    # T = kit.tools()
DATASETS = ("scifact",)             # the kNN document graph of text-to-graphs gives a real degree distribution
EXTRAS = "graphs"

_DEPTH = ("The learner has worked through the text-to-graphs topic (graph representations, BFS, DFS, weak and "
          "strong components, Dijkstra with similarities turned into distances). Explain in three stages and stop at the "
          "first unless the student asks for more or the cell itself shows the formula: (1) the idea in words, with "
          "paths_toy, reply_toy or a StudyHub user; (2) the mechanics, the rule or formula with every symbol defined "
          "and the toy's numbers; (3) going deeper, why it works and what it costs. Cells headed 'Optional' are stage "
          "3; never suggest that skipping them is a failure. Always state what an edge means and what its cost means "
          "before giving a path or a reachability result.")

STAGES = {
    "data": stage("The StudyHub forum (fictional, generated with a fixed seed): users table and replies table", BLACK_BOX,
                  how="T.forum(), T.forum_graph(), T.default_start()",
                  note="say that the forum is invented and what an edge means: author -> replier, the replier replied "
                       "to the author's posts, so information flows from author to replier"),
    "toys": stage("paths_toy (km and messages per week on each link) and reply_toy (directed, disconnected)", FOCUS,
                  how="T.paths_toy(), T.reply_toy(), T.draw_paths_toy(), T.draw_reply_toy(), T.bfs_trace(), "
                      "T.bfs_hops()", note=_DEPTH),
    "path_meaning": stage("What an edge cost means: hops, a distance (km) and an interaction frequency turned into a cost "
                          "(1/f); why a frequency cannot be used as a cost; ties; keeping the path definition with the result",
                          FOCUS, how="T.describe_path(), T.frequency_to_cost(), T.add_costs(), T.path_cost(), T.shortest_route(), "
                              "T.all_shortest()", note=_DEPTH),
    "dijkstra": stage("Dijkstra's algorithm, recalled", ON_REQUEST, how="networkx shortest_path(weight=...)",
                      taught_in="the text-to-graphs topic", note="recall the idea; explain the code only when asked"),
    "reachability": stage("Reachability along directed edges: reachable sets, who can reach a node, weak versus strong "
                          "connectivity, why weak connectivity does not guarantee a directed path, unreachable users and "
                          "the reasons", FOCUS, how="T.reachable_from(), T.can_reach(), T.explain_unreachable()", note=_DEPTH),
    "condensation": stage("The condensation of a directed graph (strong components as nodes form a DAG)", ON_REQUEST,
                          how="networkx.condensation", note="optional depth"),
    "views": stage("Graph views: directed, undirected (either replied), reciprocal (both replied), strong ties; what each "
                   "conversion drops", FOCUS, how="T.view(), T.structure_table()", note=_DEPTH),
    "structure": stage("Degree (in, out, total), strength, density, local and average clustering, transitivity, and a "
                       "random baseline", FOCUS, how="T.structure_row(), T.random_baseline_clustering()", note=_DEPTH),
    "distributions": stage("Degree distributions: histogram, log-log histogram, CCDF; hubs and heavy tails; why a "
                           "straight-looking log-log plot does not prove a power law", FOCUS,
                           how="T.degree_counts(), T.ccdf(), T.plot_distribution(), T.scifact_knn_indegree(), "
                               "T.load_scifact()", note=_DEPTH),
    "tail_fit": stage("Fitting and comparing power-law and log-normal tails (Clauset, Shalizi and Newman)", ON_REQUEST,
                      how="T.tail_fit()", note="optional depth: maximum likelihood, likelihood ratio, Vuong's test; "
                                                "keep to the idea unless asked"),
    "generator": stage("How the forum data were generated", BLACK_BOX,
                       note="do not reveal the generator's planted groups; the student should find them from the graph"),
    "drawing": stage("Drawing graphs with Matplotlib", BLACK_BOX, how="T.draw()"),
    "checks": stage("The setup notebook's self-check and the practice notebook's checks", BLACK_BOX,
                    how="T.quiz(), T.check(), T.reference_summary(), T.reference()",
                    note="never give the solution; explain the hint a FAIL line gives"),
    "centrality": stage("Betweenness, closeness, eigenvector centrality, PageRank and HITS", OUT_OF_SCOPE,
                        taught_in="the centrality-and-link-analysis topic"),
}
