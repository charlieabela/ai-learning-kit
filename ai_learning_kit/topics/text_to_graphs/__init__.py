"""Text to graphs: turning a corpus into graphs and exploring them (term-document graphs and their projections,
co-occurrence with count and lift thresholds, k-nearest-neighbour document graphs, traversal, components and
shortest paths, and what construction and preprocessing choices do to the result).

The topic's notebooks check themselves. They work first on a miniature of six abstracts, each reduced to four
index terms, so that every number can be checked by hand, and then repeat each idea on SciFact (5,183
biomedical abstracts, downloaded from its owner on first use). Learners can bring their own preprocessing
output (tokens, or TF-IDF weights) in place of the kit's reference pipeline.

Every concept is offered at three depths: the idea, the mechanics (a calculation by hand on the miniature) and
going deeper; the notebooks mark the depth cells as optional.
"""
from ...core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

TOPIC = "text-to-graphs"
TITLE = "From text to graphs"
LEVEL = "intermediate"
BUILDS_ON = ()                      # tokenisation, stop words and TF-IDF are assumed (see the "preprocessing" stage)
PROBLEM = None
TASKS = {}
TOOLS = ".tools"
DATASETS = ("scifact",)
EXTRAS = "graphs"                   # networkx, scikit-learn and NLTK (for the Porter variant)

_DEPTH = ("Explain in three stages and stop at the first unless the learner asks for more: (1) the idea in words, "
          "with the miniature; (2) the mechanics, a calculation by hand on the miniature with the printed numbers; "
          "(3) the depth, why the method behaves as it does and what it costs. Cells headed 'Optional' are the "
          "depth stage; never make a learner feel that skipping one is a failure.")

STAGES = {
    "representations": stage("Graphs from text: the bipartite term-document graph, its projections onto terms "
                             "(co-occurrence) and documents (shared terms), directed against undirected, weighted, "
                             "and adjacency lists, matrices and edge lists", FOCUS, note=_DEPTH),
    "traversal": stage("Breadth-first and depth-first search: queues and stacks, hop counts, visiting orders",
                       FOCUS, how="T.bfs_hops(), T.bfs_order(), T.dfs_order()", note=_DEPTH),
    "components": stage("Connected components; weak and strong components of a directed graph; isolated nodes",
                        FOCUS, how="T.weak_components(), T.strong_components()", note=_DEPTH),
    "shortest_paths": stage("Shortest paths: fewest hops with BFS, Dijkstra's algorithm with a priority queue, and "
                            "why a similarity must be turned into a distance first (1 - s, -ln s, 1/s)", FOCUS,
                            how="T.dijkstra(), T.add_distances(), T.path_to()", note=_DEPTH),
    "construction": stage("Construction choices: choosing concepts by document frequency, count thresholds against "
                          "lift, k in a kNN graph, directed, symmetric and mutual kNN graphs, similarity thresholds, "
                          "and how each changes size, density and components", FOCUS,
                          how="T.cooccurrence_graph(), T.knn_graph(), T.summary(), T.sweep_cooccurrence(), "
                              "T.sweep_knn()", note=_DEPTH),
    "preprocessing_effect": stage("What preprocessing choices (stop words, hyphenated terms, stemming) do to the "
                                  "concept graph", FOCUS, how="T.corpus_tokens(variant), T.tokens_from_file()"),
    "preprocessing": stage("Tokenisation, stop-word removal and TF-IDF weighting themselves", ON_REQUEST,
                           how="T.tokenize(), T.tfidf()",
                           note="assumed from earlier work on text representation: say what goes in and out; explain "
                                "the code only when asked"),
    "data": stage("Downloading and reading SciFact", BLACK_BOX, how="kit.start, T.load_scifact()"),
    "drawing": stage("Drawing graphs with Matplotlib", BLACK_BOX, how="T.draw(), T.draw_bipartite()"),
    "setup": stage("Checking the libraries and the Neo4j connection before the topic starts", BLACK_BOX,
                   how="T.check_environment(), T.check_neo4j()",
                   note="say what a FAIL line asks the learner to do; never ask for or repeat a password"),
    "efficient_scc": stage("Tarjan's and Kosaraju's algorithms for strong components", ON_REQUEST,
                           note="mentioned in an optional cell; explain at the level of the idea only when asked"),
    "centrality": stage("Centrality measures, communities and link analysis (PageRank)", OUT_OF_SCOPE,
                        taught_in="the centrality-and-link-analysis topic; communities in later graph-analysis "
                                  "topics"),
}
