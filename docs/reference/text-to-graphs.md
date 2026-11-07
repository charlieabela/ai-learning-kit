# From text to graphs

`text-to-graphs` · level **intermediate** · builds on no other topic · pip extra `graphs`

Text to graphs: turning a corpus into graphs and exploring them (term-document graphs and their projections, co-occurrence with count and lift thresholds, k-nearest-neighbour document graphs, traversal, components and shortest paths, and what construction and preprocessing choices do to the result).

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("text-to-graphs")
T = kit.tools()
```

Written for the **intermediate** level: Learners who are comfortable with basic Python and pandas and know the fundamentals of data analysis, but are new to the methods of this topic.

**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it is kept):

- `scifact`: SciFact (Allen Institute for AI): corpus of 5,183 abstracts and the claims, from Allen Institute for AI (Wadden et al.), GitHub allenai/scifact. The dataset card lists CC BY-NC 2.0 (non-commercial use with attribution). The kit downloads the archive from the Allen Institute's address rather than copying it.

## The three layers

### In focus: learners do these themselves

- Graphs from text: the bipartite term-document graph, its projections onto terms (co-occurrence) and documents (shared terms), directed against undirected, weighted, and adjacency lists, matrices and edge lists
- Breadth-first and depth-first search: queues and stacks, hop counts, visiting orders (with `T.bfs_hops(), T.bfs_order(), T.dfs_order()`)
- Connected components; weak and strong components of a directed graph; isolated nodes (with `T.weak_components(), T.strong_components()`)
- Shortest paths: fewest hops with BFS, Dijkstra's algorithm with a priority queue, and why a similarity must be turned into a distance first (1 - s, -ln s, 1/s) (with `T.dijkstra(), T.add_distances(), T.path_to()`)
- Construction choices: choosing concepts by document frequency, count thresholds against lift, k in a kNN graph, directed, symmetric and mutual kNN graphs, similarity thresholds, and how each changes size, density and components (with `T.cooccurrence_graph(), T.knn_graph(), T.summary(), T.sweep_cooccurrence(), T.sweep_knn()`)
- What preprocessing choices (stop words, hyphenated terms, stemming) do to the concept graph (with `T.corpus_tokens(variant), T.tokens_from_file()`)

### On request: the kit does these, and learners may look inside

- Tokenisation, stop-word removal and TF-IDF weighting themselves (with `T.tokenize(), T.tfidf()`)
- Tarjan's and Kosaraju's algorithms for strong components

### Support: the kit does these for the learner

- Downloading and reading SciFact (with `kit.start, T.load_scifact()`)
- Drawing graphs with Matplotlib (with `T.draw(), T.draw_bipartite()`)
- Checking the libraries and the Neo4j connection before the topic starts (with `T.check_environment(), T.check_neo4j()`)

### Left out of this topic

- Centrality measures, communities and link analysis (PageRank) (taught in the centrality-and-link-analysis topic; communities in later graph-analysis topics)

## Helpers: `T = kit.tools()`

Text to graphs: helpers returned by T = kit.tools() after kit.start("text-to-graphs").

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`T.corpus_tokens(variant: str = 'reference') -> dict`**: {doc_id: [tokens]} for SciFact (title and abstract) from the reference pipeline or one of its variants.
- **`T.tokens_from_file(path) -> dict`**: Your own preprocessing output as {doc_id: [tokens]}. Accepted shapes: .json {doc_id: [tokens]} (or {doc_id: "tokens separated by spaces"}) .jsonl one object per line with doc_id (or _id) and tokens (a list or a string) .csv / .tsv columns doc_id and tokens, the tokens separated by spaces .pkl / .pickle a dict {doc_id: [tokens]}
- **`T.cooccurrence_graph(tok: dict, top_n: int | None = 300, terms=None, min_weight: int = 1, min_lift: float | None = None, window: int | None = None) -> nx.Graph`**: The concept co-occurrence graph: the chosen terms (the top_n by document frequency, or the terms given), linked when they occur in the same document, with weight = the number of such documents and lift = weight / (df(u) x df(v) / N). Edges need weight >= min_weight and, if given, lift >= min_lift. window=w counts two terms only when they occur within w consecutive tokens of each other.
- **`T.knn_graph(X, ids, k: int = 5, symmetric: bool = False, mutual: bool = False) -> nx.Graph`**: Each document points to its k most similar documents (cosine similarity of the rows of X; only positive similarities; ties broken by position). Edge attribute similarity. symmetric=True: undirected, an edge when either document lists the other. mutual=True: undirected, an edge only when both list each other.
- **`T.add_distances(G, method: str = '1-s')`**: Turn each edge's similarity into a distance, stored as the edge attribute distance: "1-s" (1 - s), "-log" (-ln s) or "1/s". Returns the same graph.
- **`T.bfs_hops(G, source) -> dict`**: Breadth-first search: {node: number of edges on a shortest path from source}, for reachable nodes.
- **`T.bfs_order(G, source) -> list`**: The order in which breadth-first search discovers the nodes reachable from source.
- **`T.dfs_order(G, source) -> list`**: The order in which depth-first search visits the nodes reachable from source (neighbours in sorted order, always continuing from the node reached last).
- **`T.weak_components(G) -> list`**: Connected components (for a directed graph, ignoring direction), largest first.
- **`T.strong_components(G) -> list`**: Strongly connected components of a directed graph, largest first: the nodes reachable from s intersected with the nodes that can reach s, repeated for every node not yet assigned.
- **`T.dijkstra(G, source, weight: str = 'distance')`**: Dijkstra's algorithm with a priority queue. Returns (dist, prev): dist[v] is the length of a shortest path from source, prev[v] the node before v on it. Every edge weight must be zero or positive.
- **`T.path_to(prev: dict, target) -> list`**: The path from the source to target, read back from a prev dictionary ([] if target was not reached).
- **`T.summary(G) -> dict`**: Size, density, mean degree, isolated nodes and components of a graph (weak and strong for a directed graph).
- **`T.sweep_cooccurrence(tok: dict, thresholds=(1, 2, 5, 10, 20, 50, 100), top_n: int = 300, by: str = 'weight', min_weight: int = 1) -> pd.DataFrame`**: summary() of the concept graph for each threshold: by="weight" varies the count threshold, by="lift" the lift threshold (with edges also needing min_weight shared documents).
- **`T.sweep_knn(X, ids, ks=(1, 2, 3, 5, 10)) -> pd.DataFrame`**: For each k: the directed kNN graph's size and components, and the mutual graph's edges, components and isolated documents.

### On request (look inside)

- **`T.tokenize(text: str, variant: str = 'reference') -> list`**: One text as tokens. reference: lower case, terms of at least two characters with at least one letter (so numbers such as 1 or 2019 are dropped, but il-6 and p53 are kept), hyphenated terms kept whole, scikit-learn's English stop words removed, no stemming. The variants change one choice each: keep_stopwords, split_hyphens (anti-inflammatory becomes anti, inflammatory) or porter (adds the Porter stemmer).
- **`T.tfidf(tok: dict, min_df: int = 2)`**: TF-IDF vectors as in scikit-learn's TfidfVectorizer (smoothed idf, l2-normalised rows) for documents already split into tokens. Returns (X sparse matrix, ids, vocabulary), rows in the order of tok.

### Support

- **`T.load_scifact() -> pd.DataFrame`**: SciFact's corpus: one row per abstract (doc_id as text, title, abstract as one string).
- **`T.draw(G, pos=None, ax=None, title: str = '', boxes: bool = False, font_size: int = 10, edge_labels=None, node_color='#cfe3f5')`**: Draw a graph with labelled nodes; boxes=True writes the labels in boxes instead of circles, edge_labels="weight" (or another attribute) labels the edges. Arrows are drawn for a directed graph.
- **`T.draw_bipartite(B, title: str = '', ax=None)`**: Draw a bipartite term-document graph: documents on the left, terms on the right.
- **`T.check_environment() -> list`**: One PASS or FAIL line each for Python, NetworkX, scikit-learn, SciPy, NLTK and the Neo4j driver.
- **`T.check_neo4j(credentials=None) -> list`**: Check the Neo4j driver and, with a credentials file next to the notebook (or NEO4J_* environment variables), that your database answers: PASS, SKIP (no credentials yet) or FAIL with a hint. The password is read from the file and never printed.

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.miniature() -> pd.DataFrame`**: The six miniature abstracts: doc_id, title and their four index terms.
- **`T.miniature_tokens() -> dict`**: {doc_id: [terms]} for the miniature.
- **`T.miniature_positions(kind: str = 'terms') -> dict`**: Fixed positions for drawing the miniature: kind "terms" or "docs".
- **`T.miniature_similarity_graph() -> nx.Graph`**: The miniature's documents linked when their TF-IDF cosine similarity is positive (rounded to 3 places).
- **`T.load_claims() -> pd.DataFrame`**: SciFact's claims (train and dev, with their evidence document ids).
- **`T.document_frequency(tok: dict) -> Counter`**: How many documents contain each term.
- **`T.concepts(tok: dict, top_n: int = 300) -> list`**: The top_n terms that occur in the most documents (ties in alphabetical order).
- **`T.tfidf_from_file(path)`**: Your own TF-IDF weights, a JSON file {doc_id: {term: weight}}, as (X, ids, vocabulary) with each row scaled to length 1, so that dot products are cosine similarities.
- **`T.similarity_matrix(X) -> np.ndarray`**: Cosine similarity of every pair of rows of an l2-normalised matrix, as a dense array.
- **`T.bipartite_graph(tok: dict) -> nx.Graph`**: Documents and terms as two kinds of node (attribute side = "doc" or "term"); an edge when a document contains a term.
- **`T.project(B: nx.Graph, side: str = 'term') -> nx.Graph`**: Keep one kind of node (side "term" or "doc") and link two that share a neighbour on the other side, with weight = the number of shared neighbours.
- **`T.lift(tok: dict, a: str, b: str) -> dict`**: Observed co-occurrences of two terms against those expected if they were unrelated: lift = observed / (df(a) x df(b) / N).
- **`T.threshold_graph(X, ids, min_sim: float = 0.3) -> nx.Graph`**: Documents linked when their cosine similarity is at least min_sim (undirected).
- **`T.bfs_tree(G, source) -> dict`**: {node: the node from which breadth-first search first reached it} (source: None), for path_to().
- **`T.hubs(G, n: int = 10) -> pd.DataFrame`**: The n nodes with the most edges (ties in alphabetical order).
- **`T.self_check(question: str, answer=None) -> None`**: The setup notebook's predict-then-check questions: self_check("Q1", 10). Says whether the prediction was right and why.
- **`T.check(exercise: str, *args, **kwargs) -> None`**: Check a practice exercise: one PASS or FAIL line per fact, with a hint when something fails. check("E1", bfs), check("E2", dfs), check("E3", weak_components, strong=strong_components), check("E4", dijkstra), check("E5", table, kind="cooccurrence" or "knn").

### Constants

`T.DISTANCES` (dict), `T.MINIATURE` (list), `T.ROOT` (Path), `T.SEED` (int), `T.TOY_ARCS` (list), `T.TOY_EDGES` (list), `T.VARIANTS` (tuple)
