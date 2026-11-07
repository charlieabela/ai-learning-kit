# Paths, connectivity and structure

`paths-and-connectivity` · level **intermediate** · builds on [`text-to-graphs`](text-to-graphs.md) · pip extra `graphs`

Paths, connectivity and structure: what a shortest path, a reachable set and a structural measure mean.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("paths-and-connectivity")
T = kit.tools()
```

Written for the **intermediate** level: Learners who are comfortable with basic Python and pandas and know the fundamentals of data analysis, but are new to the methods of this topic.

**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it is kept):

- `scifact`: SciFact (Allen Institute for AI): corpus of 5,183 abstracts and the claims, from Allen Institute for AI (Wadden et al.), GitHub allenai/scifact. The dataset card lists CC BY-NC 2.0 (non-commercial use with attribution). The kit downloads the archive from the Allen Institute's address rather than copying it.

## The three layers

### In focus: learners do these themselves

- paths_toy (km and messages per week on each link) and reply_toy (directed, disconnected) (with `T.paths_toy(), T.reply_toy(), T.draw_paths_toy(), T.draw_reply_toy(), T.bfs_trace(), T.bfs_hops()`)
- What an edge cost means: hops, a distance (km) and an interaction frequency turned into a cost (1/f); why a frequency cannot be used as a cost; ties; keeping the path definition with the result (with `T.describe_path(), T.frequency_to_cost(), T.add_costs(), T.path_cost(), T.shortest_route(), T.all_shortest()`)
- Reachability along directed edges: reachable sets, who can reach a node, weak versus strong connectivity, why weak connectivity does not guarantee a directed path, unreachable users and the reasons (with `T.reachable_from(), T.can_reach(), T.explain_unreachable()`)
- Graph views: directed, undirected (either replied), reciprocal (both replied), strong ties; what each conversion drops (with `T.view(), T.structure_table()`)
- Degree (in, out, total), strength, density, local and average clustering, transitivity, and a random baseline (with `T.structure_row(), T.random_baseline_clustering()`)
- Degree distributions: histogram, log-log histogram, CCDF; hubs and heavy tails; why a straight-looking log-log plot does not prove a power law (with `T.degree_counts(), T.ccdf(), T.plot_distribution(), T.scifact_knn_indegree(), T.load_scifact()`)

### On request: the kit does these, and learners may look inside

- Dijkstra's algorithm, recalled (with `networkx shortest_path(weight=...)`; taught in the text-to-graphs topic)
- The condensation of a directed graph (strong components as nodes form a DAG) (with `networkx.condensation`)
- Fitting and comparing power-law and log-normal tails (Clauset, Shalizi and Newman) (with `T.tail_fit()`)

### Support: the kit does these for the learner

- The StudyHub forum (fictional, generated with a fixed seed): users table and replies table (with `T.forum(), T.forum_graph(), T.default_start()`)
- How the forum data were generated
- Drawing graphs with Matplotlib (with `T.draw()`)
- The setup notebook's self-check and the practice notebook's checks (with `T.quiz(), T.check(), T.reference_summary(), T.reference()`)

### Left out of this topic

- Betweenness, closeness, eigenvector centrality, PageRank and HITS (taught in the centrality-and-link-analysis topic)

## Helpers: `T = kit.tools()`

Paths, connectivity and structure: readable helpers.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`T.paths_toy() -> nx.Graph`**: Eight students (invented). An edge means the two know each other. Each edge carries: km : the distance between their homes, a cost (a printed handout passed hand to hand travels it); msgs : messages they exchange per week, an interaction frequency (a strength, not a cost).
- **`T.reply_toy() -> nx.DiGraph`**: Eleven forum users (invented). u -> v means v replied to u at least once, so information u posted reached v. Kurt registered but never posted or replied: he is a node with no edges.
- **`T.draw_paths_toy(G=None, show='both', highlight=None, ax=None, title=None)`**: Draw paths_toy with its edge labels. show: "both" (km and messages per week), "km", "msgs", "cost" (1/msgs) or None. highlight: a route (list of nodes) drawn in orange.
- **`T.draw_reply_toy(R=None, start=None, ax=None, title=None)`**: Draw reply_toy (u -> v: v replied to u). With start, the start is orange, the users it reaches are blue and the users it cannot reach are grey.
- **`T.bfs_trace(G, source, reverse: bool = False) -> pd.DataFrame`**: Breadth-first search written out step by step: at each step the front of the queue is taken out, its neighbours that have not been seen yet are discovered (in alphabetical order) and join the back of the queue. Directed graphs are followed along the edges (reverse=True follows them backwards: who can reach source). Returns one row per step: the node taken out, its hop count, the nodes it discovered and the queue after.
- **`T.frequency_to_cost(f, method: str = '1/f')`**: Turn an interaction frequency (a strength: more is closer) into a cost (more is farther). "1/f" : one over the frequency (20 messages a week cost 0.05, one message costs 1); "-log" : -ln(f / f_max) for a given f_max (pass method=("-log", f_max)); the strongest link costs 0.
- **`T.add_costs(G, source: str = 'msgs', target: str = 'cost', method='1/f')`**: Add a cost attribute computed from a frequency attribute with frequency_to_cost(). Returns G.
- **`T.path_cost(G, path, weight)`**: The sum of an edge attribute along a path (weight=None counts edges).
- **`T.shortest_route(G, source, target, weight=None)`**: One shortest path (fewest edges when weight is None, smallest total weight otherwise), with ties broken by the alphabetical order of the paths, so that every notebook gets the same answer.
- **`T.all_shortest(G, source, target, weight=None)`**: Every shortest path between source and target (there can be several: a tie).
- **`T.describe_path(G, source, target, weight=None, definition=None)`**: A shortest path together with what it means: the route, its length in edges, its total cost and the definition of an edge cost. Keep the definition with the result.
- **`T.reachable_from(G, source) -> set`**: Every node that can be reached from source along the edges (directions respected), source excluded.
- **`T.can_reach(G, target) -> set`**: Every node from which target can be reached (the nodes whose information can arrive at target).
- **`T.explain_unreachable(G, source, users=None) -> pd.DataFrame`**: Every user that cannot be reached from source, with the reason (see REASONS). users: the full list of users (for example the users table's user column); users missing from G are reported as isolated.
- **`T.view(G: nx.DiGraph, kind: str = 'directed', min_replies: int = 1)`**: A view of a directed, weighted interaction graph, each answering a different question. "directed" : as it is (u -> v: v replied to u); "undirected" : an edge when either replied to the other (direction dropped; weights added); "reciprocal" : an edge only when both replied to each other (a two-way relationship); "strong" : directed, keeping only pairs with at least min_replies replies. Every view keeps all nodes, so that isolated users still count.
- **`T.structure_row(H, name: str) -> dict`**: Nodes, edges, density, mean degree, isolated nodes, components and clustering of one view. Clustering is computed on the undirected version, and only reported for undirected views.
- **`T.structure_table(G: nx.DiGraph, min_replies: int = 3) -> pd.DataFrame`**: structure_row() for the four views of G (the strong view keeps pairs with at least min_replies replies).
- **`T.random_baseline_clustering(H: nx.Graph, trials: int = 5, seed: int = 2205) -> float`**: Average clustering of random graphs with the same number of nodes and edges (G(n, m)): what clustering would be if links were placed at random. For a random graph it is about the density.
- **`T.degree_counts(G, kind: str = 'out') -> pd.Series`**: How many nodes have each degree. kind: "out", "in" or "total" (directed graphs), or "total" for undirected.
- **`T.ccdf(values) -> pd.DataFrame`**: The complementary cumulative distribution: for each value x, the share of nodes with degree >= x.
- **`T.plot_distribution(values, title='', axes=None, color='#2F5D8C')`**: Three views of the same degrees: a histogram on linear axes, the same histogram on log-log axes, and the complementary cumulative distribution on log-log axes.
- **`T.scifact_knn_indegree(k: int = 5)`**: In-degrees of SciFact's directed kNN document graph from text-to-graphs (TF-IDF cosine, reference pipeline): how many abstracts list each abstract among their k nearest neighbours. Returns (graph, in-degree dict).

### On request (look inside)

- **`T.tail_fit(values, xmin: int) -> dict`**: Optional depth. Fits two models to the tail (values >= xmin) and compares them, following the approach of Clauset, Shalizi and Newman (2009) in a simplified, continuous approximation: - a power law, p(x) ~ x^-alpha, with the maximum-likelihood estimate alpha = 1 + n / sum ln(x / (xmin - 0.5)); - a log-normal truncated at xmin, fitted by maximum likelihood. R is the log-likelihood ratio (positive favours the power law); p is Vuong's two-sided p-value: a large p means the data cannot tell the two apart. A straight-looking log-log plot is not, by itself, evidence.

### Support

- **`T.draw(G, pos=None, ax=None, labels=True, edge_labels=None, highlight=None, node_size=900, title=None, node_color='#2F5D8C', font_size=10, colors=None, boxes=False)`**: Draw a small graph. Directed edges get arrowheads, and a pair of opposite edges is drawn as two curves so both arrows show. edge_labels: an edge attribute to print on edges. highlight: a list of nodes forming a path. colors: {node: colour}. boxes=True draws each node as a labelled box (better for names) instead of a circle.
- **`T.forum(seed: int = 2205)`**: StudyHub, a fictional course forum (invented users; generated with a fixed seed). Returns (users, replies): users has one row per registered user (user, role: member, tutor or bot); replies has one row per pair (author, replier, replies): replier replied `replies` times to posts of author. Some rows have author == replier (users replying in their own threads).
- **`T.forum_graph(seed: int = 2205, include_users: bool = True, drop_self_loops: bool = True) -> nx.DiGraph`**: The forum as a directed graph: author -> replier means the replier replied to the author's posts, so information the author posted reached the replier. Edge attributes: replies (an interaction frequency) and inv_replies = 1 / replies (a cost for strongest-chain paths). include_users adds users with no edges.
- **`T.default_start(seed: int = 2205) -> str`**: The suggested starting user for the practice notebook: the tutor who replied most.
- **`T.quiz(qid: str, answer=None) -> None`**: Self-check: compare your prediction with the answer, and say where to revisit the topic.
- **`T.reference_summary(G: nx.DiGraph, users, start) -> dict`**: The connectivity summary the practice notebook asks for (P1), computed by the kit.
- **`T.check(exercise: str, answer=None, **more)`**: Check a practice exercise, e.g. T.check("P1", summary). Prints one PASS or FAIL line per fact.
- **`T.reference() -> dict`**: The frozen results of the run used on the slides and in the notes (reference/results.json).

### From other topics

Supplied ready-made here; each is taught in its own topic.

- `T.bfs_hops`, from [`text-to-graphs`](text-to-graphs.md)
- `T.load_scifact`, from [`text-to-graphs`](text-to-graphs.md)

### Constants

`T.BLUE` (str), `T.DEFINITIONS` (dict), `T.GREEN` (str), `T.GREY` (str), `T.NAVY` (str), `T.ORANGE` (str), `T.PATHS_POS` (dict), `T.PURPLE` (str), `T.REASONS` (dict), `T.REPLY_POS` (dict), `T.ROOT` (Path), `T.SEED` (int), `T.VERSION` (str)
