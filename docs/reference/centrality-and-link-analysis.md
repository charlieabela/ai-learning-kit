# Centrality and link analysis

`centrality-and-link-analysis` · level **intermediate** · builds on [`text-to-graphs`](text-to-graphs.md), [`paths-and-connectivity`](paths-and-connectivity.md) · pip extra `graphs`

Centrality and link analysis: what "important" means in a graph, and how degree, betweenness, closeness, PageRank and HITS each answer it.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("centrality-and-link-analysis")
T = kit.tools()
```

Written for the **intermediate** level: Learners who are comfortable with basic Python and pandas and know the fundamentals of data analysis, but are new to the methods of this topic.

## The three layers

### In focus: learners do these themselves

- trio (three users) and mention_toy (thirteen users; u -> v: u mentioned v, weight = mentions) (with `T.trio(), T.mention_toy(), T.draw_mention_toy()`)
- What 'important' means: popularity, activity, brokerage, reach, recursive importance; the question decides the measure (with `T.rank_table()`)
- In-degree, out-degree and strength; whether repeated mentions should count (with `T.degree_table()`)
- Betweenness through shortest paths, pair by pair; directed and undirected interpretations; why weights that are strengths must not be used as distances (with `T.betweenness_by_hand(), networkx betweenness_centrality`)
- Closeness on disconnected data: the classic formula, the Wasserman-Faust scaling and harmonic closeness (with `T.closeness_table()`)
- PageRank as probability flow: transition matrix, repeated matrix multiplication, dangling nodes, teleportation and damping; weighted PageRank (with `T.transition_matrix(), T.flow(), T.pagerank_power(), networkx pagerank`)
- HITS hubs and authorities, contrasted with PageRank (with `T.hits_table()`)
- Comparing rankings (Spearman, Kendall, top-k overlap) and perturbing one edge (with `T.compare_rankings(), T.ranks(), T.perturb(), T.rank_shift()`)

### On request: the kit does these, and learners may look inside

- Brandes' algorithm for betweenness and its cost (with `networkx betweenness_centrality`)
- Why PageRank converges: Markov chains, stationary distributions, the Google matrix as a stochastic, irreducible, aperiodic matrix, the eigenvector view and the rate (1 - damping)

### Support: the kit does these for the learner

- StudyHub (the fictional forum of paths-and-connectivity) as an attention graph: u -> v when u replied to v's posts (with `T.attention_graph(), T.forum(), T.default_start()`)
- How the forum data were generated
- Drawing graphs with Matplotlib (with `T.draw()`)
- The setup notebook's self-check and the practice notebook's checks (with `T.quiz(), T.check(), T.reference_ranking(), T.reference()`)

### Left out of this topic

- Using graph scores inside retrieval and ranking documents (taught in later topics on graphs inside retrieval)
- Node embeddings and learned representations (taught in the graph-embeddings topic)

## Helpers: `T = kit.tools()`

Centrality and link analysis: readable helpers.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`T.trio() -> nx.DiGraph`**: Three users: Amy mentions Ben and Clara, Ben mentions Amy, Clara mentions Ben (one mention each).
- **`T.mention_toy() -> nx.DiGraph`**: Thirteen forum users (invented). Edge u -> v: u mentioned v; attribute mentions = how many times.
- **`T.draw_mention_toy(G=None, scores=None, ax=None, title=None, labels: bool = True, highlight=None, colors=None)`**: Draw mention_toy. labels: write the number of mentions on each edge. scores: a dict or Series; the score is written under each name, and the colour shows its third of the range (orange: top, blue: middle, grey: bottom). highlight: a route (list of nodes) in orange.
- **`T.degree_table(G, weight: str = 'mentions') -> pd.DataFrame`**: In-degree and out-degree (how many different users), and in- and out-strength (how many mentions/replies).
- **`T.betweenness_by_hand(G, v, weight=None) -> pd.DataFrame`**: Betweenness written out: for every ordered pair (s, t) with s, t != v and a path from s to t, the number of shortest paths from s to t, how many pass through v, and the share. Betweenness (unnormalised) is the sum of the shares. For an undirected graph each unordered pair is listed once.
- **`T.closeness_table(G, direction: str = 'out') -> pd.DataFrame`**: Three versions of closeness for every node. For a directed graph, direction="out" measures distances from the node to the others (how quickly its mentions reach them), "in" from the others to the node. reachable: how many others are reachable; raw: reachable / (sum of their distances), the classic formula applied inside what is reachable; scaled: raw x reachable / (n - 1) (Wasserman and Faust); harmonic: the mean of 1 / distance over all n - 1 others, with 1 / infinity = 0.
- **`T.transition_matrix(G, weight=None, nodelist=None)`**: Row-stochastic transition matrix of a random walk that follows out-edges: P[u, v] = w(u, v) / out-strength(u). A node with no out-edges (dangling) has a row of zeros. Returns (P as a DataFrame, list of dangling nodes).
- **`T.flow(x0, P, steps: int = 1) -> pd.DataFrame`**: Repeated matrix multiplication x_{t+1} = x_t P, starting from x0 (a dict or Series over P's nodes). Returns one row per step (step 0 is x0) and a 'total' column: the probability still in the walk.
- **`T.pagerank_power(G, alpha: float = 0.85, weight='mentions', dangling: str = 'uniform', steps: int = 1000, tol: float = 1e-10, start=None)`**: PageRank by power iteration, written out. Each step: with probability alpha the walker follows an out-edge (chosen in proportion to its weight), with probability 1 - alpha it teleports to a node chosen uniformly. dangling="uniform": a walker on a node with no out-edges teleports (networkx's default); dangling="leak": its probability is lost, so the total falls below 1. Returns (scores as a Series, history as a DataFrame).
- **`T.hits_table(G, weight=None) -> pd.DataFrame`**: HITS hub and authority scores (each sums to 1). A good hub points to good authorities; a good authority is pointed to by good hubs. With weight, the iteration uses the weighted adjacency matrix.
- **`T.rank_table(G, weight: str = 'mentions', alpha: float = 0.85) -> pd.DataFrame`**: Several definitions of importance side by side (directed graph G; weight names the edge weight): in_degree, in_strength, out_degree, betweenness (directed, unweighted, normalised), betweenness_undirected, harmonic closeness (directed, outgoing distances), pagerank (weighted, alpha), hub and authority (HITS).
- **`T.ranks(scores, ascending: bool = False) -> pd.Series`**: Rank positions (1 = top); ties share the best position.
- **`T.compare_rankings(a, b, k: int = 5) -> dict`**: How far two rankings agree: Spearman and Kendall rank correlations over all nodes, and the overlap of their top-k sets (ties at the k-th place broken by node name).
- **`T.perturb(G, add=None, remove=None, weight: str = 'mentions')`**: A copy of G with one edge added (add=(u, v) or (u, v, w)) and/or one removed (remove=(u, v)).
- **`T.rank_shift(before, after, k: int = None) -> pd.DataFrame`**: Scores and rank positions before and after a change, with the change in rank (positive = moved up). Scores are rounded to 6 decimals before ranking, so that numerical noise does not separate users who tie.

### Support

- **`T.attention_graph(seed: int = 2205, include_users: bool = True) -> nx.DiGraph`**: StudyHub as an attention graph: u -> v when u replied to v's posts; attribute replies = how many times. It is the flow graph of paths-and-connectivity (author -> replier) with every edge reversed; self-loops removed, and every registered user is a node (include_users=True).
- **`T.quiz(qid: str, answer=None) -> None`**: Self-check: compare your prediction with the answer, and say where to revisit the topic.
- **`T.reference_ranking(alpha: float = 0.85) -> pd.DataFrame`**: The P1 table computed by the kit on the attention graph (every registered user, self-loops removed): in_degree, in_strength (replies received), out_degree, pagerank (weighted by replies, alpha, dangling nodes teleport) and betweenness (directed, unweighted, normalised).
- **`T.check(exercise: str, answer=None, **more)`**: Check a practice exercise, e.g. T.check("P1", table). Prints one PASS or FAIL line per fact.
- **`T.reference() -> dict`**: The frozen results of the run used on the slides and in the notes (reference/results.json).

### From other topics

Supplied ready-made here; each is taught in its own topic.

- `T.default_start`, from [`paths-and-connectivity`](paths-and-connectivity.md)
- `T.draw`, from [`paths-and-connectivity`](paths-and-connectivity.md)
- `T.forum`, from [`paths-and-connectivity`](paths-and-connectivity.md)

### Constants

`T.BLUE` (str), `T.GREEN` (str), `T.GREY` (str), `T.MENTION_POS` (dict), `T.NAVY` (str), `T.ORANGE` (str), `T.P1_COLUMNS` (list), `T.PURPLE` (str), `T.ROOT` (Path), `T.SEED` (int), `T.TRIO_POS` (dict), `T.VERSION` (str)
