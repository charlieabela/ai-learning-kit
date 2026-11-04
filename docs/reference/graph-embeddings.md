# Graph embeddings

`graph-embeddings` · level **advanced** · builds on no other topic · pip extra `graphs`

Graph embeddings: random walks, skip-gram, node2vec, evaluation and TransE.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("graph-embeddings")
T = kit.tools()
```

Written for the **advanced** level: Learners who are fluent in Python and read code and documentation independently, at the level of a postgraduate course.

**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it is kept):

- `cora`: Cora citation network (LINQS), from LINQS Statistical Relational Learning Group, UC Santa Cruz. The archive's README describes provenance and schema; it states no redistribution licence, so the kit downloads the archive from LINQS rather than copying it.

## The three layers

### In focus: learners do these themselves

- Random walks and (centre, context) pairs from a sliding window
- Skip-gram with negative sampling: training node vectors from the pairs (with `T.fit_sgns() / T.embed()`)
- node2vec: biased walks with the return parameter p and the in-out parameter q
- Evaluating embeddings: node classification against handcrafted and text features, link prediction without leakage, choosing settings on validation data only
- Transductive versus inductive: features for nodes not seen in training
- Knowledge-graph embeddings: TransE on a handful of triples (with `T.fit_transe()`)

### Support: the kit does these for the learner

- Loading Cora: reading the archive, building the citation graph, word features, labels and the fixed 541/541/1626 split (with `T.load_cora()`)

## Helpers: `T = kit.tools()`

Graph embeddings: inspectable CPU teaching helpers, not a benchmark library.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`T.fit_sgns(pairs, n_nodes, dim=16, negatives=3, epochs=2, lr=0.15, batch=256, seed=5111)`**: Fixed-window skip-gram negative sampling. Returns source Z and context C. Repeated visits may create self-node pairs at different positions. Noise is context frequency ** .75, sampled with replacement; accidental positives are retained as noise (not asserted to be nonedges). Batch summed SGD gradients approximate successive updates. This is an explicit teaching variant.
- **`T.embed(g, **kwargs)`**: (no description)
- **`T.fit_transe(triples, n_ent, n_rel, dim=2, epochs=800, lr=0.01, margin=0.5, seed=5111)`**: Minimal TransE (Bordes et al., 2013): train h + r close to t for true triples, with a margin ranking loss against triples whose head or tail is replaced at random. Entity vectors are kept inside the unit ball. A teaching sketch for a handful of triples, not a library implementation.

### Support

- **`T.load_cora()`**: (no description)

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.toy_graph()`**: (no description)
- **`T.draw_toy(ax=None, highlight=())`**: (no description)
- **`T.transition(g, previous, current, p=1.0, q=1.0, weighted=False)`**: Second-order node2vec transition, for an undirected graph, p,q > 0.
- **`T.walks(g, n_walks=3, length=16, p=1.0, q=1.0, seed=5111, weighted=False)`**: Stops at isolated nodes. Caches transitions; can be costly on high-degree graphs.
- **`T.context_pairs(walk_list, window=2)`**: (no description)
- **`T.classify(features, y, split, test=False)`**: Select regularisation with validation labels only. Test evaluation is explicit.
- **`T.structural_features(g)`**: (no description)
- **`T.split_links(g, seed=5111)`**: Exclude held-out positives before feature computation; negatives disjoint.
- **`T.common_neighbour_auc(train_graph, positive, negative)`**: (no description)
- **`T.save_handover(g, ids, split, z, config, directory='s02_handover')`**: (no description)
- **`T.karate()`**: Zachary's karate club (networkx copy) and the faction of each member: 0 = Mr. Hi, 1 = Officer.
- **`T.lesmis()`**: Les Miserables co-appearance network (Knuth, 1993), nodes relabelled 0..76; names in attribute 'name'.
- **`T.sgns_snapshots(pairs, n_nodes, dim=2, negatives=5, epochs=30, lr=0.025, batch=64, seed=5111)`**: Same algorithm as fit_sgns, but returns a copy of Z after every epoch (snapshot 0 = random start) and the loss per positive pair for every epoch. Used to watch a small embedding train.
- **`T.walk_spread(g, walk_list)`**: Mean hop distance between the first and last node of each walk, and mean number of distinct nodes.
- **`T.neighbour_mean_features(g, x, visible)`**: Inductive features without training: [own features | mean features of the visible neighbours]. visible: boolean array; edges to invisible nodes are ignored (a new paper can cite only papers that already exist). Nodes without a visible neighbour get zeros in the second block.
- **`T.edge_features(z, pairs)`**: Hadamard (element-wise) product of the two node vectors: one feature vector per node pair.

### Constants

`T.ROOT` (Path), `T.SEED` (int), `T.VERSION` (str)
