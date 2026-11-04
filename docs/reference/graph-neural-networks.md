# Graph neural networks

`graph-neural-networks` · level **advanced** · builds on [`graph-embeddings`](graph-embeddings.md) · pip extra `advanced`

Graph neural networks: message passing, GCN, depth, GraphSAGE, limits and tasks.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("graph-neural-networks")
T = kit.tools()
```

Written for the **advanced** level: Learners who are fluent in Python and read code and documentation independently, at the level of a postgraduate course.

**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it is kept):

- `cora`: Cora citation network (LINQS), from LINQS Statistical Relational Learning Group, UC Santa Cruz. The archive's README describes provenance and schema; it states no redistribution licence, so the kit downloads the archive from LINQS rather than copying it.

## The three layers

### In focus: learners do these themselves

- Neural networks without a graph: weighted sums, softmax and cross-entropy, gradient descent, layers and ReLU, dropout and weight decay, train/validation/test
- Why an MLP on adjacency rows fails: node numbering is arbitrary, so layers must treat neighbours as an unordered multiset (permutation equivariance)
- Message passing: aggregate the neighbours, update the node; receptive fields
- The graph convolutional network: A_hat = D^-1/2 (A+I) D^-1/2, H' = ReLU(A_hat H W); training on Cora with 140 labels; SGC as a propagation-only baseline (with `T.GCN, T.train_node_classifier`)
- Depth and over-smoothing: more layers make nodes alike; SGC accuracy and mean cosine as K grows
- GraphSAGE: concatenate self and neighbour mean, sample neighbours, inductive use (with `T.SAGE`)
- Limits of GNNs: over-smoothing, heterophily, the 1-WL limit on what message passing can distinguish, cost on large graphs, over-squashing, attention is not explanation
- Node classification and link prediction with a GCN encoder (graph auto-encoder), with the graph-embeddings link split and the leakage check (with `T.train_gae`)

### On request: the kit does these, and learners may look inside

- The full-batch training loop: Adam, loss on the labelled nodes only, keeping the epoch with the best validation accuracy (with `T.train_node_classifier(), T.repeat()`)
- Building a neighbour-mean matrix from sampled neighbours (with `T.sampled_mean_matrix(), T.computation_tree()`)
- Weisfeiler-Lehman colour refinement and the random sum-GNN (with `T.wl_colours(), T.wl_same(), T.sum_gnn_vector()`)
- Generating graphs with a chosen share of same-class links (illustrative data) (with `T.synthetic_graph(), T.homophily_experiment()`)

### Support: the kit does these for the learner

- Loading Cora: the citation graph, normalised word features, labels and the low-label split (20 labelled papers per topic, 500 validation, 1,000 test) (with `T.load_cora()`; taught in the graph-embeddings topic)

### Left out of this topic

- Graph attention networks (taught in the further reading of this topic)
- Spectral graph theory and the graph Laplacian's eigenvectors (taught in the optional depth sections of this topic's notes)

## Helpers: `T = kit.tools()`

Graph neural networks: readable CPU teaching helpers in plain PyTorch.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`class T.GCN(n_in, n_out, hidden=16, n_layers=2, dropout=0.5)`**: Graph convolutional network: before every linear layer, average over the neighbourhood with A_hat (H' = A_hat H W). n_layers counts the propagation steps (2 is the usual choice).
- **`class T.SAGE(n_in, n_out, hidden=16, n_layers=2, dropout=0.5)`**: GraphSAGE with the mean aggregator: W [h_v || mean of the neighbours' h_u]. The node's own representation keeps its own half of the weights, so it is never diluted into the average.
- **`T.train_node_classifier(model, x, a, y, split, epochs=200, lr=0.01, weight_decay=0.0005, seed=0, resample=None)`**: Full-batch training: predictions for every node, loss on the training nodes only. Keeps the weights of the epoch with the best validation accuracy and reports their test accuracy. resample: optional function(epoch) -> a new propagation matrix (GraphSAGE neighbour sampling).
- **`T.train_gae(x, message_graph, positive, negative, epochs=200, lr=0.01, seed=0)`**: Train on the training links (with fresh random non-links each epoch); message_graph is the graph the encoder propagates over; x is normally cora['X01'], the 0/1 words, as in Kipf and Welling (2016). Returns validation and test AUC from the epoch with the best validation AUC.

### On request (look inside)

- **`T.repeat(make_model, x, a, y, split, seeds=(0, 1, 2, 3, 4), **kw)`**: Train a fresh model per seed (make_model() builds it after the seed is set); mean and std of test accuracy.
- **`T.computation_tree(g, node, fanouts=None, seed=5111)`**: Number of entries in a node's computation tree for len(fanouts) layers; fanouts=None uses every neighbour, otherwise at most fanouts[i] sampled neighbours at layer i (GraphSAGE).
- **`T.sampled_mean_matrix(g, fanout, seed)`**: Neighbour-mean matrix built from at most `fanout` sampled neighbours per node (a new sample per call).
- **`T.wl_colours(g, rounds=3)`**: 1-WL colour refinement from a single starting colour. Returns one sorted colour histogram per round (round 0 first). Colours are made comparable across graphs by using the signatures themselves.
- **`T.wl_same(g1, g2, rounds=3)`**: True when 1-WL cannot tell the two graphs apart within `rounds` rounds.
- **`T.sum_gnn_vector(g, layers=2, width=8, seed=5111)`**: A randomly initialised GNN with sum aggregation and a sum readout: one vector per graph.
- **`T.synthetic_graph(h, n=1200, n_classes=4, dim=16, degree=10, signal=0.55, seed=5111)`**: Synthetic graph (illustrative) whose share h of links joins nodes of the same class, with noisy class-dependent features. Returns G, X (tensor), y (tensor) and a 20/20/60 split (240/240/720).
- **`T.homophily_experiment(hs=(0.9, 0.7, 0.5, 0.3, 0.1), graphs=3, epochs=200)`**: MLP, GCN and GraphSAGE on synthetic graphs as homophily falls; mean test accuracy over `graphs` graphs.

### Support

- **`T.load_cora()`**: Cora as a dictionary: G (networkx), A (scipy sparse), X (row-normalised word features, a sparse tensor: about 18 of the 1,433 entries per paper are non-zero), X01 (the same words as 0/1, used for link prediction), x_raw (0/1 words, scipy), y (tensor), classes, ids, split (low-label, 140/500/1000) and split_embeddings (the 541/541/1626 split of graph-embeddings).

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.primer_tensors()`**: The one-paper example of the primer as tensors: x (4 words), W (3 x 4), b (3), true class.
- **`T.softmax(z)`**: Scores to probabilities: exponentiate, then divide by the total (subtracting the max first is safe).
- **`T.five_papers()`**: The five-paper graph: G (nodes 'A'..'E'), the feature matrix X (5 x 2) in node order, the names.
- **`T.draw_five(ax=None, labels=None, title=None, highlight=None)`**: Draw the five-paper graph; labels maps a node to the text written under it (e.g. its vector).
- **`T.adjacency(g, nodelist=None)`**: Dense 0/1 adjacency matrix of an undirected graph, in the order of nodelist.
- **`T.mean_adjacency(a, self_loops=True)`**: Row-normalised adjacency: each row averages a node's neighbours (and itself, with self_loops).
- **`T.gcn_adjacency(a)`**: The GCN's normalised matrix A_hat = D^-1/2 (A + I) D^-1/2 (Kipf and Welling, 2017).
- **`T.mean_aggregate(a, h, rounds=1)`**: Message passing at its simplest: every node averages itself and its neighbours, `rounds` times.
- **`T.position_layer_demo(seed=5111)`**: Why a plain MLP on adjacency rows fails: the same graph, numbered two ways, gives different outputs with a layer that has one weight per position, but the same outputs with a neighbour sum.
- **`T.low_label_split(y, per_class=20, n_val=500, n_test=1000, seed=5111)`**: 20 labelled papers per topic for training, then 500 validation and 1,000 test papers, all disjoint.
- **`T.as_tensor(m)`**: A scipy or NumPy matrix as a dense float32 tensor.
- **`T.as_sparse_tensor(m)`**: A matrix as a sparse PyTorch tensor: only the non-zero entries are stored (about 13,000 of Cora's 7.3 million), which makes multiplying by it fast.
- **`T.propagation_matrices(a)`**: The two matrices the models use, as sparse tensors: A_hat for the GCN and the neighbour mean (no self-loop) for GraphSAGE.
- **`T.propagate(a, h)`**: a @ h for a dense or sparse matrix a: every node's new row is a weighted sum of its neighbours' rows.
- **`T.dropout(x, p, training)`**: Dropout that also works on a sparse feature matrix. On Cora's words only the stored (non-zero) entries can be silenced, since a zero stays zero, so this matches ordinary dropout and is much faster.
- **`T.linear(layer, x)`**: layer(x) for a dense or a sparse x: x W^T + b.
- **`class T.MLP(n_in, n_out, hidden=16, dropout=0.5)`**: Two layers on a node's own features; the graph is ignored. The baseline every GNN must beat.
- **`T.n_parameters(model)`**: Number of trainable numbers in a model.
- **`T.plot_history(result, ax=None, title=None)`**: Training loss and accuracies per epoch, with the chosen epoch marked.
- **`T.logistic_eval(features, y, split, Cs=(0.1, 1.0, 10.0, 100.0))`**: Multinomial logistic regression with C chosen on the validation papers; test accuracy reported.
- **`T.sgc_features(a_hat, x, k=2)`**: Simplified graph convolution (Wu et al., 2019): propagate the features k times, A_hat^k X, no learning.
- **`T.mean_cosine(h, n_pairs=2000, seed=5111)`**: Average cosine similarity between randomly chosen pairs of rows: near 1 means everyone looks alike.
- **`T.deepwalk_baseline(cora)`**: graph-embeddings' DeepWalk vectors (same settings as its slides) evaluated on this topic's split.
- **`T.edge_homophily(g, y, classes=None)`**: Share of edges joining two nodes of the same class, overall and for edges touching each class.
- **`T.receptive_field(g, node, hops)`**: Number of nodes within `hops` steps of node (not counting the node itself).
- **`T.receptive_field_summary(g, hops=(1, 2, 3), sample=300, seed=5111)`**: Mean and maximum receptive-field size over a random sample of nodes.
- **`T.without_nodes(g, nodes)`**: Copy of g with every edge touching `nodes` removed (the nodes stay, isolated), as if they did not exist yet.
- **`T.aggregate_multisets()`**: Mean, max and sum of two pairs of neighbour multisets that differ only in counts.
- **`class T.GAE(n_in, hidden=32, out=16)`**: Graph auto-encoder (Kipf and Welling, 2016): a two-layer GCN encoder; a pair scores z_u . z_v.
- **`T.reference()`**: The frozen results shown on the slides, as a dictionary (reference/results.json).

### From other topics

Supplied ready-made here; each is taught in its own topic.

- `T.common_neighbour_auc`, from [`graph-embeddings`](graph-embeddings.md)
- `T.split_links`, from [`graph-embeddings`](graph-embeddings.md)

### Constants

`T.FIVE_EDGES` (list), `T.FIVE_FEATURES` (dict), `T.FIVE_FEATURE_NAMES` (tuple), `T.FIVE_POS` (dict), `T.PRIMER` (dict), `T.ROOT` (Path), `T.SEED` (int), `T.SEEDS` (tuple), `T.TOPICS_SHORT` (dict), `T.VERSION` (str)
