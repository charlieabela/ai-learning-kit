"""ICS5111 Session 03 (graph neural networks): readable CPU teaching helpers in plain PyTorch.

Used in the three Session 03 notebooks as T = kit.tools() after kit.start("ICS5111", "S03").
Every layer is written out in a few lines (no graph library), so that the code can be read
next to the equations in the student notes. Cora comes from its owner (LINQS) through
ai_learning_kit.datasets and is parsed exactly as in Session 02; the split used here is a
low-label split in the style of Yang, Cohen and Salakhutdinov (2016): 20 labelled papers per
topic, 500 for validation and 1,000 for test, drawn with seed 5111 (it is not the published
Planetoid split, so the numbers are close to, but not the same as, published ones).
reference/ holds the frozen results of the runs used on the slides.

Results of neural networks vary slightly with the machine (the order of floating-point sums),
so a third decimal may differ from the slides; differences in the second decimal mean
something has changed.
"""
from pathlib import Path
import json, time
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..s02 import tools as _s02          # Session 02's Cora parser and link split, unchanged

ROOT = Path(__file__).parent              # ROOT / 'reference' holds the frozen reference results
SEED = 5111
VERSION = '2026-10-30'
SEEDS = (0, 1, 2, 3, 4)                   # the five seeds behind every mean and standard deviation
TOPICS_SHORT = {'Case_Based': 'Case based', 'Genetic_Algorithms': 'Genetic algorithms',
                'Neural_Networks': 'Neural networks', 'Probabilistic_Methods': 'Probabilistic methods',
                'Reinforcement_Learning': 'Reinforcement learning', 'Rule_Learning': 'Rule learning',
                'Theory': 'Theory'}

# ---------------------------------------------------------------------------------------------
# Part 1: a neural network without a graph (the primer)
# ---------------------------------------------------------------------------------------------
PRIMER = {
    'words': ['graph', 'image', 'network', 'learning'],
    'classes': ['Graph mining', 'Computer vision', 'Machine learning'],
    'x': [1., 0., 1., 1.],
    'W': [[1.0, -0.5, 0.8, 0.2], [-0.6, 1.2, -0.2, 0.3], [0.1, 0.1, 0.4, 0.9]],
    'b': [0., 0.1, -0.2],
    'true_class': 0,
    'W1': [[0.5, -0.4, 0.6, 0.0], [-0.3, 0.8, -0.1, 0.2]],
    'W2': [[1.0, -1.0], [-0.8, 1.2], [0.4, 0.4]],
}

def primer_tensors():
    """The one-paper example of the primer as tensors: x (4 words), W (3 x 4), b (3), true class."""
    return (torch.tensor(PRIMER['x']), torch.tensor(PRIMER['W']), torch.tensor(PRIMER['b']),
            PRIMER['true_class'])

def softmax(z):
    """Scores to probabilities: exponentiate, then divide by the total (subtracting the max first is safe)."""
    z = np.asarray(z, dtype=float); e = np.exp(z - z.max()); return e / e.sum()

# ---------------------------------------------------------------------------------------------
# The five-paper graph used for every calculation by hand
# ---------------------------------------------------------------------------------------------
FIVE_EDGES = [('A', 'B'), ('A', 'C'), ('A', 'D'), ('B', 'D'), ('B', 'E'), ('D', 'E')]
FIVE_FEATURES = {'A': (1, 0), 'B': (1, 1), 'C': (0, 1), 'D': (1, 0), 'E': (0, 1)}
FIVE_FEATURE_NAMES = ('mentions graphs', 'mentions neural networks')
FIVE_POS = {'A': (0, 0), 'B': (1.2, 0.7), 'C': (-1.2, 0.5), 'D': (1.2, -0.7), 'E': (2.4, 0)}

def five_papers():
    """The five-paper graph: G (nodes 'A'..'E'), the feature matrix X (5 x 2) in node order, the names."""
    g = nx.Graph(FIVE_EDGES); names = ['A', 'B', 'C', 'D', 'E']
    x = np.array([FIVE_FEATURES[n] for n in names], dtype=float)
    return g, x, names

def draw_five(ax=None, labels=None, title=None, highlight=None):
    """Draw the five-paper graph; labels maps a node to the text written under it (e.g. its vector)."""
    if ax is None: _, ax = plt.subplots(figsize=(5.5, 2.8))
    g, x, names = five_papers()
    colours = ['#D37B35' if (highlight and n in highlight) else '#4279B7' for n in names]
    nx.draw(g, FIVE_POS, ax=ax, nodelist=names, with_labels=True, node_color=colours, font_color='white',
            node_size=900, width=2, edge_color='#9CA9B6', font_weight='bold')
    text = labels if labels is not None else {n: str(tuple(int(v) for v in FIVE_FEATURES[n])) for n in names}
    for n in names:
        px, py = FIVE_POS[n]; ax.text(px, py - 0.42, text[n], ha='center', va='top', fontsize=10)
    ax.set_xlim(-1.8, 3.0); ax.set_ylim(-1.5, 1.2)
    if title: ax.set_title(title)
    return ax

def adjacency(g, nodelist=None):
    """Dense 0/1 adjacency matrix of an undirected graph, in the order of nodelist."""
    nodelist = list(nodelist if nodelist is not None else sorted(g))
    return nx.to_numpy_array(g, nodelist=nodelist)

def mean_adjacency(a, self_loops=True):
    """Row-normalised adjacency: each row averages a node's neighbours (and itself, with self_loops)."""
    a = np.asarray(a, dtype=float) + (np.eye(len(a)) if self_loops else 0)
    deg = a.sum(1, keepdims=True); return a / np.maximum(deg, 1)

def gcn_adjacency(a):
    """The GCN's normalised matrix A_hat = D^-1/2 (A + I) D^-1/2 (Kipf and Welling, 2017)."""
    a = np.asarray(a, dtype=float) + np.eye(len(a)); d = 1 / np.sqrt(a.sum(1))
    return a * d[:, None] * d[None, :]

def mean_aggregate(a, h, rounds=1):
    """Message passing at its simplest: every node averages itself and its neighbours, `rounds` times."""
    m = mean_adjacency(a); out = [np.asarray(h, dtype=float)]
    for _ in range(rounds): out.append(m @ out[-1])
    return out

def position_layer_demo(seed=SEED):
    """Why a plain MLP on adjacency rows fails: the same graph, numbered two ways, gives different outputs
    with a layer that has one weight per position, but the same outputs with a neighbour sum."""
    g, x, names = five_papers(); rng = np.random.default_rng(seed)
    w = rng.normal(0, 1, 5); value = x[:, 0] + 2 * x[:, 1]          # one number per paper
    out = {}
    for label, order in (('A,B,C,D,E', names), ('E,D,C,B,A', names[::-1])):
        a = adjacency(g, order); v = np.array([value[names.index(n)] for n in order])
        positional = (a * v[None, :]) @ w                            # weight j multiplies position j
        neighbour_sum = a @ v
        out[label] = pd.DataFrame({'position layer': positional, 'neighbour sum': neighbour_sum}, index=order)
    return out

# ---------------------------------------------------------------------------------------------
# Cora as tensors
# ---------------------------------------------------------------------------------------------
def low_label_split(y, per_class=20, n_val=500, n_test=1000, seed=SEED):
    """20 labelled papers per topic for training, then 500 validation and 1,000 test papers, all disjoint."""
    rng = np.random.default_rng(seed); y = np.asarray(y)
    train = np.concatenate([rng.choice(np.flatnonzero(y == c), per_class, replace=False) for c in np.unique(y)])
    rest = rng.permutation(np.setdiff1d(np.arange(len(y)), train))
    return {'train': np.sort(train), 'val': np.sort(rest[:n_val]), 'test': np.sort(rest[n_val:n_val + n_test])}

def load_cora():
    """Cora as a dictionary: G (networkx), A (scipy sparse), X (row-normalised word features, a sparse
    tensor: about 18 of the 1,433 entries per paper are non-zero), X01 (the same words as 0/1, used for
    link prediction), x_raw (0/1 words, scipy), y (tensor), classes, ids, split (low-label, 140/500/1000) and
    split_s02 (the 541/541/1626 split of Session 02)."""
    g, x, y, ids, classes, split_s02 = _s02.load_cora()
    a = nx.to_scipy_sparse_array(g, nodelist=range(len(ids)), format='csr')
    rows = np.asarray(x.sum(1)).ravel(); xn = sparse.diags(1 / np.maximum(rows, 1)) @ x
    return {'G': g, 'A': a, 'X': as_sparse_tensor(xn), 'X01': as_sparse_tensor(x), 'x_raw': x,
            'y': torch.tensor(y, dtype=torch.long), 'classes': [TOPICS_SHORT.get(c, c) for c in classes],
            'ids': ids, 'split': low_label_split(y), 'split_s02': split_s02}

def as_tensor(m):
    """A scipy or NumPy matrix as a dense float32 tensor."""
    if sparse.issparse(m): m = m.toarray()
    return torch.tensor(np.asarray(m), dtype=torch.float32)

def as_sparse_tensor(m):
    """A matrix as a sparse PyTorch tensor: only the non-zero entries are stored (about 13,000 of Cora's
    7.3 million), which makes multiplying by it fast."""
    m = sparse.coo_matrix(m)
    idx = torch.tensor(np.vstack([m.row, m.col]), dtype=torch.long)
    return torch.sparse_coo_tensor(idx, torch.tensor(m.data, dtype=torch.float32), m.shape,
                                   check_invariants=False).coalesce()

def propagation_matrices(a):
    """The two matrices the models use, as sparse tensors: A_hat for the GCN and the neighbour mean
    (no self-loop) for GraphSAGE."""
    a = sparse.csr_matrix(a, dtype=float); n = a.shape[0]
    a_tilde = a + sparse.eye(n); d = 1 / np.sqrt(np.asarray(a_tilde.sum(1)).ravel())
    a_hat = sparse.diags(d) @ a_tilde @ sparse.diags(d)
    deg = np.asarray(a.sum(1)).ravel(); a_mean = sparse.diags(1 / np.maximum(deg, 1)) @ a
    return {'gcn': as_sparse_tensor(a_hat), 'mean': as_sparse_tensor(a_mean)}

def propagate(a, h):
    """a @ h for a dense or sparse matrix a: every node's new row is a weighted sum of its neighbours' rows."""
    return torch.sparse.mm(a, h) if a.is_sparse else a @ h

# ---------------------------------------------------------------------------------------------
# Models: each layer in a few lines
# ---------------------------------------------------------------------------------------------
def dropout(x, p, training):
    """Dropout that also works on a sparse feature matrix. On Cora's words only the stored (non-zero)
    entries can be silenced, since a zero stays zero, so this matches ordinary dropout and is much faster."""
    if not training or p == 0: return x
    if x.is_sparse:
        x = x.coalesce(); keep = (torch.rand(x.values().shape[0]) >= p).float() / (1 - p)
        return torch.sparse_coo_tensor(x.indices(), x.values() * keep, x.shape, check_invariants=False)
    return F.dropout(x, p, True)

def linear(layer, x):
    """layer(x) for a dense or a sparse x: x W^T + b."""
    if not x.is_sparse: return layer(x)
    out = torch.sparse.mm(x, layer.weight.t())
    return out if layer.bias is None else out + layer.bias

class MLP(nn.Module):
    """Two layers on a node's own features; the graph is ignored. The baseline every GNN must beat."""
    def __init__(self, n_in, n_out, hidden=16, dropout=0.5):
        super().__init__(); self.l1 = nn.Linear(n_in, hidden); self.l2 = nn.Linear(hidden, n_out); self.p = dropout
    def forward(self, x, a=None):
        h = F.relu(linear(self.l1, dropout(x, self.p, self.training)))
        return self.l2(dropout(h, self.p, self.training))

class GCN(nn.Module):
    """Graph convolutional network: before every linear layer, average over the neighbourhood with
    A_hat (H' = A_hat H W). n_layers counts the propagation steps (2 is the usual choice)."""
    def __init__(self, n_in, n_out, hidden=16, n_layers=2, dropout=0.5):
        super().__init__(); sizes = [n_in] + [hidden] * (n_layers - 1) + [n_out]
        self.layers = nn.ModuleList(nn.Linear(i, o) for i, o in zip(sizes[:-1], sizes[1:])); self.p = dropout
    def forward(self, x, a_hat):
        h = x
        for i, layer in enumerate(self.layers):
            h = propagate(a_hat, linear(layer, dropout(h, self.p, self.training)))   # transform, then average
            if i < len(self.layers) - 1: h = F.relu(h)
        return h
    def embed(self, x, a_hat):
        """The representation after the first layer (16 numbers per node)."""
        return F.relu(propagate(a_hat, linear(self.layers[0], x)))

class SAGE(nn.Module):
    """GraphSAGE with the mean aggregator: W [h_v || mean of the neighbours' h_u]. The node's own
    representation keeps its own half of the weights, so it is never diluted into the average."""
    def __init__(self, n_in, n_out, hidden=16, n_layers=2, dropout=0.5):
        super().__init__(); sizes = [n_in] + [hidden] * (n_layers - 1) + [n_out]
        self.self_w = nn.ModuleList(nn.Linear(i, o) for i, o in zip(sizes[:-1], sizes[1:]))
        self.neigh_w = nn.ModuleList(nn.Linear(i, o, bias=False) for i, o in zip(sizes[:-1], sizes[1:]))
        self.p = dropout
    def forward(self, x, a_mean):
        h = x
        for i, (ws, wn) in enumerate(zip(self.self_w, self.neigh_w)):
            h = dropout(h, self.p, self.training)
            h = linear(ws, h) + propagate(a_mean, linear(wn, h))   # = W [h || mean of neighbours h_u]
            if i < len(self.self_w) - 1: h = F.relu(h)
        return h

def _seed_all(seed):
    torch.manual_seed(seed); np.random.seed(seed)

def train_node_classifier(model, x, a, y, split, epochs=200, lr=0.01, weight_decay=5e-4, seed=0,
                          resample=None):
    """Full-batch training: predictions for every node, loss on the training nodes only. Keeps the
    weights of the epoch with the best validation accuracy and reports their test accuracy.
    resample: optional function(epoch) -> a new propagation matrix (GraphSAGE neighbour sampling)."""
    _seed_all(seed); opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    tr, va, te = (torch.as_tensor(split[k]) for k in ('train', 'val', 'test'))
    hist = {'loss': [], 'train_acc': [], 'val_acc': []}; best = (-1, None, 0); t0 = time.perf_counter()
    for epoch in range(epochs):
        model.train(); opt.zero_grad()
        out = model(x, resample(epoch) if resample else a)
        loss = F.cross_entropy(out[tr], y[tr]); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(x, a).argmax(1)
            acc = lambda idx: float((pred[idx] == y[idx]).float().mean())
            hist['loss'].append(float(loss)); hist['train_acc'].append(acc(tr)); hist['val_acc'].append(acc(va))
            if best[1] is None or hist['val_acc'][-1] > best[0]:
                best = (hist['val_acc'][-1], {k: v.clone() for k, v in model.state_dict().items()}, epoch)
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad(): pred = model(x, a).argmax(1)
    return {'val_acc': best[0], 'best_epoch': best[2],
            'train_acc': float((pred[tr] == y[tr]).float().mean()),
            'test_acc': float((pred[te] == y[te]).float().mean()), 'history': hist,
            'seconds': time.perf_counter() - t0, 'model': model}

def repeat(make_model, x, a, y, split, seeds=SEEDS, **kw):
    """Train a fresh model per seed (make_model() builds it after the seed is set); mean and std of test accuracy."""
    accs = []
    for s in seeds:
        _seed_all(s); accs.append(train_node_classifier(make_model(), x, a, y, split, seed=s, **kw)['test_acc'])
    return {'mean': float(np.mean(accs)), 'std': float(np.std(accs)), 'runs': [float(v) for v in accs]}

def n_parameters(model):
    """Number of trainable numbers in a model."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))

def plot_history(result, ax=None, title=None):
    """Training loss and accuracies per epoch, with the chosen epoch marked."""
    if ax is None: _, ax = plt.subplots(figsize=(6.5, 3.2))
    h = result['history']; ax.plot(h['train_acc'], label='training accuracy'); ax.plot(h['val_acc'], label='validation accuracy')
    ax.plot(np.array(h['loss']) / max(h['loss']), ':', label='training loss (scaled)')
    ax.axvline(result['best_epoch'], color='grey', lw=1); ax.set_xlabel('epoch'); ax.legend(fontsize=8)
    if title: ax.set_title(title)
    return ax

# ---------------------------------------------------------------------------------------------
# Baselines without a neural network
# ---------------------------------------------------------------------------------------------
def logistic_eval(features, y, split, Cs=(0.1, 1., 10., 100.)):
    """Multinomial logistic regression with C chosen on the validation papers; test accuracy reported."""
    if torch.is_tensor(features) and features.is_sparse: features = features.to_dense()
    f = features.numpy() if torch.is_tensor(features) else (features.toarray() if sparse.issparse(features) else features)
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y); best = None
    for C in Cs:
        m = LogisticRegression(C=C, max_iter=3000).fit(f[split['train']], y[split['train']])
        v = accuracy_score(y[split['val']], m.predict(f[split['val']]))
        if best is None or v > best[0]: best = (v, C, m)
    return {'C': best[1], 'val_acc': float(best[0]),
            'test_acc': float(accuracy_score(y[split['test']], best[2].predict(f[split['test']])))}

def sgc_features(a_hat, x, k=2):
    """Simplified graph convolution (Wu et al., 2019): propagate the features k times, A_hat^k X, no learning."""
    h = x.to_dense() if torch.is_tensor(x) and x.is_sparse else x
    for _ in range(k): h = propagate(a_hat, h)
    return h

def mean_cosine(h, n_pairs=2000, seed=SEED):
    """Average cosine similarity between randomly chosen pairs of rows: near 1 means everyone looks alike."""
    h = h.numpy() if torch.is_tensor(h) else np.asarray(h); rng = np.random.default_rng(seed)
    i, j = rng.integers(len(h), size=(2, n_pairs)); keep = i != j; i, j = i[keep], j[keep]
    hn = h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12)
    return float(np.mean(np.sum(hn[i] * hn[j], axis=1)))

def deepwalk_baseline(cora):
    """Session 02's DeepWalk vectors (same settings as its slides) evaluated on this session's split."""
    nodes, z, info = _s02.embed(cora['G'], n_walks=3, length=16, window=2, dim=16, negatives=3, epochs=2, lr=.03)
    return logistic_eval(z, cora['y'], cora['split'])

# ---------------------------------------------------------------------------------------------
# Looking at the graph: homophily, receptive fields, sampling
# ---------------------------------------------------------------------------------------------
def edge_homophily(g, y, classes=None):
    """Share of edges joining two nodes of the same class, overall and for edges touching each class."""
    y = y.numpy() if torch.is_tensor(y) else np.asarray(y)
    e = np.array(list(g.edges())); same = y[e[:, 0]] == y[e[:, 1]]
    per = {}
    for c in np.unique(y):
        touch = (y[e[:, 0]] == c) | (y[e[:, 1]] == c)
        per[classes[c] if classes else int(c)] = float(same[touch].mean())
    return float(same.mean()), per

def receptive_field(g, node, hops):
    """Number of nodes within `hops` steps of node (not counting the node itself)."""
    return len(nx.single_source_shortest_path_length(g, node, cutoff=hops)) - 1

def receptive_field_summary(g, hops=(1, 2, 3), sample=300, seed=SEED):
    """Mean and maximum receptive-field size over a random sample of nodes."""
    rng = np.random.default_rng(seed); nodes = rng.choice(sorted(g), min(sample, len(g)), replace=False)
    rows = {h: [receptive_field(g, n, h) for n in nodes] for h in hops}
    return pd.DataFrame({'mean': {h: float(np.mean(v)) for h, v in rows.items()},
                         'max': {h: int(np.max(v)) for h, v in rows.items()}}).rename_axis('hops')

def computation_tree(g, node, fanouts=None, seed=SEED):
    """Number of entries in a node's computation tree for len(fanouts) layers; fanouts=None uses every
    neighbour, otherwise at most fanouts[i] sampled neighbours at layer i (GraphSAGE)."""
    rng = np.random.default_rng(seed); level = [node]; total = 1
    for f in (fanouts if fanouts is not None else [None, None]):
        nxt = []
        for v in level:
            nb = list(g[v])
            if f is not None and len(nb) > f: nb = list(rng.choice(nb, f, replace=False))
            nxt.extend(nb)
        total += len(nxt); level = nxt
    return total

def sampled_mean_matrix(g, fanout, seed):
    """Neighbour-mean matrix built from at most `fanout` sampled neighbours per node (a new sample per call)."""
    rng = np.random.default_rng(seed); n = len(g); rows, cols = [], []
    for v in range(n):
        nb = list(g[v])
        if len(nb) > fanout: nb = list(rng.choice(nb, fanout, replace=False))
        rows += [v] * len(nb); cols += nb
    m = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    deg = np.asarray(m.sum(1)).ravel()
    return as_sparse_tensor(sparse.diags(1 / np.maximum(deg, 1)) @ m)

# ---------------------------------------------------------------------------------------------
# Inductive use: train without the test papers
# ---------------------------------------------------------------------------------------------
def without_nodes(g, nodes):
    """Copy of g with every edge touching `nodes` removed (the nodes stay, isolated), as if they did not exist yet."""
    h = g.copy(); h.remove_edges_from([e for e in g.edges(nodes)]); return h

# ---------------------------------------------------------------------------------------------
# Limits: Weisfeiler-Lehman, aggregators, heterophily
# ---------------------------------------------------------------------------------------------
def wl_colours(g, rounds=3):
    """1-WL colour refinement from a single starting colour. Returns one sorted colour histogram per round
    (round 0 first). Colours are made comparable across graphs by using the signatures themselves."""
    colour = {v: '0' for v in g}; hist = [sorted(colour.values())]
    for _ in range(rounds):
        colour = {v: colour[v] + '(' + ','.join(sorted(colour[u] for u in g[v])) + ')' for v in g}
        hist.append(sorted(colour.values()))
    return hist

def wl_same(g1, g2, rounds=3):
    """True when 1-WL cannot tell the two graphs apart within `rounds` rounds."""
    return all(a == b for a, b in zip(wl_colours(g1, rounds), wl_colours(g2, rounds)))

def sum_gnn_vector(g, layers=2, width=8, seed=SEED):
    """A randomly initialised GNN with sum aggregation and a sum readout: one vector per graph."""
    rng = np.random.default_rng(seed); a = adjacency(g); h = np.ones((len(g), 1))
    for _ in range(layers):
        w = rng.normal(0, 1, (h.shape[1], width)); h = np.maximum(0, (h + a @ h) @ w)
    return h.sum(0)

def aggregate_multisets():
    """Mean, max and sum of two pairs of neighbour multisets that differ only in counts."""
    a, b = np.array([1., 0.]), np.array([0., 1.])
    sets = {'{a, a}': [a, a], '{a}': [a], '{a, b}': [a, b], '{a, a, b, b}': [a, a, b, b]}
    return pd.DataFrame({k: {'mean': tuple(np.mean(v, 0).round(2)), 'max': tuple(np.max(v, 0)),
                             'sum': tuple(np.sum(v, 0))} for k, v in sets.items()}).T

def synthetic_graph(h, n=1200, n_classes=4, dim=16, degree=10, signal=0.55, seed=SEED):
    """Synthetic graph (illustrative) whose share h of links joins nodes of the same class, with noisy
    class-dependent features. Returns G, X (tensor), y (tensor) and a 20/20/60 split (240/240/720)."""
    rng = np.random.default_rng(seed); y = np.repeat(np.arange(n_classes), n // n_classes)
    means = rng.normal(0, 1, (n_classes, dim)); x = signal * means[y] + rng.normal(0, 1, (n, dim))
    g = nx.Graph(); g.add_nodes_from(range(n)); by_class = [np.flatnonzero(y == c) for c in range(n_classes)]
    others = [np.flatnonzero(y != c) for c in range(n_classes)]
    for v in range(n):
        for _ in range(degree // 2):
            pool = by_class[y[v]] if rng.random() < h else others[y[v]]
            u = int(rng.choice(pool))
            if u != v: g.add_edge(v, u)
    order = rng.permutation(n); k = n // 5                       # 20% train, 20% validation, 60% test
    split = {'train': np.sort(order[:k]), 'val': np.sort(order[k:2 * k]), 'test': np.sort(order[2 * k:])}
    return g, torch.tensor(x, dtype=torch.float32), torch.tensor(y), split

def homophily_experiment(hs=(0.9, 0.7, 0.5, 0.3, 0.1), graphs=3, epochs=200):
    """MLP, GCN and GraphSAGE on synthetic graphs as homophily falls; mean test accuracy over `graphs` graphs."""
    rows = {}
    for h in hs:
        acc = {'MLP (features only)': [], 'GCN': [], 'GraphSAGE': []}
        for k in range(graphs):
            g, x, y, split = synthetic_graph(h, seed=SEED + k)
            p = propagation_matrices(nx.to_scipy_sparse_array(g, nodelist=range(len(g))))
            for name, make, mat in (('MLP (features only)', lambda: MLP(x.shape[1], 4, hidden=32), None),
                                    ('GCN', lambda: GCN(x.shape[1], 4, hidden=32), p['gcn']),
                                    ('GraphSAGE', lambda: SAGE(x.shape[1], 4, hidden=32), p['mean'])):
                _seed_all(k); acc[name].append(train_node_classifier(make(), x, mat, y, split, epochs=epochs, seed=k)['test_acc'])
        rows[h] = {k: float(np.mean(v)) for k, v in acc.items()}
    return pd.DataFrame(rows).rename_axis('model')

# ---------------------------------------------------------------------------------------------
# Link prediction with a GCN encoder (graph auto-encoder)
# ---------------------------------------------------------------------------------------------
split_links = _s02.split_links            # the Session 02 link split (527 validation and 527 test links)
common_neighbour_auc = _s02.common_neighbour_auc

class GAE(nn.Module):
    """Graph auto-encoder (Kipf and Welling, 2016): a two-layer GCN encoder; a pair scores z_u . z_v."""
    def __init__(self, n_in, hidden=32, out=16):
        super().__init__(); self.l1 = nn.Linear(n_in, hidden); self.l2 = nn.Linear(hidden, out)
    def encode(self, x, a_hat):
        return propagate(a_hat, self.l2(F.relu(propagate(a_hat, self.l1(x)))))
    @staticmethod
    def score(z, pairs):
        pairs = torch.as_tensor(np.asarray(pairs)); return (z[pairs[:, 0]] * z[pairs[:, 1]]).sum(1)

def train_gae(x, message_graph, positive, negative, epochs=200, lr=0.01, seed=0):
    """Train on the training links (with fresh random non-links each epoch); message_graph is the graph the
    encoder propagates over; x is normally cora['X01'], the 0/1 words, as in Kipf and Welling (2016). Returns validation and test AUC from the epoch with the best validation AUC."""
    _seed_all(seed); n = x.shape[0]
    a_hat = propagation_matrices(nx.to_scipy_sparse_array(message_graph, nodelist=range(n)))['gcn']
    model = GAE(x.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = torch.as_tensor(positive['train']); rng = np.random.default_rng(seed); best = (-1, 0, 0)
    auc = lambda z, k: roc_auc_score([1] * len(positive[k]) + [0] * len(negative[k]),
                                     torch.cat([GAE.score(z, positive[k]), GAE.score(z, negative[k])]).numpy())
    for epoch in range(epochs):
        model.train(); opt.zero_grad(); z = model.encode(x, a_hat)
        neg = torch.as_tensor(rng.integers(n, size=(len(pos), 2)))
        logits = torch.cat([GAE.score(z, pos), GAE.score(z, neg)])
        labels = torch.cat([torch.ones(len(pos)), torch.zeros(len(neg))])
        loss = F.binary_cross_entropy_with_logits(logits, labels); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            z = model.encode(x, a_hat); v = auc(z, 'val')
            if v > best[0]: best = (v, auc(z, 'test'), epoch)
    return {'val_auc': float(best[0]), 'test_auc': float(best[1]), 'best_epoch': best[2]}

# ---------------------------------------------------------------------------------------------
# Reference results
# ---------------------------------------------------------------------------------------------
def reference():
    """The frozen results shown on the slides, as a dictionary (reference/results.json)."""
    return json.loads((ROOT / 'reference' / 'results.json').read_text())
