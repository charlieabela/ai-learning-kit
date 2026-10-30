"""ICS5111 Session 03: graph neural networks (message passing, GCN, depth, GraphSAGE, limits, tasks).

A session without graded tasks, like Session 02: its three notebooks (Prepare, Explore,
Practice) check themselves with prediction-then-check cells, and the practice notebook
compares the student's own functions with the helpers. The kit supplies the helpers
(T = kit.tools(), plain PyTorch), Cora (downloaded from LINQS on first use, as in Session 02)
and the reference results of the runs used on the slides.

Every concept is offered at three depths: the idea (words and pictures), the mechanics
(equations and a calculation by hand on the five-paper graph) and the depth (why the method
behaves as it does). The notebooks mark the mechanics and depth cells as optional.
"""
from ....core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

UNIT = "ICS5111"
SESSION = "S03"
TITLE = "Graph neural networks"
PROBLEM = None                      # no case problem: Cora, the five-paper graph and synthetic graphs
TASKS = {}                          # no answer book: the notebooks check themselves
TOOLS = ".tools"                    # T = kit.tools()
DATASETS = ("cora",)                # downloaded by kit.start, so a failure shows in the first cell
EXTRAS = "advanced"                 # pip extra the notebooks install: PyTorch (and PyTorch Geometric)

_DEPTH = ("Explain in three stages and stop at the first unless the student asks for more or the cell itself "
          "shows the equation: (1) the idea in words, with the five-paper graph or a domain example; "
          "(2) the mechanics, the equation with every symbol defined and the numbers of the hand calculation; "
          "(3) the depth, why the method behaves as it does. Cells headed 'Optional' are the mechanics or depth "
          "stage: when asked about one, go through the equation symbol by symbol and check it against the "
          "printed numbers; never make a student feel that skipping it is a failure.")

STAGES = {
    "loading": stage("Loading Cora: the citation graph, normalised word features, labels and the low-label split "
                     "(20 labelled papers per topic, 500 validation, 1,000 test)", BLACK_BOX, how="T.load_cora()",
                     taught_in="ICS5111 Session 02",
                     note="say what comes out and why only 140 labels are used (semi-supervised setting); "
                          "the parsing is Session 02's"),
    "primer": stage("Neural networks without a graph: weighted sums, softmax and cross-entropy, gradient descent, "
                    "layers and ReLU, dropout and weight decay, train/validation/test", FOCUS,
                    note="a brief primer for students without a neural-network background; " + _DEPTH),
    "permutation": stage("Why an MLP on adjacency rows fails: node numbering is arbitrary, so layers must treat "
                         "neighbours as an unordered multiset (permutation equivariance)", FOCUS, note=_DEPTH),
    "message_passing": stage("Message passing: aggregate the neighbours, update the node; receptive fields", FOCUS,
                             note=_DEPTH),
    "gcn": stage("The graph convolutional network: A_hat = D^-1/2 (A+I) D^-1/2, H' = ReLU(A_hat H W); training on "
                 "Cora with 140 labels; SGC as a propagation-only baseline", FOCUS, how="T.GCN, T.train_node_classifier",
                 note="the spectral origin of the name is optional depth: mention it only if asked. " + _DEPTH),
    "depth": stage("Depth and over-smoothing: more layers make nodes alike; SGC accuracy and mean cosine as K grows",
                   FOCUS, note=_DEPTH),
    "graphsage": stage("GraphSAGE: concatenate self and neighbour mean, sample neighbours, inductive use", FOCUS,
                       how="T.SAGE", note=_DEPTH),
    "limits": stage("Limits of GNNs: over-smoothing, heterophily, the 1-WL limit on what message passing can "
                    "distinguish, cost on large graphs, over-squashing, attention is not explanation", FOCUS,
                    note="proofs of the WL bound are out of scope; state the result and show the six-cycle example. "
                         + _DEPTH),
    "tasks": stage("Node classification and link prediction with a GCN encoder (graph auto-encoder), with the "
                   "Session 02 link split and the leakage check", FOCUS, how="T.train_gae"),
    "training_loop": stage("The full-batch training loop: Adam, loss on the labelled nodes only, keeping the epoch with "
                           "the best validation accuracy", ON_REQUEST, how="T.train_node_classifier(), T.repeat()",
                           note="the idea (predict every node, learn from the 140 labelled ones) is in focus; the loop's "
                                "code is optional"),
    "sampling_code": stage("Building a neighbour-mean matrix from sampled neighbours", ON_REQUEST,
                           how="T.sampled_mean_matrix(), T.computation_tree()"),
    "wl_code": stage("Weisfeiler-Lehman colour refinement and the random sum-GNN", ON_REQUEST,
                     how="T.wl_colours(), T.wl_same(), T.sum_gnn_vector()"),
    "synthetic": stage("Generating graphs with a chosen share of same-class links (illustrative data)", ON_REQUEST,
                       how="T.synthetic_graph(), T.homophily_experiment()"),
    "attention": stage("Graph attention networks", OUT_OF_SCOPE, taught_in="the further reading of Session 03",
                       note="if asked, say that GAT learns how much each neighbour counts from the content"),
    "spectral": stage("Spectral graph theory and the graph Laplacian's eigenvectors", OUT_OF_SCOPE,
                      taught_in="the optional depth sections of the Session 03 student notes"),
}
