"""ICS5111 Session 02: graph embeddings (random walks, skip-gram, node2vec, evaluation, TransE).

A session without graded tasks: its three notebooks (Prepare, Explore, Practice) check
themselves with prediction-then-check cells, and the practice notebook compares the
student's own functions with the helpers. The kit supplies the helpers (T = kit.tools()),
Cora (downloaded from LINQS on first use) and the reference results of the run used in
the slides.
"""
from ....core.session import BLACK_BOX, FOCUS, stage

UNIT = "ICS5111"
SESSION = "S02"
TITLE = "Graph embeddings"
PROBLEM = None                      # no case problem: the session works on Cora and small classic graphs
TASKS = {}                          # no answer book: the notebooks check themselves
TOOLS = ".tools"                    # T = kit.tools()
DATASETS = ("cora",)                # downloaded by kit.start, so a failure shows in the first cell
EXTRAS = "graphs"                   # pip extra the notebooks install: networkx and scikit-learn

STAGES = {
    "loading": stage("Loading Cora: reading the archive, building the citation graph, word features, labels and "
                     "the fixed 541/541/1626 split", BLACK_BOX, how="T.load_cora()",
                     note="say what comes out (G, X, y, paper_ids, classes, split) and why the split is fixed; "
                          "the parsing itself is not part of the session"),
    "walks": stage("Random walks and (centre, context) pairs from a sliding window", FOCUS),
    "sgns": stage("Skip-gram with negative sampling: training node vectors from the pairs", FOCUS,
                  how="T.fit_sgns() / T.embed()",
                  note="explain the objective and what the loss shows; the NumPy implementation can be read "
                       "but is not a task"),
    "node2vec": stage("node2vec: biased walks with the return parameter p and the in-out parameter q", FOCUS),
    "evaluation": stage("Evaluating embeddings: node classification against handcrafted and text features, "
                        "link prediction without leakage, choosing settings on validation data only", FOCUS),
    "inductive": stage("Transductive versus inductive: features for nodes not seen in training", FOCUS),
    "transe": stage("Knowledge-graph embeddings: TransE on a handful of triples", FOCUS,
                    how="T.fit_transe()", note="a teaching sketch, not a library implementation"),
}
