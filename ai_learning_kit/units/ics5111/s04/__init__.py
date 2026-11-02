"""ICS5111 Session 04: knowledge graphs (property graphs and RDF, Neo4j and Cypher, graph algorithms in the
database, TransE for link prediction, coverage and bias).

A session without graded tasks, like Sessions 02 and 03: its three notebooks (Prepare, Explore, Practice)
check themselves. The kit supplies the helpers (T = kit.tools()), Hetionet v1.0 (downloaded from its owner's
GitHub repository on first use), a small drug-interaction example in RDF and as a property graph, a Neo4j
connection that falls back to saved results when no database is connected, and the reference results of the
runs used on the slides.

Every concept is offered at three depths: the idea, the mechanics (equations and a calculation by hand) and
going deeper; the notebooks mark the mechanics and depth cells as optional.
"""
from ....core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

UNIT = "ICS5111"
SESSION = "S04"
TITLE = "Knowledge graphs"
PROBLEM = None
TASKS = {}
TOOLS = ".tools"
DATASETS = ("hetionet_nodes", "hetionet_edges")
EXTRAS = "advanced"                 # PyTorch, PyTorch Geometric, the Neo4j driver, rdflib and owlrl

_DEPTH = ("Explain in three stages and stop at the first unless the student asks for more or the cell itself "
          "shows the equation: (1) the idea in words, with the drug-interaction example or Hetionet; (2) the "
          "mechanics, the equation or query with every symbol defined and the numbers of the worked example; "
          "(3) the depth, why the method behaves as it does. Cells headed 'Optional' are the mechanics or depth "
          "stage: when asked about one, go through it step by step and check it against the printed numbers; "
          "never make a student feel that skipping it is a failure.")

STAGES = {
    "models": stage("Property graphs and RDF: nodes, relationships and properties against triples and IRIs; "
                    "schemas, ontologies (RDFS/OWL), the open-world assumption, SPARQL and Cypher on the same facts",
                    FOCUS, note="the drug-interaction example has 17 drugs and 4 enzymes; reasoning is done by "
                                "owlrl. SHACL is mentioned, not used. " + _DEPTH),
    "hetionet": stage("Reading Hetionet and choosing the subset loaded into Neo4j; its schema (metagraph)", FOCUS,
                      how="T.hetionet_nodes(), T.hetionet_edges(), T.subset(), T.schema()"),
    "loading": stage("Loading nodes and relationships into Neo4j with constraints, UNWIND batches and MERGE",
                     FOCUS, how="T.load_into_neo4j(), T.LOAD_CYPHER",
                     note="the three Cypher statements are in focus; the batching loop is a convenience"),
    "connection": stage("Connecting to Neo4j (Sandbox, Desktop or AuraDB) from a credentials file, and the saved "
                        "results used when no database is connected", ON_REQUEST, how="T.connect(), db.run()",
                        note="a convenience so that the notebooks run without a database; never print or "
                             "repeat a student's password"),
    "cypher": stage("Cypher: MATCH patterns, WHERE, aggregation, multi-hop and variable-length patterns, "
                    "existence subqueries, shortest paths", FOCUS, note=_DEPTH),
    "algorithms": stage("Graph algorithms in the database (GDS projection, PageRank, Louvain) compared with "
                        "NetworkX: score scales, dangling nodes, iterations, non-deterministic communities", FOCUS,
                        how="T.GDS_QUERIES, T.nx_pagerank(), T.nx_louvain(), T.compare_rankings(), "
                            "T.compare_partitions()", note=_DEPTH),
    "transe": stage("TransE: entities and relations as vectors, h + r close to t, margin loss with corrupted "
                    "triples, trained with PyTorch Geometric", FOCUS, how="T.make_kge(), T.train_kge()",
                    note=_DEPTH),
    "evaluation": stage("Link-prediction evaluation: ranking every entity, the filtered setting, MRR and Hits@k, "
                        "per-relation results, a random baseline, choosing settings on validation data", FOCUS,
                        how="T.filtered_ranks(), T.rank_metrics()", note=_DEPTH),
    "ranking_code": stage("The vectorised ranking code (torch.cdist and masking known answers)", ON_REQUEST,
                          how="T.score_all(), T.filtered_ranks()"),
    "variants": stage("RotatE, ComplEx and DistMult at awareness level: which relation patterns each can "
                      "represent (symmetry, inversion, composition)", FOCUS,
                      note="awareness only: the idea of each scoring function and one comparison table; no "
                           "derivations unless asked"),
    "bias": stage("Coverage gaps and bias in public knowledge graphs: literature-curated against systematic "
                  "sources, degree concentration, missing relations, popularity bias in link prediction", FOCUS),
    "shacl_owl_detail": stage("Full OWL 2 semantics, description logics and SHACL validation", OUT_OF_SCOPE,
                              taught_in="the further reading of Session 04"),
    "rgcn": stage("Relational GCNs and other GNN encoders for knowledge graphs", OUT_OF_SCOPE,
                  taught_in="the further reading of Session 04 (builds on Session 03)"),
}
