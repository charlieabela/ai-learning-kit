# Knowledge graphs

`knowledge-graphs` · level **advanced** · builds on [`graph-embeddings`](graph-embeddings.md), [`graph-neural-networks`](graph-neural-networks.md) · pip extra `advanced`

Knowledge graphs: property graphs and RDF, Neo4j and Cypher, graph algorithms in the database, TransE for link prediction, coverage and bias.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("knowledge-graphs")
T = kit.tools()
```

Written for the **advanced** level: Learners who are fluent in Python and read code and documentation independently, at the level of a postgraduate course.

**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it is kept):

- `hetionet_nodes`: Hetionet v1.0 nodes (TSV), from Hetionet project (Himmelstein et al.), GitHub hetio/hetionet. Hetionet's original content is CC0; it integrates 29 resources whose own licences vary (the hetionet repository links a per-source table). The kit downloads the files from the hetionet repository rather than copying them.
- `hetionet_edges`: Hetionet v1.0 edges (gzipped SIF), from Hetionet project (Himmelstein et al.), GitHub hetio/hetionet. Hetionet's original content is CC0; it integrates 29 resources whose own licences vary (the hetionet repository links a per-source table). The kit downloads the files from the hetionet repository rather than copying them.

## The three layers

### In focus: learners do these themselves

- Property graphs and RDF: nodes, relationships and properties against triples and IRIs; schemas, ontologies (RDFS/OWL), the open-world assumption, SPARQL and Cypher on the same facts
- Reading Hetionet and choosing the subset loaded into Neo4j; its schema (metagraph) (with `T.hetionet_nodes(), T.hetionet_edges(), T.subset(), T.schema()`)
- Loading nodes and relationships into Neo4j with constraints, UNWIND batches and MERGE (with `T.load_into_neo4j(), T.LOAD_CYPHER`)
- Cypher: MATCH patterns, WHERE, aggregation, multi-hop and variable-length patterns, existence subqueries, shortest paths
- Graph algorithms in the database (GDS projection, PageRank, Louvain) compared with NetworkX: score scales, dangling nodes, iterations, non-deterministic communities (with `T.GDS_QUERIES, T.nx_pagerank(), T.nx_louvain(), T.compare_rankings(), T.compare_partitions()`)
- TransE: entities and relations as vectors, h + r close to t, margin loss with corrupted triples, trained with PyTorch Geometric (with `T.make_kge(), T.train_kge()`)
- Link-prediction evaluation: ranking every entity, the filtered setting, MRR and Hits@k, per-relation results, a random baseline, choosing settings on validation data (with `T.filtered_ranks(), T.rank_metrics()`)
- RotatE, ComplEx and DistMult at awareness level: which relation patterns each can represent (symmetry, inversion, composition)
- Coverage gaps and bias in public knowledge graphs: literature-curated against systematic sources, degree concentration, missing relations, popularity bias in link prediction

### On request: the kit does these, and learners may look inside

- Connecting to Neo4j (Sandbox, Desktop or AuraDB) from a credentials file, and the saved results used when no database is connected (with `T.connect(), db.run()`)
- The vectorised ranking code (torch.cdist and masking known answers) (with `T.score_all(), T.filtered_ranks()`)

### Left out of this topic

- Full OWL 2 semantics, description logics and SHACL validation (taught in the further reading of this topic)
- Relational GCNs and other GNN encoders for knowledge graphs (taught in the further reading of this topic (builds on graph-neural-networks))

## Helpers: `T = kit.tools()`

Knowledge graphs: helpers for RDF, Neo4j, graph algorithms and TransE.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`T.hetionet_nodes()`**: All 47,031 Hetionet nodes: id, name, kind.
- **`T.hetionet_edges(metaedges=None)`**: Hetionet edges (source, metaedge, target), optionally only the listed metaedges.
- **`T.subset(metaedges)`**: (nodes, edges) for the listed metaedges: only nodes that take part in at least one of the edges, with Neo4j labels and relationship types added.
- **`T.schema(nodes, edges)`**: The schema of a subset: node counts per label and, per relationship type, the labels it joins.
- **`T.load_into_neo4j(db, nodes, edges, batch=5000, progress=True)`**: Load a Hetionet subset into Neo4j with the three statements in T.LOAD_CYPHER: one uniqueness constraint per label, then nodes and relationships in batches of `batch` rows (UNWIND). Safe to run twice (MERGE).
- **`T.nx_pagerank(g, alpha=0.85)`**: NetworkX PageRank (scores sum to 1; dangling nodes spread their score over all nodes).
- **`T.nx_louvain(g, seed=5111, resolution=1.0)`**: NetworkX Louvain communities as a Series node -> community number (largest community first).
- **`T.compare_rankings(a, b, k=20)`**: Agreement of two score Series on the same nodes: Spearman correlation and overlap of the top k.
- **`T.compare_partitions(g, p1, p2)`**: Agreement of two partitions: adjusted Rand index, normalised mutual information, and each modularity.
- **`T.make_kge(name, n_ent, n_rel, dim=64, margin=None)`**: A PyTorch Geometric knowledge-graph embedding model: 'TransE', 'RotatE', 'ComplEx' or 'DistMult'.
- **`T.train_kge(model, train, epochs=40, lr=0.01, batch=1024, seed=5111, log_every=0)`**: Train with the model's own loss (PyG corrupts a head or a tail at random for each positive triple).
- **`T.filtered_ranks(model, test, known, n_ent, batch=64, candidates=None)`**: Filtered ranks of the true tail and the true head of every test triple.
- **`T.rank_metrics(ranks)`**: MRR, Hits@1, Hits@3, Hits@10 and mean rank from a list of ranks.

### On request (look inside)

- **`T.connect(folder='.', quiet=False)`**: Connect to your Neo4j database with the credentials found by _credentials(); otherwise return a session that uses the kit's saved results. Never prints the password.
- **`T.score_all(model, part, side, ents)`**: Scores of every entity as the missing tail (side='tail') or head of each triple in `part` (B x n_ent). For TransE the distances to all entities are computed at once (torch.cdist), which is much faster than scoring the candidate triples one by one and gives the same numbers as the model's own forward().

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.ddi_rdf(extra_ttl='')`**: The drug-interaction example as an rdflib Graph (optionally with extra Turtle appended).
- **`T.short(term)`**: An RDF term without the example namespace (ex:warfarin -> warfarin).
- **`T.sparql(g, query)`**: Run a SPARQL query on an rdflib graph and return a DataFrame of short names.
- **`T.reason(g)`**: Expand g in place with the OWL 2 RL rules (owlrl) and return the number of triples added.
- **`T.drugs_typed(g)`**: Sorted names of everything that is rdf:type ex:Drug in g.
- **`T.interacting_pairs(g)`**: (pairs, self_pairs): unordered drug pairs joined by ex:interactsWith, and drugs said to interact with themselves.
- **`T.ddi_property_graph()`**: The same facts as a property graph (networkx MultiDiGraph): nodes with a label and name, edges with a type and, for the asserted interaction, properties of its own.
- **`T.ddi_cypher_params()`**: The parameters for DDI_CYPHER_CREATE, taken from the RDF version so that both hold the same facts.
- **`T.metaedge_table()`**: One row per metaedge: source kind, relation, target kind, edges and the share from systematic sources.
- **`T.gene_network()`**: The gene-gene interaction network (GiG) as an undirected networkx Graph with gene names.
- **`T.project_genes(db)`**: Project the gene-interaction network for GDS, trying the self-managed form first and then the AuraDB Graph Analytics form (which needs a memory size). Returns the projection's summary.
- **`class T.Neo4jSession(driver=None, database=None)`**: Runs Cypher on a Neo4j database, or returns the kit's saved result when none is connected.
- **`T.capture(db, folder=<kit>/topics/knowledge_graphs/saved, load=True)`**: Lecturer tool: on an EMPTY connected database, load the drug-interaction example and the Hetionet subset, run every named query in the order the notebooks use them, and save each result as folder/<name>.json (the kit's saved results). Returns {name: rows}.
- **`T.gds_formula_pagerank(g, damping=0.85, iterations=20)`**: PageRank as Neo4j GDS documents it, re-implemented here only to explain the differences: every node starts at 1 - d and receives (1 - d) + d * sum(score / out-degree); scores are not normalised. This is NOT a GDS result; the real one comes from your database (db.run('gds_pagerank')).
- **`T.modularity(g, partition)`**: Modularity of a partition given as node -> community.
- **`T.transe_hand_step(v={'Paris': [0.12, -0.45, 0.31], 'capital_of': [0.05, 0.33, -0.21], 'France': [0.67, 0.23, -0.18], 'Berlin': [-0.23, 0.41, 0.19], 'margin': 1.0, 'lr': 0.1})`**: One margin-loss step for (Paris, capital_of, France) against the corruption (Berlin, capital_of, France).
- **`T.kge_triples(metaedges=('CbG', 'CtD', 'CpD', 'CrC', 'DaG', 'DrD', 'DpS', 'DlA', 'PCiC'))`**: Integer-coded triples for embedding: (triples array [n, 3] of head, relation, tail), entity ids, relation names (metaedge abbreviations) and the entity names.
- **`T.split_triples(t, frac=0.1, seed=5111)`**: Random train/validation/test split of the triples (frac each for validation and test). A held-out triple whose head or tail would not appear in training is moved back to training, because a TransE model has no vector for an entity it never saw (the transductive limit).
- **`T.type_candidates(triples, ents)`**: For each relation, the entities of the right kind as heads and tails (taken from the Hetionet id prefix, e.g. 'Compound::DB00331'), so that ranking can be restricted to entities of the right type.
- **`T.metrics_by_relation(ranks, test, rels)`**: rank_metrics for each relation (tail and head ranks together).
- **`T.random_mrr(n_candidates)`**: Expected MRR of a uniformly random ranking among n candidates: (1/n) * sum(1/k).
- **`T.gini(x)`**: Gini coefficient of non-negative values: 0 when all are equal, near 1 when a few hold almost everything.
- **`T.degree_concentration(metaedge, side='target', top=0.01)`**: How unevenly one metaedge's edges fall on its nodes: Gini of the degrees and the share of edges held by the top `top` fraction of nodes (for example genes in CbG, 'compound binds gene').
- **`T.coverage_table()`**: Simple coverage checks on drugs and diseases: how many have at least one edge of each kind.
- **`T.reference()`**: The frozen results shown on the slides (reference/results.json).

### Constants

`T.DDI_CYPHER_CREATE` (str), `T.DDI_PREFIX` (str), `T.DDI_RULE_CYPHER` (str), `T.DDI_RULE_SPARQL` (str), `T.DDI_TTL` (str), `T.GDS_QUERIES` (dict), `T.KGE_METAEDGES` (tuple), `T.LABELS` (dict), `T.LOAD_CYPHER` (dict), `T.METAEDGES` (dict), `T.NEO4J_METAEDGES` (tuple), `T.QUERIES` (dict), `T.REL_TYPES` (dict), `T.ROOT` (Path), `T.SAVED` (Path), `T.SEED` (int), `T.TRANSE_HAND` (dict), `T.VERSION` (str)
