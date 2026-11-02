"""ICS5111 Session 04 (knowledge graphs): helpers for RDF, Neo4j, graph algorithms and TransE.

Used in the three Session 04 notebooks as T = kit.tools() after kit.start("ICS5111", "S04").

- A small drug-interaction example (17 drugs, 4 enzymes) written both as RDF (rdflib, with
  reasoning by owlrl) and as a property graph, so that the two data models can be compared on
  the same facts.
- Hetionet v1.0 (Himmelstein et al., 2017), a public biomedical knowledge graph, read from its
  owner's GitHub repository through ai_learning_kit.datasets, and the subsets used in the
  session: the part loaded into Neo4j, the gene-interaction network used for PageRank and
  Louvain, and the drug-disease-gene triples used for TransE.
- A Neo4j connection that runs Cypher on the student's own database (Neo4j Sandbox, Neo4j
  Desktop or AuraDB) and, when no database is connected, returns the kit's saved result of the
  same named query, so that every notebook also runs without a database.
- NetworkX counterparts of the in-database algorithms and functions to compare the two.
- TransE (and RotatE, ComplEx, DistMult) with PyTorch Geometric, trained on CPU, and the
  filtered ranking evaluation (MRR, Hits@k) written out in a few readable lines.
- Measures of coverage and bias in the knowledge graph.

reference/ holds the frozen results shown on the slides; saved/ holds the saved results of the
named Cypher queries. Results of neural networks vary slightly between machines, so a third
decimal may differ from the slides.
"""
from pathlib import Path
import json, os, re, time, gzip, hashlib
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from .... import datasets

ROOT = Path(__file__).parent
SAVED = ROOT / "saved"
SEED = 5111
VERSION = '2026-10-31'

# =============================================================================================
# Part 1: two data models for the same facts (the drug-interaction example)
# =============================================================================================
DDI_PREFIX = "http://example.org/ddi#"
DDI_TTL = """@prefix ex:   <http://example.org/ddi#> .
@prefix rdf:  <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl:  <http://www.w3.org/2002/07/owl#> .

# --- ontology: classes ---------------------------------------------------------------
ex:Drug a owl:Class .            ex:Enzyme a owl:Class .
ex:Statin rdfs:subClassOf ex:Drug .
ex:Antibiotic rdfs:subClassOf ex:Drug .      ex:Macrolide rdfs:subClassOf ex:Antibiotic .
ex:Antifungal rdfs:subClassOf ex:Drug .      ex:Azole rdfs:subClassOf ex:Antifungal .
ex:Benzodiazepine rdfs:subClassOf ex:Drug .  ex:Anticoagulant rdfs:subClassOf ex:Drug .
ex:Antiplatelet rdfs:subClassOf ex:Drug .    ex:Opioid rdfs:subClassOf ex:Drug .
ex:Antiarrhythmic rdfs:subClassOf ex:Drug .  ex:ProtonPumpInhibitor rdfs:subClassOf ex:Drug .
ex:SSRI rdfs:subClassOf ex:Drug .

# --- ontology: relations ---------------------------------------------------------------
ex:inhibits      rdfs:domain ex:Drug ; rdfs:range ex:Enzyme .
ex:metabolisedBy rdfs:domain ex:Drug ; rdfs:range ex:Enzyme .
ex:interactsWith a owl:SymmetricProperty .

# --- data: drugs (none is asserted to be a Drug) ---------------------------------------
ex:simvastatin a ex:Statin ;  ex:metabolisedBy ex:CYP3A4 .
ex:atorvastatin a ex:Statin ; ex:metabolisedBy ex:CYP3A4 .
ex:rosuvastatin a ex:Statin .
ex:pravastatin a ex:Statin .
ex:midazolam a ex:Benzodiazepine ; ex:metabolisedBy ex:CYP3A4 .
ex:warfarin a ex:Anticoagulant ;   ex:metabolisedBy ex:CYP2C9 .
ex:clopidogrel a ex:Antiplatelet ; ex:metabolisedBy ex:CYP2C19 .
ex:codeine a ex:Opioid ;           ex:metabolisedBy ex:CYP2D6 .
ex:aspirin a ex:Antiplatelet .
ex:clarithromycin a ex:Macrolide ; ex:inhibits ex:CYP3A4 .
ex:azithromycin a ex:Macrolide .
ex:itraconazole a ex:Azole ; ex:inhibits ex:CYP3A4 .
ex:fluconazole a ex:Azole ;  ex:inhibits ex:CYP3A4 , ex:CYP2C9 , ex:CYP2C19 .
ex:amiodarone a ex:Antiarrhythmic ;    ex:inhibits ex:CYP2C9 .
ex:omeprazole a ex:ProtonPumpInhibitor ; ex:inhibits ex:CYP2C19 .
ex:fluoxetine a ex:SSRI ; ex:inhibits ex:CYP2D6 .
ex:paroxetine a ex:SSRI ; ex:inhibits ex:CYP2D6 .
ex:warfarin ex:interactsWith ex:aspirin .
"""
DDI_RULE_SPARQL = """PREFIX ex: <http://example.org/ddi#>
SELECT ?perpetrator ?victim ?enzyme WHERE {
  ?perpetrator ex:inhibits      ?enzyme .
  ?victim      ex:metabolisedBy ?enzyme .
} ORDER BY ?enzyme ?perpetrator ?victim"""
DDI_RULE_CYPHER = ("MATCH (p:Drug)-[:INHIBITS]->(e:Enzyme)<-[:METABOLISED_BY]-(v:Drug)\n"
                   "RETURN p.name AS perpetrator, v.name AS victim, e.name AS enzyme\n"
                   "ORDER BY enzyme, perpetrator, victim")

def ddi_rdf(extra_ttl=""):
    """The drug-interaction example as an rdflib Graph (optionally with extra Turtle appended)."""
    import rdflib
    return rdflib.Graph().parse(data=DDI_TTL + "\n" + extra_ttl, format="turtle")

def short(term):
    """An RDF term without the example namespace (ex:warfarin -> warfarin)."""
    return str(term).replace(DDI_PREFIX, "")

def sparql(g, query):
    """Run a SPARQL query on an rdflib graph and return a DataFrame of short names."""
    res = g.query(query)
    return pd.DataFrame([[short(v) for v in row] for row in res], columns=[str(v) for v in res.vars])

def reason(g):
    """Expand g in place with the OWL 2 RL rules (owlrl) and return the number of triples added."""
    import owlrl
    before = len(g); owlrl.DeductiveClosure(owlrl.OWLRL_Semantics).expand(g); return len(g) - before

def drugs_typed(g):
    """Sorted names of everything that is rdf:type ex:Drug in g."""
    import rdflib
    return sorted(short(s) for s in g.subjects(rdflib.RDF.type, rdflib.URIRef(DDI_PREFIX + "Drug")))

def interacting_pairs(g):
    """(pairs, self_pairs): unordered drug pairs joined by ex:interactsWith, and drugs said to interact with
    themselves."""
    import rdflib
    p = rdflib.URIRef(DDI_PREFIX + "interactsWith")
    pairs = {tuple(sorted((short(a), short(b)))) for a, b in g.subject_objects(p) if a != b}
    selfp = sorted({short(a) for a, b in g.subject_objects(p) if a == b})
    return sorted(pairs), selfp

def ddi_property_graph():
    """The same facts as a property graph (networkx MultiDiGraph): nodes with a label and name, edges with a
    type and, for the asserted interaction, properties of its own."""
    g = nx.MultiDiGraph(); r = ddi_rdf()
    import rdflib
    ex = lambda x: rdflib.URIRef(DDI_PREFIX + x)
    for s, o in r.subject_objects(rdflib.RDF.type):
        if (o, rdflib.RDFS.subClassOf, None) in r or short(o) in ("Statin",):
            g.add_node(short(s), label="Drug", drug_class=short(o))
    for rel, typ in (("inhibits", "INHIBITS"), ("metabolisedBy", "METABOLISED_BY")):
        for s, o in r.subject_objects(ex(rel)):
            g.add_node(short(o), label="Enzyme"); g.add_edge(short(s), short(o), type=typ)
    g.add_edge("warfarin", "aspirin", type="INTERACTS_WITH", mechanism="pharmacodynamic", effect="bleeding")
    return g

DDI_CYPHER_CREATE = """CREATE CONSTRAINT drug_name IF NOT EXISTS FOR (d:Drug) REQUIRE d.name IS UNIQUE;
CREATE CONSTRAINT enzyme_name IF NOT EXISTS FOR (e:Enzyme) REQUIRE e.name IS UNIQUE;
UNWIND $drugs AS row MERGE (d:Drug {name: row.name}) SET d.drug_class = row.drug_class;
UNWIND $enzymes AS name MERGE (:Enzyme {name: name});
UNWIND $inhibits AS row MATCH (d:Drug {name: row[0]}), (e:Enzyme {name: row[1]}) MERGE (d)-[:INHIBITS]->(e);
UNWIND $metabolised AS row MATCH (d:Drug {name: row[0]}), (e:Enzyme {name: row[1]}) MERGE (d)-[:METABOLISED_BY]->(e);
MATCH (w:Drug {name:'warfarin'}), (a:Drug {name:'aspirin'})
MERGE (w)-[:INTERACTS_WITH {mechanism:'pharmacodynamic', effect:'bleeding'}]->(a)"""

def ddi_cypher_params():
    """The parameters for DDI_CYPHER_CREATE, taken from the RDF version so that both hold the same facts."""
    g = ddi_property_graph()
    drugs = [{"name": n, "drug_class": d.get("drug_class")} for n, d in g.nodes(data=True) if d.get("label") == "Drug"]
    enz = sorted(n for n, d in g.nodes(data=True) if d.get("label") == "Enzyme")
    e = lambda t: sorted([u, v] for u, v, d in g.edges(data=True) if d["type"] == t)
    return {"drugs": sorted(drugs, key=lambda r: r["name"]), "enzymes": enz, "inhibits": e("INHIBITS"),
            "metabolised": e("METABOLISED_BY")}

# =============================================================================================
# Hetionet
# =============================================================================================
# The metaedges of Hetionet v1.0 (hetio/hetionet describe/edges/metaedges.tsv): abbreviation ->
# (source kind, relation, target kind, edges, edges from systematic "unbiased" sources).
METAEDGES = {
    "AdG": ("Anatomy", "downregulates", "Gene", 102240, 102240),
    "AeG": ("Anatomy", "expresses", "Gene", 526407, 453477),
    "AuG": ("Anatomy", "upregulates", "Gene", 97848, 97848),
    "CbG": ("Compound", "binds", "Gene", 11571, 0),
    "CcSE": ("Compound", "causes", "Side Effect", 138944, 0),
    "CdG": ("Compound", "downregulates", "Gene", 21102, 21102),
    "CpD": ("Compound", "palliates", "Disease", 390, 0),
    "CrC": ("Compound", "resembles", "Compound", 6486, 6486),
    "CtD": ("Compound", "treats", "Disease", 755, 0),
    "CuG": ("Compound", "upregulates", "Gene", 18756, 18756),
    "DaG": ("Disease", "associates", "Gene", 12623, 1284),
    "DdG": ("Disease", "downregulates", "Gene", 7623, 7623),
    "DlA": ("Disease", "localizes", "Anatomy", 3602, 0),
    "DpS": ("Disease", "presents", "Symptom", 3357, 0),
    "DrD": ("Disease", "resembles", "Disease", 543, 0),
    "DuG": ("Disease", "upregulates", "Gene", 7731, 7731),
    "GcG": ("Gene", "covaries", "Gene", 61690, 61690),
    "GiG": ("Gene", "interacts", "Gene", 147164, 15517),
    "GpBP": ("Gene", "participates", "Biological Process", 559504, 0),
    "GpCC": ("Gene", "participates", "Cellular Component", 73566, 0),
    "GpMF": ("Gene", "participates", "Molecular Function", 97222, 0),
    "GpPW": ("Gene", "participates", "Pathway", 84372, 0),
    "Gr>G": ("Gene", "regulates", "Gene", 265672, 265672),
    "PCiC": ("Pharmacologic Class", "includes", "Compound", 1029, 0),
}
LABELS = {"Compound": "Compound", "Disease": "Disease", "Gene": "Gene", "Anatomy": "Anatomy",
          "Symptom": "Symptom", "Side Effect": "SideEffect", "Pharmacologic Class": "PharmacologicClass",
          "Biological Process": "BiologicalProcess", "Cellular Component": "CellularComponent",
          "Molecular Function": "MolecularFunction", "Pathway": "Pathway"}
REL_TYPES = {"binds": "BINDS", "treats": "TREATS", "palliates": "PALLIATES", "resembles": "RESEMBLES",
             "associates": "ASSOCIATES", "presents": "PRESENTS", "localizes": "LOCALIZES", "includes": "INCLUDES",
             "downregulates": "DOWNREGULATES", "upregulates": "UPREGULATES", "interacts": "INTERACTS",
             "causes": "CAUSES", "expresses": "EXPRESSES", "covaries": "COVARIES", "participates": "PARTICIPATES",
             "regulates": "REGULATES"}

# The part of Hetionet loaded into Neo4j: drugs, diseases, genes, symptoms, anatomy and classes
# (about 24,000 nodes and 243,000 relationships; the large gene-annotation edges are left out).
NEO4J_METAEDGES = ("CbG", "CtD", "CpD", "CrC", "DaG", "DrD", "DpS", "DlA", "PCiC",
                   "CdG", "CuG", "DdG", "DuG", "GiG")
# The triples used for TransE: everything above except the gene-gene and expression edges.
KGE_METAEDGES = ("CbG", "CtD", "CpD", "CrC", "DaG", "DrD", "DpS", "DlA", "PCiC")

_cache = {}

def hetionet_nodes():
    """All 47,031 Hetionet nodes: id, name, kind."""
    if "nodes" not in _cache:
        _cache["nodes"] = pd.read_csv(datasets.fetch("hetionet_nodes", quiet=True), sep="\t")
    return _cache["nodes"].copy()

def hetionet_edges(metaedges=None):
    """Hetionet edges (source, metaedge, target), optionally only the listed metaedges."""
    if "edges" not in _cache:
        _cache["edges"] = pd.read_csv(datasets.fetch("hetionet_edges", quiet=True), sep="\t")
    e = _cache["edges"]
    return (e[e.metaedge.isin(metaedges)] if metaedges is not None else e).reset_index(drop=True).copy()

def metaedge_table():
    """One row per metaedge: source kind, relation, target kind, edges and the share from systematic sources."""
    t = pd.DataFrame([(k, *v) for k, v in METAEDGES.items()],
                     columns=["metaedge", "source", "relation", "target", "edges", "unbiased"])
    t["share_systematic"] = (t.unbiased / t.edges).round(3)
    return t.sort_values("edges", ascending=False).reset_index(drop=True)

def subset(metaedges):
    """(nodes, edges) for the listed metaedges: only nodes that take part in at least one of the edges,
    with Neo4j labels and relationship types added."""
    e = hetionet_edges(metaedges); n = hetionet_nodes()
    used = pd.unique(pd.concat([e.source, e.target]))
    n = n[n.id.isin(used)].reset_index(drop=True); n["label"] = n.kind.map(LABELS)
    e["type"] = e.metaedge.map(lambda m: REL_TYPES[METAEDGES[m][1]])
    return n, e

def schema(nodes, edges):
    """The schema of a subset: node counts per label and, per relationship type, the labels it joins."""
    kind = dict(zip(nodes.id, nodes.label))
    s = edges.assign(source_label=edges.source.map(kind), target_label=edges.target.map(kind))
    rels = (s.groupby(["source_label", "type", "target_label", "metaedge"]).size().rename("edges")
            .reset_index().sort_values("edges", ascending=False).reset_index(drop=True))
    return nodes.label.value_counts().rename_axis("label").rename("nodes").reset_index(), rels

def gene_network():
    """The gene-gene interaction network (GiG) as an undirected networkx Graph with gene names."""
    n = hetionet_nodes(); names = dict(zip(n.id, n.name)); e = hetionet_edges(["GiG"])
    g = nx.Graph(); g.add_edges_from(zip(e.source, e.target))
    nx.set_node_attributes(g, {v: names.get(v, v) for v in g}, "name")
    return g

# =============================================================================================
# Neo4j
# =============================================================================================
# Named queries used in the notebooks. Each has a saved result in saved/<name>.json that is
# returned when no database is connected (and checked against the live result when one is).
QUERIES = {
    "count_nodes": "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS nodes ORDER BY nodes DESC, label",
    "count_rels": ("MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS relationships "
                   "ORDER BY relationships DESC, type"),
    "schema_pairs": ("MATCH (a)-[r]->(b) RETURN labels(a)[0] AS source, type(r) AS type, labels(b)[0] AS target, "
                     "count(*) AS relationships ORDER BY relationships DESC, source, type"),
    "ddi_rule": DDI_RULE_CYPHER,
    "ddi_cleanup": "MATCH (n) WHERE n:Drug OR n:Enzyme DETACH DELETE n RETURN count(*) AS deleted",
    "treats_epilepsy": ("MATCH (c:Compound)-[:TREATS]->(d:Disease {name: 'epilepsy syndrome'}) "
                        "RETURN c.name AS compound ORDER BY compound"),
    "most_treated": ("MATCH (c:Compound)-[:TREATS]->(d:Disease) RETURN d.name AS disease, count(c) AS compounds "
                     "ORDER BY compounds DESC, disease LIMIT 10"),
    "untreated_diseases": ("MATCH (d:Disease) WHERE NOT EXISTS { (:Compound)-[:TREATS]->(d) } "
                           "RETURN count(d) AS diseases_without_a_treatment"),
    "two_hop_genes": ("MATCH (c:Compound {name: 'Warfarin'})-[:BINDS]->(g:Gene)<-[:ASSOCIATES]-(d:Disease) "
                      "WITH d, g ORDER BY g.name "
                      "WITH d, collect(DISTINCT g.name) AS genes "
                      "RETURN d.name AS disease, size(genes) AS shared_genes, genes[..5] AS genes "
                      "ORDER BY shared_genes DESC, disease LIMIT 10"),
    "repurposing": ("MATCH (c:Compound)-[:BINDS]->(g:Gene)<-[:ASSOCIATES]-(d:Disease {name: 'multiple sclerosis'}) "
                    "WHERE NOT EXISTS { (c)-[:TREATS]->(d) } "
                    "WITH c, g ORDER BY g.name "
                    "WITH c, collect(DISTINCT g.name) AS genes "
                    "RETURN c.name AS compound, size(genes) AS genes, genes[..4] AS via "
                    "ORDER BY genes DESC, compound LIMIT 10"),
    "similar_compounds": ("MATCH p = (c:Compound {name: 'Warfarin'})-[:RESEMBLES*1..2]-(o:Compound) WHERE o <> c "
                          "WITH o, min(length(p)) AS hops RETURN hops, count(o) AS compounds ORDER BY hops"),
    "shortest_paths": ("MATCH (c:Compound {name: 'Warfarin'}), (d:Disease {name: \"Alzheimer's disease\"}) "
                       "MATCH p = allShortestPaths((c)-[*..4]-(d)) "
                       "RETURN length(p) AS hops, [n IN nodes(p) | n.name] AS path, "
                       "[r IN relationships(p) | type(r)] AS types ORDER BY path"),
    "gene_degree_top": ("MATCH (g:Gene) RETURN g.name AS gene, COUNT { (g)--() } AS degree "
                        "ORDER BY degree DESC, gene LIMIT 10"),
}
GDS_QUERIES = {
    # The projection: an in-memory copy of the gene-interaction network, treated as undirected. Self-managed
    # GDS (Neo4j Desktop, Sandbox) takes the first form; AuraDB's Graph Analytics sessions need a memory size.
    "gds_project": ("MATCH (a:Gene)-[:INTERACTS]->(b:Gene) "
                    "WITH gds.graph.project('genes', a, b, {}, {undirectedRelationshipTypes: ['*']}) AS g "
                    "RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS relationships"),
    "gds_project_aura": ("MATCH (a:Gene)-[:INTERACTS]->(b:Gene) "
                         "WITH gds.graph.project('genes', a, b, {}, {undirectedRelationshipTypes: ['*'], memory: '2GB'}) AS g "
                         "RETURN g.graphName AS graph, g.nodeCount AS nodes, g.relationshipCount AS relationships"),
    # Results are written back as node properties and then read with plain Cypher, which works in both settings.
    "gds_pagerank_write": ("CALL gds.pageRank.write('genes', {dampingFactor: 0.85, maxIterations: 20, "
                           "writeProperty: 'pagerank'}) YIELD nodePropertiesWritten, ranIterations, didConverge "
                           "RETURN nodePropertiesWritten AS written, ranIterations AS iterations, didConverge AS converged"),
    "gds_pagerank_read": ("MATCH (g:Gene) WHERE g.pagerank IS NOT NULL "
                          "RETURN g.id AS id, g.name AS gene, g.pagerank AS score ORDER BY score DESC, gene"),
    "gds_louvain_write": ("CALL gds.louvain.write('genes', {writeProperty: 'louvain'}) "
                          "YIELD communityCount, modularity RETURN communityCount AS communities, modularity"),
    "gds_louvain_read": "MATCH (g:Gene) WHERE g.louvain IS NOT NULL RETURN g.id AS id, g.louvain AS community",
    "gds_drop": "CALL gds.graph.drop('genes', false) YIELD graphName RETURN graphName AS dropped",
}

def project_genes(db):
    """Project the gene-interaction network for GDS, trying the self-managed form first and then the AuraDB
    Graph Analytics form (which needs a memory size). Returns the projection's summary."""
    try:
        return db.run("gds_project")
    except Exception as first:
        if not db.connected:
            raise
        try:
            return db.run("gds_project_aura")
        except Exception:
            raise first

def _credentials(folder="."):
    """Neo4j credentials from the environment (NEO4J_URI, NEO4J_USERNAME, NEO4J_PASSWORD) or from a
    credentials .txt file in the notebook's folder, as Neo4j Sandbox and AuraDB let you download it."""
    env = {k: os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")}
    if env["NEO4J_URI"] and env["NEO4J_PASSWORD"]:
        return env["NEO4J_URI"], env["NEO4J_USERNAME"] or "neo4j", env["NEO4J_PASSWORD"]
    for f in sorted(Path(folder).glob("*.txt")):
        text = f.read_text(errors="ignore")
        vals = dict(re.findall(r"^\s*(NEO4J_\w+)\s*=\s*(\S+)\s*$", text, re.M))
        if vals.get("NEO4J_URI") and vals.get("NEO4J_PASSWORD"):
            return vals["NEO4J_URI"], vals.get("NEO4J_USERNAME", "neo4j"), vals["NEO4J_PASSWORD"]
    return None

class Neo4jSession:
    """Runs Cypher on a Neo4j database, or returns the kit's saved result when none is connected.

    db.run("count_nodes")              a named query from T.QUERIES / T.GDS_QUERIES
    db.run("MATCH (n) RETURN count(n)") any Cypher (needs a connected database)
    db.connected                       True when a database answered
    """
    def __init__(self, driver=None, database=None):
        self.driver, self.database = driver, database
        self.connected = driver is not None
        self.last_source = None

    def run(self, query, **params):
        name = query if query in QUERIES or query in GDS_QUERIES else None
        text = QUERIES.get(query) or GDS_QUERIES.get(query) or query
        if self.connected:
            with self.driver.session(database=self.database) as s:
                records = [r.data() for r in s.run(text, **params)]
            self.last_source = "your database"
            return pd.DataFrame(records)
        saved = SAVED / f"{name}.json" if name else None
        if saved is not None and saved.exists():
            self.last_source = "the kit's saved result (no database connected)"
            return pd.DataFrame(json.loads(saved.read_text())["rows"])
        raise RuntimeError("No Neo4j database is connected, and the kit has no saved result for this query. "
                           "Connect a database (see T.connect) to run your own Cypher.") from None

    def maybe(self, query, **params):
        """Like run(), but when neither a database nor a saved result is available, print a short note and
        return None instead of raising (used for the GDS cells, whose saved results come from the lecturer's
        capture run)."""
        try:
            return self.run(query, **params)
        except Exception as e:
            print(f"Not run: {e}".split(". Connect")[0] + ".")
            return None

    def run_script(self, script, **params):
        """Run several Cypher statements separated by ';' (needs a connected database)."""
        if not self.connected:
            raise RuntimeError("No Neo4j database is connected (see T.connect).")
        for stmt in [s.strip() for s in script.split(";") if s.strip()]:
            used = {k: v for k, v in params.items() if "$" + k in stmt}
            with self.driver.session(database=self.database) as s:
                s.run(stmt, **used).consume()

    def close(self):
        if self.driver: self.driver.close()

def connect(folder=".", quiet=False):
    """Connect to your Neo4j database with the credentials found by _credentials(); otherwise return a session
    that uses the kit's saved results. Never prints the password."""
    cred = _credentials(folder)
    if cred is None:
        if not quiet:
            print("No Neo4j credentials found: the named queries will return the kit's saved results.\n"
                  "To use your own database, add its credentials file (NEO4J_URI=..., NEO4J_USERNAME=..., "
                  "NEO4J_PASSWORD=...) next to this notebook, then run this cell again.")
        return Neo4jSession()
    from neo4j import GraphDatabase
    uri, user, pwd = cred
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pwd)); driver.verify_connectivity()
    except Exception as e:
        if not quiet:
            print(f"Could not connect to {uri} ({type(e).__name__}). The named queries will return the kit's saved "
                  "results. Check that your database is running and the credentials are current.")
        return Neo4jSession()
    if not quiet: print(f"Connected to {uri} as {user}.")
    return Neo4jSession(driver)

LOAD_CYPHER = {
    "constraint": "CREATE CONSTRAINT {label}_id IF NOT EXISTS FOR (n:{label}) REQUIRE n.id IS UNIQUE",
    "nodes": "UNWIND $rows AS row MERGE (n:{label} {{id: row.id}}) SET n.name = row.name",
    "edges": ("UNWIND $rows AS row MATCH (a:{src} {{id: row.source}}), (b:{dst} {{id: row.target}}) "
              "MERGE (a)-[:{type}]->(b)"),
}

def load_into_neo4j(db, nodes, edges, batch=5000, progress=True):
    """Load a Hetionet subset into Neo4j with the three statements in T.LOAD_CYPHER: one uniqueness constraint
    per label, then nodes and relationships in batches of `batch` rows (UNWIND). Safe to run twice (MERGE)."""
    if not db.connected:
        raise RuntimeError("No Neo4j database is connected (see T.connect).")
    t0 = time.perf_counter(); kind = dict(zip(nodes.id, nodes.label))
    for label in sorted(nodes.label.unique()):
        db.run_script(LOAD_CYPHER["constraint"].format(label=label))
    for label, part in nodes.groupby("label"):
        rows = part[["id", "name"]].to_dict("records")
        for i in range(0, len(rows), batch):
            db.run_script(LOAD_CYPHER["nodes"].format(label=label), rows=rows[i:i + batch])
    e = edges.assign(src=edges.source.map(kind), dst=edges.target.map(kind))
    for (src, typ, dst), part in e.groupby(["src", "type", "dst"]):
        rows = part[["source", "target"]].to_dict("records")
        for i in range(0, len(rows), batch):
            db.run_script(LOAD_CYPHER["edges"].format(src=src, dst=dst, type=typ), rows=rows[i:i + batch])
        if progress: print(f"  {src}-[:{typ}]->{dst}: {len(rows):,}")
    return time.perf_counter() - t0

def capture(db, folder=SAVED, load=True):
    """Lecturer tool: on an EMPTY connected database, load the drug-interaction example and the Hetionet subset,
    run every named query in the order the notebooks use them, and save each result as folder/<name>.json
    (the kit's saved results). Returns {name: rows}."""
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True); out = {}
    def keep(name, df):
        rows = json.loads(df.to_json(orient="records"))
        (folder / f"{name}.json").write_text(json.dumps({"query": (QUERIES | GDS_QUERIES)[name],
            "captured": time.strftime("%Y-%m-%d"), "source": "captured from Neo4j", "rows": rows}))
        out[name] = len(rows)
    if load:
        db.run_script(DDI_CYPHER_CREATE, **ddi_cypher_params())
    keep("ddi_rule", db.run("ddi_rule")); keep("ddi_cleanup", db.run("ddi_cleanup"))
    if load:
        nodes, edges = subset(NEO4J_METAEDGES); load_into_neo4j(db, nodes, edges, progress=False)
    for name in QUERIES:
        if name not in ("ddi_rule", "ddi_cleanup"):
            keep(name, db.run(name))
    db.run("gds_drop")
    summary = project_genes(db); (folder / "gds_project.json").write_text(json.dumps(
        {"query": GDS_QUERIES["gds_project"], "captured": time.strftime("%Y-%m-%d"), "source": "captured from Neo4j",
         "rows": json.loads(summary.to_json(orient="records"))})); out["gds_project"] = len(summary)
    for name in ("gds_pagerank_write", "gds_pagerank_read", "gds_louvain_write", "gds_louvain_read", "gds_drop"):
        keep(name, db.run(name))
    return out

# =============================================================================================
# Graph algorithms: NetworkX counterparts and comparisons
# =============================================================================================
def nx_pagerank(g, alpha=0.85):
    """NetworkX PageRank (scores sum to 1; dangling nodes spread their score over all nodes)."""
    return pd.Series(nx.pagerank(g, alpha=alpha, max_iter=100, tol=1e-6)).rename("pagerank")

def gds_formula_pagerank(g, damping=0.85, iterations=20):
    """PageRank as Neo4j GDS documents it, re-implemented here only to explain the differences: every node
    starts at 1 - d and receives (1 - d) + d * sum(score / out-degree); scores are not normalised. This is
    NOT a GDS result; the real one comes from your database (db.run('gds_pagerank'))."""
    nodes = list(g); idx = {v: i for i, v in enumerate(nodes)}
    a = nx.to_scipy_sparse_array(g, nodelist=nodes, format="csr").astype(float)
    deg = np.asarray(a.sum(1)).ravel(); s = np.full(len(nodes), 1 - damping)
    for _ in range(iterations):
        s = (1 - damping) + damping * (a.T @ (s / np.maximum(deg, 1)))
    return pd.Series(s, index=nodes).rename("pagerank_gds_formula")

def nx_louvain(g, seed=SEED, resolution=1.0):
    """NetworkX Louvain communities as a Series node -> community number (largest community first)."""
    comms = sorted(nx.community.louvain_communities(g, seed=seed, resolution=resolution), key=len, reverse=True)
    return pd.Series({v: i for i, c in enumerate(comms) for v in c}).rename("community")

def modularity(g, partition):
    """Modularity of a partition given as node -> community."""
    groups = {}
    for v, c in partition.items(): groups.setdefault(c, set()).add(v)
    return float(nx.community.modularity(g, list(groups.values())))

def compare_rankings(a, b, k=20):
    """Agreement of two score Series on the same nodes: Spearman correlation and overlap of the top k."""
    from scipy.stats import spearmanr
    common = a.index.intersection(b.index)
    rho = spearmanr(a[common], b[common]).statistic
    top = len(set(a[common].nlargest(k).index) & set(b[common].nlargest(k).index)) / k
    return {"nodes": len(common), "spearman": float(rho), f"top{k}_overlap": float(top)}

def compare_partitions(g, p1, p2):
    """Agreement of two partitions: adjusted Rand index, normalised mutual information, and each modularity."""
    from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
    common = p1.index.intersection(p2.index)
    return {"ARI": float(adjusted_rand_score(p1[common], p2[common])),
            "NMI": float(normalized_mutual_info_score(p1[common], p2[common])),
            "communities_1": int(p1.nunique()), "communities_2": int(p2.nunique()),
            "modularity_1": modularity(g, p1), "modularity_2": modularity(g, p2)}

# =============================================================================================
# Knowledge-graph embedding with PyTorch Geometric
# =============================================================================================
TRANSE_HAND = {   # the worked example of the student notes (normalisation omitted for a hand calculation)
    "Paris": [0.12, -0.45, 0.31], "capital_of": [0.05, 0.33, -0.21],
    "France": [0.67, 0.23, -0.18], "Berlin": [-0.23, 0.41, 0.19], "margin": 1.0, "lr": 0.1}

def transe_hand_step(v=TRANSE_HAND):
    """One margin-loss step for (Paris, capital_of, France) against the corruption (Berlin, capital_of, France)."""
    p, r, f, b = (np.array(v[k]) for k in ("Paris", "capital_of", "France", "Berlin"))
    dp, dn = p + r - f, b + r - f; d1, d2 = np.linalg.norm(dp), np.linalg.norm(dn)
    loss = max(0, v["margin"] + d1 - d2); u, u2 = dp / d1, dn / d2; lr = v["lr"]
    p2, f2, b2, r2 = p - lr * u, f + lr * u - lr * u2, b + lr * u2, r - lr * (u - u2)
    e1, e2 = np.linalg.norm(p2 + r2 - f2), np.linalg.norm(b2 + r2 - f2)
    return {"d_pos": d1, "d_neg": d2, "loss": loss, "grad_r": u - u2, "Paris": p2, "capital_of": r2,
            "France": f2, "Berlin": b2, "d_pos_after": e1, "d_neg_after": e2,
            "loss_after": max(0, v["margin"] + e1 - e2)}

def kge_triples(metaedges=KGE_METAEDGES):
    """Integer-coded triples for embedding: (triples array [n, 3] of head, relation, tail), entity ids, relation
    names (metaedge abbreviations) and the entity names."""
    e = hetionet_edges(metaedges); n = hetionet_nodes()
    ents = pd.Index(pd.unique(pd.concat([e.source, e.target]))); rels = pd.Index(sorted(e.metaedge.unique()))
    t = np.column_stack([ents.get_indexer(e.source), rels.get_indexer(e.metaedge), ents.get_indexer(e.target)])
    names = dict(zip(n.id, n.name))
    return t.astype(np.int64), ents, rels, [names.get(x, x) for x in ents]

def split_triples(t, frac=0.1, seed=SEED):
    """Random train/validation/test split of the triples (frac each for validation and test). A held-out triple
    whose head or tail would not appear in training is moved back to training, because a TransE model has no
    vector for an entity it never saw (the transductive limit)."""
    rng = np.random.default_rng(seed); order = rng.permutation(len(t)); k = int(frac * len(t))
    val, test, train = t[order[:k]], t[order[k:2 * k]], t[order[2 * k:]]
    seen = set(train[:, 0]) | set(train[:, 2])
    ok = lambda x: np.array([h in seen and tt in seen for h, _, tt in x], dtype=bool)
    mv, mt = ok(val), ok(test)
    train = np.vstack([train, val[~mv], test[~mt]])
    return {"train": train, "val": val[mv], "test": test[mt], "moved_back": int((~mv).sum() + (~mt).sum())}

def make_kge(name, n_ent, n_rel, dim=64, margin=None):
    """A PyTorch Geometric knowledge-graph embedding model: 'TransE', 'RotatE', 'ComplEx' or 'DistMult'."""
    import torch_geometric.nn.kge as kge
    cls = getattr(kge, name)
    kw = {"margin": margin} if margin is not None and name in ("TransE", "RotatE", "DistMult") else {}
    return cls(n_ent, n_rel, hidden_channels=dim, **kw)

def train_kge(model, train, epochs=40, lr=0.01, batch=1024, seed=SEED, log_every=0):
    """Train with the model's own loss (PyG corrupts a head or a tail at random for each positive triple)."""
    import torch
    torch.manual_seed(seed); np.random.seed(seed)
    t = torch.as_tensor(train)
    loader = model.loader(head_index=t[:, 0], rel_type=t[:, 1], tail_index=t[:, 2], batch_size=batch, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=lr); losses = []; t0 = time.perf_counter()
    for epoch in range(epochs):
        total = 0.
        for h, r, tt in loader:
            opt.zero_grad(); loss = model.loss(h, r, tt); loss.backward(); opt.step(); total += float(loss.detach()) * len(h)
        losses.append(total / len(train))
        if log_every and (epoch + 1) % log_every == 0: print(f"epoch {epoch + 1}: loss {losses[-1]:.4f}")
    return {"loss": losses, "seconds": time.perf_counter() - t0}

def score_all(model, part, side, ents):
    """Scores of every entity as the missing tail (side='tail') or head of each triple in `part` (B x n_ent).
    For TransE the distances to all entities are computed at once (torch.cdist), which is much faster than
    scoring the candidate triples one by one and gives the same numbers as the model's own forward()."""
    import torch, torch.nn.functional as F
    B, n = len(part), len(ents)
    if type(model).__name__ == "TransE":
        E = F.normalize(model.node_emb.weight, p=model.p_norm, dim=-1); R = model.rel_emb.weight
        if side == "tail":
            q = E[part[:, 0]] + R[part[:, 1]]                     # h + r, compared with every t
        else:
            q = E[part[:, 2]] - R[part[:, 1]]                     # t - r, compared with every h
        return -torch.cdist(q, E, p=model.p_norm)
    rep = lambda c: part[:, c].repeat_interleave(n); cand = ents.repeat(B)
    return (model(rep(0), rep(1), cand) if side == "tail" else model(cand, rep(1), rep(2))).view(B, n)

def filtered_ranks(model, test, known, n_ent, batch=64, candidates=None):
    """Filtered ranks of the true tail and the true head of every test triple.

    For (h, r, ?) every entity e gets the score of (h, r, e); the other entities that make a KNOWN true triple
    (in training, validation or test) are removed, so the model is not penalised for ranking another correct
    answer first; the rank of the true tail is 1 + the number of remaining entities scored higher (ties count
    half). The same is done for (?, r, t). Returns an array of 2 * len(test) ranks: tails first, then heads.

    candidates: optional dict relation -> (allowed heads, allowed tails) as arrays of entity indices, to rank
    only entities of the right type (for example only diseases as the tail of 'treats')."""
    import torch
    tails, heads = {}, {}
    for h, r, t in known:
        tails.setdefault((h, r), []).append(t); heads.setdefault((r, t), []).append(h)
    allowed = None
    if candidates is not None:
        allowed = {}
        for r, (hs, ts) in candidates.items():
            for side, ids in (("head", hs), ("tail", ts)):
                m = torch.ones(n_ent, dtype=torch.bool); m[torch.as_tensor(ids)] = False; allowed[(r, side)] = m
    ranks = []; model.eval(); ents = torch.arange(n_ent)
    with torch.no_grad():
        for side in ("tail", "head"):
            for i in range(0, len(test), batch):
                part = torch.as_tensor(test[i:i + batch]); B = len(part)
                s = score_all(model, part, side, ents)
                true = part[:, 2] if side == "tail" else part[:, 0]
                target = s[torch.arange(B), true].clone()
                rows, cols = [], []                       # known true answers other than the one being ranked
                for j, (hh, rr, tt) in enumerate(part.tolist()):
                    others = tails[(hh, rr)] if side == "tail" else heads[(rr, tt)]
                    rows += [j] * len(others); cols += others
                s[torch.as_tensor(rows), torch.as_tensor(cols)] = -float("inf")
                if allowed is not None:
                    for j, rr in enumerate(part[:, 1].tolist()):
                        s[j, allowed[(rr, side)]] = -float("inf")
                higher = (s > target[:, None]).sum(1).float()
                ties = (s == target[:, None]).sum(1).float()        # the true answer itself was set to -inf above
                ranks.append((1 + higher + 0.5 * ties).numpy())
    return np.concatenate(ranks)

def type_candidates(triples, ents):
    """For each relation, the entities of the right kind as heads and tails (taken from the Hetionet id prefix,
    e.g. 'Compound::DB00331'), so that ranking can be restricted to entities of the right type."""
    kind = np.array([e.split("::")[0] for e in ents]); out = {}
    for r in np.unique(triples[:, 1]):
        sub = triples[triples[:, 1] == r]
        hk, tk = kind[sub[0, 0]], kind[sub[0, 2]]
        out[int(r)] = (np.flatnonzero(kind == hk), np.flatnonzero(kind == tk))
    return out

def rank_metrics(ranks):
    """MRR, Hits@1, Hits@3, Hits@10 and mean rank from a list of ranks."""
    r = np.asarray(ranks, dtype=float)
    return {"MRR": float(np.mean(1 / r)), "Hits@1": float(np.mean(r <= 1)), "Hits@3": float(np.mean(r <= 3)),
            "Hits@10": float(np.mean(r <= 10)), "MR": float(np.mean(r)), "n": int(len(r))}

def metrics_by_relation(ranks, test, rels):
    """rank_metrics for each relation (tail and head ranks together)."""
    r = np.asarray(ranks); rel = np.concatenate([test[:, 1], test[:, 1]])
    return pd.DataFrame({rels[k]: rank_metrics(r[rel == k]) for k in np.unique(rel)}).T

def random_mrr(n_candidates):
    """Expected MRR of a uniformly random ranking among n candidates: (1/n) * sum(1/k)."""
    return float(np.mean(1 / np.arange(1, n_candidates + 1)))

# =============================================================================================
# Coverage and bias
# =============================================================================================
def gini(x):
    """Gini coefficient of non-negative values: 0 when all are equal, near 1 when a few hold almost everything."""
    x = np.sort(np.asarray(x, dtype=float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum())) if x.sum() else 0.0

def degree_concentration(metaedge, side="target", top=0.01):
    """How unevenly one metaedge's edges fall on its nodes: Gini of the degrees and the share of edges held by
    the top `top` fraction of nodes (for example genes in CbG, 'compound binds gene')."""
    e = hetionet_edges([metaedge]); d = e[side].value_counts()
    k = max(1, int(round(top * len(d))))
    return {"metaedge": metaedge, "nodes": int(len(d)), "edges": int(len(e)), "gini": gini(d.values),
            f"top{int(top * 100)}pct_share": float(d.iloc[:k].sum() / d.sum())}

def coverage_table():
    """Simple coverage checks on drugs and diseases: how many have at least one edge of each kind."""
    n = hetionet_nodes(); e = hetionet_edges(["CtD", "CpD", "CbG", "CcSE", "DaG"])
    comp = set(n.id[n.kind == "Compound"]); dis = set(n.id[n.kind == "Disease"])
    has = lambda m, side: set(e[e.metaedge == m][side])
    rows = [("Compounds that treat at least one disease (CtD)", len(has("CtD", "source") & comp), len(comp)),
            ("Compounds with at least one target gene (CbG)", len(has("CbG", "source") & comp), len(comp)),
            ("Compounds with at least one side effect (CcSE)", len(has("CcSE", "source") & comp), len(comp)),
            ("Diseases with at least one treatment (CtD)", len(has("CtD", "target") & dis), len(dis)),
            ("Diseases with at least one associated gene (DaG)", len(has("DaG", "source") & dis), len(dis))]
    t = pd.DataFrame(rows, columns=["check", "count", "of"]); t["share"] = (t["count"] / t["of"]).round(3)
    return t

def reference():
    """The frozen results shown on the slides (reference/results.json)."""
    return json.loads((ROOT / "reference" / "results.json").read_text())
