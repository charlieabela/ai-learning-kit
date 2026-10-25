"""ICS5111 Mining Large Scale Data (MSc in AI and MSc in Data Science).

The graph block covers graph analysis, graph
representation learning including GNNs, knowledge graphs, and RAG / GraphRAG,
with a different corpus per session matched to the technique (biomedical
abstracts for text-to-graph work, financial data for anomaly detection, legal
text among the options). Sessions here will bind such problems to tasks pitched
at postgraduate level. See docs/AUTHORING_A_SESSION.md.
"""

UNIT = {
    "code": "ICS5111",
    "title": "Mining Large Scale Data",
    "programme": "MSc Artificial Intelligence; MSc Data Science",
    "level": "postgraduate",
    "notes": ("Students can be expected to read code and documentation independently; guidance should "
              "focus on the method and its assumptions rather than on Python syntax."),
}

SESSIONS = {
    "S02": ".s02",      # Graph embeddings (no case problem; Cora from LINQS; notebooks check themselves)
}
