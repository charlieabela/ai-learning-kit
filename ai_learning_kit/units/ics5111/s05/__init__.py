"""ICS5111 Session 05: retrieval-augmented generation (RAG) and Graph-RAG.

A session without graded tasks, like Sessions 02 to 04: its notebooks check themselves. The kit supplies the
helpers (T = kit.tools()): a minimal RAG pipeline written in plain Python (chunking, BM25, latent semantic analysis,
fusion, a small learned reranker, an extractive generator, a grounding check, retrieval metrics) and Graph-RAG
components (entity linking, subgraph retrieval from the Session 04 graph and Hetionet, verbalising triples,
personalised PageRank over an entity graph, community summaries). Two collections ship with the session: a
20-passage formulary written for teaching and a CC BY 4.0 subset of MedQuAD; Hetionet comes from Session 04.

No language model is required: generation is extractive by default, and a model is used only when the kit can
reach one. Every concept is offered at three depths (the idea, the mechanics, going deeper); the notebooks mark the
mechanics and depth cells as optional.
"""
from ....core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

UNIT = "ICS5111"
SESSION = "S05"
TITLE = "RAG and Graph-RAG"
PROBLEM = None
TASKS = {}
TOOLS = ".tools"
DATASETS = ("hetionet_nodes", "hetionet_edges")
EXTRAS = "advanced"

_DEPTH = ("Explain in three stages and stop at the first unless the student asks for more or the cell itself shows "
          "the equation: (1) the idea in words, with the formulary or a MedQuAD example; (2) the mechanics, the "
          "formula with every symbol defined and the numbers of the worked example; (3) the depth, why the method "
          "behaves as it does. Cells headed 'Optional' are the mechanics or depth stage: when asked about one, go "
          "through it step by step and check it against the printed numbers; never make a student feel that skipping "
          "it is a failure.")

STAGES = {
    "pipeline": stage("The RAG pipeline: indexing offline (chunking, sparse and dense indexes) and answering online "
                      "(retrieve, rerank, build the prompt, generate with citations); why retrieval helps a language "
                      "model and what it does not fix", FOCUS, note=_DEPTH),
    "chunking": stage("Chunking: chunk size and overlap, whole sentences, and their effect on retrieval", FOCUS,
                      how="T.chunk(), T.chunk_collection()"),
    "sparse": stage("Sparse retrieval with BM25: term frequency saturation, document-length normalisation, idf",
                    FOCUS, how="T.BM25", note=_DEPTH),
    "dense": stage("Dense retrieval: vectors and cosine similarity; latent semantic analysis (TF-IDF and SVD) computed "
                   "here; neural sentence encoders where available; reciprocal rank fusion", FOCUS,
                   how="T.LSA, T.neural(), T.rrf()", note=_DEPTH),
    "rerank": stage("Reranking: a second stage that rescores a short list with richer features (here a logistic "
                    "regression; cross-encoders in practice)", FOCUS, how="T.Reranker, T.rerank_features()",
                    note=_DEPTH),
    "generation": stage("Generation: the prompt with numbered evidence and citation instructions; an extractive "
                        "generator; an optional language model", FOCUS, how="T.build_prompt(), "
                        "T.extractive_answer(), T.llm_answer()",
                        note="a language model is never required; say so if a student has none configured"),
    "graphrag": stage("Graph-RAG: entity linking, subgraph retrieval from a typed knowledge graph, verbalising "
                      "triples, an entity co-occurrence graph with personalised PageRank, community summaries for "
                      "global questions, fusing graph and text evidence", FOCUS, note=_DEPTH),
    "evaluation": stage("Evaluating retrieval (context recall and precision, Recall@k, MRR, nDCG) and grounding "
                        "(faithfulness and citation precision); language-model judges and their biases", FOCUS,
                        how="T.evaluate(), T.grounding()", note=_DEPTH),
    "risks": stage("Risks: hallucination despite retrieval, indirect prompt injection, access control, extraction "
                   "errors, stale or biased knowledge graphs, citations that do not support the claim, cost", FOCUS),
    "data_prep": stage("Building the MedQuAD subset and linking its topics to Hetionet diseases", BLACK_BOX,
                       how="T.medquad()", note="say what is in the subset and why distractors are included"),
    "helper_code": stage("The code of the helpers (BM25 loop, SVD, PageRank call, grounding check)", ON_REQUEST,
                         how="kit.source(T.<name>)"),
    "frameworks": stage("RAG frameworks (LangChain, LlamaIndex), vector databases and their APIs", OUT_OF_SCOPE,
                        taught_in="students' own projects",
                        note="the session deliberately builds the pipeline without frameworks"),
}
