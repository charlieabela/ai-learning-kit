# RAG and Graph-RAG

`retrieval-augmented-generation` · level **advanced** · builds on [`knowledge-graphs`](knowledge-graphs.md) · pip extra `advanced`

Retrieval-augmented generation (RAG) and Graph-RAG.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("retrieval-augmented-generation")
T = kit.tools()
```

Written for the **advanced** level: Learners who are fluent in Python and read code and documentation independently, at the level of a postgraduate course.

**Data downloaded from its owners** (checked against a checksum; `kit.datasets.status()` shows where it is kept):

- `hetionet_nodes`: Hetionet v1.0 nodes (TSV), from Hetionet project (Himmelstein et al.), GitHub hetio/hetionet. Hetionet's original content is CC0; it integrates 29 resources whose own licences vary (the hetionet repository links a per-source table). The kit downloads the files from the hetionet repository rather than copying them.
- `hetionet_edges`: Hetionet v1.0 edges (gzipped SIF), from Hetionet project (Himmelstein et al.), GitHub hetio/hetionet. Hetionet's original content is CC0; it integrates 29 resources whose own licences vary (the hetionet repository links a per-source table). The kit downloads the files from the hetionet repository rather than copying them.

## The three layers

### In focus: learners do these themselves

- The RAG pipeline: indexing offline (chunking, sparse and dense indexes) and answering online (retrieve, rerank, build the prompt, generate with citations); why retrieval helps a language model and what it does not fix
- Chunking: chunk size and overlap, whole sentences, and their effect on retrieval (with `T.chunk(), T.chunk_collection()`)
- Sparse retrieval with BM25: term frequency saturation, document-length normalisation, idf (with `T.BM25`)
- Dense retrieval: vectors and cosine similarity; latent semantic analysis (TF-IDF and SVD) computed here; neural sentence encoders where available; reciprocal rank fusion (with `T.LSA, T.neural(), T.rrf()`)
- Reranking: a second stage that rescores a short list with richer features (here a logistic regression; cross-encoders in practice) (with `T.Reranker, T.rerank_features()`)
- Generation: the prompt with numbered evidence and citation instructions; an extractive generator; an optional language model (with `T.build_prompt(), T.extractive_answer(), T.llm_answer()`)
- Graph-RAG: entity linking, subgraph retrieval from a typed knowledge graph, verbalising triples, an entity co-occurrence graph with personalised PageRank, community summaries for global questions, fusing graph and text evidence
- Evaluating retrieval (context recall and precision, Recall@k, MRR, nDCG) and grounding (faithfulness and citation precision); language-model judges and their biases (with `T.evaluate(), T.grounding()`)
- Risks: hallucination despite retrieval, indirect prompt injection, access control, extraction errors, stale or biased knowledge graphs, citations that do not support the claim, cost

### On request: the kit does these, and learners may look inside

- The code of the helpers (BM25 loop, SVD, PageRank call, grounding check) (with `kit.source(T.<name>)`)

### Support: the kit does these for the learner

- Building the MedQuAD subset and linking its topics to Hetionet diseases (with `T.medquad()`)

### Left out of this topic

- RAG frameworks (LangChain, LlamaIndex), vector databases and their APIs (taught in students' own projects)

## Helpers: `T = kit.tools()`

RAG and Graph-RAG: a minimal retrieval-augmented pipeline written in plain Python.

`kit.source(T.<name>)` prints a helper's code, `help(T.<name>)` its full description.

### Named in the focus

- **`T.chunk(text, size=100, overlap=20, by='words')`**: Split one text into chunks of about `size` words with `overlap` words shared between neighbours (by='words'), or into groups of whole sentences up to `size` words (by='sentences').
- **`T.chunk_collection(df, text_col='answer', id_col='id', **kw)`**: Chunk every document; each chunk keeps its parent document id (so that a hit on any chunk counts for the document when evaluating).
- **`class T.BM25(texts, k1=1.5, b=0.75, tokenizer=<function tokens>)`**: BM25 (Robertson and Zaragoza, 2009) written out: score(D, Q) = sum over query words w in D of idf(w) * f(w, D) * (k1 + 1) / (f(w, D) + k1 * (1 - b + b * |D| / avg|D|)), idf(w) = ln(1 + (N - n_w + 0.5) / (n_w + 0.5)).
- **`class T.LSA(texts, dim=100, seed=5111)`**: Dense vectors by latent semantic analysis: TF-IDF vectors compressed with a truncated SVD to `dim` dimensions; query and documents are compared by cosine similarity. Trained on the collection itself.
- **`T.neural(texts, **kw)`**: A NeuralEncoder, or None with a short note when the library or the model is not available here.
- **`T.rrf(rankings, k=60)`**: Reciprocal rank fusion (Cormack, Clarke and Buettcher, 2009): score(d) = sum over rankings of 1 / (k + rank).
- **`T.rerank_features(question, cand, bm_scores, dense_scores, docs)`**: One row of features per candidate document: the two first-stage scores and ranks, the share of query words the passage contains, whether the passage is about the condition named in the question, and whether its heading (question type) matches the question's type.
- **`class T.Reranker()`**: Logistic regression on rerank_features, trained on questions whose relevant passages are known.
- **`T.build_prompt(question, evidence)`**: The prompt a language model would receive: instructions, the question and the numbered evidence.
- **`T.extractive_answer(question, evidence, max_sentences=5, min_overlap=1)`**: A generator that needs no language model: it picks the evidence sentences that share the most content words with the question (and adds no words of its own), each followed by its citation [n].
- **`T.llm_answer(question, evidence, max_tokens=400)`**: The answer of a language model, when the kit can reach one (Notebook Tutor, or a key set with kit.llm.configure); otherwise None. A model is never required by the notebooks.
- **`T.grounding(answer, evidence, threshold=0.6)`**: Check every answer sentence against the evidence it cites: support = the share of the sentence's content words found in the cited evidence. Returns a table and the share of sentences supported (faithfulness proxy) and the share of citations that point to evidence containing the sentence's words (citation precision).
- **`T.evaluate(run, questions, ks=(1, 3, 5, 10))`**: Mean Recall@k, MRR and nDCG@10 of a run: {question index: ranked list of passage ids}.

### Support

- **`T.medquad()`**: The MedQuAD subset: one row per answer passage (id, source, focus, synonyms, qtype, question, answer, hetionet_disease: the Hetionet disease name the focus links to, or None for distractors).

### Not yet placed in a layer

No stage of the topic names these helpers, so a tutor cannot tell whether learners are meant to write this kind of code, look inside it, or only use it. Name each in a stage's `how` to place it.

- **`T.formulary()`**: The formulary as a DataFrame: id (the drug), text.
- **`T.two_hop_questions()`**: The 13 two-hop questions and their gold evidence (the passage on the named drug plus every drug linked to it through a shared enzyme in the opposite role).
- **`T.medquad_questions(mq, seed=5111, n=None)`**: Evaluation questions: the MedQuAD question of every passage about a Hetionet disease, with gold = all passages on the same focus and question type (from any NIH source). Optionally a random sample of n.
- **`T.with_synonym(questions)`**: The same questions with the focus replaced by its first MedQuAD synonym (when one exists), to test retrieval when the question and the passage use different words for the same condition.
- **`T.sentences(text)`**: Split text into sentences (a simple rule: a full stop, question or exclamation mark followed by a space).
- **`T.stem(w)`**: A deliberately small suffix stripper, so that inhibits, inhibiting, inhibitor and inhibition meet as 'inhibit' (real systems use the Porter or Snowball stemmer, or none with a neural encoder).
- **`T.tokens(text)`**: Lower-cased, lightly stemmed word tokens without stop words (the same function for queries and documents).
- **`T.bm25_hand()`**: The three-document example of the notes: scores and per-word contributions.
- **`class T.NeuralEncoder(texts, model='sentence-transformers/all-MiniLM-L6-v2')`**: A pretrained sentence encoder (sentence-transformers, default all-MiniLM-L6-v2). Only works where the library is installed and the model can be downloaded (Colab, Notebook Tutor's advanced install); see neural().
- **`T.rank(scores, k=None)`**: Indices of the documents in decreasing order of score (ties broken by index).
- **`T.question_type(question)`**: A guess of the question's type from its words (symptoms, treatment, causes, ...).
- **`T.link_entities(text, names, aliases=None)`**: Entity linking by dictionary: every known name (or alias) that occurs in the text as whole words, returned as the canonical names in order of appearance. aliases maps an alias to its canonical name.
- **`T.entity_graph(passages, names=('simvastatin', 'atorvastatin', 'rosuvastatin', 'pravastatin', 'midazolam', 'warfarin', 'clopidogrel', 'codeine', 'aspirin', 'clarithromycin', 'azithromycin', 'itraconazole', 'fluconazole', 'amiodarone', 'omeprazole', 'fluoxetine', 'paroxetine', 'metformin', 'amoxicillin', 'paracetamol', 'CYP3A4', 'CYP2C9', 'CYP2C19', 'CYP2D6'))`**: Entity co-occurrence graph built from text: entities are nodes; two entities named in the same passage are joined. Returns (G, mentions) where mentions[passage id] lists the entities the passage names.
- **`T.ppr_rank(G, mentions, seeds, alpha=0.5)`**: Personalised PageRank from the question's entities (HippoRAG): teleport only to the seeds; each passage is scored by the probability of the entities it names, divided by the square root of their number.
- **`T.typed_graph()`**: The formulary's typed knowledge graph (as in knowledge-graphs): drug -[METABOLISED_BY]-> enzyme, drug -[INHIBITS]-> enzyme.
- **`T.kg_retrieve(G, entity)`**: Subgraph retrieval for one drug: its enzyme edges, then the drugs on the other side of each enzyme in the opposite role (two hops: drug -> enzyme <- drug). Returns a list of triples (head, relation, tail).
- **`T.verbalise(triples, source='knowledge graph')`**: Turn triples into short sentences a generator can quote, each with its provenance.
- **`T.hetionet_facts(disease, kinds=('CtD', 'CpD', 'DpS', 'DaG'), limit=None)`**: Hetionet triples about one disease (treatments, palliation, symptoms, associated genes), as (head, relation, tail) with names, from the knowledge-graphs helpers.
- **`T.context_recall(retrieved, gold, k)`**: Share of the gold passages that are among the top k retrieved.
- **`T.context_precision(retrieved, gold, k)`**: Share of the top k retrieved passages that are gold.
- **`T.mrr(retrieved, gold)`**: Reciprocal rank of the first relevant passage (0 if none is retrieved).
- **`T.ndcg(retrieved, gold, k)`**: nDCG@k with binary relevance: sum of 1 / log2(rank + 1) over relevant passages in the top k, divided by the best possible value.
- **`T.reference()`**: The frozen results shown on the slides (reference/results.json).

### Constants

`T.BM25_HAND` (dict), `T.DATA` (Path), `T.ENZYMES` (tuple), `T.FORMULARY` (dict), `T.FORMULARY_NAMES` (tuple), `T.GLOBAL_QUESTION` (str), `T.INHIBITS` (dict), `T.INJECTION` (str), `T.MEDQUAD_CITATION` (str), `T.PROMPT` (str), `T.QTYPE_WORDS` (dict), `T.RERANK_FEATURES` (tuple), `T.ROOT` (Path), `T.SEED` (int), `T.STOP` (set), `T.SUBSTRATE_OF` (dict), `T.VERBS` (dict), `T.VERSION` (str)
