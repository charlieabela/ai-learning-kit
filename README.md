# ai-learning-kit

A library for learning data science and AI by doing. It is organised by **topic**, what is
learnt (knowledge graphs, data cleaning, graph neural networks), independently of any course,
and it works at four **levels**: introductory, intermediate, advanced and research.

The kit names no course: courses choose the topics they use and the level they work at. A
notebook starts the topic it is about. The kit then supplies everything that is not the
topic's focus, so that the learner can concentrate on the topic:

- **Focus**: the few ideas and functions the notebook is about, which learners write, change
  or inspect.
- **Support**: everything else the topic needs (data, steps taught in other topics,
  connections to databases and language models), one call each.
- **Look inside**: a readable version of a support helper, for learners who want to see how it
  works (`kit.source(T.<name>)`), or to ask a tutor about it.

A topic either supplies **helpers and data** to notebooks that check themselves, or binds a
**problem** from the kit's library (a frozen set of data sources in several formats) to a
sequence of graded **tasks** with an answer book and formative feedback. The notebooks
themselves are not stored here: they are kept and distributed with a course's material, and run
unchanged in Notebook Tutor and in Colab because everything they depend on comes from the kit.

All companies, people and documents in the problems are fictional. Real datasets (Cora,
Hetionet, SciFact) are not copied into the kit: it downloads them from their owners, as described under
*Data* below.

## Notebooks

Learners open a topic's notebooks in
[Notebook Tutor](https://github.com/charlieabela/notebook-tutor), which ships the kit and keeps
each learner's copy, with their work, in its own clone; or in Colab. Every notebook follows one
contract (`docs/NOTEBOOK_CONTRACT.md`): it starts with the kit's install line and
`kit.start("<topic>")`, and nothing in it depends on where it runs. Check and stamp a notebook
file wherever it is:

```
python -m ai_learning_kit.dev.notebooks check 1_prepare.ipynb
python -m ai_learning_kit.dev.notebooks stamp 1.0.0 1_prepare.ipynb
```

## Use in a notebook (Notebook Tutor, Colab, Jupyter or VS Code)

```python
%pip install -q "ai-learning-kit[advanced]@https://github.com/charlieabela/ai-learning-kit/archive/refs/tags/v1.0.0.zip"
import ai_learning_kit as kit
kit.start("knowledge-graphs")      # downloads the topic's datasets from their owners, once
T = kit.tools()                    # the topic's helper functions
nodes = T.hetionet_nodes()
```

A topic with graded tasks keeps an answer book next to the notebook:

```python
kit.start("<topic>", student="Your Name")
kit.task("T1.1")                   # what to do, why, a typical example, what to record
kit.answer("T1.1", value=..., reasoning="...")
kit.progress(); kit.submit()
```

The install line uses GitHub's archive of the release, so it needs only pip, not git.
`kit.catalog()` lists every topic, problem and dataset in the installed version;
`kit.explain(fn)` and `kit.source(fn)` show what any kit function does and how.

**Where work is kept.** `kit.start` finds the folder of the running notebook and makes it the
current folder: the answer book (`answers/`), the case files and every file saved with a plain
name sit next to the notebook. In Colab the kit connects Google Drive and finds the notebook
there; `kit.sync()` pushes pending writes, and `kit.submit()` calls it. `folder=` chooses another
folder (in Colab, relative to My Drive).

**What to install.** The plain line covers the core libraries (pandas, NumPy, Matplotlib,
scikit-learn, requests, Beautiful Soup). Topics on graphs add `[graphs]` (networkx, NLTK), and topics on neural networks,
graph databases and RDF add `[advanced]` (PyTorch, PyTorch Geometric, the Neo4j driver, rdflib
and owlrl). Each topic records its extra, and its notebooks install it.

**Data.** `ai_learning_kit.datasets` records, for every external dataset, its owner, download
address, checksum, licence note and citation. The kit downloads it from the owner the first time
a notebook needs it, checks the checksum, and keeps it in its data folder. When the owner's site
cannot be reached, it uses a copy from somewhere else, checked the same way:

```python
kit.datasets.use("cora", "~/Downloads/cora.tgz")                       # a file on this computer
kit.datasets.add_folder("~/Library/CloudStorage/GoogleDrive-me/My Drive/data")   # a synced folder
kit.datasets.use("cora", "https://drive.google.com/file/d/<id>/view")   # a shared copy (Drive, OneDrive, Dropbox)
kit.datasets.status()                                                   # what is where
```

A file dropped next to the notebook, or in its `data/` folder, is found too. The data folder is
the `data/` folder of the clone when the kit or Notebook Tutor runs from a git clone (ignored by
git), My Drive › ai-learning-kit › data in Colab with Drive connected, and the user's cache folder
otherwise. Small research datasets (the karate club, Les Misérables, the Florentine families,
Davis's Southern women) are stored in the kit with their citations: `kit.datasets.small("karate")`.

**Language models are optional.** With a model configured (a Gemini, Anthropic or OpenAI API
key, or an open-weight model behind Ollama, vLLM or any OpenAI-compatible server),
`kit.hint(task)` gives one next step and `kit.ask(task, question)` explains a concept. Without
one, everything else works unchanged. See `docs/LLM.md`.

## Topics in this version

Each topic's level is the level its material is written for by default; a notebook or a tutor
can pitch the same topic at another level. A topic's notebooks may assume the topics it builds on.

| Topic | Level | What it covers | Data | Extra |
|---|---|---|---|---|
| `data-formats` | introductory | reading CSV, JSON and XML, choosing formats for extracted data, converting and combining sources, with graded tasks and an answer book | the `brightwater_q3` problem | |
| `web-data` | introductory | requesting web pages and web APIs, reading HTML with Beautiful Soup and JSON replies, scraping responsibly (robots.txt, delays) | the fictional Campus Lets website and the Open-Meteo API, with saved copies so notebooks run offline | |
| `data-cleaning` | introductory | finding missing values, making text, numbers and dates consistent, removing duplicates, filling gaps and measuring the error of each method | the data web-data collects, supplied ready-collected | |
| `exploring-data` | introductory | descriptive statistics, distributions, outliers, correlation and visualisation; a preview of sampling | the Palmer penguins, a year of daily weather for Malta, Anscombe's quartet | |
| `modelling` | introductory | linear regression, logistic regression and k-nearest neighbours with scikit-learn, with the data prepared by hand (encoding, splits, scaling) | the Palmer penguins, Auto MPG | |
| `text-to-graphs` | intermediate | graphs as data structures (adjacency lists and matrices), BFS and DFS, connected and strongly connected components, shortest paths and turning similarity into distance, concept co-occurrence graphs (counts and lift) and k-nearest-neighbour document graphs, and what preprocessing choices do to a graph | SciFact abstracts (Wadden et al., 2020, from its owners), a six-abstract miniature | `graphs` |
| `graph-embeddings` | advanced | random walks, skip-gram, node2vec, evaluating embeddings, TransE | Cora (from LINQS), small classic graphs | `graphs` |
| `graph-neural-networks` | advanced | message passing, GCN, depth and over-smoothing, GraphSAGE, the limits of GNNs, node classification and link prediction | Cora (from LINQS), a five-paper graph, synthetic graphs | `advanced` |
| `knowledge-graphs` | advanced | property graphs and RDF, Neo4j and Cypher, graph algorithms in the database, TransE for link prediction, coverage and bias | Hetionet v1.0 (from GitHub), a drug-interaction example | `advanced` |
| `retrieval-augmented-generation` | advanced | a RAG pipeline in plain Python (chunking, BM25, dense retrieval, fusion, reranking, grounding, retrieval metrics) and Graph-RAG; no language model required | a teaching formulary, a MedQuAD subset (CC BY 4.0), Hetionet | `advanced` |

**brightwater_q3**: the Q3 2026 expenditure compliance review at Brightwater Logistics Ltd.
Seven sources in seven formats (DOCX policy, CSV payments, JSON bank feed, XML approvals, XLSX
vendor list, TXT emails, PDF board report), eleven planted breaches, two split purchases,
duplicated bank lines and an over-counting Board report.

## Structure

```
ai_learning_kit/topics     one package per topic: its helpers, data, reference results and scope
ai_learning_kit/core       the engine: topics, levels, checks, feedback, answer book, work folder
ai_learning_kit/problems   the problem library (frozen case data with helper code)
ai_learning_kit/llm        optional language-model providers and the tutor helpers
ai_learning_kit/datasets   external datasets: owner, address, checksum, licence; download and cache
ai_learning_kit/dev        validator (problems, topics), the notebook checker, and the release step
templates/                 skeletons for a new problem, a new topic, and a topic with graded tasks
docs/                      ARCHITECTURE, AUTHORING_A_PROBLEM, AUTHORING_A_TOPIC, LLM, prompts/
```

`docs/ARCHITECTURE.md` explains the layers and contracts. To add a problem or a topic, follow the
authoring guides; `docs/prompts/` holds prompts that draft either with a language model, for the
lecturer to review. `python -m ai_learning_kit.dev.validate` checks every contract.

## Development

```
pip install -e ".[dev,graphs]"
pytest -q
python -m ai_learning_kit.dev.validate          # problems and topics
python -m ai_learning_kit.dev.release 1.0.1     # a release: version and docs
```

CI runs the tests and the validator on Linux, macOS and Windows, and checks that
every registered dataset's owner still serves the exact file (an owner that refuses
the connection is reported as a warning).

## Licence

Copyright © 2026 Charlie Abela, University of Malta.

ai-learning-kit is free software: you can redistribute it and/or modify it under the terms of the
GNU Affero General Public License, version 3 (AGPL-3.0-only), as published by the Free Software
Foundation; see [LICENSE](LICENSE). It is distributed in the hope that it will be useful, but
without any warranty; without even the implied warranty of merchantability or fitness for a
particular purpose.

In short: you may use, study, change and share the kit, including in teaching. If you distribute
the kit or a work based on it, or let people use a modified version over a network (for example
inside a hosted notebook service), you must make the complete source of that version available
under the same licence. Releases up to v0.11.0 were published under GPL-3.0-or-later.

Datasets the kit downloads or stores keep their own licences, recorded with each dataset
(`kit.datasets.catalog()` and the `data/README.md` of each topic that stores data).
