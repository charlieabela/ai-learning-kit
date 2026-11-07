# ai-learning-kit reference

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Topics

| Topic | Level | Builds on | Extra | What it covers |
|---|---|---|---|---|
| [`data-formats`](data-formats.md) | introductory | — | — | Data formats: CSV, JSON, XML and documents. Tasks, questions and formative checks on the Brightwater Q3 case. |
| [`web-data`](web-data.md) | introductory | data-formats | — | Web data: obtaining data from web pages and web APIs. |
| [`data-cleaning`](data-cleaning.md) | introductory | web-data | — | Data cleaning: finding missing values, making formats consistent, removing duplicates and filling gaps. |
| [`exploring-data`](exploring-data.md) | introductory | data-cleaning | — | Exploring data: descriptive statistics, distributions, correlation and visualisation. |
| [`modelling`](modelling.md) | introductory | exploring-data | — | Modelling: linear regression, logistic regression and k-nearest neighbours. |
| [`text-to-graphs`](text-to-graphs.md) | intermediate | — | graphs | Text to graphs: turning a corpus into graphs and exploring them (term-document graphs and their projections, co-occurrence with count and lift thresholds, k-nearest-neighbour document graphs, traversal, components and shortest paths, and what construction and preprocessing choices do to the result). |
| [`paths-and-connectivity`](paths-and-connectivity.md) | intermediate | text-to-graphs | graphs | Paths, connectivity and structure: what a shortest path, a reachable set and a structural measure mean. |
| [`centrality-and-link-analysis`](centrality-and-link-analysis.md) | intermediate | text-to-graphs, paths-and-connectivity | graphs | Centrality and link analysis: what "important" means in a graph, and how degree, betweenness, closeness, PageRank and HITS each answer it. |
| [`graph-embeddings`](graph-embeddings.md) | advanced | — | graphs | Graph embeddings: random walks, skip-gram, node2vec, evaluation and TransE. |
| [`graph-neural-networks`](graph-neural-networks.md) | advanced | graph-embeddings | advanced | Graph neural networks: message passing, GCN, depth, GraphSAGE, limits and tasks. |
| [`knowledge-graphs`](knowledge-graphs.md) | advanced | graph-embeddings, graph-neural-networks | advanced | Knowledge graphs: property graphs and RDF, Neo4j and Cypher, graph algorithms in the database, TransE for link prediction, coverage and bias. |
| [`retrieval-augmented-generation`](retrieval-augmented-generation.md) | advanced | knowledge-graphs | advanced | Retrieval-augmented generation (RAG) and Graph-RAG. |

## Levels

- **introductory**: Learners who are new to programming and to data science, still getting used to Python, NumPy, pandas and Matplotlib.
- **intermediate**: Learners who are comfortable with basic Python and pandas and know the fundamentals of data analysis, but are new to the methods of this topic.
- **advanced**: Learners who are fluent in Python and read code and documentation independently, at the level of a postgraduate course.
- **research**: Learners working at research level, for example doctoral students, who relate a method to the literature and to their own problems.

## The kit's own functions

Available in every topic, as `kit.<name>` after `import ai_learning_kit as kit`.

- **`kit.start(topic: str, student: str = '', folder=None, drive=None) -> None`**: Start (or continue) a topic: choose the work folder, fetch the topic's data and, for a topic with graded tasks, open your answer book.
- **`kit.tools()`**: The current topic's helper functions, used as T = kit.tools() (after kit.start).
- **`kit.task(task_id: str) -> None`**: Explain a task: what to do, why it matters, a typical example and what to record.
- **`kit.answer(task_id: str, value=None, reasoning: str | None = None, feedback: bool = True) -> None`**: Save your answer to a task (a value, your reasoning, or both) and show feedback.
- **`kit.check(task_id: str, value=None, reasoning: str | None = None) -> None`**: Give formative feedback on a value and/or reasoning without saving it or counting an attempt.
- **`kit.progress() -> pd.DataFrame`**: Show which tasks have been answered.
- **`kit.submit(download: bool = True)`**: Finalise the answer book, list anything missing, and download it in Colab.
- **`kit.sync() -> None`**: Make sure everything saved so far has reached Google Drive (Colab only).
- **`kit.sources() -> pd.DataFrame`**: List the current problem's evidence sources: short name, file name, size and description.
- **`kit.workspace(folder=None, overwrite: bool = False)`**: Copy the current problem's case files into a folder next to this notebook and return its path.
- **`kit.peek(name: str, lines: int = 12) -> None`**: Print the first lines of a source exactly as stored, without interpreting it.
- **`kit.path(name: str)`**: The path of a source file inside the installed kit (students normally use workspace()).
- **`kit.explain(obj) -> None`**: Print the plain-language description of a kit function or module.
- **`kit.source(obj) -> None`**: Print the Python source code of a kit function, so you can see how it works.
- **`kit.hint(task_id: str) -> None`**: Ask the configured language model for one hint on a task, never the answer.
- **`kit.ask(task_id: str, question: str) -> None`**: Ask the configured language model a question about a task (what a term means, why a step is needed).
- **`kit.catalog(include_tasks: bool = False) -> dict`**: Every topic, problem and dataset in this version of the kit, as plain data.
- **`kit.problem()`**: The current problem (its sources, files and helper modules).
- **`kit.topic()`**: The current topic module (TOPIC, TITLE, LEVEL, PROBLEM, TASKS, ...).
