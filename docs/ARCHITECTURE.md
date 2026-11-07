# ai-learning-kit: architecture

## Purpose

A library for learning data science and AI by doing, independent of any course. It is
organised by **topic**: what is learnt. A topic supplies everything that is not its
focus (data, steps taught elsewhere, connections to databases and language models) as
**helpers**, so that learners concentrate on the topic, and every helper can be opened
for those who want to look inside. A topic may use a case **problem** from the problem
library (frozen fictional data with helper code that does the heavy lifting) and pose
graded **tasks**, whose answers are saved in an **answer book** next to the notebook
with **formative feedback** given at once; most topics have no tasks and their
notebooks check themselves. A **level** (introductory, intermediate, advanced,
research) says who is learning, so the same topic can serve different cohorts.
Assessment tooling for the lecturer is a separate, private artefact that reads the
answer books.

## Layers

```
ai_learning_kit/
  core/          the engine, independent of any topic or problem
    problem.py     Problem: sources, data folder, workspace(), peek(), helper modules
    session.py     stage(), task(), attach_guides(), render_guide(), scope_text(): the topic contract
    levels.py      the four levels and describe(level), the paragraph a tutor pitches its help to
    checks.py      numeric_check(), written(), any_of/all_of/at_least, saved_only
    feedback.py    Feedback (verdict, details, suggestion after N attempts), combine()
    answers.py     AnswerBook: JSON answer book, attempts, results, submit
    workfolder.py  where work is kept (notebook's folder; Drive in Colab; folder=)
    registry.py    topics/problems discovery; tutor_context(); catalog() as plain data
    help.py        explain(), source(): look inside a helper
  topics/        one package per topic
    __init__.py    REGISTRY {"knowledge-graphs": ".knowledge_graphs", ...}
    <topic>/       __init__.py (the contract: TOPIC, TITLE, LEVEL, STAGES, ...),
                   tools.py (T = kit.tools()), and its data/, saved/, site/ or reference/
  problems/      the problem library; each problem is independent of any topic
    __init__.py    REGISTRY {id: package}
    brightwater_q3/  case.py (PROBLEM), data/, extract.py, reference.py, pipeline.py, report.py
  datasets/      external datasets: owner, address, checksum, licence, citation;
                 fetch() downloads from the owner, checks and caches; prefetch()
  llm/           optional language-model support (never required)
    providers.py   OpenAICompatible (OpenAI, Ollama, vLLM, LM Studio...), Anthropic, Null
    __init__.py    configure(), autodetect(), complete(), hint(), ask()
  dev/           tools for authors
    validate.py    python -m ai_learning_kit.dev.validate [--problem id | --topic name | --notebook path]
    notebooks.py   the notebook contract for notebook files anywhere: check_notebook(), stamp(), install_line()
    saved_copies.py  compare (and capture) the saved copies of web pages and API responses
    reference.py   docs/reference/: one page per topic, generated from its stages and helpers' docstrings;
                   page(name) gives one page to a front end
    release.py     python -m ai_learning_kit.dev.release X.Y.Z: version and docs
  __init__.py    the learner's API: start, tools, task, answer, check, progress, submit, sync,
                 sources, workspace, peek, path, extract/pipeline/report/reference (proxies),
                 explain, source, hint, ask, catalog, topic
templates/       problem/ and topic/ skeletons to copy
docs/            this file, the authoring guides, the generation prompts, LLM.md, sites/
tests/           pytest suite, including simulated Colab and a fake LLM server
```

Dependencies point downwards only: `topics` import `core`, `problems` and, where one
topic reuses another's data or helpers, that topic (data-cleaning reuses web-data's
loaders; graph-neural-networks reuses graph-embeddings' Cora parser); `problems`
import `core`; `core` imports nothing above it; `llm` is imported by the learner's
API and may be used by topics, but no topic requires it.

No file of the package names a study unit or a numbered course session:
`python -m ai_learning_kit.dev.validate` fails if one does. Courses decide which topics
they use and at which level; the kit does not know about them.

## The contracts

**Topic** (`core/session.py`, defaults in `core/registry.py`, guide in
`docs/AUTHORING_A_TOPIC.md`): a package with `TOPIC` (lower case words joined by
hyphens) and `TITLE`, and optionally `LEVEL` (the level its material is written for),
`BUILDS_ON` (topics whose ideas are assumed), `PROBLEM` (None for a topic without a
case problem), `TASKS` ({} when the notebooks check themselves), `TOOLS` (the helper
module `kit.tools()` returns), `DATASETS` (external datasets `kit.start` fetches),
`EXTRAS` (the pip extra its notebooks install) and `STAGES`, which say what is in
focus, what is available on request, what the kit does for the learner and what is out
of scope. Each task has `title`, `question`, `needs_value`, `needs_reasoning`, `check`
(value to Feedback), `written` (text to Feedback) and a four-part `guide` (what, why,
example, record). Checks compare with reference values computed from the data, never
typed by hand.

**Problem** (`core/problem.py`): a dataclass with `id`, `title`, `question`,
`summary`, `domain`, `sources` (short name to file), `descriptions`,
`text_sources`, `peek_hints`, `modules` (namespace to module path), `workspace_name`,
`tags`. The data folder is `problems/<id>/data/` and is shipped in the wheel.
Helper modules are ordinary Python modules; the kit exposes them to learners as
`kit.<namespace>` for the problem of the current topic.

**Feedback** (`core/feedback.py`): `ok` (True, False, or None when the lecturer
reads the answer), a one-line `summary`, itemised `details` with evidence, an
optional `suggestion` shown after `REVEAL_AFTER` attempts, and `open_points` for
written answers with checklist points not found.

## What a learner sees

```python
import ai_learning_kit as kit
kit.start("knowledge-graphs")          # a topic whose notebooks check themselves
T = kit.tools()
nodes = T.hetionet_nodes()

kit.start("data-formats", student="Name")      # a topic with graded tasks
kit.workspace(); kit.peek("erp"); kit.task("T3.3")
kit.extract.policy_rules()                     # the problem's helpers, by namespace
kit.answer("T3.3", value=39, reasoning="...")  # saved at once; feedback printed
kit.hint("T3.3")                               # only if a language model is configured
kit.submit()
```

The same calls work for any topic and any problem, because `kit.extract` and
friends are proxies to the current topic's problem.

## Where files go

`kit.start` finds the notebook's folder (Jupyter and VS Code start the kernel
there; in Colab the kit connects Drive, asks the runtime which notebook is open and
locates it in Drive) and makes it the current folder. The answer book is
`answers/<topic>_<student>.json`; the case files are copied to
`<workspace_name>/`; anything saved with a plain name lands beside them. A
notebook that is not in the learner's own Drive falls back to
`My Drive/ai-learning-kit/<topic>` with advice to save a copy.
`folder=` overrides this (in Colab, relative to My Drive).

## Language models

Optional. One provider class covers every OpenAI-compatible server, so an
open-weight model behind Ollama, vLLM or LM Studio is a base URL, not new code.
Autodetection reads environment variables and Colab secrets. The model is used
only by `kit.hint()`, `kit.ask()` and whatever a topic author chooses; it never
grades, and the answer book records only the kit's deterministic checks.

## Data

External datasets are downloaded from their owners, never copied into the kit
(`datasets/`): each is recorded with its owner, address, checksum, licence note and
citation. `kit.start` fetches a topic's datasets: from the data folder, beside the
notebook, from added folders (synced Google Drive or OneDrive folders, Drive in
Colab), from the owner, from lecturer-approved course copies, and from locations
given with `datasets.use()` (files, folders, or Google Drive, OneDrive/SharePoint and
Dropbox links). Every copy is checked against the checksum and kept in the data
folder: `AILK_DATA_DIR` (Notebook Tutor sets it to its clone's `data/`), the kit
clone's `data/`, My Drive › ai-learning-kit › data in Colab, or the user's cache
folder. Small research datasets are stored in `datasets/small_data/` with their
citations; fictional case files and reference results are packaged with their
problem or topic.

## Notebooks

The kit stores no notebooks: the lecturer keeps and distributes them, Notebook Tutor
keeps each learner's copy in its clone, and Colab opens them from Drive.
`docs/NOTEBOOK_CONTRACT.md` is the rule book: every notebook opens with the install
line and `kit.start("<topic>")`, contains nothing that depends on where it runs,
names no course and links to no private Drive, and carries
`metadata["ai_learning_kit"]`. `python -m ai_learning_kit.dev.notebooks check|stamp`
works on notebook files wherever they are.

## Versioning and compatibility

Releases are tagged; notebooks install a tag (`@v1.2.0`). A problem's data and a
released topic's helpers and tasks are never changed afterwards: fix a problem by
adding a new id, change a helper by adding a new function. Earlier tags remain
installable, so a notebook keeps working against the tag it installs (notebooks
written for releases before 1.0.0 start a course session and keep installing the
release they name).

## Notebook tutors and other front ends

A notebook tutor can treat this kit as its back end:

- `kit.catalog(include_tasks=True)` returns every topic (with its level), task,
  problem and dataset as JSON-able data, and the four levels;
- each notebook carries `metadata["ai_learning_kit"]` with the topic, title, role,
  kit version and install line, and optionally a level, so a tutor can open a
  notebook and know what it is about; `registry.tutor_context(topic, level=...)`
  gives the level's description, the scope, the tasks, the helpers and the datasets;
- `dev.reference.page(topic)` is the topic's reference page as Markdown (what is in focus, on request and
  support, and every helper), generated from the installed kit, for showing to learners;
- answer books are plain JSON with attempts, results and timestamps per task;
- `Feedback` objects (not only their printed form) are returned by `check`
  functions, so a tutor can render them its own way;
- the LLM layer is pluggable: a tutor can set `llm.configure(...)` or register
  its own provider.
