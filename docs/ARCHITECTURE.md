# ai-learning-kit: architecture

## Purpose

A sandbox library for the AI study units at the University of Malta. It holds a
library of **problems** (frozen datasets with helper code that does the heavy
lifting) and, for each **unit**, **sessions** that bind a problem to a sequence of
**tasks**. A session hides the steps that are not its topic, so students
concentrate on the step that is. Each task poses a question; answers are saved in
an **answer book** next to the notebook; **formative feedback** is given at once.
Assessment tooling for the lecturer is a separate, private artefact that reads
the answer books.

## Layers

```
ai_learning_kit/
  core/          the engine, independent of any unit or problem
    problem.py     Problem: sources, data folder, workspace(), peek(), helper modules
    session.py     task(), attach_guides(), render_guide(): the session contract
    checks.py      numeric_check(), written(), any_of/all_of/at_least, saved_only
    feedback.py    Feedback (verdict, details, suggestion after N attempts), combine()
    answers.py     AnswerBook: JSON answer book, attempts, results, submit
    workfolder.py  where work is kept (notebook's folder; Drive in Colab; folder=)
    registry.py    units/sessions/problems discovery; catalog() as plain data
    help.py        explain(), source(): look inside the black box
  problems/      the problem library; each problem is unit-independent
    __init__.py    REGISTRY {id: package}
    brightwater_q3/  case.py (PROBLEM), data/, extract.py, reference.py, pipeline.py, report.py
  units/         one package per study unit
    __init__.py    REGISTRY {code: package}; session lookup
    ari1101/       UNIT dict, SESSIONS {"S02": ".s02"}, s02.py, s02_guide.py
    ics2205/       UNIT dict, SESSIONS {}  (no sessions yet)
    ics5111/       UNIT dict, SESSIONS {"S02": ".s02"}
      s02/           session package: __init__.py (contract), tools.py (T = kit.tools()), reference/
  datasets/      external datasets: owner, address, checksum, licence, citation;
                 fetch() downloads from the owner, checks and caches; prefetch()
  llm/           optional language-model support (never required)
    providers.py   OpenAICompatible (OpenAI, Ollama, vLLM, LM Studio...), Anthropic, Null
    __init__.py    configure(), autodetect(), complete(), hint(), ask()
  dev/           tools for authors
    validate.py    python -m ai_learning_kit.dev.validate [--problem id | --session UNIT SNN | --notebook path]
    notebooks.py   the notebook contract: install_line(), check_notebook(), stamp(), colab_url()
    release.py     python -m ai_learning_kit.dev.release X.Y.Z: version, stamps, Colab index, docs
  __init__.py    the student API: start, tools, task, answer, check, progress, submit, sync,
                 sources, workspace, peek, path, extract/pipeline/report/reference (proxies),
                 explain, source, hint, ask, catalog
templates/       problem/ and session/ skeletons to copy
docs/            this file, the authoring guides, the generation prompts, LLM.md
notebooks/       student notebooks (notebooks/<UNIT>/<SESSION>/), README.md with Open-in-Colab links
tests/           pytest suite, including simulated Colab and a fake LLM server
```

Dependencies point downwards only: `units` import `core` and `problems`;
`problems` import `core`; `core` imports nothing above it; `llm` is imported by
the student API and may be used by sessions, but no session requires it.

## The three contracts

**Problem** (`core/problem.py`): a dataclass with `id`, `title`, `question`,
`summary`, `domain`, `sources` (short name to file), `descriptions`,
`text_sources`, `peek_hints`, `modules` (namespace to module path), `workspace_name`,
`tags`. The data folder is `problems/<id>/data/` and is shipped in the wheel.
Helper modules are ordinary Python modules; the kit exposes them to students as
`kit.<namespace>` for the problem of the current session.

**Session** (`core/session.py`, defaults in `core/registry.py`): a module or package
with `UNIT`, `SESSION` and `TITLE`, and optionally `PROBLEM` (None for a session
without a case problem), `TASKS` ({} when the notebooks check themselves), `TOOLS`
(the helper module `kit.tools()` returns), `DATASETS` (external datasets `kit.start`
fetches), `EXTRAS` (the pip extra its notebooks install), `NOTEBOOKS` (role to path)
and `STAGES`. Each task has `title`, `question`, `needs_value`,
`needs_reasoning`, `check` (value to Feedback), `written` (text to Feedback) and a
four-part `guide` (what, why, example, record). Checks compare with reference
values computed from the data, never typed by hand.

**Feedback** (`core/feedback.py`): `ok` (True, False, or None when the lecturer
reads the answer), a one-line `summary`, itemised `details` with evidence, an
optional `suggestion` shown after `REVEAL_AFTER` attempts, and `open_points` for
written answers with checklist points not found.

## What a student sees

```python
import ai_learning_kit as kit
kit.start("ARI1101", "S02", student="Name")   # work folder = the notebook's folder
kit.workspace(); kit.peek("erp"); kit.task("T3.3")
kit.extract.policy_rules()                     # the problem's helpers, by namespace
kit.answer("T3.3", value=39, reasoning="...")  # saved at once; feedback printed
kit.hint("T3.3")                               # only if a language model is configured
kit.submit()
```

The same calls work for any unit and any problem, because `kit.extract` and
friends are proxies to the current session's problem.

## Where files go

`kit.start` finds the notebook's folder (Jupyter and VS Code start the kernel
there; in Colab the kit connects Drive, asks the runtime which notebook is open and
locates it in Drive) and makes it the current folder. The answer book is
`answers/<UNIT>_<SESSION>_<student>.json`; the case files are copied to
`<workspace_name>/`; anything saved with a plain name lands beside them. A
notebook that is not in the student's own Drive falls back to
`My Drive/ai-learning-kit/<UNIT>/<SESSION>` with advice to save a copy.
`folder=` overrides this (in Colab, relative to My Drive).

## Language models

Optional. One provider class covers every OpenAI-compatible server, so an
open-weight model behind Ollama, vLLM or LM Studio is a base URL, not new code.
Autodetection reads environment variables and Colab secrets. The model is used
only by `kit.hint()`, `kit.ask()` and whatever a session author chooses; it never
grades, and the answer book records only the kit's deterministic checks.

## Data

External datasets are downloaded from their owners, never copied into the kit
(`datasets/`): each is recorded with its owner, address, checksum, licence note and
citation. `kit.start` fetches a session's datasets: from the data folder, beside the
notebook, from added folders (synced Google Drive or OneDrive folders, Drive in
Colab), from the owner, from lecturer-approved course copies, and from locations
given with `datasets.use()` (files, folders, or Google Drive, OneDrive/SharePoint and
Dropbox links). Every copy is checked against the checksum and kept in the data
folder: `AILK_DATA_DIR` (Notebook Tutor sets it to its clone's `data/`), the kit
clone's `data/`, My Drive › ai-learning-kit › data in Colab, or the user's cache
folder. Small research datasets are stored in `datasets/small_data/` with their
citations; fictional case files and reference results are packaged with their
problem or session.

## Notebooks

`docs/NOTEBOOK_CONTRACT.md` is the rule book: every notebook opens with the install
line and `kit.start`, contains nothing that depends on where it runs, and carries
`metadata["ai_learning_kit"]`. The validator checks it, CI executes every notebook on
three systems, and the release step stamps the version and rebuilds the index of
Open-in-Colab links.

## Versioning and compatibility

Releases are tagged; notebooks install a tag (`@v0.5.0`). A problem's data and a
released session's tasks are never changed afterwards: fix a problem by adding a
new id, extend a session by adding a new session. Earlier tags remain
installable, so a notebook keeps working against the tag it installs.

## Notebook tutors and other front ends

A notebook tutor can treat this kit as its back end:

- `kit.catalog(include_tasks=True)` returns every unit, session, task and problem
  as JSON-able data;
- each student notebook carries `metadata["ai_learning_kit"]` with the unit,
  session, role, kit version and install line, so a tutor can open a notebook
  and know which session it belongs to; `registry.tutor_context()` gives the
  cohort, the scope, the tasks, the helpers and the datasets;
- answer books are plain JSON with attempts, results and timestamps per task;
- `Feedback` objects (not only their printed form) are returned by `check`
  functions, so a tutor can render them its own way;
- the LLM layer is pluggable: a tutor can set `llm.configure(...)` or register
  its own provider.
