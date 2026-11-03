# Authoring a topic

A topic is what is learnt: knowledge graphs, data cleaning, graph neural networks. It is
independent of any course: the kit never names a study unit, a programme or a week. Which
topics a course uses, in which order and at which level is part of the course's own material
(the notebooks' titles, the reading sheets), not of the kit.

A notebook starts the topic it is about and gets its helpers:

```python
import ai_learning_kit as kit
kit.start("knowledge-graphs")
T = kit.tools()
```

A topic is one of two kinds:

- **Notebooks that check themselves** (prediction-then-check cells, comparisons with the
  helpers): the topic supplies the helpers, the data and the reference results. Most topics
  are of this kind (`knowledge-graphs`).
- **Graded tasks**: the topic binds one problem from the problem library to a sequence of
  tasks, recorded in an answer book with formative feedback. See *Topics with graded tasks*.

Either kind's notebooks follow `docs/NOTEBOOK_CONTRACT.md`; they are kept and distributed with
the course's material, not in the kit.

## Keep the learner on the focus: three layers

Every topic is built in three layers, so that a learner can concentrate on one thing without
having to go into anything else:

| Layer | What it is | In the topic module |
|---|---|---|
| **Focus** | the few ideas and functions the notebook is about; learners write, change or inspect them | `FOCUS` stages |
| **Support** | everything the topic needs but does not teach (loading and splitting data, steps taught in other topics, connections to databases or language models): one call each | `BLACK_BOX` stages, and helpers in `tools.py` |
| **Look inside** | a readable version of a support helper, for learners who want to see how it works, or to ask a tutor about it | `ON_REQUEST` stages; `kit.source(T.<name>)` prints the helper |

Write that table before any code: list what learners do, what the kit does for them, and which
support steps deserve a look inside. A step that is support in one topic can be the focus of
another (splitting a graph's edges is support in `knowledge-graphs` and focus in
`graph-embeddings`); say so with `taught_in="the graph-embeddings topic"`.

## Declare the scope

Record the layers in the topic module as `STAGES`, so that `kit.hint()`, `kit.ask()` and
Notebook Tutor pitch their explanations to them:

```python
from ...core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

STAGES = {
    "cypher": stage("Cypher: MATCH patterns, WHERE, aggregation and paths", FOCUS),
    "loading": stage("Loading nodes and relationships with UNWIND and MERGE", ON_REQUEST,
                     how="T.load_into_neo4j()", note="the three Cypher statements are in focus"),
    "split": stage("Splitting edges into training, validation and test sets", BLACK_BOX,
                   how="T.split_edges()", taught_in="the graph-embeddings topic"),
    "rgcn": stage("Relational GCNs", OUT_OF_SCOPE, taught_in="the further reading of this topic"),
}
```

`FOCUS`: learners do it; explain in depth. `ON_REQUEST`: a helper does it and the notebook
offers an optional look inside; a tutor says what goes in and out, and explains how it works
only when asked, from the helper's code. `BLACK_BOX`: a helper does it; say what goes in and out
and why, not how. `OUT_OF_SCOPE`: do not bring it up; if asked, say briefly where it is taught.
Write `ON_REQUEST` helpers to be read, because `kit.source(T.name)` prints them.
`scope_text(module)` shows the instructions a tutor receives.

## Level

A topic declares the level its material is written for, `LEVEL = "advanced"`, one of
`introductory`, `intermediate`, `advanced` and `research` (`ai_learning_kit/core/levels.py`
describes each). The level describes the learner, not the content: the same topic can be used
at another level, and a notebook can record the level it is pitched at in its metadata. A tutor
combines the level (who is learning) with the topic's stages (what is in scope).

## Steps

1. Choose the name: lower-case words joined by hyphens, naming the subject
   (`text-to-graphs`, `data-cleaning`). Create the package
   `ai_learning_kit/topics/<name_with_underscores>/`.
2. In its `__init__.py` define, with a docstring that says what the topic covers:
   - `TOPIC = "<name>"`, `TITLE`, `LEVEL`, and `BUILDS_ON` (topics whose ideas are assumed);
   - `PROBLEM = None` (or a problem id) and `TASKS = {}` (or the graded tasks);
   - `TOOLS = ".tools"`: the helper module that `kit.tools()` returns;
   - `DATASETS = ("cora",)`: external datasets `kit.start` downloads from their owners
     (register new ones in `ai_learning_kit/datasets/__init__.py` with owner, address,
     checksum, size, licence note and citation);
   - `EXTRAS`: the pip extra the notebooks install (`""`, `"graphs"` or `"advanced"`);
   - `STAGES`, as above.
3. Write `tools.py`, with a docstring that says what the module is for. Read external data with
   `datasets.fetch(name)`; small research datasets from the library that ships them; synthetic
   data from a seeded generator. Give every check a clear PASS or FAIL line and a hint.
4. Keep frozen results of the runs shown on the slides in `reference/`, read as
   `ROOT / "reference" / ...`.
5. Register the topic in `ai_learning_kit/topics/__init__.py` (`REGISTRY`), add tests in
   `tests/test_topic_<name>.py`, and run
   `python -m ai_learning_kit.dev.validate --topic <name>`. The validator also fails if a
   topic's files name a study unit or a numbered course session; refer to other topics by name.
6. Write the notebooks to the contract, run
   `python -m ai_learning_kit.dev.notebooks check <the notebook files>`, execute each notebook
   end to end against the kit, and distribute them executed.

A topic whose notebooks **read the web** keeps a saved copy of every page and API response its
notebooks request, so that they run offline and give the same results every time: the helper
`T.get()` sends the request and falls back to the copy when the site cannot be reached. List the
API requests in the helpers' `SAVED_REQUESTS` with their files in `saved/`, and keep a fictional
website in `site/` (published with GitHub Pages from `docs/sites/<site>/`; a test keeps the two
identical). CI runs `python -m ai_learning_kit.dev.saved_copies check`.

## Topics with graded tasks

The topic binds one problem from the problem library to a sequence of tasks (start from
`templates/topic_with_tasks/`; `data-formats` is the worked example). List the tasks in
the order of the work, each with what is recorded (a value, written reasoning, or both), how it
is checked, and the common wrong answers you expect.

1. Build `_ref()` from the problem's `reference` module, including the values that common
   mistakes produce, so that feedback can name the likely cause.
2. Write the tasks with `task(...)`: numbers with `numeric_check(...)`, written answers with
   `written(points, model)` (the model answer passes every point), anything else with a function
   returning `Feedback` with itemised evidence. Feedback says what was recorded, what is wrong,
   the evidence in the data and how to verify; the right answer appears only as the
   `suggestion`, after several attempts.
3. Write the guide with four parts per task (what, why, example, record), using a smaller
   example case than the problem so that the example never gives the answer away, and attach it
   with `attach_guides(TASKS, GUIDE, example_case=...)`. The depth of the guide follows the
   topic's `LEVEL`.

## Rules that keep old notebooks working

- Never change a released topic's task ids, questions or checks. Add a new topic instead.
- Never change a problem's data. Add a new problem id instead.
- Never change a released helper's results. Add a function instead.
- Never rename a released topic: notebooks start it by name.
