# Authoring a session

A session is one of two kinds:

- **With graded tasks**: it binds one problem from the problem library to a
  sequence of tasks, recorded in an answer book with formative feedback (ARI1101
  S02). Most of this guide is about this kind.
- **Without graded tasks**: its notebooks check themselves (prediction-then-check
  cells, comparisons with the helpers), and the session supplies the helpers,
  the data and the reference results (ICS5111 S02). See *Sessions whose notebooks
  check themselves* below.

Either kind's notebooks follow `docs/NOTEBOOK_CONTRACT.md`; they are kept and
distributed with the session's material, not in the kit.

A session with tasks binds one problem to a sequence of tasks for a unit's learning
outcomes. The session decides the split: which steps students do themselves
(the topic of the session) and which steps the kit does for them (everything
else). The same problem can be used by several sessions and units with a
different split each time.

## Design the split first

Write one table before any code:

| Students do (this session's topic) | The kit does (hidden, via helpers) |
|---|---|
| ... | ... |

Then list the tasks in the order of the work, each with: what is recorded (a
value, written reasoning, or both), how it is checked, and the common wrong
answers you expect. Aim for 15 to 30 tasks for a two-hour session, mixing
quick checks with three or four written answers.

## Declare the scope for tutors

Record the split in the session module as `STAGES`, so that `kit.hint()`,
`kit.ask()` and Notebook Tutor pitch their explanations to it:

```python
from ..core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

STAGES = {
    "parse": stage("Reading CSV, JSON and XML with pandas, json and ElementTree", FOCUS),
    "crawl": stage("Following 'Next page' links automatically", ON_REQUEST, how="T.scrape_listings()",
                   note="students loop over known pages themselves; explain the helper only when asked"),
    "doc_extract": stage("Extracting rules from Word, Excel and PDF", BLACK_BOX, how="kit.extract",
                         taught_in="ICS2205 Part A", note="say only that the kit reads the document"),
    "cleaning": stage("Cleaning and de-duplicating records", OUT_OF_SCOPE, taught_in="the ARI1101 Scrub session"),
}
```

The four levels are: `FOCUS` (students do it; explain in depth), `ON_REQUEST` (a
helper does it and the notebook offers an optional *look inside*: a tutor says what
goes in and out, and explains how it works only when the student asks, from the
helper's code), `BLACK_BOX` (a helper does it; never explained inside) and
`OUT_OF_SCOPE`. Use `ON_REQUEST` for steps that are harder than the cohort needs but
that a curious student may reasonably want to understand; write those helpers to be
read, because `kit.source(T.name)` prints them.

A step that is a black box in one unit can be in focus in another: ICS2205 can
make `doc_extract` its topic with the same problem. `scope_text(module)` shows
the instructions a tutor receives.

## Pitch

Read the unit's `UNIT["notes"]`. For a first-year unit, every task gets a
step-by-step "what you do", names the variables in the cell, and shows the idea
on a tiny example. For a postgraduate unit, guidance can focus on the method and
its assumptions. The four-part guide (what, why, example, record) is required
either way; its depth is what changes.

## Steps

1. Copy `templates/session/sNN.py` and `sNN_guide.py` into
   `ai_learning_kit/units/<unit>/` with the real session code.
2. Define `UNIT`, `SESSION`, `TITLE`, `PROBLEM`, and `STAGES` from the split.
3. Build `_ref()` from the problem's `reference` module, including the values that
   common mistakes produce (the unconverted count, the inner-merge count, the
   `>=` count), so feedback can name the likely cause.
4. Write the tasks with `task(...)`:
   - numbers: `numeric_check(_ref, key, how_to_check, explanation, wrong, evidence, right)`;
   - written answers: `written(points, model)`, where each point is (what a good
     answer covers, a key-word test, a hint if missing) and the model answer
     passes every point;
   - anything else: a function returning `Feedback` with itemised evidence.
   Feedback must say what was recorded, what is wrong, the evidence in the data,
   and how to verify; the right answer appears only as the `suggestion`, after
   several attempts.
5. Write the guide with four parts per task; use a different, smaller example case
   than the problem so the example never gives the answer away.
6. `attach_guides(TASKS, GUIDE, example_case=...)`.
7. Register the session in the unit's `SESSIONS` and run
   `python -m ai_learning_kit.dev.validate --session <UNIT> <SNN>`.
8. Build the notebook from the session with the lecturer's builder (the task
   cells are generated from the guide, so notebook and `kit.task()` agree) and run
   the worked-answers copy end to end.

## Sessions whose notebooks check themselves

Make the session a package, `ai_learning_kit/units/<unit>/sNN/`, holding:

- `__init__.py` with `UNIT`, `SESSION`, `TITLE`, `PROBLEM = None` (or a problem id),
  `TASKS = {}`, and
  - `TOOLS = ".tools"`: the helper module that `kit.tools()` returns;
  - `DATASETS = ("cora",)`: external datasets `kit.start` downloads from their owners
    (register new ones in `ai_learning_kit/datasets/__init__.py` with owner, address,
    checksum, size, licence note and citation);
  - `EXTRAS = "graphs"`: the pip extra the notebooks install;
  - `STAGES`, as above, so that tutors know what is in focus;
- `tools.py`: the helpers, with a docstring that says what the module is for. Read
  external data with `datasets.fetch(name)`; small research datasets from the
  library that ships them (networkx, scikit-learn); synthetic data from a seeded
  generator;
- `reference/`: frozen results of the run shown on the slides, read as
  `ROOT / "reference" / ...`.

A session whose notebooks **read the web** (ARI1101 S03) keeps a saved copy of every
page and API response its notebooks request, so that they run offline and give the
same results every time: the helper `T.get()` sends the request and falls back to the
copy when the site cannot be reached. List the API requests in the helpers'
`SAVED_REQUESTS` with their files in `saved/` (`saved.json` records status and content
type), and keep a course website in `site/` (published with GitHub Pages from
`docs/sites/<unit>-<session>/`; a test keeps the two identical). CI runs
`python -m ai_learning_kit.dev.saved_copies check`, which compares every copy with the
live service and attaches the live response when it differs or is missing;
`... saved_copies capture` writes them on a machine that can reach the services.

Then write the notebooks to the contract, run
`python -m ai_learning_kit.dev.validate --session <UNIT> <SNN>` and
`python -m ai_learning_kit.dev.notebooks check <the notebook files>`, execute each
notebook end to end against the kit, and distribute them executed.

## Rules that keep old notebooks working

- Never change a released session's task ids, questions or checks. Add a new
  session instead.
- Never change a problem's data. Add a new problem id instead.
- Never change a released helper's results. Add a function, or a new session.
- Tag every release with `python -m ai_learning_kit.dev.release`; notebooks install a tag.

To draft a session from a topic and a problem with a language model, use
`docs/prompts/new_session.md`.
