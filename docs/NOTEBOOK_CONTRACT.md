# The notebook contract

A notebook built on the kit runs unchanged in **Notebook Tutor** (on the student's own computer)
and in **Colab**, and on a best-effort basis in plain Jupyter. That works because nothing in the
notebook depends on where it runs: the kit finds the work folder, provides the helpers and fetches
the data.

A notebook is about one **topic** (`knowledge-graphs`, `data-cleaning`), and the kit never names a
course: which topics a course uses, and when, is part of the course's own material.

**Where notebooks live.** The kit stores no notebooks. The lecturer keeps them with the rest of the
course's material (Drive, the VLE) and students take them from there: in Notebook Tutor they open
the file, and the tutor keeps their copy, with their work, in its own clone; in Colab they open it
from Drive. The kit holds what notebooks depend on (each topic's helpers, data sources and
reference results) and checks a notebook file wherever it is:

```
python -m ai_learning_kit.dev.notebooks check  1_prepare_setup.ipynb 2_explore_walkthrough.ipynb ...
python -m ai_learning_kit.dev.notebooks stamp 1.0.0 1_prepare_setup.ipynb ...     # install line + metadata
```

`check` tests every rule marked **(checked)** below. Before a notebook is handed to students it is
also executed from top to bottom against the kit version it names, and saved with its outputs.

## 1. Names

`<n>_<role>_<name>.ipynb`, for example `2_explore_walkthrough.ipynb`, with any prefix the course
uses for its own ordering; no spaces. Roles: `setup` (checks the learner's environment before the
topic starts), `prepare`, `explore` (the walkthrough), `practice`, or `complete` for one notebook that
covers every stage. Solved (lecturer) versions are kept with the lecturer's resources and never
given to students. A practice notebook runs from top to bottom with its blanks unfilled (each
exercise returns `None` or prints a prompt until it is completed), so it can be executed and checked
in its student form.

## 2. The opening: two cells, always the same

```python
%pip install -q "ai-learning-kit[graphs]@https://github.com/charlieabela/ai-learning-kit/archive/refs/tags/v1.2.0.zip"
```

```python
import ai_learning_kit as kit
kit.start("knowledge-graphs")     # fetches the topic's datasets the first time
T = kit.tools()                   # the topic's helper functions

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
SEED = T.SEED
```

- The first code cell is exactly the install line for the kit version the notebook was checked
  with, with the topic's pip extra (`EXTRAS` in the topic module: none for the core libraries,
  `graphs` for networkx, `advanced` for PyTorch, PyTorch Geometric, Neo4j and RDF). It uses GitHub's archive of the release,
  which needs only pip, and is written without spaces around `@`, which lets Notebook Tutor
  recognise it and skip it (the tutor ships the kit). `stamp` writes it. **(checked)**
- The second code cell starts the topic and, when the topic has helpers, gets them as `T`.
  Imports the rest of the notebook relies on go in the same cell. (`kit.start` may come in the
  third cell after an imports cell; it must still come before any other use of the kit.)
  **(checked)**
- A topic with graded tasks starts with `kit.start(topic, student="Your Name")`; a topic whose
  notebooks check themselves needs no name.
- The markdown cell before the opening says, in two or three sentences, what the cells do and that
  nothing has to be downloaded or uploaded by hand when the download works. It does not explain
  Drive folders, ZIPs or paths.

## 3. Nothing that depends on the environment

No cell (after the install line) may **(checked)**:

- import `google.colab`, mount Drive, or call `files.upload()`;
- open, extract or mention ZIP files;
- change `sys.path`, or import helper files that sit next to the notebook;
- install packages (`%pip`, `!pip`, `subprocess ... pip`): a library the topic needs is added
  to the kit's pip extras instead;
- hard-code paths such as `/content/drive/...` or `MyDrive`.

Files a notebook writes use plain relative names (`outputs/comparison.csv`): `kit.start` has made
the notebook's folder (or, in Colab, its Drive folder) the current folder.

## 4. Data

- **Course datasets published by someone else** (Cora) are never copied into the kit or the
  notebook. Each is registered in `ai_learning_kit/datasets/__init__.py` with the owner, the
  owner's download address, the checksum of the exact file, its size, the licence note and the
  citation, and listed in the topic's `DATASETS`. `kit.start` fetches them, so a problem shows in
  the opening cell, and helpers open them with `datasets.fetch(name)`.
- **Where the kit looks**: its data folder; the notebook's folder and its `data/`; added folders
  (`kit.datasets.add_folder`, `AILK_DATA_PATH`); the owner's address; the course copies
  (`Source.mirrors`, then `data-copies.json` in the repository); locations given with
  `kit.datasets.use(name, file_folder_or_link)`; in Colab, an upload button in the cell. Every copy
  is checked against the checksum and kept in the data folder.
- **Course copies.** When an owner's site does not answer (LINQS refuses many cloud machines,
  Colab's among them), the kit downloads the same file from a course copy: an asset of a `data-*`
  release of the kit's repository (`data-1` holds `cora.tgz`), so students never notice. A copy is
  added only with the lecturer's approval, because it redistributes someone else's file. A copy
  listed in `data-copies.json` on `main` also reaches notebooks that install an earlier kit version.
  CI checks that every course copy serves the exact file.
- **When nothing works**, the message (shown without a traceback) says how to add the file where the
  student is: in Notebook Tutor with **Data files › Add data file…**, in Colab with the upload button
  the kit shows (or the Files panel), in Jupyter by putting it next to the notebook, using the copy
  the lecturer provides (for example on the VLE). In Notebook Tutor and Jupyter the student adds it
  once; the kit checks it and keeps it.
- **Each clone has its own data.** When the kit or Notebook Tutor runs from a git clone, the data
  folder is that clone's `data/` (ignored by git). It holds only the datasets of the topics that
  learner opened.
- **A student's own data** is not the kit's business: they add it with **Data files** next to their
  notebook and read it by path. Nothing is registered or checked.
- **Small research datasets** (Zachary's karate club, Les Misérables, the Florentine families,
  Davis's Southern women) are stored in the kit with their citations: `datasets.small("karate")`.
  Add another by storing it in `ai_learning_kit/datasets/small_data/` the same way.
- **Reference results** (frozen outputs of the run shown on the slides) are packaged with the
  topic, in `topics/<topic>/reference/`, and read through `T.ROOT / "reference"`.
- **Web pages and API responses** that a notebook requests are sent through the topic's
  helper (`T.get`, which behaves like `requests.get`), and the kit keeps a saved copy of each
  one: when the site cannot be reached, the copy is used, so the notebook still runs and gives
  the same results. A request the kit has no copy of is sent as written.

## 5. Helpers: focus, support and look inside

- A topic's helpers are one module (`TOOLS = ".tools"`), returned by `kit.tools()`. Every
  `T.<name>` a notebook uses must exist there. **(checked)**
- The notebook works on the topic's focus. Everything else it needs is a support helper, one call
  each, so that the learner never has to go into a step that is not the focus. Support steps that
  a curious learner may want to understand are offered as an optional *look inside*
  (`kit.source(T.<name>)`, or a cell headed "Optional: look inside"). The topic's `STAGES` record
  which is which (docs/AUTHORING_A_TOPIC.md).
- Helpers never change once a version is released, so a notebook that names that version keeps
  giving the numbers on its slides. A change is a new function.

## 6. Practice: from question to code

Learners can follow what finished code does and still not see how it came about. In a practice notebook, give
each exercise a **plan**: the steps that turn the question into code, before any code is written. Append it to
the exercise's statement (the markdown just before the exercise cell, which a tutor reads as the task), with a
bold label that names the stage, and repeat the steps as numbered comments in the starter code:

```markdown
**From question to code (plan given, with the tools).** *Goal:* one line per step.

1. Keep the payments above the threshold (a boolean filter, `df[df["amount"] > 5000]`).
2. Count them (`len`).
```

```python
# 1 keep the payments above the threshold     (a boolean filter)
# 2 count them                                 (len)
n_large = None   # TODO
```

The plans **fade** across the notebook, and the notebook's first cell says how:

| Stage | Label | The learner gets |
|---|---|---|
| 1 | plan given, with the tools | the steps, each with the function or method it needs |
| 2 | steps given, tools yours | the steps only |
| 3 | steps as labels only | numbered comments in the code |
| 4 | your plan | a goal; the learner writes the numbered comments first |
| 5 | questions instead of steps, or goal only | for evaluation and design exercises, and the last one |

The pace follows the level: introductory notebooks stay at stage 1 for two or three exercises and reach stage 4
at most once, at the end; advanced notebooks fade faster. The answers copy keeps the same comments with the
solution lines under them, so the plan and the code can be read together. Every step must match the code under
it. Notebook Tutor recognises the label and the numbered comments: its code check says which steps are done, and
its hints follow the plan. `dev.notebooks check` lists, as a note, the exercises of a practice notebook that have
no plan. The approach rests on subgoal labels (Margulieux, Guzdial and Catrambone, ICER 2012) and faded worked
examples (Renkl and Atkinson, *Educational Psychologist*, 2003).

## 7. Metadata

`metadata["ai_learning_kit"]` holds **(checked)**:

| Key | Meaning |
|---|---|
| `topic` | the topic in the kit, e.g. `"knowledge-graphs"` |
| `title` | the notebook's title as students see it |
| `role` | `setup`, `prepare`, `explore`, `practice` or `complete` |
| `level` | optional: `introductory`, `intermediate`, `advanced` or `research`, the level the notebook is pitched at (the topic's `LEVEL` when absent) |
| `kit_version`, `install` | written by `stamp` |
| `tasks` | the task ids the notebook answers (empty when it checks itself) |
| `runs_unfilled` | `false` only when blanks raise errors until filled (older notebooks) |

Notebook Tutor reads this block to recognise the notebook, to fetch the topic's scope and helpers
for its explanations at the notebook's level, and to say so when the notebook was written for
another kit version. (Notebooks handed out before 1.0.0 record a course and a session instead; they
install the earlier kit version they name and keep working.)

## 8. Saved outputs

Notebooks are distributed **executed**, so that a student who cannot run a cell still sees the
expected result. Saved outputs contain no errors, except in cells tagged `raises-exception` (a
cell that demonstrates an error on purpose). **(checked)**

## 9. Shared with others

Notebooks are given to learners and may be passed on, so **(checked)**:

- no cell names a study unit (three capitals and four digits) or a numbered course session
  ("Session 04"): refer to other topics by name ("as in graph-embeddings"), and let the course's own
  materials say where a notebook sits in a course;
- no cell links to Google Drive (`drive.google.com`, `docs.google.com`): the lecturer's Drive is
  private, and data comes from the kit.

## 10. Releasing

`python -m ai_learning_kit.dev.release X.Y.Z` sets the version, updates the docs and runs the
validator. After the pull request is merged, publish the release `vX.Y.Z` on main. The kit is developed
in a private repository, and a release there is for development and testing only. It reaches students
and notebooks when it is made public by hand with the `publish` workflow (Actions › publish › Run
workflow, with the tag), which copies the release's files as one commit, its tag and its notes to the
public repository that notebooks install from, and copies `data-N` releases with their files. Running
it with `all` fills gaps (earlier versions not yet public) without making a newer version public. Then
update Notebook Tutor to bundle the release. Notebooks that should move to the new version are stamped with
`python -m ai_learning_kit.dev.notebooks stamp X.Y.Z <files>`, checked, executed and distributed
again; notebooks left at an earlier version keep working, because they install that version.
