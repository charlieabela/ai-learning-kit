# The notebook contract

A notebook built on the kit runs unchanged in **Notebook Tutor** (on the student's own computer)
and in **Colab**, and on a best-effort basis in plain Jupyter. That works because nothing in the
notebook depends on where it runs: the kit finds the work folder, provides the helpers and fetches
the data.

**Where notebooks live.** The kit stores no notebooks. The lecturer keeps them with the rest of a
session's material (Drive, the VLE) and students take them from there: in Notebook Tutor they open
the file, and the tutor keeps their copy, with their work, in its own clone; in Colab they open it
from Drive. The kit holds what notebooks depend on (each session's helpers, data sources and
reference results) and checks a notebook file wherever it is:

```
python -m ai_learning_kit.dev.notebooks check  S02_1_prepare_setup.ipynb S02_2_explore_walkthrough.ipynb ...
python -m ai_learning_kit.dev.notebooks stamp 0.5.0 S02_1_prepare_setup.ipynb ...     # install line + metadata
```

`check` tests every rule marked **(checked)** below. Before a notebook is handed to students it is
also executed from top to bottom against the kit version it names, and saved with its outputs.

## 1. Names

`<SESSION>_<n>_<role>_<name>.ipynb`, for example `S02_2_explore_walkthrough.ipynb`; no spaces.
Roles: `prepare`, `explore` (the walkthrough), `practice`, or `session` for one notebook that
covers a whole session. Solved (lecturer) versions are kept with the lecturer's resources and never
given to students. A practice notebook runs from top to bottom with its blanks unfilled (each
exercise returns `None` or prints a prompt until it is completed), so it can be executed and checked
in its student form.

## 2. The opening: two cells, always the same

```python
%pip install -q "ai-learning-kit[graphs]@https://github.com/charlieabela/ai-learning-kit/archive/refs/tags/v0.8.0.zip"
```

```python
import ai_learning_kit as kit
kit.start("ICS5111", "S02")       # fetches the session's datasets the first time
T = kit.tools()                   # the session's helper functions

import numpy as np, pandas as pd, matplotlib.pyplot as plt
from IPython.display import display
SEED = T.SEED
```

- The first code cell is exactly the install line for the kit version the notebook was checked
  with, with the session's pip extra (`EXTRAS` in the session module; none for ARI1101, `graphs`
  for networkx and scikit-learn, `advanced` for PyTorch). It uses GitHub's archive of the release,
  which needs only pip, and is written without spaces around `@`, which lets Notebook Tutor
  recognise it and skip it (the tutor ships the kit). `stamp` writes it. **(checked)**
- The second code cell starts the session and, when the session has helpers, gets them as `T`.
  Imports the rest of the notebook relies on go in the same cell. (Older notebooks may start the
  session in the third cell after an imports cell; `kit.start` must still come before any other
  use of the kit.) **(checked)**
- A session with graded tasks starts with `kit.start(unit, session, student="Your Name")`; a
  session whose notebooks check themselves needs no name.
- The markdown cell before the opening says, in two or three sentences, what the cells do and that
  nothing has to be downloaded or uploaded by hand when the download works. It does not explain
  Drive folders, ZIPs or paths.

## 3. Nothing that depends on the environment

No cell (after the install line) may **(checked)**:

- import `google.colab`, mount Drive, or call `files.upload()`;
- open, extract or mention ZIP files;
- change `sys.path`, or import helper files that sit next to the notebook;
- install packages (`%pip`, `!pip`, `subprocess ... pip`): a library the session needs is added
  to the kit's pip extras instead;
- hard-code paths such as `/content/drive/...` or `MyDrive`.

Files a notebook writes use plain relative names (`outputs/comparison.csv`): `kit.start` has made
the notebook's folder (or, in Colab, its Drive folder) the current folder.

## 4. Data

- **Course datasets published by someone else** (Cora) are never copied into the kit or the
  notebook. Each is registered in `ai_learning_kit/datasets/__init__.py` with the owner, the
  owner's download address, the checksum of the exact file, its size, the licence note and the
  citation, and listed in the session's `DATASETS`. `kit.start` fetches them, so a problem shows in
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
  folder is that clone's `data/` (ignored by git). It holds only the course datasets of the
  sessions that student opened: an ARI1101 student's clone never fetches Cora.
- **A student's own data** is not the kit's business: they add it with **Data files** next to their
  notebook and read it by path. Nothing is registered or checked.
- **Small research datasets** (Zachary's karate club, Les Misérables, the Florentine families,
  Davis's Southern women) are stored in the kit with their citations: `datasets.small("karate")`.
  Add another by storing it in `ai_learning_kit/datasets/small_data/` the same way.
- **Reference results** (frozen outputs of the run shown on the slides) are packaged with the
  session, in `units/<unit>/<session>/reference/`, and read through `T.ROOT / "reference"`.
- **Web pages and API responses** that a notebook requests are sent through the session's
  helper (`T.get`, which behaves like `requests.get`), and the kit keeps a saved copy of each
  one: when the site cannot be reached, the copy is used, so the notebook still runs and gives
  the same results. A request the kit has no copy of is sent as written.

## 5. Helpers

- A session's helpers are one module (`TOOLS = ".tools"`), returned by `kit.tools()`. Every
  `T.<name>` a notebook uses must exist there. **(checked)**
- Helpers never change once a version is released, so a notebook that names that version keeps
  giving the numbers on its slides. A change is a new function, or a new session version.

## 6. Metadata

`metadata["ai_learning_kit"]` holds **(checked)**:

| Key | Meaning |
|---|---|
| `unit`, `session` | the session in the kit, e.g. `"ICS5111"`, `"S02"` |
| `title` | the notebook's title as students see it |
| `role` | `prepare`, `explore`, `practice` or `session` |
| `kit_version`, `install` | written by `stamp` |
| `tasks` | the task ids the notebook answers (empty when it checks itself) |
| `runs_unfilled` | `false` only when blanks raise errors until filled (older notebooks) |

Notebook Tutor reads this block to recognise the notebook, to fetch the session's scope and helpers
for its explanations, and to say so when the notebook was written for another kit version.

## 7. Saved outputs

Notebooks are distributed **executed**, so that a student who cannot run a cell still sees the
expected result. Saved outputs contain no errors, except in cells tagged `raises-exception` (a
cell that demonstrates an error on purpose). **(checked)**

## 8. Releasing

`python -m ai_learning_kit.dev.release X.Y.Z` sets the version, updates the docs and runs the
validator. After the pull request is merged, create the tag `vX.Y.Z` on main, then update Notebook
Tutor to bundle the release. Notebooks that should move to the new version are stamped with
`python -m ai_learning_kit.dev.notebooks stamp X.Y.Z <files>`, checked, executed and distributed
again; notebooks left at an earlier version keep working, because they install that version.
