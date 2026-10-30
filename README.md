# ai-learning-kit

A sandbox library for the AI study units at the University of Malta
(ARI1101 Foundations of Data Science, ICS2205 Web Intelligence, ICS5111 Mining
Large Scale Data, and others to come).

The kit holds a **library of problems**, each a frozen set of data sources in
several formats with helper code that does the heavy lifting, and, for each
**unit**, **sessions**. A session either binds a problem to a sequence of graded
**tasks** with an answer book, or supplies **helpers and data** to notebooks that
check themselves. The notebooks themselves are not stored here: they are kept and
distributed with the rest of a session's material, and run unchanged in Notebook
Tutor and in Colab because everything they depend on comes from the kit. A session
hides the steps that are not its topic, so that students concentrate on the
step that is: given the text of a policy, the kit extracts the rules and the
students choose a format to store them; given tables of transactions, the kit
checks compliance and the students decide what to count. Each task poses a
question; answers are saved in an answer book next to the notebook; formative
feedback says what is right, what is not, and the evidence.

All companies, people and documents in the problems are fictional. Real
datasets (Cora, for example) are not copied into the kit: it downloads them from
their owners, as described under *Data* below.

## Notebooks

Students open a session's notebooks in
[Notebook Tutor](https://github.com/charlieabela/notebook-tutor), which ships the
kit and keeps each student's copy, with their work, in its own clone; or in Colab,
from Drive. Every notebook follows one contract (`docs/NOTEBOOK_CONTRACT.md`): it
starts with the kit's install line and `kit.start(...)`, and nothing in it depends
on where it runs. Check and stamp a notebook file wherever it is:

```
python -m ai_learning_kit.dev.notebooks check S02_1_prepare_setup.ipynb
python -m ai_learning_kit.dev.notebooks stamp 0.5.0 S02_1_prepare_setup.ipynb
```

## Use in a notebook (Notebook Tutor, Colab, Jupyter or VS Code)

A session with graded tasks:

```python
%pip install -q "ai-learning-kit@https://github.com/charlieabela/ai-learning-kit/archive/refs/tags/v0.7.0.zip"
import ai_learning_kit as kit

kit.start("ARI1101", "S02", student="Your Name")   # work is kept next to this notebook
kit.workspace()                                     # copy the problem's case files there
kit.task("T1.1")                                    # what to do, why, a typical example, what to record
kit.peek("erp")                                     # look at a raw file
rules = kit.extract.policy_rules()                  # the problem's helpers: extract, pipeline, report, reference
kit.answer("T2.1", value=len(rules), reasoning="rules is a list of dictionaries")
kit.progress(); kit.submit()
```

A session whose notebooks check themselves (no answer book, no name needed):

```python
%pip install -q "ai-learning-kit[graphs]@https://github.com/charlieabela/ai-learning-kit/archive/refs/tags/v0.7.0.zip"
import ai_learning_kit as kit
kit.start("ICS5111", "S02")        # downloads the session's datasets from their owners
T = kit.tools()                    # the session's helper functions
G, X, y, paper_ids, classes, split = T.load_cora()
```

The install line uses GitHub's archive of the release, so it needs only pip, not git.

Always install a tagged version, so that a notebook keeps working when later
sessions are added. `kit.catalog()` lists every unit, session and problem in
the installed version; `kit.explain(fn)` and `kit.source(fn)` show what any kit
function does and how.

**Where work is kept.** `kit.start` finds the folder of the running notebook and
makes it the current folder: the answer book (`answers/`), the case files and
every file saved with a plain name sit next to the notebook. In Colab the kit
connects Google Drive and finds the notebook there; `kit.sync()` pushes pending
writes, and `kit.submit()` calls it. `folder=` chooses another folder (in Colab,
relative to My Drive).

**What to install.** The plain line covers ARI1101. Sessions on graphs add
`[graphs]` (networkx, scikit-learn), and sessions on embeddings and graph neural
networks add `[advanced]` (PyTorch, PyTorch Geometric):
`%pip install -q "ai-learning-kit[advanced]@https://github.com/charlieabela/ai-learning-kit/archive/refs/tags/v0.7.0.zip"`.
Each session records its extra, and its notebooks install it.

**Data.** `ai_learning_kit.datasets` records, for every external dataset, its
owner, download address, checksum, licence note and citation. The kit downloads
it from the owner the first time a notebook needs it, checks the checksum, and
keeps it in its data folder. When the owner's site cannot be reached, it uses a
copy from somewhere else, checked the same way:

```python
kit.datasets.use("cora", "~/Downloads/cora.tgz")                       # a file on this computer
kit.datasets.add_folder("~/Library/CloudStorage/GoogleDrive-me/My Drive/ICS5111")   # a synced folder
kit.datasets.use("cora", "https://drive.google.com/file/d/<id>/view")   # a shared copy (Drive, OneDrive, Dropbox)
kit.datasets.status()                                                   # what is where
```

A file dropped next to the notebook, or in its `data/` folder, is found too. The
data folder is the `data/` folder of the clone when the kit or Notebook Tutor runs
from a git clone (ignored by git), My Drive › ai-learning-kit › data in Colab with
Drive connected, and the user's cache folder otherwise. Small research datasets
(the karate club, Les Misérables, the Florentine families, Davis's Southern women)
are stored in the kit with their citations: `kit.datasets.small("karate")`.

**Language models are optional.** With a model configured (a Gemini, Anthropic
or OpenAI API key, or an open-weight model behind Ollama, vLLM or any
OpenAI-compatible server),
`kit.hint(task)` gives one next step and `kit.ask(task, question)` explains a
concept. Without one, everything else works unchanged. See `docs/LLM.md`.

## Contents of this version

| Unit | Sessions | Problem |
|---|---|---|
| ARI1101 Foundations of Data Science | S02 Data formats: CSV, JSON, XML and documents | `brightwater_q3` |
| | S03 Web pages, web APIs and data cleaning (helpers, the fictional Campus Lets website, saved Open-Meteo responses) | none (helpers and data) |
| ICS2205 Web Intelligence | none yet | |
| ICS5111 Mining Large Scale Data | S02 Graph embeddings (helpers, Cora from LINQS, reference results) | none (helpers and data) |
| ICS5111 Mining Large Scale Data | S03 Graph neural networks (plain-PyTorch helpers, Cora from LINQS, reference results; needs `[advanced]`) | none (helpers and data) |

**brightwater_q3**: the Q3 2026 expenditure compliance review at Brightwater
Logistics Ltd. Seven sources in seven formats (DOCX policy, CSV payments, JSON
bank feed, XML approvals, XLSX vendor list, TXT emails, PDF board report), eleven
planted breaches, two split purchases, duplicated bank lines and an over-counting
Board report.

## Structure

```
ai_learning_kit/core       the engine: problems, sessions, checks, feedback, answer book, work folder
ai_learning_kit/problems   the problem library (unit-independent)
ai_learning_kit/units      one package per unit; sessions bind a problem to tasks
ai_learning_kit/llm        optional language-model providers and the tutor helpers
ai_learning_kit/datasets   external datasets: owner, address, checksum, licence; download and cache
ai_learning_kit/dev        validator (problems, sessions), the notebook checker, and the release step
templates/                 skeletons for a new problem and a new session
docs/                      ARCHITECTURE, AUTHORING_A_PROBLEM, AUTHORING_A_SESSION, LLM, prompts/
```

`docs/ARCHITECTURE.md` explains the layers and contracts. To add a problem or a
session, follow the authoring guides; `docs/prompts/` holds prompts that draft
either from a topic and guidelines with a language model, for the lecturer to
review. `python -m ai_learning_kit.dev.validate` checks every contract.

## Development

```
pip install -e ".[dev,graphs]"
pytest -q
python -m ai_learning_kit.dev.validate          # problems, sessions and every notebook named by a session
python -m ai_learning_kit.dev.release 0.6.0     # a release: version and docs
```

CI runs the tests and the validator on Linux, macOS and Windows, and checks that
every registered dataset's owner still serves the exact file (an owner that refuses
the connection is reported as a warning).

Licence: GPL-3.0-or-later.
