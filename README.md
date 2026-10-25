# ai-learning-kit

A sandbox library for the AI study units at the University of Malta
(ARI1101 Foundations of Data Science, ICS2205 Web Intelligence, ICS5111 Mining
Large Scale Data, and others to come).

The kit holds a **library of problems**, each a frozen set of data sources in
several formats with helper code that does the heavy lifting, and, for each
**unit**, **sessions** that bind a problem to a sequence of **tasks**. A session
hides the steps that are not its topic, so that students concentrate on the
step that is: given the text of a policy, the kit extracts the rules and the
students choose a format to store them; given tables of transactions, the kit
checks compliance and the students decide what to count. Each task poses a
question; answers are saved in an answer book next to the notebook; formative
feedback says what is right, what is not, and the evidence.

All companies, people and documents in the problems are fictional.

## Use in a notebook (Colab, Jupyter or VS Code)

```python
%pip install -q "git+https://github.com/charlieabela/ai-learning-kit@v0.3.0"
import ai_learning_kit as kit

kit.start("ARI1101", "S02", student="Your Name")   # work is kept next to this notebook
kit.workspace()                                     # copy the problem's case files there
kit.task("T1.1")                                    # what to do, why, a typical example, what to record
kit.peek("erp")                                     # look at a raw file
rules = kit.extract.policy_rules()                  # the problem's helpers: extract, pipeline, report, reference
kit.answer("T2.1", value=len(rules), reasoning="rules is a list of dictionaries")
kit.progress(); kit.submit()
```

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

**What to install.** The line above covers ARI1101. Sessions on graphs add
`[graphs]` (networkx, scikit-learn), and sessions on embeddings and graph neural
networks add `[advanced]` (PyTorch, PyTorch Geometric):
`%pip install -q "ai-learning-kit[advanced] @ git+https://github.com/charlieabela/ai-learning-kit@v0.3.0"`.
[Notebook Tutor](https://github.com/charlieabela/notebook-tutor) installs the kit
for students who work on their own computers.

**Language models are optional.** With a model configured (a Gemini, Anthropic
or OpenAI API key, or an open-weight model behind Ollama, vLLM or any
OpenAI-compatible server),
`kit.hint(task)` gives one next step and `kit.ask(task, question)` explains a
concept. Without one, everything else works unchanged. See `docs/LLM.md`.

## Contents of this version

| Unit | Sessions | Problem |
|---|---|---|
| ARI1101 Foundations of Data Science | S02 Data formats: CSV, JSON, XML and documents | `brightwater_q3` |
| ICS2205 Web Intelligence | none yet | |
| ICS5111 Mining Large Scale Data | none yet | |

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
ai_learning_kit/dev        validator for problems and sessions
templates/                 skeletons for a new problem and a new session
docs/                      ARCHITECTURE, AUTHORING_A_PROBLEM, AUTHORING_A_SESSION, LLM, prompts/
notebooks/<UNIT>/          student notebooks
```

`docs/ARCHITECTURE.md` explains the layers and contracts. To add a problem or a
session, follow the authoring guides; `docs/prompts/` holds prompts that draft
either from a topic and guidelines with a language model, for the lecturer to
review. `python -m ai_learning_kit.dev.validate` checks every contract.

## Development

```
pip install -e ".[dev]"
pytest -q
python -m ai_learning_kit.dev.validate
```

Licence: GPL-3.0-or-later.
