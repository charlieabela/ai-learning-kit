# Changelog

## 0.8.0 · ARI1101 S04, exploring data

- **ARI1101 S04 (Explore: descriptive statistics, distributions, correlation and visualisation)**, a session whose notebooks check themselves. `T = kit.tools()` provides the **Palmer penguins** (CC0, stored in the session's `data/` with its citation), **a year of daily weather for Malta** (2025, Open-Meteo archive API, ERA5, stored as the API returned it and compared with the live service in CI), **Anscombe's quartet**, helpers students may look inside (`iqr_outliers`, `pearson_by_hand`, `scatter_with_trend`, `corr_heatmap`, `sample_means`) and checks for the practice notebook (`T.check("E1")` ... `T.check("E5")`). The notebooks are distributed with the session's material.
- `scipy` becomes a requirement: pandas needs it for `.corr(method="spearman")`, which plain installs of the kit lacked.
- `dev.saved_copies check` reports matching copies as plain lines and keeps GitHub's annotations (ten per step) for differences and captured responses.

## 0.7.0 · ICS5111 S03: graph neural networks

- **ICS5111 S03 (graph neural networks)**: a session without graded tasks, like S02. `T = kit.tools()` gives readable CPU helpers in plain PyTorch, each layer written out in a few lines so that the code can be read next to the equations: the primer example, the five-paper graph used for calculations by hand, mean aggregation, the GCN's normalised adjacency, `MLP`, `GCN` and `SAGE` (GraphSAGE with the mean aggregator and optional neighbour sampling), full-batch training that keeps the epoch with the best validation accuracy, SGC and over-smoothing measures, edge homophily, receptive fields and computation trees, Weisfeiler–Lehman colour refinement, synthetic graphs of chosen homophily, and a graph auto-encoder for link prediction on the S02 link split. Cora is parsed by the S02 helpers, unchanged; S03 adds a low-label split (20 labelled papers per topic, 500 validation, 1,000 test, seed 5111). The reference results of the runs used on the slides ship with the session.
- **Depth on request**: the session's `STAGES` tell a tutor to explain in three stages (the idea, the mechanics, going deeper) and to stop at the first unless the student asks or the cell shows the equation; the notebooks mark the mechanics and depth cells as optional. The training loop, neighbour sampling, Weisfeiler–Lehman refinement and the synthetic-graph generator use 0.6.0's `ON_REQUEST` level, so a tutor explains their code only when asked.
- S03 needs the `[advanced]` extra (PyTorch); Notebook Tutor's advanced install already has it. Helpers use sparse matrices, so every Cora model trains in a few seconds on a laptop CPU.
- CI installs `[advanced]` (CPU wheels of PyTorch) so that the S03 helpers are validated and tested on every system; the S03 tests need no download.

## 0.6.0 · ARI1101 S03, steps explained on request, saved web responses

- **ARI1101 S03 (Obtain and Scrub: web pages, web APIs, data cleaning)**, a session whose notebooks check themselves. `T = kit.tools()` provides `T.get()` (a GET request like `requests.get`, which falls back to the kit's saved copy when the site cannot be reached), the fictional **Campus Lets** course website (36 flat listings and 20 textbooks, deliberately untidy), saved responses of the **Open-Meteo** archive and geocoding APIs (Luqa, October 2025; Victoria, Gozo, November 2025; one rejected request), helpers to look inside (`scrape_listings`, `allowed`, `crawl_delay`, `knn_impute`, `parse_dates`), gaps for imputation exercises (`weather_with_gaps`) and checks for the two practice notebooks (`T.check("S1")` ... `T.check("C6")`). The notebooks are distributed with the session's material.
- **A fourth stage visibility, `ON_REQUEST`**: a helper does the step and the notebook offers an optional look inside; tutors say what goes in and out, and explain how it works only when the student asks, from the helper's code. `scope_text()` tells tutors so.
- **The course website is published with GitHub Pages** from `docs/sites/ari1101-s03/` (`docs/.nojekyll` keeps the files as they are); the package holds the same files, and a test keeps them identical.
- **`python -m ai_learning_kit.dev.saved_copies check | capture`**: compares saved API responses and pages with the live ones (a CI step on Linux, which attaches the live response when it differs) and captures them again.
- `requests` and `beautifulsoup4` become requirements, so Notebook Tutor and the plain install have them; pandas 2.0 or later is required.

## 0.5.1 · Course copies of datasets, no failure in Colab

- **Course copy of Cora.** LINQS does not answer from Colab or GitHub's machines, so the kit now falls back, after 10 seconds, to a course copy of the same file: the `cora.tgz` asset of the repository's `data-1` release, accepted only if its checksum matches LINQS's original. The student sees one extra line, or nothing (inside `kit.start`).
- **Course copies added later reach older notebooks.** When the owner and the copies built into a version fail, the kit reads `data-copies.json` from the repository's `main` branch; a URL added there serves every kit version from 0.5.1 on. It can only add places to look: the checksum stays the one in the release.
- **Colab asks for the file instead of failing.** When nothing works in Colab, the cell shows an upload button for the lecturer's copy and checks the chosen file.
- **The missing-data message is shown without a traceback** in Jupyter, Colab and Notebook Tutor.
- `kit.start` says when it is fetching a dataset for the first time; `datasets.available(name)`.
- CI checks that every course copy serves the exact file.

## 0.5.0 · Sessions without answer books, datasets from their owners, ICS5111 S02

- **ICS5111 S02 (graph embeddings)**: the helpers of the v2 session pack (`s02_tools.py`, version 2026-10-27), unchanged in every computation, are `T = kit.tools()`; the reference results ship with the session. The pack's Prepare, Explore and Practice notebooks were rebuilt on the kit (the ZIP and Drive set-up cell replaced by the kit's two opening cells) and reproduce every printed result and figure of the pack; they are distributed with the session's material, not stored in the kit. `adr_graph()` was left out: no Session 02 notebook uses it.
- **Sessions without graded tasks**: `PROBLEM` and `TASKS` are optional; such a session has helpers (`TOOLS`), and `kit.start(unit, session)` needs no student name and connects Google Drive only when work must be kept (`SAVES_WORK`, default: when there are tasks). `kit.tools()` returns the helpers. Sessions also declare `DATASETS` and `EXTRAS`.
- **Course datasets from their owners, with fallbacks** (`ai_learning_kit.datasets`): each external dataset is recorded with owner, address, checksum, size, licence note and citation; `kit.start` fetches it. When the owner cannot be reached, the kit takes a checked copy from the notebook's folder or its `data/`, from added folders (`datasets.add_folder`: synced Google Drive or OneDrive folders, Drive in Colab), from lecturer-approved course copies, or from `datasets.use(name, file_folder_or_link)` (Google Drive, OneDrive/SharePoint and Dropbox links become download links). The message for a missing dataset says how to add it where the student is (Notebook Tutor's Data files button, Colab's Files panel, the notebook's folder in Jupyter). Copies are kept in the data folder, which is the clone's `data/` when the kit or Notebook Tutor runs from a git clone, so each clone holds only the datasets its student's sessions used. Cora comes from LINQS, whose server refuses some cloud machines (GitHub's CI runners among them).
- **Small research datasets stored in the kit** (`datasets.small`): Zachary's karate club, Les Misérables, the Florentine families and Davis's Southern women, identical to the networkx copies, with citations.
- **The notebook contract** (`docs/NOTEBOOK_CONTRACT.md`) and `python -m ai_learning_kit.dev.notebooks check|stamp` for notebook files wherever they are. **The kit no longer stores notebooks**: the ARI1101 S02 notebook leaves `notebooks/` (it remains in the history and in the tags up to v0.4.0).
- **Install without git**: notebooks install GitHub's archive of the release (`ai-learning-kit[extra]@https://.../archive/refs/tags/vX.Y.Z.zip`), which needs only pip.
- **One release step**: `python -m ai_learning_kit.dev.release X.Y.Z` sets the version, updates the docs and runs the validator.
- **CI** on Linux, macOS and Windows: tests, the validator, a check that each dataset owner still serves the exact file (a refused connection is a warning), and a regression test of the Session 02 helpers on the stored karate club that needs no download.
- After 0.4.0 and before this release, the ARI1101 S02 notebook and the docs were moved to install `v0.4.0`.

## 0.4.0 · One package name

- The compatibility package kept from before 0.3.0 is removed: the distribution now installs only `ai_learning_kit`. Use `import ai_learning_kit as kit` and `kit.start("ARI1101", "S02", student=...)`.
- Nothing else changed: no code in `ai_learning_kit` depended on the removed package, and the released notebooks already use the new name. The ARI1101 S02 notebook now installs `v0.3.2` (visible install line and `metadata["ai_learning_kit"]`), so `kit.hint()` and `kit.ask()` work with thinking models in Colab and Notebook Tutor no longer reports a version mismatch; a notebook written before 0.3.0 keeps working because it installs its own earlier tag.

## 0.3.2 · Models that think before answering

- Claude 5.5, Gemini 2.5 and later, and OpenAI's reasoning models think before they answer, and the thinking counts against the output limit. With the kit's small limits the answer was cut off, or JSON arrived incomplete. Every request now allows at least 4,096 output tokens (only tokens used are billed).
- Claude: requests ask for low thinking effort (`Anthropic.effort = "low"`), which is faster and cheaper for tutoring; models without an effort setting get the request again without it.
- `provider.truncated` and `provider.last_stop` say whether the last reply reached the limit; an answer lost entirely to thinking raises `LLMError` with code `truncated`.

## 0.3.1 · Language models that refuse some settings

- Claude 5.5 models refuse the `temperature` setting (400: "'temperature' is deprecated for this model"), and OpenAI's reasoning models refuse `temperature` and want `max_completion_tokens` instead of `max_tokens`. The Anthropic and OpenAI-compatible providers now retry without the refused setting and remember that for the model, so `kit.hint()`, `kit.ask()` and Notebook Tutor work with these models.
- Notebooks are unchanged and keep installing `v0.3.0`; nothing in a session changed.

## 0.3.0 · ari1101kit becomes ai-learning-kit

A restructuring, not a rewrite: all of the Session 02 work, the Brightwater problem, the feedback, the answer book and the work-folder logic are kept and moved into a layered package that serves several study units.

- **Package**: `ai_learning_kit` (distribution `ai-learning-kit`). `import ari1101kit as kit` still works through a shim.
- **Layers**: `core` (engine: Problem and session contracts, check builders, Feedback, AnswerBook, work folder, registry), `problems` (the unit-independent problem library; `brightwater_q3` moved here with its data and helpers), `units` (`ari1101` with S02; `ics2205` and `ics5111` registered with no sessions yet), `llm` (optional), `dev` (validator).
- **Student API**: `kit.start("ARI1101", "S02", student=...)`; `kit.extract`, `kit.pipeline`, `kit.report`, `kit.reference` are proxies to the current session's problem, so notebook cells are the same for every problem; `kit.catalog()` lists units, sessions, tasks and problems as plain data; `kit.problem()` and `kit.session()`.
- **Answer book**: `answers/<UNIT>_<SESSION>_<student>.json`, with unit, session, problem and kit version recorded.
- **Language models (optional)**: `ai_learning_kit.llm` with a Google Gemini provider (an API key from Google AI Studio), one provider for every OpenAI-compatible server (OpenAI, Ollama, vLLM, LM Studio, ...), an Anthropic provider, autodetection from environment variables and Colab secrets, and `kit.hint()` / `kit.ask()` that never give the answer. Nothing requires a model; the model never grades. For front ends such as Notebook Tutor: `llm.build()` makes a provider without changing the notebook's choice, `list_models()` asks the provider which models it offers, and `LLMError.code` says what went wrong (`bad_key`, `forbidden`, `model_not_found`, `rate_limited`, `offline`, ...).
- **Session scope**: a session may declare `STAGES`, each step marked as `focus` (students do it; explain in depth), `black_box` (the kit does it; say what goes in and out and where it is taught) or `out_of_scope`. `scope_text()` turns them into instructions for a tutor, `kit.hint()` and `kit.ask()` include them, and `registry.tutor_context(unit, session)` gives a tutor the cohort, the problem, the scope and the tasks with their guidance. ARI1101 S02 declares its stages.
- **Install tiers**: the base install covers ARI1101; `[graphs]` adds networkx and scikit-learn; `[advanced]` adds PyTorch and PyTorch Geometric for embeddings and graph neural networks.
- **Authoring**: `templates/problem`, `templates/session`, `docs/AUTHORING_A_PROBLEM.md`, `docs/AUTHORING_A_SESSION.md`, `docs/prompts/new_problem.md`, `docs/prompts/new_session.md`, and `python -m ai_learning_kit.dev.validate`.
- **Notebooks**: moved to `notebooks/<UNIT>/`; each carries `metadata["ai_learning_kit"]` (unit, session, problem, version, install line, task ids) so a notebook tutor can recognise it.

## 0.2.4 · Fix: work saved to Colab's temporary disk instead of Drive

- Fixed: `kit.start(..., folder="/content/drive/...")` did not connect Google Drive, so it created a look-alike folder on Colab's temporary disk and answers never reached Drive. Drive is now always connected before a Drive folder is used, and "is Drive connected" is decided by the mount point, not by a folder of that name existing.
- If such a look-alike folder exists (from v0.2.3), `kit.start` moves it aside, connects Drive, and copies its files in (only where Drive has no newer copy).
- `folder=` in Colab is relative to My Drive: `folder="ARI1101/S02"`, `"My Drive/ARI1101/S02"` and `"/content/drive/My Drive/ARI1101/S02"` all mean My Drive › ARI1101 › S02. Elsewhere a relative folder is taken from the notebook's folder, and calling `kit.start` twice no longer nests folders.
- Every save is forced out of the file cache; `kit.sync()` sends everything to Drive immediately, and `kit.submit()` calls it before downloading.

## 0.2.3 · Work saved next to the notebook

- `kit.start` keeps the answer book, the case files and files saved with plain names in the notebook's own folder, in Colab as well as in Jupyter and VS Code.
- In Colab the kit connects Google Drive, asks the runtime which notebook is open, and finds it in Drive: by name (Colab Notebooks first), and through the Drive API when the name is not unique.
- A notebook that is not in the student's own Drive (opened from GitHub or a shared link) falls back to `My Drive/ARI1101/<session>`, with the reason and the advice to use File › Save a copy in Drive.
- In Jupyter and VS Code the notebook's path is read from `JPY_SESSION_NAME` or `__vsc_ipynb_file__` when available, otherwise the kernel's starting folder.

## 0.2.2 · Task guidance, feedback on written answers, work folder

- Every S02 task has plain-language guidance: what to do, why it matters, a typical example on a small library case, and what to record. `kit.task("T1.2")` shows it, and the notebook's task cells are generated from the same text.
- Written answers get checklist feedback: the points a good answer covers, marked as found (✓) or not found (○), with a hint for each missing point and a model answer after three attempts. The checks look for key words and never mark a written answer wrong; the lecturer reads every written answer.
- Numeric checks add evidence from the data that helps without giving the number away, and recognise more common slips (for example `len(bank_doc)`, a failed `findall`, the count of compliant instead of non-compliant payments).
- `kit.start` chooses a work folder: Google Drive (`My Drive/ARI1101/<session>`) in Colab, the notebook's folder elsewhere, or a folder given with `folder=`. The answer book, the case files and files saved with plain names go there, and every save prints the full path.
- Every save counts as an attempt, including reasoning added later; a later call with only a value keeps the saved reasoning.

## 0.2.1 · Richer formative feedback

- Feedback now says which parts of an answer are right and which are not, what the student recorded, the evidence in the case files that shows why (file name, first bytes, structure), and a line of code to check it.
- Common wrong values are recognised and explained, for example an unconverted USD count, `>=` instead of `>`, an inner merge, or the number of groups instead of payments.
- `kit.answer()` counts attempts. After three attempts that are not yet right, a suggested answer with an explanation is shown. The answer book stores the attempts, the last result and the attempt at which the answer first became right.
- `kit.check()` gives the same feedback without saving or counting an attempt.
- Adding reasoning later with `kit.answer(task, reasoning=...)` keeps the value already saved.
- Session modules may return a `Feedback` object or the older `(ok, message)` pair.

## 0.2.0 · Session 02 (Data formats)

- Core: frozen Brightwater Q3 2026 case data (seven sources in seven formats).
- Core: `extract` (policy rules and thresholds from DOCX, vendor list from XLSX, Board report from PDF, emails from TXT), `reference` loaders, `pipeline.compliance_check` and `pipeline.split_purchases`, `report.compliance_report` (HTML and DOCX), answer book with formative feedback, `explain` and `source`.
- Session S02: 26 tasks (T1.1 to T8.2) and the student sandbox notebook.
