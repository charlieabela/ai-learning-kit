# Changelog

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
