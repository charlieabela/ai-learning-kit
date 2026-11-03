# Changelog

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
