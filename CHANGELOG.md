# Changelog

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
