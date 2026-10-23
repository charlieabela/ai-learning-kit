# Prompt: draft a new session for ai-learning-kit

Use this with Claude or another capable model. Fill in the inputs, attach or
paste `docs/AUTHORING_A_SESSION.md`, `ai_learning_kit/core/session.py`,
`ai_learning_kit/core/checks.py`, `templates/session/*`, the problem's `case.py`
and `reference.py`, and, as a worked example, `ai_learning_kit/units/ari1101/s02.py`
and `s02_guide.py`. Review everything it produces.

---

You are helping a university lecturer add a session to **ai-learning-kit**. Read
the attached authoring guide, the session contract, the check builders, the
template and the worked example before writing.

**Inputs**

- Unit and level: `<e.g. ICS5111, MSc; see the unit's UNIT["notes"]>`
- Session code and topic: `<e.g. S01, graph construction from text>`
- Learning outcomes for the session: `<three to five, in the lecturer's words>`
- Problem to use: `<problem id>` (its `case.py` and `reference.py` are attached)
- Time available and setting: `<e.g. 2 hours, Colab, students work alone>`
- Guidelines: `<e.g. keep CSV reading visible, hide the entity linking; include
  two written reflections; one optional extension>`

**Produce, in this order**

1. **The split**: a two-column table, "students do" and "the kit does", justified
   by the learning outcomes.
2. **The task list**: 15 to 30 tasks in working order, each with id (`T<part>.<n>`),
   title, question, what is recorded (value, reasoning or both), the check type
   (`numeric_check`, `written`, custom `Feedback`, or `saved_only`), the
   reference value it uses, and the two or three wrong answers you expect with
   their causes.
3. **`_ref()`**: the dictionary of reference values built from the problem's
   `reference` module, including the wrong-answer values.
4. **`sNN.py`**: the complete module, ready to paste, following the template:
   custom checks return `Feedback` with itemised evidence; `written` points have
   key-word tests and a model answer that passes them.
5. **`sNN_guide.py`**: four parts per task (what, why, example, record), pitched
   to the unit's level, with every example on a different, smaller case named in
   `EXAMPLE_CASE`.
6. **Model answers** for every written task (used by the lecturer's worked-answers
   notebook).
7. **Notebook outline**: the markdown and code cells in order, with `___` blanks
   where students fill in, and the exact `kit.answer(...)` call for each task.

**Constraints**

- Feedback never marks a written answer wrong; it reports points found and not found.
- The right value appears only as the `suggestion`, never in the first feedback.
- Do not change the problem's data or helpers; if a helper is missing, list it
  under "helpers the problem would need".
- Mark anything you are unsure about with "CHECK:".
