# Prompt: draft a new topic with graded tasks for ai-learning-kit

Use this with Claude or another capable model. Fill in the inputs, attach or
paste `docs/AUTHORING_A_TOPIC.md`, `ai_learning_kit/core/session.py`,
`ai_learning_kit/core/levels.py`, `ai_learning_kit/core/checks.py`, `templates/topic_with_tasks/*`,
the problem's `case.py` and `reference.py`, and, as a worked example,
`ai_learning_kit/topics/data_formats/__init__.py` and `guide.py`. Review everything it produces.
(A topic whose notebooks check themselves, without tasks, starts from `templates/topic/` instead.)

---

You are helping a university lecturer add a topic to **ai-learning-kit**. Read
the attached authoring guide, the topic contract, the levels, the check builders, the
template and the worked example before writing. The kit is independent of any course:
nothing you write may name a study unit, a programme or a numbered course session.

**Inputs**

- Topic name and title: `<e.g. graph-construction, "Building graphs from text">`
- Level: `<introductory, intermediate, advanced or research; see core/levels.py>`
- Topics it builds on: `<e.g. text-to-graphs>`
- Learning outcomes for the topic: `<three to five, in the lecturer's words>`
- Problem to use: `<problem id>` (its `case.py` and `reference.py` are attached)
- Time available and setting: `<e.g. 2 hours, Colab, students work alone>`
- Guidelines: `<e.g. keep CSV reading visible, hide the entity linking; include
  two written reflections; one optional extension>`

**Produce, in this order**

1. **The split**: the topic's `STAGES`: what learners do (focus), what the kit does in
   one call (support), what they may look inside on request, and what is out of scope,
   justified by the learning outcomes.
2. **The task list**: 15 to 30 tasks in working order, each with id (`T<part>.<n>`),
   title, question, what is recorded (value, reasoning or both), the check type
   (`numeric_check`, `written`, custom `Feedback`, or `saved_only`), the
   reference value it uses, and the two or three wrong answers you expect with
   their causes.
3. **`_ref()`**: the dictionary of reference values built from the problem's
   `reference` module, including the wrong-answer values.
4. **`__init__.py`**: the complete topic module, ready to paste, following the template:
   custom checks return `Feedback` with itemised evidence; `written` points have
   key-word tests and a model answer that passes them.
5. **`guide.py`**: four parts per task (what, why, example, record), pitched
   to the topic's level, with every example on a different, smaller case named in
   `EXAMPLE_CASE`.
6. **Model answers** for every written task (used by the lecturer's worked-answers
   notebook).
7. **Notebook outline**: the markdown and code cells in order, with `___` blanks
   where learners fill in, and the exact `kit.answer(...)` call for each task.

**Constraints**

- Feedback never marks a written answer wrong; it reports points found and not found.
- The right value appears only as the `suggestion`, never in the first feedback.
- Do not change the problem's data or helpers; if a helper is missing, list it
  under "helpers the problem would need".
- Mark anything you are unsure about with "CHECK:".
