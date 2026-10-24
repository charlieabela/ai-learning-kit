# Prompt: draft a new problem for ai-learning-kit

Use this with Claude or another capable model. Fill in the three inputs, attach
or paste `docs/AUTHORING_A_PROBLEM.md`, `ai_learning_kit/core/problem.py`,
`templates/problem/*` and, as a worked example, `ai_learning_kit/problems/brightwater_q3/case.py`.
Review everything it produces: the model drafts, you decide.

---

You are helping a university lecturer add a problem to **ai-learning-kit**, a
Python sandbox library for AI study units. Read the attached authoring guide,
the `Problem` dataclass, the template and the worked example before writing.

**Inputs**

- Topic / domain: `<e.g. hospital bed scheduling; municipal waste collection; a
  small e-commerce returns desk>`
- Intended units and levels: `<e.g. ARI1101 (BSc year 1, foundations of data
  science) and ICS5111 (MSc, mining large-scale data)>`
- Guidelines and constraints: `<e.g. must include a free-text source and a
  PDF; one boundary case per rule; English; Maltese setting; at most 500
  records per table; no real organisations>`

**Produce, in this order**

1. **The question** the organisation asks, in one sentence, and a one-paragraph
   story: who is asking, why, what decision depends on the answer.
2. **The sources**: four to seven, each with a short name, file name, format, a
   one-line description, and the part of the answer it carries. Say why each
   format is the natural one for that source.
3. **Planted issues** (the private key): a numbered list of eight to twelve
   issues with known answers, each stating where it sits, which sources reveal it,
   and the correct finding. Include at least one boundary case, one unit or
   currency conversion, one duplicate or missing record across sources, one
   claim in a document that the data contradicts, and one issue visible only by
   combining rows or reading free text.
4. **The reference values** a session would check: a table of name, definition,
   and how it is computed from the data; include the values common mistakes
   would produce.
5. **The generator plan**: a description of a seeded script (not the data) that
   produces every file, with record counts, distributions, naming conventions,
   and how each planted issue is injected.
6. **`case.py`**: the completed `Problem(...)` definition, ready to paste.
7. **Helper module outline**: for `extract.py`, `reference.py` and (if useful)
   `pipeline.py` and `report.py`, the public functions with one-paragraph plain
   docstrings, no code bodies.
8. **Session ideas**: for each intended unit, the "students do / kit does" split
   and five example task titles, so the problem is known to work at that level.

**Constraints**

- Everything fictional; no real people, companies or addresses.
- Plain language in anything a student may read; name the real concept rather
  than a metaphor.
- Do not write the data itself; the lecturer generates it privately.
- Mark anything you are unsure about with "CHECK:".
