# Authoring a problem

A problem is a self-contained case: a question, a story, several data sources in
several formats, helper code that does the heavy lifting, and reference values.
It is independent of any unit; sessions decide which parts students do themselves.

## What makes a good problem

1. **One question an organisation would actually ask**, answerable only by
   combining sources. "How many payments above €5,000 were made and were they
   approved as the policy requires?" works; "analyse this dataset" does not.
2. **Several sources in several formats**, each carrying a different part of the
   answer: a structured export (CSV), a nested feed (JSON), a tree (XML), a
   document written for people (DOCX or PDF), a spreadsheet with human
   formatting (XLSX), free text (emails). Every format should be there for a reason.
3. **Planted issues with known answers**: boundary cases (exactly on a threshold),
   unit or currency conversions, duplicates, records missing from one source,
   a claim in a report that the data contradicts, deliberate evasion visible only
   when rows are combined or text is read. Keep a private key of every planted issue.
4. **Frozen, generated data** with a seed, small enough to ship (a few MB at most),
   fictional names and organisations, and no real personal data. Generate it with
   a private script outside the public repository; the repository holds only the
   files.
5. **Helpers with plain docstrings**, because students can call `kit.explain()`
   and `kit.source()` on any of them. Helpers do what a session hides; they do
   not do what a session teaches.
6. **Reference values computed, not typed**: the `reference` module derives every
   correct value from the data, so sessions never hard-code a number.

## Steps

1. Copy `templates/problem/` to `ai_learning_kit/problems/<problem_id>/`.
2. Fill in `case.py` (the `Problem` object). Short source names are what students
   type (`kit.peek("erp")`), so keep them short and meaningful.
3. Put the data files in `data/`. Add a `data/README.md` line naming the private
   generator and seed.
4. Write `extract.py` (documents to data), `reference.py` (correct values, cached),
   and optionally `pipeline.py` (the end-to-end check a session may hide) and
   `report.py` (output layout). Register each under a namespace in `modules`.
5. Add the id to `ai_learning_kit/problems/REGISTRY`.
6. Run `python -m ai_learning_kit.dev.validate --problem <problem_id>` and fix
   anything it reports.
7. Add a test that pins the key reference values (so the frozen data cannot drift).
8. Keep the generator, the key of planted issues and worked answers in the
   lecturer's private pack, never in the repository.

## Checklist before a pull request

- [ ] `validate --problem` passes
- [ ] every source has a description and, if binary, a peek hint
- [ ] every helper function has a one-paragraph docstring in plain language
- [ ] reference values are cached functions of the data
- [ ] a test pins the headline reference values
- [ ] the story, names and documents are fictional; `fictional=True`
- [ ] total data size is reasonable for a wheel

To draft a new problem from a topic with a language model, use
`docs/prompts/new_problem.md`.
