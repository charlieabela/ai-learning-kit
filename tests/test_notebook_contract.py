"""The notebook contract, checked on notebook files wherever they are (the kit stores no notebooks)."""
import json

import pytest

from ai_learning_kit.dev import notebooks as nbc


def notebook(tmp_path, cells, name="1_prepare.ipynb", topic="graph-embeddings", version=None, markdown=()):
    version = version or nbc._current()
    install = nbc.install_line(version, "graphs")
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
              "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
              "ai_learning_kit": {"topic": topic, "title": "Graph embeddings · test", "role": "prepare",
                                  "kit_version": version, "install": install.lstrip("%"), "tasks": []}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": src} for i, src in enumerate([install, *cells])]
                   + [{"cell_type": "markdown", "id": f"m{i}", "metadata": {}, "source": text}
                      for i, text in enumerate(markdown)]}
    path = tmp_path / name
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


START = 'import ai_learning_kit as kit\nkit.start("graph-embeddings")\nT = kit.tools()'


def fails(path):
    return [msg for ok, msg in nbc.check_notebook(path) if not ok]


def test_a_notebook_anywhere_passes_the_contract(tmp_path):
    assert fails(notebook(tmp_path, [START, "G, X, y, ids, classes, split = T.load_cora()"])) == []


@pytest.mark.parametrize("bad,why", [
    ("from google.colab import drive\ndrive.mount('/content/drive')", "Colab or Google Drive"),
    ("import zipfile", "ZIP"),
    ("import sys\nsys.path.insert(0, 'resources')", "sys.path"),
    ("%pip install torch", "installs packages"),
    ("T.not_a_helper()", "missing: not_a_helper"),
])
def test_environment_dependent_code_is_refused(tmp_path, bad, why):
    assert any(why in m for m in fails(notebook(tmp_path, [START, bad])))


def test_the_topic_must_start_before_the_kit_is_used(tmp_path):
    assert any("kit.start" in m for m in fails(notebook(tmp_path, ["T = kit.tools()", START])))


def test_stamp_moves_a_notebook_to_another_version(tmp_path):
    path = notebook(tmp_path, [START], version="0.4.0")
    assert any("stamped with kit version" in m for m in fails(path))
    assert nbc.stamp(path, nbc._current()) is True and fails(path) == []
    assert nbc.stamp(path, nbc._current()) is False


def test_command_line(tmp_path, capsys):
    path = notebook(tmp_path, [START])
    assert nbc.main(["check", str(path)]) == 0 and "OK" in capsys.readouterr().out
    assert nbc.main(["stamp", nbc._current(), str(path)]) == 0
    assert nbc.main(["nonsense"]) == 2


@pytest.mark.parametrize("text,why", [
    ("Welcome to ABC1234, week 3.", "study unit"),
    ("As we saw in Session 02, random walks ...", "numbered course session"),
    ("The slides are [here](https://drive.google.com/file/d/abc/view).", "Google Drive"),
])
def test_shared_notebooks_name_no_course_and_link_no_drive(tmp_path, text, why):
    assert any(why in m for m in fails(notebook(tmp_path, [START], markdown=[text])))


def test_a_notebook_for_an_unknown_topic_is_reported(tmp_path):
    assert any("does not have" in m for m in fails(notebook(tmp_path, [START], topic="not-a-topic")))


def test_plan_notes_list_practice_exercises_without_a_plan(tmp_path):
    import json
    from ai_learning_kit.dev import notebooks as nbc

    def md(t):
        return {"cell_type": "markdown", "metadata": {}, "source": t}

    def code(t):
        return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": t}
    cells = [md("# Practice"), md("## E1 · Count\n\n**From question to code (plan given, with the tools).** ..."),
             code("# 1 count them   (len)\nn = None   # TODO"), md("## E2 · Sum"), code("total = None   # TODO"),
             md("## E3 · Average\n\n**From question to code (your plan).** ..."), code("data = [1, 2]"),
             md("Now write it."), code("mean = None   # TODO")]      # the plan sits above code the exercise uses
    for role, expected in (("practice", 1), ("explore", 0)):
        p = tmp_path / f"{role}.ipynb"
        p.write_text(json.dumps({"nbformat": 4, "metadata": {"ai_learning_kit": {"role": role}}, "cells": cells}))
        notes = nbc.plan_notes(p)
        assert len(notes) == expected and (not notes or "[5]" in notes[0]), notes
