"""The notebook contract, checked on notebook files wherever they are (the kit stores no notebooks)."""
import json

import pytest

from ai_learning_kit.dev import notebooks as nbc


def notebook(tmp_path, cells, name="S02_1_prepare_setup.ipynb", unit="ICS5111", session="S02", version=None):
    version = version or nbc._current()
    install = nbc.install_line(version, "graphs" if unit == "ICS5111" else "")
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {
              "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
              "ai_learning_kit": {"unit": unit, "session": session, "title": "Session 02 · test", "role": "prepare",
                                  "kit_version": version, "install": install.lstrip("%"), "tasks": []}},
          "cells": [{"cell_type": "code", "id": f"c{i}", "metadata": {}, "outputs": [], "execution_count": None,
                     "source": src} for i, src in enumerate([install, *cells])]}
    path = tmp_path / name
    path.write_text(json.dumps(nb), encoding="utf-8")
    return path


START = 'import ai_learning_kit as kit\nkit.start("ICS5111", "S02")\nT = kit.tools()'


def fails(path):
    return [msg for ok, msg in nbc.check_notebook(path) if not ok]


def test_a_notebook_anywhere_passes_the_contract(tmp_path):
    assert fails(notebook(tmp_path, [START, "G, X, y, ids, classes, split = T.load_cora()"])) == []


@pytest.mark.parametrize("bad,why", [
    ("from google.colab import drive\ndrive.mount('/content/drive')", "Colab or Google Drive"),
    ("import zipfile", "ZIP"),
    ("import sys\nsys.path.insert(0, 's02_resources')", "sys.path"),
    ("%pip install torch", "installs packages"),
    ("T.not_a_helper()", "missing: not_a_helper"),
])
def test_environment_dependent_code_is_refused(tmp_path, bad, why):
    assert any(why in m for m in fails(notebook(tmp_path, [START, bad])))


def test_the_session_must_start_before_the_kit_is_used(tmp_path):
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
