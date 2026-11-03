"""Topics: the registry, levels, kit.start("<topic>"), the tutor context and the notebook contract, without
downloads or optional libraries."""
import json
import re
from pathlib import Path

import pytest

import ai_learning_kit as kit
from ai_learning_kit import topics
from ai_learning_kit.core import levels, registry
from ai_learning_kit.dev import notebooks as nbc
from ai_learning_kit.dev import validate


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(topic=None, problem=None, book=None, folder=None, tools=None)


def test_topic_names_are_normalised():
    assert topics.normalise_topic("Knowledge graphs") == "knowledge-graphs"
    assert topics.normalise_topic("knowledge_graphs") == "knowledge-graphs"
    assert topics.is_topic("knowledge-graphs") and not topics.is_topic("XYZ1234")
    with pytest.raises(KeyError, match="knowledge-graphs"):
        topics.topic("no-such-topic")


def test_every_topic_has_the_required_fields_and_a_valid_level():
    for name in topics.names():
        mod = registry.topic(name)
        assert mod.TOPIC == name and mod.TITLE.strip()
        assert mod.LEVEL in levels.LEVELS
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", mod.TOPIC)


def test_levels_describe_learners_not_courses():
    assert list(levels.LEVELS) == ["introductory", "intermediate", "advanced", "research"]
    for name in levels.LEVELS:
        text = levels.describe(name)
        assert text.startswith(levels.LEVELS[name]["label"])
        assert not validate.COURSE_CODE.search(text)
    assert levels.normalise_level(None) == levels.DEFAULT_LEVEL
    with pytest.raises(ValueError):
        levels.normalise_level("year-one")


def test_no_topic_file_names_a_study_unit():
    for name in topics.names():
        rep = validate.validate_topic(name)
        assert "FAIL no study-unit codes" not in str(rep), str(rep)


def test_tutor_context_uses_the_level_given():
    ctx = registry.tutor_context("knowledge-graphs", level="introductory")
    json.dumps(ctx)
    assert ctx["level"] == "introductory" and ctx["cohort"].startswith("Introductory level.")
    assert "unit" not in ctx and ctx["scope"].startswith("SCOPE OF THIS TOPIC")


def test_catalog_lists_topics_and_levels():
    cat = registry.catalog(include_tasks=False)
    assert {"topic": "knowledge-graphs", "title": "Knowledge graphs", "level": "advanced"} in cat["topics"]
    assert set(cat["levels"]) == set(levels.LEVELS)


def test_start_a_topic_names_it_by_its_title(capsys, monkeypatch):
    from ai_learning_kit import datasets
    mod = registry.topic("knowledge-graphs")
    monkeypatch.setattr(mod, "DATASETS", ())          # no downloads in this test
    kit.start("knowledge-graphs")
    out = capsys.readouterr().out
    assert f"· {mod.TITLE}" in out and not validate.COURSE_CODE.search(out)
    assert kit._state["topic"] is mod and kit.tools().__name__.endswith("knowledge_graphs.tools")


def test_start_without_a_known_topic_says_which_exist():
    with pytest.raises(KeyError, match="knowledge-graphs"):
        kit.start("graphs-and-things")


def _topic_notebook(tmp_path, start='import ai_learning_kit as kit\nkit.start("knowledge-graphs")\nT = kit.tools()',
                    meta_extra=None):
    install = nbc.install_line(None, "advanced")
    meta = {"topic": "knowledge-graphs", "title": "Knowledge graphs: setup", "role": "prepare", "level": "advanced",
            "kit_version": kit.__version__, "install": install.lstrip("%")}
    meta.update(meta_extra or {})
    nb = {"nbformat": 4, "nbformat_minor": 5, "metadata": {"ai_learning_kit": meta},
          "cells": [{"cell_type": "markdown", "metadata": {}, "source": "# Setup"},
                    {"cell_type": "code", "metadata": {}, "outputs": [], "source": install},
                    {"cell_type": "code", "metadata": {}, "outputs": [], "source": start},
                    {"cell_type": "code", "metadata": {}, "outputs": [], "source": "nodes = T.hetionet_nodes()"}]}
    p = Path(tmp_path) / "1_prepare_setup.ipynb"
    p.write_text(json.dumps(nb), encoding="utf-8")
    return p


def test_a_topic_notebook_meets_the_contract(tmp_path):
    results = nbc.check_notebook(_topic_notebook(tmp_path))
    assert all(ok for ok, _ in results), [m for ok, m in results if not ok]


def test_the_contract_catches_a_wrong_start_and_an_unknown_level(tmp_path):
    p = _topic_notebook(tmp_path, start='import ai_learning_kit as kit\nkit.start("graph-embeddings")\nT = kit.tools()',
                        meta_extra={"level": "postgraduate"})
    failed = [m for ok, m in nbc.check_notebook(p) if not ok]
    assert any("kit.start(\"knowledge-graphs\"" in m for m in failed)
    assert any("level 'postgraduate'" in m for m in failed)


def test_stamp_rewrites_a_topic_notebook(tmp_path):
    p = _topic_notebook(tmp_path)
    assert nbc.stamp(p, "9.9.9")
    meta = json.loads(p.read_text())["metadata"]["ai_learning_kit"]
    assert meta["kit_version"] == "9.9.9" and meta["install"].endswith("v9.9.9.zip\"")
