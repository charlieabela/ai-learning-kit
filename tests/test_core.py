"""The engine: registry, catalog, proxies, answer book naming."""
import json

import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry
from ai_learning_kit import problems, topics


@pytest.fixture(autouse=True)
def _isolated_folder(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(topic=None, problem=None, book=None, folder=None)


def test_catalog_is_plain_data_and_lists_everything():
    cat = kit.catalog(include_tasks=True)
    json.dumps(cat)                                   # JSON-able, for notebook tutors
    assert "units" not in cat and [t["topic"] for t in cat["topics"]] == topics.names()
    fmt = cat["topics"][0]
    assert fmt["topic"] == "data-formats" and fmt["problem"] == "brightwater_q3" and len(fmt["tasks"]) == 26
    assert set(cat["levels"]) == {"introductory", "intermediate", "advanced", "research"}
    assert cat["problems"][0]["id"] == "brightwater_q3" and "extract" in cat["problems"][0]["helpers"]


def test_topic_lookup_is_forgiving_about_spelling_and_names_no_courses():
    for name in ("data-formats", "Data Formats", "data_formats", " DATA-FORMATS "):
        assert registry.topic(name).TOPIC == "data-formats"
    with pytest.raises(KeyError, match="This version of the kit has"):
        registry.topic("XYZ9999")
    with pytest.raises(KeyError, match="Unknown topic"):
        kit.start("XYZ9999", "S01")            # a course and a session no longer start anything


def test_every_topic_module_has_the_contract_fields():
    for name in topics.names():
        mod = registry.topic(name)
        assert mod.TOPIC == name and mod.LEVEL in ("introductory", "intermediate", "advanced", "research")
        assert all(b in topics.REGISTRY for b in mod.BUILDS_ON), name
        assert mod.PROBLEM is None or mod.PROBLEM in problems.REGISTRY
        assert mod.TASKS or mod.TOOLS, f"{name} has neither tasks nor helpers"
        assert not hasattr(mod, "UNIT") and not hasattr(mod, "SESSION"), name
        for tid, t in mod.TASKS.items():
            assert t["title"] and t["question"], tid
            assert t["guide"] is not None, tid


def test_no_file_in_the_package_names_a_course():
    from ai_learning_kit.dev import validate
    assert validate.validate_package_text().failures == 0


def test_helpers_are_proxies_to_the_topics_problem():
    assert "no topic started" in repr(kit.extract)
    with pytest.raises(RuntimeError, match="kit.start"):
        kit.extract.policy_rules()
    kit.start("data-formats", student="Proxy Test")
    assert len(kit.extract.policy_rules()) == 9
    assert kit.explain is not None and "policy_rules" in dir(kit.extract)
    assert kit.problem().id == "brightwater_q3" and kit.topic().TOPIC == "data-formats"


def test_start_needs_a_name_when_the_topic_has_tasks(tmp_path):
    with pytest.raises(TypeError, match="your name"):
        kit.start("data-formats")
    kit.start("data-formats", student="Name Test")
    assert kit._state["book"].path.name == "data-formats_Name_Test.json"
    assert kit._state["book"].path.parent == tmp_path.resolve() / "answers"
    book = json.loads(kit._state["book"].path.read_text())
    assert book["topic"] == "data-formats" and book["problem"] == "brightwater_q3" and book["kit"] == "ai-learning-kit"


def test_workspace_copies_the_problems_files_next_to_the_notebook(tmp_path):
    kit.start("data-formats", student="WS")
    folder = kit.workspace()
    assert folder.resolve() == tmp_path.resolve() / "brightwater_q3"
    assert sorted(p.name for p in folder.iterdir()) == sorted(kit.problem().sources.values())


def test_topic_stages_give_the_tutor_its_scope():
    from ai_learning_kit.core.session import scope_text, stage, FOCUS
    mod = registry.topic("data-formats")
    assert {st["visibility"] for st in mod.STAGES.values()} == {"focus", "black_box", "out_of_scope"}
    text = scope_text(mod)
    assert "In focus" in text and "kit.extract" in text and "data-cleaning topic" in text
    ctx = registry.tutor_context("data-formats")
    json.dumps(ctx)
    assert ctx["level"] == "introductory" and ctx["cohort"].startswith("Introductory level") and ctx["scope"] == text
    assert ctx["tasks"]["T3.3"]["guide"].startswith("### Task T3.3")
    with pytest.raises(ValueError):
        stage("x", "somewhat visible")


def test_validator_passes_on_everything_registered(capsys):
    pytest.importorskip("torch")                 # some topics need the 'advanced' extra (CI installs it)
    from ai_learning_kit.dev import validate
    assert validate.main([]) == 0
    assert "stages, each with a label and a valid visibility" in capsys.readouterr().out
