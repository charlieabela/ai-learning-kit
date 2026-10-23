"""The engine: registry, catalog, proxies, answer book naming, compatibility shim."""
import json

import pytest

import ai_learning_kit as kit
from ai_learning_kit.core import registry
from ai_learning_kit import problems, units


@pytest.fixture(autouse=True)
def _isolated_folder(tmp_path, monkeypatch):
    from ai_learning_kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit._state.update(session=None, problem=None, book=None, folder=None)


def test_catalog_is_plain_data_and_lists_everything():
    cat = kit.catalog(include_tasks=True)
    json.dumps(cat)                                   # JSON-able, for notebook tutors
    assert [u["code"] for u in cat["units"]] == ["ARI1101", "ICS2205", "ICS5111"]
    s02 = cat["units"][0]["sessions"][0]
    assert s02["session"] == "S02" and s02["problem"] == "brightwater_q3" and len(s02["tasks"]) == 26
    assert cat["problems"][0]["id"] == "brightwater_q3" and "extract" in cat["problems"][0]["helpers"]


def test_session_lookup_is_forgiving_about_codes():
    for unit, sess in (("ari1101", "s02"), ("ARI1101", "2"), ("ARI 1101", "Session 02"), ("ARI1101", "S2")):
        assert registry.session(unit, sess).SESSION == "S02"
    with pytest.raises(KeyError, match="no session"):
        registry.session("ICS5111", "S01")
    with pytest.raises(KeyError, match="Unknown unit"):
        registry.session("XYZ9999", "S01")


def test_every_session_module_has_the_contract_fields():
    for u in units.REGISTRY:
        pkg = units.unit(u)
        for code in pkg.SESSIONS:
            mod = registry.session(u, code)
            assert mod.UNIT == pkg.UNIT["code"] and mod.SESSION == code
            assert mod.PROBLEM in problems.REGISTRY
            for tid, t in mod.TASKS.items():
                assert t["title"] and t["question"], tid
                assert t["guide"] is not None, tid


def test_helpers_are_proxies_to_the_sessions_problem():
    assert "no session started" in repr(kit.extract)
    with pytest.raises(RuntimeError, match="kit.start"):
        kit.extract.policy_rules()
    kit.start("ARI1101", "S02", student="Proxy Test")
    assert len(kit.extract.policy_rules()) == 9
    assert kit.explain is not None and "policy_rules" in dir(kit.extract)
    assert kit.problem().id == "brightwater_q3" and kit.session().SESSION == "S02"


def test_start_accepts_unit_slash_session_and_needs_a_name(tmp_path):
    with pytest.raises(TypeError, match="your name"):
        kit.start("ARI1101/S02")
    kit.start("ARI1101/S02", student="Slash Test")
    assert kit._state["book"].path.name == "ARI1101_S02_Slash_Test.json"
    assert kit._state["book"].path.parent == tmp_path.resolve() / "answers"
    book = json.loads(kit._state["book"].path.read_text())
    assert book["unit"] == "ARI1101" and book["problem"] == "brightwater_q3" and book["kit"] == "ai-learning-kit"


def test_workspace_copies_the_problems_files_next_to_the_notebook(tmp_path):
    kit.start("ARI1101", "S02", student="WS")
    folder = kit.workspace()
    assert folder.resolve() == tmp_path.resolve() / "brightwater_q3"
    assert sorted(p.name for p in folder.iterdir()) == sorted(kit.problem().sources.values())


def test_ari1101kit_shim_still_works(capsys):
    import ari1101kit as old
    old.start("S02", student="Old Notebook")
    assert "now part of ai-learning-kit" in capsys.readouterr().out
    assert old.extract.policy_rules()[0]["rule_id"] == "R1"
    old.answer("T2.1", value=9, reasoning="rules is a list of dicts")
    assert kit._state["book"].data["answers"]["T2.1"]["result"] == "correct"
