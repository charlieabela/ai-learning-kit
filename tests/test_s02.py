"""Regression tests: the frozen case must keep producing the same reference values."""
import ari1101kit as kit
from ari1101kit.sessions import s02
import pytest as _pytest


@_pytest.fixture(autouse=True)
def _isolated_folder(tmp_path, monkeypatch):
    """Every test starts in its own folder, which also counts as the notebook's folder."""
    from ari1101kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)


def test_sources_present():
    assert len(kit.sources()) == 7


def test_policy_extraction():
    rules = {r["rule_id"]: r for r in kit.extract.policy_rules()}
    assert set(rules) == {"R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "E1"}
    assert rules["R2"]["approvers"]["roles"] == ["Budget Holder", "Finance Manager"]
    assert rules["R7"]["rate_eur_per_usd"] == 0.92
    assert rules["R4"]["window_days"] == 7


def test_reference_values():
    ref = s02._ref()
    assert ref["erp_shape"] == (426, 9)
    assert ref["erp_high"] == 39
    assert ref["bank_high"] == 42
    assert ref["requests"] == 37
    assert ref["noncompliant"] == 11
    assert ref["split_payments"] == 5
    assert ref["stated"] == 41


def test_every_task_has_a_question():
    for tid, t in s02.TASKS.items():
        assert t["question"] and t["title"], tid


# --------------------------------------------------------------------------- feedback (0.2.1)
CHARLIE_T11 = {
    "policy": {"format": "docx", "shape": "document"},
    "erp": {"format": "csv", "shape": "table"},
    "bank": {"format": "json", "shape": "nested"},
    "approvals": {"format": "xml", "shape": "tree"},
    "vendors": {"format": "docx", "shape": "document"},
    "emails": {"format": "txt", "shape": "text"},
    "board_report": {"format": "docx", "shape": "document"},
}


def test_t11_feedback_names_each_wrong_field_with_evidence():
    fb = s02.TASKS["T1.1"]["check"](CHARLIE_T11)
    assert fb.ok is False
    text = "\n".join(fb.details)
    assert "vendors, format: you wrote 'docx'" in text and "approved_vendors.xlsx" in text
    assert "vendors, shape: you wrote 'document'" in text
    assert "board_report, format: you wrote 'docx'" in text and "%PDF" in text
    assert "board_report, shape" not in text          # 'document' is right for the board report
    assert "policy, erp, bank, approvals, emails" in text


def test_t11_correct():
    good = dict(CHARLIE_T11, vendors={"format": "xlsx", "shape": "table"},
                board_report={"format": "pdf", "shape": "document"})
    assert s02.TASKS["T1.1"]["check"](good).ok is True


def test_numeric_feedback_explains_common_mistakes():
    def details(tid, v):
        return " ".join(s02.TASKS[tid]["check"](v).details)
    assert "USD amounts were not converted" in details("T3.3", 40)
    assert "EUR rows only" in details("T3.3", 37)
    assert "number of rows in the ERP file" in details("T4.2", 426)
    assert "number of groups" in details("T7.2", 2)
    assert "<= 2" in details("T5.3", 34)
    assert s02.TASKS["T3.3"]["check"](39).ok is True


def test_suggestion_only_after_three_attempts():
    fb = s02.TASKS["T3.3"]["check"](40)
    assert "Suggested answer" not in fb.render("T3.3", 1)
    assert "Suggested answer" in fb.render("T3.3", 3)


def test_answer_book_counts_attempts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    kit.start("S02", student="Attempt Test")
    kit.answer("T3.3", value=40)
    kit.answer("T3.3", value=39)
    kit.answer("T3.3", reasoning="added later")
    entry = kit._state["book"].data["answers"]["T3.3"]
    assert entry["attempts"] == 3 and entry["value"] == 39     # every save counts, including reasoning added later
    assert entry["result"] == "correct" and entry["first_correct_attempt"] == 2


# --------------------------------------------------------------------------- guidance and written answers (0.2.2)
def test_every_task_has_guidance():
    from ari1101kit.sessions.s02_guide import GUIDE
    for tid, t in s02.TASKS.items():
        g = GUIDE[tid]
        assert all(g[k].strip() for k in ("what", "why", "example", "record")), tid
        assert s02.render_markdown(tid, t["title"]).startswith(f"### Task {tid}")
        if t["needs_reasoning"]:
            assert t["written"] is not None, tid


def test_written_feedback_lists_found_and_missing_points():
    weak = "The csv and json files can be read by pandas."
    fb = s02.TASKS["T1.2"]["written"](weak)
    assert fb.ok is None and fb.open_points
    text = "\n".join(fb.details)
    assert "✓ It names the sources that can be read directly" in text
    assert "○ Not found: it explains that the question needs several sources combined" in text
    model = s02.TASKS["T1.2"]["written"](fb.suggestion)
    assert not model.open_points            # the model answer passes its own checklist


def test_model_answers_pass_their_checklists():
    for key, check in s02.WRITTEN.items():
        model = check("x").suggestion
        assert not check(model).open_points, key


def test_placeholder_reasoning_is_flagged():
    fb = s02.TASKS["T3.4"]["written"]("\nWrite your answer here.\n")
    assert fb.open_points and "No reasoning yet" in fb.summary


def test_local_work_folder_is_the_current_folder(tmp_path, monkeypatch):
    from ari1101kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit.start("S02", student="Local Student")
    assert kit._state["folder"] == tmp_path.resolve()
    assert kit._state["book"].path.parent == tmp_path.resolve() / "ari1101_answers"


# --------------------------------------------------------------------------- work folder = notebook folder (0.2.3)
import sys
import types

import pytest


@pytest.fixture
def fake_colab(tmp_path, monkeypatch):
    """A pretend Colab: google.colab importable, Drive mounted (on request) under tmp_path/drive."""
    from ari1101kit.core import workfolder
    drive_root = tmp_path / "drive"
    state = {"mounted": False, "mounts": 0, "flushes": 0}

    def mount(p):
        if drive_root.exists() and any(drive_root.iterdir()):
            raise ValueError("Mountpoint must not already contain files")
        state["mounted"] = True
        state["mounts"] += 1
        (drive_root / "MyDrive").mkdir(parents=True, exist_ok=True)

    def flush_and_unmount():
        state["flushes"] += 1
        state["mounted"] = False            # files stay on disk in this simulation

    colab = types.ModuleType("google.colab")
    colab.drive = types.SimpleNamespace(mount=mount, flush_and_unmount=flush_and_unmount)
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(workfolder, "COLAB_DRIVE", drive_root)
    monkeypatch.setattr(workfolder, "COLAB_LOCAL", tmp_path / "content")
    (tmp_path / "content").mkdir()
    monkeypatch.setattr(workfolder, "drive_mounted", lambda: state["mounted"])
    monkeypatch.chdir(tmp_path)

    def mount_now():
        mount(str(drive_root))
    return types.SimpleNamespace(root=drive_root, my=drive_root / "MyDrive", state=state, wf=workfolder,
                                 mp=monkeypatch, mount_now=mount_now, tmp=tmp_path)


def _notebook(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}")
    return path


def test_colab_saves_next_to_a_copy_in_colab_notebooks(fake_colab):
    fake_colab.mp.setattr(fake_colab.wf, "colab_session", lambda: {"name": "S02.ipynb", "path": "fileId=abc123"})
    fake_colab.mount_now()
    _notebook(fake_colab.my / "Colab Notebooks" / "S02.ipynb")
    kit.start("S02", student="Colab Student")
    expected = (fake_colab.my / "Colab Notebooks").resolve()
    assert kit._state["folder"] == expected
    assert kit._state["book"].path == expected / "ari1101_answers" / "S02_Colab_Student.json"


def test_colab_finds_a_notebook_in_a_subfolder(fake_colab):
    fake_colab.mp.setattr(fake_colab.wf, "colab_session", lambda: {"name": "S02.ipynb", "path": "fileId=abc123"})
    fake_colab.mount_now()
    _notebook(fake_colab.my / "Uni" / "ARI1101" / "S02.ipynb")
    kit.start("S02", student="Colab Student")
    assert kit._state["folder"] == (fake_colab.my / "Uni" / "ARI1101").resolve()


def test_colab_uses_the_drive_api_when_the_name_is_not_unique(fake_colab):
    fake_colab.mp.setattr(fake_colab.wf, "colab_session", lambda: {"name": "S02.ipynb", "path": "fileId=abc123"})
    fake_colab.mount_now()
    _notebook(fake_colab.my / "A" / "S02.ipynb")
    _notebook(fake_colab.my / "B" / "S02.ipynb")
    fake_colab.mp.setattr(fake_colab.wf, "path_from_drive_api", lambda fid: fake_colab.my / "B")
    kit.start("S02", student="Colab Student")
    assert kit._state["folder"] == (fake_colab.my / "B").resolve()


def test_colab_notebook_from_github_falls_back_with_advice(fake_colab, capsys):
    fake_colab.mp.setattr(fake_colab.wf, "colab_session",
                          lambda: {"name": "S02.ipynb", "path": "/v2/external/notebooks/github/x.ipynb"})
    kit.start("S02", student="Colab Student")
    assert fake_colab.state["mounts"] == 1          # Drive was not mounted yet, so kit.start mounted it
    assert kit._state["folder"] == (fake_colab.my / "ARI1101" / "S02").resolve()
    out = capsys.readouterr().out
    assert "not stored in Google Drive" in out and "Save a copy in Drive" in out


def test_colab_shared_notebook_falls_back_with_advice(fake_colab, capsys):
    fake_colab.mp.setattr(fake_colab.wf, "colab_session", lambda: {"name": "S02.ipynb", "path": "fileId=shared1"})

    def refuse(fid):
        raise PermissionError("not yours")
    fake_colab.mp.setattr(fake_colab.wf, "path_from_drive_api", refuse)
    kit.start("S02", student="Colab Student")
    assert kit._state["folder"] == (fake_colab.my / "ARI1101" / "S02").resolve()
    assert "shared with you" in capsys.readouterr().out


def test_drive_api_path_walks_up_the_parent_folders(fake_colab):
    tree = {"nb": {"name": "S02.ipynb", "parents": ["f2"]}, "f2": {"name": "ARI1101", "parents": ["f1"]},
            "f1": {"name": "Uni", "parents": ["root"]}, "root": {"name": "My Drive"}}
    files = types.SimpleNamespace(get=lambda fileId, **kw: types.SimpleNamespace(execute=lambda: tree[fileId]))
    auth = types.SimpleNamespace(authenticate_user=lambda: None)
    sys.modules["google.colab"].auth = auth
    discovery = types.ModuleType("googleapiclient.discovery")
    discovery.build = lambda *a, **k: types.SimpleNamespace(files=lambda: files)
    fake_colab.mp.setitem(sys.modules, "googleapiclient", types.ModuleType("googleapiclient"))
    fake_colab.mp.setitem(sys.modules, "googleapiclient.discovery", discovery)
    assert fake_colab.wf.path_from_drive_api("nb") == fake_colab.my / "Uni" / "ARI1101"


def test_colab_folder_argument_is_relative_to_my_drive_and_mounts_drive(fake_colab):
    kit.start("S02", student="Charlie", folder="AI-Lectures/ari1101-Foundations of Data Science/tools")
    expected = (fake_colab.my / "AI-Lectures" / "ari1101-Foundations of Data Science" / "tools").resolve()
    assert fake_colab.state["mounts"] == 1 and kit._state["folder"] == expected


@pytest.mark.parametrize("given", ["My Drive/ARI1101/S02", "MyDrive/ARI1101/S02", "ARI1101/S02",
                                   "/content/drive/My Drive/ARI1101/S02"])
def test_colab_folder_spellings(fake_colab, given):
    fake_colab.mp.setattr(fake_colab.wf, "COLAB_DRIVE", __import__("pathlib").Path("/content/drive"))
    assert fake_colab.wf.resolve_folder(given) == __import__("pathlib").Path("/content/drive/MyDrive/ARI1101/S02")


def test_colab_rescues_files_written_while_drive_was_not_connected(fake_colab, capsys):
    # what v0.2.3 did with folder=...: a look-alike Drive folder on the temporary disk
    ghost = fake_colab.my / "X" / "ari1101_answers" / "S02_Charlie.json"
    ghost.parent.mkdir(parents=True)
    ghost.write_text('{"answers": {"T1.1": {}}}')
    real_drive_copy = {}

    def mount(p):                            # the real Drive does not yet have the file
        fake_colab.state["mounted"] = True
        (fake_colab.my).mkdir(parents=True, exist_ok=True)
        real_drive_copy["existed"] = ghost.exists()
    sys.modules["google.colab"].drive.mount = mount
    kit.start("S02", student="Charlie", folder="X")
    assert real_drive_copy["existed"] is False          # the ghost folder was moved aside before mounting
    assert '"T1.1"' in ghost.read_text()                # ...and its answer book copied into Drive
    assert kit._state["book"].data["answers"].get("T1.1") is not None
    assert "while Drive was not connected" in capsys.readouterr().out


def test_sync_flushes_and_reconnects_drive(fake_colab):
    fake_colab.mount_now()
    kit.start("S02", student="Charlie", folder="ARI1101/S02")
    kit.sync()
    assert fake_colab.state["flushes"] == 1 and fake_colab.state["mounted"]
    import os
    assert os.getcwd() == str(kit._state["folder"])


def test_local_relative_folder_is_taken_from_the_notebook_folder(tmp_path, monkeypatch):
    from ari1101kit.core import workfolder
    monkeypatch.setattr(workfolder, "START_DIR", tmp_path)
    monkeypatch.chdir(tmp_path)
    kit.start("S02", student="Local Student", folder="work")
    kit.start("S02", student="Local Student", folder="work")      # a second call must not nest folders
    assert kit._state["folder"] == (tmp_path / "work").resolve()


def test_jupyter_session_name_gives_the_notebook_folder(tmp_path, monkeypatch):
    nb = _notebook(tmp_path / "notebooks" / "S02.ipynb")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("JPY_SESSION_NAME", str(nb))
    kit.start("S02", student="Lab Student")
    assert kit._state["folder"] == nb.parent.resolve()
