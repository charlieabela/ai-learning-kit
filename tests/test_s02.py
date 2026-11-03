"""Regression tests: the frozen case must keep producing the same reference values."""
import ari1101kit as kit
from ari1101kit.sessions import s02


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
    monkeypatch.chdir(tmp_path)
    kit.start("S02", student="Local Student")
    assert kit._state["folder"] == tmp_path.resolve()
    assert kit._state["book"].path.parent == tmp_path.resolve() / "ari1101_answers"


def test_colab_work_folder_is_on_drive(tmp_path, monkeypatch):
    import sys, types
    from ari1101kit.core import workfolder
    mounted = []
    drive = types.SimpleNamespace(mount=lambda p: (mounted.append(p), (tmp_path / "drive" / "MyDrive").mkdir(parents=True)))
    colab = types.ModuleType("google.colab"); colab.drive = drive
    google = types.ModuleType("google"); google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    monkeypatch.setattr(workfolder, "COLAB_DRIVE", tmp_path / "drive")
    monkeypatch.chdir(tmp_path)
    kit.start("S02", student="Colab Student")
    expected = (tmp_path / "drive" / "MyDrive" / "ARI1101" / "S02").resolve()
    assert mounted and kit._state["folder"] == expected
    assert kit._state["book"].path == expected / "ari1101_answers" / "S02_Colab_Student.json"
