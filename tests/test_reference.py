"""docs/reference/ is generated from the kit and must match it."""
import pytest


def test_the_topic_reference_matches_the_kit():
    pytest.importorskip("torch")                  # the graph topics' helpers need the 'advanced' extra (CI has it)
    from ai_learning_kit.dev import reference
    assert reference.stale() == [], ("run python -m ai_learning_kit.dev.reference write and commit docs/reference/\n"
                                     + reference.differences())


def test_a_page_has_the_three_layers_and_the_helpers():
    from ai_learning_kit.dev import reference
    page = reference.topic_page("data-cleaning")
    assert page.startswith("# ") and "## The three layers" in page and "### In focus" in page
    assert "**`T.knn_impute(df: pd.DataFrame, column: str" in page          # signatures without quoted annotations
    assert "`T.textbooks_raw`, from [`web-data`](web-data.md)" in page    # supplied by another topic
    assert "do not edit by hand" in page


def test_pages_do_not_depend_on_where_the_kit_is_installed():
    pytest.importorskip("torch")
    from pathlib import Path
    from ai_learning_kit.dev import reference
    page = reference.topic_page("knowledge-graphs")
    assert str(Path.home()) not in page and "PosixPath" not in page and "WindowsPath" not in page
    assert "folder=<kit>/topics/knowledge_graphs/saved" in page


def test_page_gives_the_index_or_a_topic():
    from ai_learning_kit.dev import reference
    assert reference.page().startswith("# ai-learning-kit reference")
    assert reference.page("README.md") == reference.page()
    assert reference.page("data-cleaning.md") == reference.topic_page("data-cleaning")


def test_a_topic_whose_extra_is_missing_still_has_a_page(monkeypatch):
    from ai_learning_kit.core import registry
    from ai_learning_kit.dev import reference

    def no_torch(mod):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")
    monkeypatch.setattr(registry, "tools", no_torch)
    page = reference.topic_page("graph-neural-networks")
    assert "## The three layers" in page and "`torch` is not installed" in page and "T = kit.tools()" in page
