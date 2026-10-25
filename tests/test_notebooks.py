"""Every notebook named by a session: the contract, and (with AILK_EXECUTE=1) a full run.

The run skips the install cell, as Notebook Tutor does, because the kit under test is already
installed. Notebooks whose metadata says runs_unfilled = false (exercise blanks that raise
until a student fills them) are run up to and including kit.start only.
"""
import os
from pathlib import Path

import pytest

from ai_learning_kit import units
from ai_learning_kit.core import registry
from ai_learning_kit.dev.notebooks import NOTEBOOK_ROOT, check_notebook

NOTEBOOKS = [(code, scode, role, rel)
             for code in units.REGISTRY
             for scode in getattr(units.unit(code), "SESSIONS", {})
             for role, rel in registry.session(code, scode).NOTEBOOKS.items()]


@pytest.mark.parametrize("code,scode,role,rel", NOTEBOOKS, ids=[n[3] for n in NOTEBOOKS])
def test_notebook_follows_the_contract(code, scode, role, rel):
    failures = [msg for ok, msg in check_notebook(NOTEBOOK_ROOT / rel, registry.session(code, scode)) if not ok]
    assert not failures, failures


@pytest.mark.skipif(not os.environ.get("AILK_EXECUTE"), reason="set AILK_EXECUTE=1 to execute the notebooks")
@pytest.mark.parametrize("code,scode,role,rel", NOTEBOOKS, ids=[n[3] for n in NOTEBOOKS])
def test_notebook_runs(code, scode, role, rel, tmp_path):
    nbformat = pytest.importorskip("nbformat")
    from nbclient import NotebookClient
    path = NOTEBOOK_ROOT / rel
    from ai_learning_kit import datasets
    for name in registry.session(code, scode).DATASETS:
        try:
            datasets.fetch(name, quiet=True)
        except datasets.DatasetUnavailable as e:
            if os.environ.get("AILK_REQUIRE_DATA", "").lower() == "true":
                raise
            pytest.skip(f"dataset {name!r} is not available here: {str(e).splitlines()[0]}")
    nb = nbformat.read(path, as_version=4)
    code_cells = [c for c in nb.cells if c.cell_type == "code"]
    code_cells[0].source = "# install cell skipped: the kit under test is installed"
    if nb.metadata.get("ai_learning_kit", {}).get("runs_unfilled", True) is False:
        start = next(i for i, c in enumerate(code_cells) if "kit.start(" in c.source)
        keep = set(map(id, code_cells[: start + 1]))
        nb.cells = [c for c in nb.cells if c.cell_type != "code" or id(c) in keep]
        for c in nb.cells:
            if "kit.start(" in getattr(c, "source", ""):
                c.source = c.source.replace('student="Your Name"', 'student="CI Run"')
    work = tmp_path / "work"
    work.mkdir()
    NotebookClient(nb, timeout=1800, kernel_name="python3", allow_errors=True,
                   resources={"metadata": {"path": str(work)}}).execute()
    errors = [(i, o.get("ename"), o.get("evalue")) for i, c in enumerate(nb.cells) if c.cell_type == "code"
              for o in c.get("outputs", []) if o.get("output_type") == "error"]
    assert not errors, errors
