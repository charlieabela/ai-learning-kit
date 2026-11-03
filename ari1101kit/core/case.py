"""Access to the frozen case files: inventory, paths, raw previews and a working copy."""
from __future__ import annotations

import shutil
from importlib import resources
from pathlib import Path

import pandas as pd

CASE_NAME = "brightwater_q3"

#: Short source names used throughout the kit, mapped to file names.
SOURCES = {
    "policy": "expenditure_policy_v3_1.docx",
    "erp": "erp_payments_q3.csv",
    "bank": "bank_feed_q3.json",
    "approvals": "approvals_pws_q3.xml",
    "vendors": "approved_vendors.xlsx",
    "emails": "finance_emails_q3.txt",
    "board_report": "q3_finance_summary.pdf",
}

DESCRIPTIONS = {
    "policy": "Expenditure Approval Policy v3.1 (Word document)",
    "erp": "Payments posted in the finance system (ERP) in Q3",
    "bank": "Bank statement feed for Q3, as returned by the bank's API",
    "approvals": "Approval requests exported from the Procurement Workflow System (PWS)",
    "vendors": "Approved Vendor List maintained by Procurement (Excel workbook)",
    "emails": "Finance-related emails collected for the review",
    "board_report": "Q3 Finance Summary presented to the Board (PDF)",
}

TEXT_SOURCES = {"erp", "bank", "approvals", "emails"}


def _data_dir() -> Path:
    return Path(str(resources.files("ari1101kit") / "data" / CASE_NAME))


def path(name: str) -> Path:
    """Return the path of a source file inside the installed kit.

    `name` is one of the short names listed by `sources()`, for example "erp".
    Students normally use the copies made by `workspace()` instead.
    """
    if name not in SOURCES:
        raise KeyError(f"Unknown source {name!r}. Choose one of: {', '.join(SOURCES)}")
    return _data_dir() / SOURCES[name]


def sources() -> pd.DataFrame:
    """List the case's evidence sources: short name, file name, size and description."""
    rows = []
    for name, fname in SOURCES.items():
        p = _data_dir() / fname
        rows.append({"source": name, "file": fname, "size_kB": round(p.stat().st_size / 1024, 1),
                     "description": DESCRIPTIONS[name]})
    return pd.DataFrame(rows)


def workspace(folder: str | Path = CASE_NAME, overwrite: bool = False) -> Path:
    """Copy the case files into a working folder (default ./brightwater_q3) and return its path.

    Your own code should read the files from this folder, for example
    pd.read_csv("brightwater_q3/erp_payments_q3.csv"). Existing files are kept
    unless overwrite=True, so your own files in the folder are never deleted.
    """
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for fname in SOURCES.values():
        target = folder / fname
        if overwrite or not target.exists():
            shutil.copy2(_data_dir() / fname, target)
    print(f"Case files are in: {folder}/")
    for name, fname in SOURCES.items():
        print(f"  {name:<13} {fname}")
    return folder


def peek(name: str, lines: int = 12) -> None:
    """Print the first lines of a source exactly as stored, without interpreting it.

    Text-based sources (CSV, JSON, XML, TXT) are shown as raw characters. Binary
    documents (DOCX, XLSX, PDF) cannot be read as plain text; for those the kit
    explains which extraction function to use.
    """
    p = path(name)
    if name not in TEXT_SOURCES:
        with open(p, "rb") as f:
            start = f.read(24)
        print(f"{p.name} is a binary file. Its first bytes are {start!r}.")
        hint = {"policy": "kit.extract.policy_text(), policy_rules(), policy_thresholds()",
                "vendors": "kit.extract.vendors()",
                "board_report": "kit.extract.board_report()"}[name]
        if start.startswith(b"PK"):
            print("PK is the signature of a ZIP archive: Word and Excel files are ZIP folders of XML files.")
        elif start.startswith(b"%PDF"):
            print("%PDF is the signature of a PDF file: text, fonts and images laid out on pages.")
        print(f"It cannot be read as plain text. Use {hint}.")
        return
    with open(p, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= lines:
                print(f"... ({p.name} continues)")
                break
            print(line.rstrip("\n"))
