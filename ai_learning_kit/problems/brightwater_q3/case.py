"""The Brightwater Q3 2026 expenditure compliance review: problem definition and sources."""
from __future__ import annotations

from ...core.problem import Problem

PROBLEM = Problem(
    id="brightwater_q3",
    title="The Q3 2026 Expenditure Compliance Review at Brightwater Logistics Ltd",
    question=("How many payments above €5,000 were made in Q3 2026, and were they in line with the "
              "company's Expenditure Approval Policy?"),
    summary=("Brightwater Logistics Ltd, a fictional Maltese logistics company, has asked its data scientist "
             "whether the quarter's large payments followed its approval policy. The evidence sits in seven "
             "sources in seven formats: the policy (Word), the finance system's payment export (CSV), the bank's "
             "statement feed (JSON), the approval workflow export (XML), the approved vendor list (Excel), a set "
             "of emails (plain text) and the Board's own finance summary (PDF). Eleven planted breaches, two "
             "split purchases, duplicated bank lines and a Board report that over-counts make the review "
             "non-trivial at every OSEMN stage."),
    domain="finance / compliance",
    sources={
        "policy": "expenditure_policy_v3_1.docx",
        "erp": "erp_payments_q3.csv",
        "bank": "bank_feed_q3.json",
        "approvals": "approvals_pws_q3.xml",
        "vendors": "approved_vendors.xlsx",
        "emails": "finance_emails_q3.txt",
        "board_report": "q3_finance_summary.pdf",
    },
    descriptions={
        "policy": "Expenditure Approval Policy v3.1 (Word document)",
        "erp": "Payments posted in the finance system (ERP) in Q3",
        "bank": "Bank statement feed for Q3, as returned by the bank's API",
        "approvals": "Approval requests exported from the Procurement Workflow System (PWS)",
        "vendors": "Approved Vendor List maintained by Procurement (Excel workbook)",
        "emails": "Finance-related emails collected for the review",
        "board_report": "Q3 Finance Summary presented to the Board (PDF)",
    },
    text_sources={"erp", "bank", "approvals", "emails"},
    peek_hints={"policy": "kit.extract.policy_text(), policy_rules(), policy_thresholds()",
                "vendors": "kit.extract.vendors()",
                "board_report": "kit.extract.board_report()"},
    modules={
        "extract": "ai_learning_kit.problems.brightwater_q3.extract",
        "reference": "ai_learning_kit.problems.brightwater_q3.reference",
        "pipeline": "ai_learning_kit.problems.brightwater_q3.pipeline",
        "report": "ai_learning_kit.problems.brightwater_q3.report",
    },
    workspace_name="brightwater_q3",
    tags=["data formats", "csv", "json", "xml", "docx", "pdf", "xlsx", "extraction", "merging",
          "duplicates", "currency conversion", "rules", "reporting", "OSEMN", "BSc year 1"],
)

# Short names kept for the helper modules in this folder.
SOURCES = PROBLEM.sources
DESCRIPTIONS = PROBLEM.descriptions
TEXT_SOURCES = PROBLEM.text_sources


def path(name: str):
    return PROBLEM.path(name)
