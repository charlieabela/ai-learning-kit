"""Session 02: Data formats. Tasks, questions and formative checks for the Q3 compliance case.

In this session students read CSV, JSON and XML themselves, choose formats for
what they extract, and convert between formats. Document extraction, the
compliance check and report layout are done by the kit.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..core import extract, pipeline, reference
from ..core.case import SOURCES

SESSION = "S02"
TITLE = "Data formats: gathering the evidence for the Q3 compliance review"


# --------------------------------------------------------------------------- reference values
@lru_cache(maxsize=None)
def _ref():
    erp = reference.erp()
    bank = reference.bank()
    req = reference.requests()
    hv = reference.high_value()
    findings = pipeline.compliance_check(hv, verbose=False)
    splits = pipeline.split_purchases()
    return {
        "erp_shape": tuple(pd.read_csv(reference.path("erp")).shape),
        "erp_gt5000_any": int((erp["amount"] > 5000).sum()),
        "erp_gt5000_eur_only": int(((erp["currency"] == "EUR") & (erp["amount"] > 5000)).sum()),
        "erp_ge5000_any": int((erp["amount"] >= 5000).sum()),
        "erp_high": len(hv),
        "bank_keys": ["bank", "account", "statement_period", "generated_at", "transactions"],
        "bank_n": len(bank),
        "bank_high": int((bank["amount_eur"] > 5000).sum()),
        "bank_dupes": sorted(bank.loc[bank["erp_ref"].duplicated(), "erp_ref"].dropna().unique()),
        "requests": len(req),
        "req_lt2": int((req["n_approvers"] < 2).sum()),
        "rules": len(extract.policy_rules()),
        "noncompliant": int((~findings["compliant"]).sum()),
        "split_payments": int(splits["n_payments"].sum()),
        "stated": extract.board_report()["stated_high_value_count"],
    }


def _num(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- checks
FORMATS = {"policy": "docx", "erp": "csv", "bank": "json", "approvals": "xml",
           "vendors": "xlsx", "emails": "txt", "board_report": "pdf"}
SHAPES = {"erp": "table", "bank": "nested", "approvals": "tree"}


def _check_t11(v):
    if not isinstance(v, dict):
        return False, "Record a dictionary with one entry per source, e.g. {'erp': {'format': 'csv', 'shape': 'table'}, ...}."
    missing = [s for s in SOURCES if s not in v]
    if missing:
        return False, f"Missing sources: {', '.join(missing)}."
    off = [s for s in SOURCES if FORMATS[s] not in str(v[s].get("format", "")).lower()]
    shapes = [s for s, sh in SHAPES.items() if sh not in str(v[s].get("shape", "")).lower()]
    if off:
        return False, f"Look again at the file extension and the raw content of: {', '.join(off)}."
    if shapes:
        return False, f"Formats are right. Reconsider the shape of: {', '.join(shapes)} (table, nested or tree?)."
    return True, "Formats and shapes match the kit's description."


def _check_equal(key, hints=None):
    def check(v):
        expected = _ref()[key]
        if _num(v) == expected:
            return True, "Matches the kit's reference value."
        for wrong, hint in (hints or {}).items():
            if _num(v) == _ref()[wrong]:
                return False, hint
        return False, "Does not match the kit's reference value. Check your filter and print intermediate results."
    return check


def _check_shape(v):
    try:
        ok = tuple(v) == _ref()["erp_shape"]
    except TypeError:
        ok = False
    return (True, "Matches.") if ok else (False, "Record df.shape, a (rows, columns) pair, for the ERP file read with pd.read_csv.")


def _check_true(v):
    return (True, "The round trip preserved the rules exactly.") if v is True else \
        (False, "Compare the reloaded object with the original using ==, and record the result (True or False).")


def _check_t23(v):
    if isinstance(v, str) and v.strip().startswith(("{", "[")):
        return True, "Yes: after the CSV round trip the nested value comes back as a string, not a dictionary."
    return None, "Record the value of one reloaded 'approvers' cell and its type; what kind of object is it now?"


def _check_t24(v):
    if not isinstance(v, pd.DataFrame):
        return False, "Record the DataFrame you built from the thresholds table."
    col = next((c for c in v.columns if "approver" in str(c).lower()), None)
    if col is None or len(v) != 3:
        return False, "Expected 3 bands and a column with the number of approvers."
    if not pd.api.types.is_numeric_dtype(v[col]):
        return False, f"The column {col!r} is still text. Convert it with pd.to_numeric."
    return True, "Three bands, with the number of approvers stored as a number."


def _check_keys(v):
    try:
        ok = set(v) == set(_ref()["bank_keys"])
    except TypeError:
        ok = False
    return (True, "Matches.") if ok else (False, "Record list(data.keys()) for the loaded JSON document.")


def _check_cols(v):
    cols = set(map(str, v)) if isinstance(v, (list, tuple, pd.Index)) else set()
    if {"amount.value", "amount.currency", "counterparty.name"} <= cols:
        return True, "The nested amount and counterparty objects were flattened into columns."
    return False, "Flatten the 'transactions' list with pd.json_normalize and record list(df.columns)."


def _check_dupes(v):
    try:
        ok = sorted(map(str, v)) == _ref()["bank_dupes"]
    except TypeError:
        ok = False
    return (True, "These ERP references appear twice in the bank feed.") if ok else \
        (False, "Extract the TX reference from remittance_info, then use .duplicated() on that column.")


def _check_t52(v):
    if not isinstance(v, pd.DataFrame):
        return False, "Record the DataFrame with one row per <Request>."
    if len(v) != _ref()["requests"]:
        return False, "Expected one row per <Request> element; compare len(df) with the number of requests."
    need = {"erp_ref", "n_approvers"}
    if not need <= set(v.columns):
        return False, f"Include the columns: {', '.join(sorted(need))}."
    if "amount" in v.columns and not pd.api.types.is_numeric_dtype(v["amount"]):
        return False, "The amount attribute is still text. Convert it with pd.to_numeric."
    return True, "One row per request, with the number of approvers counted."


def _check_t61(v):
    if not isinstance(v, pd.DataFrame):
        return False, "Record your combined DataFrame of high-value transactions."
    if "erp_ref" not in v.columns:
        return False, "Keep the erp_ref column; it is the key that links the sources."
    if v["erp_ref"].duplicated().any():
        return False, "Some erp_ref values appear more than once. Did the merge multiply rows?"
    if len(v) != _ref()["erp_high"]:
        return False, "The number of rows differs from the kit's selection. Did you convert USD to euro and use '>'?"
    return True, "Same transactions as the kit's selection, one row each."


def _check_sizes(v):
    if isinstance(v, dict) and {"csv", "json"} <= {str(k).lower() for k in v}:
        return True, "Sizes recorded."
    return False, "Record a dictionary such as {'csv': size_in_bytes, 'json': size_in_bytes}."


def _check_t81(v):
    if isinstance(v, dict) and _num(v.get("stated")) == _ref()["stated"] and _num(v.get("ours")) == _ref()["erp_high"]:
        return True, "Both counts match. Now explain the difference."
    return False, "Record {'stated': count claimed in the board report, 'ours': your count from T3.3}."


def _saved(v):
    return None, "Saved. This answer is reviewed by your lecturer, not checked automatically."


# --------------------------------------------------------------------------- task register
TASKS = {
    "T1.1": dict(title="Format and shape of each source", needs_value=True, check=_check_t11,
                 question="For each of the seven sources, record its file format and the shape of its data "
                          "(table, nested, tree, or document/text) as a dictionary."),
    "T1.2": dict(title="What can be read directly", needs_reasoning=True, check=_saved,
                 question="Which sources can you read with pandas, json or ElementTree, and which need "
                          "extraction? Why can the review question not be answered from one source?"),
    "T2.1": dict(title="Number of policy rules", needs_value=True, check=_check_equal("rules"),
                 question="How many rules and exceptions does kit.extract.policy_rules() return, and what Python type is the result?"),
    "T2.2": dict(title="Rules round trip through JSON", needs_value=True, check=_check_true,
                 question="Save the rules to policy_rules.json, read them back, and record whether the reloaded "
                          "object equals the original."),
    "T2.3": dict(title="Rules round trip through CSV", needs_value=True, needs_reasoning=True, check=_check_t23,
                 question="Save the rules as CSV and read them back. Record one reloaded 'approvers' value. What "
                          "happened to the nested fields, and which format suits the rules?"),
    "T2.4": dict(title="Thresholds table as a DataFrame", needs_value=True, check=_check_t24,
                 question="Turn the policy's threshold table into a DataFrame with the number of approvers stored as a number."),
    "T3.1": dict(title="ERP shape", needs_value=True, check=_check_shape,
                 question="Read the ERP CSV and record its shape."),
    "T3.2": dict(title="ERP rows with amount > 5000", needs_value=True,
                 check=_check_equal("erp_gt5000_any", {"erp_gt5000_eur_only": "That is the EUR-only count. This task asks for all rows, whatever the currency."}),
                 question="How many ERP rows have an amount greater than 5000, ignoring currency?"),
    "T3.3": dict(title="ERP payments above €5,000 in euro", needs_value=True,
                 check=_check_equal("erp_high", {"erp_gt5000_any": "Some amounts are in USD. Convert them with the rate in rule R7 first.",
                                                 "erp_ge5000_any": "The policy says 'above €5,000'. Use > rather than >=, and convert USD."}),
                 question="How many ERP payments are above €5,000 once USD amounts are converted at the R7 rate?"),
    "T3.4": dict(title="Boundary and currency cases", needs_reasoning=True, check=_saved,
                 question="Find the payment of exactly 5,000.00 and the USD payment of 5,300.00. Should either "
                          "be in the review? Justify each with the policy's wording."),
    "T4.1": dict(title="Bank feed top-level keys", needs_value=True, check=_check_keys,
                 question="Load the bank feed with json.load and record its top-level keys."),
    "T4.2": dict(title="Bank transactions", needs_value=True, check=_check_equal("bank_n"),
                 question="How many transactions does the bank feed contain?"),
    "T4.3": dict(title="Flattened bank columns", needs_value=True, check=_check_cols,
                 question="Flatten the transactions with pd.json_normalize and record the column names."),
    "T4.4": dict(title="Bank payments above €5,000", needs_value=True, needs_reasoning=True,
                 check=_check_equal("bank_high"),
                 question="How many bank transactions are above €5,000 in euro? Why might this differ from your ERP count?"),
    "T4.5": dict(title="Duplicated bank entries", needs_value=True, check=_check_dupes,
                 question="Which ERP references appear more than once in the bank feed?"),
    "T5.1": dict(title="Number of approval requests", needs_value=True, check=_check_equal("requests"),
                 question="How many <Request> elements does the PWS export contain?"),
    "T5.2": dict(title="Requests as a DataFrame", needs_value=True, check=_check_t52,
                 question="Build a DataFrame with one row per request: erp_ref, amount (numeric), currency, "
                          "n_approvers and the roles that approved."),
    "T5.3": dict(title="Requests with fewer than two approvers", needs_value=True, check=_check_equal("req_lt2"),
                 question="How many requests have fewer than two approvers recorded?"),
    "T5.4": dict(title="Attributes and element text", needs_reasoning=True, check=_saved,
                 question="Which values did you read from attributes and which from element text? Which needed "
                          "a type conversion, and why?"),
    "T6.1": dict(title="Combined high-value table", needs_value=True, check=_check_t61,
                 question="Combine the ERP payments above €5,000 with the approval requests into one table, "
                          "one row per payment."),
    "T6.2": dict(title="Choosing a format", needs_value=True, needs_reasoning=True, check=_check_sizes,
                 question="Save the combined table as CSV and as JSON. Record both file sizes. Which format "
                          "would you send to the finance team, and what does each format lose when read back?"),
    "T6.3": dict(title="Extension: convert back to XML", needs_reasoning=True, check=_saved,
                 question="Optional. Write the combined table as XML with df.to_xml(). How does it compare with "
                          "the PWS export's structure?"),
    "T7.1": dict(title="Transactions not compliant", needs_value=True, check=_check_equal("noncompliant"),
                 question="Run kit.pipeline.compliance_check on your saved table. How many transactions are not compliant?"),
    "T7.2": dict(title="Split purchases and emails", needs_value=True, needs_reasoning=True,
                 check=_check_equal("split_payments"),
                 question="How many payments are involved in possible split purchases? What would a review "
                          "that only looks at transactions above €5,000 miss, and what do the emails add?"),
    "T8.1": dict(title="Board report against your count", needs_value=True, needs_reasoning=True, check=_check_t81,
                 question="Record the number of large payments stated in the board report and your own count. "
                          "Explain the difference."),
    "T8.2": dict(title="Your findings and the report", needs_value=True, needs_reasoning=True, check=_saved,
                 question="Write your findings (3 to 6 sentences) and generate the report. Which format would "
                          "you send to the Board, and why not CSV or JSON?"),
}
