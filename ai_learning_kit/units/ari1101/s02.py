"""ARI1101 Session 02: Data formats. Tasks, questions and formative checks on the Brightwater Q3 case.

In this session students read CSV, JSON and XML themselves, choose formats for
what they extract, and convert between formats. Document extraction, the
compliance check and report layout are done by the kit.

Every check returns a Feedback: which parts of the answer are right, which are
not, the evidence in the case files that shows why, and how to verify it. After
several attempts a suggested answer with an explanation is added.
"""
from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd

from ...core.checks import all_of, any_of, as_int, at_least, numeric_check, saved_only, written
from ...core.feedback import Feedback
from ...core.session import attach_guides, task
from ...problems.brightwater_q3 import extract, pipeline, reference
from ...problems.brightwater_q3.case import SOURCES
from .s02_guide import EXAMPLE_CASE, GUIDE

UNIT = "ARI1101"
SESSION = "S02"
TITLE = "Data formats: gathering the evidence for the Q3 compliance review"
PROBLEM = "brightwater_q3"


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
        "erp_n": len(erp),
        "erp_gt5000_any": int((erp["amount"] > 5000).sum()),
        "erp_gt5000_eur_only": int(((erp["currency"] == "EUR") & (erp["amount"] > 5000)).sum()),
        "erp_ge5000_any": int((erp["amount"] >= 5000).sum()),
        "erp_high": len(hv),
        "bank_keys": ["bank", "account", "statement_period", "generated_at", "transactions"],
        "bank_n": len(bank),
        "bank_high": int((bank["amount_eur"] > 5000).sum()),
        "bank_no_conversion": int((bank["amount.value"] > 5000).sum()),
        "bank_eur_only": int(((bank["amount.currency"] == "EUR") & (bank["amount.value"] > 5000)).sum()),
        "bank_dupes": sorted(bank.loc[bank["erp_ref"].duplicated(), "erp_ref"].dropna().unique()),
        "requests": len(req),
        "req_lt2": int((req["n_approvers"] < 2).sum()),
        "req_le2": int((req["n_approvers"] <= 2).sum()),
        "high_with_request": len(hv.merge(req, on="erp_ref")),
        "rules": len(extract.policy_rules()),
        "noncompliant": int((~findings["compliant"]).sum()),
        "compliant": int(findings["compliant"].sum()),
        "split_payments": int(splits["n_payments"].sum()),
        "split_groups": len(splits),
        "stated": extract.board_report()["stated_high_value_count"],
    }


# --------------------------------------------------------------------------- T1.1
FORMAT_EVIDENCE = {
    "policy": ("docx", "the file is expenditure_policy_v3_1.docx (see kit.sources()). kit.peek('policy') shows it "
                       "starts with PK, the signature of a ZIP archive, which is how Word and Excel files are stored."),
    "erp": ("csv", "the file is erp_payments_q3.csv; kit.peek('erp') shows plain text lines with values separated by commas."),
    "bank": ("json", "the file is bank_feed_q3.json; kit.peek('bank') shows text starting with '{' and \"key\": value pairs."),
    "approvals": ("xml", "the file is approvals_pws_q3.xml; kit.peek('approvals') starts with <?xml ...?> and tags such as <Request>."),
    "vendors": ("xlsx", "the file is approved_vendors.xlsx, an Excel workbook (see kit.sources()). kit.peek('vendors') "
                        "shows the bytes start with PK: Word (.docx) and Excel (.xlsx) are both ZIP archives, so the "
                        "first bytes cannot tell them apart and the extension decides."),
    "emails": ("txt", "the file is finance_emails_q3.txt; kit.peek('emails') shows plain text with From:, To:, Subject: lines."),
    "board_report": ("pdf", "the file is q3_finance_summary.pdf (see kit.sources()); kit.peek('board_report') shows its "
                            "first bytes are %PDF, the signature of a PDF file, not PK as for Word."),
}
SHAPE_EVIDENCE = {
    "policy": ({"document", "text"}, "document",
               "the policy is numbered sections of prose with one table; there is no row-per-record structure."),
    "erp": ({"table"}, "table", "a header line and then one line per payment, with the same nine fields on every line."),
    "bank": ({"nested"}, "nested",
             "one JSON object holding a list of transactions, and each transaction holds further objects "
             "(amount, counterparty): objects inside objects."),
    "approvals": ({"tree"}, "tree",
                  "elements inside elements: <ApprovalLog> holds <Request>, which holds <Vendor> and <Approvers>, "
                  "which holds <Approver>. Values also sit in attributes."),
    "vendors": ({"table"}, "table",
                "one row per vendor with the same five columns (Vendor ID, name, category, approved since, status). "
                "Run display(kit.extract.vendors()) to see it. The title lines above the header do not make it a document."),
    "emails": ({"document", "text"}, "document",
               "free text: each email has a few header lines and a body of sentences, with no fixed fields to fill."),
    "board_report": ({"document"}, "document",
                     "a laid-out report: paragraphs, a table and a chart on a page. It contains a table, "
                     "but as a whole it is a document."),
}


def _check_t11(v):
    if not isinstance(v, dict):
        return Feedback(False, f"Record a dictionary; you recorded a {type(v).__name__}.",
                        ["Use the survey dictionary from the cell, with one entry per source."])
    missing = [s for s in SOURCES if s not in v]
    details, right = [], []
    for s in SOURCES:
        if s in missing:
            continue
        entry = v[s] if isinstance(v[s], dict) else {}
        fmt = str(entry.get("format", "")).strip().lower().lstrip(".")
        shape = str(entry.get("shape", "")).strip().lower()
        exp_fmt, fmt_ev = FORMAT_EVIDENCE[s]
        ok_shapes, exp_shape, shape_ev = SHAPE_EVIDENCE[s]
        problems = []
        if fmt != exp_fmt:
            problems.append(f"✗ {s}, format: you wrote {fmt or 'nothing'!r}. Evidence: {fmt_ev}")
        if shape not in ok_shapes:
            problems.append(f"✗ {s}, shape: you wrote {shape or 'nothing'!r}. Evidence: {shape_ev}")
        if problems:
            details += problems
        else:
            right.append(s)
    if missing:
        details.append(f"✗ Missing from your dictionary: {', '.join(missing)}.")
    if not details:
        return Feedback(True, "All seven formats and shapes match the evidence in the files.")
    n_ok = len(right)
    if right:
        details.insert(0, f"✓ Right: {', '.join(right)}.")
    suggestion = "\n".join(f"{s:<13} format {FORMAT_EVIDENCE[s][0]!r:<7} shape {SHAPE_EVIDENCE[s][1]!r}"
                           for s in SOURCES)
    return Feedback(False, f"{n_ok} of {len(SOURCES)} sources fully right.", details, suggestion=suggestion)


# --------------------------------------------------------------------------- other checks
def _check_shape(v):
    rows, cols = _ref()["erp_shape"]
    try:
        r, c = (int(x) for x in v)
    except (TypeError, ValueError):
        return Feedback(False, f"Record erp.shape, a (rows, columns) pair. You recorded {v!r}.",
                        ["erp.shape has no brackets after it: it is an attribute, not a method."],
                        suggestion=f"({rows}, {cols}): {rows} payments and {cols} columns in the CSV file.")
    if (r, c) == (rows, cols):
        return Feedback(True, f"({r}, {c}) matches: one row per payment and nine columns.")
    details = [f"You recorded ({r}, {c})."]
    if r != rows:
        details.append(f"✗ Rows: {r} is {'more' if r > rows else 'fewer'} than the file holds. "
                       "Did you read the whole file, and is the header line counted as column names (the default)?")
    else:
        details.append("✓ Rows are right.")
    if c != cols:
        details.append(f"✗ Columns: {c}. The file's header line names nine fields. If you added a column "
                       "(for example amount_eur) before recording the shape, record the shape straight after reading.")
    else:
        details.append("✓ Columns are right.")
    return Feedback(False, "Not yet right.", details,
                    suggestion=f"({rows}, {cols}): print(erp.shape) immediately after pd.read_csv.")


def _check_true(v):
    if v is True:
        return Feedback(True, "The round trip through JSON preserved every rule exactly.")
    if v is False:
        return Feedback(False, "You recorded False: the reloaded object differs from the original.",
                        ["Check that you dumped `rules` (not the file name or another variable) and loaded it "
                         "with json.load(f), then compared with rules_reloaded == rules.",
                         "Print both and compare rule R2: the nested dictionaries should be identical."],
                        suggestion="True. json.dump writes dictionaries, lists, numbers and text, and json.load "
                                   "rebuilds exactly the same structure, so rules_reloaded == rules.")
    return Feedback(False, f"Record the result of the comparison (True or False). You recorded {v!r}.",
                    ["same = rules_reloaded == rules gives a single True or False."],
                    suggestion="True, from same = rules_reloaded == rules.")


def _check_t23(v):
    if isinstance(v, str) and v.strip().startswith(("{", "[")):
        return Feedback(True, "Yes: after the CSV round trip the nested value is a string that only looks like a dictionary.",
                        ["Check your reasoning says why: CSV stores text only, so the structure is lost."])
    if isinstance(v, dict):
        return Feedback(False, "You recorded a dictionary. Did you take it from rules_df (before saving) "
                               "instead of rules_csv (after reading the CSV back)?",
                        ["Use rules_csv.loc[1, 'approvers'] and look at type(value)."],
                        suggestion="A string such as \"{'count': 2, 'roles': [...]}\": CSV turned the dictionary into text.")
    return Feedback(None, f"You recorded {v!r} ({type(v).__name__}). Look at type(value): is it still a dictionary?")


def _check_t24(v):
    if not isinstance(v, pd.DataFrame):
        return Feedback(False, f"Record the thresholds DataFrame; you recorded a {type(v).__name__}.")
    details = []
    col = next((c for c in v.columns if "approver" in str(c).lower()), None)
    if len(v) != 3:
        details.append(f"✗ The policy table has 3 bands (A, B, C); your DataFrame has {len(v)} rows.")
    if col is None:
        details.append("✗ No column for the number of approvers was found. Keep 'Approvers required'.")
    elif not pd.api.types.is_numeric_dtype(v[col]):
        details.append(f"✗ {col!r} has dtype {v[col].dtype}, so the numbers are still text, as written in the Word table. "
                       "Convert with pd.to_numeric(thresholds['Approvers required']) and assign the result back to the column.")
    if not details:
        return Feedback(True, f"Three bands, with {col!r} stored as {v[col].dtype}.")
    return Feedback(False, "Not yet right.", details,
                    suggestion="thresholds['Approvers required'] = pd.to_numeric(thresholds['Approvers required'])\n"
                               "print(thresholds.dtypes) should then show int64 for that column.")


def _check_keys(v):
    expected = _ref()["bank_keys"]
    try:
        got = [str(k) for k in v]
    except TypeError:
        return Feedback(False, f"Record list(bank_doc.keys()). You recorded {v!r}.")
    if set(got) == set(expected):
        return Feedback(True, "These are the five top-level keys; the payments are inside 'transactions'.")
    details = [f"You recorded {got}."]
    if {"amount", "counterparty", "bank_ref"} & set(got):
        details.append("Those are the keys of one transaction, one level down. Record the keys of the whole document, bank_doc.")
    missing, extra = set(expected) - set(got), set(got) - set(expected)
    if missing:
        details.append(f"Missing: {sorted(missing)}.")
    if extra:
        details.append(f"Not top-level keys of the document: {sorted(extra)}.")
    return Feedback(False, "Not yet right.", details, suggestion=f"{expected}, from list(bank_doc.keys()).")


def _check_cols(v):
    cols = [str(c) for c in v] if isinstance(v, (list, tuple, pd.Index)) else []
    need = {"amount.value", "amount.currency", "counterparty.name", "counterparty.iban"}
    if need <= set(cols):
        return Feedback(True, "The nested amount and counterparty objects became dotted columns.")
    details = [f"You recorded {cols or v!r}."]
    if "amount" in cols or "counterparty" in cols:
        details.append("'amount' and 'counterparty' are still single columns holding dictionaries: the list was not "
                       "flattened. Use pd.json_normalize(transactions), not pd.DataFrame(transactions).")
    elif "transactions" in cols:
        details.append("You flattened the whole document. Pass the list bank_doc['transactions'] to pd.json_normalize.")
    else:
        details.append(f"Expected columns such as {sorted(need)}.")
    return Feedback(False, "Not yet right.", details,
                    suggestion="bank = pd.json_normalize(transactions) gives bank_ref, value_date, remittance_info, "
                               "amount.value, amount.currency, counterparty.name and counterparty.iban.")


def _check_dupes(v):
    expected = _ref()["bank_dupes"]
    try:
        got = sorted({str(x) for x in v})
    except TypeError:
        return Feedback(False, f"Record a list of ERP references. You recorded {v!r}.")
    if got == expected:
        return Feedback(True, f"{len(got)} ERP references appear twice in the bank feed.")
    details = [f"You recorded {len(got)} reference(s)."]
    if any(x in ("nan", "None") for x in got):
        details.append("Your list contains a missing value. The bank line without an ERP reference is not a duplicate; "
                       "drop it before looking for duplicates, or use .dropna().")
    if not got:
        details.append("An empty list: check that the erp_ref column was filled by .str.extract and is not all NaN.")
    right = set(got) & set(expected)
    if right and len(right) < len(expected):
        details.append(f"{len(right)} of your references {'is a duplicate' if len(right) == 1 else 'are duplicates'}; "
                       f"{len(expected) - len(right)} more exist(s).")
    wrong = set(got) - set(expected) - {"nan", "None"}
    if wrong:
        details.append(f"{len(wrong)} of your references appear only once in the bank feed.")
    details.append("To check: bank[bank['erp_ref'].duplicated(keep=False)] lists every row whose reference occurs twice.")
    return Feedback(False, "Not yet right.", details, suggestion=f"{expected}: each appears on two bank lines one day apart.")


def _check_t52(v):
    if not isinstance(v, pd.DataFrame):
        return Feedback(False, f"Record the pws DataFrame; you recorded a {type(v).__name__}.")
    details = []
    n = _ref()["requests"]
    if len(v) != n:
        details.append(f"✗ {len(v)} rows. There should be one row per <Request> element; len(requests) tells you how many.")
    missing = {"erp_ref", "n_approvers"} - set(v.columns)
    if missing:
        details.append(f"✗ Missing columns: {sorted(missing)}.")
    if "amount" in v.columns and not pd.api.types.is_numeric_dtype(v["amount"]):
        details.append("✗ 'amount' is still text: XML attributes are always strings. Use pd.to_numeric(pws['amount']).")
    if "n_approvers" in v.columns and not pd.api.types.is_numeric_dtype(v["n_approvers"]):
        details.append("✗ 'n_approvers' should be a number: len(req.findall('Approvers/Approver')).")
    if not details:
        return Feedback(True, f"{len(v)} requests, one row each, with amount and number of approvers as numbers.")
    return Feedback(False, "Not yet right.", details,
                    suggestion="amount = req.get('amount'), currency = req.get('currency'),\n"
                               "n_approvers = len(approvers), then pws['amount'] = pd.to_numeric(pws['amount']).")


def _check_t61(v):
    if not isinstance(v, pd.DataFrame):
        return Feedback(False, f"Record the review DataFrame; you recorded a {type(v).__name__}.")
    ref = _ref()
    details = []
    if "erp_ref" not in v.columns:
        details.append("✗ The erp_ref column is missing; it is the key that links the sources.")
    elif v["erp_ref"].duplicated().any():
        details.append(f"✗ {int(v['erp_ref'].duplicated().sum())} erp_ref value(s) repeat. A merge multiplies rows "
                       "when the right-hand table has the same key more than once.")
    if len(v) == ref["high_with_request"] and len(v) != ref["erp_high"]:
        details.append(f"✗ {len(v)} rows: this is the number of payments that have a PWS request. An inner merge "
                       "(the default) dropped the payments without one, which are among the most important. Use how='left'.")
    elif len(v) != ref["erp_high"]:
        details.append(f"✗ {len(v)} rows. The table should hold exactly the payments from T3.3, one row each. "
                       "Check that `high` uses the euro amounts and '>'.")
    if not details:
        return Feedback(True, f"{len(v)} payments, one row each, including those without an approval request.")
    return Feedback(False, "Not yet right.", details,
                    suggestion=f"high.merge(pws[['erp_ref', 'n_approvers', 'roles']], on='erp_ref', how='left') "
                               f"gives {ref['erp_high']} rows.")


def _check_sizes(v):
    if isinstance(v, dict) and {"csv", "json"} <= {str(k).lower() for k in v}:
        s = {str(k).lower(): val for k, val in v.items()}
        try:
            ratio = float(s["json"]) / float(s["csv"])
            return Feedback(True, f"Sizes recorded: the JSON file is {ratio:.1f} times the size of the CSV file.",
                            ["Explain where the difference comes from in your reasoning."])
        except (TypeError, ValueError, ZeroDivisionError):
            pass
    return Feedback(False, f"Record a dictionary {{'csv': bytes, 'json': bytes}}. You recorded {v!r}.",
                    ["The cell builds `sizes` for you with Path(...).stat().st_size; record that variable."])


def _check_t81(v):
    ref = _ref()
    if not isinstance(v, dict):
        return Feedback(False, f"Record a dictionary {{'stated': ..., 'ours': ...}}. You recorded {v!r}.")
    details = []
    stated, ours = as_int(v.get("stated")), as_int(v.get("ours"))
    if stated == ref["stated"]:
        details.append(f"✓ stated: {stated}, as in the Board report.")
    else:
        details.append(f"✗ stated: you recorded {v.get('stated')!r}. Use board['stated_high_value_count'], "
                       "the number the report claims in its overview.")
    if ours == ref["erp_high"]:
        details.append(f"✓ ours: {ours}, your count from T3.3.")
    else:
        details.append(f"✗ ours: you recorded {v.get('ours')!r}. Use your count from T3.3, len(high).")
    ok = stated == ref["stated"] and ours == ref["erp_high"]
    return Feedback(ok, "Both counts are right. Now explain the difference in your reasoning." if ok else "Not yet right.",
                    details, suggestion=f"{{'stated': {ref['stated']}, 'ours': {ref['erp_high']}}}")




# --------------------------------------------------------------------------- written answers
_SOURCES_DIRECT = at_least(2, any_of(r"\bcsv\b", r"\berp\b"), any_of(r"\bjson\b", r"\bbank"), any_of(r"\bxml\b", r"approval", r"\bpws\b"))
_EXTRACT = all_of(any_of(r"polic", r"\bword\b", r"docx", r"\bpdf\b", r"board", r"e-?mail"),
                any_of(r"extract", r"document", r"sentence", r"paragraph", r"layout", r"binary", r"unstructured",
                     r"not structured", r"free text", r"written", r"text"))
_COMBINE = any_of(r"combin", r"together", r"more than one", r"several", r"no single", r"not .{0,20}single", r"\blink",
                r"merg", r"\bjoin", r"each source", r"cross", r"\bboth\b", r"all of them", r"multiple")
_ROLES = at_least(3, any_of(r"\berp\b", r"payment", r"paid"), any_of(r"approv", r"\bpws\b"),
                any_of(r"polic", r"\brule"), any_of(r"vendor", r"supplier"), any_of(r"\bbank"))

WRITTEN = {
    "T1.2": written([
        ("It names the sources that can be read directly (ERP as CSV, bank feed as JSON, approvals as XML).",
         _SOURCES_DIRECT, "Name the sources in the first group and the tool that reads each: pd.read_csv, json.load, ET.parse."),
        ("It says which sources need extraction (policy, board report, emails) and why.",
         _EXTRACT, "Say why a Word or PDF document cannot simply be loaded as data: the facts are inside sentences, "
                   "paragraphs and page layout."),
        ("It explains that the question needs several sources combined.",
         _COMBINE, "Say why one source is not enough on its own."),
        ("It says which source gives which part of the answer.",
         _ROLES, "For example: the ERP says what was paid, the approvals say who approved it, the policy says who "
                 "should have, the vendor list says who may be paid."),
    ], model=(
        "The ERP export (CSV), the bank feed (JSON) and the approvals (XML) are already structured data, so "
        "pd.read_csv, json.load and ET.parse read them directly. The policy (Word), the board report (PDF) and the "
        "emails hold their information in sentences and page layout, so it has to be extracted first; the Excel "
        "vendor list can be opened with pd.read_excel once you skip its title lines. No single source answers the "
        "question: the ERP says what was paid, the bank confirms it left the account, the approvals say who "
        "authorised it, the policy says who should have, and the vendor list says who may be paid.")),
    "T2.1": written([
        ("It says that rules is a list.", any_of(r"\blist"), "type(rules) shows the type of the whole object."),
        ("It says that each rule is a dictionary.", any_of(r"dict"), "type(rules[0]) shows the type of one item."),
    ], model="rules is a list of dictionaries, one dictionary per rule or exception."),
    "T2.3": written([
        ("It says the reloaded value is a string (text), not a dictionary.", any_of(r"\bstr", r"string", r"\btext"),
         "Look at the type printed next to the value."),
        ("It says the nested structure was lost.", any_of(r"lost", r"lose", r"flat", r"no longer", r"structure", r"nest"),
         "What can you no longer do with the value, for example read its 'roles' list?"),
        ("It chooses JSON for the rules.", any_of(r"\bjson"), "Which format gave the rules back unchanged in T2.2?"),
    ], model=(
        "After the CSV round trip the approvers value is a string that only looks like a dictionary: CSV stores text "
        "only, so the nested structure is lost and the roles cannot be used without re-parsing the text. JSON gave "
        "the rules back identical (T2.2), so JSON is the right format for the rules.")),
    "T3.4": written([
        ("It excludes the payment of exactly €5,000, because the policy says 'above'.",
         all_of(any_of(r"5[,.]?000"), any_of(r"above", r"more than", r"greater", r"strictly", r">", r"exceed")),
         "Quote the word the policy uses and say what it means for a payment of exactly 5,000.00."),
        ("It converts the USD payment to euro and finds it below the threshold.",
         any_of(r"4[,.]?876", r"convert", r"euro equivalent", r"0\.92", r"\br7\b"),
         "Apply rule R7: what is 5,300 dollars in euro?"),
        ("It reaches a decision for each payment.",
         any_of(r"not .{0,25}(includ|review|part)", r"exclud", r"neither", r"should not", r"does not qualify",
              r"out of scope", r"\bbelow\b", r"not above"),
         "Say clearly whether each payment is in or out of the review."),
    ], model=(
        "Neither payment belongs in the review. Rule R1 applies to expenditure 'above €5,000', so a payment of exactly "
        "€5,000.00 does not qualify. Rule R7 says thresholds apply to the euro equivalent: USD 5,300 × 0.92 = €4,876, "
        "which is below €5,000.")),
    "T4.4": written([
        ("It suggests that some payments appear twice in the bank feed.",
         any_of(r"duplicat", r"twice", r"two times", r"double", r"repeat", r"more than once"),
         "Could the bank have paid the same invoice twice?"),
        ("It suggests that some bank payments were never recorded in the ERP.",
         any_of(r"not recorded", r"missing", r"unrecorded", r"not in the erp", r"no erp", r"never .{0,15}(record|enter)",
              r"not .{0,10}entered", r"outside", r"without"),
         "Could money leave the account without a matching ERP entry?"),
    ], model=(
        "The bank shows three more large payments than the ERP. A bank statement records the money that actually "
        "left the account, so the same invoice can appear twice if it was paid twice, and a payment can appear "
        "that finance never recorded in the ERP. T4.5 finds both: two duplicated references and one payment with "
        "no ERP reference.")),
    "T5.4": written([
        ("It names values read from attributes.", all_of(any_of(r"attribut"), any_of(r"amount", r"erp_ref", r"role", r"currency", r"decision", r"date")),
         "Which values sit inside the opening tag, like amount=\"...\"?"),
        ("It names values that are element text.", all_of(any_of(r"\btext"), any_of(r"name", r"vendor", r"description", r"approver")),
         "Which value sits between <Approver ...> and </Approver>?"),
        ("It says that amount had to be converted because XML stores text.",
         all_of(any_of(r"to_numeric", r"convert", r"numeric", r"number", r"float"), any_of(r"text", r"string", r"\bstr")),
         "What type did req.get('amount') return?"),
    ], model=(
        "erp_ref, amount and currency are attributes of <Request>; role, decision and date are attributes of "
        "<Approver>. The approver's name, the vendor name and the description are element text. XML stores "
        "everything as text, so amount had to be converted with pd.to_numeric (and dates would need pd.to_datetime "
        "before comparing them with payment dates).")),
    "T6.2": written([
        ("It explains why the JSON file is bigger.",
         any_of(r"repeat", r"field names", r"column names", r"\bkeys\b", r"every record", r"each record", r"verbose",
              r"names .{0,20}(each|every)"),
         "Print the start of the JSON file: what is written again in every record?"),
        ("It chooses a format for the finance team and says why.",
         all_of(any_of(r"\bcsv"), any_of(r"excel", r"finance", r"spreadsheet", r"smaller", r"simple", r"open")),
         "Which format opens directly in Excel?"),
        ("It says what the formats lose or keep when read back.",
         any_of(r"dtype", r"\btype", r"date", r"lost", r"lose", r"keep", r"nest", r"as text"),
         "Look at the dtypes table: which column types survive, and would a date or a list survive?"),
    ], model=(
        "The table is flat, one row per payment, so CSV represents it fully and is less than half the size of the "
        "JSON, which repeats every field name in every record. The finance team can open CSV in Excel, so CSV is "
        "the better choice for them. Both files return the same dtypes here only because posting_date is still "
        "text: neither format has a date type, so dates come back as text. JSON would be the choice for a program "
        "or an API, or if a row needed nested values such as a list of approvers.")),
    "T6.3": written([
        ("It describes the structure pandas wrote (one <row> per row, one element per column).",
         any_of(r"\brow", r"element per", r"each column", r"one element"), "Look at the first <row> in the printout."),
        ("It compares with the PWS export (attributes, nesting).",
         any_of(r"attribut", r"nest", r"flat"), "Where did the PWS export keep erp_ref, and how were approvers stored?"),
    ], model=(
        "pandas writes one <row> element per row with one child element per column, all as element text. The PWS "
        "export used attributes for key fields and nested <Approver> elements; the table has no nesting, so the "
        "roles survive only as one joined string. XML written from a DataFrame is a flat table dressed as a tree.")),
    "T7.2": written([
        ("It says the narrow question misses payments below €5,000 that add up to more.",
         all_of(any_of(r"below", r"under", r"less than", r"smaller"),
              any_of(r"split", r"separate", r"several", r"combin", r"add up", r"together", r"total")),
         "What happens to three payments of €4,800 to the same supplier in one week, if you filter amount > 5000?"),
        ("It says what the emails add (intent or context the tables cannot show).",
         all_of(any_of(r"e-?mail"), any_of(r"intent", r"deliberate", r"on purpose", r"avoid", r"ask", r"request", r"why",
                                    r"context", r"evidence", r"explain")),
         "Read the Marketing email: what does it show that the payments table cannot?"),
    ], model=(
        "Five payments in two groups are each below €5,000 but add up to more than €5,000 for the same supplier "
        "within 7 days, so a filter on 'amount > 5000' never sees them. The Marketing email shows that the Kestrel "
        "split was requested deliberately to avoid the approval route: text adds intent and context that the "
        "tables cannot carry.")),
    "T8.1": written([
        ("It compares the report's wording ('5,000 or more') with the policy's ('above').",
         any_of(r"or more", r">=", r"at least", r"equal", r"exactly", r"inclusive", r"above"),
         "Read board['stated_threshold'] and compare it with rule R1."),
        ("It explains the role of currency (the USD payment counted unconverted).",
         any_of(r"usd", r"dollar", r"currenc", r"convert", r"euro equivalent"),
         "Was the USD 5,300 payment counted in dollars or in euro?"),
        ("It accounts for the size of the difference.",
         any_of(r"\btwo\b", r"\b2\b", r"both", r"each"),
         "How many payments does each cause add?"),
    ], model=(
        "The Board report counts 41 payments 'of EUR 5,000 or more', using the amounts as recorded, while the policy "
        "asks for payments above €5,000 in euro. The difference of two is the payment of exactly €5,000.00 "
        "(included by 'or more') and the USD 5,300 payment (counted in dollars, but €4,876 in euro). The report "
        "also states that all such payments complied with the policy, which the review contradicts.")),
    "T8.2v": written([
        ("It gives the answer: the number of payments above €5,000 and how many did not comply.",
         all_of(any_of(r"\b39\b", r"thirty-nine"), any_of(r"\b11\b", r"eleven")),
         "State both numbers from T3.3 and T7.1."),
        ("It names the main kinds of breach.",
         at_least(2, any_of(r"approv"), any_of(r"supplier", r"vendor", r"suspend"), any_of(r"late", r"after payment", r"timing"),
                any_of(r"reject"), any_of(r"cfo", r"finance manager"), any_of(r"split")),
         "Look at the issues column of findings: missing approvers, late approvals, suspended or unlisted suppliers..."),
        ("It states a limitation of the review.",
         any_of(r"limit", r"\bonly\b", r"could not", r"cannot", r"not includ", r"rel(y|ies|ied) on", r"assum", r"not see",
              r"outside"),
         "What could the review not see, given the sources it used?"),
    ], model=(
        "In Q3 2026 Brightwater made 39 payments above €5,000 in euro terms. Eleven of them, worth about €168,000, did "
        "not comply with the Expenditure Approval Policy: most lacked a required approver or were approved late, and "
        "some had no approval in PWS or were paid to a suspended or unlisted supplier. Five further payments below "
        "€5,000 form two possible split purchases, one of them requested deliberately. The review relies on the PWS "
        "export and the emails provided; approvals recorded elsewhere would not have been seen."), what="findings"),
    "T8.2": written([
        ("It chooses a document format for the Board (Word, PDF or the HTML page).",
         any_of(r"word", r"docx", r"\bpdf", r"html", r"report", r"document"), "Which file did the kit create for people to read?"),
        ("It says why CSV or JSON would not suit the Board.",
         all_of(any_of(r"\bcsv", r"\bjson"), any_of(r"raw", r"data", r"chart", r"narrative", r"explan", r"read", r"machine",
                                               r"program", r"row", r"context", r"layout")),
         "What does a Board member need that a CSV or JSON file cannot give?"),
    ], model=(
        "The Board needs a readable document with explanation, a chart and selected tables, so the Word or PDF report "
        "(or the HTML page) is right. CSV and JSON are formats for data that will be processed further; they carry "
        "no narrative, no chart and no emphasis, and a Board member would have to interpret raw rows.")),
}


TASKS = {
    "T1.1": task("Format and shape of each source",
                  "For each of the seven sources, record its file format and the shape of its data "
                  "(table, nested, tree or document) in the survey dictionary.",
                  _check_t11, needs_value=True),
    "T1.2": task("What can be read directly",
                  "(a) Which sources can Python read directly with pandas, json or ElementTree, and which need their "
                  "information extracted first? (b) Which source gives which part of the answer, and why is no single "
                  "source enough?",
                  needs_reasoning=True, written=WRITTEN["T1.2"]),
    "T2.1": task("Number of policy rules",
                  "How many items does kit.extract.policy_rules() return, and what Python type is the result and each item?",
                  numeric_check(_ref, "rules", "print(len(rules)) and [r['rule_id'] for r in rules]",
                                 "Eight rules (R1 to R8) and one exception (E1), one dictionary each.",
                                 wrong={8: "you counted the rules R1 to R8 but not the exception E1, which is also an item "
                                           "in the list. len(rules) counts every item."},
                                 evidence="the list holds one dictionary per rule or exception; each has a 'rule_id' key.",
                                 right="Eight rules and one exception."),
                  needs_value=True, needs_reasoning=True, written=WRITTEN["T2.1"]),
    "T2.2": task("Rules round trip through JSON",
                  "Save the rules to policy_rules.json, read them back, and record whether the reloaded object equals the original.",
                  _check_true, needs_value=True),
    "T2.3": task("Rules round trip through CSV",
                  "Save the rules as CSV and read them back. Record the reloaded 'approvers' value of rule R2. What happened "
                  "to the nested fields, and which format suits the rules?",
                  _check_t23, needs_value=True, needs_reasoning=True, written=WRITTEN["T2.3"]),
    "T2.4": task("Thresholds table as a DataFrame",
                  "Turn the policy's threshold table into a DataFrame with the number of approvers stored as a number.",
                  _check_t24, needs_value=True),
    "T3.1": task("ERP shape", "Read the ERP CSV file and record its shape (rows, columns).", _check_shape, needs_value=True),
    "T3.2": task("ERP rows with amount > 5000",
                  "How many ERP rows have an amount greater than 5000, ignoring currency?",
                  numeric_check(_ref, "erp_gt5000_any", "(erp['amount'] > 5000).sum()",
                                 "Every row with an amount above 5000, whatever its currency.",
                                 wrong={"erp_gt5000_eur_only": "that is the number of EUR rows only. This task ignores "
                                                               "currency: include the USD rows too (T3.3 deals with currency).",
                                        "erp_high": "that is the euro-converted count, which is T3.3. Here, compare the "
                                                    "amount column with 5000 as it stands.",
                                        "erp_ge5000_any": "that count includes a payment of exactly 5000.00. 'Greater than' "
                                                          "means >, not >=."},
                                 evidence="the ERP has 426 rows; erp['amount'].describe() shows the range of amounts, "
                                          "and one payment is exactly 5000.00."),
                  needs_value=True),
    "T3.3": task("ERP payments above €5,000 in euro",
                  "How many ERP payments are above €5,000 once USD amounts are converted at the R7 rate?",
                  numeric_check(_ref, "erp_high", "erp['amount_eur'] = erp['amount'].where(erp['currency'] == 'EUR', "
                                             "erp['amount'] * rate), then (erp['amount_eur'] > 5000).sum()",
                                 "USD amounts are converted at 0.92 EUR per USD (rule R7), then payments strictly above "
                                 "5000 are counted. The USD 5,300 payment becomes €4,876 and drops out; the payment of "
                                 "exactly €5,000.00 is not above €5,000.",
                                 wrong={"erp_gt5000_any": "that is the same as T3.2. Either the USD amounts were not "
                                                          "converted (print erp[erp['currency'] == 'USD'] and look at "
                                                          "amount_eur), or you used >= and included the payment of exactly 5,000.00.",
                                        "erp_gt5000_eur_only": "that counts EUR rows only. Convert the USD rows instead of "
                                                               "leaving them out: some of them are above €5,000 in euro.",
                                        "erp_ge5000_any": "that count uses >= on the unconverted amounts: it includes the "
                                                          "payment of exactly 5,000.00 and treats dollars as euro."},
                                 evidence="17 ERP payments are in USD (erp['currency'].value_counts()). Three of them are "
                                          "above 4,000 dollars: look at erp[(erp['currency'] == 'USD') & (erp['amount'] > 4000)] "
                                          "and work out which stay above €5,000 after conversion.",
                                 right="USD amounts converted, and only payments strictly above €5,000 counted."),
                  needs_value=True),
    "T3.4": task("Boundary and currency cases",
                  "Should the payment of exactly €5,000.00 and the USD 5,300.00 payment be in the review? Justify each "
                  "with the policy's wording.",
                  needs_reasoning=True, written=WRITTEN["T3.4"]),
    "T4.1": task("Bank feed top-level keys", "Load the bank feed with json.load and record its top-level keys.",
                  _check_keys, needs_value=True),
    "T4.2": task("Bank transactions", "How many transactions does the bank feed contain?",
                  numeric_check(_ref, "bank_n", "len(bank_doc['transactions'])",
                                 "The 'transactions' list holds 429 items, three more than the ERP's 426 payments.",
                                 wrong={"erp_n": "that is the number of rows in the ERP file (T3.1). Count the items in "
                                                 "the bank feed's 'transactions' list instead.",
                                        5: "len(bank_doc) counts the five top-level keys of the document. The payments "
                                           "are one level down, in bank_doc['transactions'].",
                                        "bank_high": "that is the number above €5,000 (T4.4). This task asks for all transactions."},
                                 evidence="bank_doc['transactions'] is a list with one dictionary per transaction."),
                  needs_value=True),
    "T4.3": task("Flattened bank columns", "Flatten the transactions with pd.json_normalize and record the column names.",
                  _check_cols, needs_value=True),
    "T4.4": task("Bank payments above €5,000",
                  "How many bank transactions are above €5,000 in euro? Why might this differ from your ERP count?",
                  numeric_check(_ref, "bank_high", "bank['amount_eur'] = bank['amount.value'].where(bank['amount.currency'] "
                                              "== 'EUR', bank['amount.value'] * rate), then (bank['amount_eur'] > 5000).sum()",
                                 "Convert USD at 0.92 and count amount_eur > 5000. The bank shows three more than the ERP: "
                                 "two payments appear twice and one payment has no ERP record (T4.5).",
                                 wrong={"bank_no_conversion": "that is what you get without converting USD, or with >= "
                                                              "instead of >. Check both.",
                                        "bank_eur_only": "that counts EUR transactions only. Convert the USD ones instead "
                                                         "of leaving them out.",
                                        "erp_high": "that is the ERP count from T3.3. Count the rows of the bank DataFrame instead."},
                                 evidence="17 bank transactions are in USD (bank['amount.currency'].value_counts()); the "
                                          "amounts are in the column 'amount.value'."),
                  needs_value=True, needs_reasoning=True, written=WRITTEN["T4.4"]),
    "T4.5": task("Duplicated bank entries", "Which ERP references appear more than once in the bank feed?",
                  _check_dupes, needs_value=True),
    "T5.1": task("Number of approval requests", "How many <Request> elements does the PWS export contain?",
                  numeric_check(_ref, "requests", "len(root.findall('Request'))",
                                 "The export holds 37 <Request> elements: two high-value payments have no request at all.",
                                 wrong={0: "findall found nothing. Tags are case-sensitive: use 'Request' with a capital R, "
                                           "and call findall on root."},
                                 evidence="kit.peek('approvals', lines=14) shows that each request starts with <Request ...> "
                                          "directly inside <ApprovalLog>."),
                  needs_value=True),
    "T5.2": task("Requests as a DataFrame",
                  "Build a DataFrame with one row per request: erp_ref, amount (numeric), currency, n_approvers and roles.",
                  _check_t52, needs_value=True),
    "T5.3": task("Requests with fewer than two approvers", "How many requests have fewer than two approvers recorded?",
                  numeric_check(_ref, "req_lt2", "(pws['n_approvers'] < 2).sum()",
                                 "Three requests have only the Budget Holder's approval.",
                                 wrong={"req_le2": "that count uses <= 2. 'Fewer than two' means < 2.",
                                        31: "that is the number of requests with exactly two approvers."},
                                 evidence="pws['n_approvers'].value_counts() shows how many requests have 1, 2 or 3 approvers."),
                  needs_value=True),
    "T5.4": task("Attributes and element text",
                  "Which values did you read from attributes and which from element text? Which needed a type "
                  "conversion, and why?",
                  needs_reasoning=True, written=WRITTEN["T5.4"]),
    "T6.1": task("Combined high-value table",
                  "Combine the ERP payments above €5,000 with the approval requests into one table, one row per payment.",
                  _check_t61, needs_value=True),
    "T6.2": task("Choosing a format",
                  "Save the combined table as CSV and as JSON and record both file sizes. Which format would you send to "
                  "the finance team, and what does each format lose when read back?",
                  _check_sizes, needs_value=True, needs_reasoning=True, written=WRITTEN["T6.2"]),
    "T6.3": task("Extension: convert back to XML",
                  "Optional. Write the combined table as XML with df.to_xml(). How does it compare with the PWS export's structure?",
                  needs_reasoning=True, written=WRITTEN["T6.3"]),
    "T7.1": task("Transactions not compliant",
                  "Run kit.pipeline.compliance_check on your saved table. How many transactions are not compliant?",
                  numeric_check(_ref, "noncompliant", "(~findings['compliant']).sum()",
                                 "11 of the 39 payments break at least one rule.",
                                 wrong={"compliant": "that is the number of compliant payments. Count those where compliant is False.",
                                        "erp_high": "that is the number of payments checked (all rows of findings). "
                                                    "Count only the rows where compliant is False."},
                                 evidence="findings['compliant'].value_counts() shows how many payments are True and False."),
                  needs_value=True),
    "T7.2": task("Split purchases and emails",
                  "How many payments are involved in possible split purchases? What would a review that only looks at "
                  "payments above €5,000 miss, and what do the emails add?",
                  numeric_check(_ref, "split_payments", "splits['n_payments'].sum()",
                                 "Two groups: three Kestrel Print payments and two Harbour IT payments.",
                                 wrong={"split_groups": "that is the number of groups (rows of splits). Count the payments in them.",
                                        3: "that is one group only. Add up n_payments over all the groups."},
                                 evidence="display(splits) shows one row per group; n_payments says how many payments each holds."),
                  needs_value=True, needs_reasoning=True, written=WRITTEN["T7.2"]),
    "T8.1": task("Board report against your count",
                  "Record the number of large payments stated in the Board report and your own count. Explain the difference.",
                  _check_t81, needs_value=True, needs_reasoning=True, written=WRITTEN["T8.1"]),
    "T8.2": task("Your findings and the report",
                  "Write your findings (3 to 6 sentences) and generate the report. Which format would you send to the "
                  "Board, and why not CSV or JSON?",
                  WRITTEN["T8.2v"], needs_value=True, needs_reasoning=True, written=WRITTEN["T8.2"]),
}

attach_guides(TASKS, GUIDE, example_case=EXAMPLE_CASE)
