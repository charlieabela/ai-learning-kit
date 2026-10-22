"""Session 02: Data formats. Tasks, questions and formative checks for the Q3 compliance case.

In this session students read CSV, JSON and XML themselves, choose formats for
what they extract, and convert between formats. Document extraction, the
compliance check and report layout are done by the kit.

Every check returns a Feedback: which parts of the answer are right, which are
not, the evidence in the case files that shows why, and how to verify it. After
several attempts a suggested answer with an explanation is added.
"""
from __future__ import annotations

from functools import lru_cache

import pandas as pd

from ..core import extract, pipeline, reference
from ..core.case import SOURCES
from ..core.feedback import Feedback

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


def _num(value):
    try:
        if isinstance(value, bool):
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _numeric_check(key, task_hint, how, wrong=None):
    """Feedback for a single count: says what was recorded, explains known wrong values,
    and otherwise says whether the count is too high or too low."""
    def check(v):
        expected = _ref()[key]
        got = _num(v)
        if got is None:
            return Feedback(False, f"Record a whole number. You recorded {v!r} ({type(v).__name__}).",
                            [task_hint], suggestion=f"{expected}. {how}")
        if got == expected:
            return Feedback(True, f"{got} matches the kit's reference value.")
        details = [f"You recorded {got}."]
        for wrong_key, explanation in (wrong or {}).items():
            if got == _ref()[wrong_key]:
                details.append(explanation)
                break
        else:
            details.append(f"Your count is {'higher' if got > expected else 'lower'} than the reference.")
        details.append(f"To check: {task_hint}")
        return Feedback(False, "Not yet right.", details, suggestion=f"{expected}. {how}")
    return check


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
    stated, ours = _num(v.get("stated")), _num(v.get("ours"))
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


def _saved(v):
    return Feedback(None, "Saved. Your lecturer reads this answer; it is not checked automatically.")


# --------------------------------------------------------------------------- task register
TASKS = {
    "T1.1": dict(title="Format and shape of each source", needs_value=True, check=_check_t11,
                 question="For each of the seven sources, record its file format and the shape of its data "
                          "(table, nested, tree, or document/text) as a dictionary."),
    "T1.2": dict(title="What can be read directly", needs_reasoning=True, check=_saved,
                 question="Which sources can you read with pandas, json or ElementTree, and which need "
                          "extraction? Why can the review question not be answered from one source?"),
    "T2.1": dict(title="Number of policy rules", needs_value=True,
                 check=_numeric_check("rules", "len(rules) counts the items in the list returned by kit.extract.policy_rules().",
                                      "Eight rules (R1 to R8) and one exception (E1)."),
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
                 check=_numeric_check("erp_gt5000_any",
                                      "(erp['amount'] > 5000).sum() counts every row, whatever its currency.",
                                      "All rows with amount above 5000, in any currency.",
                                      {"erp_gt5000_eur_only": "That is the number of EUR rows only. This task ignores currency: "
                                                              "include the USD rows too (T3.3 deals with currency).",
                                       "erp_high": "That is the euro-converted count, which is T3.3. Here, compare the "
                                                   "amount column with 5000 as it stands.",
                                       "erp_ge5000_any": "That count includes a payment of exactly 5000.00. "
                                                         "'Greater than' means >, not >=."}),
                 question="How many ERP rows have an amount greater than 5000, ignoring currency?"),
    "T3.3": dict(title="ERP payments above €5,000 in euro", needs_value=True,
                 check=_numeric_check("erp_high",
                                      "convert with the R7 rate, then (erp['amount_eur'] > 5000).sum().",
                                      "USD amounts are converted at 0.92 EUR per USD (rule R7), then payments strictly "
                                      "above 5000 are counted. The USD 5,300 payment becomes €4,876 and drops out; "
                                      "the payment of exactly €5,000.00 is not above €5,000.",
                                      {"erp_gt5000_any": "That is the same as T3.2. Either the USD amounts were not converted "
                                                         "(print erp[erp['currency'] == 'USD'] before and after), or you used >= "
                                                         "and included the payment of exactly 5,000.00.",
                                       "erp_gt5000_eur_only": "That counts EUR rows only. Convert the USD rows instead of "
                                                              "leaving them out: two of them are above €5,000 in euro.",
                                       "erp_ge5000_any": "That count uses >= on the unconverted amounts: it includes the "
                                                         "payment of exactly 5,000.00 and the USD amounts as if they were euro."}),
                 question="How many ERP payments are above €5,000 once USD amounts are converted at the R7 rate?"),
    "T3.4": dict(title="Boundary and currency cases", needs_reasoning=True, check=_saved,
                 question="Find the payment of exactly 5,000.00 and the USD payment of 5,300.00. Should either "
                          "be in the review? Justify each with the policy's wording."),
    "T4.1": dict(title="Bank feed top-level keys", needs_value=True, check=_check_keys,
                 question="Load the bank feed with json.load and record its top-level keys."),
    "T4.2": dict(title="Bank transactions", needs_value=True,
                 check=_numeric_check("bank_n", "len(bank_doc['transactions']) counts the items in the transactions list.",
                                      "The 'transactions' list holds 429 items, three more than the ERP's 426 payments.",
                                      {"erp_n": "That is the number of rows in the ERP file (T3.1). Count the items in "
                                                "the bank feed's 'transactions' list instead.",
                                       "bank_high": "That is the number above €5,000 (T4.4). This task asks for all transactions."}),
                 question="How many transactions does the bank feed contain?"),
    "T4.3": dict(title="Flattened bank columns", needs_value=True, check=_check_cols,
                 question="Flatten the transactions with pd.json_normalize and record the column names."),
    "T4.4": dict(title="Bank payments above €5,000", needs_value=True, needs_reasoning=True,
                 check=_numeric_check("bank_high",
                                      "convert amount.value with the R7 rate where amount.currency is USD, then count > 5000.",
                                      "Convert USD at 0.92 and count amount_eur > 5000. The bank shows three more than "
                                      "the ERP: two payments appear twice and one payment has no ERP record (T4.5).",
                                      {"bank_no_conversion": "That count is what you get without converting USD, or with >= "
                                                             "instead of >. Check both.",
                                       "bank_eur_only": "That counts EUR transactions only. Convert the USD ones instead of "
                                                        "leaving them out.",
                                       "erp_high": "That is the ERP count from T3.3. Count the rows of the bank DataFrame instead."}),
                 question="How many bank transactions are above €5,000 in euro? Why might this differ from your ERP count?"),
    "T4.5": dict(title="Duplicated bank entries", needs_value=True, check=_check_dupes,
                 question="Which ERP references appear more than once in the bank feed?"),
    "T5.1": dict(title="Number of approval requests", needs_value=True,
                 check=_numeric_check("requests", "len(root.findall('Request')) counts the <Request> children of the root.",
                                      "The export holds 37 <Request> elements: two high-value payments have no request at all."),
                 question="How many <Request> elements does the PWS export contain?"),
    "T5.2": dict(title="Requests as a DataFrame", needs_value=True, check=_check_t52,
                 question="Build a DataFrame with one row per request: erp_ref, amount (numeric), currency, "
                          "n_approvers and the roles that approved."),
    "T5.3": dict(title="Requests with fewer than two approvers", needs_value=True,
                 check=_numeric_check("req_lt2", "pws[pws['n_approvers'] < 2] keeps the requests with a single approver.",
                                      "Three requests have only the Budget Holder's approval.",
                                      {"req_le2": "That count uses <= 2. 'Fewer than two' means < 2."}),
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
    "T7.1": dict(title="Transactions not compliant", needs_value=True,
                 check=_numeric_check("noncompliant", "(~findings['compliant']).sum() counts the rows where compliant is False.",
                                      "11 of the 39 payments break at least one rule.",
                                      {"compliant": "That is the number of compliant payments. Count those where compliant is False."}),
                 question="Run kit.pipeline.compliance_check on your saved table. How many transactions are not compliant?"),
    "T7.2": dict(title="Split purchases and emails", needs_value=True, needs_reasoning=True,
                 check=_numeric_check("split_payments", "splits['n_payments'].sum() adds up the payments in all groups.",
                                      "Two groups: three Kestrel Print payments and two Harbour IT payments.",
                                      {"split_groups": "That is the number of groups. Count the payments in them."}),
                 question="How many payments are involved in possible split purchases? What would a review "
                          "that only looks at transactions above €5,000 miss, and what do the emails add?"),
    "T8.1": dict(title="Board report against your count", needs_value=True, needs_reasoning=True, check=_check_t81,
                 question="Record the number of large payments stated in the board report and your own count. "
                          "Explain the difference."),
    "T8.2": dict(title="Your findings and the report", needs_value=True, needs_reasoning=True, check=_saved,
                 question="Write your findings (3 to 6 sentences) and generate the report. Which format would "
                          "you send to the Board, and why not CSV or JSON?"),
}
