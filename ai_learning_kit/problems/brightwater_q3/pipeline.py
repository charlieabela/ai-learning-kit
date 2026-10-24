"""The compliance check: applies the policy rules to a table of transactions.

This is the part of the work that the Session 02 sandbox does for you. You
provide a table of high-value transactions (a DataFrame, or the path of a CSV
or JSON file you saved) and the kit checks each one against rules R1 to R8 and
exception E1, using the approvals, vendor list and emails.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import extract, reference


def _load_table(table) -> pd.DataFrame:
    if isinstance(table, pd.DataFrame):
        return table.copy()
    p = Path(table)
    if p.suffix.lower() == ".csv":
        return pd.read_csv(p)
    if p.suffix.lower() == ".json":
        return pd.read_json(p)
    raise ValueError("Pass a DataFrame, or the path of a .csv or .json file.")


def _find_ref_column(df: pd.DataFrame) -> str:
    for c in df.columns:
        if str(c).strip().lower() in {"erp_ref", "erp ref", "erpref", "ref", "reference"}:
            return c
    raise KeyError("Your table needs a column with the ERP reference, named 'erp_ref'.")


def _rules() -> dict:
    return {r["rule_id"]: r for r in extract.policy_rules()}


def band(amount_eur: float) -> str:
    """Return the approval band (A, B or C) for a euro amount, using the policy thresholds."""
    rules = _rules()
    if amount_eur > rules["R3"]["applies_to_eur"]["above"]:
        return "C"
    if amount_eur > rules["R2"]["applies_to_eur"]["above"]:
        return "B"
    return "A"


def compliance_check(table, verbose: bool = True) -> pd.DataFrame:
    """Check each transaction in `table` against the expenditure policy.

    `table` must contain an `erp_ref` column; other columns are ignored, because
    the kit looks up amounts, dates and vendors in the ERP itself. The result
    has one row per transaction with its band, the approvals recorded in PWS,
    a `compliant` flag and a plain-language list of `issues`.

    With verbose=True the function also tells you whether your table contains
    the same transactions as the kit's own selection of payments above €5,000.
    """
    student = _load_table(table)
    ref_col = _find_ref_column(student)
    refs = student[ref_col].astype(str).str.strip().unique().tolist()

    rules = _rules()
    erp = reference.erp().set_index("erp_ref")
    hv = set(reference.high_value()["erp_ref"])
    if verbose:
        mine = set(refs)
        print(f"Your table: {len(mine)} transactions. Kit selection above €5,000 (euro equivalent): {len(hv)}.")
        if mine - set(erp.index):
            print(f"  {len(mine - set(erp.index))} reference(s) in your table are not in the ERP.")
        if hv - mine:
            print(f"  {len(hv - mine)} high-value transaction(s) are missing from your table.")
        if (mine & set(erp.index)) - hv:
            print(f"  {len((mine & set(erp.index)) - hv)} transaction(s) in your table are not above €5,000 in euro.")

    approvals = reference.approvals()
    vend = extract.vendors().set_index("vendor_id")
    mails = extract.emails()
    upper_b = rules["R2"]["applies_to_eur"]["up_to"]
    e1_days = rules["E1"]["deadline_days"]

    rows = []
    for ref in refs:
        if ref not in erp.index:
            continue
        tx = erp.loc[ref]
        b = band(tx["amount_eur"])
        needed = (rules["R3"] if tx["amount_eur"] > upper_b else rules["R2"])["approvers"]["roles"] if b != "A" else []
        a = approvals[approvals["erp_ref"] == ref]
        issues = []

        if b != "A" and a.empty:
            mailed = [m for m in mails if ref in m["erp_refs"]]
            if mailed:
                issues.append(f"No PWS request; approved by email only and not recorded in PWS within "
                              f"{e1_days} working days (R1, E1)")
            else:
                issues.append("No approval request in PWS (R1)")
        for role in needed:
            if a.empty:
                break
            r = a[a["role"] == role]
            if r.empty:
                issues.append(f"Missing approval: {role} ({'R3' if b == 'C' else 'R2'})")
                continue
            if (r["decision"] == "rejected").any():
                issues.append(f"Rejected by {role} but paid (R8)")
            if (r["decision_date"] > tx["posting_date"]).any():
                issues.append(f"{role} approval dated after payment (R6)")
        if b != "A":
            vid = tx["vendor_id"]
            if vid not in vend.index:
                issues.append("Supplier not on the Approved Vendor List (R5)")
            elif vend.loc[vid, "status"] != "Active":
                issues.append(f"Supplier status is {vend.loc[vid, 'status']} (R5)")

        recorded = "; ".join(f"{r.role} ({r.decision})" for r in a.itertuples()) if not a.empty else ""
        rows.append({"erp_ref": ref, "posting_date": tx["posting_date"].date(),
                     "vendor_name": tx["vendor_name"], "amount": tx["amount"], "currency": tx["currency"],
                     "amount_eur": tx["amount_eur"], "band": b, "approvals_recorded": recorded,
                     "compliant": not issues, "issues": "; ".join(issues)})
    out = pd.DataFrame(rows).sort_values("posting_date").reset_index(drop=True)
    if verbose:
        print(f"Checked {len(out)} transactions: {int(out['compliant'].sum())} compliant, "
              f"{int((~out['compliant']).sum())} not compliant.")
    return out


def split_purchases() -> pd.DataFrame:
    """Find possible split purchases (rule R4) across all ERP payments.

    Looks for payments of €5,000 or less to the same supplier within the window
    stated in R4 whose combined value is above the R4 limit. Returns one row per
    group, with the payments involved and their combined euro value.
    """
    r4 = _rules()["R4"]
    window, limit = r4["window_days"], r4["applies_to_eur"]["above"]
    df = reference.erp()
    small = df[df["amount_eur"] <= limit]
    groups = []
    for vid, g in small.groupby("vendor_id"):
        g = g.sort_values("posting_date")
        for _, row in g.iterrows():
            w = g[(g["posting_date"] - row["posting_date"]).dt.days.abs() <= window]
            if len(w) > 1 and w["amount_eur"].sum() > limit:
                groups.append(tuple(w["erp_ref"]))
    rows = []
    for i, refs in enumerate(sorted(set(groups)), start=1):
        g = df[df["erp_ref"].isin(refs)]
        rows.append({"group": f"S{i}", "vendor_name": g["vendor_name"].iloc[0],
                     "n_payments": len(g), "erp_refs": ", ".join(refs),
                     "first_date": g["posting_date"].min().date(), "last_date": g["posting_date"].max().date(),
                     "combined_eur": round(g["amount_eur"].sum(), 2),
                     "issue": f"Combined value above €{limit:,.0f} within {window} days without PWS approval (R4, R1)"})
    return pd.DataFrame(rows)
