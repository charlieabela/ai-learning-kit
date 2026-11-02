"""Reference loaders used inside the kit.

These functions read the raw case files into clean DataFrames. The kit's own
pipeline uses them so that later steps work even when a session does not ask
students to do this part themselves. In the session where a step is the topic,
students write their own version and can compare it with these.
"""
from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from functools import lru_cache

import pandas as pd

from .case import path
from . import extract


@lru_cache(maxsize=1)
def usd_rate() -> float:
    """Return the EUR-per-USD rate stated in policy rule R7."""
    for rule in extract.policy_rules():
        if "rate_eur_per_usd" in rule:
            return rule["rate_eur_per_usd"]
    raise ValueError("The policy does not state a USD conversion rate.")


def to_eur(amount, currency):
    """Convert an amount (or a Series of amounts) to euro using the policy rate."""
    if isinstance(amount, pd.Series):
        return amount.where(currency.eq("EUR"), (amount * usd_rate()).round(2))
    return amount if currency == "EUR" else round(amount * usd_rate(), 2)


def erp() -> pd.DataFrame:
    """ERP payments with a proper date type and an amount_eur column."""
    df = pd.read_csv(path("erp"), parse_dates=["posting_date"])
    df["amount_eur"] = to_eur(df["amount"], df["currency"])
    return df


def bank() -> pd.DataFrame:
    """Bank feed flattened to one row per transaction, with the ERP reference extracted."""
    with open(path("bank"), encoding="utf-8") as f:
        doc = json.load(f)
    df = pd.json_normalize(doc["transactions"])
    df["value_date"] = pd.to_datetime(df["value_date"])
    df["amount_eur"] = to_eur(df["amount.value"], df["amount.currency"])
    df["erp_ref"] = df["remittance_info"].str.extract(r"(TX-\d{4}-\d{5})", expand=False)
    return df


def approvals() -> pd.DataFrame:
    """PWS approvals, one row per approver decision (request fields repeated)."""
    root = ET.parse(path("approvals")).getroot()
    rows = []
    for req in root.findall("Request"):
        base = {"request_id": req.get("id"), "erp_ref": req.get("erp_ref"),
                "amount": float(req.get("amount")), "currency": req.get("currency"),
                "submitted": req.get("submitted")}
        for appr in req.findall("Approvers/Approver"):
            rows.append({**base, "role": appr.get("role"), "approver": appr.text,
                         "decision": appr.get("decision"), "decision_date": appr.get("date")})
    df = pd.DataFrame(rows)
    df["submitted"] = pd.to_datetime(df["submitted"])
    df["decision_date"] = pd.to_datetime(df["decision_date"])
    return df


def requests() -> pd.DataFrame:
    """PWS requests, one row per request, with the number of approvers recorded."""
    a = approvals()
    out = (a.groupby(["request_id", "erp_ref", "amount", "currency"], as_index=False)
             .agg(n_approvers=("role", "count"), roles=("role", lambda s: "; ".join(s))))
    return out


def high_value(threshold: float = 5000.0) -> pd.DataFrame:
    """ERP payments whose euro equivalent is above the threshold (default €5,000)."""
    df = erp()
    return df[df["amount_eur"] > threshold].reset_index(drop=True)
