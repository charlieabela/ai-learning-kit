# ari1101kit

A teaching sandbox for **ARI1101 Foundations of Data Science** at the University of Malta.

The unit follows one running case through the OSEMN lifecycle: the **Q3 2026 expenditure compliance review** at Brightwater Logistics Ltd, a fictional company. The question is how many payments above €5,000 were made in the quarter, and whether they followed the company's Expenditure Approval Policy.

Answering that question involves extraction, conversion, cleaning, de-duplication, exploration and reporting, which is too much for the first weeks of the unit. The kit hides the steps that are not the topic of the current session, so that students can concentrate on the step that is. Each session hands a different step back to the students.

All companies, people, accounts and documents in the case are fictional.

## Use in Google Colab

```python
%pip install -q "git+https://github.com/charlieabela/ari1101kit@v0.2.2"
import ari1101kit as kit

kit.start("S02", student="Your Name")   # in Colab, connects Google Drive for the work folder
kit.workspace()                          # copies the case files into <work folder>/brightwater_q3
kit.task("T1.1")                         # explains a task: what to do, why, an example, what to record
```

**Where work is kept.** `kit.start` chooses a work folder and makes it the current folder. In Colab it connects the student's Google Drive and uses `My Drive/ARI1101/<session>`, so the answer book and every file saved with a plain name survive a disconnect. On a student's own computer it uses the folder the notebook runs in. `kit.start(..., folder="...")` chooses another folder; `drive=False` skips Drive in Colab.

Always install a tagged version (`@v0.2.2`), so that a notebook keeps working when later sessions are added.

## The case files

| Source | File | Format |
|---|---|---|
| Expenditure Approval Policy v3.1 | `expenditure_policy_v3_1.docx` | Word |
| ERP payments, Q3 | `erp_payments_q3.csv` | CSV |
| Bank statement feed, Q3 | `bank_feed_q3.json` | JSON |
| Approval workflow (PWS) export | `approvals_pws_q3.xml` | XML |
| Approved Vendor List | `approved_vendors.xlsx` | Excel |
| Finance emails | `finance_emails_q3.txt` | plain text |
| Q3 Finance Summary for the Board | `q3_finance_summary.pdf` | PDF |

The files are frozen: every version of the kit ships the same data, so results discussed in one session remain valid in the next.

## What the kit provides

| Module | Purpose |
|---|---|
| `kit.sources()`, `kit.workspace()`, `kit.peek()`, `kit.path()` | list, copy and preview the case files |
| `kit.extract` | text, rules and tables from Word, Excel, PDF and email files |
| `kit.reference` | the kit's own clean versions of the ERP, bank and approval data |
| `kit.pipeline` | the compliance check and the split-purchase search |
| `kit.report` | assemble findings into an HTML and Word report |
| `kit.start()`, `kit.answer()`, `kit.check()`, `kit.progress()`, `kit.submit()` | the answer book (formative) |
| `kit.explain()`, `kit.source()` | describe any kit function, or show its code |

## Sessions and versions

| Version | Session | Students do | The kit does |
|---|---|---|---|
| 0.2.0 | S02 Data formats | read CSV, JSON and XML; store extracted rules and tables as JSON or CSV; combine sources; choose and compare formats | document extraction, cleaning inside its checks, compliance check, report layout |

Each new session adds a module in `ari1101kit/sessions/` and a new tagged release; earlier modules and the data are not changed. See `CHANGELOG.md`.

## Notebooks

`notebooks/S02_Sandbox_Q3_Compliance_Review.ipynb` is the student notebook for Session 02.
