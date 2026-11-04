# Data formats: gathering the evidence for the Q3 compliance review

`data-formats` · level **introductory** · builds on no other topic · pip extra `none`

Data formats: CSV, JSON, XML and documents. Tasks, questions and formative checks on the Brightwater Q3 case.

*Generated from the kit by `python -m ai_learning_kit.dev.reference write`: do not edit by hand.*

## Starting the topic

```python
import ai_learning_kit as kit
kit.start("data-formats", student="Your Name")
```

Written for the **introductory** level: Learners who are new to programming and to data science, still getting used to Python, NumPy, pandas and Matplotlib.

**Case problem:** `brightwater_q3`, The Q3 2026 Expenditure Compliance Review at Brightwater Logistics Ltd. How many payments above €5,000 were made in Q3 2026, and were they in line with the company's Expenditure Approval Policy?

## The three layers

### In focus: learners do these themselves

- Identifying file formats and data shapes
- Reading CSV, JSON and XML with pandas, json and ElementTree
- Storing extracted data as JSON or CSV and reading it back; comparing formats
- Currency conversion and boolean filters to select payments
- Linking sources by key and merging tables

### Support: the kit does these for the learner

- Extracting rules, tables and text from Word, Excel, PDF and emails (with `kit.extract`; taught in text retrieval and information extraction, beyond this topic)
- Checking each payment against the policy rules (with `kit.pipeline.compliance_check`; taught in the exploring-data topic)
- Finding split purchases across payments (with `kit.pipeline.split_purchases`)
- Laying out the Board report (with `kit.report.compliance_report`)

### Left out of this topic

- Cleaning and de-duplicating records (taught in the data-cleaning topic)
- Statistics, machine learning and prediction (taught in the exploring-data and modelling topics)

## Graded tasks

`kit.task("<id>")` explains a task; `kit.answer(...)` records and checks it.

| Task | Title | Question |
|---|---|---|
| T1.1 | Format and shape of each source | For each of the seven sources, record its file format and the shape of its data (table, nested, tree or document) in the survey dictionary. |
| T1.2 | What can be read directly | (a) Which sources can Python read directly with pandas, json or ElementTree, and which need their information extracted first? (b) Which source gives which part of the answer, and why is no single source enough? |
| T2.1 | Number of policy rules | How many items does kit.extract.policy_rules() return, and what Python type is the result and each item? |
| T2.2 | Rules round trip through JSON | Save the rules to policy_rules.json, read them back, and record whether the reloaded object equals the original. |
| T2.3 | Rules round trip through CSV | Save the rules as CSV and read them back. Record the reloaded 'approvers' value of rule R2. What happened to the nested fields, and which format suits the rules? |
| T2.4 | Thresholds table as a DataFrame | Turn the policy's threshold table into a DataFrame with the number of approvers stored as a number. |
| T3.1 | ERP shape | Read the ERP CSV file and record its shape (rows, columns). |
| T3.2 | ERP rows with amount > 5000 | How many ERP rows have an amount greater than 5000, ignoring currency? |
| T3.3 | ERP payments above €5,000 in euro | How many ERP payments are above €5,000 once USD amounts are converted at the R7 rate? |
| T3.4 | Boundary and currency cases | Should the payment of exactly €5,000.00 and the USD 5,300.00 payment be in the review? Justify each with the policy's wording. |
| T4.1 | Bank feed top-level keys | Load the bank feed with json.load and record its top-level keys. |
| T4.2 | Bank transactions | How many transactions does the bank feed contain? |
| T4.3 | Flattened bank columns | Flatten the transactions with pd.json_normalize and record the column names. |
| T4.4 | Bank payments above €5,000 | How many bank transactions are above €5,000 in euro? Why might this differ from your ERP count? |
| T4.5 | Duplicated bank entries | Which ERP references appear more than once in the bank feed? |
| T5.1 | Number of approval requests | How many <Request> elements does the PWS export contain? |
| T5.2 | Requests as a DataFrame | Build a DataFrame with one row per request: erp_ref, amount (numeric), currency, n_approvers and roles. |
| T5.3 | Requests with fewer than two approvers | How many requests have fewer than two approvers recorded? |
| T5.4 | Attributes and element text | Which values did you read from attributes and which from element text? Which needed a type conversion, and why? |
| T6.1 | Combined high-value table | Combine the ERP payments above €5,000 with the approval requests into one table, one row per payment. |
| T6.2 | Choosing a format | Save the combined table as CSV and as JSON and record both file sizes. Which format would you send to the finance team, and what does each format lose when read back? |
| T6.3 | Extension: convert back to XML | Optional. Write the combined table as XML with df.to_xml(). How does it compare with the PWS export's structure? |
| T7.1 | Transactions not compliant | Run kit.pipeline.compliance_check on your saved table. How many transactions are not compliant? |
| T7.2 | Split purchases and emails | How many payments are involved in possible split purchases? What would a review that only looks at payments above €5,000 miss, and what do the emails add? |
| T8.1 | Board report against your count | Record the number of large payments stated in the Board report and your own count. Explain the difference. |
| T8.2 | Your findings and the report | Write your findings (3 to 6 sentences) and generate the report. Which format would you send to the Board, and why not CSV or JSON? |
