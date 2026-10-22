"""Session 02: plain-language guidance for every task.

Each entry explains what to do, why it matters in the case, a typical example
on a different, small case (a school library), and what to record. The same text
is shown by kit.task("T1.1") and in the task cells of the S02 notebook.
"""
from __future__ import annotations

GUIDE = {
# ----------------------------------------------------------------------------------------- 1
"T1.1": dict(
what="""
Use the output of the previous cell (`kit.peek`) and the table from `kit.sources()`. For each of the seven sources, write down two things:

1. its **format**: the kind of file. The file extension tells you (`.csv`, `.json`, `.xml`, `.docx`, `.xlsx`, `.txt`, `.pdf`), and the first characters confirm it.
2. its **shape**: how the information inside is organised, using one of the four words above (`table`, `nested`, `tree`, `document`).

Replace each `___` with a word in quotes, for example `"json"`.""",
why="""
The format decides which Python tool can open the file. The shape decides how much work is needed before you can count or filter: a table is ready to use, a nested structure has to be flattened, a tree has to be walked element by element, and a document has to be searched for the pieces you need.""",
example="""
A school library keeps three files:

| file | first characters | format | shape | because |
|---|---|---|---|---|
| `loans.csv` | `member_id,book_id,date_out` then `M017,B220,2026-03-02` | `csv` | `table` | a header line, then one line per loan with the same fields |
| `catalogue.json` | `{"books": [{"id": 1, "title": "Dune", "authors": ["Herbert"]}` | `json` | `nested` | a list of books inside an object, and a list of authors inside each book |
| `rules.pdf` | `%PDF-1.7` | `pdf` | `document` | paragraphs written for people to read |

Word (`.docx`) and Excel (`.xlsx`) files both start with the bytes `PK`, because both are ZIP archives. The first bytes cannot tell them apart, so use the file name in `kit.sources()`.""",
record="""
`value`: the completed `survey` dictionary. No reasoning is needed.""",
),
"T1.2": dict(
what="""
Two short questions, answered in your own words.

**(a) Which sources can Python read directly, and which need extraction?** Sort the seven sources into two groups:

- *read directly*: the file is already organised data, so one standard call loads it: `pd.read_csv` for CSV, `json.load` for JSON, `ET.parse` (ElementTree) for XML;
- *needs extraction*: the information sits inside a document (Word, PDF, the text of an email), so someone has to find the right words or numbers and turn them into data. In this session the kit does that for you.

The Excel vendor list is a borderline case: `pd.read_excel` can open it, but there are title lines above the table, so you must tell pandas where the header is. Put it in either group, as long as you say why.

**(b) Why can the review question not be answered from any single source?** The question has two parts: *how many payments were above €5,000* and *were they approved as the policy requires*. Say which source gives which part of the answer.""",
why="""
Real data science questions rarely come with one tidy file. Knowing which sources are ready to use and which need work is how you estimate the effort, and knowing which source holds which fact is how you plan to combine them.""",
example="""
Library audit question: *which members borrowed more books than their limit?*

> `loans.csv` and `members.json` can be read directly, with `pd.read_csv` and `json.load`, because they are already structured. `rules.pdf` needs extraction: the limits are written in a sentence ("students may borrow 5 books"), not stored as data. No single file answers the question: the loans say who borrowed how many books, `members.json` says whether each member is a student or a teacher, and only the rules say what each type may borrow. The three have to be combined.""",
record="""
`reasoning`: three to five sentences covering (a) and (b). No value is needed.""",
),
# ----------------------------------------------------------------------------------------- 2
"T2.1": dict(
what="""
`kit.extract.policy_rules()` read the Word policy and returned a Python object, stored in the variable `rules`. Find out:

- how many items it holds: `len(rules)`;
- what kind of object it is, and what kind each item is: `type(rules)` and `type(rules[0])`.

The cell above already printed these; read the output.""",
why="""
Before you use any data in Python you need to know its type, because the type decides what you can do with it: you loop over a list, look up a key in a dictionary, and filter a DataFrame.""",
example="""
```python
shelves = [{"id": "S1", "topic": "History"},
           {"id": "S2", "topic": "Maths"}]
len(shelves)        # 2
type(shelves)       # <class 'list'>
type(shelves[0])    # <class 'dict'>
```
So: *shelves is a list of dictionaries, one per shelf*.""",
record="""
`value`: the number of items. `reasoning`: complete the sentence "rules is a ___ of ___".""",
),
"T2.2": dict(
what="""
Save `rules` to a JSON file and read it back:

1. `with open("policy_rules.json", "w", encoding="utf-8") as f:` opens a file for **w**riting and calls it `f`;
2. `json.dump(rules, f, ...)` writes the object into the file;
3. a second `with open(...)` (without `"w"`) opens it for reading, and `json.load(f)` rebuilds the object;
4. `rules_reloaded == rules` compares the two and gives `True` or `False`.""",
why="""
Data you extract once should be stored so that it can be reused without extracting it again. A good storage format gives you back exactly what you saved.""",
example="""
```python
shelf = {"id": "S1", "books": ["Dune", "Emma"]}

with open("shelf.json", "w", encoding="utf-8") as f:
    json.dump(shelf, f)

with open("shelf.json", encoding="utf-8") as f:
    shelf_again = json.load(f)

shelf_again == shelf      # True: the list inside came back as a list
```""",
record="""
`value`: the result of the comparison, `same`.""",
),
"T2.3": dict(
what="""
Now store the same rules as CSV. The cell does the steps for you: it turns `rules` into a DataFrame, writes it to `policy_rules.csv`, reads it back into `rules_csv`, and prints the `approvers` value of rule R2 (row 1) together with its type.

Look closely at that output and answer: what happened to the nested fields (`approvers`, `applies_to_eur`)? Which format would you choose to store the rules, and why?""",
why="""
CSV is the most common data format, but it can only hold a grid of text. Knowing what it loses is how you decide when *not* to use it.""",
example="""
```python
df = pd.DataFrame([{"id": "S1", "books": ["Dune", "Emma"]}])
df.to_csv("shelf.csv", index=False)
back = pd.read_csv("shelf.csv")

back.loc[0, "books"]          # "['Dune', 'Emma']"   it looks like a list...
type(back.loc[0, "books"])    # <class 'str'>        ...but it is text
back.loc[0, "books"][0]       # "["                  the first character, not "Dune"
```
A CSV cell holds only text, so the list was written as its printed form and came back as a string.""",
record="""
`value`: the reloaded `approvers` value (the cell records it). `reasoning`: what happened to the nested fields, and which format you would use for the rules.""",
),
"T2.4": dict(
what="""
The policy's table of approval bands comes back exactly as written in Word, so every value is text, including the numbers in `Approvers required`. Convert that column to numbers with `pd.to_numeric` and assign the result back to the same column. Print `thresholds.dtypes` before and after: `object` means text, `int64` means whole numbers.""",
why="""
Numbers stored as text look right on screen but cannot be compared or added: `"3" > "10"` is `True` for text. Every time data crosses a format boundary, check the types.""",
example="""
```python
t = pd.DataFrame({"band": ["A", "B"], "max_books": ["3", "5"]})
t.dtypes                       # max_books    object   (text)
t["max_books"] = pd.to_numeric(t["max_books"])
t.dtypes                       # max_books    int64
t["max_books"] + 1             # 4, 6
```""",
record="""
`value`: the `thresholds` DataFrame after the conversion.""",
),
# ----------------------------------------------------------------------------------------- 3
"T3.1": dict(
what="""
Read the ERP file into a DataFrame called `erp` with `pd.read_csv`. The file is in the case folder: `folder / "erp_payments_q3.csv"` joins the folder and the file name into a path. Then look at the first rows (`erp.head()`), the column types (`erp.dtypes`) and the size (`erp.shape`).""",
why="""
The first look at a new table tells you how many records there are, what each column holds and whether pandas guessed the types correctly. Every later step builds on it.""",
example="""
```python
loans = pd.read_csv(folder / "loans.csv")
loans.head()      # the first 5 rows
loans.shape       # (250, 3): 250 rows (loans), 3 columns
```
`shape` has no brackets after it: it is a value stored with the DataFrame, not a function you call.""",
record="""
`value`: `erp.shape`, recorded straight after reading, before you add any column.""",
),
"T3.2": dict(
what="""
Count the rows where `amount` is greater than 5000. A **boolean filter** does this in two steps:

1. `erp["amount"] > 5000` compares every amount with 5000 and gives a column of `True`/`False`;
2. `.sum()` adds them up, counting each `True` as 1.

Ignore the currency for now; that is the next task.""",
why="""
Counting the rows that meet a condition is the most common question asked of any table. This first count is also a baseline: the next tasks show how the answer changes when the definition is made precise.""",
example="""
```python
books = pd.DataFrame({"title": ["Dune", "Emma", "Ulysses"],
                      "price": [12.0, 30.0, 8.5]})
books["price"] > 10              # True, True, False
(books["price"] > 10).sum()      # 2
```""",
record="""
`value`: the count, a whole number.""",
),
"T3.3": dict(
what="""
Some ERP payments are in US dollars, and rule R7 says the €5,000 threshold applies to the **euro equivalent**. Three steps:

1. **Find the rate in `rules`**, rather than typing it: the R7 dictionary has a key `rate_eur_per_usd`. One way:
   `rate = [r for r in rules if r["rule_id"] == "R7"][0]["rate_eur_per_usd"]`
   (make a list of the rules whose id is R7, take the first, read its rate).
2. **Add a column `amount_eur`**: keep the amount where the currency is EUR, and multiply by the rate otherwise. `.where(condition, other)` keeps a value where the condition is `True` and uses `other` where it is `False`.
3. **Filter** the rows where `amount_eur` is **above** 5000 and keep them in `high`.""",
why="""
The policy defines the population to review. Counting with the wrong currency or the wrong comparison (`>=` instead of `>`) gives a different set of payments, and every later step inherits the mistake.""",
example="""
```python
fines = pd.DataFrame({"amount": [4.0, 10.0, 8.0],
                      "currency": ["EUR", "GBP", "EUR"]})
rate = 1.17                                   # EUR per GBP
fines["amount_eur"] = fines["amount"].where(fines["currency"] == "EUR",
                                            fines["amount"] * rate)
fines["amount_eur"]                           # 4.0, 11.7, 8.0
over = fines[fines["amount_eur"] > 5]         # the rows above €5
len(over)                                     # 2
```""",
record="""
`value`: `len(high)`, the number of payments above €5,000 in euro.""",
),
"T3.4": dict(
what="""
The cell shows two payments that sit on the edge of the definition: one of exactly 5,000.00 euro, and one of 5,300.00 US dollars. For each, decide whether it belongs in the review, and justify your decision by quoting the words of the policy (rules R1 and R7).""",
why="""
Edge cases are where definitions matter. A reviewer who cannot say why a borderline payment is in or out cannot defend the count.""",
example="""
Library rule: *"members may borrow **more than** 5 books only with the librarian's permission."*

> A member with exactly 5 books needs no permission, because "more than 5" means 6 or more. A member with 6 books does.""",
record="""
`reasoning`: one or two sentences per payment, each quoting the policy.""",
),
# ----------------------------------------------------------------------------------------- 4
"T4.1": dict(
what="""
Load the bank feed with `json.load(f)` and store it in `bank_doc`. JSON objects become Python dictionaries, so `bank_doc.keys()` lists the names at the top level of the document.""",
why="""
A JSON document from an API usually wraps the data you want inside a few layers of information about the data (who sent it, for which period). Looking at the top level first tells you where the records are.""",
example="""
```python
with open("catalogue.json", encoding="utf-8") as f:
    catalogue = json.load(f)
type(catalogue)              # <class 'dict'>
list(catalogue.keys())       # ['library', 'updated', 'books']
```
The books are one level down, in `catalogue["books"]`.""",
record="""
`value`: `list(bank_doc.keys())`, the top-level keys of the whole document.""",
),
"T4.2": dict(
what="""
The payments are inside the document, under the key `"transactions"`. Take that list out into a variable called `transactions`, count its items with `len`, and print the first item to see how it is nested.""",
why="""
Counting the records in each source is the first check that two systems agree. Here the bank and the ERP should describe the same quarter.""",
example="""
```python
books = catalogue["books"]   # a list of dictionaries
len(books)                   # 312
books[0]                     # {'id': 1, 'title': 'Dune', 'price': {'value': 12.5, 'currency': 'EUR'}}
```""",
record="""
`value`: `len(transactions)`.""",
),
"T4.3": dict(
what="""
Each transaction holds two objects inside it, `amount` and `counterparty`. `pd.json_normalize(transactions)` turns the list into a DataFrame and **flattens** the inner objects: their keys become separate columns with dotted names such as `amount.value`.""",
why="""
pandas can filter and count only columns that hold single values. A column of dictionaries has to be flattened before you can ask "which amounts are above 5000?".""",
example="""
```python
books = [{"id": 1, "title": "Dune", "price": {"value": 12.5, "currency": "EUR"}}]
pd.DataFrame(books).columns         # id, title, price          price still holds a dictionary
pd.json_normalize(books).columns    # id, title, price.value, price.currency
```
Dotted columns are read with brackets: `df["price.value"]`, not `df.price.value`.""",
record="""
`value`: `list(bank.columns)`.""",
),
"T4.4": dict(
what="""
Repeat the euro conversion from T3.3 for the bank: use the columns `amount.value` and `amount.currency`, and the same `rate`. Count the transactions above €5,000.

Your count will not match T3.3. Before doing T4.5, suggest in your reasoning what could make a bank statement show more large payments than the finance system.""",
why="""
Comparing two independent records of the same events (what finance recorded, and what the bank actually paid) is one of the strongest checks in any audit. A difference is a lead to follow, not an error in your code.""",
example="""
The library's till shows 52 fines paid this month, but the fines register lists 50. Possible reasons: a fine was paid twice and both payments went through the till, or a fine was paid but never entered in the register.""",
record="""
`value`: the count. `reasoning`: one or two possible reasons for the difference.""",
),
"T4.5": dict(
what="""
Each bank line has a `remittance_info` text that starts with the ERP reference, for example `TX-2026-01002 Compressor repair`. Two steps:

1. **Extract** the reference into its own column with a regular expression. In `r"(TX-\\d{4}-\\d{5})"`, `TX-` is matched literally, `\\d{4}` means four digits, `\\d{5}` means five digits, and the brackets say "this is the part to keep". Lines without a reference get `NaN` (missing).
2. **Find duplicates**: `.duplicated(keep=False)` marks every row whose reference appears more than once. The cell then lists the references.""",
why="""
A shared key is what lets you link two sources. Once the bank lines carry the ERP reference, you can see payments that appear twice and payments that finance never recorded.""",
example="""
```python
notes = pd.Series(["LN-0001 late return", "LN-0002", "lost card", "LN-0001 reminder"])
refs = notes.str.extract(r"(LN-\\d{4})", expand=False)
refs                              # LN-0001, LN-0002, NaN, LN-0001
refs.duplicated(keep=False)       # True, False, False, True
```""",
record="""
`value`: `duplicated_refs`, the sorted list of references that appear more than once.""",
),
# ----------------------------------------------------------------------------------------- 5
"T5.1": dict(
what="""
`ET.parse(...).getroot()` reads the XML file and gives you its outermost element, the **root**. `root.findall("Request")` returns a list of the root's direct children whose tag is `Request`. Store it in `requests` and count it. Tags are case-sensitive: `request` would find nothing.""",
why="""
In XML, each record is an element. Finding the right elements is the XML equivalent of reading the rows of a table.""",
example="""
```python
# library.xml:
# <library>
#   <book id="1" price="12.50"><title>Dune</title></book>
#   <book id="2" price="9.99"><title>Emma</title></book>
# </library>
root = ET.parse("library.xml").getroot()
root.tag                     # 'library'
books = root.findall("book")
len(books)                   # 2
```""",
record="""
`value`: `len(requests)`.""",
),
"T5.2": dict(
what="""
Turn the XML into a table, one row per request. XML stores values in two places:

- **attributes**, inside the opening tag: `<Request erp_ref="TX-2026-01003" amount="21572.59" ...>`. Read them with `req.get("amount")`.
- **element text**, between the tags: `<Approver ...>Lara Grech</Approver>`. Read it with `.text`.

The loop builds one dictionary per request and adds it to `rows`; `pd.DataFrame(rows)` makes the table. Fill in `amount` and `currency` (attributes) and `n_approvers` (how many `<Approver>` elements: `len(approvers)`). Finally convert `amount` to a number: everything read from XML is text.""",
why="""
Building a table from a tree, one record at a time, is the general way to get any XML (or nested JSON) into pandas when a ready-made function does not fit.""",
example="""
```python
rows = []
for b in books:                                    # the <book> elements from T5.1's example
    rows.append({"id": b.get("id"),                # attribute
                 "price": b.get("price"),          # attribute, text "12.50"
                 "title": b.find("title").text})   # element text
df = pd.DataFrame(rows)
df["price"] = pd.to_numeric(df["price"])           # now a number
```""",
record="""
`value`: the `pws` DataFrame.""",
),
"T5.3": dict(
what="""
Use a boolean filter, as in T3.2, to keep the requests in `pws` with **fewer than two** approvers, and store them in `too_few`. "Fewer than two" means `< 2`.""",
why="""
Rule R2 requires two approvers above €5,000. A request with only one is a likely breach, worth flagging before the full check.""",
example="""
```python
books[books["n_authors"] < 2]      # books with a single author
```""",
record="""
`value`: `len(too_few)`.""",
),
"T5.4": dict(
what="""
Look back at the XML and at your loop in T5.2, and at the printout of one `<Approver>` element. Answer three questions:

1. Which values did you read from **attributes**? (In which tag?)
2. Which values are **element text**?
3. Which values needed a **type conversion**, and why?""",
why="""
The same information can be put in an attribute or in an element. You have to know which, to write the right code, and you have to remember that XML does not record types.""",
example="""
For the library XML: `id` and `price` are attributes of `<book>`, and the title is the element text of `<title>`. `price` had to be converted with `pd.to_numeric` because XML stores everything as text, so `"12.50"` arrives as a string.""",
record="""
`reasoning`: two to four sentences answering the three questions.""",
),
# ----------------------------------------------------------------------------------------- 6
"T6.1": dict(
what="""
Combine two tables into one: `high` (ERP payments above €5,000) and `pws` (approval requests). `merge` matches rows that share the same value in a key column, here `erp_ref`.

The `how` argument decides what happens to payments that have **no** matching request:

- `how="inner"` (the default) drops them;
- `how="left"` keeps every row of the left table (`high`) and leaves the request columns empty (`NaN`).""",
why="""
Payments without any approval request are exactly the ones the review must find. An inner merge would remove them silently.""",
example="""
```python
loans = pd.DataFrame({"book_id": [1, 2, 3]})
books = pd.DataFrame({"book_id": [1, 2], "title": ["Dune", "Emma"]})
loans.merge(books, on="book_id")                # 2 rows: the loan of book 3 is gone
loans.merge(books, on="book_id", how="left")    # 3 rows: book 3 has title NaN
```
Book 3 is the interesting one: a loan of a book that is not in the catalogue.""",
record="""
`value`: the `review` DataFrame.""",
),
"T6.2": dict(
what="""
The cell saves `review` twice, as `q3_high_value.csv` and as `q3_high_value.json`, compares the file sizes, reads both back and shows the column types side by side. Study the output and answer:

1. Which file is bigger, and why? (Open the JSON with `print(open("q3_high_value.json").read()[:500])` to see.)
2. Which format would you send to the **finance team**, who work in Excel?
3. What does each format lose or keep when the file is read back? Look at the dtypes table.""",
why="""
Choosing a format is a decision about the reader: who will open the file, with what tool, and what they need to keep.""",
example="""
The library's loans table (250 rows) is 9 KB as CSV and 25 KB as JSON, because JSON repeats the column names in every record. In both, the dates come back as text unless you convert them. For the librarian, who uses Excel, CSV is the natural choice; for the library's website, which reads data through a program, JSON is.""",
record="""
`value`: `sizes` (the cell records it). `reasoning`: your answers to the three questions.""",
),
"T6.3": dict(
what="""
Optional. Write `review` as XML with `review.to_xml(...)` and print the beginning of the file. Compare its structure with the PWS export you read in Part 5: where are the values (attributes or element text)? Is anything nested?""",
why="""
Converting a table to a tree format does not create structure that was not there. Seeing that helps you choose formats deliberately.""",
example="""
The library loans table written as XML becomes `<data><row><member_id>M017</member_id><book_id>B220</book_id>...</row>...</data>`: one `<row>` per row and one element per column, with no attributes and no nesting.""",
record="""
`reasoning`: two or three sentences. This task is optional.""",
),
# ----------------------------------------------------------------------------------------- 7
"T7.1": dict(
what="""
`findings` has one row per payment you selected, with a column `compliant` that is `True` or `False`. Count the payments where it is `False`. `~` reverses `True` and `False`, so `(~findings["compliant"]).sum()` counts the `False` values.""",
why="""
This number is half of the answer to the review question.""",
example="""
```python
checks = pd.DataFrame({"member": ["M1", "M2", "M3"], "ok": [True, False, False]})
(~checks["ok"]).sum()          # 2
(checks["ok"] == False).sum()  # 2, the same count written another way
```""",
record="""
`value`: `n_not_compliant`.""",
),
"T7.2": dict(
what="""
Rule R4 says that payments to the **same supplier within 7 days** count as a single expenditure. `kit.pipeline.split_purchases()` searches all payments, including those below €5,000, for groups like this. Each row of `splits` is one group, and the column `n_payments` says how many payments it contains.

1. Record how many **payments** are involved in total (add up `n_payments`).
2. Read the Marketing email (`print(kit.extract.emails()[0]["body"])`), then explain: what would a review that only looks at payments above €5,000 miss? What do the emails add that the tables cannot?""",
why="""
A question worded as "payments above €5,000" invites a filter on single rows. Splitting a purchase into smaller payments is the classic way around such a rule, and only combining rows, or reading what people wrote, reveals it.""",
example="""
The library limit is 5 books. A member borrows 3 books on Monday and 3 on Tuesday: each visit is within the limit, but together they exceed it. An email from the member saying "can you check them out in two goes so it doesn't flag?" shows it was deliberate.""",
record="""
`value`: `n_split_payments`, a number of payments (not of groups). `reasoning`: what the narrow question misses, and what the emails add.""",
),
# ----------------------------------------------------------------------------------------- 8
"T8.1": dict(
what="""
The kit extracted the Board report's claim: how many large payments it says were made, and how it defines "large" (`board["stated_threshold"]`). Record that number and your own count from T3.3, then explain the difference. Compare the report's wording with the policy's ("above €5,000"), and think about the USD payments.""",
why="""
A report is a claim about the data. Checking a claim against the data, and explaining exactly why two numbers differ, is a core skill: it is how you find out whether the difference is a mistake or a different definition.""",
example="""
The library's annual report says "40 members borrowed 5 or more books". Your count of members who borrowed *more than* 5 is 37. The 3 members who borrowed exactly 5 explain the difference: the two numbers answer different questions.""",
record="""
`value`: `{"stated": board["stated_high_value_count"], "ours": len(high)}`. `reasoning`: the explanation of the difference.""",
),
"T8.2": dict(
what="""
Write your findings for the Board in 3 to 6 sentences, in plain language. Cover:

1. the answer to the question: how many payments above €5,000, and how many did not comply;
2. the main kinds of breach (look at the `issues` column of `findings`);
3. one limitation of the review: what it could not see.

The kit then lays out the report (HTML and Word) around your text. In the reasoning, say which format you would send to the Board, and why not CSV or JSON.""",
why="""
Analysis is only useful once someone who did not do it can understand and act on it. The format of the final output is chosen for its reader, just like every format before it.""",
example="""
> The library audit found that 12 of 250 members borrowed more than their limit in the spring term; 9 of them were students with over 5 books, and 3 were teachers with over 10. Most cases were in the week before exams. The audit used the loan records only, so books taken without being checked out are not included.""",
record="""
`value`: your findings text (`my_findings`). `reasoning`: the format for the Board, and why.""",
),
}


def render_markdown(task_id: str, title: str) -> str:
    """The guidance for one task as Markdown, as shown in the notebook."""
    g = GUIDE[task_id]
    return "\n\n".join([
        f"### Task {task_id} · {title}",
        "**What you do**\n\n" + g["what"].strip("\n"),
        "**Why it matters**\n\n" + g["why"].strip("\n"),
        "**A typical example** (a different, smaller case: a school library)\n\n" + g["example"].strip("\n"),
        "**What to record**\n\n" + g["record"].strip("\n"),
    ])
