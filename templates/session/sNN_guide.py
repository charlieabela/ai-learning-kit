"""<UNIT> Session NN: plain-language guidance for every task.

Four parts per task. The example uses a different, smaller case than the problem,
so that it shows the idea without giving the answer away.
"""
from __future__ import annotations

EXAMPLE_CASE = "a different, smaller case: <e.g. a school library>"

GUIDE = {
"T1.1": dict(
what="""
<What to do, step by step, naming the variables in the cell and the functions to call.>""",
why="""
<Why this step matters for the problem and in general.>""",
example="""
<The same idea on the small case, with code and its printed result where possible.>""",
record="""
`value`: <what to pass>. <`reasoning`: what to write, if any.>""",
),
"T1.2": dict(what="""...""", why="""...""", example="""...""", record="""..."""),
"T1.3": dict(what="""...""", why="""...""", example="""...""", record="""..."""),
}
