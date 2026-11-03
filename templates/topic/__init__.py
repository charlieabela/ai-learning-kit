"""<Title>: <one sentence on what the topic covers>.

<What the notebooks do, what the kit supplies (helpers, data, reference results), and which steps are
support rather than focus. Copy this folder to ai_learning_kit/topics/<name_with_underscores>/ and
register the topic in ai_learning_kit/topics/__init__.py. See docs/AUTHORING_A_TOPIC.md.>
"""
from ...core.session import BLACK_BOX, FOCUS, ON_REQUEST, OUT_OF_SCOPE, stage

TOPIC = "<name>"                    # lower-case words joined by hyphens, as given to kit.start
TITLE = "<Title>"
LEVEL = "intermediate"              # introductory, intermediate, advanced or research
BUILDS_ON = ()                      # topics whose ideas are assumed
PROBLEM = None                      # or an id from the problem library
TASKS = {}                          # graded tasks; {} when the notebooks check themselves
TOOLS = ".tools"
DATASETS = ()                       # external datasets fetched by kit.start
EXTRAS = ""                         # "", "graphs" or "advanced"

STAGES = {
    "idea": stage("<what learners do themselves>", FOCUS),
    "support": stage("<a step the kit does in one call>", BLACK_BOX, how="T.<helper>()",
                     taught_in="the <other> topic"),
    "inside": stage("<a support step learners may look inside>", ON_REQUEST, how="T.<helper>()"),
    "later": stage("<something not brought up>", OUT_OF_SCOPE, taught_in="the further reading of this topic"),
}
