"""Topics: what is learnt, each in its own package, independent of any course.

A notebook starts the topic it is about with kit.start("<topic>"), for example
kit.start("knowledge-graphs"), and gets the topic's helpers with T = kit.tools().

Each topic package defines (see docs/AUTHORING_A_TOPIC.md):
    TOPIC     the name used in kit.start, lower case with hyphens ("knowledge-graphs")
    TITLE     the title shown to learners
    LEVEL     the level the topic's material is written for by default (see core.levels)
    TOOLS     the helper module returned by kit.tools() (".tools")
    DATASETS  external datasets the topic fetches when it starts
    EXTRAS    the pip extra the topic needs ("", "graphs" or "advanced")
    STAGES    what is in focus, what the kit does for the learner, and what is out of scope
and optionally BUILDS_ON (topics whose ideas are assumed), PROBLEM and TASKS.

Every concept a topic does not teach is a helper: one call that does it, with an optional
look inside (kit.source(T.<name>)). A learner works on the topic's focus without having to go
into anything else. Released helpers never change; a change is a new function.
"""
from importlib import import_module

REGISTRY = {
    "data-formats": ".data_formats",
    "web-data": ".web_data",
    "data-cleaning": ".data_cleaning",
    "exploring-data": ".exploring_data",
    "modelling": ".modelling",
    "text-to-graphs": ".text_to_graphs",
    "paths-and-connectivity": ".paths_and_connectivity",
    "centrality-and-link-analysis": ".centrality_and_link_analysis",
    "graph-embeddings": ".graph_embeddings",
    "graph-neural-networks": ".graph_neural_networks",
    "knowledge-graphs": ".knowledge_graphs",
    "retrieval-augmented-generation": ".retrieval_augmented_generation",
}


def normalise_topic(name: str) -> str:
    return str(name).strip().lower().replace("_", "-").replace(" ", "-")


def names() -> list:
    return list(REGISTRY)


def is_topic(name) -> bool:
    return isinstance(name, str) and normalise_topic(name) in REGISTRY


def topic(name: str):
    """The topic package registered under a name, e.g. topic("knowledge-graphs")."""
    key = normalise_topic(name)
    if key not in REGISTRY:
        raise KeyError(f"Unknown topic {name!r}. This version of the kit has: {', '.join(REGISTRY)}")
    return import_module(REGISTRY[key], package=__name__)
