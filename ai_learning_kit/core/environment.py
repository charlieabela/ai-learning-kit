"""Checks a setup notebook runs before a topic starts: are the libraries there, and does a database answer?

Each check prints one line per item, starting with PASS, SKIP or FAIL, and a hint after every FAIL that says
what to do. The functions return the results as a list of (status, item, detail), so a notebook can also test
them. Nothing here prints a password.
"""
from __future__ import annotations

import os
import platform
import re
import sys
from importlib import import_module
from pathlib import Path

PASS, SKIP, FAIL = "PASS", "SKIP", "FAIL"

# The name a learner knows, the module to import, the oldest version that works, and the extra that brings it.
LIBRARIES = {
    "NetworkX": ("networkx", "3.2", "graphs"),
    "scikit-learn": ("sklearn", "1.3", ""),
    "SciPy": ("scipy", "1.10", ""),
    "NLTK": ("nltk", "3.8", "graphs"),
    "Neo4j driver": ("neo4j", "5.14", "graphs"),
    "pandas": ("pandas", "2.0", ""),
    "PyTorch": ("torch", "2.2", "advanced"),
    "PyTorch Geometric": ("torch_geometric", "2.5", "advanced"),
}


def _say(status: str, item: str, detail: str, hint: str = "") -> tuple:
    print(f"{status}  {item}: {detail}" + (f"\n      {hint}" if hint and status == FAIL else ""))
    return status, item, detail


def _version_tuple(v: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", str(v))[:3])


def environment(names=("NetworkX", "scikit-learn", "SciPy", "NLTK", "Neo4j driver"), python="3.10") -> list:
    """Check the Python version and each library in names (keys of LIBRARIES); one PASS or FAIL line each."""
    out = []
    ok = sys.version_info[:2] >= _version_tuple(python)[:2]
    out.append(_say(PASS if ok else FAIL, "Python", platform.python_version(),
                    f"Python {python} or newer is needed: use Notebook Tutor, Colab, or a newer Python."))
    for name in names:
        module, oldest, extra = LIBRARIES[name]
        install = f'the install line in the first cell (with [{extra}])' if extra else "the install line in the first cell"
        try:
            version = getattr(import_module(module), "__version__", "installed")
        except Exception as e:                    # ImportError, or a library that fails while loading
            out.append(_say(FAIL, name, f"not available ({type(e).__name__})",
                            f"Run {install} again, then restart the kernel (Colab: Runtime › Restart session)."))
            continue
        ok = version == "installed" or _version_tuple(version) >= _version_tuple(oldest)
        out.append(_say(PASS if ok else FAIL, name, str(version),
                        f"Version {oldest} or newer is needed: run {install} again and restart the kernel."))
    return out


def neo4j_credentials(credentials: str | os.PathLike | None = None, folder=".") -> tuple | None:
    """(uri, user, password) from a credentials file (as Aura and Neo4j Desktop let you download it,
    NEO4J_URI=... lines), or from the NEO4J_URI, NEO4J_USERNAME and NEO4J_PASSWORD environment variables;
    with no file named, the first credentials file in folder. None when there are none."""
    files = [Path(folder) / credentials] if credentials else sorted(Path(folder).glob("*.txt"))
    for f in files:
        if not f.is_file():
            continue
        vals = dict(re.findall(r"^\s*(NEO4J_\w+)\s*=\s*(\S+)\s*$", f.read_text(errors="ignore"), re.M))
        if vals.get("NEO4J_URI") and vals.get("NEO4J_PASSWORD"):
            return vals["NEO4J_URI"], vals.get("NEO4J_USERNAME", "neo4j"), vals["NEO4J_PASSWORD"]
    env = {k: os.environ.get(k) for k in ("NEO4J_URI", "NEO4J_USERNAME", "NEO4J_PASSWORD")}
    if env["NEO4J_URI"] and env["NEO4J_PASSWORD"]:
        return env["NEO4J_URI"], env["NEO4J_USERNAME"] or "neo4j", env["NEO4J_PASSWORD"]
    return None


def neo4j(credentials: str | os.PathLike | None = None, folder=".", timeout: float = 15) -> list:
    """Check the Neo4j driver and, when credentials are found, that the database answers a query.

    credentials: the name of the credentials file next to the notebook (None: look for one, then for the
    NEO4J_* environment variables). The database line is SKIP when there are no credentials yet."""
    out = []
    try:
        import neo4j as driver_module
        out.append(_say(PASS, "Neo4j driver", driver_module.__version__))
    except Exception:
        out.append(_say(FAIL, "Neo4j driver", "not installed",
                        "Run the install line in the first cell again, then restart the kernel."))
        return out
    if credentials and not (Path(folder) / credentials).is_file():
        out.append(_say(FAIL, "Neo4j database", f"no file called {credentials} next to this notebook",
                        "Put the credentials file you downloaded next to the notebook (in Notebook Tutor: "
                        "Data files › Add data file…) and check the name, including .txt."))
        return out
    cred = neo4j_credentials(credentials, folder)
    if cred is None:
        out.append(_say(SKIP, "Neo4j database", "no credentials yet: name your credentials file and run this "
                                                "cell again"))
        return out
    uri, user, password = cred
    where = re.sub(r"//[^@/]*@", "//", uri)            # never show anything that could hold a secret
    try:
        with driver_module.GraphDatabase.driver(uri, auth=(user, password), connection_timeout=timeout) as d:
            d.verify_connectivity()
            records, _, _ = d.execute_query("RETURN 1 AS ok")
            ok = records and records[0]["ok"] == 1
        out.append(_say(PASS if ok else FAIL, "Neo4j database", f"{where} answered as {user}"))
    except Exception as e:
        kind = type(e).__name__
        hint = {"AuthError": "The user name or password is wrong: download the credentials file again, or reset "
                             "the password in the Aura console or Neo4j Desktop.",
                "ServiceUnavailable": "The database did not answer: start it (Neo4j Desktop), or resume it in the "
                                      "Aura console (a free instance pauses after a few days without use). Colab "
                                      "cannot reach a database on your own computer: use Aura with Colab."}
        out.append(_say(FAIL, "Neo4j database", f"{where} did not answer ({kind})",
                        hint.get(kind, "Check the address in the credentials file and that the database is running.")))
    return out
