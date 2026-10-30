"""Saved copies of web pages and API responses that session notebooks request.

A session whose helpers send web requests (ARI1101 S03) keeps a copy of every response its
notebooks use, so that the notebooks run when the network does not. This tool compares the
copies with the live sites and services, and captures them again:

    python -m ai_learning_kit.dev.saved_copies check      # compare every copy with the live response
    python -m ai_learning_kit.dev.saved_copies capture    # write the live responses into the package

`check` never fails: a site that cannot be reached, or a response that changed, is reported
(as a GitHub annotation in CI, with the live response attached, so it can be reviewed and
saved). `capture` needs a network that reaches every service; review the changes with git
before committing them, because helpers' results must not change once released.
"""
from __future__ import annotations

import base64
import gzip
import json
import os
import sys
from datetime import date

VOLATILE = ("generationtime_ms",)       # fields that differ on every request


def _sessions():
    from ..core import registry
    from .. import units
    for code in units.REGISTRY:
        for scode in getattr(units.unit(code), "SESSIONS", {}):
            mod = registry.session(code, scode)
            helpers = registry.tools(mod)
            if helpers is not None and hasattr(helpers, "SAVED_REQUESTS"):
                yield mod, helpers


def _comparable(body: bytes, content_type: str):
    if "json" in content_type:
        try:
            data = json.loads(body)
            if isinstance(data, dict):
                for k in VOLATILE:
                    data.pop(k, None)
            return data
        except ValueError:
            pass
    return body


def _note(level: str, title: str, text: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        text = text.replace("%", "%25").replace("\r", "").replace("\n", "%0A")
        print(f"::{level} title={title}::{text}")
    else:
        print(f"[{level}] {title}: {text}")


def _attach(title: str, status: int, content_type: str, body: bytes, size: int = 3000) -> None:
    payload = base64.b64encode(gzip.compress(body, mtime=0)).decode("ascii")
    parts = [payload[i:i + size] for i in range(0, len(payload), size)] or [""]
    for i, part in enumerate(parts, 1):
        _note("notice", f"{title} live {i}/{len(parts)}", f"status={status} type={content_type} gzip-base64={part}")


def _live(helpers, url, params):
    import requests
    return requests.get(url, params=params, headers=helpers.HEADERS, timeout=30)


def check() -> int:
    for mod, h in _sessions():
        index = h._index()
        for name, (url, params) in h.SAVED_REQUESTS.items():
            title = f"{mod.UNIT} {mod.SESSION} {name}"
            try:
                r = _live(h, url, params)
            except Exception as e:
                _note("warning", title, f"could not reach {url}: {type(e).__name__}: {e}")
                continue
            ctype = r.headers.get("Content-Type", "")
            path = h.SAVED / name
            if name not in index or not path.is_file():
                _note("warning", title, f"no saved copy yet; live status {r.status_code}")
                _attach(title, r.status_code, ctype, r.content)
            elif r.status_code != index[name]["status"] or _comparable(r.content, ctype) != _comparable(path.read_bytes(), ctype):
                _note("warning", title, f"the live response differs from the saved copy (live status {r.status_code})")
                _attach(title, r.status_code, ctype, r.content)
            else:
                _note("notice", title, "the saved copy matches the live response")
        site_files = getattr(h, "SITE_FILES", None)
        if site_files is not None:
            import requests
            for path in sorted(site_files.iterdir()):
                title = f"{mod.UNIT} {mod.SESSION} site/{path.name}"
                try:
                    r = requests.get(h.SITE + path.name, headers=h.HEADERS, timeout=30)
                except Exception as e:
                    _note("warning", title, f"could not reach {h.SITE}: {type(e).__name__}")
                    continue
                if r.status_code != 200:
                    _note("warning", title, f"{h.SITE}{path.name} returned {r.status_code} (is the site published?)")
                elif r.content != path.read_bytes():
                    _note("warning", title, "the published page differs from the kit's copy")
                else:
                    _note("notice", title, "the published page matches the kit's copy")
    return 0


def capture() -> int:
    for mod, h in _sessions():
        h.SAVED.mkdir(parents=True, exist_ok=True)
        index = h._index()
        for name, (url, params) in h.SAVED_REQUESTS.items():
            r = _live(h, url, params)
            (h.SAVED / name).write_bytes(r.content)
            index[name] = {"status": r.status_code, "content_type": r.headers.get("Content-Type", ""),
                           "url": r.url, "captured": date.today().isoformat()}
            print(f"{mod.UNIT} {mod.SESSION} {name}: {r.status_code}, {len(r.content)} bytes")
        (h.SAVED / "saved.json").write_text(json.dumps(index, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["check"]:
        return check()
    if argv == ["capture"]:
        return capture()
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
