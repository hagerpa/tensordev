"""MkDocs hook: include README sections in the pages.

A page line  --8<-- "README.md:NAME"  is replaced by the README text between the markers
<!-- --8<-- [start:NAME] -->  and  <!-- --8<-- [end:NAME] -->.  Links in the README that
should point to a page of the site are rewritten with ANCHORS.  A missing marker fails the
build.  docs/scripts/check_snippets.py uses `resolve` to see the pages as MkDocs does.
"""
import re
from pathlib import Path

README = Path(__file__).resolve().parent.parent / "README.md"
INCLUDE = re.compile(r'^--8<-- "README\.md:(\w+)"$', re.MULTILINE)
ANCHORS = {}   # "](old)": "](page.md#anchor)"


def _section(text, name):
    start = f"<!-- --8<-- [start:{name}] -->\n"
    end = f"<!-- --8<-- [end:{name}] -->"
    if start not in text or end not in text:
        raise ValueError(f"README.md has no section {name!r} (markers {start.strip()} ... {end})")
    i = text.index(start) + len(start)
    return text[i:text.index(end, i)]


def resolve(markdown, readme=None):
    """Return `markdown` with every README include replaced by the README section."""
    readme = README.read_text(encoding="utf-8") if readme is None else readme

    def include(match):
        body = _section(readme, match.group(1))
        for old, new in ANCHORS.items():
            body = body.replace(old, new)
        return body

    return INCLUDE.sub(include, markdown)


def on_page_markdown(markdown, page, config, files):
    return resolve(markdown)
