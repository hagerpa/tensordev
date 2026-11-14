"""Run the Python code blocks of the documentation pages.

Run from the repository root with the Python environment that has JAX installed:

    python docs/scripts/check_snippets.py                    # every docs/*.md page
    python docs/scripts/check_snippets.py docs/volterra.md   # some pages
    python docs/scripts/check_snippets.py README.md          # the README on its own
    python docs/scripts/check_snippets.py --list             # show the blocks, run nothing

README includes are resolved with docs/hooks.py first, so a page is checked as MkDocs renders
it.  The ```python (or ```py) blocks of a page run in order in one fresh Python process, with
PYTHONPATH=src and the repository root as working directory; later blocks see the names that
earlier blocks defined.  Pages do not share state.

A block that must not run, for example because it needs a GPU, is marked by an HTML comment
on the nearest non-blank line above its fence:

    <!-- check_snippets: skip (needs a GPU) -->

The comment is invisible on the site, on GitHub and on PyPI.  Give the reason in parentheses.
The script exits with status 1 when a block fails, a page times out, or a marker is malformed.
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"

_spec = importlib.util.spec_from_file_location("tensordev_docs_hooks", DOCS / "hooks.py")
hooks = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(hooks)

FENCE_OPEN = re.compile(r"^(?P<indent>[ \t]*)(?P<fence>`{3,}|~{3,})[ \t]*(?P<info>[^`\n]*?)[ \t]*$")
DIRECTIVE = re.compile(r"^<!--\s*check_snippets:(?P<body>.*?)-->$")
SKIP = re.compile(r"^\s*skip(?:\s*\((?P<reason>[^)]*)\))?\s*$")
PYTHON = {"python", "py", "python3"}

DRIVER = r"""
import json, linecache, resource, sys, traceback
blocks = json.load(open(sys.argv[1], encoding="utf-8"))
namespace = {"__name__": "__main__"}
status = 0
for block in blocks:
    linecache.cache[block["label"]] = (len(block["code"]), None, block["code"].splitlines(True), block["label"])
    try:
        exec(compile(block["code"], block["label"], "exec"), namespace)
    except BaseException:
        traceback.print_exc()
        print(f"FAILED: {block['label']}", file=sys.stderr)
        status = 1
        break
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)
json.dump({"peak_rss_bytes": peak}, open(sys.argv[2], "w"))
sys.exit(status)
"""


@dataclass
class Block:
    code: str
    label: str
    skip: str | None   # the reason, when the block is marked to be skipped


class SnippetError(Exception):
    pass


def _language(info):
    info = info.strip()
    if info.startswith("{"):
        classes = [t[1:] for t in info.strip("{}").split() if t.startswith(".")]
        return classes[0].lower() if classes else ""
    return info.split()[0].lower() if info else ""


def _origin(code, name, page_text, readme_text, resolved_line):
    """Where a block comes from, as `file:line` of its first code line."""
    first = code.splitlines()[0] if code.strip() else ""
    for label, text in ((name, page_text), ("README.md", readme_text)):
        i = text.find(code)
        if i >= 0:
            return f"{label}:{text.count(chr(10), 0, i) + 1}"
    return f"{name}:{resolved_line} (after includes; first line {first!r})"


def extract(path):
    """Return the code blocks of a page after README includes are resolved."""
    page_text = path.read_text(encoding="utf-8")
    readme_text = hooks.README.read_text(encoding="utf-8")
    text = hooks.resolve(page_text, readme_text)
    name = str(path.resolve().relative_to(ROOT)) if path.resolve().is_relative_to(ROOT) else str(path)
    lines = text.split("\n")
    blocks, pending_skip, i = [], None, 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        directive = DIRECTIVE.match(stripped)
        if directive:
            skip = SKIP.match(directive.group("body"))
            if not skip:
                raise SnippetError(f"{name}:{i + 1}: unknown directive {stripped!r}")
            if pending_skip is not None:
                raise SnippetError(f"{name}:{i + 1}: two skip markers in a row")
            pending_skip = (skip.group("reason") or "no reason given").strip(), i + 1
            i += 1
            continue
        opening = FENCE_OPEN.match(line)
        if opening:
            indent, fence = opening.group("indent"), opening.group("fence")
            closing = re.compile(rf"^[ \t]*{re.escape(fence[0])}{{{len(fence)},}}[ \t]*$")
            j = i + 1
            while j < len(lines) and not closing.match(lines[j]):
                j += 1
            if j == len(lines):
                raise SnippetError(f"{name}:{i + 1}: code fence is never closed")
            body = [ln[len(indent):] if ln.startswith(indent) else ln.lstrip() for ln in lines[i + 1:j]]
            code = "\n".join(body) + "\n"
            if _language(opening.group("info")) in PYTHON:
                origin = _origin(code, name, page_text, readme_text, i + 2)
                blocks.append(Block(code, origin, pending_skip[0] if pending_skip else None))
            elif pending_skip is not None:
                raise SnippetError(f"{name}:{pending_skip[1]}: skip marker above a non-Python block")
            pending_skip, i = None, j + 1
            continue
        if stripped and pending_skip is not None:
            raise SnippetError(f"{name}:{pending_skip[1]}: skip marker is not followed by a code block")
        i += 1
    if pending_skip is not None:
        raise SnippetError(f"{name}:{pending_skip[1]}: skip marker at the end of the page")
    return name, blocks


def run_page(name, blocks, python, timeout):
    """Run the blocks of one page; return (ok, message)."""
    runnable = [{"code": b.code, "label": b.label} for b in blocks if b.skip is None]
    skipped = len(blocks) - len(runnable)
    counts = f"{len(runnable)} run, {skipped} skipped"
    if not runnable:
        return True, f"{name}: no blocks to run ({counts})"
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [str(ROOT / "src"), env.get("PYTHONPATH")]))
    env.setdefault("MPLBACKEND", "Agg")
    with tempfile.TemporaryDirectory() as tmp:
        payload, stats = Path(tmp) / "blocks.json", Path(tmp) / "stats.json"
        payload.write_text(json.dumps(runnable), encoding="utf-8")
        started = time.perf_counter()
        try:
            proc = subprocess.run([python, "-c", DRIVER, str(payload), str(stats)], cwd=ROOT, env=env,
                                  capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return False, f"{name}: timed out after {timeout} s ({counts})"
        seconds = time.perf_counter() - started
        peak = json.loads(stats.read_text())["peak_rss_bytes"] if stats.exists() else None
    memory = "" if peak is None else f", peak RSS {peak / 2**30:.2f} GiB"
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).rstrip()
        return False, f"{name}: FAILED ({counts}, {seconds:.1f} s{memory})\n{detail}"
    return True, f"{name}: ok ({counts}, {seconds:.1f} s{memory})"


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("pages", nargs="*", type=Path, help="Markdown files (default: docs/*.md)")
    parser.add_argument("--python", default=sys.executable, help="interpreter for the blocks (default: this one)")
    parser.add_argument("--timeout", type=float, default=900, help="seconds per page (default: 900)")
    parser.add_argument("--jobs", "-j", type=int, default=1, help="pages run in parallel (default: 1)")
    parser.add_argument("--list", action="store_true", help="list the blocks and run nothing")
    args = parser.parse_args()

    pages = args.pages or sorted(DOCS.glob("*.md"))
    try:
        extracted = [extract(page) for page in pages]
    except SnippetError as error:
        sys.exit(f"error: {error}")

    if args.list:
        for name, blocks in extracted:
            print(f"{name}: {len(blocks)} Python blocks")
            for block in blocks:
                status = f"skip ({block.skip})" if block.skip else "run"
                print(f"  {block.label:<60} {status}")
        return

    print(f"Python: {args.python}", flush=True)
    failures = 0
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = [pool.submit(run_page, name, blocks, args.python, args.timeout) for name, blocks in extracted]
        for future in futures:
            ok, message = future.result()
            failures += not ok
            print(message, flush=True)
    if failures:
        sys.exit(f"{failures} of {len(extracted)} pages failed")


if __name__ == "__main__":
    main()
