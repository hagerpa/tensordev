"""Griffe extension: render the reStructuredText bits of the docstrings as Markdown.

The docstrings are NumPy style but use Sphinx markup in places.  This extension rewrites, for
the API reference only (the source stays unchanged):

- roles such as :func:`make_core` or :class:`~tensordev.sss.FSSK` to code spans;
- :math:`x` to inline math and ``.. math::`` directives to display math;
- literal blocks introduced by a trailing ``::`` to fenced code blocks;
- double backticks in the type of a Parameters, Returns, ... item, which the type column shows
  verbatim, and around a whole line of an indented code block;
- in prose, ``<`` before a letter (raw HTML to Markdown) and ``**`` between word characters, as
  in ``d**k`` (bold to Markdown), are escaped.
"""
import re

import griffe

ROLE = re.compile(r":(?:py:)?(?:func|class|meth|attr|mod|data|exc|obj|const):`(?P<target>[^`]+)`")
MATH_ROLE = re.compile(r":math:`(?P<tex>[^`]+)`")
MATH_DIRECTIVE = re.compile(r"^(?P<indent>[ \t]*)\.\. math::[ \t]*(?P<rest>.*)$")
LITERAL = re.compile(r"^(?P<indent>[ \t]*)(?P<text>.*?)(?P<space>[ \t]?)::[ \t]*$")


def _role(match):
    target = match.group("target")
    if "<" in target and target.endswith(">"):          # :func:`title <target>`
        return f"`{target[:target.index('<')].strip()}`"
    target = target.lstrip("!")
    if target.startswith("~"):
        target = target[1:].rsplit(".", 1)[-1]
    return f"`{target}`"


def _indented_block(lines, i, indent):
    """Lines after line i that are blank or indented deeper than `indent`."""
    j = i + 1
    while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip()) > len(indent)):
        j += 1
    while j > i + 1 and not lines[j - 1].strip():
        j -= 1
    body = lines[i + 1:j]
    while body and not body[0].strip():
        body = body[1:]
    depth = min((len(ln) - len(ln.lstrip()) for ln in body if ln.strip()), default=0)
    return j, [ln[depth:] for ln in body]


def convert(text):
    lines = text.split("\n")
    out, i = [], 0
    while i < len(lines):
        line = lines[i]
        math = MATH_DIRECTIVE.match(line)
        literal = LITERAL.match(line) if not line.strip().startswith("..") else None
        if math:
            indent = math.group("indent")
            j, body = _indented_block(lines, i, indent)
            tex = [math.group("rest")] if math.group("rest") else []
            tex += [ln for ln in body if ln.strip()]
            out += ["", f"{indent}$$"] + [f"{indent}{ln}" for ln in tex] + [f"{indent}$$", ""]
            i = j
            continue
        if literal and i + 1 < len(lines) and not lines[i + 1].strip():
            indent, text_ = literal.group("indent"), literal.group("text")
            j, body = _indented_block(lines, i, indent)
            if body:
                if text_.strip():
                    out.append(f"{indent}{text_}{'' if literal.group('space') else ':'}")
                out += ["", f"{indent}```text"] + [f"{indent}{ln}".rstrip() for ln in body]
                out += [f"{indent}```", ""]
                i = j
                continue
        out.append(MATH_ROLE.sub(lambda m: f"${m.group('tex').strip()}$", ROLE.sub(_role, line)))
        i += 1
    return "\n".join(out)


ITEM_SECTIONS = {"parameters", "other parameters", "attributes", "returns", "yields", "receives",
                 "raises", "warns"}
UNDERLINE = re.compile(r"^[ \t]*-{3,}[ \t]*$")
FENCE = re.compile(r"^[ \t]*(`{3,}|~{3,})")
PROTECTED = re.compile(r"(?<!`)(`+)(?!`).+?(?<!`)\1(?!`)|\$[^$\n]+\$")   # code spans, inline math
BACKTICKS = re.compile(r"``([^`]+)``|`([^`]+)`")
CODE_LINE = re.compile(r"^(?P<indent>[ \t]*)``(?P<code>[^`]+)``(?P<punct>[,.;:]?)[ \t]*$")
ANGLE = re.compile(r"<(?=[A-Za-z])(?!https?://)")
POWER = re.compile(r"(?<=[\w)\]])\*\*(?=[\w(\[])")


def _indent(line):
    return len(line) - len(line.lstrip())


def _prose(line):
    """Escape raw-HTML `<` and exponent `**` outside code spans and inline math."""
    out, last = [], 0
    for match in PROTECTED.finditer(line):
        out += [POWER.sub(r"\\*\\*", ANGLE.sub("&lt;", line[last:match.start()])), match.group(0)]
        last = match.end()
    out.append(POWER.sub(r"\\*\\*", ANGLE.sub("&lt;", line[last:])))
    return "".join(out)


def _item(line, section):
    """Drop the backticks of the type part of a section item, `name : type` or a bare type."""
    name, sep, kind = line.partition(" : ")
    if not sep:
        name, kind = "", line
        if section in {"parameters", "other parameters", "attributes"}:
            return line
    plain = BACKTICKS.sub(lambda m: m.group(1) or m.group(2), kind)
    if plain != kind and _is_expression(plain):
        # Griffe would render `Array, shape (n, d)` as the Python tuple it parses to; a no-break
        # space keeps it a string and looks the same.
        plain = plain.replace(" ", "\u00a0")
    return f"{name}{sep}{plain}"


def _is_expression(text):
    try:
        compile(text.strip(), "<type>", "eval")
    except (SyntaxError, ValueError):
        return False
    return True


def tidy(text):
    lines = text.split("\n")
    out, section, fence, base, previous = [], None, None, 0, "blank"
    for i, line in enumerate(lines):
        opening = FENCE.match(line)
        if fence or opening:
            if opening and not fence:
                fence = opening.group(1)
            elif opening and opening.group(1)[0] == fence[0] and line.strip() == fence:
                fence = None
            out.append(line)
            previous = "fence"
            continue
        if not line.strip():
            out.append(line)
            previous = "code" if previous == "code" else "blank"
            continue
        if UNDERLINE.match(line) or (i + 1 < len(lines) and UNDERLINE.match(lines[i + 1])):
            if not UNDERLINE.match(line):
                section = line.strip().lower()
            out.append(line)
            base, previous = _indent(line), "text"
            continue
        if section in ITEM_SECTIONS and _indent(line) == 0:
            out.append(_item(line, section))
            base, previous = 0, "text"
            continue
        if previous in ("blank", "code") and _indent(line) >= base + 4:   # indented code block
            code = CODE_LINE.match(line)
            out.append(f"{code['indent']}{code['code']}{code['punct']}" if code else line)
            previous = "code"
            continue
        out.append(_prose(line))
        base, previous = _indent(line), "text"
    return "\n".join(out)


class RstToMarkdown(griffe.Extension):
    def on_instance(self, *, obj, **kwargs):
        if obj.docstring is not None:
            obj.docstring.value = tidy(convert(obj.docstring.value))
