"""Comment-preserving JSONC/JSON5 editing for tool config files.

Editors such as Zed store their settings as JSONC/JSON5 -- JSON with ``//`` and
``/* */`` comments and trailing commas. Strict :func:`json.loads` chokes on
these, and the integration code historically responded to a parse failure by
writing a brand-new file containing only the keys it wanted to set, destroying
the user's entire settings file in the process.

This module fixes that by:

1. Parsing with json5 (Dirk Pranke's ``json5`` package) so JSONC/JSON5 files
   parse successfully instead of being mistaken for corrupt JSON.
2. Applying edits by splicing serialized values into the *original text* at the
   spans they occupy, so all surrounding formatting and comments survive.
3. Refusing to touch a file that genuinely cannot be parsed, rather than
   overwriting it.

json5 has no comment-preserving AST (its parser yields plain Python values), so
we locate spans ourselves with a small JSONC-aware scanner. Every edited result
is re-parsed and compared against the intended value before it is trusted; if a
splice ever fails to round-trip, callers fall back to a full reserialize.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import json5


class JsoncError(ValueError):
    """Raised when a JSONC/JSON5 document cannot be parsed."""


def loads(text: str) -> Any:
    """Parse ``text`` as JSONC/JSON5, raising :class:`JsoncError` on failure."""
    try:
        return json5.loads(text)
    except Exception as e:  # json5 raises ValueError subclasses / recursion, etc.
        raise JsoncError(str(e)) from e


# ---------------------------------------------------------------------------
# Low-level scanning over raw JSONC text.
#
# These helpers operate on ``(text, index)`` pairs and return the index that
# sits just past whatever they consumed. They assume the document already
# parsed cleanly via json5, so they only need to handle well-formed input.
# ---------------------------------------------------------------------------


def _skip_trivia(text: str, i: int) -> int:
    """Skip whitespace and ``//`` / ``/* */`` comments starting at ``i``."""
    n = len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
        elif text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
        elif text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
        else:
            break
    return i


def _scan_string(text: str, i: int) -> int:
    """Return the index past a quoted string that starts at ``i``.

    Handles both double- and single-quoted strings (JSON5) and backslash
    escapes. ``i`` must point at the opening quote.
    """
    quote = text[i]
    n = len(text)
    i += 1
    while i < n:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        if c == quote:
            return i + 1
        i += 1
    raise JsoncError("unterminated string")


def _scan_bracketed(text: str, i: int) -> int:
    """Return the index past a ``{...}`` or ``[...]`` value starting at ``i``."""
    depth = 0
    n = len(text)
    while i < n:
        if text.startswith("//", i):
            nl = text.find("\n", i)
            i = n if nl == -1 else nl + 1
            continue
        if text.startswith("/*", i):
            end = text.find("*/", i + 2)
            i = n if end == -1 else end + 2
            continue
        c = text[i]
        if c in "\"'":
            i = _scan_string(text, i)
            continue
        if c in "{[":
            depth += 1
            i += 1
            continue
        if c in "}]":
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        i += 1
    raise JsoncError("unbalanced brackets")


def _scan_value(text: str, i: int) -> int:
    """Return the index just past the value that starts at (or after) ``i``."""
    i = _skip_trivia(text, i)
    c = text[i]
    if c in "\"'":
        return _scan_string(text, i)
    if c in "{[":
        return _scan_bracketed(text, i)
    # A primitive: number / true / false / null / Infinity / NaN. Read until a
    # structural delimiter, whitespace, or comment.
    n = len(text)
    j = i
    while j < n:
        c = text[j]
        if (
            c in ",}]"
            or c.isspace()
            or text.startswith("//", j)
            or text.startswith("/*", j)
        ):
            break
        j += 1
    return j


class _Member:
    """A single ``"key": value`` pair located within an object's source text."""

    __slots__ = ("key", "key_start", "value_start", "value_end")

    def __init__(self, key: str, key_start: int, value_start: int, value_end: int):
        self.key = key
        self.key_start = key_start
        self.value_start = value_start
        self.value_end = value_end


def _object_members(text: str, open_i: int) -> tuple[list[_Member], int]:
    """Scan the object whose ``{`` is at ``open_i``.

    Returns the ordered members and the index of the matching ``}``.
    """
    if text[open_i] != "{":
        raise JsoncError("expected '{'")
    n = len(text)
    i = open_i + 1
    members: list[_Member] = []
    while True:
        i = _skip_trivia(text, i)
        if i >= n:
            raise JsoncError("unterminated object")
        if text[i] == "}":
            return members, i
        key_start = i
        if text[i] in "\"'":
            key_end = _scan_string(text, i)
            key = loads(text[key_start:key_end])
        else:
            # Unquoted (JSON5) identifier key.
            j = i
            while j < n and (text[j].isalnum() or text[j] in "_$"):
                j += 1
            key_end = j
            key = text[key_start:key_end]
        i = _skip_trivia(text, key_end)
        if i >= n or text[i] != ":":
            raise JsoncError("expected ':' after key")
        value_start = _skip_trivia(text, i + 1)
        value_end = _scan_value(text, value_start)
        members.append(_Member(key, key_start, value_start, value_end))
        i = _skip_trivia(text, value_end)
        if i < n and text[i] == ",":
            i += 1


def _line_indent(text: str, pos: int) -> str:
    """Return the leading whitespace of the line containing ``pos``."""
    line_start = text.rfind("\n", 0, pos) + 1
    j = line_start
    while j < len(text) and text[j] in " \t":
        j += 1
    return text[line_start:j]


def _serialize(value: Any, continuation_indent: str) -> str:
    """Serialize ``value`` for insertion after ``"key": ``.

    The first line stays inline; subsequent lines are indented by
    ``continuation_indent`` so nested structure lines up under the key.
    """
    raw = json.dumps(value, indent=2, ensure_ascii=False)
    lines = raw.split("\n")
    return lines[0] + "".join("\n" + continuation_indent + ln for ln in lines[1:])


def _leading_trivia(text: str) -> str:
    """Return the comments/whitespace preceding the root object (or all of it)."""
    root = _skip_trivia(text, 0)
    if root >= len(text) or text[root] != "{":
        return ""
    return text[:root]


# ---------------------------------------------------------------------------
# Splicing edits into the raw text.
# ---------------------------------------------------------------------------


def _set_in_object(text: str, open_i: int, path: tuple[str, ...], value: Any) -> str:
    """Set ``path`` to ``value`` within the object whose ``{`` is at ``open_i``."""
    members, close_i = _object_members(text, open_i)
    key = path[0]
    match = next((m for m in members if m.key == key), None)

    if len(path) == 1:
        if match is not None:
            indent = _line_indent(text, match.key_start)
            new_val = _serialize(value, indent)
            return text[: match.value_start] + new_val + text[match.value_end :]
        return _insert_member(text, open_i, members, close_i, key, value)

    # More path to traverse.
    if match is not None:
        vstart = _skip_trivia(text, match.value_start)
        if text[vstart] == "{":
            return _set_in_object(text, vstart, path[1:], value)
        # Existing non-object where we need to descend: replace it wholesale
        # with the nested structure.
        indent = _line_indent(text, match.key_start)
        new_val = _serialize(_nest(path[1:], value), indent)
        return text[: match.value_start] + new_val + text[match.value_end :]

    return _insert_member(text, open_i, members, close_i, key, _nest(path[1:], value))


def _nest(path: tuple[str, ...], value: Any) -> Any:
    """Wrap ``value`` in nested objects following ``path``."""
    for key in reversed(path):
        value = {key: value}
    return value


def _insert_member(
    text: str,
    open_i: int,
    members: list[_Member],
    close_i: int,
    key: str,
    value: Any,
) -> str:
    """Insert a new ``"key": value`` member into an object, preserving layout."""
    if members:
        indent = _line_indent(text, members[0].key_start)
    else:
        indent = _line_indent(text, open_i) + "  "
    entry = f"\n{indent}{json.dumps(key)}: {_serialize(value, indent)}"

    if members:
        # Insert as the first member followed by a comma. This is valid
        # regardless of whether the object already ends in a trailing comma,
        # and never introduces one before the closing brace.
        return text[: open_i + 1] + entry + "," + text[open_i + 1 :]

    # Empty object: place the sole member and close on its own indented line.
    closing_indent = _line_indent(text, open_i)
    return text[: open_i + 1] + entry + "\n" + closing_indent + text[close_i:]


def set_path(text: str, path: tuple[str, ...], value: Any) -> str:
    """Return ``text`` with ``path`` set to ``value``, preserving comments.

    ``text`` must contain a JSONC object at its root.
    """
    root = _skip_trivia(text, 0)
    if root >= len(text) or text[root] != "{":
        raise JsoncError("root value is not an object")
    return _set_in_object(text, root, path, value)


# ---------------------------------------------------------------------------
# Diffing two dicts into the minimal set of leaf edits.
# ---------------------------------------------------------------------------


def _diff(
    old: Any, new: Any, prefix: tuple[str, ...] = ()
) -> tuple[list[tuple[tuple[str, ...], Any]], bool]:
    """Compute the edits that turn ``old`` into ``new``.

    Returns ``(sets, has_removals)`` where ``sets`` is a list of
    ``(path, value)`` assignments. ``has_removals`` is True when a key present
    in ``old`` is gone in ``new`` -- surgical splicing does not handle deletions,
    so callers reserialize instead when that happens.
    """
    if old == new:
        return [], False
    if not (isinstance(old, dict) and isinstance(new, dict)):
        return [(prefix, new)], False

    sets: list[tuple[tuple[str, ...], Any]] = []
    has_removals = any(k not in new for k in old)
    for key, nval in new.items():
        path = prefix + (key,)
        if key not in old:
            sets.append((path, nval))
        elif old[key] != nval:
            sub_sets, sub_removed = _diff(old[key], nval, path)
            sets.extend(sub_sets)
            has_removals = has_removals or sub_removed
    return sets, has_removals


def _get_path(obj: Any, path: tuple[str, ...]) -> Any:
    for key in path:
        obj = obj[key]
    return obj


def apply_updates(text: str, updater: Callable[[dict], None]) -> str:
    """Apply ``updater`` to the parsed document and return updated JSONC text.

    ``updater`` mutates a plain ``dict`` in place (the same contract as the
    existing JSON config writers). The difference is preserved by splicing only
    the changed values back into ``text``; comments and untouched formatting
    survive. Raises :class:`JsoncError` if ``text`` does not parse or is not an
    object.

    The spliced result is re-parsed and compared against the intended document;
    if they differ (or the diff involves key removals, which splicing does not
    handle), a full reserialize is returned instead -- still preserving the
    leading comment block.
    """
    original = loads(text)
    if not isinstance(original, dict):
        raise JsoncError("root value is not an object")

    merged = copy.deepcopy(original)
    updater(merged)

    sets, has_removals = _diff(original, merged)
    if not has_removals:
        try:
            out = text
            for path, value in sets:
                out = set_path(out, path, value)
            if loads(out) == merged:
                return out
        except JsoncError:
            pass

    # Fall back to a clean reserialize, keeping the leading comment/header block.
    body = json.dumps(merged, indent=2, ensure_ascii=False)
    return _leading_trivia(text) + body + "\n"
