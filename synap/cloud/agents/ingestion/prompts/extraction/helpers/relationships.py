"""Relationship-prompt helper that is logic, not prompt text: does the content speak in the first person.

Moved byte for byte out of cloud/relationships.py so Core can ship it; that module imports it back.
"""
import re

_FIRST_PERSON = re.compile(r"\bI\b|\b(?i:me|my|mine|myself)\b")

# Double-quoted speech: its "I" is the person quoted. Single quotes are left
# alone, they are apostrophes far more often than quotation marks.
_QUOTED = re.compile(r'"[^"\n]*"|\u201c[^\u201d\n]*\u201d')

def writer_speaks(content: str) -> bool:
    """True when the content says "I", "me" or "my" (English only).

    The writer is named to the model only then. Content with no first person
    gets the prompt it always got, so naming the writer cannot change how a
    third-person note is read. "we" and "our" alone do not count: they are the
    writer's company or team as often as the writer. Nor does an "I" inside
    double quotes: that is the person quoted.
    """
    return bool(_FIRST_PERSON.search(_QUOTED.sub(" ", content or "")))
