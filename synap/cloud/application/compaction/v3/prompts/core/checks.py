"""Core's repair messages for the v3 code checks.

Each function builds one message from the facts the check found, using the line with its name in
``core_prompts/compaction_v3/problem_messages.txt``. Same names and arguments as the Cloud module.
"""

from synap.cloud.shared.prompt_switch import fill, pack_text


def _lines() -> dict:
    out = {}
    for line in pack_text("compaction_v3", "problem_messages.txt").splitlines():
        key, sep, text = line.partition(": ")
        if sep and key.strip():
            out[key.strip()] = text
    return out


_MESSAGES = _lines()


def _say(kind: str, **fields: object) -> str:
    return fill(_MESSAGES[kind], **fields)


def action_quote_missing(what, who, text, cited):
    return _say("quote_action", action=what, m=cited or "none", quote=text)


def action_no_quote(what):
    return _say("quote_none", action=what)


def ledger_value_missing(value, label, cited):
    return _say("quote_value", value=value, label=label, m=cited or "any turn")


def literal_missing(literal, label):
    return _say("literal", value=literal, m=label)


def inventory_not_in_record(text, m):
    return _say("coverage", item=text, m=m)


def promise_not_carried(label, sentence):
    return _say("promise", m=label, sentence=sentence)


def suggestion_not_carried(label, sentence):
    return _say("suggestion", m=label, sentence=sentence)


def response_mismatch(item, reply_m):
    return _say("response", item=item, m=reply_m)


def header_not_allowed(header):
    return _say("structure_header", header=header)


def no_such_section(section_id):
    return _say("structure_unknown_id", id=section_id)


def size_over(growth, high):
    return _say("size_over", growth=growth, max=high)


def size_under(growth, new_tokens, low):
    return _say("size_under", growth=growth, min=low)
