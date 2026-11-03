"""Logic that used to sit inside prompt modules (flags, small predicates, merges).

Each module here is shipped code, not prompt text. The Cloud prompt module imports its
helpers back, so Cloud sees the same function objects, and in Core the interface of a
prompt without a Core text serves these names (synap/cloud/shared/prompt_switch.py).
"""
