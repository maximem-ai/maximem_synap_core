"""check_use_case.py decides whether a use-case file may be sent to Synap Cloud. These tests pin what it blocks."""
import pytest

from conftest import EXAMPLES

SECTIONS = ["Agent Objective", "Target Users", "Interaction Pattern", "Task Examples", "Behavioral Guidelines",
            "Role Descriptions", "Compliance and Data Sensitivity", "Memory Priorities", "Additional Context"]
EXAMPLE_LINES = ('- **User**: "I am vegetarian."\n  **Agent**: Remembers the diet.\n'
                 '- **User**: "I fly to Lisbon in March."\n  **Agent**: Remembers the trip.\n'
                 '- **User**: "My sister is Priya."\n  **Agent**: Remembers the relation.\n')


def build(**changes):
    """A valid use-case file as text. A section set to None is left out; any other value replaces its body."""
    bodies = {name: "Plain words about this agent." for name in SECTIONS}
    bodies["Task Examples"] = EXAMPLE_LINES
    bodies.update(changes)
    parts = ["# Use-Case Description\n"]
    for name in SECTIONS:
        if bodies[name] is not None:
            parts.append(f"## {name}\n\n{bodies[name]}\n")
    return "\n".join(parts)


@pytest.fixture
def check(run, tmp_path):
    def go(text, raw=False):
        path = tmp_path / "use-case.md"
        path.write_bytes(text if raw else text.encode())
        return run("check_use_case.py", "--file", str(path))
    return go


def last_lines(done):
    out = done.stdout.strip().splitlines()
    return out[-2], out[-1]


@pytest.mark.parametrize("name", ["use-case.md", "use-case-personal-chatbot.md"])
def test_the_shipped_examples_pass(run, name):
    done = run("check_use_case.py", "--file", str(EXAMPLES / name))
    assert done.returncode == 0, done.stdout
    assert last_lines(done) == ("RESULT: PASS", "EXIT CODE: 0")


def test_a_complete_file_passes(check):
    done = check(build())
    assert done.returncode == 0, done.stdout
    assert "3 task examples" in done.stdout


def test_a_missing_file_fails_and_says_where_to_start(run, tmp_path):
    done = run("check_use_case.py", "--file", str(tmp_path / "nope.md"))
    assert done.returncode == 1
    assert "does not exist" in done.stdout


def test_an_empty_file_fails(check):
    assert check("   \n").returncode == 1


@pytest.mark.parametrize("section", ["Agent Objective", "Target Users", "Task Examples"])
def test_a_missing_required_section_fails_and_names_it(check, section):
    done = check(build(**{section: None}))
    assert done.returncode == 1
    assert f"the section '{section}' is missing" in done.stdout


def test_a_required_section_that_says_only_not_stated_fails(check):
    done = check(build(**{"Agent Objective": "not stated"}))
    assert done.returncode == 1
    assert "says only `not stated`" in done.stdout


def test_a_missing_optional_section_only_warns(check):
    done = check(build(**{"Memory Priorities": None}))
    assert done.returncode == 0
    assert "[WARN] the section 'Memory Priorities' is missing" in done.stdout


def test_headings_are_matched_loosely(check):
    text = build().replace("## Compliance and Data Sensitivity", "## compliance & data sensitivity")
    done = check(text)
    assert done.returncode == 0, done.stdout
    assert "'Compliance and Data Sensitivity' is missing" not in done.stdout


def test_task_examples_without_the_user_agent_form_fail(check):
    done = check(build(**{"Task Examples": "Some sentences about what people say."}))
    assert done.returncode == 1
    assert "no examples in the form" in done.stdout


def test_two_examples_warn_but_pass(check):
    two = "\n".join(EXAMPLE_LINES.splitlines()[:4])
    done = check(build(**{"Task Examples": two}))
    assert done.returncode == 0
    assert "2 example(s)" in done.stdout


def test_a_file_over_256_kb_fails(check):
    done = check(build(**{"Additional Context": "x" * (256 * 1024 + 1)}))
    assert done.returncode == 1
    assert "the limit is 256 KB" in done.stdout


def test_a_file_that_is_not_utf8_fails(check):
    done = check(b"\xff\xfe## Agent Objective\n", raw=True)
    assert done.returncode == 1
    assert "not valid UTF-8" in done.stdout


def test_control_characters_fail(check):
    done = check(build(**{"Additional Context": "bad\x01char"}))
    assert done.returncode == 1
    assert "control characters" in done.stdout


# -- secrets: the text is built at run time, so no key-shaped string is committed ------------------------------
SECRETS = {
    "an OpenAI-style key": "sk-" + "Ab1" * 8,
    "an Anthropic key": "sk-" + "ant-" + "Zy9" * 6,
    "an AWS access key": "AK" + "IA" + "7" * 16,
    "a GitHub token": "gh" + "p_" + "q" * 30,
    "a Slack token": "xo" + "xb-" + "1" * 14,
    "a bearer token": "Bearer " + "T" * 24,
    "a private key block": "-----BEGIN " + "RSA PRIVATE KEY-----",
    "a key, token or password assignment": "api_key = " + "z" * 20,
}


@pytest.mark.parametrize("kind", SECRETS)
def test_a_secret_is_refused_by_kind_and_line_and_never_printed(check, kind):
    secret = SECRETS[kind]
    text = build(**{"Additional Context": f"Before.\n{secret}\nAfter."})
    done = check(text)
    line = text.splitlines().index(secret) + 1
    assert done.returncode == 1
    assert f"line {line} looks like {kind}" in done.stdout
    assert secret not in done.stdout + done.stderr
    assert secret.split()[-1] not in done.stdout + done.stderr


def test_a_file_without_secrets_says_so(check):
    assert "no key, token or password found" in check(build()).stdout


def test_an_email_address_warns_and_is_not_printed(check):
    address = "someone" + "@" + "example.org"
    done = check(build(**{"Additional Context": f"Write to {address}."}))
    assert done.returncode == 0
    assert "an email address" in done.stdout
    assert address not in done.stdout


def test_a_long_digit_run_warns(check):
    done = check(build(**{"Additional Context": "Number 1234 5678 9012 3456 here."}))
    assert done.returncode == 0
    assert "a long digit run" in done.stdout


def test_placeholder_text_warns_with_its_line(check):
    text = build(**{"Additional Context": "TODO write this"})
    done = check(text)
    assert done.returncode == 0
    assert f"placeholder text on line(s) {text.splitlines().index('TODO write this') + 1}" in done.stdout
