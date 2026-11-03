"""set_env.py writes settings into .env and must never print or accept a secret on the command line."""
import re

import pytest


def read_env(path):
    return path.read_text()


def lines_for(path, key):
    return [l for l in read_env(path).splitlines() if l.startswith(f"{key}=")]


def test_it_creates_the_file_from_the_example_and_sets_the_values(run, env_example):
    done = run("set_env.py", "--file", str(env_example), "SYNAP_LLM_PROVIDER=openai", "SYNAP_LLM_MODEL=gpt-4o-mini")
    assert done.returncode == 0, done.stdout
    assert lines_for(env_example, "SYNAP_LLM_PROVIDER") == ["SYNAP_LLM_PROVIDER=openai"]
    assert lines_for(env_example, "SYNAP_LLM_MODEL") == ["SYNAP_LLM_MODEL=gpt-4o-mini"]
    assert (env_example.stat().st_mode & 0o777) == 0o600


def test_a_commented_setting_is_switched_on_once_and_a_second_run_adds_nothing(run, env_example):
    for _ in range(2):
        assert run("set_env.py", "--file", str(env_example), "OLLAMA_NUM_CTX=8192").returncode == 0
    assert lines_for(env_example, "OLLAMA_NUM_CTX") == ["OLLAMA_NUM_CTX=8192"]


def test_a_secret_on_the_command_line_is_refused_and_not_echoed(run, env_example):
    fake = "sk-" + "Qw7" * 8
    done = run("set_env.py", "--file", str(env_example), f"OPENAI_API_KEY={fake}")
    assert done.returncode == 2
    assert "--from-env" in done.stdout
    assert fake not in done.stdout + done.stderr
    assert fake not in read_env(env_example)


@pytest.mark.parametrize("pair,reason", [
    ("NOT_A_SETTING=1", "is not a setting Synap Core reads"),
    ("SYNAP_LLM_PROVIDER=skynet", "must be one of"),
    ('SYNAP_LLM_MODEL="gpt-4o"', "quotes or spaces"),
    ("SYNAP_LLM_MODEL=", "empty"),
    ("justaword", "is not KEY=VALUE"),
])
def test_bad_pairs_are_refused_with_a_reason(run, env_example, pair, reason):
    done = run("set_env.py", "--file", str(env_example), pair)
    assert done.returncode == 2
    assert reason in done.stdout


def test_from_env_copies_a_secret_without_showing_it(run, env_example):
    secret = "sk-" + "Lm4" * 9
    done = run("set_env.py", "--file", str(env_example), "--from-env", "OPENAI_API_KEY=MY_OWN_VARIABLE",
               env={"MY_OWN_VARIABLE": secret})
    assert done.returncode == 0, done.stdout
    assert lines_for(env_example, "OPENAI_API_KEY") == [f"OPENAI_API_KEY={secret}"]
    assert secret not in done.stdout + done.stderr
    assert f"{len(secret)} characters" in done.stdout


def test_from_env_with_nothing_to_copy_is_refused(run, env_example):
    done = run("set_env.py", "--file", str(env_example), "--from-env", "OPENAI_API_KEY")
    assert done.returncode == 2
    assert "is not set in this shell" in done.stdout


def test_from_env_is_for_secrets_only(run, env_example):
    done = run("set_env.py", "--file", str(env_example), "--from-env", "SYNAP_LLM_MODEL", env={"SYNAP_LLM_MODEL": "x"})
    assert done.returncode == 2
    assert "secrets only" in done.stdout


def test_a_generated_token_is_written_and_not_printed(run, env_example):
    done = run("set_env.py", "--file", str(env_example), "--generate-token")
    assert done.returncode == 0, done.stdout
    (line,) = lines_for(env_example, "SYNAP_API_TOKEN")
    token = line.split("=", 1)[1]
    assert len(token) >= 40
    assert token not in done.stdout
    assert re.fullmatch(r"[A-Za-z0-9_-]+", token)


def test_unset_empties_a_setting_and_keeps_the_line(run, env_example):
    run("set_env.py", "--file", str(env_example), "--from-env", "SYNAP_CLOUD_API_KEY=SRC", env={"SRC": "abc" * 5})
    assert lines_for(env_example, "SYNAP_CLOUD_API_KEY") == ["SYNAP_CLOUD_API_KEY=" + "abc" * 5]
    done = run("set_env.py", "--file", str(env_example), "--unset", "SYNAP_CLOUD_API_KEY")
    assert done.returncode == 0
    assert lines_for(env_example, "SYNAP_CLOUD_API_KEY") == ["SYNAP_CLOUD_API_KEY="]


def test_a_url_is_shown_without_its_credentials_but_saved_whole(run, env_example):
    url = "https://user:hunter2@gateway.example.com/v1?token=abc123"
    done = run("set_env.py", "--file", str(env_example), f"SYNAP_LLM_BASE_URL={url}")
    assert done.returncode == 0
    assert "hunter2" not in done.stdout and "abc123" not in done.stdout
    assert "credentials or query hidden" in done.stdout
    assert lines_for(env_example, "SYNAP_LLM_BASE_URL") == [f"SYNAP_LLM_BASE_URL={url}"]


def test_nothing_to_do_is_refused(run, env_example):
    done = run("set_env.py", "--file", str(env_example))
    assert done.returncode == 2
    assert "Nothing to do" in done.stdout


def test_a_missing_example_is_reported(run, tmp_path):
    done = run("set_env.py", "--file", str(tmp_path / ".env"), "SYNAP_LLM_MODEL=x")
    assert done.returncode == 2
    assert "no .env.example" in done.stdout
