"""check_env.py says whether the server can start with a .env, and prints names and lengths, never values."""
import pytest

GOOD = {"SYNAP_LLM_PROVIDER": "openai", "SYNAP_LLM_MODEL": "gpt-4o-mini", "OPENAI_API_KEY": "sk-" + "Hj2" * 10}


@pytest.fixture
def check(run, tmp_path):
    def go(values, mode=0o600, raw=None):
        path = tmp_path / ".env"
        path.write_text(raw if raw is not None else "".join(f"{k}={v}\n" for k, v in values.items()))
        path.chmod(mode)
        return run("check_env.py", "--file", str(path))
    return go


def test_a_good_openai_file_passes_and_the_key_is_only_counted(check):
    done = check(GOOD)
    assert done.returncode == 0, done.stdout
    assert "RESULT: PASS" in done.stdout
    assert GOOD["OPENAI_API_KEY"] not in done.stdout
    assert f"OPENAI_API_KEY is set ({len(GOOD['OPENAI_API_KEY'])} characters)" in done.stdout


def test_a_missing_file_fails(run, tmp_path):
    done = run("check_env.py", "--file", str(tmp_path / ".env"))
    assert done.returncode == 1
    assert "does not exist" in done.stdout


def test_the_unedited_example_fails_on_the_placeholders(run, tmp_path):
    from conftest import REPO
    path = tmp_path / ".env"
    path.write_text((REPO / ".env.example").read_text())
    path.chmod(0o600)
    done = run("check_env.py", "--file", str(path))
    assert done.returncode == 1
    assert "SYNAP_LLM_PROVIDER is still the placeholder" in done.stdout
    assert "SYNAP_LLM_MODEL is still the placeholder" in done.stdout


def test_a_provider_that_does_not_exist_fails(check):
    done = check({**GOOD, "SYNAP_LLM_PROVIDER": "skynet"})
    assert done.returncode == 1
    assert "must be one of" in done.stdout


@pytest.mark.parametrize("provider,key", [("openai", "OPENAI_API_KEY"), ("anthropic", "ANTHROPIC_API_KEY")])
def test_a_hosted_provider_needs_its_key(check, provider, key):
    done = check({"SYNAP_LLM_PROVIDER": provider, "SYNAP_LLM_MODEL": "some-model"})
    assert done.returncode == 1
    assert f"{key} is empty" in done.stdout


def test_ollama_needs_no_key(check):
    done = check({"SYNAP_LLM_PROVIDER": "ollama", "SYNAP_LLM_MODEL": "llama3"})
    assert done.returncode == 0, done.stdout


def test_openai_compatible_needs_an_address(check):
    done = check({"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m"})
    assert done.returncode == 1
    assert "SYNAP_LLM_BASE_URL is empty" in done.stdout


def test_an_address_without_a_scheme_fails(check):
    done = check({"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m", "SYNAP_LLM_BASE_URL": "example.com/v1"})
    assert done.returncode == 1
    assert "must start with http" in done.stdout


def test_an_address_with_credentials_is_not_shown(check):
    done = check({"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m",
                  "SYNAP_LLM_BASE_URL": "https://user:hunter2@gateway.example.com/v1?token=abc123"})
    assert "hunter2" not in done.stdout and "abc123" not in done.stdout
    assert "has credentials in its address" in done.stdout


def test_localhost_is_flagged_because_the_server_runs_in_a_container(check):
    done = check({"SYNAP_LLM_PROVIDER": "openai-compatible", "SYNAP_LLM_MODEL": "m", "SYNAP_LLM_BASE_URL": "http://localhost:11434/v1"})
    assert "host.docker.internal" in done.stdout


def test_quotes_around_a_value_fail(check):
    done = check({**GOOD, "SYNAP_LLM_MODEL": '"gpt-4o"'})
    assert done.returncode == 1
    assert "quotes or spaces" in done.stdout


def test_a_stray_character_in_a_key_fails_and_the_key_is_not_printed(check):
    bad = "sk-" + "Pq3" * 6 + "​" + "Rt5" * 3
    done = check({**GOOD, "OPENAI_API_KEY": bad})
    assert done.returncode == 1
    assert "not plain ASCII" in done.stdout
    assert bad not in done.stdout and "Pq3" * 6 not in done.stdout


def test_a_space_inside_a_key_fails(check):
    done = check({**GOOD, "OPENAI_API_KEY": "sk-" + "ab" * 5 + " " + "cd" * 5})
    assert done.returncode == 1
    assert "has a space inside it" in done.stdout


def test_a_file_other_users_can_read_warns(check):
    done = check(GOOD, mode=0o644)
    assert done.returncode == 0
    assert "chmod 600" in done.stdout


def test_a_key_for_another_provider_warns_and_does_not_fail(check):
    done = check({**GOOD, "ANTHROPIC_API_KEY": "sk-" + "ant-" + "Kd8" * 6})
    assert done.returncode == 0
    assert "ANTHROPIC_API_KEY is set but the provider is openai" in done.stdout


def test_no_token_warns_and_a_token_is_only_counted(check):
    assert "SYNAP_API_TOKEN is not set" in check(GOOD).stdout
    token = "Vx" * 20
    done = check({**GOOD, "SYNAP_API_TOKEN": token})
    assert token not in done.stdout
    assert f"SYNAP_API_TOKEN is set ({len(token)} characters)" in done.stdout


def test_a_leftover_cloud_key_warns(check):
    done = check({**GOOD, "SYNAP_CLOUD_API_KEY": "mk-" + "Zz1" * 6})
    assert "SYNAP_CLOUD_API_KEY is set" in done.stdout
    assert "Zz1" * 6 not in done.stdout
