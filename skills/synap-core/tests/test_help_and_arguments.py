"""Every script answers --help, and the ones that need a server refuse bad arguments before they touch it."""
import importlib.util

import pytest

from conftest import SCRIPTS, run_script

ALL = sorted(p.name for p in SCRIPTS.glob("*.py"))


def test_the_eight_scripts_are_all_covered():
    assert len(ALL) == 8, ALL


@pytest.mark.parametrize("name", ALL)
def test_help_exits_zero_and_says_what_the_script_does(name):
    done = run_script(name, "--help")
    assert done.returncode == 0, done.stderr
    assert "usage:" in done.stdout.lower()


def test_generate_config_needs_a_use_case_file_and_a_relationship():
    done = run_script("generate_config.py")
    assert done.returncode == 2
    assert "--use-case" in done.stderr


def test_generate_config_refuses_an_unknown_relationship():
    done = run_script("generate_config.py", "--use-case", "use-case.md", "--relationship", "b3b")
    assert done.returncode == 2
    assert "invalid choice" in done.stderr


@pytest.mark.parametrize("port", ["8000", "8010"])
def test_preflight_checks_the_port_that_compose_publishes(port, monkeypatch):
    monkeypatch.setenv("SYNAP_PORT", port)
    spec = importlib.util.spec_from_file_location("preflight_under_test", SCRIPTS / "preflight.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.PORT == int(port)
