"""The ChangeListener's sslmode must be settable, and its default must not move.

A hardcoded sslmode=disable silently broke live registry updates
where Postgres refuses unencrypted connections. The index still loaded and
ingestion still worked, so the only symptom was the listener reconnecting every
five seconds forever. Every test here CALLS listener_dsn().
"""

import pytest

from synap.cloud.agents.ingestion.resolution.startup import listener_dsn

URL = "postgresql+asyncpg://u:p@h:5432/db"


@pytest.fixture(autouse=True)
def no_inherited_env(monkeypatch):
    monkeypatch.delenv("SYNAP_ER_LISTENER_SSLMODE", raising=False)


def test_the_default_is_unchanged(monkeypatch):
    """The environments where the SSL-upgrade hang was seen must not change."""
    assert listener_dsn(URL).endswith("sslmode=disable")


def test_the_env_var_sets_it(monkeypatch):
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "require")
    dsn = listener_dsn(URL)
    assert dsn.endswith("sslmode=require")
    assert "disable" not in dsn


def test_an_sslmode_already_on_the_url_wins_over_the_env(monkeypatch):
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "disable")
    dsn = listener_dsn("postgresql+asyncpg://u:p@h:5432/db?sslmode=verify-full")
    assert "sslmode=verify-full" in dsn
    assert dsn.count("sslmode=") == 1


def test_an_empty_or_blank_env_var_falls_back_to_the_default(monkeypatch):
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "   ")
    assert listener_dsn(URL).endswith("sslmode=disable")


@pytest.mark.parametrize(
    "url",
    [
        "postgres://u:p@h:5432/db",
        "postgresql://u:p@h:5432/db",
        "postgresql+asyncpg://u:p@h:5432/db",
    ],
)
def test_every_url_form_yields_a_plain_asyncpg_dsn(monkeypatch, url):
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "require")
    dsn = listener_dsn(url)
    assert dsn.startswith("postgresql://")
    assert "+asyncpg" not in dsn
    assert dsn.endswith("sslmode=require")


def test_an_existing_query_string_gets_an_ampersand_not_a_second_question_mark(
    monkeypatch,
):
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "require")
    dsn = listener_dsn("postgresql+asyncpg://u:p@h:5432/db?application_name=synap")
    assert dsn.count("?") == 1
    assert "&sslmode=require" in dsn


def test_the_credentials_and_target_survive_untouched(monkeypatch):
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "require")
    dsn = listener_dsn("postgresql+asyncpg://user:pa%40ss@db.example:6543/synap_db")
    assert "user:pa%40ss@db.example:6543/synap_db" in dsn


def test_the_prod_shape_gets_require_when_configured(monkeypatch):
    """The exact failure: a URL with no sslmode against a TLS-only server."""
    monkeypatch.setenv("SYNAP_ER_LISTENER_SSLMODE", "require")
    dsn = listener_dsn("postgresql+asyncpg://synap-prod-user:s@10.0.0.1:5432/synap_db_prod")
    assert dsn == (
        "postgresql://synap-prod-user:s@10.0.0.1:5432/synap_db_prod?sslmode=require"
    )
