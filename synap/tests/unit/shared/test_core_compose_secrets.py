"""The generated passwords file can be read by the user the server runs as (found at the first Docker boot)."""
from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[4] / "docs" / "oss" / "core_repo" / "deploy" / "docker-compose.yml"

pytestmark = pytest.mark.skipif(not COMPOSE.is_file(), reason="the Core repo's compose file is not in this tree")

SERVER_USER = "${SYNAP_USER:-10001:10001}"


@pytest.fixture(scope="module")
def services():
    return yaml.safe_load(COMPOSE.read_text())["services"]


def test_the_secrets_step_hands_the_file_to_the_server_user_and_locks_it(services):
    script = services["secrets"]["command"][-1]
    # after both branches (generated and already there), so a volume from an older run is repaired too
    assert script.index("fi\n") < script.index(f"chown {SERVER_USER} /state/secrets.env") < script.index("chmod 400 /state/secrets.env")


def test_everything_that_reads_the_file_as_the_server_image_runs_as_the_same_user(services):
    assert services["synap"]["user"] == SERVER_USER
    assert services["schema"]["user"] == SERVER_USER


def test_the_server_image_is_built_not_pulled_while_none_is_published(services):
    for name in ("schema", "first-start", "synap"):
        assert services[name]["pull_policy"] == "build"


def test_models_load_from_the_volume_with_no_network_and_nothing_is_reported():
    env = yaml.safe_load(COMPOSE.read_text())["x-synap-env"]
    assert env["HF_HUB_OFFLINE"] == "1"
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1" and env["DO_NOT_TRACK"] == "1"
    assert env["ANONYMIZED_TELEMETRY"] == "FALSE"


def test_the_stores_are_told_not_to_report_usage(services):
    # Neo4j reports usage to Neo4j by default (checked on the pinned image: dbms.usage_report.enabled is true).
    # The setting is the Docker form of that name: dots become single underscores, underscores double.
    assert services["neo4j"]["environment"]["NEO4J_dbms_usage__report_enabled"] == "false"
    assert services["chroma"]["environment"]["ANONYMIZED_TELEMETRY"] == "FALSE"


def test_the_setup_step_leaves_the_developers_config_folder_alone(services):
    # The MACA plugin writes there as the server's user. On Linux that user is the developer's own when
    # the stack is started with SYNAP_USER; the setup step must not make the folder open to everyone.
    secrets = services["secrets"]
    assert not any("/config" in v for v in secrets["volumes"])
    assert "/config" not in secrets["command"][-1]
    assert "../config:/app/config" in services["synap"]["volumes"]
