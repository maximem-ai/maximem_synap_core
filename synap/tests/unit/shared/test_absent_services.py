"""A start-up service whose package is not in the build is one INFO line, not a warning with a traceback."""
import logging
import subprocess
import sys
from pathlib import Path

import pytest

from synap.cloud.shared import edition as ed
from synap.cloud.shared.absent_services import AbsentServiceFilter


@pytest.fixture
def log():
    logger = logging.getLogger("test.absent_services")
    logger.setLevel(logging.DEBUG)
    logger.addFilter(AbsentServiceFilter())
    records = []

    class Keep(logging.Handler):
        def emit(self, record):
            records.append(record)

    handler = Keep()
    logger.addHandler(handler)
    yield logger, records
    logger.removeHandler(handler)
    logger.filters.clear()


def _raise_missing(logger, level, message):
    try:
        import synap.cloud.privacy_not_in_this_build  # noqa: F401
    except ModuleNotFoundError:
        logger.log(level, message, exc_info=True)


def test_a_missing_package_becomes_one_info_line_with_no_traceback(log):
    logger, records = log
    _raise_missing(logger, logging.ERROR, "PII ingress gate FAILED to start — documents will be ingested without detection")
    _raise_missing(logger, logging.WARNING, "Credit allowance worker failed to start")
    (pii, credit) = records
    assert pii.levelno == credit.levelno == logging.INFO
    assert pii.getMessage() == ("PII ingress gate: not started, "
                                "synap.cloud.privacy_not_in_this_build is not part of this build")
    assert credit.getMessage().startswith("Credit allowance worker: not started")
    assert pii.exc_info is None and credit.exc_info is None


def test_the_word_failed_does_not_survive_in_the_info_line(log):
    logger, records = log
    _raise_missing(logger, logging.WARNING, "Observability phase-2 metrics bootstrap failed; continuing without metrics")
    assert records[0].getMessage() == (
        "Observability phase-2 metrics bootstrap: not started, synap.cloud.privacy_not_in_this_build is not part of this build")


def test_any_other_failure_keeps_its_level_and_traceback(log):
    logger, records = log
    try:
        raise RuntimeError("boom")
    except RuntimeError:
        logger.warning("Compaction worker failed to start", exc_info=True)
    logger.warning("plain warning")
    assert records[0].levelno == logging.WARNING and records[0].exc_info is not None
    assert records[0].getMessage() == "Compaction worker failed to start"
    assert records[1].levelno == logging.WARNING


def _filters_on_main(edition):
    code = (
        "import logging;from synap.cloud.application import main;"
        "from synap.cloud.shared.absent_services import AbsentServiceFilter as F;"
        "print(any(isinstance(f, F) for f in main.logger.filters))"
    )
    import os
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                         env={**os.environ, "SYNAP_EDITION": edition}, timeout=300)
    return out.stdout.strip().splitlines()[-1] if out.stdout.strip() else out.stderr[-500:]


def test_only_core_quiets_absent_services():
    assert ed.resolve({}).quiet_absent_services is False
    assert ed.resolve({"SYNAP_EDITION": "core"}).quiet_absent_services is True


@pytest.mark.skipif(not Path(__file__).resolve().parents[2].joinpath("cloud", "cloud_hooks").is_dir(),
                    reason="the Cloud edition cannot start without cloud_hooks, which Core does not have")
def test_main_installs_the_filter_in_core_and_not_in_cloud():
    assert _filters_on_main("core") == "True"
    assert _filters_on_main("cloud") == "False"
