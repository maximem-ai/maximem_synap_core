"""The loop around ``run_cycle``. Same shape as the other maintenance workers."""

from __future__ import annotations

import asyncio
import logging

from synap.cloud.application.retention import sweep

logger = logging.getLogger(__name__)


class RetentionWorker:
    def __init__(self, tick_seconds: int | None = None):
        self.tick_seconds = tick_seconds or sweep.interval_seconds()
        self._stopped = asyncio.Event()
        self._ledger = sweep.CleanupLedger()

    def stop(self) -> None:
        self._stopped.set()

    async def run_forever(self) -> None:
        logger.info("retention_worker_starting tick_seconds=%s", self.tick_seconds)
        try:
            deps = sweep._real_deps()
        except Exception:
            logger.exception("retention_worker_could_not_start")
            return
        while not self._stopped.is_set():
            try:
                await sweep.run_cycle(deps, self._ledger)
            except Exception:
                logger.exception("retention_cycle_failed")
            try:
                await asyncio.wait_for(self._stopped.wait(), timeout=self.tick_seconds)
            except asyncio.TimeoutError:
                continue
