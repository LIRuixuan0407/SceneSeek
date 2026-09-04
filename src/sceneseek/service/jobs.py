from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any


@dataclass(slots=True)
class JobState:
    state: str = "idle"
    operation: str | None = None
    current: int = 0
    total: int = 0
    message: str = "就绪"
    started_at: str | None = None
    finished_at: str | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = JobState()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return asdict(self._state)

    def update_progress(self, current: int, total: int, message: str) -> None:
        with self._lock:
            self._state.current = current
            self._state.total = total
            self._state.message = message

    def start(self, operation: str, callback: Callable[[], dict[str, Any]]) -> None:
        with self._lock:
            if self._state.state == "running":
                raise RuntimeError("已有索引任务正在运行")
            self._state = JobState(
                state="running",
                operation=operation,
                message="正在准备",
                started_at=datetime.now(UTC).isoformat(),
            )

        thread = threading.Thread(
            target=self._run,
            args=(callback,),
            name=f"sceneseek-{operation}",
            daemon=True,
        )
        thread.start()

    def _run(self, callback: Callable[[], dict[str, Any]]) -> None:
        try:
            result = callback()
        except Exception as error:  # noqa: BLE001 - background boundary records the failure
            with self._lock:
                self._state.state = "failed"
                self._state.error = str(error)
                self._state.message = "任务失败"
                self._state.finished_at = datetime.now(UTC).isoformat()
        else:
            with self._lock:
                self._state.state = "complete"
                self._state.result = result
                self._state.current = self._state.total
                self._state.message = "任务完成"
                self._state.finished_at = datetime.now(UTC).isoformat()
