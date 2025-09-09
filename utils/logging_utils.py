from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional
import csv


def get_logs_dir() -> Path:
    root = Path(__file__).resolve().parent.parent
    logs_dir = root / "data" / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir


class EpisodeCSVLogger:
    """
    Structured CSV logger with schema:
    episode,step,reward,avg_wait_time,queue_length
    """

    def __init__(self, filename: str = "training_log.csv") -> None:
        logs_dir = get_logs_dir()
        self.filepath = logs_dir / filename
        self._ensure_header()

    def _ensure_header(self) -> None:
        if not self.filepath.exists():
            with self.filepath.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["episode", "step", "reward", "avg_wait_time", "queue_length"])

    def log_row(self, episode: int, step: int, reward: float,
                avg_wait_time: Optional[float] = None,
                queue_length: Optional[int] = None) -> None:
        with self.filepath.open("a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([episode, step, reward, avg_wait_time, queue_length])


