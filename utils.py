import json
import time
import threading
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def load_jsonl(file_path: str) -> pd.DataFrame:
    """Load a JSONL file and extract user questions + ground-truth assistant facts."""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line.strip())
            messages = obj.get("messages", [])

            user_message = None
            assistant_message = None
            for msg in messages:
                if msg["role"] == "user":
                    user_message = msg["content"]
                elif msg["role"] == "assistant":
                    assistant_message = msg["content"]

            if user_message and assistant_message:
                try:
                    gt_facts = json.loads(assistant_message)
                except json.JSONDecodeError:
                    gt_facts = assistant_message

                data.append({"question": user_message, "gt_facts": gt_facts})

    return pd.DataFrame(data)


class ProgressTracker:
    """Thread-safe progress tracker with ETA logging."""

    def __init__(self, total_tasks: int):
        self.total_tasks = total_tasks
        self.completed_tasks = 0
        self.lock = threading.Lock()
        self.start_time = time.time()

    def update(self, increment: int = 1):
        with self.lock:
            self.completed_tasks += increment
            if self.completed_tasks % 10 == 0 or self.completed_tasks == self.total_tasks:
                elapsed = time.time() - self.start_time
                rate = self.completed_tasks / elapsed if elapsed > 0 else 0
                eta = (self.total_tasks - self.completed_tasks) / rate if rate > 0 else 0
                logger.info(
                    f"Progress: {self.completed_tasks}/{self.total_tasks} "
                    f"({self.completed_tasks / self.total_tasks * 100:.1f}%) "
                    f"Rate: {rate:.2f} samples/sec  ETA: {eta / 60:.1f} min"
                )
