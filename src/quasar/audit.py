from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class AuditLog:
    def __init__(self, path: str | Path = "outputs/audit.sqlite3") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS audit_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                )"""
            )

    def record(self, event_type: str, payload: dict[str, Any]) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT INTO audit_events (event_type, payload_json) VALUES (?, ?)",
                (event_type, json.dumps(payload, sort_keys=True, default=str)),
            )

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        with sqlite3.connect(self.path) as connection:
            rows = connection.execute(
                "SELECT id, timestamp, event_type, payload_json FROM audit_events ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [
            {"id": row[0], "timestamp": row[1], "event_type": row[2], "payload": json.loads(row[3])}
            for row in rows
        ]

