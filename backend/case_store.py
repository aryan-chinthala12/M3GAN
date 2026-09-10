"""
backend/case_store.py

Lightweight SQLite-backed case persistence for the NHAA SVI module.

Design notes:
  - Stores only REDACTED narrative previews + assessment metadata.
  - Thread-safe via a lock (FastAPI threadpool executes sync handlers).
  - Zero external dependencies; swap for Postgres in production by
    keeping this class's interface.
"""

import json
import sqlite3
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

from backend.schemas import CaseSummary, StatsResponse

DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cases.db"


class CaseStore:
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = str(db_path or DEFAULT_DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    # -------------------------------------------------------------
    # Connection / schema
    # -------------------------------------------------------------

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS cases (
                    case_id    TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    channel    TEXT NOT NULL,
                    language   TEXT,
                    svi_score  REAL,
                    risk_band  TEXT,
                    risk_color TEXT,
                    status     TEXT DEFAULT 'logged',
                    preview    TEXT,
                    payload    TEXT
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_cases_created "
                "ON cases (created_at DESC)"
            )

    # -------------------------------------------------------------
    # Write
    # -------------------------------------------------------------

    def create_case(
        self,
        channel: str,
        language: str,
        svi_score: float,
        risk_band: str,
        risk_color: str,
        status: str = "logged",
        text: str = "",
        payload: Optional[dict] = None,
    ) -> str:
        case_id = f"NH-{uuid.uuid4().hex[:8].upper()}"
        created_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        preview = (text or "")[:280]

        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO cases (
                    case_id, created_at, channel, language, svi_score,
                    risk_band, risk_color, status, preview, payload
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    case_id,
                    created_at,
                    channel,
                    language,
                    float(svi_score),
                    risk_band,
                    risk_color,
                    status,
                    preview,
                    json.dumps(payload) if payload else None,
                ),
            )
        return case_id

    def update_status(self, case_id: str, status: str) -> bool:
        with self._lock, self._connect() as conn:
            cursor = conn.execute(
                "UPDATE cases SET status = ? WHERE case_id = ?",
                (status, case_id),
            )
            return cursor.rowcount > 0

    # -------------------------------------------------------------
    # Read
    # -------------------------------------------------------------

    def list_cases(self, limit: int = 100) -> List[CaseSummary]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT case_id, created_at, channel, language, svi_score,
                       risk_band, risk_color, status, preview
                FROM cases
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            CaseSummary(
                case_id=row["case_id"],
                created_at=row["created_at"],
                channel=row["channel"],
                language=row["language"] or "English",
                svi_score=round(float(row["svi_score"] or 0.0), 2),
                risk_band=row["risk_band"] or "LOW",
                status=row["status"] or "logged",
                preview=row["preview"] or "",
                risk_color=row["risk_color"] or "#2E7D32",
            )
            for row in rows
        ]

    def get_case(self, case_id: str) -> Optional[dict]:
        with self._lock, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM cases WHERE case_id = ?",
                (case_id,),
            ).fetchone()

        if row is None:
            return None

        data = dict(row)
        if data.get("payload"):
            try:
                data["payload"] = json.loads(data["payload"])
            except json.JSONDecodeError:
                data["payload"] = None
        return data

    def get_stats(self) -> StatsResponse:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT svi_score, risk_band, channel FROM cases"
            ).fetchall()

        by_risk = {"LOW": 0, "MODERATE": 0, "HIGH": 0, "CRITICAL": 0}
        by_channel = {}
        scores = []

        for row in rows:
            band = row["risk_band"] or "LOW"
            by_risk[band] = by_risk.get(band, 0) + 1

            channel = row["channel"] or "other"
            by_channel[channel] = by_channel.get(channel, 0) + 1

            if row["svi_score"] is not None:
                scores.append(float(row["svi_score"]))

        total = len(rows)
        avg = round(sum(scores) / len(scores), 2) if scores else 0.0

        return StatsResponse(
            total_cases=total,
            by_risk=by_risk,
            by_channel=by_channel,
            avg_svi=avg,
            critical_cases=by_risk.get("CRITICAL", 0),
        )
