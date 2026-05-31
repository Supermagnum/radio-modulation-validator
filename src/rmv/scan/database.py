"""SQLite audit database for rmv scan (.rmv_findings.db)."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_scan_timestamp(value: str) -> str:
    """Parse a scan timestamp and return UTC ISO-8601 (lexicographically comparable)."""
    text = value.strip()
    if not text:
        msg = "Timestamp must not be empty"
        raise ValueError(msg)
    if text.endswith("Z"):
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    else:
        try:
            dt = datetime.fromisoformat(text)
        except ValueError as exc:
            msg = f"Invalid timestamp: {value!r} (use e.g. 2026-05-31T15:00:00 or ...Z)"
            raise ValueError(msg) from exc
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY,
    path TEXT UNIQUE NOT NULL,
    name TEXT NOT NULL,
    gr_version TEXT,
    readme_path TEXT,
    last_scanned TEXT,
    scan_status TEXT
);

CREATE TABLE IF NOT EXISTS blocks (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    block_name TEXT NOT NULL,
    block_file TEXT,
    expected_family TEXT,
    expected_order TEXT,
    gr_version TEXT
);

CREATE TABLE IF NOT EXISTS validations (
    id INTEGER PRIMARY KEY,
    block_id INTEGER REFERENCES blocks(id),
    run_at TEXT NOT NULL,
    iq_file TEXT,
    predicted_family TEXT,
    predicted_order TEXT,
    family_confidence REAL,
    order_confidence REAL,
    family_pass INTEGER,
    order_pass INTEGER,
    hard_fail INTEGER,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS issues (
    id INTEGER PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id),
    block_id INTEGER REFERENCES blocks(id),
    detected_at TEXT NOT NULL,
    severity TEXT,
    description TEXT,
    resolved INTEGER DEFAULT 0
);
"""


@dataclass
class PurgePreview:
    """Counts of rows that would be removed by purge_keep_latest."""

    validations_to_delete: int
    issues_to_delete: int
    validations_to_keep: int
    issues_to_keep: int


@dataclass
class IssueRow:
    id: int
    project_id: int | None
    block_id: int | None
    detected_at: str
    severity: str
    description: str
    resolved: int


class FindingsDB:
    """Append-only validations and issues; projects/blocks are upserted."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path))
            self._conn.row_factory = sqlite3.Row
            self._conn.executescript(SCHEMA_SQL)
            self._conn.commit()
        return self._conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def upsert_project(
        self,
        *,
        path: str,
        name: str,
        gr_version: str,
        readme_path: str | None,
        scan_status: str,
    ) -> int:
        conn = self.connect()
        now = _utc_now()
        conn.execute(
            """
            INSERT INTO projects (path, name, gr_version, readme_path, last_scanned, scan_status)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                gr_version=excluded.gr_version,
                readme_path=excluded.readme_path,
                last_scanned=excluded.last_scanned,
                scan_status=excluded.scan_status
            """,
            (path, name, gr_version, readme_path, now, scan_status),
        )
        conn.commit()
        row = conn.execute("SELECT id FROM projects WHERE path = ?", (path,)).fetchone()
        assert row is not None
        return int(row["id"])

    def upsert_block(
        self,
        *,
        project_id: int,
        block_name: str,
        block_file: str | None,
        expected_family: str,
        expected_order: str,
        gr_version: str,
    ) -> int:
        conn = self.connect()
        existing = conn.execute(
            """
            SELECT id FROM blocks
            WHERE project_id = ? AND block_name = ?
            """,
            (project_id, block_name),
        ).fetchone()
        if existing is not None:
            block_id = int(existing["id"])
            conn.execute(
                """
                UPDATE blocks SET block_file=?, expected_family=?, expected_order=?, gr_version=?
                WHERE id=?
                """,
                (block_file, expected_family, expected_order, gr_version, block_id),
            )
            conn.commit()
            return block_id
        cur = conn.execute(
            """
            INSERT INTO blocks (
                project_id, block_name, block_file, expected_family, expected_order, gr_version
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (project_id, block_name, block_file, expected_family, expected_order, gr_version),
        )
        conn.commit()
        return int(cur.lastrowid)

    def add_validation(
        self,
        *,
        block_id: int,
        iq_file: str,
        predicted_family: str,
        predicted_order: str,
        family_confidence: float,
        order_confidence: float,
        family_pass: bool,
        order_pass: bool,
        hard_fail: bool,
        notes: str,
    ) -> int:
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT INTO validations (
                block_id, run_at, iq_file, predicted_family, predicted_order,
                family_confidence, order_confidence, family_pass, order_pass, hard_fail, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                block_id,
                _utc_now(),
                iq_file,
                predicted_family,
                predicted_order,
                family_confidence,
                order_confidence,
                int(family_pass),
                int(order_pass),
                int(hard_fail),
                notes,
            ),
        )
        conn.commit()
        return int(cur.lastrowid)

    def add_issue(
        self,
        *,
        project_id: int,
        block_id: int | None,
        severity: str,
        description: str,
    ) -> int:
        conn = self.connect()
        cur = conn.execute(
            """
            INSERT INTO issues (project_id, block_id, detected_at, severity, description, resolved)
            VALUES (?, ?, ?, ?, ?, 0)
            """,
            (project_id, block_id, _utc_now(), severity, description),
        )
        conn.commit()
        return int(cur.lastrowid)

    def supersede_open_issues_for_block(
        self,
        *,
        project_id: int,
        block_id: int,
    ) -> int:
        """Mark unresolved issues for a block resolved before recording a new scan result."""
        conn = self.connect()
        cur = conn.execute(
            """
            UPDATE issues
            SET resolved = 1,
                description = description || ' | superseded by re-scan'
            WHERE project_id = ? AND block_id = ? AND resolved = 0
            """,
            (project_id, block_id),
        )
        conn.commit()
        return int(cur.rowcount)

    def resolve_issue(self, issue_id: int, note: str | None = None) -> bool:
        conn = self.connect()
        desc = conn.execute("SELECT description FROM issues WHERE id = ?", (issue_id,)).fetchone()
        if desc is None:
            return False
        new_desc = str(desc["description"])
        if note:
            new_desc = f"{new_desc} | resolved: {note}"
        conn.execute(
            "UPDATE issues SET resolved = 1, description = ? WHERE id = ?",
            (new_desc, issue_id),
        )
        conn.commit()
        return True

    def list_issues(
        self,
        *,
        project_name: str | None = None,
        severity: str | None = None,
        unresolved_only: bool = True,
        detected_since: str | None = None,
    ) -> list[dict[str, Any]]:
        conn = self.connect()
        query = """
            SELECT i.*, p.name AS project_name, b.block_name
            FROM issues i
            LEFT JOIN projects p ON i.project_id = p.id
            LEFT JOIN blocks b ON i.block_id = b.id
            WHERE 1=1
        """
        params: list[Any] = []
        if unresolved_only:
            query += " AND i.resolved = 0"
        if project_name:
            query += " AND p.name LIKE ?"
            params.append(f"%{project_name}%")
        if severity:
            query += " AND i.severity = ?"
            params.append(severity)
        if detected_since is not None:
            query += " AND i.detected_at >= ?"
            params.append(normalize_scan_timestamp(detected_since))
        query += " ORDER BY i.detected_at DESC"
        rows = conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def preview_purge_keep_latest(self) -> PurgePreview:
        """Count validation and issue rows that purge_keep_latest would delete."""
        conn = self.connect()
        val_delete = conn.execute(
            """
            SELECT COUNT(*) AS c FROM validations v
            JOIN blocks b ON v.block_id = b.id
            WHERE v.id NOT IN (
                SELECT v2.id FROM validations v2
                JOIN (
                    SELECT block_id, MAX(run_at) AS max_run
                    FROM validations
                    GROUP BY block_id
                ) latest ON v2.block_id = latest.block_id AND v2.run_at = latest.max_run
            )
            """
        ).fetchone()
        issue_delete = conn.execute(
            """
            SELECT COUNT(*) AS c FROM issues i
            WHERE (
                i.project_id IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM blocks b
                    JOIN validations v ON v.block_id = b.id
                    WHERE b.project_id = i.project_id
                )
                AND i.detected_at < (
                    SELECT MIN(latest_run) FROM (
                        SELECT MAX(v.run_at) AS latest_run
                        FROM validations v
                        JOIN blocks b ON v.block_id = b.id
                        WHERE b.project_id = i.project_id
                        GROUP BY b.id
                    )
                )
            )
            OR (
                i.project_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM blocks b
                    JOIN validations v ON v.block_id = b.id
                    WHERE b.project_id = i.project_id
                )
                AND i.detected_at < COALESCE(
                    (SELECT last_scanned FROM projects WHERE id = i.project_id),
                    '1970-01-01T00:00:00Z'
                )
            )
            """
        ).fetchone()
        val_keep = conn.execute("SELECT COUNT(*) AS c FROM validations").fetchone()
        issue_keep = conn.execute("SELECT COUNT(*) AS c FROM issues").fetchone()
        to_delete_v = int(val_delete["c"]) if val_delete else 0
        to_delete_i = int(issue_delete["c"]) if issue_delete else 0
        total_v = int(val_keep["c"]) if val_keep else 0
        total_i = int(issue_keep["c"]) if issue_keep else 0
        return PurgePreview(
            validations_to_delete=to_delete_v,
            issues_to_delete=to_delete_i,
            validations_to_keep=total_v - to_delete_v,
            issues_to_keep=total_i - to_delete_i,
        )

    def purge_keep_latest(self) -> PurgePreview:
        """
        Delete validations and issues from older scan runs.

        Keeps the newest validation row per block and issues detected at or after
        the start of that project's latest run. Does not delete projects or blocks.
        """
        preview = self.preview_purge_keep_latest()
        conn = self.connect()
        conn.execute(
            """
            DELETE FROM validations
            WHERE id NOT IN (
                SELECT v.id FROM validations v
                JOIN (
                    SELECT block_id, MAX(run_at) AS max_run
                    FROM validations
                    GROUP BY block_id
                ) latest ON v.block_id = latest.block_id AND v.run_at = latest.max_run
            )
            """
        )
        conn.execute(
            """
            DELETE FROM issues
            WHERE (
                project_id IS NOT NULL
                AND EXISTS (
                    SELECT 1 FROM blocks b
                    JOIN validations v ON v.block_id = b.id
                    WHERE b.project_id = issues.project_id
                )
                AND detected_at < (
                    SELECT MIN(latest_run) FROM (
                        SELECT MAX(v.run_at) AS latest_run
                        FROM validations v
                        JOIN blocks b ON v.block_id = b.id
                        WHERE b.project_id = issues.project_id
                        GROUP BY b.id
                    )
                )
            )
            OR (
                project_id IS NOT NULL
                AND NOT EXISTS (
                    SELECT 1 FROM blocks b
                    JOIN validations v ON v.block_id = b.id
                    WHERE b.project_id = issues.project_id
                )
                AND detected_at < COALESCE(
                    (SELECT last_scanned FROM projects WHERE id = issues.project_id),
                    '1970-01-01T00:00:00Z'
                )
            )
            """
        )
        conn.commit()
        return preview

    def list_projects(self) -> list[dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            "SELECT * FROM projects ORDER BY last_scanned DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def status_summary(self) -> dict[str, Any]:
        conn = self.connect()
        projects = conn.execute("SELECT COUNT(*) AS c FROM projects").fetchone()
        validations = conn.execute("SELECT COUNT(*) AS c FROM validations").fetchone()
        issues = conn.execute(
            """
            SELECT severity, COUNT(*) AS c FROM issues WHERE resolved = 0 GROUP BY severity
            """
        ).fetchall()
        last = conn.execute(
            "SELECT MAX(last_scanned) AS m FROM projects"
        ).fetchone()
        return {
            "projects": int(projects["c"]) if projects else 0,
            "validations": int(validations["c"]) if validations else 0,
            "open_issues": {str(r["severity"]): int(r["c"]) for r in issues},
            "last_scan": last["m"] if last else None,
        }

    def latest_validations_for_project(self, project_name: str) -> list[dict[str, Any]]:
        conn = self.connect()
        rows = conn.execute(
            """
            SELECT v.*, b.block_name, b.expected_family, b.expected_order
            FROM validations v
            JOIN blocks b ON v.block_id = b.id
            JOIN projects p ON b.project_id = p.id
            WHERE p.name = ?
            ORDER BY v.run_at DESC, b.block_name
            """,
            (project_name,),
        ).fetchall()
        latest: dict[str, dict[str, Any]] = {}
        for row in rows:
            name = str(row["block_name"])
            if name not in latest:
                latest[name] = dict(row)
        return sorted(latest.values(), key=lambda r: str(r["block_name"]))

    def count_issues(self, *, unresolved_only: bool = True) -> int:
        conn = self.connect()
        if unresolved_only:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM issues WHERE resolved = 0"
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM issues").fetchone()
        return int(row["c"]) if row else 0
