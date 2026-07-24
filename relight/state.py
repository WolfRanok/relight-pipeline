"""Relight 流水线 SQLite 状态，保证付费任务可安全续跑。"""

from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ACTIVE_STAGES = {"pending", "selected"}
TERMINAL_STAGES = {"completed", "skipped", "failed"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class RelightState:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relight_items (
                item_id TEXT PRIMARY KEY,
                source_path TEXT NOT NULL,
                output_path TEXT NOT NULL,
                stage TEXT NOT NULL DEFAULT 'pending',
                source_sha256 TEXT,
                width INTEGER,
                height INTEGER,
                image_format TEXT,
                aspect_ratio TEXT,
                selection_json TEXT,
                vl_attempts INTEGER NOT NULL DEFAULT 0,
                generation_attempts INTEGER NOT NULL DEFAULT 0,
                upload_url TEXT,
                oss_input_key TEXT,
                oss_output_prefix TEXT,
                business_id TEXT,
                submission_started INTEGER NOT NULL DEFAULT 0,
                task_id TEXT,
                result_json TEXT,
                error TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        # 旧运行目录自动补齐图片元数据列，不移动或重建用户已有状态库。
        columns = {
            str(row["name"])
            for row in self.connection.execute("PRAGMA table_info(relight_items)")
        }
        for name, data_type in (
            ("width", "INTEGER"),
            ("height", "INTEGER"),
            ("image_format", "TEXT"),
            ("aspect_ratio", "TEXT"),
            ("oss_input_key", "TEXT"),
            ("oss_output_prefix", "TEXT"),
            ("submission_started", "INTEGER NOT NULL DEFAULT 0"),
        ):
            if name not in columns:
                self.connection.execute(
                    f"ALTER TABLE relight_items ADD COLUMN {name} {data_type}"
                )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relight_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                category TEXT NOT NULL,
                item_id TEXT,
                message TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS relight_metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def run_id(self) -> str:
        """返回持久化运行UUID；同目录续跑稳定，目录删除重建后不会复用旧ID。"""

        row = self.connection.execute(
            "SELECT value FROM relight_metadata WHERE key='run_id'"
        ).fetchone()
        if row is not None:
            return str(row["value"])
        value = uuid.uuid4().hex
        self.connection.execute(
            "INSERT INTO relight_metadata(key,value) VALUES('run_id',?)", (value,)
        )
        self.connection.commit()
        return value

    def recover_invalid_response_failures(self) -> int:
        """恢复旧版本因HTTP 200异常JSON而误判失败的可查询生图任务。"""

        cursor = self.connection.execute(
            "UPDATE relight_items SET stage='selected',generation_attempts=0,"
            "submission_started=1,result_json=NULL,error=NULL,updated_at=? "
            "WHERE stage='failed' AND selection_json IS NOT NULL "
            "AND business_id IS NOT NULL "
            "AND error LIKE '%HTTP 200返回非JSON响应%'",
            (_now(),),
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def recover_oss_cache_miss_failures(self) -> int:
        """恢复旧版本把OSS首次HEAD 404误判为永久失败的未提交任务。"""

        cursor = self.connection.execute(
            "UPDATE relight_items SET stage='selected',generation_attempts=0,"
            "result_json=NULL,error=NULL,updated_at=? "
            "WHERE stage='failed' AND selection_json IS NOT NULL "
            "AND task_id IS NULL AND submission_started=0 "
            "AND oss_input_key IS NULL AND error LIKE '%OSS input upload failed%' "
            "AND (error LIKE '%Http Status Code: 404%' "
            "OR error LIKE '%NoSuchKey%')",
            (_now(),),
        )
        self.connection.commit()
        return int(cursor.rowcount)

    def add_items(self, items: Iterable[tuple[str, str, str]]) -> None:
        self.connection.executemany(
            "INSERT OR IGNORE INTO relight_items"
            "(item_id,source_path,output_path,updated_at) VALUES(?,?,?,?)",
            ((item_id, source, output, _now()) for item_id, source, output in items),
        )
        self.connection.commit()

    def get(self, item_id: str) -> dict[str, Any]:
        row = self.connection.execute(
            "SELECT * FROM relight_items WHERE item_id=?", (item_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"Relight任务不存在：{item_id}")
        return dict(row)

    def rows(self, stages: set[str] | None = None) -> list[dict[str, Any]]:
        if not stages:
            cursor = self.connection.execute(
                "SELECT * FROM relight_items ORDER BY item_id"
            )
        else:
            placeholders = ",".join("?" for _ in stages)
            cursor = self.connection.execute(
                f"SELECT * FROM relight_items WHERE stage IN ({placeholders}) "
                "ORDER BY item_id",
                sorted(stages),
            )
        return [dict(row) for row in cursor]

    def active(self, limit: int) -> list[dict[str, Any]]:
        # 优先恢复已经完成VL选图的付费生图阶段，再审核新图。
        cursor = self.connection.execute(
            "SELECT * FROM relight_items WHERE stage IN ('selected','pending') "
            "ORDER BY CASE stage WHEN 'selected' THEN 0 ELSE 1 END,item_id LIMIT ?",
            (limit,),
        )
        return [dict(row) for row in cursor]

    def update(self, item_id: str, stage: str | None = None, **fields: Any) -> None:
        if stage is not None:
            fields["stage"] = stage
        fields["updated_at"] = _now()
        assignments = ",".join(f"{name}=?" for name in fields)
        values = [fields[name] for name in fields]
        self.connection.execute(
            f"UPDATE relight_items SET {assignments} WHERE item_id=?",
            [*values, item_id],
        )
        self.connection.commit()

    def increment_attempt(self, item_id: str, field: str) -> int:
        if field not in {"vl_attempts", "generation_attempts"}:
            raise ValueError(f"非法尝试字段：{field}")
        self.connection.execute(
            f"UPDATE relight_items SET {field}={field}+1,error=NULL,updated_at=? "
            "WHERE item_id=?",
            (_now(), item_id),
        )
        self.connection.commit()
        return int(self.get(item_id)[field])

    def save_json(
        self,
        item_id: str,
        field: str,
        payload: dict[str, Any],
        *,
        stage: str | None = None,
        **fields: Any,
    ) -> None:
        if field not in {"selection_json", "result_json"}:
            raise ValueError(f"非法JSON字段：{field}")
        fields[field] = json.dumps(payload, ensure_ascii=False)
        self.update(item_id, stage=stage, **fields)

    def counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT stage,COUNT(*) AS count FROM relight_items GROUP BY stage"
        ).fetchall()
        return {str(row["stage"]): int(row["count"]) for row in rows}

    def completed_count(self) -> int:
        row = self.connection.execute(
            "SELECT COUNT(*) AS count FROM relight_items WHERE stage='completed'"
        ).fetchone()
        return int(row["count"])

    def decrement_attempt(self, item_id: str, field: str) -> None:
        """熔断或远端仍在运行时撤销本次计数，避免耗尽单图重试额度。"""

        if field not in {"vl_attempts", "generation_attempts"}:
            raise ValueError(f"非法尝试字段：{field}")
        self.connection.execute(
            f"UPDATE relight_items SET {field}=MAX({field}-1,0),updated_at=? "
            "WHERE item_id=?",
            (_now(), item_id),
        )
        self.connection.commit()

    def add_event(
        self, event_type: str, category: str, message: str, item_id: str | None
    ) -> None:
        """持久化运行级事件；历史熔断不会自动阻止下一次续跑。"""

        self.connection.execute(
            "INSERT INTO relight_events(event_type,category,item_id,message,created_at) "
            "VALUES(?,?,?,?,?)",
            (event_type, category, item_id, message[:2000], _now()),
        )
        self.connection.commit()

    def events(self) -> list[dict[str, Any]]:
        rows = self.connection.execute(
            "SELECT * FROM relight_events ORDER BY id"
        ).fetchall()
        return [dict(row) for row in rows]
