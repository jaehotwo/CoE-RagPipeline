"""Generic job status tracking backed by Redis.

This module replaces the former ITSD-specific job status store with a reusable
implementation that can be used by any long running ingestion task.
"""

from __future__ import annotations

import os
import time
import uuid
from typing import Any, Dict, Optional

import redis


def _redis_client() -> redis.Redis:
    """Create a Redis client using environment configuration."""
    host = os.getenv("REDIS_HOST", "localhost")
    port = int(os.getenv("REDIS_PORT", "6379"))
    password = os.getenv("REDIS_PASSWORD")
    db = int(os.getenv("REDIS_AUTH_DB", "0"))
    return redis.Redis(host=host, port=port, password=password, db=db, decode_responses=True)


class JobStatusStore:
    """Lightweight job status storage backed by Redis hashes.

    Keys follow the pattern `job:{job_id}` and store status metadata as a hash.
    """

    def __init__(self) -> None:
        self._redis = _redis_client()

    def _key(self, job_id: str) -> str:
        return f"job:{job_id}"

    def create_job(self, task: str, filename: Optional[str] = None) -> Dict[str, Any]:
        job_id = uuid.uuid4().hex
        now = str(int(time.time()))
        key = self._key(job_id)
        self._redis.hset(
            key,
            mapping={
                "status": "queued",
                "task": task,
                "filename": filename or "",
                "created_at": now,
                "updated_at": now,
            },
        )
        # Expire after 24h by default (overridable via env)
        ttl = int(os.getenv("JOB_STATUS_TTL_SECONDS", "86400"))
        if ttl > 0:
            self._redis.expire(key, ttl)
        return {"job_id": job_id, "status": "queued"}

    def start_job(self, job_id: str) -> None:
        now = str(int(time.time()))
        self._redis.hset(self._key(job_id), mapping={"status": "running", "updated_at": now})

    def complete_job(self, job_id: str, result: Optional[Dict[str, Any]] = None) -> None:
        import json

        now = str(int(time.time()))
        mapping: Dict[str, Any] = {"status": "completed", "updated_at": now}
        if result is not None:
            mapping["result_json"] = json.dumps(result, ensure_ascii=False)
        self._redis.hset(self._key(job_id), mapping=mapping)

    def fail_job(self, job_id: str, error: str) -> None:
        now = str(int(time.time()))
        self._redis.hset(
            self._key(job_id),
            mapping={"status": "failed", "error": error, "updated_at": now},
        )

    def set_progress(self, job_id: str, progress: float | int, stage: Optional[str] = None) -> None:
        """Update progress (0-100) and optional stage description for a job."""
        try:
            pct = float(progress)
        except Exception:
            pct = 0.0
        pct = max(0.0, min(100.0, pct))
        now = str(int(time.time()))
        mapping: Dict[str, Any] = {"progress": str(int(pct)), "updated_at": now}
        if stage is not None:
            mapping["stage"] = stage
        self._redis.hset(self._key(job_id), mapping=mapping)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        import json

        data = self._redis.hgetall(self._key(job_id))
        if not data:
            return None
        output: Dict[str, Any] = dict(data)
        if "result_json" in output:
            try:
                output["result"] = (
                    json.loads(output["result_json"]) if output.get("result_json") else None
                )
            except Exception:
                output["result"] = None
        return output

