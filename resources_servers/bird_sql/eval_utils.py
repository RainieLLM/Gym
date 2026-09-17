# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Execution-based evaluation utilities for BIRD text-to-SQL.

Mirrors the official BIRD eval (DAMO-ConvAI): execute both predicted and
ground-truth SQL against the per-db_id SQLite file, compare result sets via
``set(predicted) == set(ground_truth)``.

SQL execution uses ``asyncio.to_thread`` rather than Ray remote tasks.
When this resource server runs alongside a Ray-coordinated multi-node DP
vLLM (``gym env start`` attaches to the same Ray cluster), the vLLM engines
consume the Ray slots and ``@ray.remote`` sqlite tasks sit in the scheduler
queue past the per-query timeout, get cancelled, and every rollout reports
``gold_execution_error``. SQLite queries are fast and self-contained — no
reason to cross process boundaries; run them in the asyncio event loop's
default thread pool under a semaphore for bounded concurrency.
"""

import asyncio
import sqlite3
from pathlib import Path
from typing import Any, Literal, NamedTuple, Optional


ResultRow = tuple[Any, ...]
ResultSet = list[ResultRow]


class SqlExecutionResult(NamedTuple):
    """Rows or a safe error code for one SQLite query."""

    rows: Optional[ResultSet]
    error: Optional[Literal["sql_error", "timeout"]] = None


def execute_sqlite(db_path: Path, sql: str) -> SqlExecutionResult:
    """Execute SQL against a SQLite database file and return rows or an error.

    The result does not include SQLite error text, which may expose query data.
    """
    try:
        with sqlite3.connect(str(db_path)) as conn:
            conn.text_factory = lambda b: b.decode(errors="ignore")
            cur = conn.cursor()
            cur.execute(sql)
            return SqlExecutionResult(cur.fetchall())
    except sqlite3.Error:
        return SqlExecutionResult(None, "sql_error")


async def execute_sqlite_async(
    db_path: Path,
    sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> SqlExecutionResult:
    """Execute SQL asynchronously in a worker thread, bounded by semaphore.

    A timeout has a distinct result from a SQLite error.
    """
    async with semaphore:
        try:
            return await asyncio.wait_for(asyncio.to_thread(execute_sqlite, db_path, sql), timeout=timeout_s)
        except asyncio.TimeoutError:
            return SqlExecutionResult(None, "timeout")


def result_sets_match(gold: ResultSet, pred: ResultSet) -> bool:
    """BIRD's result-set comparison: unordered set equality over tuple rows."""
    try:
        return set(gold) == set(pred)
    except TypeError:
        # Rows may contain unhashable types (bytes, lists) — fall back to sorted lists.
        try:
            return sorted(map(repr, gold)) == sorted(map(repr, pred))
        except Exception:
            return False


async def execute_and_compare(
    db_path: Path,
    gold_sql: str,
    pred_sql: str,
    semaphore: asyncio.Semaphore,
    timeout_s: float = 30.0,
) -> tuple[bool, Optional[ResultSet], Optional[ResultSet], Optional[str]]:
    """Execute both queries and compare. Returns (match, gold, pred, error_tag)."""
    gold = await execute_sqlite_async(db_path, gold_sql, semaphore, timeout_s)
    if gold.error:
        return False, None, None, "gold_sql_timeout" if gold.error == "timeout" else "gold_sql_error"
    if gold.rows is None:
        raise ValueError("Reference SQL returned neither rows nor an error")

    pred = await execute_sqlite_async(db_path, pred_sql, semaphore, timeout_s)
    if pred.error:
        return False, gold.rows, None, "pred_sql_timeout" if pred.error == "timeout" else "pred_sql_error"
    if pred.rows is None:
        raise ValueError("Predicted SQL returned neither rows nor an error")

    return result_sets_match(gold.rows, pred.rows), gold.rows, pred.rows, None
