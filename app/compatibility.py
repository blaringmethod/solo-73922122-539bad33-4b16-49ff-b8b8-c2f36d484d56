"""Compare the catalog snapshot that produced a package with the live source.

A job's SQL package is generated from one consistent catalog snapshot (see
:func:`app.jobs.run_job`).  This module re-reads the *whole* closure from the
current source inside a single REPEATABLE READ transaction, so existence
checks, introspection and closure inspection also share one snapshot.  Only
``pg_catalog`` is queried; no table row is ever read.

The comparison is deliberately conservative: anything that cannot be proven
identical (missing table, unsupported structure, unreadable source, no
snapshot) is reported as *not applicable*, never as applicable.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import psycopg

from .db import source_snapshot_connection
from .introspect import IntrospectionError, introspect_tables
from .models import (
    CompatibilityReport,
    CompatibilityVerdict,
    GenerationSnapshotInfo,
    JobStatus,
    SchemaChange,
    TableSchema,
    UnsupportedObject,
)

_KIND_LABELS = {
    "column_type": "列类型",
    "column_nullability": "列可空性",
    "column_added": "新增列",
    "column_removed": "删除列",
    "primary_key": "主键",
    "unique_key": "唯一键",
    "foreign_key": "外键",
    "closure": "外键闭包",
    "table_missing": "表缺失",
}


@dataclass(frozen=True)
class GenerationSnapshot:
    """Immutable catalog snapshot captured when the package was generated."""

    captured_at: str
    schema_name: str
    requested_tables: tuple[str, ...]
    selected_tables: tuple[str, ...]
    schemas: dict[str, TableSchema]

    def info(self) -> GenerationSnapshotInfo:
        return GenerationSnapshotInfo(
            captured_at=self.captured_at,
            schema_name=self.schema_name,
            requested_tables=list(self.requested_tables),
            selected_tables=list(self.selected_tables),
        )


def capture_snapshot(
    schema_name: str,
    requested_tables: list[str],
    selected_tables: list[str],
    schemas: dict[str, TableSchema],
) -> GenerationSnapshot:
    """Freeze the exact structures used to render the SQL package."""
    return GenerationSnapshot(
        captured_at=datetime.now(timezone.utc).isoformat(),
        schema_name=schema_name,
        requested_tables=tuple(requested_tables),
        selected_tables=tuple(selected_tables),
        schemas={t: schemas[t].model_copy(deep=True) for t in selected_tables},
    )


def _relation_kinds(conn, schema: str, tables: list[str]) -> dict[str, str]:
    """Return relname -> relkind for the given tables (catalogs only)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.relname, c.relkind
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = %s AND c.relname = ANY(%s)
            """,
            (schema, tables),
        )
        return {name: kind for name, kind in cur.fetchall()}


def _change(kind: str, obj: str, detail: str) -> SchemaChange:
    return SchemaChange(kind=kind, object=obj, detail=detail)


def _compare_columns(
    schema: str, table: str, old: TableSchema, new: TableSchema
) -> list[SchemaChange]:
    changes: list[SchemaChange] = []
    old_cols = {c.name: c for c in old.columns}
    new_cols = {c.name: c for c in new.columns}
    for name in sorted(old_cols):
        obj = f"{schema}.{table}.{name}"
        if name not in new_cols:
            changes.append(_change("column_removed", obj, "生成时存在的列已从源表删除"))
            continue
        if old_cols[name].pg_type != new_cols[name].pg_type:
            changes.append(
                _change(
                    "column_type",
                    obj,
                    f"列类型由 {old_cols[name].pg_type} 变为 {new_cols[name].pg_type}",
                )
            )
        if old_cols[name].nullable != new_cols[name].nullable:
            before = "可空" if old_cols[name].nullable else "NOT NULL"
            after = "可空" if new_cols[name].nullable else "NOT NULL"
            changes.append(
                _change("column_nullability", obj, f"可空性由 {before} 变为 {after}")
            )
    for name in sorted(set(new_cols) - set(old_cols)):
        changes.append(
            _change(
                "column_added",
                f"{schema}.{table}.{name}",
                f"源表新增列 {new_cols[name].pg_type}（生成包中不存在该列）",
            )
        )
    return changes


def _compare_keys(
    schema: str, table: str, old: TableSchema, new: TableSchema
) -> list[SchemaChange]:
    def inventory(ts: TableSchema):
        return {(u.is_primary, tuple(u.columns)): u.name for u in ts.unique_constraints}

    old_map = inventory(old)
    new_map = inventory(new)
    changes: list[SchemaChange] = []
    for key in sorted(set(old_map) - set(new_map), key=lambda k: (k[0], k[1])):
        is_pk, cols = key
        changes.append(
            _change(
                "primary_key" if is_pk else "unique_key",
                f"{schema}.{table}.{old_map[key]}",
                ("主键" if is_pk else "唯一键")
                + f" ({', '.join(cols)}) 已移除或列组成已变化",
            )
        )
    for key in sorted(set(new_map) - set(old_map), key=lambda k: (k[0], k[1])):
        is_pk, cols = key
        changes.append(
            _change(
                "primary_key" if is_pk else "unique_key",
                f"{schema}.{table}.{new_map[key]}",
                ("主键" if is_pk else "唯一键")
                + f" ({', '.join(cols)}) 为生成后新增或列组成已变化",
            )
        )
    return changes


def _fk_signature(fk) -> tuple:
    return (
        fk.ref_table,
        tuple(fk.ref_columns),
        fk.match.value,
        fk.on_update,
        fk.on_delete,
        fk.not_deferrable,
        fk.initially_deferred,
    )


def _compare_fks(
    schema: str, table: str, old: TableSchema, new: TableSchema
) -> list[SchemaChange]:
    """Compare FKs grouped by their local column list.

    Grouping by columns makes a changed local column list surface as an
    explicit removal + addition, while same-column changes are itemised field
    by field: target table, referenced columns, MATCH type, ON UPDATE /
    ON DELETE actions and deferrability.
    """
    changes: list[SchemaChange] = []
    old_groups: dict[tuple[str, ...], list] = {}
    new_groups: dict[tuple[str, ...], list] = {}
    for fk in old.foreign_keys:
        old_groups.setdefault(tuple(fk.columns), []).append(fk)
    for fk in new.foreign_keys:
        new_groups.setdefault(tuple(fk.columns), []).append(fk)

    for cols in sorted(set(old_groups) | set(new_groups)):
        label = f"({', '.join(cols)})"
        remaining_old = list(old_groups.get(cols, []))
        remaining_new = list(new_groups.get(cols, []))

        # Drop unchanged definitions from both sides first.
        for fk in list(remaining_old):
            match = next(
                (n for n in remaining_new if _fk_signature(n) == _fk_signature(fk)),
                None,
            )
            if match is not None:
                remaining_old.remove(fk)
                remaining_new.remove(match)

        # Pair leftovers positionally: same columns, different definition.
        for fk, nfk in zip(remaining_old, remaining_new):
            obj = f"{schema}.{table}.{fk.name}"
            if fk.ref_table != nfk.ref_table:
                changes.append(
                    _change(
                        "foreign_key",
                        obj,
                        f"外键列 {label} 的目标由 {fk.ref_table} 变为 {nfk.ref_table}",
                    )
                )
            if fk.ref_columns != nfk.ref_columns:
                changes.append(
                    _change(
                        "foreign_key",
                        obj,
                        f"外键列 {label} 的被引用列由 ({', '.join(fk.ref_columns)}) "
                        f"变为 ({', '.join(nfk.ref_columns)})",
                    )
                )
            if fk.match != nfk.match:
                changes.append(
                    _change(
                        "foreign_key",
                        obj,
                        f"外键列 {label} 的匹配方式由 {fk.match.value} 变为 {nfk.match.value}",
                    )
                )
            if fk.on_update != nfk.on_update:
                changes.append(
                    _change(
                        "foreign_key",
                        obj,
                        f"外键列 {label} 的 ON UPDATE 由 {fk.on_update} 变为 {nfk.on_update}",
                    )
                )
            if fk.on_delete != nfk.on_delete:
                changes.append(
                    _change(
                        "foreign_key",
                        obj,
                        f"外键列 {label} 的 ON DELETE 由 {fk.on_delete} 变为 {nfk.on_delete}",
                    )
                )
            if (fk.not_deferrable, fk.initially_deferred) != (
                nfk.not_deferrable,
                nfk.initially_deferred,
            ):
                changes.append(
                    _change("foreign_key", obj, f"外键列 {label} 的可延迟性发生变化")
                )

        paired = min(len(remaining_old), len(remaining_new))
        for fk in remaining_old[paired:]:
            changes.append(
                _change(
                    "foreign_key",
                    f"{schema}.{table}.{fk.name}",
                    f"外键列 {label} 在源表中已移除",
                )
            )
        for nfk in remaining_new[paired:]:
            changes.append(
                _change(
                    "foreign_key",
                    f"{schema}.{table}.{nfk.name}",
                    f"源表新增外键列 {label}，引用 {nfk.ref_table}"
                    f"({', '.join(nfk.ref_columns)})，生成包中不存在该约束",
                )
            )
    return changes


def _compare_tables(
    snapshot: GenerationSnapshot, current: dict[str, TableSchema]
) -> list[SchemaChange]:
    schema = snapshot.schema_name
    changes: list[SchemaChange] = []
    for table in snapshot.selected_tables:
        old = snapshot.schemas[table]
        new = current[table]
        changes.extend(_compare_columns(schema, table, old, new))
        changes.extend(_compare_keys(schema, table, old, new))
        changes.extend(_compare_fks(schema, table, old, new))
    return changes


def _closure_changes(
    snapshot: GenerationSnapshot, current: dict[str, TableSchema]
) -> tuple[list[SchemaChange], list[str]]:
    """Flag any current FK that reaches a table outside the package closure."""
    inside = set(snapshot.selected_tables)
    outside: set[str] = set()
    changes: list[SchemaChange] = []
    for table in snapshot.selected_tables:
        for fk in current[table].foreign_keys:
            if fk.ref_table not in inside:
                outside.add(fk.ref_table)
                changes.append(
                    _change(
                        "closure",
                        f"{snapshot.schema_name}.{table}.{fk.name}",
                        f"外键引用闭包外的表 {snapshot.schema_name}.{fk.ref_table}；"
                        f"生成时的闭包为 {sorted(inside)}",
                    )
                )
    closure = list(snapshot.selected_tables) + sorted(outside - set(snapshot.selected_tables))
    return changes, closure


def _dedupe(changes: list[SchemaChange]) -> list[SchemaChange]:
    seen = set()
    out: list[SchemaChange] = []
    for ch in changes:
        key = (ch.kind, ch.object, ch.detail)
        if key in seen:
            continue
        seen.add(key)
        out.append(ch)
    return sorted(out, key=lambda c: (c.object, c.kind, c.detail))


def evaluate(snapshot: GenerationSnapshot, conn) -> dict:
    """Compare the snapshot with the source, inside one caller-owned snapshot.

    Returns a dict with ``verdict``, ``reason``, ``changes``, ``unsupported``
    and ``current_closure``.  Raises only for source read failures, which the
    caller maps to ``source_unavailable``.
    """
    schema = snapshot.schema_name
    tables = list(snapshot.selected_tables)
    kinds = _relation_kinds(conn, schema, tables)

    missing = [t for t in tables if t not in kinds]
    if missing:
        changes = [
            _change(
                "table_missing",
                f"{schema}.{t}",
                "源库中已不存在该表，原生成包无法再对应当前结构",
            )
            for t in missing
        ]
        return {
            "verdict": CompatibilityVerdict.TABLE_MISSING,
            "reason": (
                "源库中已不存在生成时的表："
                + "、".join(f"{schema}.{t}" for t in missing)
                + "；不能判定为适用，请重新生成。"
            ),
            "changes": _dedupe(changes),
            "unsupported": [],
            "current_closure": [t for t in tables if t in kinds],
        }

    try:
        current = introspect_tables(conn, schema, tables)
    except IntrospectionError as exc:
        unsupported = [UnsupportedObject(object=o, reason=r) for o, r in exc.rejections]
        return {
            "verdict": CompatibilityVerdict.STRUCTURE_UNSUPPORTED,
            "reason": (
                "当前源表结构包含不受支持的对象，无法证明原生成包仍适用："
                + "；".join(f"{o}: {r}" for o, r in exc.rejections)
                + "。请修正结构后重新生成。"
            ),
            "changes": [],
            "unsupported": unsupported,
            "current_closure": tables,
        }

    changes = _compare_tables(snapshot, current)
    closure_changes, closure = _closure_changes(snapshot, current)
    changes = _dedupe(changes + closure_changes)

    if changes:
        kinds_hit = sorted({c.kind for c in changes})
        summary = "、".join(_KIND_LABELS.get(k, k) for k in kinds_hit)
        return {
            "verdict": CompatibilityVerdict.CHANGED,
            "reason": (
                f"对照生成时的结构快照检测到 {len(changes)} 处变化（{summary}），"
                "原 SQL 包不再与当前源表结构一致，建议重新生成后使用。"
            ),
            "changes": changes,
            "unsupported": [],
            "current_closure": closure,
        }

    return {
        "verdict": CompatibilityVerdict.APPLICABLE,
        "reason": (
            "当前源表结构与生成 SQL 包时的目录快照一致（列类型/可空性、主键/唯一键、"
            "外键列/目标/匹配方式/更新删除动作及闭包均未变化），原样本仍适用。"
        ),
        "changes": [],
        "unsupported": [],
        "current_closure": closure,
    }


def build_report(
    job_id: str,
    job_status: str,
    snapshot: GenerationSnapshot | None,
    result: dict,
    checked_at: str | None = None,
) -> CompatibilityReport:
    return CompatibilityReport(
        job_id=job_id,
        job_status=job_status,
        applicable=result["verdict"] == CompatibilityVerdict.APPLICABLE,
        verdict=result["verdict"],
        reason=result["reason"],
        checked_at=checked_at or datetime.now(timezone.utc).isoformat(),
        schema_name=snapshot.schema_name if snapshot else "",
        snapshot=snapshot.info() if snapshot else None,
        current_closure=result.get("current_closure"),
        changes=result.get("changes", []),
        unsupported=result.get("unsupported", []),
    )


def check_compatibility(job, snapshot: GenerationSnapshot | None, dsn: str) -> CompatibilityReport:
    """Produce a compatibility report for one job; never mutates job state.

    Ordering of the guards matters: a job that did not succeed can never get
    an "applicable" answer even if the source happens to look unchanged, and
    an unreadable source is reported as unknown rather than applicable.
    """
    if job.status != JobStatus.SUCCEEDED:
        return build_report(
            job.job_id,
            job.status.value,
            snapshot,
            {
                "verdict": CompatibilityVerdict.JOB_NOT_SUCCESSFUL,
                "reason": (
                    f"作业状态为 {job.status.value}，未成功生成并验证 SQL 包，"
                    "不能给出适用结论，也没有可沿用的检验样本。"
                ),
                "current_closure": None,
            },
        )

    if snapshot is None:
        return build_report(
            job.job_id,
            job.status.value,
            None,
            {
                "verdict": CompatibilityVerdict.SNAPSHOT_MISSING,
                "reason": (
                    "找不到该作业生成时的结构快照（可能已过期回收），"
                    "无法对照当前源库证明包仍适用；不能判定为适用。"
                ),
                "current_closure": None,
            },
        )

    try:
        # A single REPEATABLE READ, read-only transaction: existence checks,
        # strict introspection and the closure inspection all see one snapshot
        # of the catalog. Only pg_catalog is queried; no row data is read.
        with source_snapshot_connection(dsn) as conn:
            result = evaluate(snapshot, conn)
    except psycopg.Error as exc:  # connect / statement / timeout failures
        return build_report(
            job.job_id,
            job.status.value,
            snapshot,
            {
                "verdict": CompatibilityVerdict.SOURCE_UNAVAILABLE,
                "reason": (
                    "读取当前源库目录失败，无法确认结构是否仍适用（未知，不判为适用）："
                    f"{exc.__class__.__name__}: {exc}"
                ),
                "current_closure": None,
            },
        )

    return build_report(job.job_id, job.status.value, snapshot, result)
