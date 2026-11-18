"""Unit tests for the structure comparison (no database required).

These lock down each change class the API must treat as "not applicable",
including the cases that cannot easily be produced by ALTER TABLE on the
demo schema (column removal, FK column list change).
"""
from __future__ import annotations

from app.compatibility import capture_snapshot, evaluate
from app.models import (
    Column,
    ForeignKey,
    MatchType,
    TableSchema,
    UniqueConstraint,
)


class _NoDb:
    """Stand-in connection: only introspection helpers are monkeypatched away."""

    def cursor(self):  # pragma: no cover - never reached in these tests
        raise AssertionError("catalog connection must not be used")


def _col(name, pg_type="integer", nullable=False):
    return Column(
        name=name,
        pg_type=pg_type,
        ordinal=1,
        nullable=nullable,
        has_default=False,
        is_identity=False,
        is_generated=False,
    )


def _fk(name, columns, ref_table, ref_columns, match=MatchType.SIMPLE,
        on_update="NO ACTION", on_delete="NO ACTION", initially_deferred=False):
    return ForeignKey(
        name=name,
        columns=list(columns),
        ref_table=ref_table,
        ref_columns=list(ref_columns),
        match=match,
        on_update=on_update,
        on_delete=on_delete,
        not_deferrable=not initially_deferred,
        initially_deferred=initially_deferred,
    )


def _schema(name, columns, uniques=(), fks=()):
    return TableSchema(
        schema_name="public",
        name=name,
        columns=list(columns),
        unique_constraints=list(uniques),
        foreign_keys=list(fks),
    )


def _base_tables():
    parent = _schema(
        "p",
        [_col("id"), _col("v", nullable=True)],
        uniques=[UniqueConstraint(name="p_pk", columns=["id"], is_primary=True)],
    )
    child = _schema(
        "c",
        [_col("id"), _col("pid"), _col("alt", nullable=True)],
        uniques=[UniqueConstraint(name="c_pk", columns=["id"], is_primary=True)],
        fks=[_fk("c_fk", ["pid"], "p", ["id"])],
    )
    return {"p": parent, "c": child}


def _snapshot(tables):
    return capture_snapshot("public", ["c"], ["c", "p"], tables)


def _evaluate(snapshot, current):
    import app.compatibility as compat

    original = compat.introspect_tables
    original_kinds = compat._relation_kinds
    compat._relation_kinds = lambda conn, schema, tables: {t: "r" for t in tables}
    compat.introspect_tables = lambda conn, schema, tables: current
    try:
        return evaluate(snapshot, _NoDb())
    finally:
        compat.introspect_tables = original
        compat._relation_kinds = original_kinds


def test_identical_structure_is_applicable():
    tables = _base_tables()
    result = _evaluate(_snapshot(tables), _base_tables())
    assert result["verdict"].value == "applicable"
    assert result["changes"] == []


def test_removed_column_detected():
    current = _base_tables()
    current["c"] = _schema(
        "c",
        [_col("id"), _col("pid")],
        uniques=[UniqueConstraint(name="c_pk", columns=["id"], is_primary=True)],
        fks=[_fk("c_fk", ["pid"], "p", ["id"])],
    )
    result = _evaluate(_snapshot(_base_tables()), current)
    assert result["verdict"].value == "changed"
    assert any(c.kind == "column_removed" and c.object.endswith("c.alt")
               for c in result["changes"])


def test_fk_local_column_change_detected():
    current = _base_tables()
    current["c"] = _schema(
        "c",
        [_col("id"), _col("pid"), _col("alt", nullable=True)],
        uniques=[UniqueConstraint(name="c_pk", columns=["id"], is_primary=True)],
        fks=[_fk("c_fk", ["alt"], "p", ["id"])],
    )
    result = _evaluate(_snapshot(_base_tables()), current)
    kinds = {(c.kind, c.object) for c in result["changes"]}
    assert ("foreign_key", "public.c.c_fk") in kinds
    details = " ".join(c.detail for c in result["changes"])
    assert "已移除" in details and "新增外键列" in details


def test_fk_referenced_column_change_detected():
    current = _base_tables()
    current["p"] = _schema(
        "p",
        [_col("id"), _col("v", nullable=True)],
        uniques=[
            UniqueConstraint(name="p_pk", columns=["id"], is_primary=True),
            UniqueConstraint(name="p_v_uk", columns=["v"], is_primary=False),
        ],
    )
    current["c"] = _schema(
        "c",
        [_col("id"), _col("pid"), _col("alt", nullable=True)],
        uniques=[UniqueConstraint(name="c_pk", columns=["id"], is_primary=True)],
        fks=[_fk("c_fk", ["pid"], "p", ["v"])],
    )
    result = _evaluate(_snapshot(_base_tables()), current)
    details = " ".join(c.detail for c in result["changes"])
    assert "被引用列" in details


def test_fk_deferrability_change_detected():
    current = _base_tables()
    current["c"] = _schema(
        "c",
        [_col("id"), _col("pid"), _col("alt", nullable=True)],
        uniques=[UniqueConstraint(name="c_pk", columns=["id"], is_primary=True)],
        fks=[_fk("c_fk", ["pid"], "p", ["id"], initially_deferred=True)],
    )
    result = _evaluate(_snapshot(_base_tables()), current)
    details = " ".join(c.detail for c in result["changes"])
    assert "可延迟性" in details


def test_closure_extension_detected():
    current = _base_tables()
    current["c"] = _schema(
        "c",
        [_col("id"), _col("pid"), _col("alt", nullable=True)],
        uniques=[UniqueConstraint(name="c_pk", columns=["id"], is_primary=True)],
        fks=[
            _fk("c_fk", ["pid"], "p", ["id"]),
            _fk("c_alt_fk", ["alt"], "outside", ["id"]),
        ],
    )
    result = _evaluate(_snapshot(_base_tables()), current)
    assert any(c.kind == "closure" for c in result["changes"])
    assert "outside" in result["current_closure"]


def test_missing_table_short_circuits():
    current = {"p": _base_tables()["p"]}

    import app.compatibility as compat

    original_kinds = compat._relation_kinds
    compat._relation_kinds = lambda conn, schema, tables: {"p": "r"}
    try:
        result = evaluate(_snapshot(_base_tables()), _NoDb())
    finally:
        compat._relation_kinds = original_kinds
    assert result["verdict"].value == "table_missing"
    assert any(c.kind == "table_missing" for c in result["changes"])
