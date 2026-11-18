"""The catalog snapshot connection must expose one fixed, read-only view."""
from __future__ import annotations

import secrets

import psycopg
import pytest

from app.db import source_snapshot_connection


def _dsn_with_db(base_dsn: str, dbname: str) -> str:
    info = psycopg.conninfo.conninfo_to_dict(base_dsn)
    info["dbname"] = dbname
    return psycopg.conninfo.make_conninfo(**info)


@pytest.fixture()
def scratch_source(source_dsn):
    name = f"snap_{secrets.token_hex(6)}"
    with psycopg.connect(source_dsn, autocommit=True) as adm:
        adm.execute(f'CREATE DATABASE "{name}"')
    dsn = _dsn_with_db(source_dsn, name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(
            "CREATE TABLE snap_t (id integer PRIMARY KEY, v integer)"
        )
    yield dsn
    try:
        with psycopg.connect(source_dsn, autocommit=True) as adm:
            adm.execute(f'ALTER DATABASE "{name}" RENAME TO "retired_{name}"')
    except psycopg.Error:
        pass


def _columns(conn, table: str) -> list[str]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT a.attname
            FROM pg_catalog.pg_attribute a
            JOIN pg_catalog.pg_class c ON c.oid = a.attrelid
            JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relname = %s
              AND a.attnum > 0 AND NOT a.attisdropped
            ORDER BY a.attnum
            """,
            (table,),
        )
        return [r[0] for r in cur.fetchall()]


def test_snapshot_is_stable_across_concurrent_ddl(scratch_source):
    with source_snapshot_connection(scratch_source) as conn:
        before = _columns(conn, "snap_t")
        with psycopg.connect(scratch_source, autocommit=True) as ddl:
            ddl.execute("ALTER TABLE snap_t ADD COLUMN extra integer")
        # Same repeatable-read snapshot: the new column must not appear.
        after = _columns(conn, "snap_t")
    assert before == ["id", "v"]
    assert after == before

    # A fresh snapshot sees the new structure.
    with source_snapshot_connection(scratch_source) as conn:
        assert _columns(conn, "snap_t") == ["id", "v", "extra"]


def test_snapshot_connection_is_read_only(scratch_source):
    with source_snapshot_connection(scratch_source) as conn:
        with pytest.raises(psycopg.Error) as exc:
            conn.execute("INSERT INTO snap_t VALUES (1, 1)")
        assert exc.value.sqlstate == "25006"  # read_only_sql_transaction
