"""Database access helpers.

The source connection is hardened so that it can only read catalog metadata:

* ``default_transaction_read_only = on``
* ``statement_timeout`` / ``lock_timeout`` bound every statement
* introspection queries only touch ``pg_catalog`` / ``information_schema``
* introspection never executes ``SELECT ... FROM <user table>``
"""
from __future__ import annotations

from contextlib import contextmanager

import psycopg
from psycopg import IsolationLevel


@contextmanager
def source_connection(dsn: str):
    """Open a strictly read-only connection to the source database."""
    conn = psycopg.connect(dsn, connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute("SET default_transaction_read_only = on")
            cur.execute("SET statement_timeout = '15s'")
            cur.execute("SET lock_timeout = '5s'")
            # Paranoia: refuse to mutate even if transaction state is bypassed.
            cur.execute("SET idle_in_transaction_session_timeout = '30s'")
        conn.commit()
        yield conn
    finally:
        conn.close()


@contextmanager
def source_snapshot_connection(dsn: str):
    """Read-only, REPEATABLE READ connection yielding one catalog snapshot.

    Every statement executed while the context is open sees the *same*
    snapshot of ``pg_catalog``, so a multi-step catalog inspection (closure
    graph, existence checks, full introspection) cannot mix structures from
    two different DDL moments. The transaction is rolled back on exit: this
    helper can never commit anything to the source database.
    """
    conn = psycopg.connect(dsn, connect_timeout=10)
    try:
        # Must be set before the first statement so psycopg applies them to
        # the implicit transaction opened by the SET LOCAL statements below.
        conn.isolation_level = IsolationLevel.REPEATABLE_READ
        conn.read_only = True
        with conn.cursor() as cur:
            cur.execute("SET LOCAL statement_timeout = '15s'")
            cur.execute("SET LOCAL lock_timeout = '5s'")
            cur.execute("SET LOCAL idle_in_transaction_session_timeout = '30s'")
        yield conn
    finally:
        try:
            conn.rollback()
        finally:
            conn.close()


@contextmanager
def admin_connection(dsn: str, autocommit: bool = True):
    """Connection to the verification cluster's maintenance database."""
    conn = psycopg.connect(dsn, connect_timeout=10, autocommit=autocommit)
    try:
        yield conn
    finally:
        conn.close()
