"""Tests for GET /jobs/{job_id}/schema-compatibility.

Each test builds a job against its own throw-away source database (a copy of
the demo schema), then mutates that source and asks whether the package is
still applicable. The real isolated verification database is used for job
generation, so the jobs under test really succeeded and published packages.
"""
from __future__ import annotations

import pathlib
import secrets
from dataclasses import replace

import pgserver
import psycopg
import pytest
from fastapi.testclient import TestClient

import app.jobs as jobs_mod
import app.main as main_mod
from app.jobs import STORE, run_job
from app.main import app
from app.models import GenerateRequest, JobStatus

client = TestClient(app)

ROOT = pathlib.Path(__file__).resolve().parent.parent
DEMO_SQL = (ROOT / "sql/00_demo_source_schema.sql").read_text()


def _dsn_with_db(base_dsn: str, dbname: str) -> str:
    info = psycopg.conninfo.conninfo_to_dict(base_dsn)
    info["dbname"] = dbname
    return psycopg.conninfo.make_conninfo(**info)


@pytest.fixture()
def compat_source(source_dsn):
    """Disposable source database containing the demo cyclic schema."""
    name = f"compat_{secrets.token_hex(6)}"
    with psycopg.connect(source_dsn, autocommit=True) as adm:
        adm.execute(f'CREATE DATABASE "{name}"')
    dsn = _dsn_with_db(source_dsn, name)
    with psycopg.connect(dsn, autocommit=True) as conn:
        conn.execute(DEMO_SQL)
    yield dsn
    # Never block on DROP DATABASE in restricted sandboxes: rename aside.
    try:
        with psycopg.connect(source_dsn, autocommit=True) as adm:
            adm.execute(f'ALTER DATABASE "{name}" RENAME TO "retired_{name}"')
    except psycopg.Error:
        pass


def _use_source(monkeypatch, dsn: str) -> None:
    """Point job generation and the API endpoint at the scratch source."""
    monkeypatch.setattr(
        jobs_mod, "settings", replace(jobs_mod.settings, source_dsn=dsn)
    )
    monkeypatch.setattr(
        main_mod, "settings", replace(main_mod.settings, source_dsn=dsn)
    )


def _run_job(dsn: str, monkeypatch, example="request_cyclic_four_tables.json"):
    import json

    raw = json.loads((ROOT / "examples" / example).read_text())
    req = GenerateRequest(**{k: v for k, v in raw.items() if not k.startswith("_")})
    _use_source(monkeypatch, dsn)
    job_id = STORE.create()
    run_job(job_id, req)
    return job_id, STORE.get(job_id)


def _check(job_id: str):
    r = client.get(f"/jobs/{job_id}/schema-compatibility")
    assert r.status_code == 200, r.text
    return r.json()


def _ddl(dsn: str, *statements: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as conn:
        for stmt in statements:
            conn.execute(stmt)


def _kinds(report) -> set[str]:
    return {c["kind"] for c in report["changes"]}


# --------------------------------------------------------------------------
# Applicable / no changes
# --------------------------------------------------------------------------


def test_unchanged_source_is_applicable_and_check_is_read_only(
    compat_source, monkeypatch
):
    job_id, job = _run_job(compat_source, monkeypatch)
    assert job.status == JobStatus.SUCCEEDED

    package_before = dict(STORE.package(job_id))
    status_before = STORE.get(job_id).status
    download_before = client.get(f"/jobs/{job_id}/download")
    assert download_before.status_code == 200

    report = _check(job_id)

    assert report["applicable"] is True
    assert report["verdict"] == "applicable"
    assert report["changes"] == []
    assert report["snapshot"]["selected_tables"] == [
        "dept",
        "employee",
        "project",
        "assignment",
    ]
    assert report["package_modified"] is False
    assert report["download_status_modified"] is False

    # Package bytes, package availability and download state are untouched.
    assert STORE.package(job_id) == package_before
    assert STORE.get(job_id).status == status_before
    assert client.get(f"/jobs/{job_id}/download").status_code == 200


def test_check_works_for_role_without_any_table_privilege(
    compat_source, monkeypatch
):
    """Catalog-only access proves the check never reads table rows."""
    role = f"compat_ro_{secrets.token_hex(4)}"
    _ddl(
        compat_source,
        f"CREATE ROLE {role} LOGIN PASSWORD 'compat_ro_pwd'",
        f"GRANT USAGE ON SCHEMA public TO {role}",
        f"REVOKE ALL ON ALL TABLES IN SCHEMA public FROM {role}",
    )
    job_id, job = _run_job(compat_source, monkeypatch)
    assert job.status == JobStatus.SUCCEEDED

    ro_dsn = psycopg.conninfo.make_conninfo(
        **{
            **psycopg.conninfo.conninfo_to_dict(compat_source),
            "user": role,
            "password": "compat_ro_pwd",
        }
    )
    # The row-read really is denied for this role.
    with psycopg.connect(ro_dsn) as conn:
        with pytest.raises(psycopg.Error):
            conn.execute("SELECT * FROM public.dept")

    _use_source(monkeypatch, ro_dsn)
    report = _check(job_id)
    assert report["verdict"] == "applicable"
    assert report["applicable"] is True


# --------------------------------------------------------------------------
# Column changes
# --------------------------------------------------------------------------


def test_column_type_change_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE employee ALTER COLUMN level TYPE bigint",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert report["verdict"] == "changed"
    assert "column_type" in _kinds(report)
    assert any("employee.level" in c["object"] for c in report["changes"])


def test_column_nullability_change_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE employee ALTER COLUMN mentor_id SET NOT NULL",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert "column_nullability" in _kinds(report)
    assert any("mentor_id" in c["object"] for c in report["changes"])


def test_added_column_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(compat_source, "ALTER TABLE project ADD COLUMN extra integer")
    report = _check(job_id)
    assert report["applicable"] is False
    assert "column_added" in _kinds(report)


# --------------------------------------------------------------------------
# Key changes
# --------------------------------------------------------------------------


def test_primary_key_change_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE assignment DROP CONSTRAINT assignment_pk",
        "ALTER TABLE assignment ADD CONSTRAINT assignment_uk "
        "UNIQUE (emp_id, project_id)",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert "primary_key" in _kinds(report)
    assert "unique_key" in _kinds(report)


def test_unique_key_change_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE employee ADD CONSTRAINT employee_level_uk UNIQUE (level)",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert "unique_key" in _kinds(report)
    assert any("employee_level_uk" in c["object"] for c in report["changes"])


# --------------------------------------------------------------------------
# Foreign-key changes
# --------------------------------------------------------------------------


def test_fk_target_change_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE dept DROP CONSTRAINT dept_parent_fk",
        "ALTER TABLE dept ADD CONSTRAINT dept_parent_fk "
        "FOREIGN KEY (parent_dept) REFERENCES employee (emp_id) MATCH SIMPLE "
        "ON DELETE SET NULL ON UPDATE NO ACTION NOT DEFERRABLE",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert "foreign_key" in _kinds(report)
    assert any("dept_parent_fk" in c["object"] for c in report["changes"])


def test_fk_match_and_action_change_is_not_applicable(
    compat_source, monkeypatch
):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE project DROP CONSTRAINT project_dept_fk",
        "ALTER TABLE project ADD CONSTRAINT project_dept_fk "
        "FOREIGN KEY (dept_id, dept_code) REFERENCES dept (dept_id, dept_code) "
        "MATCH SIMPLE ON DELETE CASCADE ON UPDATE NO ACTION NOT DEFERRABLE",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    details = " ".join(c["detail"] for c in report["changes"])
    assert "MATCH" in details
    assert "ON DELETE" in details


def test_fk_column_change_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE assignment DROP CONSTRAINT assignment_project_fk",
        "ALTER TABLE assignment ADD CONSTRAINT assignment_project_fk "
        "FOREIGN KEY (role_code) REFERENCES project (project_id) "
        "MATCH SIMPLE ON DELETE CASCADE ON UPDATE CASCADE NOT DEFERRABLE",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert "foreign_key" in _kinds(report)
    objects = {c["object"] for c in report["changes"]}
    assert any("assignment_project_fk" in o for o in objects)


def test_new_fk_to_table_outside_closure_is_not_applicable(
    compat_source, monkeypatch
):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "CREATE TABLE region (region_id integer NOT NULL, "
        "CONSTRAINT region_pk PRIMARY KEY (region_id))",
        "ALTER TABLE employee ADD COLUMN region_id integer",
        "ALTER TABLE employee ADD CONSTRAINT employee_region_fk "
        "FOREIGN KEY (region_id) REFERENCES region (region_id) "
        "MATCH SIMPLE ON DELETE NO ACTION ON UPDATE NO ACTION NOT DEFERRABLE",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert "closure" in _kinds(report)
    assert "region" in report["current_closure"]


# --------------------------------------------------------------------------
# Missing / unsupported / unreadable source
# --------------------------------------------------------------------------


def test_dropped_table_reports_table_missing(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(compat_source, "DROP TABLE assignment CASCADE")
    report = _check(job_id)
    assert report["applicable"] is False
    assert report["verdict"] == "table_missing"
    assert "table_missing" in _kinds(report)
    assert any("assignment" in c["object"] for c in report["changes"])


def test_unsupported_structure_is_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _ddl(
        compat_source,
        "ALTER TABLE dept ADD CONSTRAINT dept_positive "
        "CHECK (dept_id > 0)",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert report["verdict"] == "structure_unsupported"
    assert any("CHECK" in u["reason"] for u in report["unsupported"])


def test_unreadable_source_is_unknown_not_applicable(compat_source, monkeypatch):
    job_id, _ = _run_job(compat_source, monkeypatch)
    _use_source(
        monkeypatch,
        "postgresql://postgres@/postgres?host=/tmp/definitely-not-a-pg-socket",
    )
    report = _check(job_id)
    assert report["applicable"] is False
    assert report["verdict"] == "source_unavailable"
    assert report["reason"]


def test_failed_job_never_gets_applicable(compat_source, monkeypatch):
    # Unsatisfiable request: the job fails and publishes nothing.
    job_id, job = _run_job(
        compat_source, monkeypatch, example="request_unsat_conflict.json"
    )
    assert job.status == JobStatus.FAILED
    report = _check(job_id)
    assert report["applicable"] is False
    assert report["verdict"] == "job_not_successful"
    assert client.get(f"/jobs/{job_id}/download").status_code == 409


def test_unknown_job_is_404():
    assert client.get("/jobs/does-not-exist/schema-compatibility").status_code == 404


def test_missing_snapshot_is_not_applicable(compat_source, monkeypatch):
    job_id, job = _run_job(compat_source, monkeypatch)
    assert job.status == JobStatus.SUCCEEDED
    # Simulate snapshot eviction while the job itself is still around.
    STORE._snapshots.pop(job_id, None)
    report = _check(job_id)
    assert report["applicable"] is False
    assert report["verdict"] == "snapshot_missing"
