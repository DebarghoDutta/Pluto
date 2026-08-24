"""
database.py
===========
Low-level SQLite persistence layer for Pluto's registered owners.

This module ONLY performs raw SQL operations (create table, insert, select,
update, delete). It has no awareness of validation rules, file handling, or
any higher-level application logic -- that responsibility belongs entirely to
owner_manager.py. Keeping this separation means database.py stays reusable
even if the business rules around "what makes a valid owner" change later.

Target platform: Raspberry Pi 5, Ubuntu. Uses Python's built-in sqlite3,
so no extra dependency is required.
"""

import os
import sqlite3
import threading

DB_PATH = os.path.join(os.path.dirname(__file__), "pluto.db")

_lock = threading.Lock()


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the owners table if it doesn't already exist."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS owners (
                owner_id        TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                dob             TEXT,
                face_dir        TEXT,
                voice_dir       TEXT,
                face_files      TEXT,   -- comma-separated filenames
                voice_files     TEXT,   -- comma-separated filenames
                registered_at   TEXT NOT NULL,
                updated_at      TEXT,
                settings_json   TEXT    -- reserved for future per-owner config
            )
            """
        )
        conn.commit()


def insert_owner(owner_row: dict):
    """Insert a new owner row. Expects keys matching the table schema."""
    with _lock, _connect() as conn:
        conn.execute(
            """
            INSERT INTO owners
                (owner_id, name, dob, face_dir, voice_dir,
                 face_files, voice_files, registered_at, updated_at, settings_json)
            VALUES
                (:owner_id, :name, :dob, :face_dir, :voice_dir,
                 :face_files, :voice_files, :registered_at, :updated_at, :settings_json)
            """,
            owner_row,
        )
        conn.commit()


def update_owner(owner_id: str, fields: dict):
    """Update arbitrary columns for a given owner_id. `fields` is column->value."""
    if not fields:
        return
    with _lock, _connect() as conn:
        set_clause = ", ".join(f"{col} = ?" for col in fields.keys())
        values = list(fields.values()) + [owner_id]
        conn.execute(f"UPDATE owners SET {set_clause} WHERE owner_id = ?", values)
        conn.commit()


def delete_owner(owner_id: str):
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM owners WHERE owner_id = ?", (owner_id,))
        conn.commit()


def get_owner(owner_id: str):
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM owners WHERE owner_id = ?", (owner_id,)
        ).fetchone()
        return dict(row) if row else None


def get_owner_by_name(name: str):
    with _lock, _connect() as conn:
        row = conn.execute(
            "SELECT * FROM owners WHERE name = ?", (name,)
        ).fetchone()
        return dict(row) if row else None


def get_all_owners():
    with _lock, _connect() as conn:
        rows = conn.execute("SELECT * FROM owners ORDER BY registered_at").fetchall()
        return [dict(r) for r in rows]


# Ensure the table exists as soon as this module is imported.
init_db()