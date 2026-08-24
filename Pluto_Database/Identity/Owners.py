import psycopg
import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def get_owner_by_name(name):
    """Recall an owner row by name, or None if this owner hasn't been seen
    in this table before."""
    with connection.cursor() as cursor:
        cursor.execute(
            "SELECT owner_id, name, role, status, last_seen FROM owners WHERE name = %s;",
            (name,),
        )
        row = cursor.fetchone()
    if not row:
        return None
    return {"owner_id": row[0], "name": row[1], "role": row[2], "status": row[3], "last_seen": row[4]}


def get_or_create_owner_row(name, role="owner"):
    """Used by face recognition: recall the owner row for a recognized name,
    creating it on first sighting so Postgres always has a row to recall
    from afterwards."""
    existing = get_owner_by_name(name)
    if existing:
        return existing
    with connection.cursor() as cursor:
        cursor.execute(
            "INSERT INTO owners (name, role, status) VALUES (%s, %s, 'Active') RETURNING owner_id;",
            (name, role),
        )
        owner_id = cursor.fetchone()[0]
    connection.commit()
    return {"owner_id": owner_id, "name": name, "role": role, "status": "Active", "last_seen": None}


def touch_owner_last_seen(owner_id):
    """Stamps last_seen the moment the owner's face is recognized."""
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE owners SET last_seen = %s, updated_at = %s WHERE owner_id = %s;",
            (datetime.now(), datetime.now(), owner_id),
        )
    connection.commit()


def set_owner_status(owner_id, status):
    """Used on owner deletion (status='Deleted') or other lifecycle changes.
    Postgres keeps the row (for event_identity/face/voice history) rather
    than hard-deleting it."""
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE owners SET status = %s, updated_at = %s WHERE owner_id = %s;",
            (status, datetime.now(), owner_id),
        )
    connection.commit()


def create_owners_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS owners (
                owner_id SERIAL PRIMARY KEY,
                name VARCHAR(100) NOT NULL,
                role VARCHAR(50),
                status VARCHAR(20) DEFAULT 'Active',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_seen TIMESTAMP
            );
        """)

    connection.commit()
    print("Owners table created successfully!")


if __name__ == "__main__":
    create_owners_table()