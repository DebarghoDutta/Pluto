import sys
import os
from datetime import datetime

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def start_session(session_type="perception", status="active", location=None):
    """Opens one session row (used by Brain.start()) and returns its session_id."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO sessions (start_time, session_type, status, location)
            VALUES (%s, %s, %s, %s)
            RETURNING session_id;
            """,
            (datetime.now(), session_type, status, location),
        )
        session_id = cursor.fetchone()[0]
    connection.commit()
    return session_id


def end_session(session_id, status="ended"):
    """Closes a session row (used by Brain.stop())."""
    with connection.cursor() as cursor:
        cursor.execute(
            "UPDATE sessions SET end_time = %s, status = %s WHERE session_id = %s;",
            (datetime.now(), status, session_id),
        )
    connection.commit()


def create_sessions_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id SERIAL PRIMARY KEY,
                start_time TIMESTAMP NOT NULL,
                end_time TIMESTAMP,
                session_type VARCHAR(50),
                status VARCHAR(20),
                location VARCHAR(100)
            );
        """)

    connection.commit()
    print("Sessions table created successfully!")


if __name__ == "__main__":
    create_sessions_table()