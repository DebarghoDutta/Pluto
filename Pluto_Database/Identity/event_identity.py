import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def insert_event_identity(session_id, source, identity_type, identity_owner, confidence, result):
    """Logs one face-recognition outcome (matched owner or unrecognized)."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO event_identity
                (source, identity_type, identity_owner, confidence, result, session_id)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING event_id;
            """,
            (source, identity_type, identity_owner, confidence, result, session_id),
        )
        event_id = cursor.fetchone()[0]
    connection.commit()
    return event_id


def create_event_identity_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS event_identity (
                event_id SERIAL PRIMARY KEY,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(10),
                identity_type VARCHAR(20),
                identity_owner VARCHAR(100),
                confidence DECIMAL(5,2),
                result VARCHAR(100),
                session_id INTEGER NOT NULL,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            );
        """)

    connection.commit()
    print("Event Identity table created successfully!")


if __name__ == "__main__":
    create_event_identity_table()