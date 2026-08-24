import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def insert_visual_obs(session_id, source, entity_id, confidence, location_data):
    """Logs one object-detection result (YOLO class + position)."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO visual_obs
                (session_id, source, entity_id, confidence, location_data)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING obs_id;
            """,
            (session_id, source, entity_id, confidence, location_data),
        )
        obs_id = cursor.fetchone()[0]
    connection.commit()
    return obs_id


def create_visual_obs_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visual_obs (
                obs_id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(100),
                entity_id TEXT,
                confidence DECIMAL(5,2),
                location_data VARCHAR(100),

                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            );
        """)

    connection.commit()
    print("Visual Observation table created successfully!")


if __name__ == "__main__":
    create_visual_obs_table()