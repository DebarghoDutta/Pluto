import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def create_visual_obs_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS visual_obs (
                obs_id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                source VARCHAR(100),
                entry_id INTEGER,
                confidence DECIMAL(5,2),
                location_data TEXT,

                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            );
        """)

    connection.commit()
    print("Visual Observation table created successfully!")


if __name__ == "__main__":
    create_visual_obs_table()