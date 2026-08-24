import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def insert_scene_observation(session_id, description, scene_id=None, location_entity_id=None,
                              primary_entity_id=None, confidence=None):
    """Logs one NLP-generated scene sentence (SceneNarrator output)."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO scene_observation
                (session_id, scene_id, location_entity_id, primary_entity_id, description, confidence)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING scene_observation_id;
            """,
            (session_id, scene_id, location_entity_id, primary_entity_id, description, confidence),
        )
        scene_observation_id = cursor.fetchone()[0]
    connection.commit()
    return scene_observation_id


def create_scene_observation_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scene_observation (
                scene_observation_id SERIAL PRIMARY KEY,
                session_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                scene_id INTEGER,
                location_entity_id INTEGER,
                primary_entity_id INTEGER,
                description TEXT,
                confidence DECIMAL(5,2),

                FOREIGN KEY (session_id)
                    REFERENCES sessions(session_id)
            );
        """)

    connection.commit()
    print("Scene Observation table created successfully!")


if __name__ == "__main__":
    create_scene_observation_table()