import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def insert_face_profile(owner_id, profile_name=None, sample_count=0, quality_score=None,
                         embedding_reference=None):
    """Logs one owner's face profile (sample count from registration/update)."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO face (owner_id, profile_name, embedding_reference, sample_count, quality_score)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING face_id;
            """,
            (owner_id, profile_name, embedding_reference, sample_count, quality_score),
        )
        face_id = cursor.fetchone()[0]
    connection.commit()
    return face_id


def create_face_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS face (
                face_id SERIAL PRIMARY KEY,
                owner_id INTEGER NOT NULL,
                profile_name VARCHAR(100),
                embedding_reference TEXT,
                sample_count INTEGER DEFAULT 0,
                quality_score DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

                FOREIGN KEY (owner_id)
                    REFERENCES owners(owner_id)
            );
        """)

    connection.commit()
    print("Face table created successfully!")


if __name__ == "__main__":
    create_face_table()