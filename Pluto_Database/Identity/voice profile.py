import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def insert_voice_profile(owner_id, profile_name=None, sample_count=0, quality_score=None,
                          embedding_reference=None):
    """Logs one owner's voice profile (sample count from registration/update)."""
    with connection.cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO voice (owner_id, profile_name, sample_count, quality_score, embedding_reference)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING voice_id;
            """,
            (owner_id, profile_name, sample_count, quality_score, embedding_reference),
        )
        voice_id = cursor.fetchone()[0]
    connection.commit()
    return voice_id


def create_voice_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS voice (
                voice_id SERIAL PRIMARY KEY,
                owner_id INTEGER NOT NULL,
                profile_name VARCHAR(100),
                sample_count INTEGER DEFAULT 0,
                quality_score DECIMAL(5,2),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                embedding_reference TEXT,

                FOREIGN KEY (owner_id)
                    REFERENCES owners(owner_id)
            );
        """)

    connection.commit()
    print("Voice table created successfully!")


if __name__ == "__main__":
    create_voice_table()