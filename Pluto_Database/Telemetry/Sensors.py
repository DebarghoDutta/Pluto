import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def create_sensors_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensors (
                sensor_id SERIAL PRIMARY KEY,
                sensor_name VARCHAR(100) NOT NULL,
                sensor_type VARCHAR(50),
                device VARCHAR(50),
                unit VARCHAR(20),
                status VARCHAR(20)
            );
        """)

    connection.commit()
    print("Sensors table created successfully!")


if __name__ == "__main__":
    create_sensors_table()