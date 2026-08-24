import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from connection import connection


def create_sensor_readings_table():
    with connection.cursor() as cursor:
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sensor_readings (
                reading_id SERIAL PRIMARY KEY,
                sensor_id INTEGER NOT NULL,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                value DECIMAL,
                unit VARCHAR(20),
                quality VARCHAR(50),
                sensor_session_id INTEGER NOT NULL,
                data TEXT,

                FOREIGN KEY (sensor_id)
                    REFERENCES sensors(sensor_id),

                FOREIGN KEY (sensor_session_id)
                    REFERENCES sessions(session_id)
            );
        """)

    connection.commit()
    print("Sensor Readings table created successfully!")


if __name__ == "__main__":
    create_sensor_readings_table()