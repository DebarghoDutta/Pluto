import os
import psycopg
from dotenv import load_dotenv

# Load the .env that lives next to this file, not the caller's CWD --
# Pluto Memory (Pi side) imports this module from a sibling folder, so a
# bare load_dotenv() would silently miss it unless launched from here.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

connection = psycopg.connect(
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT"),
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD")
)

print("Connected to PostgreSQL!")