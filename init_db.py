import psycopg
import os
from dotenv import load_dotenv

load_dotenv()
# 127.0.0.1, not "localhost": avoids the IPv6 ::1 connect stall against the
# IPv4-only Docker Postgres port on Windows. See db.py.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/capture")

def run_sql_file(path):
    with open(path, "r", encoding="utf-8") as f:
        sql = f.read()

    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(sql)

if __name__ == "__main__":
    run_sql_file("migrations/001_init.sql")
    print("Database initialized.")
