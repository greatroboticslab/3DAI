import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

# Use 127.0.0.1, not "localhost": Docker binds the Postgres port on IPv4 only,
# so "localhost" resolving to IPv6 ::1 first adds a multi-second connect stall
# per request on Windows before it falls back to IPv4.
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@127.0.0.1:55432/capture")

def get_conn():
    return psycopg.connect(DATABASE_URL)