import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres@localhost:55432/capture")

def get_conn():
    return psycopg.connect(DATABASE_URL)