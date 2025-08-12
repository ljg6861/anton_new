import os
import datetime as dt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError

# Try to use SQLCipher (encrypted SQLite). If unavailable, fallback to sqlite3
try:
    import pysqlcipher3.dbapi2 as sqlite3
    SQLCIPHER = True
except Exception:
    import sqlite3
    SQLCIPHER = False

DB_PATH = os.environ.get("AUTH_DB_PATH", "users.db")
# Only used when SQLCipher is available:
DB_KEY = os.environ.get("AUTH_DB_KEY")  # e.g. a 32+ char random string

ph = PasswordHasher()  # uses strong Argon2id defaults

def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    if SQLCIPHER:
        if not DB_KEY:
            raise RuntimeError("SQLCipher detected but AUTH_DB_KEY env var not set.")
        # Set the encryption key and sane pragmas
        conn.execute("PRAGMA key = ?;", (DB_KEY,))
        # Optional hardening
        conn.execute("PRAGMA cipher_page_size = 4096;")
        conn.execute("PRAGMA kdf_iter = 256000;")
        conn.execute("PRAGMA cipher_hmac_algorithm = HMAC_SHA512;")
        conn.execute("PRAGMA cipher_kdf_algorithm = PBKDF2_HMAC_SHA512;")
    return conn

def init_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY,
                username TEXT UNIQUE NOT NULL,
                pwd_hash TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
        """)
        conn.commit()

def create_user(username: str, password: str):
    # Hash the password (salted Argon2id)
    pwd_hash = ph.hash(password)
    try:
        with get_conn() as conn:
            conn.execute(
                "INSERT INTO users (username, pwd_hash, created_at) VALUES (?, ?, ?)",
                (username, pwd_hash, dt.datetime.utcnow().isoformat() + "Z")
            )
            conn.commit()
    except sqlite3.IntegrityError:
        raise ValueError("Username already exists")

def verify_user(username: str, password: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("SELECT pwd_hash FROM users WHERE username = ?", (username,))
        row = cur.fetchone()
        if not row:
            return False
        try:
            ph.verify(row["pwd_hash"], password)
            # Optional: if Argon2 suggests rehash (params upgraded), update stored hash
            if ph.check_needs_rehash(row["pwd_hash"]):
                new_hash = ph.hash(password)
                conn.execute("UPDATE users SET pwd_hash=? WHERE username=?", (new_hash, username))
                conn.commit()
            return True
        except VerifyMismatchError:
            return False

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quick username/password DB with Argon2 and optional SQLCipher.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init", help="Create the database schema")

    addp = sub.add_parser("add", help="Create a user")
    addp.add_argument("lucas")
    addp.add_argument("password")

    verp = sub.add_parser("verify", help="Verify login")
    verp.add_argument("lucas")
    verp.add_argument("password")

    args = parser.parse_args()

    if args.cmd == "init":
        if SQLCIPHER:
            if not DB_KEY:
                raise SystemExit("Set AUTH_DB_KEY env var for SQLCipher, e.g. a long random string.")
            print(f"[*] Using SQLCipher. DB: {DB_PATH}")
        else:
            print(f"[!] SQLCipher not available; using plain SQLite. Passwords are still Argon2-hashed.")
        init_db()
        print("[+] Initialized.")

    elif args.cmd == "add":
        init_db()
        create_user(args.username, args.password)
        print("[+] User created.")

    elif args.cmd == "verify":
        ok = verify_user(args.username, args.password)
        print("OK" if ok else "NOPE")
