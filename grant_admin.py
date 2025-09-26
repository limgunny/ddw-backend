import os
import sqlite3
import sys


def main():
    if len(sys.argv) < 2:
        print("Usage: python grant_admin.py <username>")
        sys.exit(1)

    username = sys.argv[1]
    db_path = os.path.join(os.path.dirname(__file__), 'instance', 'site.db')

    if not os.path.exists(db_path):
        print(f"DB not found: {db_path}")
        sys.exit(1)

    con = sqlite3.connect(db_path)
    try:
        cur = con.cursor()
        cur.execute("UPDATE user SET role='admin' WHERE username=?", (username,))
        con.commit()
        print("updated rows:", cur.rowcount)
        for row in con.execute("SELECT id, username, role FROM user WHERE username=?", (username,)):
            print("user:", row)
    finally:
        con.close()


if __name__ == '__main__':
    main()


