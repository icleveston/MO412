import pandas as pd
import sqlite3

conn = sqlite3.connect('data/public_transportation_bh.db')
cursor = conn.cursor()

cursor.execute("""
SELECT count(*) FROM public_transportation_bh;
""")

for linha in cursor.fetchall():
    print(f"Total Rows: {linha}")

cursor.execute("""
SELECT COUNT(DISTINCT endereco || ", " || num_rua) FROM public_transportation_bh;
""")

for linha in cursor.fetchall():
    print(f"Total Unique Nodes: {linha}")

conn.close()

