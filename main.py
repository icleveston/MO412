import pandas as pd
import sqlite3


#df = pd.read_csv('embarque_por_ped-siu-sublinha_20220406.csv', sep=";", encoding='latin-1')

#df["Endereco"] = df[' Endereço do Ponto'] + ", " + df[' Nº Imóvel'].astype(str)

#print(len(df["Endereco"].unique()))

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

