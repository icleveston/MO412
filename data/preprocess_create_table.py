import sqlite3


# Connect to database
conn = sqlite3.connect('public_transportation_bh.db')

cursor = conn.cursor()

# Create table
cursor.execute("""
CREATE TABLE public_transportation_bh (        
        id INTEGER NOT NULL PRIMARY KEY AUTOINCREMENT,
        data DATE NOT NULL,
        siu INTEGER NOT NULL,
        linha  INTEGER NOT NULL,
        sublinha INTEGER NOT NULL,
        pc INTEGER NOT NULL,
        endereco TEXT NOT NULL,
        num_rua TEXT NOT NULL,
        seq INTEGER,
        "0" INTEGER,
        "1" INTEGER,
        "2" INTEGER,
        "3" INTEGER,
        "4" INTEGER,
        "5" INTEGER,
        "6" INTEGER,
        "7" INTEGER,
        "8" INTEGER,
        "9" INTEGER,
        "10" INTEGER,
        "11" INTEGER,
        "12" INTEGER,
        "13" INTEGER,
        "14" INTEGER,
        "15" INTEGER,
        "16" INTEGER,
        "17" INTEGER,
        "18" INTEGER,
        "19" INTEGER,
        "20" INTEGER,
        "21" INTEGER,
        "22" INTEGER,
        "23" INTEGER,
        total INTEGER,
        lat REAL,
        lon REAL,
        x REAL,
        y REAL
        
);
""")

print('Tabela criada com sucesso.')      

conn.close()

