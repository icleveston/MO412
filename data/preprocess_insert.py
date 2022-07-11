import sqlite3
import os
import csv


# Connect to database
conn = sqlite3.connect('public_transportation_bh.db')

cursor = conn.cursor()

total_rows = 0
path_dir = 'csv'
# print(os.listdir('data/csv'))
# Read files
#data/csv
for filename in [f for f in os.listdir(path_dir) if os.path.isfile(os.path.join(path_dir, f))]:

    lista = []

    # Get file data
    data = filename[32:-4]

    with open(os.path.join(path_dir, filename), encoding='latin-1') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=";")
        next(spamreader)

        for row in spamreader:

            r = [data] + row

            lista.append(r)

    print(f"Inserindo arquivo {filename} com tamanho {len(lista)}")

    total_rows += len(lista)

    if len(lista) > 0:

        if len(lista[0]) == 34:

            cursor.executemany("""
            INSERT INTO public_transportation_bh (data,siu,linha,sublinha,pc,endereco,num_rua,"0","1","2","3","4","5",
            "6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23",total,x,y)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, lista)

        elif len(lista[0]) == 37:

            cursor.executemany("""
            INSERT INTO public_transportation_bh (data,siu,linha,sublinha,pc,endereco,num_rua,seq,"0","1","2","3","4","5",
            "6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23",total,lat,lon,x,y)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, lista)

        else:

            lista_proccessed = []
            for l in lista:
                lista_proccessed.append(l[:-2])

            cursor.executemany("""
            INSERT INTO public_transportation_bh (data,siu,linha,sublinha,pc,endereco,num_rua,seq,"0","1","2","3","4","5",
            "6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23",total,lat,lon,x,y)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, lista_proccessed)


        conn.commit()

print(f"Arquivos processados: {len(os.listdir('csv'))}. Total: {total_rows}")

conn.close()
