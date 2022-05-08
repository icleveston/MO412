import pandas as pd 

df = pd.read_csv('embarque_por_ped-siu-sublinha_20220406.csv', sep=";", encoding='latin-1')

df["Endereco"] = df[' Endereço do Ponto'] + ", " + df[' Nº Imóvel'].astype(str)

print(len(df["Endereco"].unique()))