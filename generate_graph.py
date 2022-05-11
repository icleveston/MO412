import networkx as nx
import sqlite3

conn = sqlite3.connect('data/public_transportation_bh.db')
cursor = conn.cursor()

cursor.execute("""
SELECT DISTINCT endereco || ", " || num_rua FROM public_transportation_bh;
""")

nodes_name_to_id = {}
for id, linha in enumerate(cursor.fetchall()):
    nodes_name_to_id.update({linha[0]: id})
    
nodes_id_to_name = {v: k for k, v in nodes_name_to_id.items()}


all_lines = []
cursor.execute("""
SELECT DISTINCT linha FROM public_transportation_bh where data = 220306
""")
for linha in cursor.fetchall():
    all_lines.append(linha[0])
   
   
print(f"All Lines: {len(all_lines)}")
   
nodes = []
edges = []
for l in all_lines:
    
    cursor.execute("""
    SELECT data, linha, seq, pc, endereco || ", " || num_rua, lat, lon FROM public_transportation_bh where seq is not null and pc=1 and linha = ? and data = 220306 order by seq;
    """, (l,))
    
    print(f"\n\nLinha: {l}")

    node_id_past = None
    for linha in cursor.fetchall():
        node_id = nodes_name_to_id[linha[4]]
        nodes.append((node_id, str(l), linha[5], linha[6]))
        
        print(linha)

        if node_id_past is None:
            node_id_past = node_id
            continue
        
        edges.append((node_id_past, node_id))
        node_id_past = node_id
    
conn.close()

G = nx.DiGraph()

print(f"Total Nodes; {len(nodes)}")
print(f"Total Edges; {len(edges)}")

G.add_nodes_from([x[0] for x in nodes])
G.add_edges_from(edges)

for node, linha, lat, lon in nodes:

    G.nodes[node]['all_linha'] = linha
    G.nodes[node]['lat'] = lat
    G.nodes[node]['lon'] = lon
    
for l in all_lines:
    for node in G.nodes:
        G.nodes[node][str(l)] = False
    
for node, linha, _, _ in nodes:
    G.nodes[node][linha] = True

# Relabel the nodes
G = nx.relabel_nodes(G, nodes_id_to_name)

# Export graph
nx.write_gexf(G, "graph.gexf")

