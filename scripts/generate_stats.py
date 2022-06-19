import networkx as nx
from networkx.algorithms import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libpysal.cg import voronoi_frames


def degree_analysis(graph):
    in_degree_sequence = sorted((d for n, d in graph.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in graph.out_degree()), reverse=True)

    fig = plt.figure("Degree Analysis", figsize=(8, 8))

    axgrid = fig.add_gridspec(4, 4)

    ax1 = fig.add_subplot(axgrid[:2, :2])
    ax1.plot(in_degree_sequence, "b-", marker="o")
    ax1.set_title("In-degree Rank Plot")
    ax1.set_ylabel("Degree")
    ax1.set_xlabel("Rank")

    ax2 = fig.add_subplot(axgrid[:2, 2:])
    ax2.bar(*np.unique(in_degree_sequence, return_counts=True))
    ax2.set_title("In-degree Histogram")
    ax2.set_xlabel("Degree")
    ax2.set_ylabel("# of Nodes")

    ax3 = fig.add_subplot(axgrid[2:, :2])
    ax3.plot(out_degree_sequence, "b-", marker="o")
    ax3.set_title("Out-degree Rank Plot")
    ax3.set_ylabel("Degree")
    ax3.set_xlabel("Rank")

    ax4 = fig.add_subplot(axgrid[2:, 2:])
    ax4.bar(*np.unique(out_degree_sequence, return_counts=True))
    ax4.set_title("Out-degree Histogram")
    ax4.set_xlabel("Degree")
    ax4.set_ylabel("# of Nodes")

    fig.tight_layout()
    plt.show()


def bridge_analysis(graph):

    graph = graph.to_undirected()

    graph_bridges = list(nx.bridges(graph))

    print(f"Has Bridges: {'Yes' if len(graph_bridges) > 0 else 'No'} - How many: {len(graph_bridges)}")

    colors = []
    for b in graph.edges:
        color = 'r' if b in graph_bridges else '#ccc'
        colors.append(color)

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in graph.nodes(data=True)}

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(graph, pos, node_size=3, node_color="b", arrowsize=3, edge_color=colors)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def connected_components_analysis(graph):
    graph_undirected = graph.to_undirected()

    connected_components = list(sorted(nx.connected_components(graph_undirected), key=len, reverse=True))

    S = [G.subgraph(c).copy() for c in connected_components]

    print(f"Is Connected: {nx.is_connected(graph_undirected)}")
    print(f"Is Strongly Connected: {nx.is_strongly_connected(graph)}")
    print(f"Is Weakly Connected: {nx.is_weakly_connected(graph)}")
    print(f"Components Size: {[len(c) for c in connected_components]}")
    print(f"Diameter of Largest Component: {nx.diameter(S[0].to_undirected())}")

    colors = [1, 2, 3, 4, 5, 6]
    color_id = 0
    for c in connected_components:
        color = colors[color_id]
        for n in c:
            graph.nodes[n]['color'] = color
        color_id += 1

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in graph.nodes(data=True)}

    node_color = []
    for n in graph.nodes(data=True):
        node_color.append(n[1]['color'])

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(graph, pos, node_size=5, node_color=node_color, arrowsize=3, vmin=1, vmax=6, edge_color='#ccc',
            cmap=plt.cm.Set1)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def spanning_tree(graph):
    graph_undirected = graph.to_undirected()

    tree_graph = nx.minimum_spanning_tree(graph_undirected)

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in tree_graph.nodes(data=True)}

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(tree_graph, pos, node_size=3, node_color="b", arrowsize=3, edge_color='#ccc')
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def voronoi_cells(graph):

    lat = []
    lon = []
    for n in graph.nodes(data=True):
        lat.append(n[1]['lat'])
        lon.append(n[1]['lon'])

    coordinates = np.column_stack((lat, lon))
    cells, _ = voronoi_frames(coordinates, clip="convex hull")
    cells["area"] = 1/cells['geometry'].area

    norm = matplotlib.colors.LogNorm(vmin=cells["area"].min(), vmax=cells["area"].max())

    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = fig.add_subplot(111)
    cells.plot(cmap=plt.cm.Spectral_r, column='area', edgecolor='white', linewidth=1, ax=ax, norm=norm)
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def betweeness_centrality(graph):
    graph = graph.to_undirected()

    centrality = nx.betweenness_centrality(graph, endpoints=True, seed=1)

    lpc = nx.community.label_propagation_communities(graph)
    community_index = {n: i for i, com in enumerate(lpc) for n in com}

    fig, ax = plt.subplots(figsize=(14, 10))
    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in graph.nodes(data=True)}
    node_color = [community_index[n] for n in graph]
    node_size = [v * 8000 for v in centrality.values()]
    nx.draw_networkx(
        graph,
        pos=pos,
        with_labels=False,
        node_color=node_color,
        node_size=node_size,
        edge_color="gainsboro",
        alpha=0.4,
        cmap=plt.cm.Dark2
    )

    plt.gca().invert_xaxis()
    fig.tight_layout()
    plt.axis("off")
    plt.show()


def plot(graph):
    plt.figure(figsize=(14, 10), dpi=300)

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in graph.nodes(data=True)}

    nx.draw(graph, pos, node_size=3, node_color="b", arrowsize=3, edge_color='#ccc')
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def plot_map_degree(graph):
    graph = graph.to_undirected()
    graph_coloring = coloring.greedy_color(graph, strategy='largest_first')

    for key, value in graph_coloring.items():
        graph.nodes[key]['color'] = value

    H = nx.Graph()
    H.add_nodes_from(sorted(graph.nodes(data=True), key=lambda x: x[1]['color']))
    H.add_edges_from(graph.edges(data=True))

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in H.nodes(data=True)}
    colors_raw = [n[1]['color']+1 for n in H.nodes(data=True)]

    colors = list(map(lambda x: x/max(colors_raw), colors_raw))

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(H, pos, node_size=list(map(lambda x: x**3, colors_raw)), node_color=colors, arrowsize=1, edge_color='#ddd',
            cmap=plt.cm.GnBu)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def plot_map_connected(graph):
    graph = graph.to_undirected()
    graph_coloring = coloring.greedy_color(graph, strategy='connected_sequential_bfs')

    for key, value in graph_coloring.items():
        graph.nodes[key]['color'] = value

    H = nx.Graph()
    H.add_nodes_from(sorted(graph.nodes(data=True), key=lambda x: x[1]['color']))
    H.add_edges_from(graph.edges(data=True))

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in H.nodes(data=True)}
    colors_raw = [n[1]['color']+1 for n in H.nodes(data=True)]

    colors = list(map(lambda x: x/max(colors_raw), colors_raw))

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(H, pos, node_size=list(map(lambda x: x**3, colors_raw)), edge_color='#ddd', node_color=colors,
            arrowsize=1, cmap=plt.cm.GnBu)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


G = nx.read_gexf('../graph.gexf')

# Remove zero degree nodes
zero_degree_nodes = [n for n, d in G.degree if d == 0]
G.remove_nodes_from(zero_degree_nodes)

print(f"Nodes: {len(G.nodes)}")
print(f"Links: {len(G.edges)}")
print(f"Average Clustering: {nx.average_clustering(G)}")
print(f"Degree Assortativity Coefficient: {nx.degree_assortativity_coefficient(G)}")

# Summarization
# G_summarized, _ = nx.dedensify(G, threshold=2)
# print(f"{G_summarized}")

#degree_analysis(G)
#bridge_analysis(G)
#connected_components_analysis(G)
#spanning_tree(G)
#voronoi_cells(G)
#betweeness_centrality(G)
#plot(G)
#plot_map_degree(G)
#plot_map_connected(G)




