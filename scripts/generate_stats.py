import networkx as nx
from networkx.algorithms import *
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from libpysal.cg import voronoi_frames

import contextily as cx
import cartopy.crs as ccrs
from cartopy.io.img_tiles import OSM


def degree_analysis(graph):

    print(f"Nodes: {len(graph.nodes)}")
    print(f"Links: {len(graph.edges)}")

    in_degree_sequence = sorted((d for n, d in graph.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in graph.out_degree()), reverse=True)

    k_in, t_in = np.unique(in_degree_sequence, return_counts=True)
    k_out, t_out = np.unique(out_degree_sequence, return_counts=True)
    N_in, N_out = np.sum(t_in), np.sum(t_out)
    p_k_in, p_k_out = t_in/N_in, t_out/N_out

    print(f"Average Incoming Degree: {np.sum(k_in*p_k_in)}")
    print(f"Average Outgoing Degree: {np.sum(k_out*p_k_out)}")

    fig = plt.figure("Degree Analysis", figsize=(8, 8))

    axgrid = fig.add_gridspec(2, 2)

    ax1 = fig.add_subplot(axgrid[0, 0])
    ax1.bar(k_in, p_k_in)
    ax1.set_xlabel("k_in")
    ax1.set_ylabel("degree distribution")
    plt.grid()
    plt.xlim([-0.5, max(in_degree_sequence)])
    plt.ylim([0, 1])
    plt.xticks(np.arange(min(in_degree_sequence), max(in_degree_sequence) + 1, 1.0))
    plt.yticks(np.arange(0, 1+0.05, 0.1))

    ax2 = fig.add_subplot(axgrid[0, 1])
    ax2.scatter(k_in, p_k_in, s=100)
    ax2.set_xlabel("k_in")
    ax2.set_ylabel("degree distribution")
    ax2.set_yscale('log')

    plt.xlim([-0.5, max(in_degree_sequence)+0.5])
    plt.ylim([0.0001, 1])
    plt.grid()
    plt.xticks(np.arange(min(in_degree_sequence), max(in_degree_sequence) + 1, 1.0))

    ax3 = fig.add_subplot(axgrid[1, 0])
    ax3.bar(k_out, p_k_out)
    ax3.set_xlabel("k_out")
    ax3.set_ylabel("degree distribution")
    plt.grid()
    plt.xlim([-0.5, max(out_degree_sequence)])
    plt.ylim([0, 1])
    plt.xticks(np.arange(min(out_degree_sequence), max(out_degree_sequence) + 1, 1.0))
    plt.yticks(np.arange(0, 1 + 0.05, 0.1))

    ax4 = fig.add_subplot(axgrid[1, 1])
    ax4.scatter(k_out, p_k_out, s=100)
    ax4.set_xlabel("k_out")
    ax4.set_ylabel("degree distribution")
    ax4.set_yscale('log')

    plt.xlim([-0.5, max(out_degree_sequence) + 0.5])
    plt.ylim([0.0001, 1])
    plt.xticks(np.arange(min(out_degree_sequence), max(out_degree_sequence) + 1, 1.0))

    fig.tight_layout()
    plt.grid()
    plt.savefig('degree_analysis.png')
    plt.show()

    _plot_map_degree(graph)


def _plot_map_degree(graph):

    graph_coloring = coloring.greedy_color(graph, strategy='largest_first')

    for key, value in graph_coloring.items():
        graph.nodes[key]['color'] = value

    H = nx.Graph()
    H.add_nodes_from(sorted(graph.nodes(data=True), key=lambda x: x[1]['color']))
    H.add_edges_from(graph.edges(data=True))

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in H.nodes(data=True)}
    colors_raw = [n[1]['color']+1 for n in H.nodes(data=True)]

    colors = list(map(lambda x: x/max(colors_raw), colors_raw))

    norm = matplotlib.colors.LogNorm(vmin=min(colors), vmax=max(colors))

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(H, pos, node_size=list(map(lambda x: x**3, colors_raw)), node_color=norm(colors), arrowsize=1, edge_color='#ddd',
            cmap=plt.cm.GnBu)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.savefig('map_degree.png')
    plt.show()


def connected_components_analysis(graph):

    print(f"Is Strongly Connected: {nx.is_strongly_connected(graph)}")
    print(f"Is Weakly Connected: {nx.is_weakly_connected(graph)}")

    strongly_connected_components = list(sorted(nx.strongly_connected_components(graph), key=len, reverse=True))

    #strongly_connected_components = [c for c in strongly_connected_components if len(c) > 1]

    S = [G.subgraph(c).copy() for c in strongly_connected_components]

    print(f"Components Size: {[len(c) for c in S]}")
    print(f"Diameter of Largest Component: {nx.diameter(S[0])}")

    colors = list(range(len(strongly_connected_components)))
    color_id = 0
    for c in strongly_connected_components:
        color = colors[color_id]
        for n in c:
            graph.nodes[n]['color'] = color
        color_id += 1

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in graph.nodes(data=True)}

    node_color = []
    for n in graph.nodes(data=True):
        node_color.append(n[1]['color'])

    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=imagery.crs)
    imagery = OSM()
    ax.set_extent([-0.14, -0.1, 51.495, 51.515], ccrs.PlateCarree())
    ax.add_image(imagery, 14)

    nx.draw(graph, pos, node_size=5, node_color=node_color, arrowsize=3, vmin=1, vmax=6, edge_color='#ccc',
            cmap=plt.cm.Set1)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()

    _plot_map_connected(graph)


def _plot_map_connected(graph):
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


G = nx.read_gexf('../graph.gexf')

# Remove zero degree nodes
zero_degree_nodes = [n for n, d in G.degree if d == 0]
G.remove_nodes_from(zero_degree_nodes)


plot(G)
degree_analysis(G)
# connected_components_analysis(G)
# bridge_analysis(G)
# spanning_tree(G)
# voronoi_cells(G)
# betweeness_centrality(G)

print(f"Average Clustering Coefficient: {nx.average_clustering(G)}")
print(f"Degree Assortativity Coefficient: {nx.degree_assortativity_coefficient(G)}")

# Summarization
# G_summarized, _ = nx.dedensify(G, threshold=2)
# print(f"{G_summarized}")
