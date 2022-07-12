import networkx as nx
from networkx.algorithms import *
import numpy as np
import scipy
from scipy.special import factorial
import matplotlib
import matplotlib.pyplot as plt
from libpysal.cg import voronoi_frames
import geopy.distance
import powerlaw
import pandas as pd


def degree_analysis(graph):

    print(f"Nodes: {len(graph.nodes)}")
    print(f"Links: {len(graph.edges)}")

    in_degree_sequence = sorted((d for n, d in graph.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in graph.out_degree()), reverse=True)

    largest_hub = sorted([{'node':n, 'degree':d} for n, d in graph.out_degree()], reverse=True, key=lambda x:x['degree'])
    print(f"Largest Hub: {largest_hub[0]}")

    k_in, t_in = np.unique(in_degree_sequence, return_counts=True)
    k_out, t_out = np.unique(out_degree_sequence, return_counts=True)
    N_in, N_out = np.sum(t_in), np.sum(t_out)
    p_k_in, p_k_out = t_in/N_in, t_out/N_out

    average_k_in = np.sum(k_in*p_k_in)
    average_k_out = np.sum(k_out*p_k_out)

    average_k_in_squared = np.sum((k_in**2)*p_k_in)

    print(f"Average Incoming Degree: {average_k_in}")
    print(f"Average Outgoing Degree: {average_k_out}")
    print(f"Squared Incoming Degree: {average_k_in_squared}")

    pmf_in = np.exp(-average_k_in) * np.power(average_k_in, k_in) / factorial(k_in)
    pmf_out = np.exp(-average_k_out) * np.power(average_k_out, k_out) / factorial(k_out)

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
    ax2.plot(k_in, pmf_in, 'g')
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
    ax4.plot(k_out, pmf_out, 'g')
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


def scale_free_analysis(graph):

    in_degree_sequence = sorted((d for n, d in graph.in_degree()), reverse=True)
    out_degree_sequence = sorted((d for n, d in graph.out_degree()), reverse=True)

    fit_in = powerlaw.Fit(in_degree_sequence, xmin=1, discrete=True, estimate_discrete=True)
    fit_out = powerlaw.Fit(out_degree_sequence, xmin=1, discrete=True, estimate_discrete=True)

    print(fit_in.power_law.alpha)
    print(fit_out.power_law.alpha)
    print(fit_in.lognormal.mu)
    print(fit_in.lognormal.sigma)

    R, p = fit_in.distribution_compare('lognormal', 'power_law', normalized_ratio=True)
    print(R, p)

    fig = plt.figure("Scale-free Analysis", figsize=(12, 8))

    axgrid = fig.add_gridspec(1, 2)

    ax1 = fig.add_subplot(axgrid[0, 0])
    fit_in.plot_pdf(color='b', linewidth=2, ax=ax1)
    fit_in.power_law.plot_pdf(color='g', linewidth=2, linestyle='--', ax=ax1)
    fit_in.lognormal.plot_pdf(color='y', linewidth=2, linestyle='--', ax=ax1)
    fit_in.exponential.plot_pdf(color='r', linewidth=2, linestyle='--', ax=ax1)
    ax1.set_xlabel("k_in")
    ax1.set_ylabel("degree distribution")
    plt.ylim([0.0000001, 10])
    plt.grid()

    ax2 = fig.add_subplot(axgrid[0, 1])
    fit_out.plot_pdf(color='b', linewidth=2, ax=ax2)
    fit_out.power_law.plot_pdf(color='g', linewidth=2, linestyle='--', ax=ax2)
    fit_out.lognormal.plot_pdf(color='y', linewidth=2, linestyle='--', ax=ax2)
    fit_out.exponential.plot_pdf(color='r', linewidth=2, linestyle='--', ax=ax2)
    ax2.set_xlabel("k_out")
    ax2.set_ylabel("degree distribution")
    plt.ylim([0.0000001, 10])
    plt.grid()

    fig.tight_layout()
    plt.savefig('scale_free_analysis.png')
    plt.show()


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

    strongly_connected_components = sorted(nx.strongly_connected_components(graph), key=len, reverse=True)

    strongly_connected_components = [c for c in strongly_connected_components if len(c) > 1]

    S = [G.subgraph(c).copy() for c in strongly_connected_components]

    print(f"Components Size: {[len(c) for c in S]}")
    print(f"Diameter of Largest Component: {nx.diameter(S[0])}")
    print(f"Average Shortest Path Length: {nx.average_shortest_path_length(S[0])}")
    print(f"Average Clustering Coefficient: {nx.average_clustering(graph)}")
    print(f"Cycles: {len(nx.find_cycle(graph, orientation='original'))}")

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
        node_color.append(n[1]['color'] if 'color' in n[1] else 0)

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(graph, pos, node_size=5, node_color=node_color, arrowsize=3, vmin=1, vmax=6, edge_color='#ccc',
            cmap=plt.cm.Set1)
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.savefig('map_connected_components.png')
    plt.show()


def robustness_analysis(graph, type='attack'):

    def remove_random_node(g, n):
        import random
        for i in range(n):
            node = random.choice(list(g.nodes.keys()))
            g.remove_node(node)

    def remove_highest_degree_node(g, n):
        import random
        for i in range(n):
            node = sorted([{'node': n, 'degree': d} for n, d in g.degree()], reverse=True,
                                 key=lambda x: x['degree'])
            g.remove_node(node[0]['node'])

    def generate_voronoi(id, cells, norm_inv):

        fig = plt.figure(figsize=(10, 6), dpi=100)
        ax = fig.add_subplot(111)
        m = cells.plot(cmap=plt.cm.Spectral_r, column='area_inv', edgecolor='white', linewidth=0, ax=ax, norm=norm_inv)

        plt.gca().invert_xaxis()
        plt.tight_layout()
        plt.axis("off")
        plt.savefig(f"robustness/map_voronoi_{id}.png", transparent=True)
        #plt.show()

    graph = graph.to_undirected()
    lat = []
    lon = []
    nodes = []
    for n in graph.nodes(data=True):
        lat.append(n[1]['lat'])
        lon.append(n[1]['lon'])
        nodes.append(n[0])

    coordinates = np.column_stack((lat, lon))
    cells, points = voronoi_frames(coordinates, clip="convex hull")
    cells["area"] = cells['geometry'].area
    cells["area_inv"] = 1 / cells["area"]
    cells["node"] = nodes
    norm_inv = matplotlib.colors.LogNorm(vmin=cells["area_inv"].min(), vmax=cells["area_inv"].max())
    id = 0

    if type == 'random':

        in_degree_sequence = sorted((d for n, d in graph.degree()), reverse=True)

        k_in, t_in = np.unique(in_degree_sequence, return_counts=True)
        N_in = np.sum(t_in)
        p_k_in = t_in/N_in

        average_k_in = np.sum(k_in*p_k_in)

        average_k_in_squared = np.sum((k_in**2)*p_k_in)
        molloy_reed = average_k_in_squared/average_k_in

        print(f"Average Degree: {average_k_in}")
        print(f"Molloy-Reed Criterion: {molloy_reed}")
        print(f"f_c: {1-(1/(molloy_reed - 1))}")
        print(f"f_c_ER: {1-(1/average_k_in)}")

        strongly_connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph = graph.subgraph(strongly_connected_components[0]).copy().to_undirected()

        graph_len = len(graph)
        print(f"{0} - Components Size: {graph_len}")
        stats = [(0, len(graph))]

        for i in range(graph_len):
            if graph_len > 1:
                remove_random_node(graph, 1)
                components = sorted(nx.connected_components(graph), key=len, reverse=True)
                graph = graph.subgraph(components[0]).to_undirected()
                graph_len = len(graph)

                cells['area_inv'] = cells['area_inv'].mask(~cells['node'].isin(list(components[0])), 0.1)

                generate_voronoi(id, cells, norm_inv)
                id += 1

            print(f"{i+1} - Components Size: {graph_len}")
            stats.append((i+1, graph_len))

        stats = np.asarray(stats)

        with open('robustness_random.npy', 'wb') as f:
            np.save(f, stats)

    elif type == 'attack':

        strongly_connected_components = sorted(nx.connected_components(graph), key=len, reverse=True)
        graph = graph.subgraph(strongly_connected_components[0]).copy().to_undirected()

        graph_len = len(graph)
        print(f"{0} - Components Size: {graph_len}")
        stats = [(0, len(graph))]

        for i in range(graph_len):
            if graph_len > 1:
                remove_highest_degree_node(graph, 1)
                components = sorted(nx.connected_components(graph), key=len, reverse=True)
                graph = graph.subgraph(components[0]).to_undirected()
                graph_len = len(graph)

                cells['area_inv'] = cells['area_inv'].mask(~cells['node'].isin(list(components[0])), 0.1)

                generate_voronoi(id, cells, norm_inv)
                id += 1

            print(f"{i + 1} - Components Size: {graph_len}")
            stats.append((i + 1, graph_len))

        stats = np.asarray(stats)

        with open('robustness_attack.npy', 'wb') as f:
            np.save(f, stats)

    elif type == 'plot':

        stats_random = np.load('robustness_random.npy')
        print(len(stats_random[:,1][stats_random[:,1] > 1]))
        stats_random = stats_random/len(stats_random[:,0])

        stats_attack = np.load('robustness_attack.npy')
        n_hubs_attack = len(stats_attack[:, 1][stats_attack[:, 1] > 1])
        print(n_hubs_attack)
        stats_attack = stats_attack / len(stats_attack[:, 0])

        for i in range(n_hubs_attack):
            fig = plt.figure("Robustness Analysis", figsize=(10, 8))
            axgrid = fig.add_gridspec(1, 1)
            ax1 = fig.add_subplot(axgrid[0, 0])
            ax1.plot(stats_random[:,0], stats_random[:,1], 'g')
            ax1.plot(stats_attack[:,0], stats_attack[:,1], 'r')
            ax1.scatter(stats_attack[i,0], stats_attack[i,1], c='r', s=100)
            ax1.set_xlabel("f")
            ax1.set_ylabel("p")

            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.xticks(np.arange(min(stats_random[:,0]), max(stats_random[:,0])+0.05, .05))
            plt.yticks(np.arange(min(stats_random[:,0]), max(stats_random[:,0])+0.05, .05))

            fig.tight_layout()
            plt.grid()
            plt.savefig(f"robustness/robustness_analysis_{i}.png")
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

    e = nx.tree.Edmonds(graph)
    e.find_optimum()

    tree_graph = nx.tree.greedy_branching(graph, kind='min')
    print(f"Links: {len(tree_graph.edges)}")

    pos = {n[0]: (n[1]['lat'], n[1]['lon']) for n in graph.nodes(data=True)}

    plt.figure(figsize=(14, 10), dpi=300)
    nx.draw(tree_graph, pos, node_size=3, node_color="b", arrowsize=3, edge_color='red')
    # plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.show()


def distance_evenly_distributed_analysis(graph):

    lat = []
    lon = []
    for n in graph.nodes(data=True):
        lat.append(n[1]['lat'])
        lon.append(n[1]['lon'])

    coordinates = np.column_stack((lat, lon))
    cells, _ = voronoi_frames(coordinates, clip="convex hull")
    cells["area"] = cells['geometry'].area
    cells["area_inv"] = 1/cells["area"]

    norm_inv = matplotlib.colors.LogNorm(vmin=cells["area_inv"].min(), vmax=cells["area_inv"].max())
    norm = matplotlib.colors.LogNorm(vmin=cells["area"].min(), vmax=cells["area"].max())

    fig = plt.figure(figsize=(14, 10), dpi=300)
    ax = fig.add_subplot(111)
    m = cells.plot(cmap=plt.cm.Spectral_r, column='area_inv', edgecolor='white', linewidth=0, ax=ax, norm=norm_inv)

    plt.gca().invert_xaxis()
    plt.tight_layout()
    plt.axis("off")
    plt.savefig('map_voronoi.png')
    plt.show()

    attrs = {(x[0], x[1]): {'distance': 0} for x in graph.edges}
    all_distances = []
    # Compute distances
    for e in graph.edges(data=True):
        n1 = graph.nodes[e[0]]
        n2 = graph.nodes[e[1]]
        coords_1 = (n1['lat'], n1['lon'])
        coords_2 = (n2['lat'], n2['lon'])

        d = geopy.distance.geodesic(coords_1, coords_2).meters
        all_distances.append(d)

        attrs[(e[0], e[1])]['distance'] = d

    nx.set_edge_attributes(graph, attrs)

    # Round and sort
    all_distances = set(map(lambda x: round(x, 3), all_distances))
    all_distances = sorted(all_distances, reverse=True)

    print(f"Mean: {np.asarray(all_distances).mean()}")
    print(f"Median: {np.median(all_distances)}")
    print(f"Std: {np.asarray(all_distances).std()}")

    fig = plt.figure("Weight Analysis", figsize=(8, 8))

    axgrid = fig.add_gridspec(1, 1)

    ax1 = fig.add_subplot(axgrid[0, 0])
    ax1.hist(all_distances, bins=1000)
    ax1.set_xlabel("distance (m)")
    ax1.set_ylabel("number of links")
    ax1.set_xscale('log')
    ax1.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax1.set_xticks([10, 30, 50, 100, 200, 300, 500, 1000, 2500, 5000, 10000])
    plt.yticks(np.arange(0, 180, 10))

    fig.tight_layout()
    plt.grid()
    plt.savefig('weight_analysis.png')
    plt.show()


def betweeness_centrality(graph):
    #graph = graph.to_undirected()

    attrs = {(x[0], x[1]): {'distance': 0} for x in graph.edges}
    # Compute distances
    for e in graph.edges(data=True):
        n1 = graph.nodes[e[0]]
        n2 = graph.nodes[e[1]]
        coords_1 = (n1['lat'], n1['lon'])
        coords_2 = (n2['lat'], n2['lon'])

        d = geopy.distance.geodesic(coords_1, coords_2).meters

        attrs[(e[0], e[1])]['distance'] = d

    nx.set_edge_attributes(graph, attrs)

    centrality = nx.betweenness_centrality(graph, weight='distance', endpoints=False, seed=1)

    lpc = nx.community.asyn_lpa_communities(graph, weight='distance', seed=1)

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
    plt.savefig('communities.png')
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


def inneficiency_analysis():
    df = pd.read_csv('../data/tempo_real_convencional_csv_080722101517.csv', delimiter=';')

    lines = df[['NL', 'VL']]
    lines = lines[lines['VL'] > 2]

    d = df[['LT', 'LG', 'VL']]
    d = d[d['VL'] > 1]

    print(d['VL'].mean())
    print(d['VL'].median())
    print(d['VL'].std())

    coordinates = d[['LT', 'LG']].to_numpy(dtype=float)
    color = d['VL'].to_numpy(dtype=float)

    fig = plt.figure("inneficiency_analysis", figsize=(14, 10), dpi=300)
    axgrid = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(axgrid[0, 0])
    ax1.scatter(coordinates[:,0], coordinates[:,1], cmap=plt.cm.Spectral, c=color)
    plt.gca().invert_xaxis()
    plt.axis("off")
    plt.savefig('inefficiency.png')
    plt.show()

    # cells, _ = voronoi_frames(coordinates, clip="convex hull")
    # cells["color"] = 1/color
    #
    # norm_inv = matplotlib.colors.LogNorm(vmin=cells["color"].min(), vmax=cells["color"].max())
    #
    # fig = plt.figure(figsize=(14, 10), dpi=300)
    # ax = fig.add_subplot(111)
    #m = cells.plot(cmap=plt.cm.Spectral_r, column='color', edgecolor='white', linewidth=0, ax=ax,
    #                norm=norm_inv)
    #
    # plt.gca().invert_xaxis()
    # plt.tight_layout()
    # plt.axis("off")
    # plt.savefig('map_voronoi.png')
    # plt.show()

    # f = scipy.interpolate.interp2d(coordinates[:,0], coordinates[:,1], color)
    # xnew = np.arange(min(coordinates[:,0]), max(coordinates[:,0]), 0.001)
    # ynew = np.arange(min(coordinates[:,1]), max(coordinates[:,1]), 0.001)
    # print(xnew)
    # znew = f(xnew,  ynew)

    h, _, _ = np.histogram2d(coordinates[:,0], coordinates[:,1], bins=1000, weights=color)
    print(np.shape(h))

    fig = plt.figure("Degree Analysis", figsize=(10, 8))
    axgrid = fig.add_gridspec(1, 1)
    ax1 = fig.add_subplot(axgrid[0, 0])
    ax1.imshow(h.T, interpolation='gaussian', cmap=plt.cm.Spectral_r) #
    plt.show()

    exit()

    vl = lines['VL'].sort_values(ascending=False).to_numpy()
    k_in, t_in = np.unique(vl, return_counts=True)

    fig = plt.figure("Degree Analysis", figsize=(10, 8))
    axgrid = fig.add_gridspec(1, 1)

    ax1 = fig.add_subplot(axgrid[0, 0])
    ax1.bar(k_in, t_in)
    ax1.set_xlabel("average speed (km/h)")
    ax1.set_ylabel("number of lanes")
    plt.grid()
    # plt.xlim([-0.5, max(in_degree_sequence)])
    # plt.ylim([0, 1])
    plt.xticks(np.arange(min(vl), max(vl) + 1, 3))
    plt.yticks(np.arange(0, 30, 3))

    plt.tight_layout()
    plt.savefig('speed.png')
    plt.show()


G = nx.read_gexf('../graph.gexf')

# Remove zero degree nodes
zero_degree_nodes = [n for n, d in G.degree if d == 0]
G.remove_nodes_from(zero_degree_nodes)

#plot(G)
#degree_analysis(G)
#scale_free_analysis(G)
#connected_components_analysis(G)
#distance_evenly_distributed_analysis(G)
#robustness_analysis(G)
inneficiency_analysis()

#bridge_analysis(G)
#betweeness_centrality(G)
#spanning_tree(G)

#print(f"Degree Assortativity Coefficient: {nx.degree_assortativity_coefficient(G)}")


