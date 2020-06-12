import json
import numpy as np
import pandas as pd
import networkx as nx
from helpers.helperFunctions import lineElevationProfile, readJSONDiGraph, getSlopeAngle, writeJSONFile


def calculate_cost(G, alpha=1.5, factor=2, save_loc=None):
    for edge in list(G.edges(data=True)):
        source = edge[0]
        target = edge[1]
        l = G.edges[source, target]['distance']
        h1 = G.nodes[source]['elevation']
        h2 = G.nodes[target]['elevation']
        theta = getSlopeAngle((h2 - h1), l)
        adj_l = l / np.cos(np.deg2rad(theta))
        if theta > 0:
            d_i = factor * theta**alpha * adj_l + adj_l
        else:
            d_i = theta**alpha * adj_l + adj_l
        G.edges[source, target]['cost'] = d_i
    
    if save_loc:
        print("Saving the graph to", save_loc)
        writeJSONFile(G, save_loc)
    
def main():
    # Random example
    A = 1094016285 # My apartment
    B = 1570382663 # Somewhere nearby 

    # lineElevationProfile(A, B, 'Long Journey')

    # TODO: Construct a graph and get the shortest path and its elevation profile
    filename = "data/roadNetwork-Directed-TokyoArea-with-elevation-v5.json"
    small_roads = readJSONDiGraph(filename)
    filename = "data/roadNetwork-Directed-TokyoArea-v4.json"
    big_roads = readJSONDiGraph(filename)
    print("Combining graphs...")
    G = nx.compose(big_roads, small_roads)
    print("Saving the composed graph at data/roadNetwork-combined-v5.json")
    writeJSONFile(G, "data/roadNetwork-combined-v5.json")
    # G = readJSONDiGraph("data/roadNetwork-combined-v5.json")
    # calculate_cost(G, save_loc="data/roadNetwork-combined-with-cost-v5.json")
    # filename = "data/roadNetwork-Directed-TokyoArea-with-cost-v5.json"
    # G = readJSONDiGraph(filename)
    # for i in range(10000, 10020):
    #     print(list(G.edges(data=True))[i][2]['cost'])
    # path = nx.dijkstra_path(G, A, B, weight='cost')
    # print(path)

    

if __name__ == "__main__":
    main()