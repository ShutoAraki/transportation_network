import os
import json
import time
import keplergl
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from helpers.helperFunctions import lineElevationProfile, readJSONDiGraph, getSlopeAngle, writeJSONFile, euclideanDistance, normRGB, getMovingAverageValues, getPolynomialFitValues, getClosestPoint, readJSON 
import matplotlib.pyplot as plt
# import seaborn as sns
import argparse

def calculate_cost(G, alpha=1.5, factor=2, save_loc=None):
    # TODO: Standardize/normalize the values (the costs are probably too small)
    thetas = []
    for edge in list(G.edges(data=True)):
        source = edge[0]
        target = edge[1]
        edge_info = G.edges[source, target]
        px1 = edge_info['x1']
        py1 = edge_info['y1']
        px2 = edge_info['x2']
        py2 = edge_info['y2']
        l = euclideanDistance(px1, py1, px2, py2) / 0.000011
        # l = G.edges[source, target]['distance']
        h1 = G.nodes[source]['elevation']
        h2 = G.nodes[target]['elevation']
        theta = getSlopeAngle((h2 - h1), l)
        thetas.append(theta)
        adj_l = l / np.cos(np.deg2rad(theta))
        if theta > 0:
            d_i = factor * (theta+1)**alpha * adj_l + adj_l
        else:
            d_i = (theta+1)**alpha * adj_l + adj_l
        G.edges[source, target]['cost'] = d_i
    # sns.distplot(thetas)
    if save_loc:
        print("Saving the graph to", save_loc)
        writeJSONFile(G, save_loc)
    return G


def elevationProfile3D(graph, path, title="Test"):
    xValues = [] # Lon
    yValues = [] # Lat
    zValues = [] # Elevation
    for node in path:
        node_info = graph.nodes[node]
        xValues.append(node_info['lon'])
        yValues.append(node_info['lat'])
        zValues.append(node_info['elevation'])
    
    ax = plt.axes(projection='3d')

    # Data for a three-dimensional line
    ax.plot3D(xValues, yValues, zValues, 'gray')

    # Data for three-dimensional scattered points
    plt.title(title)
    plt.legend()
    plt.show()


def calculate_global_score(xValues, yValues):
    try:
        slope = (yValues[-1] - yValues[0]) / (xValues[-1] - xValues[0])
    except ZeroDivisionError:
        slope = 0
    line_yValues = [slope * (xValue - xValues[0]) + yValues[0] for xValue in xValues]
    dx = (xValues[-1] - xValues[0]) / len(xValues)
    score = sum([abs(h - l)*dx for h, l in zip(yValues, line_yValues)])
    return score


def plot_graphs(xValues, yValues, title):
    fig, ax = plt.subplots(figsize=(15, 3.5))
    # last_node = len(path) - 1
    # graph.nodes[last_node]
    # plt.yticks(np.arange(min(yValues), max(yValues)+1, 1.0))
    plt.plot(xValues, yValues, c=normRGB(169,169,169,0.75), label='Raw Data')
    plt.plot(xValues, getMovingAverageValues(xValues, yValues), c=normRGB(220,20,60,0.55), label='Moving Average')
    plt.plot(xValues, getPolynomialFitValues(xValues, yValues), c=normRGB(30,144,255,0.75), label='Polynomial Fit')  ##blue

    try:
        slope = (yValues[-1] - yValues[0]) / (xValues[-1] - xValues[0])
    except ZeroDivisionError:
        slope = 0
    line_yValues = [slope * (xValue - xValues[0]) + yValues[0] for xValue in xValues]

    global_score = round(calculate_global_score(xValues, yValues), 4)
    plt.plot(xValues, line_yValues, label='Straight Line')
    plt.title(f"{title}: global score = {global_score}")
    plt.legend()
    plt.xlabel("Total distance along the path (m)")
    plt.ylabel("Elevation (m)")
    return fig


def pathElevationProfile(graph, path, title="Test", save_loc=None):
    start = time.time()
    ref_node = graph.nodes[path[0]]
    
    ref_elevation = ref_node['elevation']

    dist = 0 # Total distance
    xValues = [dist]
    yValues = [ref_elevation]
    for i in range(1, len(path)):
        source = path[i-1]
        target = path[i]
        edge_info = graph.edges[source, target]
        px1 = edge_info['x1']
        py1 = edge_info['y1']
        px2 = edge_info['x2']
        py2 = edge_info['y2']
        dist += euclideanDistance(px1, py1, px2, py2) / 0.000011  # Convert from lat-lon degrees to meters
        xValues.append(dist)
        elevation = graph.nodes[target]['elevation']
        yValues.append(elevation)

    fig = plot_graphs(xValues, yValues, title=title)
    
    if save_loc != None:
        fig.savefig(f"{save_loc}.png", dpi=150, transparent=True, bbox_inches='tight')
    pathFindTime = time.time() - start
    print("--Found Path Elevation in",pathFindTime,"seconds")


def generate_shortest_path_df(paths, links):
    result = {}
    index = 1
    # paths: list of lists
    '''
    for path in paths:
        broken_link = False
        ans = []
        for i in range(len(path)-1):
            source = path[i]
            target = path[i + 1]
            link = links.loc[(links.source == source) & (links.target == target)]
            if link.empty:
                print("Missing link!:", index)
                broken_link = True
                break
            ans.append(link)
        if not broken_link:
            result[f"path_{index}"] = pd.concat(ans)
            index += 1
    '''
    for path in paths:
        ans = []
        for i in range(len(path)-1):
            source = path[i]
            target = path[i + 1]
            link = links.loc[(links.source == source) & (links.target == target)]
            if link.empty:
                print("Missing link!:", index)
                link = links.loc[(links.source == target) & (links.target == source)]
                print(link)
            ans.append(link)
        result[f"path_{index}"] = pd.concat(ans)
        index += 1


    for path_type, s_path in result.items():
        s_path.loc[:, 'pathId'] = path_type
    return pd.concat(result.values())
    # return pd.concat(ans)


def kepler_viz(link_df, name, title="", config=None):
    if title == "":
        title = name
    if config:
        kmap = keplergl.KeplerGl(height=400,
                                 data={name: link_df},
                                 config=config)
    else:
        kmap = keplergl.KeplerGl(height=400,
                                 data={name: link_df})
    kmap.save_to_html(file_name=f"{title}.html")


def test_points(alpha, factor):

    firstPoint = [35.664947, 139.704976]    ### Aaron's home
    secondPoint = [35.659180, 139.700703]   ### Shibuya Station
    thirdPoint = [35.664598, 139.737645]    ### Roppongi Grand Tower
    forthPoint = [35.650068, 139.712641]    ### Ebisu Office
    fifthPoint = [35.681264, 139.766952]    ### Tokyo Station
    sixthPoint = [35.645701, 139.747634]    ### Tamachi Station
    seventhPoint = [35.625876, 139.771441]  ### Daiba Station
    eighthPoint = [35.635951, 139.878562]   ### Tokyo Disney
    ninthPoint = [35.613111, 140.113332]    ### Chiba Station
    tenthPoint = [35.707436, 139.958920]    ### NishiFunabashi Station
    eleventhPoint = [35.780159, 139.700633]    ### Friends Home
    twelfthPoint = [35.776210, 139.694785]    ### Shimura Sakaue Station
    thirteenthPoint = [35.709231, 139.727571] ### Shuto's home
    fourteenthPoint = [35.706521, 139.710016] ### Shuto's Hight School

    listOfPairs = [(firstPoint,secondPoint,"Home to Shibuya Station"),(firstPoint,thirdPoint,"Home to Roppongi Grand Tower"),(firstPoint,forthPoint,"Home to Ebisu Office"),(firstPoint,fifthPoint,"Home to Tokyo Station"),(secondPoint,thirdPoint,"Shibuya Station to Roppongi Grand Tower"),(secondPoint,forthPoint,"Shibuya Station to Ebisu Office"),(sixthPoint,seventhPoint,"Tamachi Station to Daiba Station"),(secondPoint,eighthPoint,"Shibuya Station to Tokyo Disney"),(eleventhPoint,twelfthPoint,"Friends Home to Shimura Sakaue Station"),(thirteenthPoint,fourteenthPoint,"Shuto's Home to Toyama HS")]
    # listOfPairs = [(firstPoint,secondPoint,"Home to Shibuya Station"), (eleventhPoint,twelfthPoint,"Friends Home to Shimura Sakaue Station"),(firstPoint,fifthPoint,"Home to Tokyo Station"),(thirteenthPoint,fourteenthPoint,"Shuto's Home to Toyama HS")]
    # listOfPairs = [(firstPoint,secondPoint,"Home to Shibuya Station")]

    # filename = "data/roadNetwork-combined-with-cost-v6.json"
    filename = "data/roadNetwork-Directed-TokyoArea-with-cost-v6.json"
    G = readJSONDiGraph(filename)
    # G = G.to_undirected()
    print("Done loading the graph!")

    nodes = pd.DataFrame([{'id': node[0], **node[1]} for node in list(G.nodes(data=True))])
    G = calculate_cost(G, alpha=alpha, factor=factor)
    print("Done calculating the weights!")
    paths = []

    for thisPair in listOfPairs:
        source = thisPair[0]
        target = thisPair[1]
        title = thisPair[2]
        sourceId = nodes.iloc[getClosestPoint(source[0], source[1], nodes), :].id
        targetId = nodes.iloc[getClosestPoint(target[0], target[1], nodes), :].id
        print(f"Source: {sourceId}")
        print(f"Target: {targetId}")

        pairStartTime = time.time()
        path = nx.dijkstra_path(G, sourceId, targetId, weight='cost')
        paths.append(path)
        # lineElevationProfile(source, target, title, saveLoc=f"data/lineElevation-{title}")
        
        save_dir = f"data/elevation_profiles/alpha={alpha}/u={factor}/"
        save_loc = f"data/elevation_profiles/alpha={alpha}/u={factor}/pathElevation-{title}-alpha:{alpha}|u:{factor}" 
        save_loc = os.path.join(save_dir, f"pathElevation-{title}-alpha:{alpha}|u:{factor}")
        # if the folder does not exist, create one
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        pathElevationProfile(G, path, title=title, save_loc=save_loc)

        # Do it again in reverse just in case the directedness does not help
        # path = nx.dijkstra_path(G, targetId, sourceId, weight='cost')
        # paths.append(path)

        print("--Completed Path Analysis and Plotting in",np.round((time.time()-pairStartTime),2),"seconds")
    
    with open(filename, encoding='utf-8-sig') as f:
        js_graph = json.load(f)
    links = pd.DataFrame(js_graph['links'])
    map_config = readJSON("shortest_path_config.txt")
    shortest_paths = generate_shortest_path_df(paths, links)
    shortest_paths.to_csv(f"data/shortest_paths-alpha:{alpha}|u:{factor}.csv", index=False)
    # kepler_viz(shortest_paths, "shortest_paths")
    kepler_viz(shortest_paths, name="shortest_paths", title=f"shortest_paths-alpha:{alpha}|u:{factor}", config=map_config)
    
    
def main():
    parser = argparse.ArgumentParser(description='Choose hyperparameters alpha and u')
    parser.add_argument('-a', type=float, default=1.5,
                    help='How bad are slopes in general')
    parser.add_argument('-u', type=float, default=2.0,
                    help='How much you hate walking uphill compared to downhill')
    args = parser.parse_args()
    alpha = args.a
    factor = args.u

    # Random example
    # A = 1094016285 # my apartment
    # B = 393130770 # near my high school
    # B = 1731350139 # Test point

    # TODO: Construct a graph and get the shortest path and its elevation profile
    # filename = "data/roadNetwork-Directed-TokyoArea-with-elevation-v5.json"
    # small_roads = readJSONDiGraph(filename)
    # filename = "data/roadNetwork-Directed-TokyoArea-with-elevation-v6.json"
    # big_roads = readJSONDiGraph(filename)
    # print("Combining graphs...")
    # G = nx.compose(big_roads, small_roads)
    # print("Saving the composed graph at data/roadNetwork-combined-v6.json")
    # writeJSONFile(G, "data/roadNetwork-combined-v6.json")
    # G = readJSONDiGraph("data/roadNetwork-combined-v6.json")
    # calculate_cost(G, save_loc="data/roadNetwork-combined-with-cost-v6.json")

    # filename = "data/roadNetwork-combined-with-cost-v6.json"
    # G = readJSONDiGraph(filename)
    # print("Done loading the graph!")
    
    # pointA = (G.nodes[A]['lat'], G.nodes[A]['lon'])
    # pointB = (G.nodes[B]['lat'], G.nodes[B]['lon'])

    '''
    nodeData = G.nodes(data=True)
    v = {'lat': pointA[0], 'lon': pointA[1]}
    print("THE CLOSEST NODE:")
    print(closestNode(nodeData, v))
    '''
    # nodes = pd.DataFrame(G.nodes(data=True))
    # this_point = getClosestPoint(pointA[0], pointA[1], nodes)
    # print(this_point)

    # lineElevationProfile(pointA, pointB, 'Small test')
    
    # path = nx.dijkstra_path(G, A, B, weight='cost')
    # print("PATH:", path)
    # pathElevationProfile(G, path, title="My home to high school")

    test_points(alpha=alpha, factor=factor)


if __name__ == "__main__":
    main()