# import sys
# sys.path.append('.')
from helpers.helperFunctions import readJSONDiGraph, readPickleFile, getLatLonElevation, addLatLonElevations, lineElevationProfile, fullFileName, printProgress

import numpy as np
import pandas as pd
import json
import os
import networkx as nx
import time
# import swifter


def load_elevation_data_to_nodes(data_filename, save_loc=None):
    nodeData = pd.read_csv(data_filename)
    N = nodeData.shape[0]
    print(N, "nodes")
    # print(f"Estimated time: {2.83*N/60} mins")
    print("Loading the boundary dictionary")
    boundaryDict = readPickleFile(fullFileName('Altitude/Elevation5mWindowFiles/boundaryDict.pkl'))
    print("Done loading! Now started adding elevation data...")
    nodeData = addLatLonElevations(nodeData, boundaryDict)

    # start = time.time()
    # nodeData.loc[:, 'elevation'] = nodeData.swifter.apply(getElevation, axis=1)
    # elapsedTime = time.time() - start
    # print(f"It took {elapsedTime} secs (average: {elapsedTime / N} secs/node)")
    
    if save_loc != None:
        nodeData.to_csv(save_loc, index=False)

def load_elevation_data_to_edges(node_data_filename, edge_data_filename, save_loc=None):
    linkData = pd.read_csv(edge_data_filename)
    nodeData = pd.read_csv(node_data_filename)

    runStartTime = time.time()
    N = linkData.shape[0]
    linkData['elevationGain'] = [None] * N
    # for i in range(910716, 910721):
    for i in range(N):
        runStartTime = printProgress(runStartTime, i, N)
        source = linkData.loc[i, 'source']
        target = linkData.loc[i, 'target']
        sourceHeight = nodeData[nodeData.id == source].elevation.values[0]
        targetHeight = nodeData[nodeData.id == target].elevation.values[0]

        # if not np.isnan(sourceHeight) and not np.isnan(targetHeight):
        linkData.loc[i, 'elevationGain'] = targetHeight - sourceHeight
        if i % 10000 == 0:
            print(f"Found an elevation gain!: {linkData.loc[i, 'elevationGain']} at {i}th row")
            linkData.to_csv("savepoint_linkElevationData.csv", index=False)

        
    if save_loc != None:
        linkData.to_csv(save_loc, index=False)



def main():
    # filename = "data/roadNetwork-Directed-TokyoArea-v2.json"
    # print("Loading a giant graph")
    # G = readJSONDiGraph(filename)
    # print(type(G))
    # load_elevation_data_to_nodes('data/nodeData-clean-TokyoArea-v2.csv', save_loc='data/elevationNodeData-TokyoArea-v4.csv')
    load_elevation_data_to_edges('data/filtered-nodeData-TokyoArea-v5.csv', 'data/filtered-linkData-TokyoArea-v5.csv', save_loc='data/filtered-elevationLinkData-TokyoArea-v5.csv')


if __name__ == "__main__":
    main()