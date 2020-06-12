import pandas as pd
import numpy as np
import json
from overpass_query import collect_node_data, aggregate_node_data
from directed_road_network import DirectedRoadGraphGenerator
from network_analysis import load_elevation_data_to_nodes

"""
This file organizes all the preprocessings done for the road network.
The pipeline goes as follows:
1. Load the data --> Save as csv (e.g., data/nodeData-TokyoArea-v2.csv)
2. Filter the nodes and edges --> Save as csv (e.g., data/)
3. Generate the graph and store it as JSON
All savepoints are stored in the current working directory.

@author: ShutoAraki (www.github.com/ShutoAraki)
@date: 06/11/2020
"""

def main(version=2, query=False, clean=False, filter=True, generate=True):
    # ===== LOAD THE DATA =====
    if query:
        print("Sending query through Overpass API...")
        collect_node_data(version=version)
        aggregate_node_data()
        print("Done with query")
    node_filename = f'data/nodeData-TokyoArea-v{version}.csv'
    edge_filename = f'data/linkData-TokyoArea-v{version}.csv'
    print(f"Loading data from {node_filename} and {edge_filename}...")
    node_data = pd.read_csv(node_filename)
    edge_data = pd.read_csv(edge_filename)

    # ===== CLEAN THE DATA =====
    if clean:
        print("Cleaning the data...")
        print("The # of edges:", edge_data.shape[0])
        print("The # of nodes:", node_data.shape[0])
        # Add source and target to the edges
        edge_data['driveSpeed'] = [None] * edge_data.shape[0]
        speedLimitByRoadType = {'motorway':80, 
                                'motorway_link':60,
                                'trunk':60,
                                'trunk_link':50,
                                'primary':50,
                                'primary_link':50,
                                'secondary':40,
                                'secondary_link':40,
                                'tertiary':30,
                                'tertiary_link':30,
                                'road':30,
                                'unclassified': 10,
                                'residential': 6,
                                'living_street': 10,
                                'pedestrian': 10}
        driveSpeedByRoadType = {'motorway':60,
                                'motorway_link':40,
                                'trunk':30,
                                'trunk_link':30,
                                'primary':30,
                                'primary_link':30,
                                'secondary':30,
                                'secondary_link':30,
                                'tertiary':30,
                                'tertiary_link':30,
                                'road':25,
                                'unclassified': 10,
                                'residential': 6,
                                'living_street': 6,
                                'pedestrian': 6}
        roadWidthByRoadType = {'motorway':21,
                               'motorway_link':10.5,
                               'trunk':14,
                               'trunk_link':7,
                               'primary':9,
                               'primary_link':4.5,
                               'secondary':6,
                               'secondary_link':3,
                               'tertiary':5.5,
                               'tertiary_link':2.75,
                               'road':6,
                               'unclassified': 10,
                               'residential': 4,
                               'living_street': 6,
                               'pedestrian': 6}

        def fillInMissingRoadData(row):
            oneWay = 0 
            speedLimit = row.speedLimit 
            roadWidth = row.roadWidth 
            driveSpeed = driveSpeedByRoadType[row.roadType]
            if row.oneWay == 'yes':
                oneWay = 1
            if row.speedLimit == '':
                speedLimit = speedLimitByRoadType[row.roadType]
            if row.roadWidth == '':
                roadWidth = roadWidthByRoadType[row.roadType]
            return (oneWay,speedLimit,driveSpeed,roadWidth)
        
        edge_data['oneWay'],edge_data['speedLimit'],edge_data['driveSpeed'],edge_data['roadWidth'] = zip(*edge_data.apply(lambda row: fillInMissingRoadData(row), axis=1))

        #----Go through linkData and convert lists of nodes to new rows of source/target pairs
        edge_dfs = []
        linkDataHeaders = list(edge_data.columns.values) + ['source','target']
        del linkDataHeaders[2] # nodes column
        del linkDataHeaders[0] # type column TODO: With the updated overpass_query, this line must be deleted.
        for index,row in edge_data.iterrows():
            thisRowNodes = json.loads(row['nodes'])
            for index in range(len(thisRowNodes)-1):
                thisRowData = [row['id'],row['roadType'],row['roadName'],row['oneWay'],row['speedLimit'],row['roadWidth'],row['driveSpeed'],thisRowNodes[index],thisRowNodes[index+1]]
                edge_dfs.append(thisRowData)
            
        #-- Convert list of lists into pandas dataframe and export  
        edge_dfs = pd.DataFrame(edge_dfs,columns=linkDataHeaders)
        print(edge_dfs.head(5))
        edge_filename = f'data/linkData-clean-TokyoArea-v{version}.csv'
        edge_dfs.to_csv(edge_filename, sep=',', encoding='utf-8-sig', index=False)
        print("Clean link data saved to", edge_filename)
        
        #-- Remove duplicates in the nodeData
        node_filename = f'data/nodeData-clean-TokyoArea-v{version}.csv'
        nodeData = pd.read_csv(node_filename).fillna('')
        nodeData.drop_duplicates(subset='id', inplace=True)
        nodeData.to_csv(node_filename, sep=',', encoding='utf-8-sig', index=False)
        print("Clean node data saved to", node_filename)


    # ===== FILTER =====
    if filter:
        print("Filtering nodes and edges based on lat lon information...")
        # Main coverage
        min_lon = 139.2019327633228
        min_lat = 35.12569127247789
        max_lon = 140.4100021566109
        max_lat = 36.10592212339746

        # Filter nodes
        filter_cond = (node_data.lon > min_lon) & (node_data.lon < max_lon) & (node_data.lat > min_lat) & (node_data.lat < max_lat)
        node_data = node_data.loc[filter_cond]

        # Filter edges
        node_set = set(node_data.id)
        edge_data = edge_data.loc[edge_data.source.map(lambda s: s in node_set) & edge_data.target.map(lambda t: t in node_set)]

        node_filename = f"data/filtered-nodeData-TokyoArea-v{version}.csv"
        edge_filename = f"data/filtered-linkData-TokyoArea-v{version}.csv"
        # print("Filtering nodes and edges based on elevation availability...")
        node_data.to_csv(node_filename, index=False)
        edge_data.to_csv(edge_filename, index=False)
    
    # ===== GENERATE GRAPH =====
    if generate:
        load_elevation_data_to_nodes(data_filename=f"data/filtered-nodeData-TokyoArea-v{version}.csv",
                                     save_loc=f"data/filtered-nodeData-with-elevation-TokyoArea-v{version}.csv")
        graph_gen = DirectedRoadGraphGenerator(version=version)
        graph_gen.create_graph(node_filename=f"filtered-nodeData-with-elevation-TokyoArea-v{version}.csv",
                               edge_filename=f"filtered-linkData-TokyoArea-v{version}.csv",
                               clean=False,
                               save=True)
    
    



if __name__ == "__main__":
    main(version=6)