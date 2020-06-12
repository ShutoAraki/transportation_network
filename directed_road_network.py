import os
import json
import networkx as nx
import numpy as np
import pandas as pd
import geopy
import geopy.distance
import codecs
from helpers.helperFunctions import fullFileName, readPickleFile

# DATA_PATH = "data/"
# DATA_PATH = os.path.join(os.environ['DATA_PATH'], "RoadNetworks")

class MyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return super(MyEncoder, self).default(obj)


class DirectedRoadGraphGenerator():

    def __init__(self, data_path = "data/", version=2, verbose=0):
        self.data_path = data_path
        self.verbose = verbose
        self.version = version
    
    def create_graph(self, node_filename, edge_filename, clean=False, save=True):
        self.node_filename = node_filename
        self.edge_filename = edge_filename
        if clean:
            self._cleanRoadData()
        self._makeDirectedRoadNetwork(save)
    
    # ===== Helper functions =====
    def _distanceBetweenLonLats(self, x1,y1,x2,y2):
        return np.round(geopy.distance.distance(geopy.Point(y1,x1), geopy.Point(y2,x2)).km, decimals=6)  

    def _makeInt(self, someNumber):
        return int(np.round(someNumber, decimals=0))

    def _writeJSONFile(self, graphData, filePathName):    
        with codecs.open(filePathName, 'w', encoding="utf-8-sig") as jsonFile:
            jsonFile.write(json.dumps(nx.readwrite.json_graph.node_link_data(graphData), cls = MyEncoder))

    def _cleanRoadData(self):
        linkData = pd.read_csv(os.path.join(self.data_path, self.edge_filename), encoding='utf-8').fillna('')
        linkData['driveSpeed'] = [None] * linkData.shape[0]

        # Get rid of the unneeded roadTypes
        # TODO: Add the removed columns to the graph
        linkData2 = linkData.loc[((linkData.roadType == 'unclassified') | (linkData.roadType == 'residential') | (linkData.roadType == 'living_street') | (linkData.roadType == 'pedestrian'))]
        # linkData2 = linkData.loc[~((linkData.roadType == 'unclassified') | (linkData.roadType == 'residential') | (linkData.roadType == 'living_street') | (linkData.roadType == 'pedestrian'))]

        # Add approx speed limits and widths for road segments that don't have them 
        # speedLimitByRoadType = {'motorway':80, 'motorway_link':60, 'trunk':60, 'trunk_link':50, 'primary':50, 'primary_link':50, 'secondary':40, 'secondary_link':40, 'tertiary':30, 'tertiary_link':30, 'road':30}
        # driveSpeedByRoadType = {'motorway':60, 'motorway_link':40, 'trunk':30, 'trunk_link':30, 'primary':30, 'primary_link':30, 'secondary':30, 'secondary_link':30, 'tertiary':30, 'tertiary_link':30, 'road':25}
        # roadWidthByRoadType = {'motorway':21, 'motorway_link':10.5, 'trunk':14, 'trunk_link':7, 'primary':9, 'primary_link':4.5, 'secondary':6, 'secondary_link':3, 'tertiary':5.5,  'tertiary_link':2.75, 'road':6}

        # Small roads
        speedLimitByRoadType = {'unclassified': 10, 'residential': 6, 'living_street': 10, 'pedestrian': 10}
        driveSpeedByRoadType = {'unclassified': 10, 'residential': 6, 'living_street': 6, 'pedestrian': 6}
        roadWidthByRoadType = {'unclassified': 10, 'residential': 4, 'living_street': 6, 'pedestrian': 6}

        # Large roads
        # def fillInMissingRoadData(row):
        #     oneWay = 0
        #     speedLimit = 30
        #     roadWidth = 6
        #     driveSpeed = 5 # New
        #     driveSpeed = driveSpeedByRoadType[row.roadType]
        #     if row.oneWay == 'yes':
        #         oneWay = 1
        #     if row.speedLimit == '':
        #         speedLimit = speedLimitByRoadType[row.roadType]
        #     if row.roadWidth == '':
        #         roadWidth = roadWidthByRoadType[row.roadType]    
        #     return (oneWay,speedLimit,driveSpeed,roadWidth)
        
        # Small roads  
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
        
        linkData2['oneWay'],linkData2['speedLimit'],linkData2['driveSpeed'],linkData2['roadWidth'] = zip(*linkData2.apply(lambda row: fillInMissingRoadData(row), axis=1))
    
        #----Go through linkData and convert lists of nodes to new rows of source/target pairs
        linkData3 = []
        linkDataHeaders = list(linkData2.columns.values) + ['source','target']
        del linkDataHeaders[2] # nodes column
        del linkDataHeaders[0] # type column TODO: With the updated overpass_query, this line must be deleted.
        for index,row in linkData2.iterrows():
            thisRowNodes = json.loads(row['nodes'])
            for index in range(len(thisRowNodes)-1):
                thisRowData = [row['id'],row['roadType'],row['roadName'],row['oneWay'],row['speedLimit'],row['roadWidth'],row['driveSpeed'],thisRowNodes[index],thisRowNodes[index+1]]
                linkData3.append(thisRowData)
            
        #-- Convert list of lists into pandas dataframe and export  
        linkData3 = pd.DataFrame(linkData3,columns=linkDataHeaders)
        print(linkData3.head(5))
        self.edge_filename = f'data/linkData-clean-TokyoArea-v{self.version}.csv' 
        linkData3.to_csv(self.edge_filename, sep=',', encoding='utf-8-sig', index=False)
        print("Link data saved to", self.edge_filename)
        
        #-- Remove duplicates in the nodeData
        nodeData = pd.read_csv(os.path.join(self.data_path, self.node_filename), encoding='utf-8').fillna('')
        nodeData.drop_duplicates(subset='id', inplace=True)
        self.node_filename = f'data/nodeData-clean-TokyoArea-v{self.version}.csv' 
        nodeData.to_csv(self.node_filename, sep=',', encoding='utf-8-sig', index=False)
        print("Node data saved to", self.node_filename)

    def _makeDirectedRoadNetwork(self, save):
        linkData = pd.read_csv(os.path.join(self.data_path, self.edge_filename), encoding='utf-8').fillna('')
        nodeData = pd.read_csv(os.path.join(self.data_path, self.node_filename), encoding='utf-8').fillna('')
        ## First make a directed network, then add a reciprical link for non-oneway roads            

        # Add lane and capacity
        # Large roads
        lanesByRoadType = {'motorway':3, 'motorway_link':1, 'trunk':2, 'trunk_link':1, 'primary':2, 'primary_link':1, 'secondary':1, 'secondary_link':1, 'tertiary':1, 'tertiary_link':1, 'road':1}
        capacityByRoadType = { 'motorway':24000, 'motorway_link':8000, 'trunk':16000, 'trunk_link':8000, 'primary':8000, 'primary_link':4000, 'secondary':4000, 'secondary_link':4000, 'tertiary':4000,  'tertiary_link':4000, 'road':2000}
        linkData['capacity'] = linkData.apply(lambda row: capacityByRoadType[row.roadType], axis=1)
        linkData['numLanes'] = linkData.apply(lambda row: lanesByRoadType[row.roadType], axis=1)
        # linkData['capacity'] = 500
        # linkData['numLanes'] = 1
        # linkData = linkData.drop(['id'], axis=1)
    
        ###----ADD THE ROAD TYPE TO THE ROAD NETWORK
        linkData['modality'] = 'road'
        nodeData['modality'] = 'road'
            
        ##----Build the network from the road links and remove nodes that are not intersections or endpoints
        ##----NOTE THAT IT IS A DIRECTED GRAPH NOW
        roadNetwork2 = nx.from_pandas_edgelist(linkData, 'source', 'target', True, nx.DiGraph())
        roadNetwork = roadNetwork2.copy()
        print("Number of nodes in unfiltered link network:",len(roadNetwork.nodes))
        numNodes = len(roadNetwork.nodes)
        
        ##-- Reduce the network by removing nodes that are not intersections
        ##-- The OSM data includes lists of nodes for each road segment, some are just for shape, and some are intersections
        ##-- Ones that are just for shape have degree == 2
        # nodesWithInDegreeOne = [node for node,inDegree in roadNetwork.in_degree() if inDegree == 1]    
        # nodesWithOutDegreeOne = [node for node,outDegree in roadNetwork.out_degree() if outDegree == 1]
        # nodesWithInOutDegreeOne = list(set(nodesWithInDegreeOne) & set(nodesWithOutDegreeOne))
        # print("Number of nodes with in- and out-degree = 1:",len(nodesWithInOutDegreeOne))
            
        ###--- Remove nodes that are not connected to any roads ... there aren't any because all nodes are created via the linklist 
        #for node in list(roadNetwork.nodes()):
        #    if roadNetwork.degree(node) == 0:
        #        roadNetwork.remove_node(node)
            
        ####----Add the GIS coords from the nodeData to the nodes
        nx.set_node_attributes(roadNetwork, nodeData.set_index('id').to_dict('index'))
        
        ####----Go through each link and use the GIS coords of the nodes to determine the approx linkLength, then using speed limits, and add time value.
        i = 0
        for edge in list(roadNetwork.edges(data=True)):
            i += 1

            x1 = roadNetwork.nodes(data=True)[edge[0]]['lon']
            y1 = roadNetwork.nodes(data=True)[edge[0]]['lat']
            x2 = roadNetwork.nodes(data=True)[edge[1]]['lon']
            y2 = roadNetwork.nodes(data=True)[edge[1]]['lat']
            distanceBetweenNodes = self._distanceBetweenLonLats(x1,y1,x2,y2)
            if self.verbose > 0:
                print("dist:",distanceBetweenNodes)
                print("speedlimit:",roadNetwork.edges[edge[0],edge[1]]['speedLimit'])
            ##-- Caluculate time based on driveSpeed rather than speedLimit, although still no penalty for turns/traffic/etc.
            timeWeight = ((distanceBetweenNodes / 1000) / roadNetwork.edges[edge[0],edge[1]]['driveSpeed']) * 60  ## the time in minutes
            roadNetwork.edges[edge[0],edge[1]]['x1'] = x1
            roadNetwork.edges[edge[0],edge[1]]['y1'] = y1
            roadNetwork.edges[edge[0],edge[1]]['x2'] = x2
            roadNetwork.edges[edge[0],edge[1]]['y2'] = y2
            roadNetwork.edges[edge[0],edge[1]]['distance'] = self._makeInt(distanceBetweenNodes)   
            roadNetwork.edges[edge[0],edge[1]]['timeWeight'] = np.round(timeWeight, decimals=1)

            ##-- Elevation gain calculation
            sourceHeight = nodeData.loc[nodeData.id == edge[0]].elevation.values[0]
            targetHeight = nodeData.loc[nodeData.id == edge[1]].elevation.values[0]
            roadNetwork.edges[edge[0],edge[1]]['elevationGain'] = targetHeight - sourceHeight
            if i % 200000 == 0:
                print(f"{i}th row: {targetHeight - sourceHeight}")
            if np.isnan(targetHeight - sourceHeight):
                print("Height is missing!")
                print(f"Source: {edge[0]}; Elevation: {sourceHeight}")
                print(f"Target: {edge[1]}; Elevation: {targetHeight}")
                print(f"Elevation gain: {targetHeight - sourceHeight}")
                print(f"Type: {type(targetHeight - sourceHeight)}")
            
            ####---- Create reciprical links for non-oneway roads
            if edge[2]['oneWay'] == 0:
                roadNetwork.add_edge(edge[1], edge[0], roadType = edge[2]['roadType'], roadName = edge[2]['roadName'], oneWay = edge[2]['oneWay'], speedLimit = edge[2]['speedLimit'], roadWidth = edge[2]['roadWidth'], driveSpeed = edge[2]['driveSpeed'], x1 = edge[2]['x1'], y1 = edge[2]['y1'], x2 = edge[2]['x2'], y2 = edge[2]['y2'], distance = edge[2]['distance'], timeWeight = edge[2]['timeWeight'], modality = edge[2]['modality'], capacity = edge[2]['capacity'], numLanes = edge[2]['numLanes'])
        
        #    print(list(roadNetwork.nodes(data=True))[10])
        #    print(list(roadNetwork.edges(data=True))[10])

        print("Number of roadNetwork nodes:",len(roadNetwork.nodes))
        print("Number of roadNetwork links:",len(roadNetwork.edges))
        
        ####### ==================== EXPORT JSON OF NETWORKX ROAD GRAPH ===================
        if save:
            print("==== Writing Road Network File ====")
            json_filename = f'data/roadNetwork-Directed-TokyoArea-with-elevation-v{self.version}.json'
            self._writeJSONFile(roadNetwork, json_filename)
            print("The complete graph file saved at", json_filename)
        self.graph = roadNetwork
    
    # ===== Load a JSON file that contains a directed graph    
    def load_graph(self, filename):
        filename = os.path.join(self.data_path, filename)
        with open(filename, encoding='utf-8-sig') as f:
            js_graph = json.load(f)
        self.graph = nx.json_graph.node_link_graph(js_graph, directed=True, multigraph=False)

    def add_elevation(self, node_filename, edge_filename, save=True):
        elevation_node_df = pd.read_csv(os.path.join(self.data_path, node_filename))
        elevation_edge_df = pd.read_csv(os.path.join(self.data_path, edge_filename))
        print("Creating a dictionary to add...")
        elevation_gains = {(key, value): elGain for key, value, elGain in zip(elevation_edge_df.source, elevation_edge_df.target, elevation_edge_df.elevationGain)}
        print("Adding the elevation data for each edge...")
        nx.set_edge_attributes(self.graph, elevation_gains, "elevationGain")
        print("Adding the elevation data for each node...")
        elevations = {node_id: elevation for node_id, elevation in zip(elevation_node_df.id, elevation_node_df.elevation)}
        nx.set_node_attributes(self.graph, elevations, "elevation")
        if save:
            print("Saving the graph...")
            self._writeJSONFile(self.graph, f'data/roadNetwork-Directed-TokyoArea-with-elevation-v{self.version}.json')




def main():
    graph_gen = DirectedRoadGraphGenerator(data_path="data/", version=6)
    # graph_gen.create_graph(node_filename="nodeData-clean-TokyoArea-v2.csv", edge_filename="linkData-clean-TokyoArea-v2.csv")
    # graph_gen.create_graph(node_filename="elevationNodeData-TokyoArea-v2.csv", edge_filename="elevationLinkData-TokyoArea-v2.csv")

    # graph_gen.load_graph("data/roadNetwork-Directed-TokyoArea-v4.json", full_name=False)
    # graph_gen.add_elevation(edge_filename="data/elevationLinkData-TokyoArea-v4.csv", node_filename="data/elevationNodeData-TokyoArea-v4.csv", full_name=False, save=True)

    # graph_gen.create_graph(node_filename="filtered_nodes.csv", edge_filename="linkData-TokyoArea-v2.csv", clean=True)
    # graph_gen.create_graph(node_filename="filtered-nodeData-TokyoArea-v5.csv",
    #                        edge_filename="filtered-elevationLinkData-TokyoArea-v5.csv",
    #                        clean=False,
    #                        save=True)
    graph_gen.create_graph(node_filename="filtered-nodeData-TokyoArea-big-road.csv",
                           edge_filename="filtered-linkData-TokyoArea-big-road.csv",
                           clean=False,
                           save=True)
    # graph_gen.load_graph("roadNetwork-Directed-TokyoArea-v5.json")
    # graph_gen.add_elevation(
    #     node_filename="filtered-nodeData-TokyoArea-v5.csv",
    #     edge_filename="filtered-elevationLinkData-TokyoArea-v5.csv"
    # )



if __name__ == "__main__":
    main()