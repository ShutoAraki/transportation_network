import os
import json
import networkx as nx
import numpy as np
import pandas as pd
import geopy
import geopy.distance
import codecs

DATA_PATH = "data/"
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

    def __init__(self, data_path = "data/", verbose=0):
        self.data_path = data_path
        self.verbose = verbose
    
    def create_graph(self, node_filename, edge_filename, clean=False):
        self.node_filename = node_filename
        self.edge_filename = edge_filename
        if clean:
            self._cleanRoadData()
        self._makeDirectedRoadNetwork()

    # ===== Helper functions =====
    def _distanceBetweenLonLats(self, x1,y1,x2,y2):
        return np.round(geopy.distance.distance(geopy.Point(y1,x1), geopy.Point(y2,x2)).km, decimals=6)  

    def _makeInt(self, someNumber):
        return int(np.round(someNumber, decimals=0))

    def _writeJSONFile(self, graphData,filePathName):    
        with codecs.open(filePathName, 'w', encoding="utf-8-sig") as jsonFile:
            jsonFile.write(json.dumps(nx.readwrite.json_graph.node_link_data(graphData), cls = MyEncoder))

    def _cleanRoadData(self):
        linkData = pd.read_csv(os.path.join(DATA_PATH, self.edge_filename), encoding='utf-8').fillna('')

        # Get rid of the unneeded roadTypes
        linkData2 = linkData.loc[~((linkData.roadType == 'unclassified') | (linkData.roadType == 'residential') | (linkData.roadType == 'living_street') | (linkData.roadType == 'pedestrian'))]

        # Add approx speed limits and widths for road segments that don't have them 
        speedLimitByRoadType = {'motorway':80, 'motorway_link':60, 'trunk':60, 'trunk_link':50, 'primary':50, 'primary_link':50, 'secondary':40, 'secondary_link':40, 'tertiary':30, 'tertiary_link':30, 'road':30}
        driveSpeedByRoadType = {'motorway':60, 'motorway_link':40, 'trunk':30, 'trunk_link':30, 'primary':30, 'primary_link':30, 'secondary':30, 'secondary_link':30, 'tertiary':30, 'tertiary_link':30, 'road':25}
        roadWidthByRoadType = {'motorway':21, 'motorway_link':10.5, 'trunk':14, 'trunk_link':7, 'primary':9, 'primary_link':4.5, 'secondary':6, 'secondary_link':3, 'tertiary':5.5,  'tertiary_link':2.75, 'road':6}

        def fillInMissingRoadData(row):
            oneWay = 0
            speedLimit = 30
            roadWidth = 6
            driveSpeed = 5 # New
            if row.oneWay == 'yes':
                oneWay = 1
            if row.speedLimit == '':
                speedLimit = speedLimitByRoadType[row.roadType]
            if row.speedLimit == '':
                driveSpeed = driveSpeedByRoadType[row.roadType]
            if row.roadWidth == '':
                roadWidth = roadWidthByRoadType[row.roadType]    
            return (oneWay,speedLimit,driveSpeed,roadWidth)
        
        linkData2['oneWay'],linkData2['speedLimit'],linkData2['driveSpeed'],linkData2['roadWidth'] = zip(*linkData2.apply(lambda row: fillInMissingRoadData(row), axis=1))
    
        ###----Go through linkData and convert lists of nodes to new rows of source/target pairs
        linkData3 = []
        linkDataHeaders = list(linkData2.columns.values)+['source','target']
        del linkDataHeaders[2] # nodes column
        del linkDataHeaders[0] # type column TODO: With the updated overpass_query, this line must be deleted.
        for index,row in linkData2.iterrows():
            thisRowNodes = json.loads(row['nodes'])
            for index in range(len(thisRowNodes)-1):
                thisRowData = [row['id'],row['roadType'],row['roadName'],row['oneWay'],row['speedLimit'],row['roadWidth'],row['driveSpeed'],thisRowNodes[index],thisRowNodes[index+1]]
                linkData3.append(thisRowData)
            
        ##-- Convert list of lists into pandas dataframe and export  
        linkData3 = pd.DataFrame(linkData3,columns=linkDataHeaders)
        print(linkData3.head(5))
        self.edge_filename = 'linkData-clean-TokyoArea-v2.csv' 
        linkData3.to_csv(self.edge_filename, sep=',', encoding='utf-8-sig', index=False)
        
        ##-- Remove duplicates in the nodeData
        nodeData = pd.read_csv(os.path.join(DATA_PATH, self.node_filename), encoding='utf-8').fillna('')
        nodeData.drop_duplicates(subset='id', inplace=True)
        self.node_filename = 'data/nodeData-clean-TokyoArea-v2.csv' 
        nodeData.to_csv(self.node_filename, sep=',', encoding='utf-8-sig', index=False)

    def _makeDirectedRoadNetwork(self, save=True):
        linkData = pd.read_csv(os.path.join(DATA_PATH, "linkData-clean-TokyoArea-v2.csv"), encoding='utf-8').fillna('')
        nodeData = pd.read_csv(os.path.join(DATA_PATH, "nodeData-clean-TokyoArea-v2.csv"), encoding='utf-8').fillna('')
        ## First make a directed network, then add a reciprical link for non-oneway roads            

        # Add lane and capacity
        lanesByRoadType = {'motorway':3, 'motorway_link':1, 'trunk':2, 'trunk_link':1, 'primary':2, 'primary_link':1, 'secondary':1, 'secondary_link':1, 'tertiary':1, 'tertiary_link':1, 'road':1}
        capacityByRoadType = { 'motorway':24000, 'motorway_link':8000, 'trunk':16000, 'trunk_link':8000, 'primary':8000, 'primary_link':4000, 'secondary':4000, 'secondary_link':4000, 'tertiary':4000,  'tertiary_link':4000, 'road':2000}
        linkData['capacity'] = linkData.apply(lambda row: capacityByRoadType[row.roadType], axis=1)
        linkData['numLanes'] = linkData.apply(lambda row: lanesByRoadType[row.roadType], axis=1)
        linkData = linkData.drop(['id'], axis=1)
    
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
        nodesWithInDegreeOne = [node for node,inDegree in roadNetwork.in_degree() if inDegree == 1]    
        nodesWithOutDegreeOne = [node for node,outDegree in roadNetwork.out_degree() if outDegree == 1]
        nodesWithInOutDegreeOne = list(set(nodesWithInDegreeOne) & set(nodesWithOutDegreeOne))
        print("Number of nodes with in- and out-degree = 1:",len(nodesWithInOutDegreeOne))
        ''' 
        ##-- Delete nodes with degree == 2 (edges of same type) and connect its two neighbors with a new link that inherits from the "left".
        for node in nodesWithInOutDegreeOne:
            edge1 = list(roadNetwork.in_edges(node,data=True))[0]         
            edge2 = list(roadNetwork.out_edges(node,data=True))[0]  
            ##-- only fuse the links if they are the same road type
            if edge1[2]['roadType'] == edge2[2]['roadType']:
                roadNetwork.add_edge(edge1[0], edge2[1], roadType = edge1[2]['roadType'], roadName = edge1[2]['roadName'], oneWay = edge1[2]['oneWay'], speedLimit = edge1[2]['speedLimit'], roadWidth = edge1[2]['roadWidth'], driveSpeed = edge1[2]['driveSpeed'], capacity = edge1[2]['capacity'], numLanes = edge1[2]['numLanes'], modality = edge1[2]['modality'])
                roadNetwork.remove_node(node)
        '''
        print("Skip filtering...")
        print("Number of nodes filtered from road network:", numNodes - len(roadNetwork.nodes))
        print("Number of nodes in filtered road network:",len(roadNetwork.nodes))
        
        ###--- ARE THERE STILL NODES WITH DEGREE == 2?  This is now reasonable if the edges are different road types
        nodesWithInDegreeOne = [node for node,inDegree in roadNetwork.in_degree() if inDegree == 1]    
        nodesWithOutDegreeOne = [node for node,outDegree in roadNetwork.out_degree() if outDegree == 1]
        nodesWithInOutDegreeOne = list(set(nodesWithInDegreeOne) & set(nodesWithOutDegreeOne))
        print("Number of filtered nodes with in- and out-degree = 1:",len(nodesWithInOutDegreeOne))
        
        ##-- Check that I didn't delete too much: this should still be zero
        nodesWithDegreeZero = [node for node,degree in roadNetwork.degree() if degree == 0]
        print("Number of filtered nodes with degree = 0:",len(nodesWithDegreeZero))
            
        ###--- Remove nodes that are not connected to any roads ... there aren't any because all nodes are created via the linklist 
        #for node in list(roadNetwork.nodes()):
        #    if roadNetwork.degree(node) == 0:
        #        roadNetwork.remove_node(node)
            
        ####----Add the GIS coords from the nodeData to the nodes
        nx.set_node_attributes(roadNetwork, nodeData.set_index('id').to_dict('index'))
        
        ####----Go through each link and use the GIS coords of the nodes to determine the approx linkLength, then using speed limits, and add time value.
        for edge in list(roadNetwork.edges(data=True)):
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
            self._writeJSONFile(roadNetwork,'data/roadNetwork-Directed-TokyoArea-v4.json')
        

def main():
    graph = DirectedRoadGraphGenerator()
    # graph.create_graph(node_filename="nodeData-clean-TokyoArea-v2.csv", edge_filename="linkData-clean-TokyoArea-v2.csv")
    graph.create_graph(node_filename="elevationNodeData-TokyoArea-v2.csv", edge_filename="elevationLinkData-TokyoArea-v2.csv")


if __name__ == "__main__":
    main()