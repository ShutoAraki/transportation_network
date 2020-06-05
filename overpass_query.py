"""
Author: Shuto Araki (s_araki@ga-tech.co.jp)
Date: 06/04/2020

What does this file do?
- It scrapes road data in Tokyo Area (Tokyo, Chiba, Saitama, and Kanagawa)
- in JSON and separate them into node and link CSV files.
So what?
- You can construct a directed road network of Tokyo Area using networkx!
- Refer to directed_road_network.py for more details.
How do I use it?
- Set your OS environment variable named DATA_PATH as the parent directory
- of where your data folder is.
What are the final CSV files?
- It is named as nodeData-TokyoArea-whatever.csv and linkData-TokyoArea-whatever.csv
"""

from __future__ import unicode_literals
from __future__ import print_function 
from datetime import datetime
import os #TODO: Adapt the DATA_PATH
import json
import overpass
import pandas as pd


class GraphDataLoader():
    # TODO: Figure out a way to directly get a prefecture data from area name
    def __init__(self, area, version=2):
        self.api = overpass.API(timeout=900)
        self.area = area.lower()
        if self.area == 'tokyo':
            self.area_code = 3601543125
        elif self.area == 'chiba':
            self.area_code = 3602679957
        elif self.area == 'saitama':
            self.area_code = 3601768185
        elif self.area == 'kanagawa':
            self.area_code = 3602689487
        else:
            print("Area not supported. Defaulting to Tokyo.")
            self.area_code = 3601543125 
        self.version = version
        
    def _save_result_json(self, result):
        now = datetime.now()
        filename = f"data/allData-{self.area}-v{self.version}.json"
        
        print(f"Save as {filename}")
            
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)

    def _save_result_csv(self, result, kind):
        now = datetime.now()
        filename = f"data/{kind}Data-{self.area}-v{self.version}.csv"
        
        print(f"Save as {filename}")
            
        saveData = pd.DataFrame(result)
        saveData.to_csv(filename, sep=',', encoding='utf-8-sig', index=False)

    def execute_query(self, format='csv', node=True):
        
        query = f"""
        (
        way["highway"~"^(motorway|trunk|primary|motorway_link|trunk_link|primary_link|secondary|secondary_link|tertiary|tertiary_link|unclassified|residential|living_street|road|pedestrian)$"](area:{self.area_code});
        );
        out body;
        >;
        out skel qt;
        """

        print("Fetching results...")
        if format == 'csv':
            if node: # Fetch node info
                # This csv is named nodeData-TokyoArea-2.csv in the data folder
                result = self.api.get(query, responseformat='csv(::id,::lat,::lon)')
                print("Done!")
                print("Saving...")
                self._save_result_csv(result, "node")
                print("Done!")
            else: # Fetch edge info
                # This csv is named linkData-TokyoArea-2.csv in the data folder
                # result = self.api.get(self.query, responseformat='csv(::id,"roadType","roadName","oneWay","speedLimit","roadWidth","source","target")')
                result = self.api.get(query, responseformat='csv(::id,::type,"highway","name","oneway","maxspeed","width","direction")')
                print("Done!")
                print("Saving...")
                self._save_result_csv(result, "link")
                print("Done!")

        elif format == 'json': # Saves everything
            result = self.api.get(query, responseformat='json') 
            print("Done!")
            print("Saving...")
            self._save_result_json(result)
            print("Done!")

        else:
            print(f"Format '{format}' is not supported. Available format: 'csv' or 'json'")
            result = None

        self.result = result

# This method might result in 429 code: too many requests due to the rate limit in Overpass API
# So... take your time!
def collect_node_data():
    loader1 = GraphDataLoader(version=2, area='tokyo')
    loader1.execute_query(format='json', node=True)
    loader2 = GraphDataLoader(version=2, area='chiba')
    loader2.execute_query(format='json', node=True)
    loader3 = GraphDataLoader(version=2, area='saitama')
    loader3.execute_query(format='json', node=True)
    loader4 = GraphDataLoader(version=2, area='kanagawa')
    loader4.execute_query(format='json', node=True)


def aggregate_node_data():
    # ===== Convert JSON to CSV =====
    wayDataIndex = ['elements.type', 'elements.id','elements.nodes', 'elements.tags.highway', 'elements.tags.name', 'elements.tags.oneway', 'elements.tags.maxspeed', 'elements.tags.width']
    nodeDataIndex = ['elements.type', 'elements.id','elements.lat','elements.lon']
    
    prefs = ['tokyo', 'chiba', 'saitama', 'kanagawa']
    for pref in prefs:
        print(f"Reading {pref}...")           
        filename = f"data/allData-{pref}-v2.json"
        
        osmData2 = []
        
        with open(filename, 'r', encoding='utf-8') as f:
            ff = json.dumps(f.read())
            df = json.loads(ff) # This is str, not pandas DataFrame
            # Very hacky way to get rid of the first metadata portion in JSON
            # {
            # "version": 0.6,
            # "generator": "Overpass API 0.7.56.1004 6cd3eaec",
            # "osm3s": {
            #     "timestamp_osm_base": "2020-06-04T05:51:03Z",
            #     "timestamp_areas_base": "2020-03-06T11:03:01Z",
            #     "copyright": "The data included in this document is from www.openstreetmap.org. The data is made available under ODbL."
            # },
            # This corresponds to 0:314, so make sure to include the first '{' as df[0]
            df = df[0] + df[315:]
            osmData = pd.read_json(df)
            osmData2 = pd.json_normalize(osmData.to_dict("records"),errors='ignore')
        
        osmDataWayList = osmData2[osmData2['elements.type'] == "way"]
        osmDataNodeList = osmData2[osmData2['elements.type'] == "node"]
        
        osmDataWayList = osmDataWayList[wayDataIndex]
        osmDataNodeList = osmDataNodeList[nodeDataIndex]
        
        osmDataWayList = osmDataWayList.rename(index=str, columns={'elements.type':'type', 'elements.id':'id','elements.nodes':'nodes', 'elements.tags.highway':'highway', 'elements.tags.name':'name', 'elements.tags.oneway':'oneway', 'elements.tags.maxspeed':'maxspeed', 'elements.tags.width':'width'})
        osmDataNodeList = osmDataNodeList.rename(index=str, columns={'elements.type':'type', 'elements.id':'id','elements.lat':'lat','elements.lon':'lon'})
        
        osmDataWayList.to_csv(f"data/TokyoArea-link-{pref}.csv", sep=',', encoding='utf-8-sig', index=False) 
        osmDataNodeList.to_csv(f"data/TokyoArea-node-{pref}.csv", sep=',', encoding='utf-8-sig', index=False)
        print(f"Done {pref}!")

    # ===== Aggregate all the node & link DataFrames =====
    nodeData1 = pd.read_csv("data/TokyoArea-node-tokyo.csv", encoding='utf-8').fillna('')
    nodeData2 = pd.read_csv("data/TokyoArea-node-chiba.csv", encoding='utf-8').fillna('')
    nodeData3 = pd.read_csv("data/TokyoArea-node-saitama.csv", encoding='utf-8').fillna('')
    nodeData4 = pd.read_csv("data/TokyoArea-node-kanagawa.csv", encoding='utf-8').fillna('')
    nodeData = pd.concat([nodeData1,nodeData2,nodeData3,nodeData4])
    nodeData = nodeData.drop(['type'], axis=1)
    nodeData.to_csv("data/nodeData-TokyoArea-v2.csv", sep=',', encoding='utf-8-sig', index=False)

    linkData1 = pd.read_csv("data/TokyoArea-link-tokyo.csv", encoding='utf-8').fillna('')
    linkData2 = pd.read_csv("data/TokyoArea-link-chiba.csv", encoding='utf-8').fillna('')
    linkData3 = pd.read_csv("data/TokyoArea-link-saitama.csv", encoding='utf-8').fillna('')
    linkData4 = pd.read_csv("data/TokyoArea-link-kanagawa.csv", encoding='utf-8').fillna('')
    linkData = pd.concat([linkData1,linkData2,linkData3,linkData4])
    linkData = linkData.rename(index=str, columns={"highway": "roadType", "name": "roadName", "oneway": "oneWay", "maxspeed": "speedLimit", "width": "roadWidth"})
    linkData = linkData.drop(['type'], axis=1)
    linkData.to_csv("data/linkData-TokyoArea-v2.csv", sep=',', encoding='utf-8-sig', index=False)


def main():
    # collect_node_data()
    aggregate_node_data()


if __name__ == "__main__":
    main()