import pandas as pd
import keplergl
import os
import json
from helpers.helperFunctions import readPickleFile, fullFileName
'''
node_filename = "data/nodeData-clean-TokyoArea-v2.csv"
node_df = pd.read_csv(node_filename)

boundaryDict = readPickleFile(fullFileName("Altitude/Elevation5mWindowFiles/boundaryDict.pkl"))
blocks = []
for k in boundaryDict.keys():
    block = pd.DataFrame(boundaryDict[k], index=[k])
    blocks.append(block)
block_df = pd.concat(blocks)
# Cast it to str so that it is JSON serializable
block_df = block_df.astype({'geometry': 'str'})

kmap = keplergl.KeplerGl(height=400,
                         data={'elevation_block': block_df, 'node': node_df})

kmap.save_to_html(file_name="elevation_analysis.html")
'''
# green_data = pd.read_csv(fullFileName("GreenMap/greenAreasData2-TokyoArea.csv"))

# elevation_data = pd.read_csv("data/elevationLinkData-TokyoArea-v4.csv")

# overall_links = pd.read_csv("data/kepler_viz_data.csv")
# print("Done loading the data!")

map_config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "m51aytg",
          "type": "line",
          "config": {
            "dataId": "all_roads",
            "label": "All Roads",
            "color": [
              255,
              203,
              153
            ],
            "columns": {
              "lat0": "y1",
              "lng0": "x1",
              "lat1": "y2",
              "lng1": "x2"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 0.8,
              "thickness": 4,
              "colorRange": {
                "name": "Uber Viz Qualitative 3",
                "type": "qualitative",
                "category": "Uber",
                "colors": [
                  "#12939A",
                  "#DDB27C",
                  "#88572C",
                  "#FF991F",
                  "#F15C17",
                  "#223F9A",
                  "#DA70BF",
                  "#125C77",
                  "#4DC19C",
                  "#776E57",
                  "#17B8BE",
                  "#F6D18A",
                  "#B7885E",
                  "#FFCB99",
                  "#F89570"
                ]
              },
              "sizeRange": [
                0,
                10
              ],
              "targetColor": None
            },
            "hidden": False,
            "textLabel": [
              {
                "field": None,
                "color": [
                  255,
                  255,
                  255
                ],
                "size": 18,
                "offset": [
                  0,
                  0
                ],
                "anchor": "start",
                "alignment": "center"
              }
            ]
          },
          "visualChannels": {
            "colorField": {
              "name": "roadType",
              "type": "string"
            },
            "colorScale": "ordinal",
            "sizeField": None,
            "sizeScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "all_roads": [
              "roadType",
              "x1",
              "y1",
              "x2",
              "y2"
            ]
          },
          "enabled": True
        },
        "brush": {
          "size": 0.5,
          "enabled": False
        },
        "geocoder": {
          "enabled": False
        },
        "coordinate": {
          "enabled": False
        }
      },
      "layerBlending": "normal",
      "splitMaps": [],
      "animationConfig": {
        "currentTime": None,
        "speed": 1
      }
    },
    "mapState": {
      "bearing": 0,
      "dragRotate": False,
      "latitude": 35.707987625391596,
      "longitude": 139.72742398788597,
      "pitch": 0,
      "zoom": 16.676937620688683,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "qn3f9ph",
      "topLayerGroups": {},
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "building": True,
        "water": True
      },
      "threeDBuildingColor": [
        194.6103322548211,
        191.81688250953655,
        185.2988331038727
      ],
      "mapStyles": {
        "qn3f9ph": {
          "accessToken": "pk.eyJ1Ijoic2h1dG9hcmFraSIsImEiOiJja2F4bGpwZGgwMXdoMnNwaTZwNzZ1N2ozIn0.4MK9evmXh1eQPTUauJQbMg",
          "custom": True,
          "icon": "https://api.mapbox.com/styles/v1/shutoaraki/ckaxlks630p1s1ilbdw4i26no/static/-122.3391,37.7922,9,0,0/400x300?access_token=pk.eyJ1Ijoic2h1dG9hcmFraSIsImEiOiJja2F4bGpwZGgwMXdoMnNwaTZwNzZ1N2ozIn0.4MK9evmXh1eQPTUauJQbMg&logo=false&attribution=false",
          "id": "qn3f9ph",
          "label": "Shuto's Classic",
          "url": "mapbox://styles/shutoaraki/ckaxlks630p1s1ilbdw4i26no"
        }
      }
    }
  }
}

# Cut the data in half
# overall_map = keplergl.KeplerGl(height=400,
#                                 data={"all_roads": overall_links.loc[:, ['roadType', 'x1', 'y1', 'x2', 'y2']]},
#                                 config=map_config)
# overall_map.save_to_html(file_name="combined_roads.html")

filename = "data/roadNetwork-Directed-TokyoArea-v5.json"
with open(filename, encoding='utf-8-sig') as f:
    js_graph = json.load(f)

elevation_small_roads = pd.DataFrame(js_graph['links']).loc[:, ['x1', 'y1', 'x2', 'y2', 'elevationGain', 'timeWeight', 'roadType']]
elevation_nodes = pd.DataFrame(js_graph['nodes']).loc[:, ['lat', 'lon', 'elevation']]

kmap = keplergl.KeplerGl(height=400,
#                         #  data={'green_data': green_data}) 
#                         #  data={'elevation_data': elevation_data}) 
#                          data={'elevation_links': links, 'elevation_nodes': nodes}) 
                         data={'elevation_small_roads': elevation_small_roads, 'elevation_nodes': elevation_nodes})

kmap.save_to_html(file_name="elevation_gains.html")