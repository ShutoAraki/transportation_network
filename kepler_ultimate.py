import json
import numpy as np
import pandas as pd
import keplergl
from helpers.helperFunctions import readJSON

map_config = readJSON("ultimateMapConfig.json")
filename = "data/roadNetwork-combined-v6.json"
with open(filename, encoding='utf-8-sig') as f:
    js_graph = json.load(f)

links = pd.DataFrame(js_graph['links']).loc[:, ['x1', 'y1', 'x2', 'y2', 'elevationGain', 'roadType', 'capacity']]
nodes = pd.DataFrame(js_graph['nodes']).loc[:, ['id', 'lat', 'lon', 'elevation']]

kmap = keplergl.KeplerGl(height=400,
                         data={'Links': links,
                               'Nodes': nodes},
                         config=map_config)

kmap.save_to_html(file_name="ultimate_roads.html")