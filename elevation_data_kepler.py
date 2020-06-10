import pandas as pd
import keplergl
import os
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

green_data = pd.read_csv(fullFileName("GreenMap/greenAreasData2-TokyoArea.csv"))

kmap = keplergl.KeplerGl(height=400,
                         data={'green_data': green_data}) 

kmap.save_to_html(file_name="green_analysis.html")