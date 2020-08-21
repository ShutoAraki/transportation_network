# -*- coding: utf-8 -*-
import time
import networkx as nx
import re
import json
import codecs
import pickle
import collections
import random
import gc
import math
import heapq

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gp

from shapely import wkt
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.geometry import MultiPoint
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from shapely.strtree import STRtree
from shapely.ops import nearest_points
from shapely.ops import unary_union
#from shapely.validation import explain_validity
#import gdal

import geopy
import geopy.distance
#from geopy import Point

#import rasterio
#import rasterio.env
#from rasterio.features import shapes

#from scipy.spatial import distance
#from scipy.spatial import cKDTree
from scipy.optimize import curve_fit   ## for the elevation profile approximation

#from sklearn.neighbors import KDTree
#from sklearn.neighbors import NearestNeighbors

#import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
#from matplotlib.colors import ListedColormap
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import contextily as ctx  ## needed for plotting on basemaps

standardCRS = 'epsg:4326'
mappingCRS = 3857
areaCalcCRS = "+proj=cea +lat_0=35.68250088833567 +lon_0=139.7671 +units=m"


####======================================================================
####====================== LOADING AND SAVING FILES ======================
####======================================================================

def getDtypes(dataType='hexData'):
    if dataType == 'hexData':
        return {'addressCode': 'str', 'prefCode': 'str', 'cityCode': 'str', 'oazaCode': 'str', 'chomeCode': 'str', 'addressName': 'str', 'prefName': 'str', 'districtName': 'str', 'cityName': 'str', 'oazaName': 'str', 'chomeName': 'str', 'totalPopulation': 'float', 'numHouseholds': 'float', 'pop_Total_A': 'float', 'pop_0-4yr_A': 'float', 'pop_5-9yr_A': 'float', 'pop_10-14yr_A': 'float', 'pop_15-19yr_A': 'float', 'pop_20-24yr_A': 'float', 'pop_25-29yr_A': 'float', 'pop_30-34yr_A': 'float', 'pop_35-39yr_A': 'float', 'pop_40-44yr_A': 'float', 'pop_45-49yr_A': 'float', 'pop_50-54yr_A': 'float', 'pop_55-59yr_A': 'float', 'pop_60-64yr_A': 'float', 'pop_65-69yr_A': 'float', 'pop_70-74yr_A': 'float', 'pop_75-79yr_A': 'float', 'pop_80-84yr_A': 'float', 'pop_85-89yr_A': 'float', 'pop_90-94yr_A': 'float', 'pop_95-99yr_A': 'float', 'pop_100yr+_A': 'float', 'pop_AgeUnknown_A': 'float', 'pop_AverageAge_A': 'float', 'pop_15yrOrLess_A': 'float', 'pop_15-64yr_A': 'float', 'pop_65yr+_A': 'float', 'pop_75yr+_A': 'float', 'pop_85yr+_A': 'float', 'pop_Foreigner_A': 'float', 'pop_0-19yr_A': 'float', 'pop_20-69yr_A': 'float', 'pop_70yr+_A': 'float', 'pop_20-29yr_A': 'float', 'pop_30-44yr_A': 'float', 'pop_percentForeigners': 'float', 'pop_percentChildren': 'float', 'pop_percentMale': 'float', 'pop_percentFemale': 'float', 'pop_percent30-44yr': 'float'}
    elif dataType == 'chomeData':
        return {'addressCode': 'str', 'prefCode': 'str', 'cityCode': 'str', 'oazaCode': 'str', 'chomeCode': 'str', 'addressName': 'str', 'prefName': 'str', 'districtName': 'str', 'cityName': 'str', 'oazaName': 'str', 'chomeName': 'str', 'totalPopulation': 'int', 'numHouseholds': 'int', 'pop_Total_A': 'int', 'pop_0-4yr_A': 'int', 'pop_5-9yr_A': 'int', 'pop_10-14yr_A': 'int', 'pop_15-19yr_A': 'int', 'pop_20-24yr_A': 'int', 'pop_25-29yr_A': 'int', 'pop_30-34yr_A': 'int', 'pop_35-39yr_A': 'int', 'pop_40-44yr_A': 'int', 'pop_45-49yr_A': 'int', 'pop_50-54yr_A': 'int', 'pop_55-59yr_A': 'int', 'pop_60-64yr_A': 'int', 'pop_65-69yr_A': 'int', 'pop_70-74yr_A': 'int', 'pop_75-79yr_A': 'int', 'pop_80-84yr_A': 'int', 'pop_85-89yr_A': 'int', 'pop_90-94yr_A': 'int', 'pop_95-99yr_A': 'int', 'pop_100yr+_A': 'int', 'pop_AgeUnknown_A': 'int', 'pop_AverageAge_A': 'float', 'pop_15yrOrLess_A': 'int', 'pop_15-64yr_A': 'int', 'pop_65yr+_A': 'int', 'pop_75yr+_A': 'int', 'pop_85yr+_A': 'int', 'pop_Foreigner_A': 'int', 'pop_0-19yr_A': 'int', 'pop_20-69yr_A': 'int', 'pop_70yr+_A': 'int', 'pop_20-29yr_A': 'int', 'pop_30-44yr_A': 'int', 'pop_percentForeigners': 'float', 'pop_percentChildren': 'float', 'pop_percentMale': 'float', 'pop_percentFemale': 'float', 'pop_percent30-44yr': 'float'}
    elif dataType == 'networkData':
        return {'modality': 'str'}
    else:
        return None

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.float):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def convertToGeoPandas(thisData, toCRS=None):
    thisData = gp.GeoDataFrame(thisData)
    if 'geometry' not in list(thisData):   ###== if there is no geometry column, try to create it
        if (('lat' in list(thisData)) & ('lon' in list(thisData))):
            thisData['geometry'] = thisData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
        elif (('latitude' in list()) & ('longitude' in list(thisData))):
            thisData['geometry'] = thisData.apply(lambda row: Point(row['longitude'], row['latitude']), axis=1)
        else:
            print("  -- Data has no geometry")
            return thisData
    else:
        thisData['geometry'] = thisData['geometry'].apply(wkt.loads)
    thisData.crs = standardCRS                     ##== 4326 corresponds to "naive geometries", normal lat/lon values
    if toCRS != None:
        thisData = thisData.to_crs(toCRS)
    gc.collect()
    return thisData

def readJSON(filename):
    with open(filename, 'r') as f:
        jsonData = f.read()
    return eval(jsonData)

def readJSONDiGraph(filename):
    with open(filename, encoding='utf-8-sig') as f:
        js_graph = json.load(f)
    return nx.json_graph.node_link_graph(js_graph, directed=True, multigraph=False)

def writeJSONFile(graphData,filePathName):
    with codecs.open(filePathName, 'w', encoding="utf-8-sig") as jsonFile:
        jsonFile.write(json.dumps(nx.json_graph.node_link_data(graphData), cls = MyEncoder))

def readPickleFile(filePathName):
    with open (filePathName, 'rb') as fp:
        return pickle.load(fp)

def readGeoPickle(filePathName, toCRS=None):
    with open (filePathName, 'rb') as fp:
        thisData = pickle.load(fp)
        thisData.crs = standardCRS  ##== 4326 corresponds to "naive geometries", normal lat/lon values
        if toCRS != None:
            thisData = thisData.to_crs(toCRS)
        return thisData

def writePickleFile(theData,filePathName):
    with open(filePathName, 'wb') as fp:
        pickle.dump(theData, fp)

def readCSV(fileName, useCols=None, fillNaN=None, theEncoding='utf-8', dtypes=None):
    useCols = [useCols] if isinstance(useCols, str) else useCols  ##-- Support entering a single text field as useCols
    dtypes = dtypes if dtypes != None else getDtypes()  ##-- Set the dTypes for whatever data, get the Dtypes for master data automatically (above)
    try:
        return pd.read_csv(fileName, encoding=theEncoding, usecols=useCols, dtype=dtypes).fillna(fillNaN)
    except:
        return pd.read_csv(fileName, encoding='shift-jis', usecols=useCols, dtype=dtypes).fillna(fillNaN)

def writeCSV(data, fileName):
    data.to_csv(fileName, sep=',', encoding='utf-8-sig', index=False)

def writeGeoCSV(data, fileName):
    pd.DataFrame(data.assign(geometry=data.geometry.apply(wkt.dumps))).to_csv(fileName, sep=',', encoding='utf-8-sig', index=False)

def writeGeoJSON(data, fileName):
    data.to_file(fileName, driver="GeoJSON")

def readGeoPandasCSV(fileName, useCols=None, toCRS=None, fillNaN=''):
    return convertToGeoPandas(pd.read_csv(fileName, encoding='utf-8', usecols=useCols).fillna(fillNaN), toCRS=toCRS)


####==== Write Pandas CSV File to S3
#import s3fs
#def writePandasToCSV(theData,theFilename,theBucket = 'geodata-processing'):
#    s3 = s3fs.S3FileSystem(anon=False)
#    with s3.open(theBucket+'/'+theFilename+'.csv','w') as f:
#        theData.to_csv(f)


####======================================================================
####====================== DATA HELPER FUNCTIONS =========================
####======================================================================

####====
def loadHexDataFromNetwork(thisNetwork, toCRS=None):
    ###=== Isolate the hex nodes
    thisNetwork = thisNetwork.subgraph([node for node,attr in thisNetwork.nodes(data=True) if attr['modality']=="hex"])
    ###=== convert node properties to pandas dataframe
    hexData = pd.DataFrame.from_dict(dict(thisNetwork.nodes(data=True)), orient='index')
    ###=== Convert pandas geometry data into actual geodata
    hexData = convertToGeoPandas(hexData, toCRS=toCRS)
    #xMin, yMin, xMax, yMax = hexData['geometry'].total_bounds
#    print()
    gc.collect()
    return hexData

####==== Read the JSON of the transportation network and extract the hexes with data as geoPandas
def loadHexDataFromNetworkFile(filename, toCRS=None):
    loadHexDataFromNetwork(readJSONDiGraph(filename), toCRS=toCRS)

####========= Master Data Helper Functions ========
def getTopicFile(thisTopic, dataType='hexData'):
    return '../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv'

###=== Read in a master data file, core or otherwise.
def readMasterCSV(fileName, useCols=None, toCRS=None, fillNaN=None):
    dataType = 'hexData' if 'hexData' in fileName else 'chomeData'
    if "Core" in fileName:
        return loadCoreData(dataType=dataType, fillNaN=fillNaN, toCRS=toCRS)
    else:
        if fillNaN != None:
            return pd.read_csv(fileName, encoding='utf-8', usecols=useCols, dtype=getDtypes(dataType)).fillna(fillNaN)
        else:
            return pd.read_csv(fileName, encoding='utf-8', usecols=useCols, dtype=getDtypes(dataType))

###--------------
def loadCoreData(dataType='hexData', fillNaN=None, toCRS=None):
    try:
        coreData = readGeoPickle('../Data/DataMasters/'+dataType+'-Core.pkl', toCRS)
    except:
        coreData = convertToGeoPandas(readMasterCSV(getTopicFile('Core', dataType), fillNaN=fillNaN), toCRS=toCRS)
    return coreData

####==== returns the topic of a variable
def getVariableDict(dataType='hexData'):
    return readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')[dataType]

def getVariableTopic(thisVariable, dataType='hexData'):
    variableLocatorDict = getVariableDict(dataType)
    thisTopic = variableLocatorDict[thisVariable] if thisVariable in list(variableLocatorDict.keys()) else None
    return thisTopic

def getVariableFile(thisVariable, dataType='hexData'):
    thisTopic = getVariableTopic(thisVariable, dataType)
    if thisTopic == None:
        print("Variable '"+thisVariable+"' not found in master data")
        return None
    else:
        return getTopicFile(thisTopic, dataType)

def getVariableList(dataType='hexData'):
    variableLocatorDict = getVariableDict(dataType)
    return list(variableLocatorDict.keys())

def getVariablesByTopic(dataType='hexData'):
    variableLocatorDict = getVariableDict(dataType)
    variablesByTopic = {thisTopic:[k for k,v in variableLocatorDict.items() if v == thisTopic] for thisTopic in list(set(variableLocatorDict.values()))}
    return variablesByTopic

def getTopicList(dataType='hexData'):
    return list(set(getVariableDict(dataType).values()))

def getVariablesForTopic(thisTopic, dataType='hexData'):
    variableLocatorDict = getVariableDict(dataType)
    return [k for k,v in variableLocatorDict.items() if v == thisTopic]


####==== For a provided variable or list of variables, return a geopandas dataframe of the core data plus selected columns
def getDataForVariables(thisVarList, dataType="hexData", fillNaN=None, toCRS=None):
    ###=== Support entering of single variable name instead of a list
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    ###=== Remove variables that are not in this master data
    thisVarList = [thisVar for thisVar in thisVarList if getVariableTopic(thisVar, dataType) is not None]
    ###=== First, get a list of topics (beyond the core data) needed for the variables listed
    theTopicList = list(set([getVariableTopic(thisVar, dataType) for thisVar in thisVarList]))
    theTopicList.remove('Core')
    ###=== Next Load and merge the files acording to the appropriate indexer
    combinedData = loadCoreData(dataType=dataType, fillNaN=fillNaN, toCRS=toCRS)

    if len(theTopicList) > 0:
        mergeKey = 'hexNum' if dataType == 'hexData' else 'addressCode'
        for thisTopic in theTopicList:
#            try:
#               thisData = readPickleFile('../Data/DataMasters/'+dataType+'-'+thisTopic+'.pkl')
#            except:
            theseVars = [var for var in thisVarList if var in getVariablesForTopic(thisTopic, dataType)] + [mergeKey]
            thisData = readMasterCSV(getTopicFile(thisTopic, dataType), useCols=theseVars, toCRS=toCRS, fillNaN=fillNaN)
#            thisData = readCSV('../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv', fillNaN=fillNaN, useCols=theseVars)
#            reportRunTime(startTime)
            combinedData = pd.merge(combinedData, thisData, on=mergeKey)
#    combinedData = combinedData[variablesToKeep]
    return combinedData

####==== After adding a new dataset to the core data, extract the new data and add it to the appropriate topic.
def addDataToTopic(thisData, thisTopic, theseVars=None, dataType='hexData'):
    print("==== Adding Data to Topic File ====")
    mergeKey = 'hexNum' if dataType == 'hexData' else 'addressCode'
    if theseVars != None:
        if mergeKey not in list(thisData):
            theseVars.append(mergeKey)
        thisData = thisData[theseVars]
    else:
        coreVariables = getVariablesForTopic("Core", dataType)  ## get a list of core variables to remove except the mergeKey
        coreVariables.remove(mergeKey)
        theseVars = [thisVar for thisVar in list(thisData) if thisVar not in coreVariables]  ## keep vars not in the core variables
        thisData = thisData[theseVars]
    ###=== If the topic already exists, add the data to that dataset, otherwise create a new topic file
    if thisTopic in getTopicList(dataType):
        thisTopicData = readMasterCSV(getTopicFile(thisTopic, dataType))  ## get the existing topic data
        thisTopicData = pd.merge(thisTopicData, thisData, on=mergeKey)
    else:
        thisTopicData = thisData
    writeCSV(thisTopicData, '../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv')
    ###=== Either way, one has to add the new variables and topic to the data catalog dictionary
    variableLocatorDict = readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')
    for thisVar in theseVars:
        variableLocatorDict[dataType][thisVar] = thisTopic
    writePickleFile(variableLocatorDict, '../Data/DataMasters/variableLocatorDict2.pkl')


####==== Remove the chosen variables from the topic files containing them for the chosen dataType
def removeVariables(thisVarList, dataType='hexData', thisTopic=None):
    print("==== Removing Variables from Topic File ====")
    ###=== Support entering of single variable name instead of a list
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    ###=== This is slow if there are multiple variables per topic, but easier to do this one variable at a time
    for thisVar in thisVarList:
        if ((thisVar != 'hexNum') & (thisVar != 'addressCode')):
            ###=== To delete from a specific topic file then specify, otherwise it will delete from where the dict says that variables is located.
            thisTopic = thisTopic if thisTopic != None else getVariableTopic(thisVar, dataType)
            thisData = readMasterCSV(getTopicFile(thisTopic, dataType))
            thisData.drop(columns=[thisVar], inplace=True)
            writeCSV(thisData, '../Data/DataMasters/'+dataType+'-'+thisTopic+'.csv')

####==== For a chosen topic, get the core data and that data together.
def getDataForTopic(thisTopic, dataType='hexData', fillNaN=None, toCRS=None):
    mergeKey = 'hexNum' if dataType == 'hexData' else 'addressCode'
    coreData = loadCoreData(dataType=dataType, fillNaN=fillNaN, toCRS=toCRS)
    if thisTopic == 'Core':
        return coreData
    else:
        thisData = readMasterCSV(getTopicFile(thisTopic, dataType), toCRS=toCRS, fillNaN=fillNaN)
        return pd.merge(coreData, thisData, on=mergeKey)



####==== check the variable locator file for multiple occurences of the same variable and report the offenders
####=== *** This is unnecessary because it's a dictionary, and dicts can't have duplicate keys!
#def checkVariableRedundancy(dataType='hexData'):
#    variableList = getVariableList(dataType)
#    duplicateVars = set(thisVar for thisVar in variableList if thisVar in seenVars or seenVars.add(thisVar))
#    report duplicateVars



####=============================================================================
####==== It looks like this converts a dataframe with multipolygons into one with only polygons, copying rows
def explode(indata):
#    indf = gp.GeoDataFrame.from_file(indata)
    indf = indata
    outdf = gp.GeoDataFrame(columns=indf.columns)
    for idx, row in indf.iterrows():
        if type(row.geometry) == Polygon:
            outdf = outdf.append(row,ignore_index=True)
        if type(row.geometry) == MultiPolygon:
            multdf = gp.GeoDataFrame(columns=indf.columns)
            recs = len(row.geometry)
            multdf = multdf.append([row]*recs,ignore_index=True)
            for geom in range(recs):
                multdf.loc[geom,'geometry'] = row.geometry[geom]
            outdf = outdf.append(multdf,ignore_index=True)
    return outdf

####=========================== HELPER FUNCTIONS =========================

####====
def flattenList(thisList):
    list(set([item for sublist in thisList for item in sublist]))

####--- Smarter handling of None and zero values when calculating percentages (esp of hex/chome values)
def percentValue(numerator, denominator):
    if ((numerator != None) & ((denominator != None))):
        if denominator != 0:
            return numerator / denominator
        else:
            return 0
    else:
        return None

####---Distance in meters between two points
def distanceBetweenLonLats(x1,y1,x2,y2):
    return np.round(geopy.distance.distance(geopy.Point(y1,x1), geopy.Point(y2,x2)).m, decimals=0)

def euclideanDistance(px1, py1, px2, py2):
    return math.sqrt((px2-px1)**2 + (py2-py1)**2)

def distWithHeight(px1, py1, px2, py2, elevation1, elevation2):
    deltaElevation = abs(elevation1 - elevation2)
    euclideanDist = math.sqrt((px2-px1)**2 + (py2-py1)**2)
    return math.sqrt((deltaElevation)**2 + (euclideanDist)**2)

####==== Calculate the great circle distance in meters for two lat/lon points
def haversineDist(lon1, lat1, lon2, lat2):
    # convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    theAngle = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    return 6367000 * 2 * math.asin(math.sqrt(theAngle))   ## distance in meters

def makeInt(someNumber):
    return int(np.round(someNumber, decimals=0))

def rnd(someNumber, decimals=3):
    return np.round(someNumber, decimals=decimals)

def getXY(pt):
    return [pt.x, pt.y]

def reportRunTime(thisStartTime):
    newStartTime = time.time()
    if newStartTime - thisStartTime < 60:
        print("  -- This block took", np.round((newStartTime - thisStartTime), decimals=2),"seconds")
    else:
        print("  -- This block took", np.round((newStartTime - thisStartTime)/60, decimals=1),"minutes")


#### Usage is runStartTime = printProgress(runStartTime,index,len(fromGeoms))
def printProgress(thisStartTime,index,totalNum):
    oneBlock = makeInt(totalNum / 100)  ## approximately how many are in 1%
    if (index % oneBlock == 0) and (index > 0):
        newStartTime = time.time()
        if newStartTime - thisStartTime < 60:
            print("  --Analyzing",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): previous batch took",rnd((newStartTime - thisStartTime),1),"seconds")
        else:
            print("  --Analyzing",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): previous batch took",rnd((newStartTime - thisStartTime)/60,2),"minutes")
        return newStartTime
    else:
        return thisStartTime

def makeNumberString(number, length=1):
    return str(int(float(number))).zfill(length)





####========================================================================
####================= GEOGRAPHIC DATA FUNCTIONS ============================
####========================================================================

def getRowsToKeep(polyGeoms, geomTree):
#    runStartTime = time.time()
    rowsToKeep = []
    for index,thisPolyGeom in enumerate(polyGeoms):
#        runStartTime = printProgress(runStartTime,index,len(polyGeoms))
        overlappingGeoms = geomTree.query(thisPolyGeom)   ### Returns the polygon indices that intersect this hex/chome
        if overlappingGeoms != []:
            ###--- this just means the bounding boxes overlap, so now check that they actually intersect
            reallyOverlap = False
            for thisGeom in overlappingGeoms:
                if thisPolyGeom.intersects(thisGeom):
                    reallyOverlap = True
            if reallyOverlap:
                rowsToKeep.append(thisPolyGeom.idx)
    #print(rowsToKeep)
    return rowsToKeep

#####=============================================================================

###=== Get the rows of a dataframe that intersect a given polygon (not necessarily from the same dataframe)
def getDataForPolygon(thisPolygon, thisData):

    polyGeoms = list(thisData['geometry'])
    polyValues = list(thisData.index)
    for index, geom in enumerate(polyGeoms):
        polyGeoms[index].idx = polyValues[index]  ##== set the idx data to be the row index of the original data
    geomTree = STRtree(polyGeoms)  ## The tree is built from the data to be filtered because a tree query returns indices from the tree data.
    rowsToKeep = getRowsToKeep([thisPolygon], geomTree)
    return thisData[thisData.index.isin(rowsToKeep)]


####=== Return a list of indices from one dataset for all polygons that intersect some polygon in another dataset
####=== For example, this is used to isolate the green areas within TokyoMain to reduce the filesize to something manageable.
def getRowsWithinArea(boundingData, dataToBeBound):
    boundingGeoms = list(boundingData['geometry'])
#    boundingValues = list(boundingData.index)
#    for index, geom in enumerate(boundingGeoms):
#        boundingGeoms[index].idx = boundingValues[index]  ##== set the idx data to be the row index of the original data

    polyGeoms = list(dataToBeBound['geometry'])
    polyValues = list(dataToBeBound.index)
    for index, geom in enumerate(polyGeoms):
        polyGeoms[index].idx = polyValues[index]  ##== set the idx data to be the row index of the original data

    geomTree = STRtree(boundingGeoms)  ## The tree is built from the data to be filtered because a tree query returns indices from the tree data.
    return getRowsToKeep(polyGeoms, geomTree)


####=== Return a list of indices from one dataset for all polygons that contain som lat/lon point
####=== Accepts a buffer (in meters) for the point so get "within radius" polygons of a point.
def getRowsForLonLat(thisLon, thisLat, thisData, thisBuffer=0):

    thisBuffer = thisBuffer * 0.000011   ##--convert buffer in meters to degrees using ave spherical at 37deg latitude
    thisPoint = Point(thisLon, thisLat).buffer(thisBuffer)

    polyGeoms = list(thisData['geometry'])
    polyValues = list(thisData.index)
    for index, geom in enumerate(polyGeoms):
        polyGeoms[index].idx = polyValues[index]  ##== set the idx data to be the row index of the original data

    geomTree = STRtree(polyGeoms)  ## The tree is built from the data to be filtered because a tree query returns indices from the tree data.
    return getRowsToKeep([thisPoint], geomTree)



####=============================================================================
####========================== INPTEROLATION ALGORITHMS =========================
####=============================================================================

####=============================================================================
####=== Isolate the geometry information and create the geometry objects with indices
def createGeomData(thisData):
    ###=== Determine if the dataset is chome, and reduce it to build geometries only from the lowest level
    if 'lowestLevel' in list(thisData):
        thisData = thisData[thisData['lowestLevel'] == True][['geometry']]
    else:
        thisData = thisData[['geometry']]

    ###=== Add index values as geometry attributes, then Build an R-tree of the hexes
    Geoms = list(thisData['geometry'])
    GeomsIdx = list(thisData.index)
    for index, geom in enumerate(Geoms):
        geom.idx = GeomsIdx[index]   ##-- store the original index value in the geom to reference the data cell later

    return Geoms


####===============================================================================
def guessCalculation(thisVar):
    if 'min' in thisVar.lower():
        return 'min'
    elif 'max' in thisVar.lower():
        return 'max'
    elif 'mean' in thisVar.lower():
        return 'mean'
    elif 'median' in thisVar.lower():
        return 'mean'
    elif 'percent' in thisVar.lower():
        return 'percent'
    else:
        return 'sum'


####===============================================================================
def aggregationCalculations(theseRows, theAdminLevel, theData, thisVarList, thisVarDict):
    for thisRow in theseRows:
        ###=== Filter to the data rows relevant to aggregating at this admin level
        if theAdminLevel == 2:
            relevantData = theData[(theData['prefCode'] == theData.at[thisRow,'prefCode']) & (theData['cityCode'] == theData.at[thisRow,'cityCode']) & (theData['oazaCode'] == theData.at[thisRow,'oazaCode']) & (theData['adminLevel'] == 3)]
        elif theAdminLevel == 1:
            relevantData = theData[(theData['prefCode'] == theData.at[thisRow,'prefCode']) & (theData['cityCode'] == theData.at[thisRow,'cityCode']) & (theData['adminLevel'] == 2)]
        else:
            relevantData = theData[(theData['prefCode'] == theData.at[thisRow,'prefCode']) & (theData['adminLevel'] == 1)]

        for thisVar in thisVarList:
            ###=== Only process rows where the variable of interest exists.
            thisRelevantData = relevantData[((relevantData[thisVar] != '') & (relevantData[thisVar].notnull()))]
            if len(thisRelevantData) > 0:
                if thisVarDict[thisVar] == 'sum':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].sum()
                elif thisVarDict[thisVar] == 'binary':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].any()
                elif thisVarDict[thisVar] == 'min':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].min()
                elif thisVarDict[thisVar] == 'max':
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].max()
                elif ((thisVarDict[thisVar] == 'mean') | (thisVarDict[thisVar] == 'median') | (thisVarDict[thisVar] == 'percent')):
#                    print("value", list(thisRelevantData[thisVar]), "    area", list(thisRelevantData['landArea']))
                    totalAreaInvolved = sum([value * area for (value,area) in zip(list(thisRelevantData[thisVar]),list(thisRelevantData['landArea']))])
                    theData.at[thisRow,thisVar] = percentValue(totalAreaInvolved, theData.at[thisRow,'landArea'])
                else: ##sum by default
                    theData.at[thisRow,thisVar] = relevantData.loc[:,thisVar].sum()
#            else:   ###=== if there is no relevant data, then leave it as None
    return theData


####=============================================================================
####==== Take binary variables at the lowest level (chome) and aggregate them up to all higher levels
####==== Currently, we make the variable true if any of the lower-level contained areas are true
def aggregateUp(theData, thisVarList=None, thisVarDict=None):

    ###=== Provide support for specifiying a single variable by converting it into a list here
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList

    ###=== If the addressCode parts are not in this data, then create them from the full AddressCOde (removed at the end)
    createAddressCodes = False
    if 'prefCode' not in list(theData):
        createAddressCodes = True
        theData['prefCode'] = theData.apply(lambda row: row['addressCode'][0:2], axis=1)
        theData['cityCode'] = theData.apply(lambda row: row['addressCode'][2:5], axis=1)
        theData['oazaCode'] = theData.apply(lambda row: row['addressCode'][5:9], axis=1)
#    print(theData.at[0,'addressCode'],"=",theData.at[0,'prefCode'],"+",theData.at[0,'cityCode'],"+",theData.at[0,'oazaCode'],"+ chomeCode")

    ###=== if the landArea column is not included, get it from the geometry whenever it is needed for a calculation (probably slow)
    createLandArea = False
    listOfCalculationTypes = [v for k,v in thisVarDict.items()]
    if any(x in listOfCalculationTypes for x in ['mean','median','percent']):
        if 'landArea' not in list(theData):
            createLandArea = True
            theData['landArea'] = theData.apply(lambda row: row['geometry'].area, axis=1)

    ####---- Start with Oaza that are not the lowest level
    nonLowestOazaRows = list(theData[(theData['lowestLevel'] == False) & (theData['adminLevel'] == 2)].index.values)
    theData = aggregationCalculations(nonLowestOazaRows, 2, theData, thisVarList, thisVarDict)
#    print(theData.head())

    cityRows = list(theData[(theData['lowestLevel'] == False) & (theData['adminLevel'] == 1)].index.values)
    theData = aggregationCalculations(cityRows, 1, theData, thisVarList, thisVarDict)
#    print(theData.head())

    prefRows = list(theData[(theData['lowestLevel'] == False) & (theData['adminLevel'] == 0)].index.values)
    theData = aggregationCalculations(prefRows, 0, theData, thisVarList, thisVarDict)
#    print(theData.head())

    ###=== Remove temporary variables
    if createAddressCodes == True:
        theData.drop(columns=['prefCode', 'cityCode', 'oazaCode'], inplace=True)
    if createLandArea == True:
        theData.drop(columns=['landArea'], inplace=True)

    return theData


####=============================================================================
####=== Take data using one set of polygons and interpolate it to a different set of polygons using overlap percents.
####=== For example, this can be used to convert between hex <=> chome, but also other "free polygon" data
def interpolateGeoData(fromGeoData, toGeoData, thisVarList=None, thisVarDict=None):

    dataType = 'hexData' if 'hexNum' in list(fromGeoData) else 'chomeData'

    if thisVarDict == None:
        if thisVarList == None:
            ## Convert all non-Core variables in fromGeoData,
            coreVariables = getVariablesForTopic("Core", dataType)  ## get a list of core variables to remove except the mergeKey
            thisVarList = [thisVar for thisVar in list(fromGeoData) if thisVar not in coreVariables]  ## keep vars not in the core variables

        ###=== Infer the interpolation calculcation from the variable names
        thisVarDict = {thisVar:guessCalculation(thisVar) for thisVar in thisVarList}
    else:
        if thisVarList == None:    ###=== if a conversion dictionary was specified, then get the varList from it
            thisVarList = [k for k,v in thisVarDict.items()]
        ###=== Else we are converting a subset of the dictionary, so keep the var list as is, but check if there is something missing from the dict
        else:
            missingDictVars = [thisVar for thisVar in thisVarList if thisVar not in [k for k,v in thisVarDict.items()] ]
            thisVarDict = {thisVar:guessCalculation(thisVar) for thisVar in missingDictVars}

    ###=== Provide support for specifiying a single variable by converting it into a list here
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    ###=== Seed the toGeoData with Nones for each variable being processed
    for thisVar in thisVarList:
        toGeoData[thisVar] = [None] * len(toGeoData)

    ###=== Many calculations require the area of the poygons and their overlaps, so we set them both to an equal-area projection
    fromGeoData = fromGeoData.to_crs(areaCalcCRS)
    toGeoData = toGeoData.to_crs(areaCalcCRS)

    ###=== Create indexed geometry object lists for both datasets for STRtree usage
    toGeoms = createGeomData(toGeoData)
    fromGeoms = createGeomData(fromGeoData)
    ## The tree is built from the fromData so the tree query returns Indices of the fromData that intersect each toData polygon
    geomTree = STRtree(fromGeoms)

    startTime = time.time()
    runStartTime = time.time()
    ###=== For each element being converted into...
    for index, thisToGeom in enumerate(toGeoms):
        runStartTime = printProgress(runStartTime,index,len(toGeoms))
        overlappingFromGeoms = geomTree.query(thisToGeom)
        for thisFromGeom in overlappingFromGeoms:
            if thisFromGeom.intersects(thisToGeom):
                if type(thisFromGeom) == Point:
                    thisFromGeoData = fromGeoData.loc[[thisFromGeom.idx]]
                    for thisVar in thisVarList:
                        currentToValue = toGeoData.at[thisToGeom.idx, thisVar]
                        fromValue = thisFromGeoData[thisVar].values[0]
                        if fromValue != None:
                            if thisVarDict[thisVar] == 'sum':
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue == None else currentToValue + fromValue

                else:
                    ## get the proportion of overlap of the from data to calculate the value to add to this toGeom
                    sourceArea = thisFromGeom.area
                    targetArea = thisToGeom.area
                    overlapArea = thisFromGeom.intersection(thisToGeom).area
                    sourceOverlapProportion = (overlapArea / sourceArea)
                    targetOverlapProportion = (overlapArea / targetArea)
                    thisFromGeoData = fromGeoData.loc[[thisFromGeom.idx]]
                    for thisVar in thisVarList:
                        currentToValue = toGeoData.at[thisToGeom.idx, thisVar]
                        fromValue = thisFromGeoData[thisVar].values[0]
                        if fromValue != None:
                            if thisVarDict[thisVar] == 'binary':
                                ##-- if the current value if either None or False, i.e. stay True if True, else set to the var
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue != True else currentToValue

                            if thisVarDict[thisVar] == 'sum':
                                toGeoData.at[thisToGeom.idx, thisVar] = (fromValue * sourceOverlapProportion) if currentToValue == None else currentToValue + (fromValue * sourceOverlapProportion)

                            if thisVarDict[thisVar] == 'max':
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue == None else max(fromValue, currentToValue)

                            if thisVarDict[thisVar] == 'min':
                                toGeoData.at[thisToGeom.idx, thisVar] = fromValue if currentToValue == None else min(fromValue, currentToValue)

                            if thisVarDict[thisVar] == 'mean':
                                toGeoData.at[thisToGeom.idx, thisVar] = (fromValue * targetOverlapProportion) if currentToValue == None else currentToValue + (fromValue * targetOverlapProportion)

                            if thisVarDict[thisVar] == 'percent':
                                toGeoData.at[thisToGeom.idx, thisVar] = targetOverlapProportion if currentToValue == None else currentToValue + targetOverlapProportion

        #            print("hexIndex:",thisToGeom.idx,"   var:",thisVar,"   value:", hexData.at[thisToGeom.idx,thisVar][0])

    ###=== If the toData is chome data, then only the lowest level has been filled in, so now aggregate up
    if 'addressCode' in list(toGeoData):
        ###--- Aggregate up from chome to their Oaza (using only Oaza that have chome..i.e., not lowest level oaza)
        toGeoData = aggregateUp(toGeoData, thisVarList, thisVarDict)

    toGeoData = toGeoData.to_crs(standardCRS)  ## Convert back to naive geometries before returning the data
    print("==== Completed Conversion in",rnd((time.time()-startTime)/60,2),"minutes ====")
    return toGeoData



####=============================================================================
####=== Take data using one set of polygons and interpolate it to a different set of polygons using overlap percents.
####=== For example, this can be used to convert between hex <=> chome, but also other "free polygon" data
#def interpolateAreaData(fromGeoData,toGeoData,thisVarList):
#
#    ###=== The land areas were computed using this CRS so they matched the values provided by the government.
#    ###=== Because we are calculating area and percent areas here, the idea is that we need to use the same projection
#    fromGeoData = fromGeoData.to_crs(areaCalcCRS)
#    toGeoData = toGeoData.to_crs(areaCalcCRS)
#
#    ###==== Check the noise polygons are all valid, ...some may be made invalid by changing the CRS, so fix them.
#    for thisIndex in fromGeoData.index.values:
#        if fromGeoData.at[thisIndex, 'geometry'].is_valid == False:
#            fromGeoData.at[thisIndex, 'geometry'] = fromGeoData.at[thisIndex, 'geometry'].buffer(0)
##            print("row:", index, "    geometry is valid:", fromGeoData.at[index, 'geometry'].is_valid )
#
##    print(toGeoData.at[0,'geometry'])
##    print("Area of 54,127m2 hexagon using cea crs is:",toGeoData.at[0,'geometry'].area)
#
#    ###=== Seed the toGeometry with Nones for each variable used from the fromGeometry
#    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
#    for thisVar in thisVarList:
#        toGeoData[thisVar] = [None] * len(toGeoData)
#    #    toGeometry[thisVar] = np.zeros(len(toGeometry)).tolist()
#
#    toGeoms = createGeomData(toGeoData)
#    fromGeoms = createGeomData(fromGeoData)
#    ## The tree is built from the fromData so the tree query returns Indices of the fromData that intersect each toData polygon
#    geomTree = STRtree(fromGeoms)
#
#    startTime = time.time()
#    runStartTime = time.time()
#    ###=== For each element being converted into...
#    for index, thisToGeom in enumerate(toGeoms):
#        runStartTime = printProgress(runStartTime,index,len(toGeoms))
#        overlappingFromGeoms = geomTree.query(thisToGeom)
#        for thisFromGeom in overlappingFromGeoms:
#            if thisFromGeom.intersects(thisToGeom):
#                ## get the overlap area and add it to this toGeom
#    #            overlapArea = thisFromGeom.intersection(thisToGeom).area
#                overlapProportion = (thisFromGeom.intersection(thisToGeom).area / thisToGeom.area)  ## the percent of the toGeom that is covered by this free geom
#                thisFromGeoData = fromGeoData.loc[[thisFromGeom.idx]]
#    #            print(overlapArea)
#                for thisVar in thisVarList:
#                    if toGeoData.at[thisToGeom.idx, thisVar] == None:
#                        toGeoData.at[thisToGeom.idx, thisVar] = thisFromGeoData[thisVar].values[0] * overlapProportion
#                    else:
#                        toGeoData.at[thisToGeom.idx, thisVar] += overlapProportion
#
#    #                toGeoData.at[thisToGeom.idx, thisVar] += overlapArea
#    #                if toGeoData.at[thisToGeom.idx,thisVar] > 54127:
#    #                    print("hexIndex:",thisToGeom.idx,"   var:",thisVar,"   value:", toGeoData.at[thisToGeom.idx,thisVar])
#
#    ###=== If the toData is chome data, then only the lowest level has been filled in, so now aggregate up
##    if 'lowestLevel' in list(toGeoData):
##        print("  -- Aggregating Variables Upwards")
##        ###--- Aggregate up from chome to their Oaza (using only Oaza that have chome..i.e., not lowest level oaza)
##        toGeoData = aggregateUpValues(toGeoData, thisVarList)
##        toGeoData = aggregateUpPercents(toGeoData, thisVarList)
##        ###--- Add a percent area variable for each area variable for the chome data
##        for thisVar in thisVarList:
##            toGeoData['percent'+thisVar] = toGeoData.apply(lambda row: percentValue(row[thisVar], row['landArea']), axis=1)
#
#    toGeoData = toGeoData.to_crs(standardCRS)  ## Convert back to naive geometries before returning the data
#    print("==== Completed Conversion in",rnd((time.time()-startTime)/60,2),"minutes ====")
#    return toGeoData









####========================================================================
#####=========================== FOR PLOTTING ==============================
####========================================================================
def ceilInt(number):
    if isinstance(number, collections.Sequence):
        return [int(np.ceil(x)) for x in number]
    else:
        return int(np.ceil(number))

def round1000s(someNumber):
    return int(np.round(someNumber, decimals=-3))

def round100s(someNumber):
    return int(np.round(someNumber, decimals=-2))

def round10s(someNumber):
    return int(np.round(someNumber, decimals=-1))

def normalizeDataPoint(thisValue, dataMin, dataMax):
    return (thisValue - dataMin)/(dataMax - dataMin)

def scaleDataPoint(thisProportion, dataMin, dataMax):
    return dataMin + (thisProportion * (dataMax - dataMin))

def scaleVariable(thisData, thisVariable, thisLevel):
    return thisData[thisVariable].min() + thisLevel * (thisData[thisVariable].max() - thisData[thisVariable].min() )

def normRGB(Red, Green, Blue, A=1.0):
    return (Red / 255.0, Green / 255.0, Blue / 255.0, A)

###=== Convert colors to hex format for the visualization layer config
def rgb2hex(r, g, b, a=1):
    baseColor = f"#{r:02x}{g:02x}{b:02x}"
    opacity = str(makeInt(a * 100)) if a < 1 else ""
    return baseColor + opacity

def makeColorMap(listOfValues,listOfColors):
    norm=plt.Normalize(min(listOfValues),max(listOfValues))
    tuples = list(zip(map(norm,listOfValues), listOfColors))
    return LinearSegmentedColormap.from_list("", tuples, N=512)


####========================================================================
#####=========================== FOR MAPPING =============================
####========================================================================

def makeLegendTicks(theData, thisVariable, numTicks=4, minVal=None, maxVal=None):
    minVal = theData[thisVariable].min() if minVal == None else minVal
    maxVal = theData[thisVariable].max() if maxVal == None else maxVal
#    thisData = list(theData[thisVariable])
#    dataMin = np.ceil(min(thisData))
#    dataMax = np.floor(max(thisData))
    tickList = []
    for thisTick in range(numTicks):
        ### If the range is small, just use the values
        if maxVal - minVal < numTicks:
            if thisTick == 0:
                tickList.append(rnd(minVal))
            elif thisTick == numTicks-1:
                tickList.append(rnd(maxVal))
            else:
                tickList.append(rnd(scaleDataPoint(thisTick/(numTicks-1), minVal, maxVal)))
        ### If the range is large enough,
        else:
            if thisTick == 0:
                tickList.append(np.ceil(minVal))
            elif thisTick == numTicks-1:
                tickList.append(np.floor(maxVal))
            else:
                if maxVal > 30000:
                    tickList.append(round1000s(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
                elif maxVal > 3000:
                    tickList.append(round100s(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
                elif maxVal > 300:
                    tickList.append(round10s(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), np.floor(maxVal))))
                else:
                    tickList.append(makeInt(scaleDataPoint(thisTick/(numTicks-1), np.ceil(minVal), maxVal)))
    return tickList

def add_basemap(ax, zoom, url='https://a.basemaps.cartocdn.com/light_all/tileZ/tileX/tileY.png'):
#    url='https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png'
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, url=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))

def add_basemap2(ax, zoom, url='https://{s}.basemaps.cartocdn.com/rastertiles/light_all/{z}/{x}/{y}.png'):
    xmin, xmax, ymin, ymax = ax.axis()
    basemap, extent = ctx.bounds2img(xmin, ymin, xmax, ymax, zoom=zoom, source=url)
    ax.imshow(basemap, extent=extent, interpolation='bilinear')
    # restore original x/y limits
    ax.axis((xmin, xmax, ymin, ymax))

def makeHexMap(theData, theVariable, theVariableName, theColormap, theLegendColormap, minVal=None, maxVal=None, numTicks=4, fileIndex=""):
    fig, ax = plt.subplots(1, figsize=(10, 7))
    minVal = theData[theVariable].min() if minVal == None else minVal
    maxVal = theData[theVariable].max() if maxVal == None else maxVal
    xMin, yMin, xMax, yMax = theData['geometry'].total_bounds
#    print("[",xMin,",", yMin,",", xMax,",", yMax,"]")
#    xMin, yMin, xMax, yMax = [15467054.4135, 4232200.62482, 15575946.2476, 4286645.82715] ##itosanken?
#    xMin, yMin, xMax, yMax = [15497318.54462218, 4232088.870202411, 15584674.986741155, 4289158.680551786]  ## Tokyo Main with Elevation
    ax.set_xlim(xMin, xMax )
    ax.set_ylim(yMin, yMax )
    ax = theData.plot(ax=ax, column=theVariable, cmap=theColormap, vmin=minVal, vmax=maxVal)
#    fig.tight_layout()
    #tileSource = 'https://maps.wikimedia.org/osm-intl/tileZ/tileX/tileY.png'
#    tileSource = ctx.sources.ST_TONER_LITE
#    add_basemap(ax, zoom=11, url=tileSource)
    add_basemap(ax, zoom=11)
    #plt.axis('scaled')
    plt.axis('equal')
    #fig.suptitle("Total Population Aged 15-65", fontname="Arial", fontsize=18, y=-0.1)
    scalarmappable = cm.ScalarMappable(cmap=theLegendColormap)
    scalarmappable.set_clim(minVal, maxVal)
    scalarmappable.set_array(theData[theVariable])
    cbaxes = inset_axes(ax, width=0.2, height=2, loc='lower left', bbox_transform=ax.transAxes, bbox_to_anchor=(0.06, 0.1, 0.1, 0.4), borderpad = 0.)
    cbaxes.set_alpha(0.75)
    cbar = plt.colorbar(scalarmappable, cax=cbaxes, ticks=makeLegendTicks(theData,theVariable,numTicks,minVal,maxVal), orientation='vertical')
    cbaxes.yaxis.set_ticks_position("left")
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel(theVariableName, rotation=270)
    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.show()
    fig.savefig("../Map Images/"+theVariable+"-Map"+fileIndex+".png", dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig("../Map Images/"+theVariable+"-Map"+fileIndex+"sm.png", dpi=72, transparent=True, bbox_inches = 'tight', pad_inches = 0)

####=== Plot geometry data on a map
def makeGeoDataMap(theData, theVariable, theVariableName, theColormap, theLegendColormap=None, minVal=None, maxVal=None, numTicks=4, fileIndex="", figWidth=10, figHeight=7, outputDPI=150, totalBounds=None):

    fig, ax = plt.subplots(1, figsize=(figWidth, figHeight))
    minVal = theData[theVariable].min() if minVal == None else minVal
    maxVal = theData[theVariable].max() if maxVal == None else maxVal
    totalBounds = list(theData['geometry'].total_bounds) if totalBounds == None else totalBounds
    xMin, yMin, xMax, yMax = totalBounds
#    print("[",xMin,",", yMin,",", xMax,",", yMax,"]")
    ax.set_xlim(xMin, xMax )
    ax.set_ylim(yMin, yMax )
    ax = theData.plot(ax=ax, column=theVariable, cmap=theColormap, vmin=minVal, vmax=maxVal)
    add_basemap2(ax, zoom=11)
    #plt.axis('scaled')
    plt.axis('equal')
#    plt.axis((xMin, yMin, xMax, yMax))
    #fig.suptitle("Total Population Aged 15-65", fontname="Arial", fontsize=18, y=-0.1)

    ###=== If theLegendColormap==False, then don't put a legend at all.
    ###=== If the theLegendColormap is not specified, then use the same colormap.
    ###=== this is only necessary until we can figure out how to get a legend that supports alpha
    if theLegendColormap != False:
        if theLegendColormap == None:
            theLegendColormap = theColormap
        scalarmappable = cm.ScalarMappable(cmap=theLegendColormap)
        scalarmappable.set_clim(minVal, maxVal)
        scalarmappable.set_array(theData[theVariable])
        cbaxes = inset_axes(ax, width=0.2, height=2, loc='lower right', bbox_transform=ax.transAxes, bbox_to_anchor=(0.06, 0.1, 0.1, 0.4), borderpad = 0.)
        cbaxes.set_alpha(0.75)
        cbar = plt.colorbar(scalarmappable, cax=cbaxes, ticks=makeLegendTicks(theData,theVariable,numTicks,minVal,maxVal), orientation='vertical')
        cbaxes.yaxis.set_ticks_position("left")
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.set_ylabel(theVariableName, rotation=270)

    ax.axis('off')
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    plt.show()
    fig.savefig("../Map Images/"+theVariable+"-Map"+fileIndex+".png", dpi=outputDPI, transparent=True, bbox_inches = 'tight', pad_inches = 0)



####========================================================================
def makeKeplerMap(theData, thisVarList, theName="someData", theConfig=None, mappingArea='in23Wards'):
    import keplergl

    dataType = 'Other'
    ###=== Some data we always want to include, depending on the file, then add the variables of interest to the base and reduce
    thisVarList = [thisVarList] if isinstance(thisVarList, str) else thisVarList
    if 'addressCode' in list(theData):
        dataType = 'chomeData'
        ###=== Check if theData has the core data, and if not add it
        if 'geometry' not in list(theData):
            theData = pd.merge(loadCoreData(dataType), theData, on='addressCode')
            theData = theData[theData['lowestLevel'] == True]
            thisVarList.extend(('geometry', 'addressName', 'addressCode', 'landArea'))

    if 'hexNum' in list(theData):
        dataType = 'hexData'
        if 'geometry' not in list(theData):
            theData = pd.merge(loadCoreData(dataType), theData, on='hexNum')
            thisVarList.extend(('geometry', 'hexNum'))
    else:
        thisVarList.append('geometry')


    ###=== Check if the dataframe is already geopandas; if not, make it one.
    if isinstance(list(theData['geometry'])[0], str):
        theData = convertToGeoPandas(theData)

    ##-- Reduce the data to the geographical area of interest
    if ((mappingArea == 'in23Wards') & ('in23Wards' in list(theData))):
        theData = theData[theData['in23Wards'] == True]
    if ((mappingArea == 'inTokyoMain') & ('inTokyoMain' in list(theData))):
        theData = theData[theData['inTokyoMain'] == True]


    ###==== Check the polygons are all valid, ...some may be made invalid by changing the CRS, so fix them.
    for thisIndex in theData.index.values:
        if theData.at[thisIndex, 'geometry'].is_valid == False:
            theData.at[thisIndex, 'geometry'] = theData.at[thisIndex, 'geometry'].buffer(0)

    for thisVar in thisVarList:
        if thisVar not in list(theData):
            thisVarList.remove(thisVar)

    ###------------------------- Safe up to here
#    print(theData.head())
#    print(thisVarList)

#    theData = theData[thisVarList]  ##-- Reduce the data to the columns of interest  ##-- this generates a recursive depth error.
#    print(theData.head())
#    print(thisVarList)


    if theConfig == None:
        kmap = keplergl.KeplerGl(height=400, data={theName: theData})
    else:
        kmap = keplergl.KeplerGl(height=400, data={theName: theData}, config=theConfig)
#    print("writing html")
    kmap.save_to_html(file_name="../Map Images/"+dataType+"-"+theName+".html")


####=============================================================================
####================= CREATE VARIABLE LISTS FOR VIZENGINE =======================
####=============================================================================

#thisVarList = ['elevationMin', 'elevationMean', 'elevationMax', 'slopeMin', 'slopeMean', 'slopeMedian', 'slopeMax']
#addVarsToLocatorDict(thisVarList, "Geography", dataType='hexData')

#addVarsToLocatorDict('totalPopulation', "Core", dataType='hexData')
##
#print(readPickleFile('../Data/DataMasters/variableLocatorDict.pkl')['hexData'])

#print(list(getDataForTopic("Population")))

#allVarList = getVisVarNames("Crime", dataType='hexData') + getVisVarNames("Economics", dataType='hexData') + getVisVarNames("Environment", dataType='hexData') + getVisVarNames("Geography", dataType='hexData') + getVisVarNames("Population", dataType='hexData') + getVisVarNames("Transportation", dataType='hexData')
#
#print(allVarList)
#print("Num hexData variables for far:", len(allVarList))

#varsToUse = ['Hex_CrimeTotalRate', 'Hex_CrimeFelonyRobberyRate', 'Hex_CrimeFelonyOtherRate', 'Hex_CrimeViolentWeaponsRate', 'Hex_CrimeViolentAssaultRate', 'Hex_CrimeViolentInjuryRate', 'Hex_CrimeViolentIntimidationRate', 'Hex_CrimeViolentExtortionRate', 'Hex_CrimeTheftBurglarySafeRate', 'Hex_CrimeTheftBurglaryEmptyHomeRate', 'Hex_CrimeTheftBurglaryHomeSleepingRate', 'Hex_CrimeTheftBurglaryHomeUnnoticedRate', 'Hex_CrimeTheftBurglaryOtherRate', 'Hex_CrimeTheftVehicleRate', 'Hex_CrimeTheftMotorcycleRate', 'Hex_CrimeTheftPickPocketRate', 'Hex_CrimeTheftPurseSnatchingRate', 'Hex_CrimeTheftBagLiftingRate', 'Hex_CrimeTheftOtherRate', 'Hex_CrimeOtherMoralIndecencyRate', 'Hex_CrimeOtherOtherRate', 'Hex_NumJobs', 'Hex_NumCompanies', 'Hex_GreenArea', 'Hex_NoiseMin', 'Hex_NoiseMean', 'Hex_NoiseMax', 'Hex_PercentCommercial', 'Hex_PercentIndustrial', 'Hex_PercentResidential', 'Hex_MeanPercentLandCoverage', 'Hex_MeanTotalPercentLandCoverage', 'Hex_ElevationMin', 'Hex_ElevationMean', 'Hex_ElevationMax', 'Hex_SlopeMin', 'Hex_SlopeMean', 'Hex_SlopeMedian', 'Hex_SlopeMax', 'Hex_NumHouseholds', 'Hex_Pop_Total_A', 'Hex_Pop_0-19yr_A', 'Hex_Pop_20-69yr_A', 'Hex_Pop_70yr+_A', 'Hex_Pop_20-29yr_A', 'Hex_Pop_30-44yr_A', 'Hex_Pop_percentForeigners', 'Hex_Pop_percentChildren', 'Hex_Pop_percentMale', 'Hex_Pop_percentFemale', 'Hex_Pop_percent30-44yr', 'Hex_TimeToTokyo', 'Hex_TimeAndCostToTokyo']
#
#for thisVar in varsToUse:
#    print('''        "'''+thisVar+'''": {
#            "colors": "white2red",
#            "reverse": false,
#            "type": "standardized",
#            "interpolate": true
#        },''')
#
#
#for thisVar in varsToUse:
#    print("               '"+thisVar+"',")









####========================================================================
#####==================== GETTING ELEVATION DATA ===========================
####========================================================================

###=== Returns the lowest index closest point's index value from a dataframe of locations to a single point
def getClosestPoint(originLat, originLon, nodes):
    dists = np.array([euclideanDistance(originLon, originLat, lon, lat) for lon, lat in zip(nodes.lon, nodes.lat)])
    return dists.argmin()

###=== Get the 5m elevation for a particular point (which can only needs a single block even if it's exactly on an edge)
def getLatLonElevation(thisLat, thisLon, thisBoundaryDict):
    thisPoint = Point(thisLon, thisLat)
    for k,v in thisBoundaryDict.items():
        if thisPoint.intersects(v["geometry"]):
            thisBlock = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
            gridGeoms = list(thisBlock['geometry'])
            gridValues = list(thisBlock['elevation'])
            for index, geom in enumerate(gridGeoms):
                gridGeoms[index].idx = gridValues[index]
            gridTree = STRtree(gridGeoms)
            thisGrid = gridTree.query(thisPoint)
            return thisGrid[0].idx

def addLatLonElevations(theData, thisBoundaryDict):
    runStartTime = time.time()
    theData.loc[:,'thisPoint'] = theData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    theData['elevation'] = [None] * len(theData)
    for k,v in thisBoundaryDict.items():
        runStartTime = printProgress(runStartTime,k,len(thisBoundaryDict))
        thisBlock = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
        gridGeoms = list(thisBlock['geometry'])
        gridValues = list(thisBlock['elevation'])
        for index, geom in enumerate(gridGeoms):
            gridGeoms[index].idx = gridValues[index]
        gridTree = STRtree(gridGeoms)

        relevantData = theData[theData['thisPoint'].intersects(v['geometry'])]
        theseNodes = list(relevantData.index.values)
        for index,thisNode in enumerate(theseNodes):
            thisPoint = theData.loc[thisNode,'thisPoint']
            thisGrid = gridTree.query(thisPoint)
            theData.at[thisNode,'elevation'] = thisGrid[0].idx

    return theData


####==== Generate a list of (x,y) values along a straight line connecting two geoPoints
####==== The y-values are the elevations sampled from the 5m grid data at chosen intervals along the line
def getLineElevations(lat1, lon1, lat2, lon2, thisBoundaryDict, pointInterval = 5):
    thisLine = LineString([Point(lon1, lat1), Point(lon2, lat2)])
    allData = gp.GeoDataFrame()
    ##collect data tiles for all areas where the line goes
    for k,v in thisBoundaryDict.items():
        if thisLine.intersects(v["geometry"]):
            thisData = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
            thisData = thisData[thisData["geometry"].intersects(thisLine)]
            allData = gp.GeoDataFrame(pd.concat([allData,thisData], ignore_index=True))

    gc.collect()
    if len(allData) > 0:
#        print(len(allData))
        ###=== Create Rtree from all the needed data tiles
        gridGeoms = list(allData['geometry'])
        gridValues = list(allData['elevation'])
        for index, geom in enumerate(gridGeoms):
            gridGeoms[index].idx = gridValues[index]
        gridTree = STRtree(gridGeoms)

        theDistance = makeInt(haversineDist(lon1, lat1, lon2, lat2))
#        print("theDistance",theDistance)
        numPoints = makeInt(theDistance / pointInterval)
        pathPoints = [thisLine.interpolate(i/float(numPoints - 1), normalized=True) for i in range(numPoints)]
#        print(pathPoints)

#        pathGrids = gp.overlay(allData, thisLine, how='intersection')
#        print(pathGrids)
#        return pathGrids
        pathElevations = []
        for index,thisPoint in enumerate(pathPoints):
            try:
                pathElevations.append([index * pointInterval, np.round(gridTree.query(thisPoint)[0].idx,1)])
            except:
                pathElevations.append([index * pointInterval, 0])  #### Assign a value of 0 when crossing a location not in elevation data
        return pathElevations
    else:
        return [0]

####==== Return the y-values after applying a moving average smoothing operation
def getMovingAverageValues(Xs, Ys, smoothFactor = 5):
    aveSmooth_yValues = np.array(Ys)
    for i in range(smoothFactor,len(aveSmooth_yValues)-smoothFactor):
        aveSmooth_yValues[i] = np.mean(aveSmooth_yValues[i-smoothFactor:i+smoothFactor])
    return aveSmooth_yValues

####==== Generate polynomial fit points that include the first and last point
def getPolynomialFitValues(Xs, Ys, polyRank = 15):
    def polyFit(x, *params):
        return np.poly1d(params)(x)
    sigma = np.ones(len(Xs))   ## Create a sequence of weights to force matching on first/last point
    sigma[[0, -1]] = 0.01           ## Set the weights of the first and last value very small (meaning strong)
    ####=== start with a high rank polynomial, and gradually decrease to linear if optimization doesn't converge
    try:
        polyFitFunc, _ = curve_fit(polyFit, Xs, Ys, np.zeros(polyRank), sigma=sigma, maxfev=5000)
    except:
        try:
            polyFitFunc, _ = curve_fit(polyFit, Xs, Ys, np.zeros(10), sigma=sigma, maxfev=5000)
        except:
            try:
                polyFitFunc, _ = curve_fit(polyFit, Xs, Ys, np.zeros(5), sigma=sigma, maxfev=5000)
            except:
                polyFitFunc = np.polyfit([Xs[0], Xs[-1]], [Ys[0], Ys[-1]], 1)
    return polyFit(Xs, *polyFitFunc)

####==== Calculate the slope in degrees from changes in Y and X values (in the same units, e.g. meters)
def getSlopeAngle(deltaY,deltaX):
    return abs(np.rad2deg(np.arctan2(deltaY, deltaX)))

####==== For a list of X and Y values, return the maxSlope, meanSlope, medianSlope, minSlope
def getSlopeStats(Xs, Ys):
    allSlopes = [getSlopeAngle(Ys[i+1] - Ys[i], Xs[i+1] - Xs[i]) for i,x in enumerate(Xs[:-1])]
    return (rnd(np.max(allSlopes)), rnd(np.mean(allSlopes)), rnd(np.median(allSlopes)), rnd(np.min(allSlopes)))

####==== Make a straight line between two points, use
def lineElevationProfile(point1, point2, thisName, saveLoc=None):
    boundaryDict = readPickleFile('../Data/Altitude/Elevation5mWindowFiles/boundaryDict.pkl')
    gc.collect()
    startTime = time.time()
    pointInterval = 5    ##distance between point samples in meters
    elevationList = getLineElevations(point1[0], point1[1], point2[0], point2[1], boundaryDict, pointInterval)
    #print(elevationList)
    pathFindTime = np.round((time.time()-startTime),2)

    ####--- Actual Values
    xValues, yValues = zip(*elevationList)

    fig, ax = plt.subplots(figsize=(15, 3.5))
    plt.plot(xValues, yValues, c=normRGB(169,169,169,0.75), label='Raw Data')  ##gray

    ####--- Moving window average smoothing
    plt.plot(xValues, getMovingAverageValues(xValues, yValues), c=normRGB(220,20,60,0.55), label='Moving Average')  ##red

    ####--- Polynomial Fit Smoothing
    #poly = np.polyfit(xValues, yValues, 5)
    #poly_yValues = np.poly1d(poly)(xValues)
    #plt.plot(xValues, poly_yValues)
    poly_yValues = getPolynomialFitValues(xValues, yValues)
    plt.plot(xValues, poly_yValues, c=normRGB(30,144,255,0.75), label='Polynomial Fit')  ##blue

    ####----Get information about the slopes (using the fitted polynomial)
    maxSlope, meanSlope, medianSlope, minSlope = getSlopeStats(xValues, poly_yValues)
    plt.title(thisName+" | Slopes: max="+str(maxSlope)+"  mean="+str(meanSlope)+"  median="+str(medianSlope)+"  min="+str(minSlope), fontsize=16)
    ####--- Change legend location so it's less likely to cover the data
    if yValues[-1] > yValues[0]:
        plt.legend(loc="lower right")
    else:
        plt.legend(loc="upper right")
    plt.xlabel("Distance (m)", fontsize=15)
    plt.ylabel("Elevation (m)", fontsize=15)
    plt.show()
    if saveLoc != False:
        saveLoc = '../Map Images/' if saveLoc == None else saveLoc
        fig.savefig(saveLoc+'elevationProfile-'+thisName+'.png', dpi=150, transparent=True, bbox_inches='tight')
    print("--Found Path Elevation in",pathFindTime,"seconds")




####========================================================================
####========================== NETWORK ALGORITHMS ==========================
####========================================================================

def convertNetworkCRS(thisNetwork, fromCRS, toCRS):
    edgesData = [{'source': e[0], 'target': e[1], **e[2]} for e in thisNetwork.edges(data=True)]
    # edgesDF = gp.GeoDataFrame([{'geometry': Point(e['x1'], e['y1']), 'target_geom': Point(e['x2'], e['y2'])} for e in edgesData])
    edgesDF = gp.GeoDataFrame({'geometry': [LineString([Point(e['x1'], e['y1']), Point(e['x2'], e['y2'])]) for e in edgesData]})
    edgesDF.crs = fromCRS
    edgesDF = edgesDF.to_crs(toCRS)
    for i in range(len(edgesDF)):
        (x1, y1), (x2, y2) = edgesDF.iloc[i, 0].coords
        edge = edgesData[i]
        source = edge['source']
        target = edge['target']
        thisNetwork.edges[source, target]['x1'] = x1
        thisNetwork.edges[source, target]['y1'] = y1
        thisNetwork.edges[source, target]['x2'] = x2
        thisNetwork.edges[source, target]['y2'] = y2
        thisNetwork.nodes[source]['lon'] = x1
        thisNetwork.nodes[source]['lat'] = y1
        thisNetwork.nodes[target]['lon'] = x2
        thisNetwork.nodes[target]['lat'] = y2

        if i % 100000 == 0:
            print(f"{i} edges processed")

###=== from https://stackoverflow.com/questions/47177493/python-point-on-a-line-closest-to-third-point
def nearestPointOnLine(startNodePoint, endNodePoint, otherPoint):
    # Returns the nearest point on a given line and its distance
    x1, y1 = startNodePoint
    x2, y2 = endNodePoint
    x3, y3 = otherPoint
    if startNodePoint == endNodePoint:
        return x1, y1, euclideanDistance(x1, y1, x3, y3)

    dx, dy = x2 - x1, y2 - y1
    det = dx * dx + dy * dy
    a = ( (dy * (y3 - y1)) + (dx * (x3 - x1)) ) / det
    # Corner cases
    if a >= 1:
        return x2, y2, euclideanDistance(x2, y2, x3, y3)
    elif a <= 0:
        return x1, y1, euclideanDistance(x1, y1, x3, y3)
    newpx = x1 + a * dx
    newpy = y1 + a * dy
    return newpx, newpy, euclideanDistance(newpx, newpy, x3, y3)

def findNearestEdge(potentialNearestEdges, otherPoint):
    # Use the min heap to get the nearest edge --> log(N) time query (but still takes ~206 hours)
    otherPoint = list(otherPoint.centroid.coords)[0]
    # start = time.time()
    heap = []
    for edge in potentialNearestEdges:
        startNodeID = int(edge.idx.split('|')[0])
        endNodeID = int(edge.idx.split('|')[1])
        startNodeCoords, endNodeCoords = list(edge.coords)
        x3, y3, d = nearestPointOnLine(startNodeCoords, endNodeCoords, otherPoint)
        heapq.heappush(heap, (d, [(x3, y3), startNodeCoords, endNodeCoords, startNodeID, endNodeID]))
    try:
        theNode_endpoints = heap[0]
    except Exception as e:
        print(e)
        print(otherPoint)
    return theNode_endpoints

# Input a network and some point data (in pandas DF) with lat/lon columns
# Return the edge geometry STRTree and the GeoDataFrame with a given buffer
def geomize(thisNetwork, storeData, thisBuffer=125):
    edgesData = [{'source': e[0], 'target': e[1], **e[2]} for e in thisNetwork.edges(data=True)]
    edgesDF = gp.GeoDataFrame({'geometry': [LineString([Point(e['x1'], e['y1']), Point(e['x2'], e['y2'])]) for e in edgesData],
                                                         'source|target': [str(e['source']) + "|" + str(e['target']) for e in edgesData]})

    edgesGeoms = list(edgesDF['geometry'])
    geomsIdx = list(edgesDF['source|target'])
    for index, geom in enumerate(edgesGeoms):
        geom.idx = geomsIdx[index]
    geomTree = STRtree(edgesGeoms)

    storeData.loc[:, 'geometry'] = storeData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    storeData = gp.GeoDataFrame(storeData)
    storeData.crs = standardCRS
    storeData = storeData.to_crs(areaCalcCRS)
    storeData.loc[:, 'geometry'] = storeData.geometry.map(lambda geom: geom.buffer(thisBuffer))

    storeGeoms = createGeomData(storeData)

    storeData.loc[:, 'lon'] = storeData.geometry.map(lambda x: list(x.centroid.coords)[0][0])
    storeData.loc[:, 'lat'] = storeData.geometry.map(lambda x: list(x.centroid.coords)[0][1])

    return geomTree, storeGeoms, storeData

def findNearestEdgeFromNetwork(thisNetwork, storeData):
    start = time.time()

    # buffer = 125 wasn't big enough...
    geomTree, storeGeoms, geoStoreData = geomize(thisNetwork, storeData, thisBuffer=100)
    N = storeData.shape[0]
    savestep = int(N * 0.01)

    print("Preprocessing done!")

    selectedStores = set()
    nodeID = max(thisNetwork.nodes()) + 1
    for i, thisGeom in enumerate(storeGeoms):
        theStoreCoords = list(thisGeom.centroid.coords)[0]
        # theStoreData = geoStoreData.loc[(geoStoreData.lon - theStoreCoords[0] < 1e-5) & (geoStoreData.lat - theStoreCoords[1] < 1e-5)]
        theStoreData = geoStoreData.loc[(~geoStoreData["index"].isin(selectedStores)) & (geoStoreData.lon == theStoreCoords[0]) & (geoStoreData.lat == theStoreCoords[1])]
        if theStoreData.shape[0] == 0:
            continue # Skip this iteration since the location was chosen already
            # print("No store found. Using the index..")
            # theStoreData = geoStoreData.iloc[thisGeom.idx, :]
        for idx in theStoreData["index"]:
            selectedStores.add(idx)
        # print(f"# of stores at location {theStoreCoords}: {theStoreData.shape[0]}")
        # Turn the geoDF into a list of dicts
        theStoreData = theStoreData.drop(['index', 'geometry', 'lat', 'lon'], axis=1).to_dict('records')
        theStoreData = {attr: [storeDict[attr] for storeDict in theStoreData] for attr in theStoreData[0].keys()}

        potentialNearestEdges = geomTree.query(thisGeom)
        if len(potentialNearestEdges) == 0:
            print("Potential Nearest Edges:", len(potentialNearestEdges))
            thisGeom = Point(theStoreCoords[0], theStoreCoords[1]).buffer(10000)
            potentialNearestEdges = geomTree.query(thisGeom)
            print("Augmented:", len(potentialNearestEdges))
        storeDist, theNode_endpoints = findNearestEdge_v3(potentialNearestEdges, thisGeom)
        theNode, startNode, endNode, startNodeID, endNodeID = theNode_endpoints

        # Add the store node
        storeNodeID = nodeID + 1
        thisNetwork.add_node(
            storeNodeID,
            lon=theStoreCoords[0],
            lat=theStoreCoords[1],
            modality='store',
            storeDist=storeDist,
            **theStoreData
        )

        # Add the store access edge
        if theNode == startNode:
                thisNetwork.add_edge(startNodeID, storeNodeID,
                                     modality='store',
                                     distance=storeDist,
                                     x1=startNode[0],
                                     y1=startNode[1],
                                     x2=theStoreCoords[0],
                                     y2=theStoreCoords[1],
                                     elevationGain=0)
                thisNetwork.add_edge(storeNodeID, startNodeID,
                                     modality='store',
                                     distance=storeDist,
                                     x1=theStoreCoords[0],
                                     y1=theStoreCoords[1],
                                     x2=startNode[0],
                                     y2=startNode[1],
                                     elevationGain=0)
#                 nx.set_node_attributes(thisNetwork, {storeNodeID: {'elevation': thisNetwork.nodes[startNodeID]['elevation']}})
        elif theNode == endNode:
                thisNetwork.add_edge(endNodeID, storeNodeID,
                                     modality='store',
                                     distance=storeDist,
                                     x1=endNode[0],
                                     y1=endNode[1],
                                     x2=theStoreCoords[0],
                                     y2=theStoreCoords[1],
                                     elevationGain=0)
                thisNetwork.add_edge(storeNodeID, endNodeID,
                                     modality='store',
                                     distance=storeDist,
                                     x1=theStoreCoords[0],
                                     y1=theStoreCoords[1],
                                     x2=endNode[0],
                                     y2=endNode[1],
                                     elevationGain=0)
#                 nx.set_node_attributes(thisNetwork, {storeNodeID: {'elevation': thisNetwork.nodes[endNodeID]['elevation']}})
        else:
            # This part still needs updates on distances and elevations
            thisNetwork.add_node(
                nodeID,
                lon=theNode[0],
                lat=theNode[1],
                parentSourceID=startNodeID,
                parentTargetID=endNodeID,
                parentSource=startNode,
                parentTarget=endNode
            )
            thisNetwork.add_edge(storeNodeID, nodeID,
                                 modality='store',
                                 distance=storeDist,
                                 x1=theStoreCoords[0],
                                 y1=theStoreCoords[1],
                                 x2=theNode[0],
                                 y2=theNode[1],
                                 elevationGain=0)
            thisNetwork.add_edge(nodeID, storeNodeID,
                                 modality='store',
                                 distance=storeDist,
                                 x1=theNode[0],
                                 y1=theNode[1],
                                 x2=theStoreCoords[0],
                                 y2=theStoreCoords[1],
                                 elevationGain=0)
#             nx.set_node_attributes(thisNetwork, {storeNodeID: {'elevation': thisNetwork.nodes[endNodeID]['elevation']}})
        nodeID += 2
        if i % savestep == 0:
            print(f"Processed {i} rows. {i//savestep}% done in {time.time() - start} seconds.")

    end = time.time()
    print(f"Total time: {end - start} seconds")

####==== Input a network and new set of points (maybe nodes) and connect the points to the network at the closest edge (or node).
###=== newPoints could be a list of coords, a list of Point objects, a list of nodes, or even a (geo)pandas dataframe of the stores with lat/lon and other attributes
###=== I'm thinking the pandas input is probably best for us.
def extendNetworkToPoint(thisNetwork, newPoints):

    ###=== Convert newPoints into geopandas dataframe with a geometry column of Point objects from the lat lon
    newPoints = convertToGeoPandas(newPoints, toCRS=areaCalcCRS)


    ###=== For each row of newPoints, find the edge or node of thisNetwork that is closest to it.




        ###=== Create a node for this point, add its attributes from Pandas.

        ###=== If the closest things is a node, just connect to it and add attributes to the edge
        ###=== If the closest thing is an edge, find the location along the edge, create a node there, connect to previous endPoints, remove old link, connect to newPoint



    return thisNetwork






















#####======================================== END OF FILE ===========================================