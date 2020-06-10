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
import os

import numpy as np
import pandas as pd
pd.set_option('display.max_columns', None)
import geopandas as gp

from shapely import wkt
from shapely.geometry import shape
from shapely.geometry import Point
from shapely.geometry import LineString
from shapely.geometry.polygon import Polygon
from shapely.geometry.multipolygon import MultiPolygon

from shapely.strtree import STRtree
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


####=================== FOR LOADING AND SAVING FILES ===================
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

def fullFileName(filename):
    return os.path.join(os.environ['DATA_PATH'], filename)
        
def convertToGeoPandas(thisData, toCRS=None, forMapping=False):
    thisData = gp.GeoDataFrame(thisData)
    thisData['geometry'] = thisData['geometry'].apply(wkt.loads)
    thisData.crs = 'epsg:4326'                     ##== 4326 corresponds to "naive geometries", normal lat/lon values
    toCRS = 3857 if forMapping == True else toCRS  ##== epsg=3857 is needed for mapping, 
    if toCRS != None:        
        thisData = thisData.to_crs(epsg=toCRS)
    gc.collect()
    return thisData

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

def writePickleFile(theData,filePathName):    
    with open(filePathName, 'wb') as fp:
        pickle.dump(theData, fp)
        
def readCSV(fileName, fillNaN=''):    
    return pd.read_csv(fileName, encoding='utf-8').fillna(fillNaN)

def writeCSV(data, fileName):
    data.to_csv(fileName, sep=',', encoding='utf-8-sig', index=False)
    
def writeGeoJSON(data, fileName):
    data.to_file(fileName, driver="GeoJSON")
    
def readGeoPandasCSV(fileName, fillNaN='', toCRS=None, forMapping=False):    
    return convertToGeoPandas(pd.read_csv(fileName, encoding='utf-8').fillna(fillNaN), toCRS=toCRS, forMapping=forMapping)
    
####==== Write Pandas CSV File to S3
#import s3fs
#def writePandasToCSV(theData,theFilename,theBucket = 'geodata-processing'):
#    s3 = s3fs.S3FileSystem(anon=False)
#    with s3.open(theBucket+'/'+theFilename+'.csv','w') as f:
#        theData.to_csv(f)
        
def loadHexDataFromNetwork(thisNetwork, forMapping=False):
    ###=== Isolate the hex nodes
    thisNetwork = thisNetwork.subgraph([node for node,attr in thisNetwork.nodes(data=True) if attr['modality']=="hex"])
    ###=== convert node properties to pandas dataframe
    hexData = pd.DataFrame.from_dict(dict(thisNetwork.nodes(data=True)), orient='index')
    ###=== Convert pandas geometry data into actual geodata
    hexData = convertToGeoPandas(hexData, forMapping=forMapping)
    #xMin, yMin, xMax, yMax = hexData['geometry'].total_bounds 
#    print()
    gc.collect()
    return hexData

def loadHexDataFromNetworkFile(filename, forMapping=False):
    loadHexDataFromNetwork(readJSONDiGraph(filename), forMapping=forMapping)

####=============================================================================
####==== It looks like this converts a dataframe with multipolygons into one with only polygons, copying rows
def explode(indata):
    indf = gp.GeoDataFrame.from_file(indata)
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
####---Distance in meters between two points
def distanceBetweenLonLats(x1,y1,x2,y2):
   return np.round(geopy.distance.distance(geopy.Point(y1,x1), geopy.Point(y2,x2)).m, decimals=0)

def Euclidean_distance(px1,py1, px2, py2):
     return math.sqrt((px2-px1)**2 + (py2-py1)**2)
 
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

def reportRunTime(startTime):
    print("Time to complete:", np.round((time.time() - startTime)/60, decimals=1),"minutes")    
 
def printProgress(thisStartTime,index,totalNum):
    oneBlock = makeInt(totalNum / 100)  ## approximately how many are in 1%
    if index % oneBlock == 0:
        newStartTime = time.time()
        if newStartTime - thisStartTime < 60:
            print("--Analyzing",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): previous batch took",rnd((newStartTime - thisStartTime),1),"seconds")
        else:
            print("--Analyzing",index,"of",totalNum,"(",rnd(100*(index/totalNum),1),"%): previous batch took",rnd((newStartTime - thisStartTime)/60,2),"minutes")
        return newStartTime
    else:
        return thisStartTime
    
def makeNumberString(number, length=1):
    return str(int(float(number))).zfill(length)
    




####=============================================================================
####========================================================================
####========================================================================


####=============================================================================
####=== Take data using one set of polygons and interpolate it to a different set of polygons using overlap percents.
####=== For example, this can be used to convert between hex <=> chome, but also other "free polygon" data
def interpolateGeoData(fromGeoData,toGeoData,thisVarList):
    ###=== Seed the toGeometry with Nones for each variable used from the fromGeometry
    for thisVar in thisVarList:
        toGeoData[thisVar] = [None] * len(toGeoData)
    #    toGeometry[thisVar] = np.zeros(len(toGeometry)).tolist()
        
    ###=== Add index values as geometry attributes, then Build an R-tree of the hexes
    toGeoms = list(toGeoData['geometry'])
    for index, geom in enumerate(toGeoms):
        toGeoms[index].idx = index
    #print([geom.idx for geom in hexGeoms[:10]])
    toGeomTree = STRtree(toGeoms)
    
    fromGeoms = list(fromGeoData['geometry'])
    for index, geom in enumerate(fromGeoms):
        fromGeoms[index].idx = index  
    
    startTime = time.time()
    runStartTime = time.time()
    for index, thisFromGeom in enumerate(fromGeoms):
        runStartTime = printProgress(runStartTime,index,len(fromGeoms))    
        thisFromGeoData = fromGeoData.iloc[[thisFromGeom.idx]]
        overlappingHexes = toGeomTree.query(thisFromGeom)
        for thisToGeom in overlappingHexes:
            overlapProportion = (thisFromGeom.intersection(thisToGeom).area / thisToGeom.area)
            for thisVar in thisVarList:
                if toGeoData.at[thisToGeom.idx, thisVar] == None:
                    toGeoData.at[thisToGeom.idx, thisVar] = thisFromGeoData[thisVar].values[0] * overlapProportion
                else:
                    toGeoData.at[thisToGeom.idx, thisVar] += thisFromGeoData[thisVar].values[0] * overlapProportion
    #            print("hexIndex:",thisHexGeom.idx,"   var:",thisVar,"   value:", hexData.at[thisHexGeom.idx,thisVar][0])
    print("==== Completed Conversion in",rnd((time.time()-startTime)/60,2),"minutes ====")
    return toGeoData

####=== Old version, should be identical in functionality
#def convertToHex(thisHexData,thisPolyData,thisVarList):
#    ###=== Seed the hexdata with Nones for each variable used from the polyData
#    for thisVar in thisVarList:
#        thisHexData[thisVar] = [None] * len(thisHexData)
#    #    hexData[thisVar] = np.zeros(len(hexData)).tolist()
#        
#    ###=== Add index values as geometry attributes, then Build an R-tree of the hexes
#    hexGeoms = list(thisHexData['geometry'])
#    for index, geom in enumerate(hexGeoms):
#        hexGeoms[index].idx = index
#    #print([geom.idx for geom in hexGeoms[:10]])
#    hexTree = STRtree(hexGeoms)
#    
#    polyGeoms = list(thisPolyData['geometry'])
#    for index, geom in enumerate(polyGeoms):
#        polyGeoms[index].idx = index  
#    
#    startTime = time.time()
#    runStartTime = time.time()
#    for index, thisPolygon in enumerate(polyGeoms):
#        runStartTime = printProgress(runStartTime,index,len(polyGeoms))    
#        polygonData = thisPolyData.iloc[[thisPolygon.idx]]
#        overlappingHexes = hexTree.query(thisPolygon)
#        for thisHexGeom in overlappingHexes:
#            overlapProportion = (thisPolygon.intersection(thisHexGeom).area / thisHexGeom.area)
#            for thisVar in thisVarList:
#                if thisHexData.at[thisHexGeom.idx, thisVar] == None:
#                    thisHexData.at[thisHexGeom.idx, thisVar] = polygonData[thisVar].values[0] * overlapProportion
#                else:
#                    thisHexData.at[thisHexGeom.idx, thisVar] += polygonData[thisVar].values[0] * overlapProportion
#    #            print("hexIndex:",thisHexGeom.idx,"   var:",thisVar,"   value:", hexData.at[thisHexGeom.idx,thisVar][0])
#    print("==== Completed Hex Conversion in",rnd((time.time()-startTime)/60,2),"minutes ====")
#    return thisHexData
#####=============================================================================







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
    
def makeHexMap(theData, theVariable, theVariableName, theColormap, theLegendColormap, minVal=None, maxVal=None, numTicks=4, fileIndex=""):
    fig, ax = plt.subplots(1, figsize=(10, 7))
    minVal = theData[theVariable].min() if minVal == None else minVal
    maxVal = theData[theVariable].max() if maxVal == None else maxVal
#    xMin, yMin, xMax, yMax = theData['geometry'].total_bounds 
#    print("[",xMin,",", yMin,",", xMax,",", yMax,"]")
#    xMin, yMin, xMax, yMax = [ 15467054.4135 , 4232200.62482 , 15575946.2476 , 4286645.82715 ]
    xMin, yMin, xMax, yMax = [ 15497318.54462218 , 4232088.870202411 , 15584674.986741155 , 4289158.680551786 ]  ## Tokyo Main with Elevation
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
    fig.savefig("G:/My Drive/Map Images/"+theVariable+"-Map"+fileIndex+".png", dpi=300, transparent=True, bbox_inches = 'tight', pad_inches = 0)
    fig.savefig("G:/My Drive/Map Images/"+theVariable+"-Map"+fileIndex+"sm.png", dpi=72, transparent=True, bbox_inches = 'tight', pad_inches = 0)





####========================================================================
#####==================== GETTING ELEVATION DATA ===========================
####========================================================================

###=== Get the 5m elevation for a particular point
def getLatLonElevation(thisLat, thisLon, thisBoundaryDict):
    thisPoint = Point(thisLon, thisLat)
    for k,v in thisBoundaryDict.items():
        if thisPoint.intersects(v["geometry"]):     
            dataFilename = fullFileName('Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
            thisData = readPickleFile(dataFilename)     
            gridGeoms = list(thisData['geometry'])
            gridValues = list(thisData['elevation'])
            for index, geom in enumerate(gridGeoms):
                gridGeoms[index].idx = gridValues[index]                         
            gridTree = STRtree(gridGeoms)
            thisGrid = gridTree.query(thisPoint)
            return thisGrid[0].idx

def addLatLonElevations(theData, thisBoundaryDict, savepoint=True):
    runStartTime = time.time()
    theData.loc[:,'thisPoint'] = theData.apply(lambda row: Point(row['lon'], row['lat']), axis=1)
    theData['elevation'] = [None] * len(theData)
    for k,v in thisBoundaryDict.items():
        runStartTime = printProgress(runStartTime,k,len(thisBoundaryDict))
        thisBlock = readPickleFile(fullFileName('Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl'))
        gridGeoms = list(thisBlock['geometry'])
        gridValues = list(thisBlock['elevation'])
        for index, geom in enumerate(gridGeoms):
            gridGeoms[index].idx = gridValues[index] 
        gridTree = STRtree(gridGeoms)
        # relevantData = theData[theData.apply(lambda row: row['thisPoint'].intersects(v['geometry']), axis=1)]
        relevantData = theData.loc[theData['thisPoint'].map(lambda p: p.intersects(v['geometry']))]
        print("RELEVANT DATA")
        print(relevantData)
        # relevantData = theData[theData['thisPoint'].intersects(v['geometry'])] 
        theseNodes = list(relevantData.index.values)
        for index,thisNode in enumerate(theseNodes):
            try:
                thisPoint = theData.loc[thisNode,'thisPoint']
                thisGrid = gridTree.query(thisPoint)
                theData.at[thisNode,'elevation'] = thisGrid[0].idx
            except IndexError as e:
                print("Ignoring", e)
                pass
            if savepoint and index % 1000 == 0:
                print("\t- Saving progress to savepoint_elevationData.csv...")
                theData.to_csv("savepoint_elevationData.csv", index=False)
        
    return theData

####==== Generate a list of (x,y) values along a straight line connecting two geoPoints
####==== The y-values are the elevations sampled from the 5m grid data at chosen intervals along the line
def getLineElevations(lat1, lon1, lat2, lon2, thisBoundaryDict, pointInterval = 5):
    thisLine = LineString([Point(lon1, lat1), Point(lon2, lat2)])
    allData = gp.GeoDataFrame()
    ##collect data tiles for all areas where the line goes
    for k,v in thisBoundaryDict.items():
        if thisLine.intersects(v["geometry"]):
            dataFilename = fullFileName('Altitude/Elevation5mWindowFiles/elevationData-5mGrid-'+str(k)+'.pkl')
            thisData = readPickleFile(dataFilename) 
            thisData = thisData[thisData["geometry"].intersects(thisLine)]
            allData = gp.GeoDataFrame(pd.concat([allData,thisData], ignore_index=True))
                                     
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
    boundaryFilename = fullFileName('Altitude/Elevation5mWindowFiles/boundaryDict.pkl')
    boundaryDict = readPickleFile(boundaryFilename) 
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
        saveLoc = 'G:/My Drive/Map Images/' if saveLoc == None else saveLoc
        fig.savefig(saveLoc+'elevationProfile-'+thisName+'.png', dpi=150, transparent=True, bbox_inches='tight')
    print("--Found Path Elevation in",pathFindTime,"seconds")
    

































#####======================================== END OF FILE ===========================================