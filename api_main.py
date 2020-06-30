"""
This API throws processed data back to frontend.
Put this file under the same level directory as your "data" directory.

@author: ShutoAraki
@date: 06/23/2020
"""

import io
import glob
import os
import pickle
import time
import numpy as np
import pandas as pd
from pydantic import BaseModel
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware


# ===== HELPER FUNCTIONS =====
def readJSON(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

def readPickleFile(filePathName):
    with open (filePathName, 'rb') as fp:
        return pickle.load(fp)

def getDtypes(dataType='hex'):
    if dataType == 'hex':
        return {'addressCode': 'str', 'prefCode': 'str', 'cityCode': 'str', 'oazaCode': 'str', 'chomeCode': 'str', 'addressName': 'str', 'prefName': 'str', 'districtName': 'str', 'cityName': 'str', 'oazaName': 'str', 'chomeName': 'str'}
    elif dataType == 'chome':
        return {'addressCode': 'str', 'prefCode': 'str', 'cityCode': 'str', 'oazaCode': 'str', 'chomeCode': 'str', 'addressName': 'str', 'prefName': 'str', 'districtName': 'str', 'cityName': 'str', 'oazaName': 'str', 'chomeName': 'str'}
    else:
        return {'modality': 'str'}

# DATA_DIR = "../Data/DataMasters" 
DATA_DIR = os.path.join(os.environ['DATA_PATH'], "DataMasters")

app = FastAPI()

origins = [
        "http://localhost:8080"
]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"]
        )

# Load the core data upon server bootup
print("Loading the initial core data")
chomeCore = readPickleFile(os.path.join(DATA_DIR, "chomeData-Core.pkl"))
hexCore = readPickleFile(os.path.join(DATA_DIR, "hexData-Core.pkl"))
print("Done loading two cores!")

class SelectedColumns(BaseModel):
    data_type: str
    selectedColumns: list

def getDataFromCols(filename, cols):
    dataType = filename.split('/')[-1].split('-')[0]
    if dataType == "hexData":
        coreData = hexCore
    elif dataType == "chomeData":
        coreData = chomeCore
    else:
        raise Exception(f"Invalid data type: {dataType}")

    mergeKey = "hexNum" if dataType == "hexData" else "addressCode"
    cols.append(mergeKey)
    thisData = pd.read_csv(filename, usecols=cols)
    combinedData = pd.merge(coreData, thisData, on=mergeKey)
    return combinedData


@app.get("/fetch")
def read_files():
    startTime = time.time()
    cwd = os.getcwd()
    csv_files = os.path.join(cwd, f"{DATA_DIR}/*.csv")
    raw_result = glob.glob(csv_files)
    result = list(filter(lambda x: 'Core' not in x, raw_result))
    filenames = [os.path.split(file_path)[-1] for file_path in result]
    ans_dict = [{'id': index, 'name': name[:-4]} for index, name in enumerate(filenames)]

    config_dir = os.path.join(DATA_DIR, 'kepler_configs')
    default_config = readJSON(os.path.join(config_dir, 'defaultConfig.json'))
    for d in ans_dict:
        filename = d['name'].lower()
        # Config settings based on keywords contained in the filename
        if 'crime' in filename:
            d['config'] = readJSON(os.path.join(config_dir, 'crimeConfig.json'))
        elif 'green' in filename:
            d['config'] = readJSON(os.path.join(config_dir, 'greenMapConfig.json'))
        else:
            d['config'] = default_config
        full_filename = result[d['id']]
        mergeKey = "hexNum" if "hex" in filename else "addressCode"
        all_columns = pd.read_csv(full_filename, nrows=1).columns.tolist()
        try:
            all_columns.remove(mergeKey)
        except:
            print(f"Removing {mergeKey} from {full_filename} does not work: {all_columns}")
        d['all_columns'] = all_columns

    print(f"Initial fetch took {time.time() - startTime} secs")

    return {"filenames": ans_dict}


# Fetch data
@app.get("/fetch/{data_info}")
async def fetch_data(data_info: str):
    startTime = time.time()

    data_list = data_info.split('|')
    if len(data_list) != 2:
        print("Invalid data format:\n", data_list)
    data_name, cols = data_list[0], data_list[1].split(',')
    print("DATA NAME", data_name)
    if data_name[-4:] == '.csv':
        filename = os.path.join(DATA_DIR, data_name)
    else:
        filename = os.path.join(DATA_DIR, data_name + '.csv')
    stream = io.StringIO()


    if 'all' in cols:
        dataset = pd.read_csv(filename)
    else:
        dataset = getDataFromCols(filename, cols)


    dataset.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={data_name}.csv"


    print(f"Loading {len(dataset.columns)} columns from {data_name} took {time.time() - startTime} secs: {filename}")

    return response


# @app.post("/columns/")
# async def select_by_columns(request: SelectedColumns):
#     # The respnose dict is {'selectedColumns': ['example.csv:[]']}
#     columns = request.dict()
#     print(columns)
#     for full_col_name in columns['selectedColumns']:
#         broken_down = full_col_name.split(':')
#         if len(broken_down) != 2:
#             print("Invalid column naming =>", full_col_name)
#         filename, colname = broken_down[0], broken_down[1]
#         # Create a dict entry for filename -> [colnames]
#         if filename in columns:
#             columns[filename].add(colname)
#         else:
#             columns[filename] = {colname}
#     del columns['selectedColumns']
#     return columns

@app.post("/agg_fetch/")
# example request
# {
#     "data_type": "Hex",
#     "selectedColumns": ["hexData-Crime:crimeTotalRate","hexData-Environment:noiseMax","hexData-Population:pop_0-4yr_A"]
# }
async def aggregate_fetch(request: SelectedColumns):
    startTime = time.time()
    columns = request.dict()
    print(columns)
    for full_col_name in columns['selectedColumns']:
        broken_down = full_col_name.split(':')
        if len(broken_down) != 2:
            print("Invalid column naming =>", full_col_name)
        filename, colname = broken_down[0], broken_down[1]
        # Create a dict entry for filename -> {colnames}
        if filename in columns:
            columns[filename].add(colname)
        else:
            columns[filename] = {colname}
    data_type = columns['data_type'].lower() # Either "hex" or "chome"
    mergeKey = "hexNum" if data_type == "hex" else "addressCode"
    combinedData = hexCore if data_type == "hex" else chomeCore
    del columns['selectedColumns']
    del columns['data_type']

    readTime = time.time()

    for topic, cols in columns.items():
        filename = os.path.join(DATA_DIR, topic + '.csv')
        usecols = cols | {mergeKey}
        thisData = pd.read_csv(filename, usecols=usecols, dtype=getDtypes(data_type))
        combinedData = pd.merge(combinedData, thisData, on=mergeKey)

    print("Read time:", time.time() - readTime)
    responseTime = time.time()
    
    topics = ''.join(list(map(lambda x: x.split("-")[-1], columns.keys())))
    stream = io.StringIO()
    combinedData.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={topics}.csv"

    print("Response time:", time.time() - responseTime)
    
    print(f"Loading {len(combinedData.columns)} columns took {time.time() - startTime} secs")
    return response