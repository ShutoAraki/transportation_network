"""
This API throws processed data back to frontend.
Put this file under the same level directory as your "data" directory.

@author: ShutoAraki
@date: 06/23/2020
"""

import os
import glob
import pickle
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

import time

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

class DataColumn(BaseModel):
    selectedColumns: list

def readJSON(filename):
    with open(filename, 'r') as f:
        data = f.read()
    return data

def readPickleFile(filePathName):
    with open (filePathName, 'rb') as fp:
        return pickle.load(fp)


@app.get("/fetch")
def read_files():
    startTime = time.time()
    cwd = os.getcwd()
    csv_files = os.path.join(cwd, f"{DATA_DIR}/*.csv")
    result = glob.glob(csv_files)
    filenames = [os.path.split(file_path)[-1] for file_path in result]
    ans_dict = [{'id': index, 'name': name[:-4]} for index, name in enumerate(filenames)]
    config_dir = os.path.join(DATA_DIR, 'kepler_configs')
    
    default_config = readJSON(os.path.join(config_dir, 'defaultConfig.json'))
    for d in ans_dict:
        filename = d['name'].lower()
        if 'crime' in filename:
            d['config'] = readJSON(os.path.join(config_dir, 'crimeConfig.json'))
        elif 'green' in filename:
            d['config'] = readJSON(os.path.join(config_dir, 'greenMapConfig.json'))
        else:
            d['config'] = default_config
        full_filename = result[d['id']]
        d['all_columns'] = pd.read_csv(full_filename, nrows=1).columns.tolist()

    print(f"Initial fetch took {time.time() - startTime} secs")

    return {"filenames": ans_dict}


# Fetch data
@app.get("/fetch/{data_info}")
async def fetch_data(data_info: str):
    startTime = time.time()

    is_pickle = False
    data_list = data_info.split('|')
    if len(data_list) != 2:
        print("Invalid data format:\n", data_list)
    data_name, cols = data_list[0], data_list[1].split(',')
    print("DATA NAME", data_name)
    if data_name[-4:] == '.csv':
        '''
        pickle_name = os.path.join(DATA_DIR, data_name[:-4] + '.pkl')
        if os.path.exists(pickle_name):
            is_pickle = True
            filename = pickle_name
        else:
            filename = f"./{DATA_DIR}/{data_name}"
            filename = os.path.join(DATA_DIR, data_name)
        '''
        filename = os.path.join(DATA_DIR, data_name)
    else:
        '''
        pickle_name = os.path.join(DATA_DIR, data_name + '.pkl')
        if os.path.exists(pickle_name):
            is_pickle = True
            filename = pickle_name
        else:
            filename = os.path.join(DATA_DIR, data_name + '.csv')
    ''' 
        filename = os.path.join(DATA_DIR, data_name + '.csv')
    stream = io.StringIO()
    if 'all' in cols:
        '''
        if is_pickle:
            dataset = readPickleFile(filename)
        else:
            dataset = pd.read_csv(filename)
        '''
        dataset = pd.read_csv(filename)
    else:
        '''
        if is_pickle:
            allDF = readPickleFile(filename)
            dataset = allDF.loc[:, cols]
        else:
            dataset = pd.read_csv(filename, usecols=cols)
       ''' 
        dataset = pd.read_csv(filename, usecols=cols)
    dataset.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename={data_name}.csv"

    print(f"Loading {len(dataset.columns)} columns from {data_name} took {time.time() - startTime} secs: {filename}; Is pickle?: {is_pickle}")

    return response

# Client throws something like this
#“/selective_fetch/[“example.csv:lat”,”example.csv:lat”,”another.csv:green”]”
# @app.get("/selective_fetch/{data_name}")
# async def fetch_data_selectively(data_name: str):


@app.post("/columns/")
async def select_by_columns(request: DataColumn):
    # The respnose dict is {'selectedColumns': ['example.csv:[]']}
    columns = request.dict()
    print(columns)
    for full_col_name in columns['selectedColumns']:
        broken_down = full_col_name.split(':')
        if len(broken_down) != 2:
            print("Invalid column naming =>", full_col_name)
        filename, colname = broken_down[0], broken_down[1]
        # Create a dict entry for filename -> [colnames]
        if filename in columns:
            columns[filename].add(colname)
        else:
            columns[filename] = {colname}
    del columns['selectedColumns']

    # Load the right csv and return the DataFrame
    '''
    totalDF = pd.concat([pd.read_csv(f"data/{filename}").loc[:, colnames] for filename, colnames in columns.items()])
    stream = io.StringIO() 
    totalDF.to_csv(stream, index=False)
    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv")
    response.headers["Content-Disposition"] = f"attachment; filename=total_data.csv"
    return response
    '''
    return columns