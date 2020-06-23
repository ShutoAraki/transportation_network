"""
This API throws processed data back to frontend.
Put this file under the same level directory as your "data" directory.

@author: ShutoAraki
@date: 06/23/2020
"""

import os
import glob
import io
import numpy as np
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

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

'''
class Item(BaseModel):
    name: str
    price: float
    is_offer: bool = None


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.put("/items/{item_id}")
def update_item(item_id: int, item: Item):
    return {"item_name": item.name, "item_id": item_id}
'''
@app.get("/fetch")
def read_files():
    cwd = os.getcwd()
    csv_files = os.path.join(cwd, "data/*.csv")
    result = glob.glob(csv_files)
    filenames = [os.path.split(file_path)[-1] for file_path in result]
    ans_dict = [{'id': index, 'name': name} for index, name in enumerate(filenames)]
    return {"filenames": ans_dict}

# Fetch data
@app.get("/fetch/{data_name}")
async def fetch_data(data_name: str):
    if data_name[-4:] == '.csv':
        filename = f"file:///Users/s_araki/local_dev/data/{data_name}"
    else:
        filename = f"file:///Users/s_araki/local_dev/data/{data_name}.csv"
    dataset = pd.read_csv(filename)
    stream = io.StringIO()
    if 'node' in data_name:
        dataset = dataset.loc[:, ['lat', 'lon']]
    # elif 'link' in data_name:
    #     dataset = dataset.loc[:, ['x1', 'y1', 'x2', 'y2']]
    dataset.to_csv(stream, index = False)
    #dataset.to_json(stream)

    response = StreamingResponse(iter([stream.getvalue()]),
                                 media_type="text/csv"
                                )

    response.headers["Content-Disposition"] = f"attachment; filename={data_name}.csv"

    return response
