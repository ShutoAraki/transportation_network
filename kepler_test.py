import pandas as pd
import keplergl
import os

chome_df = pd.read_csv("data/chomeDataMaster_v002.csv")
hex_df = pd.read_csv("data/GridData/hexData-Master_v001.csv")
elevation_filename = os.path.join(os.environ['DATA_PATH'], "Altitude/ElevationHexData/elevation-HexData-TokyoMain.csv")
elevation_df = pd.read_csv(elevation_filename)

map_config = {
  "version": "v1",
  "config": {
    "visState": {
      "filters": [],
      "layers": [
        {
          "id": "1dg0f9p",
          "type": "point",
          "config": {
            "dataId": "chome_master_data",
            "label": "Point",
            "color": [
              18,
              147,
              154
            ],
            "columns": {
              "lat": "lat",
              "lng": "lon",
              "altitude": None
            },
            "isVisible": False,
            "visConfig": {
              "radius": 10,
              "fixedRadius": False,
              "opacity": 0.8,
              "outline": False,
              "thickness": 2,
              "strokeColor": None,
              "colorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radiusRange": [
                0,
                50
              ],
              "filled": True
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
            "colorField": None,
            "colorScale": "quantile",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "sizeField": None,
            "sizeScale": "linear"
          }
        },
        {
          "id": "zqg4djn",
          "type": "hexagon",
          "config": {
            "dataId": "chome_master_data",
            "label": "chome_data_customer_pop",
            "color": [
              77,
              181,
              217
            ],
            "columns": {
              "lat": "lat",
              "lng": "lon"
            },
            "isVisible": False,
            "visConfig": {
              "opacity": 0.3,
              "worldUnitSize": 1,
              "resolution": 8,
              "colorRange": {
                "name": "ColorBrewer PuBu-8",
                "type": "sequential",
                "category": "ColorBrewer",
                "colors": [
                  "#fff7fb",
                  "#ece7f2",
                  "#d0d1e6",
                  "#a6bddb",
                  "#74a9cf",
                  "#3690c0",
                  "#0570b0",
                  "#034e7b"
                ]
              },
              "coverage": 1,
              "sizeRange": [
                0,
                500
              ],
              "percentile": [
                0,
                100
              ],
              "elevationPercentile": [
                0,
                99
              ],
              "elevationScale": 13.5,
              "colorAggregation": "sum",
              "sizeAggregation": "average",
              "enable3d": True
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
              "name": "pop_30-44yr_A",
              "type": "integer"
            },
            "colorScale": "quantile",
            "sizeField": {
              "name": "pop_30-44yr_A",
              "type": "integer"
            },
            "sizeScale": "linear"
          }
        },
        {
          "id": "r7afjb8",
          "type": "geojson",
          "config": {
            "dataId": "hex_master_data",
            "label": "hex_data",
            "color": [
              119,
              110,
              87
            ],
            "columns": {
              "geojson": "geometry"
            },
            "isVisible": True,
            "visConfig": {
              "opacity": 1,
              "strokeOpacity": 0.8,
              "thickness": 0.5,
              "strokeColor": None,
              "colorRange": {
                "name": "ColorBrewer PuBu-6",
                "type": "sequential",
                "category": "ColorBrewer",
                "colors": [
                  "#f1eef600",
                  "#d0d1e6",
                  "#a6bddb",
                  "#74a9cf",
                  "#2b8cbe",
                  "#045a8d"
                ]
              },
              "strokeColorRange": {
                "name": "Global Warming",
                "type": "sequential",
                "category": "Uber",
                "colors": [
                  "#5A1846",
                  "#900C3F",
                  "#C70039",
                  "#E3611C",
                  "#F1920E",
                  "#FFC300"
                ]
              },
              "radius": 10,
              "sizeRange": [
                0,
                10
              ],
              "radiusRange": [
                0,
                50
              ],
              "heightRange": [
                0,
                500
              ],
              "elevationScale": 5,
              "stroked": False,
              "filled": True,
              "enable3d": False,
              "wireframe": False
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
              "name": "numJobs",
              "type": "integer"
            },
            "colorScale": "quantize",
            "sizeField": None,
            "sizeScale": "linear",
            "strokeColorField": None,
            "strokeColorScale": "quantile",
            "heightField": None,
            "heightScale": "linear",
            "radiusField": None,
            "radiusScale": "linear"
          }
        }
      ],
      "interactionConfig": {
        "tooltip": {
          "fieldsToShow": {
            "chome_master_data": [
              "addressCode",
              "prefCode",
              "cityCode",
              "oazaCode",
              "chomeCode"
            ],
            "hex_master_data": [
              "numJobs",
              "numCompanies",
              "modality",
              "hexNum",
              "popTotal"
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
      "latitude": 35.679921690495966,
      "longitude": 139.48488607381515,
      "pitch": 0,
      "zoom": 9.49484659735802,
      "isSplit": False
    },
    "mapStyle": {
      "styleType": "uafryn",
      "topLayerGroups": {
        "label": True,
        "road": True,
        "border": False,
        "building": False,
        "water": False,
        "land": False
      },
      "visibleLayerGroups": {
        "label": True,
        "road": True,
        "building": True,
        "water": True
      },
      "threeDBuildingColor": [
        224.4071295378559,
        224.4071295378559,
        224.4071295378559
      ],
      "mapStyles": {
        "uafryn": {
          "accessToken": "pk.eyJ1Ijoic2h1dG9hcmFraSIsImEiOiJja2F4bGpwZGgwMXdoMnNwaTZwNzZ1N2ozIn0.4MK9evmXh1eQPTUauJQbMg",
          "custom": True,
          "icon": "https://api.mapbox.com/styles/v1/shutoaraki/ckaxlks630p1s1ilbdw4i26no/static/-122.3391,37.7922,9,0,0/400x300?access_token=pk.eyJ1Ijoic2h1dG9hcmFraSIsImEiOiJja2F4bGpwZGgwMXdoMnNwaTZwNzZ1N2ozIn0.4MK9evmXh1eQPTUauJQbMg&logo=false&attribution=false",
          "id": "uafryn",
          "label": "Shuto's Classic",
          "url": "mapbox://styles/shutoaraki/ckaxlks630p1s1ilbdw4i26no"
        }
      }
    }
  }
}

kmap = keplergl.KeplerGl(height=400,
                         data={'chome_master_data': chome_df, 'hex_master_data': hex_df, 'elevation_data': elevation_df},
                         config=map_config)

kmap.save_to_html(file_name="elevation_map.html")