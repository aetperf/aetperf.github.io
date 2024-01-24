---
title: Create a routable pedestrian network with elevation WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
- OSM
- OSMnx
---


In this blog post, we will explore how to create a routable pedestrian network with elevation using Python and [OSMnx](https://osmnx.readthedocs.io/en/stable/). OSMnx is a Python library that allows you to retrieve [OpenStreetMap](https://www.openstreetmap.org/) (OSM) data and work with street networks.

Here's a summary of the steps followed in the blog post:

- Graph download with OSMnx
- Graph processing with Pandas
- Add node elevation with rasterio
- Plot the network
- Compute the edge walking time attribute
- Save the edges and nodes to file

As the underlying motivation for this graph processing flow, our aim is to empower the network for the execution of path algorithms.

## System and package versions

We are operating on Python version 3.11.7 and running on a Linux x86_64 machine.


    contextily             : 1.5.0
    fiona                  : 1.9.5
    geopandas              : 0.14.1
    matplotlib             : 3.8.2
    numpy                  : 1.26.3
    osmnx                  : 1.8.1
    pandas                 : 2.1.4
    rasterio               : 1.3.9
    shapely                : 2.0.2


## Imports

```python
import contextily as cx
import matplotlib.pyplot as plt
import numpy as np
import osmnx as ox
import pandas as pd
import rasterio as rio
from shapely.geometry import LineString, Point

DEM_FP = "./lyon_dem.tif"
OUTPUT_NODES_FP = "./nodes_lyon_pedestrian_network.GeoJSON"
OUTPUT_EDGES_FP = "./edges_lyon_pedestrian_network.GeoJSON"
```

The file paths for Digital Elevation Model (DEM), and the output nodes and edges GeoJSON files are defined just above.

## Graph download with OSMnx

We begin by defining a bounding box using the coordinates in EPSG:4326 (WGS 84). This box encapsulates the geographical area of interest:

```python
bbox = (4.446716, 45.515971, 5.193787, 45.970243)
```

Now, let's utilize the [`ox.graph_from_bbox`](https://osmnx.readthedocs.io/en/stable/user-reference.html#osmnx.graph.graph_from_bbox) function with specific parameters, besides the bounding box:

- `network_type (string {"all_private", "all", "bike", "drive", "drive_service", "walk"})`: what type of street network to get
- `simplify (bool)`: If set to `True`, the graph topology is simplified using the `simplify_graph` function.
- `retain_all (bool)`: When `True`, the function returns the entire graph even if it's not fully connected. Otherwise, it retains only the largest weakly connected component.
- `truncate_by_edge (bool)`: Enabling this option retains nodes outside the bounding box if at least one of a node's neighbors is within the bounding box.


```python
%%time
G = ox.graph_from_bbox(
    north=bbox[3],
    south=bbox[1],
    east=bbox[0],
    west=bbox[2],
    network_type="walk",
    simplify=True,
    retain_all=True,
    truncate_by_edge=True,
)
```

    CPU times: user 1min 6s, sys: 1.19 s, total: 1min 7s
    Wall time: 1min 7s

The output graph is a [NetworkX](https://networkx.org/) object. We can explore some of the graph properties, such as whether it is directed:

```python
G.is_directed()
```


    True


Now we concert the graph into [GeoDataFrames](https://geopandas.org/en/stable/docs/reference/api/geopandas.GeoDataFrame.html) for nodes and edges, respectively:

```python
%%time
nodes_gdf, edges_gdf = ox.graph_to_gdfs(G)
```

    CPU times: user 9.25 s, sys: 141 ms, total: 9.39 s
    Wall time: 9.37 s



```python
nodes_gdf.columns
```


    Index(['y', 'x', 'street_count', 'highway', 'ref', 'geometry'], dtype='object')




```python
edges_gdf.columns
```


    Index(['osmid', 'highway', 'oneway', 'reversed', 'length', 'geometry', 'lanes',
           'ref', 'maxspeed', 'bridge', 'name', 'service', 'width', 'junction',
           'access', 'tunnel', 'est_width', 'area'],
          dtype='object')


In the next section, we will process the graph, removing many useless features for our pedestrian routing use-case.

## Graph processing with Pandas


```python
# remove most important roads
edges_gdf = edges_gdf.loc[edges_gdf.highway != "trunk"]
edges_gdf = edges_gdf.loc[edges_gdf.highway != "trunk_link"]
```

```python
edges_gdf["highway"].value_counts()[:20]
```




    highway
    service                   118730
    residential               102618
    footway                    89894
    unclassified               66840
    tertiary                   34132
    track                      33426
    path                       31814
    secondary                  24810
    primary                    15702
    corridor                    4794
    [steps, footway]            4028
    living_street               3434
    pedestrian                  3046
    steps                       2286
    [track, unclassified]       1708
    [residential, track]        1406
    [residential, footway]      1356
    [path, track]               1192
    [footway, service]           825
    [service, footway]           787
    Name: count, dtype: int64




```python
# edge column selection and renaming
edges = edges_gdf[["geometry"]].reset_index(drop=False)
edges = edges.rename(columns={"u": "tail", "v": "head"})
edges.drop("key", axis=1, inplace=True)
edges.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tail</th>
      <th>head</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2087150</td>
      <td>3209480797</td>
      <td>LINESTRING (4.53724 45.74513, 4.53756 45.74519...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2087150</td>
      <td>2025220166</td>
      <td>LINESTRING (4.53724 45.74513, 4.53684 45.74510...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2087150</td>
      <td>3209480790</td>
      <td>LINESTRING (4.53724 45.74513, 4.53775 45.74517...</td>
    </tr>
  </tbody>
</table>
</div>




```python
# make sure we get each edge once in both directions

# we select one direction between each pair couples
edges["min_vert"] = edges[["tail", "head"]].min(axis=1)
edges["max_vert"] = edges[["tail", "head"]].max(axis=1)
edges.drop_duplicates(subset=["min_vert", "max_vert"], inplace=True)
edges.drop(["min_vert", "max_vert"], axis=1, inplace=True)

# we revert these edges and concatenate them
edges_reverse = edges.copy(deep=True)
edges_reverse[["tail", "head"]] = edges_reverse[["head", "tail"]]
edges_reverse.geometry = edges_reverse.geometry.map(lambda g: g.reverse())
edges = pd.concat((edges, edges_reverse), axis=0)

# cleanup
edges = edges.sort_values(by=["tail", "head"])
edges = edges.loc[edges["tail"] != edges["head"]]  # remove loops
edges.drop_duplicates(subset=["tail", "head"], inplace=True)
edges.reset_index(drop=True, inplace=True)
edges.shape
```




    (541394, 3)




```python
# node column selection and renaming
nodes = nodes_gdf[["geometry"]].copy(deep=True)
nodes = nodes.reset_index(drop=False)
nodes.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>osmid</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2087150</td>
      <td>POINT (4.53724 45.74513)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2087196</td>
      <td>POINT (4.50251 45.73066)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2087231</td>
      <td>POINT (4.47475 45.71289)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# reindex the nodes and update the edges
nodes["id"] = nodes.index
edges = pd.merge(
    edges, nodes[["id", "osmid"]], left_on="tail", right_on="osmid", how="left"
)
edges.drop(["tail", "osmid"], axis=1, inplace=True)
edges.rename(columns={"id": "tail"}, inplace=True)
edges = pd.merge(
    edges, nodes[["id", "osmid"]], left_on="head", right_on="osmid", how="left"
)
edges.drop(["head", "osmid"], axis=1, inplace=True)
edges.rename(columns={"id": "head"}, inplace=True)
edges = edges[["tail", "head", "geometry"]]
```


```python
edges.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tail</th>
      <th>head</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145061</td>
      <td>152458</td>
      <td>LINESTRING (5.17632 45.62181, 5.17643 45.62180...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>145061</td>
      <td>152462</td>
      <td>LINESTRING (5.17632 45.62181, 5.17622 45.62183...</td>
    </tr>
    <tr>
      <th>2</th>
      <td>145061</td>
      <td>166277</td>
      <td>LINESTRING (5.17632 45.62181, 5.17627 45.62179...</td>
    </tr>
  </tbody>
</table>
</div>




```python
edges.shape
```




    (541394, 3)




```python
nodes.drop("osmid", axis=1, inplace=True)
nodes.set_index("id", inplace=True)
nodes.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POINT (4.53724 45.74513)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POINT (4.50251 45.73066)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POINT (4.47475 45.71289)</td>
    </tr>
  </tbody>
</table>
</div>




```python
nodes.shape
```




    (206284, 1)



## Add node elevation with rasterio

In this section we are going to add the elevation attribute to the graph nodes. We are going to use a raster file prepared in a previous post:

- [Lyon's Digital Terrain Model with IGN Data](https://aetperf.github.io/2023/12/26/Lyon-s-Digital-Terrain-Model-with-IGN-Data.html)

We are going to query the raster file using [rasterio](https://rasterio.readthedocs.io/en/stable/). Note that it is also possible to [add node elevation within OSMnx](https://osmnx.readthedocs.io/en/stable/getting-started.html#working-with-elevation) using the [elevation module](https://osmnx.readthedocs.io/en/stable/internals-reference.html#osmnx-elevation-module): `osmnx.elevation.add_node_elevations_raster()`
```

First we need to convert the node coordinates to the same Coordinate Reference System (CRS) as the raster file, i.e. a projected CRS: EPSG:2154, RGF93 v1 / Lambert-93 -- France.

```python
nodes = nodes.to_crs("EPSG:2154")
nodes.head(3)
```


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geometry</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POINT (819515.290 6517333.579)</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POINT (816845.594 6515675.314)</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POINT (814724.346 6513661.646)</td>
    </tr>
  </tbody>
</table>
</div>




```python
# extract point coordinates in Lambert 93
lon_lam93 = nodes.geometry.apply(lambda p: p.x)
lat_lam93 = nodes.geometry.apply(lambda p: p.y)
```


```python
nodes = nodes.to_crs("EPSG:4326")
```


```python
point_coords = list(zip(lon_lam93, lat_lam93))
```


```python
dem = rio.open(DEM_FP)
```


```python
%%time
nodes["z"] = [x[0] for x in dem.sample(point_coords)]
```

    CPU times: user 2.51 s, sys: 240 ms, total: 2.75 s
    Wall time: 2.75 s



```python
nodes.loc[nodes["z"] <= -99999, "z"] = np.nan
nodes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>geometry</th>
      <th>z</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>POINT (4.53724 45.74513)</td>
      <td>323.760010</td>
    </tr>
    <tr>
      <th>1</th>
      <td>POINT (4.50251 45.73066)</td>
      <td>393.540009</td>
    </tr>
    <tr>
      <th>2</th>
      <td>POINT (4.47475 45.71289)</td>
      <td>424.230011</td>
    </tr>
    <tr>
      <th>3</th>
      <td>POINT (4.45839 45.70706)</td>
      <td>469.100006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>POINT (4.78359 45.74445)</td>
      <td>275.119995</td>
    </tr>
  </tbody>
</table>
</div>



## Plot the network


```python
ax = edges.plot(linewidth=0.2, alpha=0.7, figsize=(12, 10))
cx.add_basemap(
    ax, source=cx.providers.CartoDB.VoyagerNoLabels, crs=edges.crs.to_string()
)
_ = plt.axis("off")
```


<p align="center">
  <img width="1200" src="/img/2024-01-11_01/output_30_0.png" alt="Pedestrian network">
</p>      


## Compute the edge walking time attribute

https://en.wikipedia.org/wiki/Tobler%27s_hiking_function

$$v = 6 e^{-3.5 \left| \tan(\theta) + 0.05 \right|}$$


```python
def walking_speed_kmh(slope_deg):
    theta = np.pi * slope_deg / 180.0
    return 6.0 * np.exp(-3.5 * np.abs(np.tan(theta) + 0.05))
```


```python
x = np.linspace(-20, 20, 100)
y = np.array(list(map(walking_speed_kmh, x)))
```


```python
fig, ax = plt.subplots(figsize=(6, 4))
_ = plt.plot(x, y, alpha=0.7)
_ = ax.set(
    title="Tobler's hiking function",
    xlabel="slope (°)",
    ylabel="Walking speed ($km.h^{-1}$)",
)
```

<p align="center">
  <img width="400" src="/img/2024-01-11_01/output_34_0.png" alt="Tobler's hiking function">
</p>      


```python
edges = pd.merge(
    edges,
    nodes[["z"]].rename(columns={"z": "tail_z"}),
    left_on="tail",
    right_index=True,
    how="left",
)
edges = pd.merge(
    edges,
    nodes[["z"]].rename(columns={"z": "head_z"}),
    left_on="head",
    right_index=True,
    how="left",
)
```


```python
edges = edges.to_crs("EPSG:2154")
edges["length"] = edges.geometry.map(lambda g: g.length)
```


```python
edges.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tail</th>
      <th>head</th>
      <th>geometry</th>
      <th>tail_z</th>
      <th>head_z</th>
      <th>length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145061</td>
      <td>152458</td>
      <td>LINESTRING (869567.830 6504812.245, 869576.816...</td>
      <td>219.820007</td>
      <td>219.820007</td>
      <td>35.998225</td>
    </tr>
    <tr>
      <th>1</th>
      <td>145061</td>
      <td>152462</td>
      <td>LINESTRING (869567.830 6504812.245, 869560.425...</td>
      <td>219.820007</td>
      <td>220.320007</td>
      <td>23.203918</td>
    </tr>
    <tr>
      <th>2</th>
      <td>145061</td>
      <td>166277</td>
      <td>LINESTRING (869567.830 6504812.245, 869564.173...</td>
      <td>219.820007</td>
      <td>219.220001</td>
      <td>86.592815</td>
    </tr>
  </tbody>
</table>
</div>



```python
linestring = edges.iloc[0].geometry
linestring
```

<p align="center">
  <img width="200" src="/img/2024-01-11_01/output_38_0.svg" alt="A curvy edge">
</p> 


```python
def compute_straight_length(linestring):
    first = Point(linestring.coords[0])
    last = Point(linestring.coords[-1])
    straight_line = LineString([first, last])
    return straight_line.length
```


```python
compute_straight_length(linestring)
```




    34.511755994005895




```python
edges["straight_length"] = edges.geometry.map(lambda g: compute_straight_length(g))
```


```python
edges = edges.to_crs("EPSG:4326")
```


```python
edges.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tail</th>
      <th>head</th>
      <th>geometry</th>
      <th>tail_z</th>
      <th>head_z</th>
      <th>length</th>
      <th>straight_length</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145061</td>
      <td>152458</td>
      <td>LINESTRING (5.17632 45.62181, 5.17643 45.62180...</td>
      <td>219.820007</td>
      <td>219.820007</td>
      <td>35.998225</td>
      <td>34.511756</td>
    </tr>
    <tr>
      <th>1</th>
      <td>145061</td>
      <td>152462</td>
      <td>LINESTRING (5.17632 45.62181, 5.17622 45.62183...</td>
      <td>219.820007</td>
      <td>220.320007</td>
      <td>23.203918</td>
      <td>22.823563</td>
    </tr>
    <tr>
      <th>2</th>
      <td>145061</td>
      <td>166277</td>
      <td>LINESTRING (5.17632 45.62181, 5.17627 45.62179...</td>
      <td>219.820007</td>
      <td>219.220001</td>
      <td>86.592815</td>
      <td>84.532070</td>
    </tr>
  </tbody>
</table>
</div>




```python
edges.straight_length.min()
```




    0.011105097809826743




```python
def compute_slope(triangle_att):
    """tail_z, head_z, straight_length"""
    tail_z = triangle_att[0]
    head_z = triangle_att[1]
    straight_length = triangle_att[2]
    x = (head_z - tail_z) / straight_length
    x = np.amin([x, 1.0])
    x = np.amax([x, -1.0])
    theta = np.arcsin(x)
    theta_deg = theta * 180.0 / np.pi
    theta_deg = np.amin([theta_deg, 20.0])
    theta_deg = np.amax([theta_deg, -20.0])
    return theta_deg
```


```python
edges["slope_deg"] = edges[["tail_z", "head_z", "straight_length"]].apply(
    compute_slope, raw=True, axis=1
)
```


```python
ax = edges.slope_deg.plot.hist(bins=25, alpha=0.7)
_ = ax.set(title="Edge slope distribution", xlabel="Slope (°)")
```

<p align="center">
  <img width="800" src="/img/2024-01-11_01/output_47_0.png" alt="Edge slope distribution">
</p> 


```python
edges["walking_speed_kmh"] = edges.slope_deg.map(lambda s: walking_speed_kmh(s))
```


```python
edges["travel_time_s"] = 3600.0 * 1.0e-3 * edges["length"] / edges.walking_speed_kmh
```


```python
edges.drop(
    ["tail_z", "head_z", "length", "straight_length", "slope_deg", "walking_speed_kmh"],
    axis=1,
    inplace=True,
)
```


```python
edges.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>tail</th>
      <th>head</th>
      <th>geometry</th>
      <th>travel_time_s</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>145061</td>
      <td>152458</td>
      <td>LINESTRING (5.17632 45.62181, 5.17643 45.62180...</td>
      <td>25.729650</td>
    </tr>
    <tr>
      <th>1</th>
      <td>145061</td>
      <td>152462</td>
      <td>LINESTRING (5.17632 45.62181, 5.17622 45.62183...</td>
      <td>17.906953</td>
    </tr>
    <tr>
      <th>2</th>
      <td>145061</td>
      <td>166277</td>
      <td>LINESTRING (5.17632 45.62181, 5.17627 45.62179...</td>
      <td>60.373345</td>
    </tr>
  </tbody>
</table>
</div>



 ## Save the edges and nodes to file


```python
import fiona

fiona.supported_drivers
```




    {'DXF': 'rw',
     'CSV': 'raw',
     'OpenFileGDB': 'raw',
     'ESRIJSON': 'r',
     'ESRI Shapefile': 'raw',
     'FlatGeobuf': 'raw',
     'GeoJSON': 'raw',
     'GeoJSONSeq': 'raw',
     'GPKG': 'raw',
     'GML': 'rw',
     'OGR_GMT': 'rw',
     'GPX': 'rw',
     'Idrisi': 'r',
     'MapInfo File': 'raw',
     'DGN': 'raw',
     'PCIDSK': 'raw',
     'OGR_PDS': 'r',
     'S57': 'r',
     'SQLite': 'raw',
     'TopoJSON': 'r'}




```python
%%time
nodes.to_file(OUTPUT_NODES_FP, driver="GeoJSON", crs="EPSG:4326")
edges.to_file(OUTPUT_EDGES_FP, driver="GeoJSON", crs="EPSG:4326")
```

    CPU times: user 22.1 s, sys: 136 ms, total: 22.2 s
    Wall time: 22.2 s



```python
!ls -l *.GeoJSON
```

    -rw-rw-r-- 1 francois francois 161497552 Jan  4 21:26 edges_lyon_pedestrian_network.GeoJSON
    -rw-rw-r-- 1 francois francois  32088554 Jan  4 21:25 nodes_lyon_pedestrian_network.GeoJSON

