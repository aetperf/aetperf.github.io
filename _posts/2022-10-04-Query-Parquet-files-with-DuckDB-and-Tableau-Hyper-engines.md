---
title: Query Parquet files with DuckDB and Tableau Hyper engines
layout: post
comments: true
author: François Pacull & Romain Ferraton
tags: 
- Python
- Graph
- Network
- Pandas
- DuckDB
- Tableau Hyper
- SQL
---

<p align="center">
  <img width="600" src="/img/2022-10-04_01/parquet_logo.jpg" alt="query_1">
</p>


In this notebook, we are going to query some [*Parquet*](https://parquet.apache.org/) files with the following SQL engines:
- [*DuckDB*](https://duckdb.org/) : an in-process SQL OLAP database management system. We are going to use its [Python Client API](https://duckdb.org/docs/api/python/reference/) (MIT license).
- [*Tableau Hyper*](https://help.tableau.com/current/api/hyper_api/en-us/reference/sql/index.html) : an in-memory data engine. We are going to interact with this engine using the [tableauhyperapi](https://help.tableau.com/current/api/hyper_api/en-us/index.html) Python package (Proprietary License).

Both of these tools are optimized for Online analytical processing (OLAP). We do not want to modify the data but launch queries that require processing a large amount of data. DuckDB and Tableau Hyper make use of some vectorized engine and some amount of parallel processing, well suited for the columnar storage format of *Parquet* files. This is very well described in [this](https://duckdb.org/2021/06/25/querying-parquet.html) post from the DuckDB team. Here is a quote from this blog post:

> DuckDB will read the Parquet files in a streaming fashion, which means you can perform queries on large Parquet files that do not fit in your main memory.  

Tableau Hyper engine has the ability to read *Parquet* files using the [`external`](
 https://help.tableau.com/current/api/hyper_api/en-us/reference/sql/external-formats.html#EXTERNAL-FORMAT-PARQUET) keyword.

> External data can be read directly in a SQL query using the set returning function external. In this case, no Hyper table is involved, so such a query can even be used if no database is attached to the current session. 


The *Parquet* files correspond to a very specific use case, since they all describe some road networks from the US or Europe. The US road networks were imported in a previous post: [Download some benchmark road networks for Shortest Paths algorithms](https://aetperf.github.io/2022/09/22/Download-some-benchmark-road-networks-for-Shortest-Paths-algorithms.html). The Europe networks were downloaded from [this](https://i11www.iti.kit.edu/resources/roadgraphs.php) web page and converted to *Parquet* files. We are only going to use the edge table, not the node coordinates one. The SQL queries in this notebook are also very specific, in a sense that they are related to the graph theory domain. Here are the things that we are going to compute: 
1. occurence of parallel edges
2. vertex and edge counts
3. count of connected vertices
4. count of vertices with one incoming and one outgoing egde
5. degree distribution
For each query and SQL engine, we are going to measure the elapsed time. In this post, we **did not** measure the memory consumption.

**Notes**:
- The *Parquet* files are not compressed.
- both engines usually make use of their own optimized file format, e.g. `.hyper` files for *Tableau hyper*. However, they both support direct querying of *CSV* or *Parquet* files.
- We are going to use *DuckDB* and *Tableau Hyper* with the **default configuration**.
- Most of the SQL queries could probably be optimized, however we believe that they are efficient enough for the comparison purpose of this short post.
- In all the elapsed time bar charts, lower is better.


## Imports

Note that both tools are very easy to install:

```bash
$ pip install duckdb  
$ pip install tableauhyperapi  
```

Here are the imports:

```python
import os
from time import perf_counter

import duckdb
import pandas as pd
from pandas.testing import assert_frame_equal
from tableauhyperapi import Connection, HyperProcess, Telemetry

pd.set_option("display.precision", 2)  # Pandas float number display
TELEMETRY = Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU  # not sending telemetry data to Tableau
FS = (12, 6)  # figure size
ALPHA = 0.8  # figure transparency
```

Package versions:

    Python         : 3.10.6
    duckdb         : 0.5.1
    tableauhyperapi: 0.0.15530
    pandas         : 1.5.0

System information:

    OS             : Linux
    Architecture   : 64bit
    CPU            : Intel(R) Core(TM) i7-7700HQ CPU @ 2.80GHz
    CPU cores      : 8
    RAM            : 32GB


## Apache Parquet files


```python
names = ["NY", "BAY", "COL", "FLA", "NW", "NE", "CAL", "LKS", "E", "W", "CTR", "USA"]

stats = {}
parquet_graph_file_paths = {}
parquet_file_sizes = []
for name in names:
    parquet_graph_file_path = f"/home/francois/Data/Disk_1/DIMACS_road_networks/{name}/USA-road-t.{name}.gr.parquet"
    stats[name] = {}
    stats[name]["parquet_file_size_MB"] = (
        os.path.getsize(parquet_graph_file_path) * 1.0e-6
    )
    parquet_graph_file_paths[name] = parquet_graph_file_path

names_osmr = ["osm-bawu", "osm-ger", "osm-eur"]
names += names_osmr
for name in names_osmr:
    parquet_graph_file_path = (
        f"/home/francois/Data/Disk_1/OSMR/{name}/{name}.gr.parquet"
    )
    stats[name] = {}
    stats[name]["parquet_file_size_MB"] = (
        os.path.getsize(parquet_graph_file_path) * 1.0e-6
    )
    parquet_graph_file_paths[name] = parquet_graph_file_path
ordered_names = ["NY", "BAY", "COL", "FLA", "NW", "NE", "CAL", "osm-bawu", "LKS", "E", "W", "CTR", "osm-ger", "USA", "osm-eur"]
```

```python
stats_df = pd.DataFrame.from_dict(stats, orient="index")
stats_df.sort_values(by="parquet_file_size_MB", ascending=True, inplace=True)
stats_df
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
      <th>parquet_file_size_MB</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NY</th>
      <td>9.11</td>
    </tr>
    <tr>
      <th>BAY</th>
      <td>10.14</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>13.32</td>
    </tr>
    <tr>
      <th>FLA</th>
      <td>31.86</td>
    </tr>
    <tr>
      <th>NW</th>
      <td>34.43</td>
    </tr>
    <tr>
      <th>NE</th>
      <td>46.36</td>
    </tr>
    <tr>
      <th>CAL</th>
      <td>56.40</td>
    </tr>
    <tr>
      <th>LKS</th>
      <td>82.57</td>
    </tr>
    <tr>
      <th>E</th>
      <td>105.59</td>
    </tr>
    <tr>
      <th>osm-bawu</th>
      <td>106.99</td>
    </tr>
    <tr>
      <th>W</th>
      <td>184.64</td>
    </tr>
    <tr>
      <th>CTR</th>
      <td>428.21</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>702.06</td>
    </tr>
    <tr>
      <th>osm-ger</th>
      <td>730.79</td>
    </tr>
    <tr>
      <th>osm-eur</th>
      <td>6112.20</td>
    </tr>
  </tbody>
</table>
</div>



## First query : parallel edges

In this first query, we want to check if there are parallel edges in the graph. This should not happen with the US networks, because we removed the parallel edges when we created the *parquet* files in a [previous post](https://aetperf.github.io/2022/09/22/Download-some-benchmark-road-networks-for-Shortest-Paths-algorithms.html). When we imported the Europe networks, we created the *parquet* files by chunks, so we can guarantee that there is no parallel edge within each chunk, but nor overall. Here is the query:


```python
query_1 = """
SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END 
FROM (
    SELECT source, target, COUNT(*)
    FROM graph_edges
    GROUP BY source, target
    HAVING COUNT(*) > 1
)"""
```

We expect this query to return 0 for each graph.

### DuckDB

```python
res_duckdb = {}

for name in names:
    parquet_graph_file_path = parquet_graph_file_paths[name]

    connection = duckdb.connect()

    # query
    start = perf_counter()
    query = query_1.replace("graph_edges", f"read_parquet('{parquet_graph_file_path}')")
    duplicates = connection.query(query).fetchone()[0]
    elapsed_time_s = perf_counter() - start

    connection.close()

    res_duckdb[name] = {}
    res_duckdb[name]["duplicates"] = duplicates

    assert duplicates == 0

    stats[name]["query_1_DuckDB"] = elapsed_time_s
connection.close()
res_duckdb_df = pd.DataFrame.from_dict(res_duckdb, orient="index")
```


### Tableau Hyper


```python
res_hyper = {}
with HyperProcess(telemetry=TELEMETRY) as hyper:
    for name in names:

        parquet_graph_file_path = parquet_graph_file_paths[name]

        with Connection(endpoint=hyper.endpoint) as connection:

            # query
            start = perf_counter()
            query = query_1.replace(
                "graph_edges",
                f"external('{parquet_graph_file_path}', FORMAT => 'parquet')",
            )
            duplicates = connection.execute_scalar_query(query)
            elapsed_time_s = perf_counter() - start

        res_hyper[name] = {}
        res_hyper[name]["duplicates"] = duplicates

        assert duplicates == 0

        stats[name]["query_1_Hyper"] = elapsed_time_s
res_hyper_df = pd.DataFrame.from_dict(res_hyper, orient="index")
```

### Validation

```python
assert_frame_equal(res_duckdb_df, res_hyper_df)
```

### Elapsed time


```python
stats_df = pd.DataFrame.from_dict(stats, orient="index")
stats_df = stats_df.loc[[c for c in ordered_names if c in stats_df.index.values]]
cols = [c for c in stats_df.columns if c.startswith("query_1")]
query_1_df = stats_df[cols]
ax = query_1_df.plot.bar(figsize=FS, grid=True, logy=True, rot=60, alpha=ALPHA)
_ = ax.legend(["DuckDB", "Hyper"])
_ = ax.set(title="Query_1", xlabel="Network", ylabel="Elapsed time (s) - Log scale")
```

<p align="center">
  <img width="800" src="/img/2022-10-04_01/output_19_0.png" alt="query_1">
</p>


### Results

There is no parallel edge in any of the networks.

## Second query : vertex and edge counts


```python
query_2 = "SELECT COUNT(*), MAX(source), MAX(target) FROM graph_edges"
```

### DuckDB


```python
res_duckdb = {}
for name in names:
    parquet_graph_file_path = parquet_graph_file_paths[name]

    connection = duckdb.connect()

    # query
    start = perf_counter()
    query = query_2.replace("graph_edges", f"read_parquet('{parquet_graph_file_path}')")
    res = connection.query(query).fetchall()[0]
    elapsed_time_s = perf_counter() - start

    connection.close()

    edge_count = res[0]
    vertex_count = max(res[1:3]) + 1

    stats[name]["vertex_count"] = vertex_count
    stats[name]["edge_count"] = edge_count
    stats[name]["query_2_DuckDB"] = elapsed_time_s

    res_duckdb[name] = {}
    res_duckdb[name]["vertex_count"] = vertex_count
    res_duckdb[name]["edge_count"] = edge_count
res_duckdb_df = pd.DataFrame.from_dict(res_duckdb, orient="index")
```

### Tableau Hyper


```python
res_hyper = {}
with HyperProcess(telemetry=TELEMETRY) as hyper:
    for name in names:
        parquet_graph_file_path = parquet_graph_file_paths[name]

        with Connection(endpoint=hyper.endpoint) as connection:

            # query
            start = perf_counter()
            query = query_2.replace(
                "graph_edges",
                f"external('{parquet_graph_file_path}', FORMAT => 'parquet')",
            )
            res = connection.execute_list_query(query)[0]
            elapsed_time_s = perf_counter() - start

        edge_count = res[0]
        vertex_count = max(res[1:3]) + 1

        stats[name]["query_2_Hyper"] = elapsed_time_s

        res_hyper[name] = {}
        res_hyper[name]["vertex_count"] = vertex_count
        res_hyper[name]["edge_count"] = edge_count
res_hyper_df = pd.DataFrame.from_dict(res_hyper, orient="index")
```


### Validation


```python
assert_frame_equal(res_duckdb_df, res_hyper_df)
```

### Elapsed time


```python
stats_df = pd.DataFrame.from_dict(stats, orient="index")
stats_df = stats_df.loc[[c for c in ordered_names if c in stats_df.index.values]]
cols = [c for c in stats_df.columns if c.startswith("query_2")]
query_2_df = stats_df[cols]
ax = query_2_df.plot.bar(figsize=FS, grid=True, logy=True, rot=60, alpha=ALPHA)
_ = ax.legend(["DuckDB", "Hyper"])
_ = ax.set(title="Query_2", xlabel="Network", ylabel="Elapsed time (s) - Log scale")
```

    
<p align="center">
  <img width="800" src="/img/2022-10-04_01/output_31_0.png" alt="query_2">
</p>

    

### Results


```python
stats_df[["vertex_count", "edge_count"]]
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
      <th>vertex_count</th>
      <th>edge_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NY</th>
      <td>264346</td>
      <td>730100</td>
    </tr>
    <tr>
      <th>BAY</th>
      <td>321270</td>
      <td>794830</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>435666</td>
      <td>1042400</td>
    </tr>
    <tr>
      <th>FLA</th>
      <td>1070376</td>
      <td>2687902</td>
    </tr>
    <tr>
      <th>NW</th>
      <td>1207945</td>
      <td>2820774</td>
    </tr>
    <tr>
      <th>NE</th>
      <td>1524453</td>
      <td>3868020</td>
    </tr>
    <tr>
      <th>CAL</th>
      <td>1890815</td>
      <td>4630444</td>
    </tr>
    <tr>
      <th>osm-bawu</th>
      <td>3064263</td>
      <td>6183798</td>
    </tr>
    <tr>
      <th>LKS</th>
      <td>2758119</td>
      <td>6794808</td>
    </tr>
    <tr>
      <th>E</th>
      <td>3598623</td>
      <td>8708058</td>
    </tr>
    <tr>
      <th>W</th>
      <td>6262104</td>
      <td>15119284</td>
    </tr>
    <tr>
      <th>CTR</th>
      <td>14081816</td>
      <td>33866826</td>
    </tr>
    <tr>
      <th>osm-ger</th>
      <td>20690320</td>
      <td>41791542</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>23947347</td>
      <td>57708624</td>
    </tr>
    <tr>
      <th>osm-eur</th>
      <td>173789185</td>
      <td>347997111</td>
    </tr>
  </tbody>
</table>
</div>


## Third query : count of connected vertices

Some vertices are isolated in the graph : this means that their degree is 0. We want to count the number of connected vertices in the graph (not isolated).


```python
query_3 = f"""
WITH edges AS (
    SELECT source, target 
    FROM graph_edges)
SELECT COUNT(*) 
FROM (
    SELECT source AS node 
    FROM edges     
        UNION     
    SELECT target AS node 
    FROM edges)"""
```

### DuckDB


```python
res_duckdb = {}
for name in names:
    parquet_graph_file_path = parquet_graph_file_paths[name]

    connection = duckdb.connect()

    # query
    start = perf_counter()
    query = query_3.replace("graph_edges", f"read_parquet('{parquet_graph_file_path}')")
    connected_vertices = connection.query(query).fetchone()[0]
    elapsed_time_s = perf_counter() - start

    connection.close()

    stats[name]["connected_vertices"] = connected_vertices
    stats[name]["mean_degree"] = stats[name]["edge_count"] / connected_vertices
    stats[name]["query_3_DuckDB"] = elapsed_time_s

    res_duckdb[name] = {}
    res_duckdb[name]["connected_vertices"] = connected_vertices
res_duckdb_df = pd.DataFrame.from_dict(res_duckdb, orient="index")
```


### Tableau Hyper


```python
res_hyper = {}
with HyperProcess(telemetry=TELEMETRY) as hyper:
    for name in names:
        parquet_graph_file_path = parquet_graph_file_paths[name]

        with Connection(endpoint=hyper.endpoint) as connection:

            # query
            start = perf_counter()
            query = query_3.replace(
                "graph_edges",
                f"external('{parquet_graph_file_path}', FORMAT => 'parquet')",
            )
            connected_vertices = connection.execute_scalar_query(query)
            elapsed_time_s = perf_counter() - start

        stats[name]["query_3_Hyper"] = elapsed_time_s

        res_hyper[name] = {}
        res_hyper[name]["connected_vertices"] = connected_vertices
res_hyper_df = pd.DataFrame.from_dict(res_hyper, orient="index")
```


### Validation


```python
assert_frame_equal(res_duckdb_df, res_hyper_df)
```

### Elapsed time


```python
stats_df = pd.DataFrame.from_dict(stats, orient="index")
stats_df = stats_df.loc[[c for c in ordered_names if c in stats_df.index.values]]
cols = [c for c in stats_df.columns if c.startswith("query_3")]
query_3_df = stats_df[cols]
ax = query_3_df.plot.bar(figsize=FS, grid=True, logy=True, rot=60, alpha=ALPHA)
_ = ax.legend(["DuckDB", "Hyper"])
_ = ax.set(title="Query_3", xlabel="Network", ylabel="Elapsed time (s) - Log scale")
```


<p align="center">
  <img width="800" src="/img/2022-10-04_01/output_44_0.png" alt="query_3">
</p>

We observe that the elapsed time measures for the largest network differ a lot between DuckDB and Tableau Hyper:


| Engine  | Elapsed time (s)  |
|---|---:|
| DuckDB | 38.71  |
| Tableau Hyper | 357.47 |


### Results

There is no isolated vertex in any of the network.


```python
stats_df[["vertex_count", "connected_vertices", "mean_degree"]]
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
      <th>vertex_count</th>
      <th>connected_vertices</th>
      <th>mean_degree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NY</th>
      <td>264346</td>
      <td>264346</td>
      <td>2.76</td>
    </tr>
    <tr>
      <th>BAY</th>
      <td>321270</td>
      <td>321270</td>
      <td>2.47</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>435666</td>
      <td>435666</td>
      <td>2.39</td>
    </tr>
    <tr>
      <th>FLA</th>
      <td>1070376</td>
      <td>1070376</td>
      <td>2.51</td>
    </tr>
    <tr>
      <th>NW</th>
      <td>1207945</td>
      <td>1207945</td>
      <td>2.34</td>
    </tr>
    <tr>
      <th>NE</th>
      <td>1524453</td>
      <td>1524453</td>
      <td>2.54</td>
    </tr>
    <tr>
      <th>CAL</th>
      <td>1890815</td>
      <td>1890815</td>
      <td>2.45</td>
    </tr>
    <tr>
      <th>osm-bawu</th>
      <td>3064263</td>
      <td>3064263</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>LKS</th>
      <td>2758119</td>
      <td>2758119</td>
      <td>2.46</td>
    </tr>
    <tr>
      <th>E</th>
      <td>3598623</td>
      <td>3598623</td>
      <td>2.42</td>
    </tr>
    <tr>
      <th>W</th>
      <td>6262104</td>
      <td>6262104</td>
      <td>2.41</td>
    </tr>
    <tr>
      <th>CTR</th>
      <td>14081816</td>
      <td>14081816</td>
      <td>2.41</td>
    </tr>
    <tr>
      <th>osm-ger</th>
      <td>20690320</td>
      <td>20690320</td>
      <td>2.02</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>23947347</td>
      <td>23947347</td>
      <td>2.41</td>
    </tr>
    <tr>
      <th>osm-eur</th>
      <td>173789185</td>
      <td>173789185</td>
      <td>2.00</td>
    </tr>
  </tbody>
</table>
</div>



## Forth query : count of vertices with one incoming and one outgoing egde

Count of degree 2 nodes with *in-degree=out-degree=1*. In the following, we refer to these vertices as *in-out vertices*.


```python
query_4 = """
    WITH TPARQUET AS (
        SELECT source, target 
        FROM graph_edges),
    TSOURCE AS (
        SELECT source AS node, COUNT(*) 
        FROM TPARQUET 
        GROUP BY source HAVING COUNT(*)=1),
    TTARGET AS (
        SELECT target AS node, COUNT(*) 
        FROM TPARQUET GROUP BY target HAVING COUNT(*)=1),
    TJOIN AS (
        SELECT s.node 
        FROM TSOURCE s 
        INNER JOIN TTARGET t ON s.node = t.node)
    SELECT COUNT(*) from TJOIN"""
```

### DuckDB


```python
res_duckdb = {}
for name in names:
    parquet_graph_file_path = parquet_graph_file_paths[name]

    connection = duckdb.connect()

    # query
    start = perf_counter()
    query = query_4.replace("graph_edges", f"read_parquet('{parquet_graph_file_path}')")
    inout_vertices = connection.query(query).fetchone()[0]
    elapsed_time_s = perf_counter() - start

    connection.close()

    stats[name]["inout_vertices"] = inout_vertices
    stats[name]["query_4_DuckDB"] = elapsed_time_s

    res_duckdb[name] = {}
    res_duckdb[name]["inout_vertices"] = inout_vertices
res_duckdb_df = pd.DataFrame.from_dict(res_duckdb, orient="index")
```

### Tableau Hyper


```python
res_hyper = {}
with HyperProcess(telemetry=TELEMETRY) as hyper:
    for name in names:
        parquet_graph_file_path = parquet_graph_file_paths[name]

        with Connection(endpoint=hyper.endpoint) as connection:

            # query
            start = perf_counter()
            query = query_4.replace(
                "graph_edges",
                f"external('{parquet_graph_file_path}', FORMAT => 'parquet')",
            )
            inout_vertices = connection.execute_scalar_query(query)
            elapsed_time_s = perf_counter() - start

        stats[name]["query_4_Hyper"] = elapsed_time_s

        res_hyper[name] = {}
        res_hyper[name]["inout_vertices"] = inout_vertices
res_hyper_df = pd.DataFrame.from_dict(res_hyper, orient="index")
```


### Validation


```python
assert_frame_equal(res_duckdb_df, res_hyper_df)
```

### Elapsed time


```python
stats_df = pd.DataFrame.from_dict(stats, orient="index")
stats_df = stats_df.loc[[c for c in ordered_names if c in stats_df.index.values]]
cols = [c for c in stats_df.columns if c.startswith("query_4")]
query_4_df = stats_df[cols]
ax = query_4_df.plot.bar(figsize=FS, grid=True, logy=True, rot=60, alpha=ALPHA)
_ = ax.legend(["DuckDB", "Hyper"])
_ = ax.set(title="Query_4", xlabel="Network", ylabel="Elapsed time (s) - Log scale")
```


<p align="center">
  <img width="800" src="/img/2022-10-04_01/output_57_0.png" alt="query_4">
</p>

### Results


```python
stats_df["ratio"] = stats_df.inout_vertices / stats_df.vertex_count
stats_df[["inout_vertices", "vertex_count", "ratio"]]
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
      <th>inout_vertices</th>
      <th>vertex_count</th>
      <th>ratio</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>NY</th>
      <td>41169</td>
      <td>264346</td>
      <td>0.16</td>
    </tr>
    <tr>
      <th>BAY</th>
      <td>73109</td>
      <td>321270</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>COL</th>
      <td>82537</td>
      <td>435666</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>FLA</th>
      <td>210849</td>
      <td>1070376</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>NW</th>
      <td>282638</td>
      <td>1207945</td>
      <td>0.23</td>
    </tr>
    <tr>
      <th>NE</th>
      <td>288695</td>
      <td>1524453</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>CAL</th>
      <td>373947</td>
      <td>1890815</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>osm-bawu</th>
      <td>398242</td>
      <td>3064263</td>
      <td>0.13</td>
    </tr>
    <tr>
      <th>LKS</th>
      <td>478263</td>
      <td>2758119</td>
      <td>0.17</td>
    </tr>
    <tr>
      <th>E</th>
      <td>764838</td>
      <td>3598623</td>
      <td>0.21</td>
    </tr>
    <tr>
      <th>W</th>
      <td>1209550</td>
      <td>6262104</td>
      <td>0.19</td>
    </tr>
    <tr>
      <th>CTR</th>
      <td>2787565</td>
      <td>14081816</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>osm-ger</th>
      <td>2874055</td>
      <td>20690320</td>
      <td>0.14</td>
    </tr>
    <tr>
      <th>USA</th>
      <td>4762005</td>
      <td>23947347</td>
      <td>0.20</td>
    </tr>
    <tr>
      <th>osm-eur</th>
      <td>20309942</td>
      <td>173789185</td>
      <td>0.12</td>
    </tr>
  </tbody>
</table>
</div>


## Fifth query : degree distribution


```python
query_5 = """
    CREATE TEMP TABLE t_edges AS 
        SELECT source, target 
        FROM graph_edges;
    CREATE TEMP TABLE t_nodes AS
        SELECT source AS node 
        FROM t_edges     
            UNION ALL
        SELECT target AS node 
        FROM t_edges;
    CREATE TEMP TABLE t_deg AS
        SELECT COUNT(*) AS deg
        FROM t_nodes
        GROUP BY node;
    SELECT deg, COUNT(*) AS n_occ 
    FROM t_deg
    GROUP BY deg
    ORDER BY deg ASC;"""
```

### DuckDB

```python
res_duckdb = {}
for name in names:
    parquet_graph_file_path = parquet_graph_file_paths[name]

    connection = duckdb.connect()
    
    # query
    start = perf_counter()
    query = query_5.replace("graph_edges", f"read_parquet('{parquet_graph_file_path}')")

    queries = query.removesuffix(";").split(";")
    for sq in queries[:-1]:
        connection.execute(sq)
    sq = queries[-1]
    res = connection.query(sq).fetchall()

    elapsed_time_s = perf_counter() - start

    stats[name]["query_5_DuckDB"] = elapsed_time_s

    connection.close()

    res_duckdb[name] = {}
    for item in res:
        degree = item[0]
        vertex_count = item[1]
        res_duckdb[name]["degree_" + str(degree).zfill(3)] = vertex_count

res_duckdb_df = pd.DataFrame.from_dict(res_duckdb, orient="index")
res_duckdb_df = res_duckdb_df.sort_index(axis=1)
res_duckdb_df = res_duckdb_df.fillna(0).astype(int)
```


### Tableau Hyper


```python
res_hyper = {}
with HyperProcess(telemetry=TELEMETRY) as hyper:
    for name in names:
        with Connection(endpoint=hyper.endpoint) as connection:

            parquet_graph_file_path = parquet_graph_file_paths[name]

            # query
            start = perf_counter()
            query = query_5.replace(
                "graph_edges",
                f"external('{parquet_graph_file_path}', FORMAT => 'parquet')",
            )

            queries = query.removesuffix(";").split(";")
            for sq in queries[:-1]:
                connection.execute_command(sq)
            sq = queries[-1]
            res = connection.execute_list_query(sq)

            elapsed_time_s = perf_counter() - start

        for item in res:
            degree = item[0]
            vertex_count = item[1]
            stats[name]["degree_" + str(degree).zfill(3)] = vertex_count
        stats[name]["query_5_Hyper"] = elapsed_time_s

        res_hyper[name] = {}
        for item in res:
            degree = item[0]
            vertex_count = item[1]
            res_hyper[name]["degree_" + str(degree).zfill(3)] = vertex_count
res_hyper_df = pd.DataFrame.from_dict(res_hyper, orient="index")
res_hyper_df = res_hyper_df.sort_index(axis=1)
res_hyper_df = res_hyper_df.fillna(0).astype(int)
```

### Validation

```python
assert_frame_equal(res_duckdb_df, res_hyper_df)
```

### Elapsed time


```python
stats_df = pd.DataFrame.from_dict(stats, orient="index")
stats_df = stats_df.loc[[c for c in ordered_names if c in stats_df.index.values]]
cols = [c for c in stats_df.columns if c.startswith("query_5")]
query_5_df = stats_df[cols]
ax = query_5_df.plot.bar(figsize=FS, grid=True, logy=True, rot=60, alpha=ALPHA)
_ = ax.legend(["DuckDB", "Hyper"])
_ = ax.set(title="Query_5", xlabel="Network", ylabel="Elapsed time (s) - Log scale")
```

    
<p align="center">
  <img width="800" src="/img/2022-10-04_01/output_71_0.png" alt="query_5">
</p>

### Results


```python
degree_cols = sorted([c for c in stats_df.columns if c.startswith("degree_")])
stats_df[degree_cols] = stats_df[degree_cols].fillna(0).astype(int)
degrees = 100 *  stats_df[degree_cols].div(stats_df["vertex_count"], axis=0)
cols = degrees.columns
degrees.columns = [int(c.split('_')[-1]) for c in cols]
tot = degrees.sum(axis=0)
degrees = degrees[list(tot[tot > 0.01].index.values)]
degrees['total'] = degrees.sum(axis=1)
styler = degrees.style.background_gradient(axis=1, cmap='YlOrRd')
styler = styler.format(precision=2)
styler 
```



<style type="text/css">
#T_9f598_row0_col0, #T_9f598_row0_col2 {
  background-color: #ffe997;
  color: #000000;
}
#T_9f598_row0_col1, #T_9f598_row0_col3, #T_9f598_row0_col5, #T_9f598_row0_col8, #T_9f598_row1_col1, #T_9f598_row1_col3, #T_9f598_row1_col5, #T_9f598_row1_col7, #T_9f598_row1_col8, #T_9f598_row2_col1, #T_9f598_row2_col3, #T_9f598_row2_col5, #T_9f598_row2_col7, #T_9f598_row2_col8, #T_9f598_row3_col1, #T_9f598_row3_col3, #T_9f598_row3_col5, #T_9f598_row3_col7, #T_9f598_row3_col8, #T_9f598_row4_col1, #T_9f598_row4_col3, #T_9f598_row4_col5, #T_9f598_row4_col7, #T_9f598_row4_col8, #T_9f598_row5_col1, #T_9f598_row5_col3, #T_9f598_row5_col5, #T_9f598_row5_col7, #T_9f598_row5_col8, #T_9f598_row6_col1, #T_9f598_row6_col3, #T_9f598_row6_col5, #T_9f598_row6_col7, #T_9f598_row6_col8, #T_9f598_row7_col5, #T_9f598_row7_col7, #T_9f598_row7_col8, #T_9f598_row8_col1, #T_9f598_row8_col3, #T_9f598_row8_col5, #T_9f598_row8_col7, #T_9f598_row8_col8, #T_9f598_row9_col1, #T_9f598_row9_col3, #T_9f598_row9_col5, #T_9f598_row9_col7, #T_9f598_row9_col8, #T_9f598_row10_col1, #T_9f598_row10_col3, #T_9f598_row10_col5, #T_9f598_row10_col7, #T_9f598_row10_col8, #T_9f598_row11_col1, #T_9f598_row11_col3, #T_9f598_row11_col5, #T_9f598_row11_col7, #T_9f598_row11_col8, #T_9f598_row12_col5, #T_9f598_row12_col7, #T_9f598_row12_col8, #T_9f598_row13_col1, #T_9f598_row13_col3, #T_9f598_row13_col5, #T_9f598_row13_col7, #T_9f598_row13_col8, #T_9f598_row14_col5, #T_9f598_row14_col7, #T_9f598_row14_col8 {
  background-color: #ffffcc;
  color: #000000;
}
#T_9f598_row0_col4 {
  background-color: #fd9640;
  color: #000000;
}
#T_9f598_row0_col6 {
  background-color: #fede82;
  color: #000000;
}
#T_9f598_row0_col7, #T_9f598_row7_col1, #T_9f598_row7_col3, #T_9f598_row12_col1, #T_9f598_row12_col3, #T_9f598_row14_col3 {
  background-color: #fffecb;
  color: #000000;
}
#T_9f598_row0_col9, #T_9f598_row1_col9, #T_9f598_row2_col9, #T_9f598_row3_col9, #T_9f598_row4_col9, #T_9f598_row5_col9, #T_9f598_row6_col9, #T_9f598_row7_col9, #T_9f598_row8_col9, #T_9f598_row9_col9, #T_9f598_row10_col9, #T_9f598_row11_col9, #T_9f598_row12_col9, #T_9f598_row13_col9, #T_9f598_row14_col9 {
  background-color: #800026;
  color: #f1f1f1;
}
#T_9f598_row1_col0, #T_9f598_row3_col2 {
  background-color: #fedd7e;
  color: #000000;
}
#T_9f598_row1_col2 {
  background-color: #fee187;
  color: #000000;
}
#T_9f598_row1_col4 {
  background-color: #fd9f44;
  color: #000000;
}
#T_9f598_row1_col6, #T_9f598_row8_col6 {
  background-color: #ffeda0;
  color: #000000;
}
#T_9f598_row2_col0, #T_9f598_row5_col0 {
  background-color: #fee38b;
  color: #000000;
}
#T_9f598_row2_col2 {
  background-color: #febe59;
  color: #000000;
}
#T_9f598_row2_col4 {
  background-color: #feb54f;
  color: #000000;
}
#T_9f598_row2_col6, #T_9f598_row13_col6 {
  background-color: #fff0a7;
  color: #000000;
}
#T_9f598_row3_col0, #T_9f598_row6_col0, #T_9f598_row11_col0, #T_9f598_row13_col0 {
  background-color: #fee288;
  color: #000000;
}
#T_9f598_row3_col4 {
  background-color: #fd9e43;
  color: #000000;
}
#T_9f598_row3_col6, #T_9f598_row5_col6, #T_9f598_row7_col0, #T_9f598_row12_col4 {
  background-color: #ffec9f;
  color: #000000;
}
#T_9f598_row4_col0 {
  background-color: #fedc7c;
  color: #000000;
}
#T_9f598_row4_col2 {
  background-color: #fecc68;
  color: #000000;
}
#T_9f598_row4_col4 {
  background-color: #feaf4b;
  color: #000000;
}
#T_9f598_row4_col6 {
  background-color: #fff2ac;
  color: #000000;
}
#T_9f598_row5_col2 {
  background-color: #fede80;
  color: #000000;
}
#T_9f598_row5_col4 {
  background-color: #fd9941;
  color: #000000;
}
#T_9f598_row6_col2 {
  background-color: #fed16e;
  color: #000000;
}
#T_9f598_row6_col4 {
  background-color: #fea848;
  color: #000000;
}
#T_9f598_row6_col6, #T_9f598_row14_col0 {
  background-color: #ffefa4;
  color: #000000;
}
#T_9f598_row7_col2 {
  background-color: #e8241f;
  color: #f1f1f1;
}
#T_9f598_row7_col4 {
  background-color: #ffeda1;
  color: #000000;
}
#T_9f598_row7_col6, #T_9f598_row12_col6 {
  background-color: #fffdc8;
  color: #000000;
}
#T_9f598_row8_col0 {
  background-color: #ffe590;
  color: #000000;
}
#T_9f598_row8_col2 {
  background-color: #fec45f;
  color: #000000;
}
#T_9f598_row8_col4 {
  background-color: #feb24c;
  color: #000000;
}
#T_9f598_row9_col0 {
  background-color: #fedf83;
  color: #000000;
}
#T_9f598_row9_col2 {
  background-color: #fed572;
  color: #000000;
}
#T_9f598_row9_col4 {
  background-color: #fea546;
  color: #000000;
}
#T_9f598_row9_col6, #T_9f598_row10_col6 {
  background-color: #fff0a8;
  color: #000000;
}
#T_9f598_row10_col0 {
  background-color: #fee289;
  color: #000000;
}
#T_9f598_row10_col2 {
  background-color: #fec863;
  color: #000000;
}
#T_9f598_row10_col4 {
  background-color: #fead4a;
  color: #000000;
}
#T_9f598_row11_col2 {
  background-color: #fec662;
  color: #000000;
}
#T_9f598_row11_col4 {
  background-color: #feb04b;
  color: #000000;
}
#T_9f598_row11_col6 {
  background-color: #ffefa5;
  color: #000000;
}
#T_9f598_row12_col0 {
  background-color: #ffeb9c;
  color: #000000;
}
#T_9f598_row12_col2 {
  background-color: #ec2c21;
  color: #f1f1f1;
}
#T_9f598_row13_col2 {
  background-color: #fec965;
  color: #000000;
}
#T_9f598_row13_col4 {
  background-color: #feae4a;
  color: #000000;
}
#T_9f598_row14_col1, #T_9f598_row14_col6 {
  background-color: #fffec9;
  color: #000000;
}
#T_9f598_row14_col2 {
  background-color: #e0181d;
  color: #f1f1f1;
}
#T_9f598_row14_col4 {
  background-color: #fff1a9;
  color: #000000;
}
</style>
<table id="T_9f598">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_9f598_level0_col0" class="col_heading level0 col0" >2</th>
      <th id="T_9f598_level0_col1" class="col_heading level0 col1" >3</th>
      <th id="T_9f598_level0_col2" class="col_heading level0 col2" >4</th>
      <th id="T_9f598_level0_col3" class="col_heading level0 col3" >5</th>
      <th id="T_9f598_level0_col4" class="col_heading level0 col4" >6</th>
      <th id="T_9f598_level0_col5" class="col_heading level0 col5" >7</th>
      <th id="T_9f598_level0_col6" class="col_heading level0 col6" >8</th>
      <th id="T_9f598_level0_col7" class="col_heading level0 col7" >10</th>
      <th id="T_9f598_level0_col8" class="col_heading level0 col8" >12</th>
      <th id="T_9f598_level0_col9" class="col_heading level0 col9" >total</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_9f598_level0_row0" class="row_heading level0 row0" >NY</th>
      <td id="T_9f598_row0_col0" class="data row0 col0" >15.57</td>
      <td id="T_9f598_row0_col1" class="data row0 col1" >0.00</td>
      <td id="T_9f598_row0_col2" class="data row0 col2" >15.27</td>
      <td id="T_9f598_row0_col3" class="data row0 col3" >0.00</td>
      <td id="T_9f598_row0_col4" class="data row0 col4" >47.11</td>
      <td id="T_9f598_row0_col5" class="data row0 col5" >0.00</td>
      <td id="T_9f598_row0_col6" class="data row0 col6" >21.56</td>
      <td id="T_9f598_row0_col7" class="data row0 col7" >0.43</td>
      <td id="T_9f598_row0_col8" class="data row0 col8" >0.06</td>
      <td id="T_9f598_row0_col9" class="data row0 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row1" class="row_heading level0 row1" >BAY</th>
      <td id="T_9f598_row1_col0" class="data row1 col0" >22.76</td>
      <td id="T_9f598_row1_col1" class="data row1 col1" >0.00</td>
      <td id="T_9f598_row1_col2" class="data row1 col2" >20.30</td>
      <td id="T_9f598_row1_col3" class="data row1 col3" >0.00</td>
      <td id="T_9f598_row1_col4" class="data row1 col4" >43.94</td>
      <td id="T_9f598_row1_col5" class="data row1 col5" >0.00</td>
      <td id="T_9f598_row1_col6" class="data row1 col6" >12.81</td>
      <td id="T_9f598_row1_col7" class="data row1 col7" >0.17</td>
      <td id="T_9f598_row1_col8" class="data row1 col8" >0.02</td>
      <td id="T_9f598_row1_col9" class="data row1 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row2" class="row_heading level0 row2" >COL</th>
      <td id="T_9f598_row2_col0" class="data row2 col0" >18.95</td>
      <td id="T_9f598_row2_col1" class="data row2 col1" >0.00</td>
      <td id="T_9f598_row2_col2" class="data row2 col2" >33.71</td>
      <td id="T_9f598_row2_col3" class="data row2 col3" >0.00</td>
      <td id="T_9f598_row2_col4" class="data row2 col4" >36.59</td>
      <td id="T_9f598_row2_col5" class="data row2 col5" >0.00</td>
      <td id="T_9f598_row2_col6" class="data row2 col6" >10.66</td>
      <td id="T_9f598_row2_col7" class="data row2 col7" >0.09</td>
      <td id="T_9f598_row2_col8" class="data row2 col8" >0.01</td>
      <td id="T_9f598_row2_col9" class="data row2 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row3" class="row_heading level0 row3" >FLA</th>
      <td id="T_9f598_row3_col0" class="data row3 col0" >19.70</td>
      <td id="T_9f598_row3_col1" class="data row3 col1" >0.00</td>
      <td id="T_9f598_row3_col2" class="data row3 col2" >22.73</td>
      <td id="T_9f598_row3_col3" class="data row3 col3" >0.00</td>
      <td id="T_9f598_row3_col4" class="data row3 col4" >44.49</td>
      <td id="T_9f598_row3_col5" class="data row3 col5" >0.00</td>
      <td id="T_9f598_row3_col6" class="data row3 col6" >12.94</td>
      <td id="T_9f598_row3_col7" class="data row3 col7" >0.13</td>
      <td id="T_9f598_row3_col8" class="data row3 col8" >0.01</td>
      <td id="T_9f598_row3_col9" class="data row3 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row4" class="row_heading level0 row4" >NW</th>
      <td id="T_9f598_row4_col0" class="data row4 col0" >23.40</td>
      <td id="T_9f598_row4_col1" class="data row4 col1" >0.00</td>
      <td id="T_9f598_row4_col2" class="data row4 col2" >28.99</td>
      <td id="T_9f598_row4_col3" class="data row4 col3" >0.00</td>
      <td id="T_9f598_row4_col4" class="data row4 col4" >38.47</td>
      <td id="T_9f598_row4_col5" class="data row4 col5" >0.00</td>
      <td id="T_9f598_row4_col6" class="data row4 col6" >9.01</td>
      <td id="T_9f598_row4_col7" class="data row4 col7" >0.13</td>
      <td id="T_9f598_row4_col8" class="data row4 col8" >0.01</td>
      <td id="T_9f598_row4_col9" class="data row4 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row5" class="row_heading level0 row5" >NE</th>
      <td id="T_9f598_row5_col0" class="data row5 col0" >18.94</td>
      <td id="T_9f598_row5_col1" class="data row5 col1" >0.00</td>
      <td id="T_9f598_row5_col2" class="data row5 col2" >21.90</td>
      <td id="T_9f598_row5_col3" class="data row5 col3" >0.00</td>
      <td id="T_9f598_row5_col4" class="data row5 col4" >45.95</td>
      <td id="T_9f598_row5_col5" class="data row5 col5" >0.00</td>
      <td id="T_9f598_row5_col6" class="data row5 col6" >12.96</td>
      <td id="T_9f598_row5_col7" class="data row5 col7" >0.23</td>
      <td id="T_9f598_row5_col8" class="data row5 col8" >0.03</td>
      <td id="T_9f598_row5_col9" class="data row5 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row6" class="row_heading level0 row6" >CAL</th>
      <td id="T_9f598_row6_col0" class="data row6 col0" >19.78</td>
      <td id="T_9f598_row6_col1" class="data row6 col1" >0.00</td>
      <td id="T_9f598_row6_col2" class="data row6 col2" >27.53</td>
      <td id="T_9f598_row6_col3" class="data row6 col3" >0.00</td>
      <td id="T_9f598_row6_col4" class="data row6 col4" >40.89</td>
      <td id="T_9f598_row6_col5" class="data row6 col5" >0.00</td>
      <td id="T_9f598_row6_col6" class="data row6 col6" >11.66</td>
      <td id="T_9f598_row6_col7" class="data row6 col7" >0.14</td>
      <td id="T_9f598_row6_col8" class="data row6 col8" >0.01</td>
      <td id="T_9f598_row6_col9" class="data row6 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row7" class="row_heading level0 row7" >osm-bawu</th>
      <td id="T_9f598_row7_col0" class="data row7 col0" >13.00</td>
      <td id="T_9f598_row7_col1" class="data row7 col1" >0.60</td>
      <td id="T_9f598_row7_col2" class="data row7 col2" >72.27</td>
      <td id="T_9f598_row7_col3" class="data row7 col3" >0.58</td>
      <td id="T_9f598_row7_col4" class="data row7 col4" >12.26</td>
      <td id="T_9f598_row7_col5" class="data row7 col5" >0.07</td>
      <td id="T_9f598_row7_col6" class="data row7 col6" >1.21</td>
      <td id="T_9f598_row7_col7" class="data row7 col7" >0.01</td>
      <td id="T_9f598_row7_col8" class="data row7 col8" >0.00</td>
      <td id="T_9f598_row7_col9" class="data row7 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row8" class="row_heading level0 row8" >LKS</th>
      <td id="T_9f598_row8_col0" class="data row8 col0" >17.34</td>
      <td id="T_9f598_row8_col1" class="data row8 col1" >0.00</td>
      <td id="T_9f598_row8_col2" class="data row8 col2" >32.01</td>
      <td id="T_9f598_row8_col3" class="data row8 col3" >0.00</td>
      <td id="T_9f598_row8_col4" class="data row8 col4" >37.75</td>
      <td id="T_9f598_row8_col5" class="data row8 col5" >0.00</td>
      <td id="T_9f598_row8_col6" class="data row8 col6" >12.77</td>
      <td id="T_9f598_row8_col7" class="data row8 col7" >0.12</td>
      <td id="T_9f598_row8_col8" class="data row8 col8" >0.01</td>
      <td id="T_9f598_row8_col9" class="data row8 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row9" class="row_heading level0 row9" >E</th>
      <td id="T_9f598_row9_col0" class="data row9 col0" >21.25</td>
      <td id="T_9f598_row9_col1" class="data row9 col1" >0.00</td>
      <td id="T_9f598_row9_col2" class="data row9 col2" >26.19</td>
      <td id="T_9f598_row9_col3" class="data row9 col3" >0.00</td>
      <td id="T_9f598_row9_col4" class="data row9 col4" >42.07</td>
      <td id="T_9f598_row9_col5" class="data row9 col5" >0.00</td>
      <td id="T_9f598_row9_col6" class="data row9 col6" >10.30</td>
      <td id="T_9f598_row9_col7" class="data row9 col7" >0.16</td>
      <td id="T_9f598_row9_col8" class="data row9 col8" >0.02</td>
      <td id="T_9f598_row9_col9" class="data row9 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row10" class="row_heading level0 row10" >W</th>
      <td id="T_9f598_row10_col0" class="data row10 col0" >19.32</td>
      <td id="T_9f598_row10_col1" class="data row10 col1" >0.00</td>
      <td id="T_9f598_row10_col2" class="data row10 col2" >30.72</td>
      <td id="T_9f598_row10_col3" class="data row10 col3" >0.00</td>
      <td id="T_9f598_row10_col4" class="data row10 col4" >39.33</td>
      <td id="T_9f598_row10_col5" class="data row10 col5" >0.00</td>
      <td id="T_9f598_row10_col6" class="data row10 col6" >10.49</td>
      <td id="T_9f598_row10_col7" class="data row10 col7" >0.13</td>
      <td id="T_9f598_row10_col8" class="data row10 col8" >0.01</td>
      <td id="T_9f598_row10_col9" class="data row10 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row11" class="row_heading level0 row11" >CTR</th>
      <td id="T_9f598_row11_col0" class="data row11 col0" >19.80</td>
      <td id="T_9f598_row11_col1" class="data row11 col1" >0.00</td>
      <td id="T_9f598_row11_col2" class="data row11 col2" >31.11</td>
      <td id="T_9f598_row11_col3" class="data row11 col3" >0.00</td>
      <td id="T_9f598_row11_col4" class="data row11 col4" >38.01</td>
      <td id="T_9f598_row11_col5" class="data row11 col5" >0.00</td>
      <td id="T_9f598_row11_col6" class="data row11 col6" >10.97</td>
      <td id="T_9f598_row11_col7" class="data row11 col7" >0.10</td>
      <td id="T_9f598_row11_col8" class="data row11 col8" >0.01</td>
      <td id="T_9f598_row11_col9" class="data row11 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row12" class="row_heading level0 row12" >osm-ger</th>
      <td id="T_9f598_row12_col0" class="data row12 col0" >13.89</td>
      <td id="T_9f598_row12_col1" class="data row12 col1" >0.60</td>
      <td id="T_9f598_row12_col2" class="data row12 col2" >70.33</td>
      <td id="T_9f598_row12_col3" class="data row12 col3" >0.66</td>
      <td id="T_9f598_row12_col4" class="data row12 col4" >13.17</td>
      <td id="T_9f598_row12_col5" class="data row12 col5" >0.08</td>
      <td id="T_9f598_row12_col6" class="data row12 col6" >1.26</td>
      <td id="T_9f598_row12_col7" class="data row12 col7" >0.01</td>
      <td id="T_9f598_row12_col8" class="data row12 col8" >0.00</td>
      <td id="T_9f598_row12_col9" class="data row12 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row13" class="row_heading level0 row13" >USA</th>
      <td id="T_9f598_row13_col0" class="data row13 col0" >19.89</td>
      <td id="T_9f598_row13_col1" class="data row13 col1" >0.00</td>
      <td id="T_9f598_row13_col2" class="data row13 col2" >30.27</td>
      <td id="T_9f598_row13_col3" class="data row13 col3" >0.00</td>
      <td id="T_9f598_row13_col4" class="data row13 col4" >38.97</td>
      <td id="T_9f598_row13_col5" class="data row13 col5" >0.00</td>
      <td id="T_9f598_row13_col6" class="data row13 col6" >10.75</td>
      <td id="T_9f598_row13_col7" class="data row13 col7" >0.12</td>
      <td id="T_9f598_row13_col8" class="data row13 col8" >0.01</td>
      <td id="T_9f598_row13_col9" class="data row13 col9" >100.00</td>
    </tr>
    <tr>
      <th id="T_9f598_level0_row14" class="row_heading level0 row14" >osm-eur</th>
      <td id="T_9f598_row14_col0" class="data row14 col0" >11.69</td>
      <td id="T_9f598_row14_col1" class="data row14 col1" >0.84</td>
      <td id="T_9f598_row14_col2" class="data row14 col2" >76.03</td>
      <td id="T_9f598_row14_col3" class="data row14 col3" >0.50</td>
      <td id="T_9f598_row14_col4" class="data row14 col4" >9.80</td>
      <td id="T_9f598_row14_col5" class="data row14 col5" >0.05</td>
      <td id="T_9f598_row14_col6" class="data row14 col6" >1.09</td>
      <td id="T_9f598_row14_col7" class="data row14 col7" >0.01</td>
      <td id="T_9f598_row14_col8" class="data row14 col8" >0.00</td>
      <td id="T_9f598_row14_col9" class="data row14 col9" >100.00</td>
    </tr>
  </tbody>
</table>


## Conclusion

*Apache Parquet* is a column-oriented data file format designed for efficient data storage and retrieval. It is widespread in the data analysis ecosystem. Querying them directly with an efficient SQL engine is really convenient. Both engines, DuckDB and Tableau Hyper are amazing tools, allowing to efficiently query *Parquet* files, among other capabilities. We only scratched the surface of this feature in this post, with a very specific use case. We observed similar timings for most of the queries. We did not measure memory usage. However, we observed that it is important to write SQL queries that are a little bit optimized regarding memory consumption, when dealing with large datasets and "large" queries. Also, it is advised to specify a temp directory to the engine, so that it can write some temporary data there (`temp_directory` setting with DuckDB).


{% if page.comments %}
<div id="disqus_thread"></div>
<script>

/**
*  RECOMMENDED CONFIGURATION VARIABLES: EDIT AND UNCOMMENT THE SECTION BELOW TO INSERT DYNAMIC VALUES FROM YOUR PLATFORM OR CMS.
*  LEARN WHY DEFINING THESE VARIABLES IS IMPORTANT: https://disqus.com/admin/universalcode/#configuration-variables*/
/*
var disqus_config = function () {
this.page.url = PAGE_URL;  // Replace PAGE_URL with your page's canonical URL variable
this.page.identifier = PAGE_IDENTIFIER; // Replace PAGE_IDENTIFIER with your page's unique identifier variable
};
*/
(function() { // DON'T EDIT BELOW THIS LINE
var d = document, s = d.createElement('script');
s.src = 'https://aetperf-github-io-1.disqus.com/embed.js';
s.setAttribute('data-timestamp', +new Date());
(d.head || d.body).appendChild(s);
})();
</script>
<noscript>Please enable JavaScript to view the <a href="https://disqus.com/?ref_noscript">comments powered by Disqus.</a></noscript>
{% endif %}