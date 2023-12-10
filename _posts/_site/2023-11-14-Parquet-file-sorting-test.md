
---
title: Parquet file sorting test WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
- TPC-H 
- benchmark
- SQL
- Parquet
- DuckDB
- Hyper
- ClickHouse
- Polars
---

A while ago, I stumbled upon a post by Mimoune Djouallah on Twitter [@mim_djo](https://twitter.com/mim_djo) about a Parquet sorting test.

<p align="center">
  <img width="1200" src="/img/2023-11-15_01/Mimoune.png" alt="Mimoune">
</p>

The goal is to read a Parquet file, sort the table and write the sorted table into a Parquet file on disk.

The table comes from the [TPC-H](https://www.tpc.org/tpch/) benchmark, with a scale factor of 10. The table that we want to sort is `lineitem`, which has 59986052 rows. It was generated with DuckDB's [TPC-H](https://duckdb.org/docs/extensions/tpch) extension, and saved into Parquet files with default compression "snappy" and row group size 122880.

## Language and package versions:

	Python 3.11.5 | packaged by conda-forge | (main, Aug 27 2023, 03:34:09) [GCC 12.3.0] on linux
    duckdb          : 0.9.2
    tableauhyperapi : 0.0.18161
    chdb            : 0.15.0
    polars          : 0.19.13
    pyarrow         : 14.0.1

These are recent versions.

## System information

The code is executed on a linux laptop with the following features:

    OS : Linux mint 21.1, based on Ubuntu 22.04  
    CPU : 12th Gen Intel© Core™ i9-12900H (10 cores)    
    RAM : 32 GB  
    Data disk : Samsung SSD 980 PRO 1TB  


## Code

Input and output parquet files:

```python
data_dir_path = "/home/francois/Data/dbbenchdata/tpch_10/"
input_file_path =  os.path.join(data_dir_path, "lineitem.parquet")
output_file_path = os.path.join(data_dir_path, "lineitem_sorted.parquet")
```

### DuckDB

```python
duckdb_file_path = os.path.join(data_dir_path, "data.duckdb")
with duckdb.connect(database=duckdb_file_path, read_only=False) as conn:
    conn.execute(
        f"""COPY (SELECT * FROM read_parquet('{input_file_path}')
        ORDER BY l_shipdate ASC)
        TO '{output_file_path}' (FORMAT PARQUET)"""
    )
```

### TableauHyperAPI

```python
hyper_file_path = os.path.join(data_dir_path, "data.hyper")
with HyperProcess(
    telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
) as hyper:
    with Connection(
        endpoint=hyper.endpoint,
        database=hyper_file_path,
        create_mode=CreateMode.CREATE_AND_REPLACE,
    ) as conn:
        sql = f"""COPY
        (SELECT * FROM external('{input_file_path}')
        ORDER BY l_shipdate ASC)
        TO '{output_file_path}' WITH ( FORMAT => 'parquet')
        """
        _ = conn.execute_command(sql)

```

Note that we also tried a version with a *temporary external table*, but without significative results:

```python
with HyperProcess(
    telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
) as hyper:
    with Connection(
        endpoint=hyper.endpoint,
        database=hyper_file_path,
        create_mode=CreateMode.CREATE_AND_REPLACE,
    ) as conn:
        sql = f"""CREATE TEMPORARY EXTERNAL TABLE lineitem
                FOR '{input_file_path}'
                WITH (FORMAT => 'parquet');"""
        _ = conn.execute_command(sql)
        sql = f"""COPY (SELECT * FROM lineitem ORDER BY l_shipdate ASC)
        TO '{output_file_path}' WITH ( FORMAT => 'parquet')"""
        _ = conn.execute_command(sql)
```

### CHDB (ClickHouse)

```python
sql = f"""SELECT * FROM file ('{input_file_path}', Parquet)
ORDER BY l_shipdate
INTO OUTFILE '{output_file_path}'  FORMAT Parquet  """
_ = chdb.query(sql, "JSON")
```

### Polars

```python
pl.read_parquet(input_file_path).sort("l_shipdate").write_parquet(output_file_path)
```

We also tried the streaming API that sounds promising:

```python
pl.scan_parquet(input_file_path).sort("l_shipdate").sink_parquet(
    path=output_file_path
)
```

but it crashed:

    pl.read_parquet(input_file_path).sort("l_shipdate").write_parquet(output_file_path)


### PyArrow

```python
table = pq.read_table(input_file_path)
sorted_table = table.sort_by([("l_shipdate", "ascending")])
pq.write_table(sorted_table, output_file_path)
```