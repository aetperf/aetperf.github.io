---
title: Parquet file sorting test
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
- GlareDB
- DataFusion
---

**Update** Nov 17, 2023 - Added results using the latest DataFusion version.


Some time ago, we came across an intriguing Parquet sorting test shared by Mimoune Djouallah on Twitter [@mim_djo](https://twitter.com/mim_djo). The test involves reading a Parquet file, sorting the table, and writing the sorted data back to a Parquet file on disk. You can find the original post [here](https://x.com/mim_djo/status/1637321688382853120?s=20).

<p align="center">
  <img width="400" src="/img/2023-11-15_01/Mimoune.png" alt="Mimoune">
</p>

The dataset under consideration is derived from the [TPC-H](https://www.tpc.org/tpch/) benchmark with a scale factor (SF) of 10, specifically the `lineitem` table, comprising 59,986,052 rows. The data was generated using DuckDB's [TPC-H](https://duckdb.org/docs/extensions/tpch) extension and saved as Parquet files with "snappy" compression and a row group size of 122,880.

Given the manageable size of the table, the sorting operation fits in memory, though this might not be the case for larger scale factors. To shed light on different approaches and packages, we compared implementations using various tools:
- [DuckDB](https://duckdb.org/)
- [Hyper](https://tableau.github.io/hyper-db/)
- [chDB](https://github.com/chdb-io/chdb)
- [Polars](https://www.pola.rs/)
- [PyArrow](https://arrow.apache.org/docs/python/index.html)
- [GlareDB](https://docs.glaredb.com/glaredb/python/)
- [DataFusion](https://arrow.apache.org/datafusion-python/index.html)

Here are the language and package versions used:

	Python 3.11.5 | packaged by conda-forge | (main, Aug 27 2023, 03:34:09) [GCC 12.3.0] on linux
    duckdb          : 0.9.2
    tableauhyperapi : 0.0.18161
    chdb            : 0.15.0
    polars          : 0.19.13
    pyarrow         : 14.0.1
    glaredb         : 0.6.1
    datafusion      : 33.0.0

The code was executed on a Linux laptop with the following specifications:

    OS : Linux mint 21.1, based on Ubuntu 22.04  
    CPU : 12th Gen Intel© Core™ i9-12900H (10 cores)    
    RAM : 32 GB  
    Data disk : Samsung SSD 980 PRO 1TB  

Below are snippets of the code using different libraries for the sorting operation.

## Code snippets

Imports:

```python
import os 

import chdb
from datafusion import SessionContext
import duckdb
import glaredb
import polars as pl
import pyarrow as pa
from tableauhyperapi import Connection, CreateMode, HyperProcess, Telemetry
```
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

**Remark:** here we gave a database file name to the `connect()` method, to read or write persistent data, however this is not used in the present case. Similarly, in the next sub-section, we also gave a file path to the Hyper engine.

### Hyper

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

Note that we also tried a version with a *temporary external table*, but without significatively different results:

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

we experimented with the streaming API, which seemed promising:

```python
pl.scan_parquet(input_file_path).sort("l_shipdate").sink_parquet(
    path=output_file_path
)
```

However, it encountered a crash:

    thread '<unnamed>' panicked at crates/polars-pipe/src/executors/sinks/io.rs:149:49:
    called `Result::unwrap()` on an `Err` value: Io(Os { code: 28, kind: StorageFull, message: "No space left on device" })
    note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

Despite ample available space on the device, we think that the issue stemmed from the API attempting to write numerous small files to disk.

### PyArrow

```python
table = pq.read_table(input_file_path)
sorted_table = table.sort_by([("l_shipdate", "ascending")])
pq.write_table(sorted_table, output_file_path)
```

### GlareDB

Despite efforts, writing a sorted table into a Parquet file using GlareDB proved challenging, potentially due to a parallel Parquet writer implementation. As a workaround, we opted to delegate this task to `pyarrow.parquet`.

```python
con = glaredb.connect()    
sql = f"""SELECT * FROM parquet_scan('{input_file_path}') ORDER BY l_shipdate ASC"""
table = con.sql(sql).to_arrow()
pq.write_table(table, where=output_file_path)
```

### DataFusion

DataFusion is writing several limited size Parquet files located in a common folder. We did not find a way to change this file size threshold. In order to get a single sorted Parquet file, we used an arrow table and `pyarrow.parquet` to write a single Parquet file.

```python
ctx = SessionContext()
ctx.register_parquet("lineitem", input_file_path)
sorted_table = ctx.sql("SELECT * FROM lineitem ORDER BY l_shipdate").to_arrow_table()
pq.write_table(sorted_table, output_file_path)
```

### Others

Unfortunately, attempts to utilize [Dask](https://docs.dask.org/en/stable/) for the were unsuccessful, resulting in crashes, and we did not investigate much. It's worth noting that it performed well for a smaller table, such as with SF1 or SF3.

## Results

<p align="center">
  <img width="800" src="/img/2023-11-15_01/elapsed_time.png" alt="elapsed_time">
</p>

In the sorting performance comparison, DuckDB demonstrated the quickest elapsed time.

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