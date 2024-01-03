---
title: Streaming data from PostgreSQL to a CSV file
layout: post
comments: true
author: François Pacull
tags: 
- Python
- SQL
- PostgreSQL
- ADBC
- COPY
- Psycopg2
- DuckDB
- TurbODBC
- PyArrow
---

In this post, we explore the process of streaming data from a PostgreSQL database to a CSV file using Python. The primary goal is to avoid loading the entire dataset into memory, enabling a more scalable and resource-efficient approach.

We'll try out various Python libraries and techniques and evaluate their performance in terms of elapsed time.

## System and package versions

We are operating on Python version 3.11.7 and running on a Linux x86_64 machine.


    adbc-driver-postgresql : 0.8.0
    duckdb                 : 0.9.2
    pandas                 : 2.1.4
    psycopg2               : 2.9.9
    pyarrow                : 14.0.2
    sqlalchemy             : 2.0.24
    turbodbc               : 4.9.0


## Imports

```python
import csv
import json
import os
import time

import adbc_driver_postgresql.dbapi
import duckdb
import pandas as pd
import psycopg2
import pyarrow as pa
import pyarrow.csv
from sqlalchemy import create_engine
from turbodbc import connect, make_options

# file paths for the CSV output and database credentials,
csv_file_path = os.path.abspath("./test_lineitem.csv")
credentials_file_path = os.path.abspath("./credentials.json")
```

While the credentials retrieved below serve the classic connection method, we also employ an ODBC connection later on, with a Data Source Name (DSN).

```python
with open(credentials_file_path, "r") as json_file:
    creds = json.load(json_file)
print("creds keys: ", list(creds.keys()))
```

    creds keys:  ['username', 'password', 'server', 'port', 'database']

```python
# initialization of the dict storing the elapsed time for different approaches
elapsed_time = {}
```

## Query

The [TPC-H](https://www.tpc.org/tpch/) benchmark stands as a widely-utilized metric for assessing the performance of database systems. The data can be created using pre-determined database sizes, referred to as *scale factors*. In this context, we are utilizing a scale factor of 10, focusing on the largest table among the 8 TPCH tables: `lineitem`. This table comprises approximately 60 million (59,986,052) rows and 16 columns. Below is a glimpse of the first 10 rows of the table:

    ┌────────────┬───────────┬───────────┬───┬────────────┬──────────────────────┐
    │ l_orderkey │ l_partkey │ l_suppkey │ … │ l_shipmode │      l_comment       │
    │   int64    │   int64   │   int64   │   │  varchar   │       varchar        │
    ├────────────┼───────────┼───────────┼───┼────────────┼──────────────────────┤
    │          1 │   1551894 │     76910 │ … │ TRUCK      │ to beans x-ray car…  │
    │          1 │    673091 │     73092 │ … │ MAIL       │  according to the …  │
    │          1 │    636998 │     36999 │ … │ REG AIR    │ ourts cajole above…  │
    │          1 │     21315 │     46316 │ … │ AIR        │ s cajole busily ab…  │
    │          1 │    240267 │     15274 │ … │ FOB        │  the regular, regu…  │
    │          1 │    156345 │      6348 │ … │ MAIL       │ rouches. special     │
    │          2 │   1061698 │     11719 │ … │ RAIL       │ re. enticingly reg…  │
    │          3 │     42970 │     17971 │ … │ AIR        │ s cajole above the…  │
    │          3 │    190355 │     65359 │ … │ RAIL       │ ecial pinto beans.…  │
    │          3 │   1284483 │     34508 │ … │ SHIP       │ e carefully fina     │
    ├────────────┴───────────┴───────────┴───┴────────────┴──────────────────────┤
    │ 10 rows                                               16 columns (5 shown) │
    └────────────────────────────────────────────────────────────────────────────┘

Additionally, we instruct the SQL engine to sort the table by the `l_orderkey` column.

```python
sql = "SELECT * FROM tpch_10.lineitem ORDER BY l_orderkey"
```
The following approaches will employ the same SQL query.

## Pandas


The code creates a [SQLAlchemy](https://www.sqlalchemy.org/) engine for PostgreSQL using the [psycopg2](https://www.psycopg.org/docs/) driver, specifying the connection details from the credentials.

The `stream_results=True` and `max_row_buffer=chunk_size` connection options are used to stream the results and limit the number of rows buffered in memory.

The code then iterates through the chunks obtained from the SQL query using `pd.read_sql` from [Pandas](https://pandas.pydata.org/). It writes each chunk to a CSV file in *append* mode after the first chunk.

```python
%%time
start_time_step = time.perf_counter()
engine = create_engine(
    f"postgresql+psycopg2://{creds['username']}:{creds['password']}@{creds['server']}:{creds['port']}/{creds['database']}"
)
chunk_size = 100_000
with engine.connect().execution_options(
    stream_results=True, max_row_buffer=chunk_size
) as conn:
    first_chunk = True
    mode = "w"
    for df in pd.read_sql(sql=sql, con=conn, chunksize=chunk_size):
        df.to_csv(
            csv_file_path, mode=mode, header=first_chunk, quoting=csv.QUOTE_ALL, index=False
        )
        if first_chunk:
            mode = "a"
            first_chunk = False
elapsed_time["Pandas"] = time.perf_counter() - start_time_step
```

    CPU times: user 6min 26s, sys: 8.84 s, total: 6min 35s
    Wall time: 7min 17s


The CSV file containing the data from the PostgreSQL table, occupies approximately 10 gigabytes of disk space. Note that if we don't activate the `stream_results` option, the data is fully loaded into memory by the `read_sql` statement, even with smaller chunks. This does not fit in the available RAM on the computer.

Next we display the first three and last three rows of the generated CSV file:

```python
!head -n 3 {csv_file_path}
```

    "l_orderkey","l_partkey","l_suppkey","l_linenumber","l_quantity","l_extendedprice","l_discount","l_tax","l_returnflag","l_linestatus","l_shipdate","l_commitdate","l_receiptdate","l_shipinstruct","l_shipmode","l_comment"
    "1","1551894","76910","1","17.0","33078.94","0.04","0.02","N","O","1996-03-13","1996-02-12","1996-03-22","DELIVER IN PERSON        ","TRUCK     ","to beans x-ray carefull"
    "1","673091","73092","2","36.0","38306.16","0.09","0.06","N","O","1996-04-12","1996-02-28","1996-04-20","TAKE BACK RETURN         ","MAIL      "," according to the final foxes. qui"



```python
!tail -n 3 {csv_file_path}
```

    "60000000","118838","93842","5","28.0","51991.24","0.0","0.08","N","O","1997-09-29","1997-11-06","1997-09-30","COLLECT COD              ","SHIP      ","regular foxes among the even depths use "
    "60000000","1294851","19864","6","48.0","88597.92","0.03","0.07","N","O","1997-11-28","1997-10-05","1997-12-06","COLLECT COD              ","MAIL      ","ual asymptotes wake af"
    "60000000","558286","33302","7","12.0","16131.12","0.02","0.05","N","O","1997-10-09","1997-10-27","1997-10-21","COLLECT COD              ","REG AIR   ","ickly according to the furiousl"


## Pandas + PyArrow

In this section we use the Pandas library with the `pyarrow` backend in combination with [PyArrow](https://arrow.apache.org/docs/python/index.html) for seamlessly handle Arrow data, i.e. write the CSV file. There should be no data copy between Pandas and PyArrow. 

We are still using a SQLAlchemy engine with the psycopg2 driver and the `stream_results=True` and `max_row_buffer=chunk_size` connection mode.

```python
%%time
start_time_step = time.perf_counter()
engine = create_engine(
    f"postgresql+psycopg2://{creds['username']}:{creds['password']}@{creds['server']}:{creds['port']}/{creds['database']}"
)
chunk_size = 100_000
write_options = pa.csv.WriteOptions(
    include_header=True,
    batch_size=2_048,
    delimiter=";",
    quoting_style="all_valid",
)
is_writer = False
with engine.connect().execution_options(
    stream_results=True, max_row_buffer=chunk_size
) as conn:
    for df in pd.read_sql(
        sql=sql, con=conn, chunksize=chunk_size, dtype_backend="pyarrow"
    ):
        table = pa.Table.from_pandas(df)
        if not is_writer:
            schema = table.schema
            writer = pa.csv.CSVWriter(
                sink=csv_file_path, schema=schema, write_options=write_options
            )
            is_writer = True
        writer.write_table(table)
    writer.close()
elapsed_time["Pandas+PyArrow"] = time.perf_counter() - start_time_step
```

    CPU times: user 5min 8s, sys: 8.2 s, total: 5min 16s
    Wall time: 5min 50s


## Turbodbc + PyArrow

[turbodbc](https://turbodbc.readthedocs.io/en/latest/) is module to access relational databases via the Open Database Connectivity (ODBC) interface. Unlike previous approaches, we establish the connection using the data source name (DSN). Notably, asynchronous I/O is activated during data retrieval, allowing the fetching of new result sets from the database in the background while Python processes the existing ones.

```python
%%time
start_time_step = time.perf_counter()
options = make_options(
    use_async_io=True,
    prefer_unicode=True,
)
write_options = pa.csv.WriteOptions(
    include_header=True,
    batch_size=2048,
    delimiter=";",
    quoting_style="all_valid",
)
with connect(connection_string="dsn=PostgreSQL", turbodbc_options=options) as conn:
    with conn.cursor() as cur:
        _ = cur.execute(sql=sql)
        is_writer = False
        for table in cur.fetcharrowbatches():
            if not is_writer:
                schema = table.schema
                writer = pa.csv.CSVWriter(
                    sink=csv_file_path, schema=schema, write_options=write_options
                )
                is_writer = True
            writer.write_table(table)
        writer.close()
elapsed_time["Turbodbc+PyArrow"] = time.perf_counter() - start_time_step
```

    CPU times: user 2min 34s, sys: 6.08 s, total: 2min 40s
    Wall time: 2min 4s


## Psycopg2

Psycopg is a wrapper for the [libpq](https://www.postgresql.org/docs/current/libpq.html) C library, the official PostgreSQL client library. We are going to use `COPY () TO STDOUT` which streams data to the client, along with the [`copy_expert`](https://www.psycopg.org/docs/cursor.html#cursor.copy_expert) method.


```python
%%time
start_time_step = time.perf_counter()
output_query = f"COPY ({sql}) TO STDOUT WITH (FORMAT CSV, HEADER true, DELIMITER ';', FORCE_QUOTE *)"
with psycopg2.connect(
    host=creds["server"],
    user=creds["username"],
    password=creds["password"],
    dbname=creds["database"],
    port=creds["port"],
) as conn:
    with conn.cursor() as cur:
        with open(csv_file_path, "w") as f:
            cur.copy_expert(output_query, f)
elapsed_time["Psycopg2"] = time.perf_counter() - start_time_step
```

    CPU times: user 9.72 s, sys: 7.31 s, total: 17 s
    Wall time: 51.9 s


## ADBC + PyArrow


In this section, we leverage ADBC in conjunction with PyArrow to fetch data as Arrow batches and subsequently write it to a CSV file. [ABDC](https://arrow.apache.org/docs/format/ADBC.html) stands for the Arrow Database Connectivity:

> ADBC aims to provide a minimal database client API standard, based on Arrow, for C, Go, and Java (with bindings for other languages). Applications code to this API standard (in much the same way as they would with JDBC or ODBC), but fetch result sets in Arrow format (e.g. via the [C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html)). They then link to an implementation of the standard: either directly to a vendor-supplied driver for a particular database, or to a driver manager that abstracts across multiple drivers. Drivers implement the standard using a database-specific API, such as Flight SQL.

The ADBC implementation used here is specifically tailored for PostgreSQL, demonstrating its compatibility with various databases, including DuckDB, Snowflake, SQLite, and any connection employing the Flight SQL Driver.

Data is fetched as Arrow batches and written with PyArrow. 

```python
%%time
start_time_step = time.perf_counter()
write_options = pa.csv.WriteOptions(
    include_header=True,
    delimiter=";",
    quoting_style="all_valid",
)
uri = f"postgresql://{creds['username']}:{creds['password']}@{creds['server']}:{creds['port']}/{creds['database']}"
with adbc_driver_postgresql.dbapi.connect(uri) as conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        is_writer = False
        for batch in cur.fetch_record_batch():
            if not is_writer:
                schema = batch.schema
                writer = pa.csv.CSVWriter(
                    sink=csv_file_path, schema=schema, write_options=write_options
                )
                is_writer = True
            writer.write_batch(batch)
        writer.close()
elapsed_time["ADBC+PyArrow"] = time.perf_counter() - start_time_step
```

    CPU times: user 1min 5s, sys: 5.95 s, total: 1min 11s
    Wall time: 1min 11s


## DuckDB

In this segment, we explore the integration of the in-process database [DuckDB](https://duckdb.org/) with its [postgres](https://duckdb.org/docs/extensions/postgres.html) extension, allowing for bidirectional data transfer between DuckDB and PostgreSQL: 

> The postgres extension allows DuckDB to directly read and write data from a running Postgres database instance. The data can be queried directly from the underlying Postgres database. Data can be loaded from Postgres tables into DuckDB tables, or vice versa.

To enable the PostgreSQL extension, a straightforward SQL command is employed: `INSTALL postgres;`. Following this installation, and the extension loading, the `ATTACH` command is utilized to make the PostgreSQL database accessible to DuckDB. In this process, an alias, in this case, `db`, is assigned to the database, and accordingly, SQL queries need to be adjusted to consider this alias. For instance, the SELECT query is modified as follows:

```python
sql_duckdb = sql.replace("tpch_", "db.tpch_")
sql_duckdb
```

    'SELECT * FROM db.tpch_10.lineitem ORDER BY l_orderkey'

An important consideration is DuckDB's default memory setting, which is configured to use 80% of available RAM. Given DuckDB's efficiency in memory usage, this default setting is sufficient for the current operation. However, when dealing with larger tables, such as the TPCH `lineitem` table generated with a scale factor of 100, a more controlled memory limit may be desired. In this example, the memory limit is set to 16 GB using the configuration command:

An important consideration is DuckDB's default memory setting, which is configured to use 80% of available RAM. Given DuckDB's efficiency in memory usage, this default setting is sufficient for the current operation. However this would not be enough if we were dealing with a larger table, for example the TPCH lineitem table generated with scale factor 100. 

Because we want to reach the memory limit  and see how it is smoothly handled by DuckDB, we set the memory limit to a smaller size : 16 GB, in order to reach the limit and see how it is smoothly handled by DuckDB. This is done with the configuration command `SET memory_limit = '16GB';`


```python
%%time
start_time_step = time.perf_counter()
output_query = (
    f"COPY ({sql_duckdb}) TO '{csv_file_path}' (HEADER, DELIMITER ';', force_quote *);"
)
conn = duckdb.connect()
conn.sql("INSTALL postgres;")
conn.sql("LOAD postgres;")
conn.sql(
    f"""ATTACH 'dbname={creds["database"]} user={creds["username"]} password={creds["password"]} host={creds["server"]} port={creds["port"]}' 
    AS db (TYPE postgres);"""
)
conn.sql("SET memory_limit = '16GB';")
conn.sql(output_query)
conn.close()
elapsed_time["DuckDB"] = time.perf_counter() - start_time_step
```

    CPU times: user 3min 38s, sys: 2min 14s, total: 5min 53s
    Wall time: 38.7 s


## Results


```python
df = pd.DataFrame.from_dict(elapsed_time, orient="index", columns=["Elapsed time (s)"])
```


```python
ax = df.sort_values(by="Elapsed time (s)").plot.barh(
    alpha=0.7, legend=False, figsize=(6, 4)
)
_ = ax.set(title="TPCH-SF10 lineitem table CSV extract", xlabel="Elapsed time (s)")
```

<p align="center">
  <img width="800" src="/img/2024-01-03_01/output_41_0.png" alt="Sample points queries mosaic">
</p>  

This example demonstrates DuckDB's effective management of system resources, showcasing its ability in handling memory constraints and efficiently leveraging CPU resources during data operations.

## Postcript

Besides the DuckDB exctraction, all the other streaming approaches use a small amout of memory, about a few gigabytes in this particular TPCH case. DuckDB stands out due to its unique capability of intentionally allocating a fixed amount of memory to speed up the processing. It's important to note that DuckDB's performance might suffer if provided with insufficient memory. To explore this further, let's experiment with different sizes of the memory_limit parameter and observe its impact on the elapsed time.

<p align="center">
  <img width="800" src="/img/2024-01-03_01/Selection_124.png" alt="DuckDB - Elapsed time vs memory limit">
</p>

Additionally, it's worth mentioning that due to the time constraints during the preparation of this post, we were unable to employ [Polars](https://pola.rs/) for the streaming extraction, leading to an out-of-memory error.

Also, it appears that [ConnectorX](https://sfu-db.github.io/connector-x/intro.html) currently lacks support for [retrieving results as Arrow batches or any type of chunks](https://github.com/sfu-db/connector-x/issues/264).


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