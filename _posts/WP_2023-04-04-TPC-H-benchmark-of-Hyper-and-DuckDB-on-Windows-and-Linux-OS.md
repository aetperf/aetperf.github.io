# TPC-H benchmark of Hyper and DuckDB on Windows and Linux OS


In this blog post, we explore the use of two SQL engines, and specifically their Python API, for querying files. The engines in focus are :
- [Tableau Hyper](https://help.tableau.com/current/api/hyper_api/en-us/index.html) / Proprietary License
- [DuckDB](https://duckdb.org/) / MIT License

They both support SQL queries and prioritize the efficiency of analytical SQL queries. To evaluate their performance, we conducted the widely-used TPC-H benchmark on a laptop with a **dual-boot Linux/Windows setup**. We ran the same queries on the same hardware, applied to the same files, with the same libraries on both operating systems, namely Windows 11 home and Linux Mint 21.1 - based on Ubuntu 22.04 Jammy Jellyfish. Our findings include a comparison of the performance of each engine on Parquet and native files, allowing us to measure the performance gap between the two platforms. 

The chosen scale factor of 10 corresponds to a dataset of modest/medium size. Let us list the different data files.

## TPC-H SF10

The TPC-H data used in this benchmark is generated using the DuckDB [TPC-H extension](https://duckdb.org/docs/extensions/overview.html#all-available-extensions) and saved into : 
- Parquet files with default compression "snappy" and row group size 122880
- an hyper database file
- a duckdb database file

A scale factor of 10 is used for data generation. The benchmark comprises 8 tables. Each table is stored in a separate Parquet file. Here's a brief overview of each table:

| Table name  | Row count  | Parquet file size |
|---|--:|--:|
| region | 5 | 1.0 kB |
| nation | 25 | 2.2 kB |
| supplier | 100 000 | 8.0 MB |
| customer | 1 500 000 | 126.3 MB |
| part | 2 000 000 | 69.5 MB |
| partsupp | 8 000 000 | 453.3 MB |
| orders | 15 000 000 | 620.8 GB |
| lineitem | 59 986 052 | 2.7 GB |

The `lineitem` table is the largest, with about 60 million rows. The `.duckdb` database file has a size of 2.6 GB, while the `.hyper` database one is around 4.5 GB.

## System information

The queries are executed on a laptop with the following features:

CPU : 12th Gen Intel© Core™ i9-12900H - 10 cores  
RAM : 32 GB   
Data disk : Samsung SSD 980 PRO 1TB   

Package versions:
```
Python          : 3.11.0 | packaged by conda-forge  
DuckDB          : 0.7.2-dev1381  
TableauHyperAPI : 0.0.16638  
```

## Files attachment to the engine

### Parquet files

The Parquet attachment process is chosen in a way that the data is not scanned, being almost instantaneous. 

- Hyper

The Parquet files are attached to the Hyper process as temporary external tables:

```sql
CREATE TEMPORARY EXTERNAL TABLE region FOR 'path-to/table.parquet'
```

- DuckDB

The Parquet files are attached to DuckDB as views:

```sql
CREATE VIEW region AS SELECT * FROM read_parquet('path-to/table.parquet')
```

### Native file formats

For the native file formats, the database files are given as an argument to the connection constructor.

- Tableau Hyper API

```python
hyper = HyperProcess(telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU)
conn = Connection(
    endpoint=hyper.endpoint,
    database='path-to/data.hyper',
    create_mode=CreateMode.NONE,
)
````

- DuckDB Python API

```python
conn = duckdb.connect(database='path-to/data.duckdb')
```

## Query execution time

In the following sections, we report the elapsed time for each of the 22 TPC-H queries. To ensure accuracy and reduce the impact of fluctuations, we executed each query three times and recorded the best elapsed time out of the three runs.

We did not include fetch time in the elapsed time. We only measure the query execution time. The data is fetched in a second step in order to check the number of returned rows.

### DuckDB engine on the duckdb file

<p align="center">
  <img width="1000" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-04-04_01/duckdb_duckdb.png" alt="duckdb_duckdb">
</p>

### DuckDB engine on the Parquet files

<p align="center">
  <img width="1000" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-04-04_01/duckdb_parquet.png" alt="duckdb_parquet">
</p>

### Hyper engine on the hyper file

<p align="center">
  <img width="1000" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-04-04_01/hyper_hyper.png" alt="hyper_hyper">
</p>

### Hyper engine on the Parquet files

<p align="center">
  <img width="1000" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-04-04_01/hyper_parquet.png" alt="hyper_parquet">
</p>

## Total TPC-H run

We executed the entire TPC-H benchmark, consisting of 22 queries, on both operating systems and recorded the best elapsed time out of three runs. The resulting total Elapsed Time [E.T.] in seconds, is shown in the table below:

<table align="center">
    <tr>
        <td>Engine - file type</td>
        <td>Linux E.T. (s)</td>
        <td>Windows E.T. (s)</td>
    </tr>
    <tr>
        <td>DuckDB - duckdb</td>
        <td>7.688</td>
        <td>11.923</td>
    </tr>
    <tr>
        <td>DuckDB - parquet</td>
        <td>9.096</td>
        <td>14.784</td>
    </tr>
    <tr>
        <td>Hyper - hyper</td>
        <td>2.773</td>
        <td>3.831</td>
    </tr>
    <tr>
        <td>Hyper - parquet</td>
        <td>5.896</td>
        <td>10.625</td>
    </tr>
</table>

<p align="center">
  <img width="600" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-04-04_01/total_et.png" alt="total_et">
</p>

## Conlusion

Our results showed that both engines were more efficient on native file formats compared to Parquet files, with Hyper outperforming DuckDB on this specific TPC-H benchmark. Additionally, we observed that the total elapsed time on Windows was on average 40-80% longer than on Linux.
