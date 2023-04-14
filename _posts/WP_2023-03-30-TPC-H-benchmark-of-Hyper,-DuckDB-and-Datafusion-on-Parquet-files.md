# TPC-H benchmark of Hyper, DuckDB and DataFusion on Parquet files


**Update** Apr 14, 2023 - An issue has been opened on the DataFusion GitHub repository regarding its poor reported performance compared to DuckDB and Hyper: [#5942](https://github.com/apache/arrow-datafusion/issues/5942). While there may be multiple factors contributing to this unexpected behavior, I might have used the API in a sub-optimal way. I will continue to update the post with new findings.

<p align="center">
  <img width="300" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-03-30_01/parquet_logo.jpg" alt="parquet">
</p>

In this blog post, we focus on directly querying Parquet files using three different SQL engines, and more specifically their Python API: 
- [Tableau Hyper](https://help.tableau.com/current/api/hyper_api/en-us/index.html) / Proprietary License
- [DuckDB](https://duckdb.org/) / MIT License
- [Apache Arrow DataFusion](https://arrow.apache.org/datafusion/) / Apache License 2.0

The [TPC-H](https://www.tpc.org/tpch/) benchmark is a widely-used measure of such systems' performance, consisting of a set of queries that have broad industry-wide relevance. We executed the TPC-H benchmark on a laptop and present our findings on the performance and capabilities of each engine on Parquet files. 

## TPC-H SF100

The TPC-H data used in this benchmark is generated using the DuckDB [TPC-H extension](https://duckdb.org/docs/extensions/overview.html#all-available-extensions) and saved into Parquet files with default compression "snappy" and row group size 122880. The benchmark comprises 8 tables, with a scale factor of 100 used for data generation. Each table is stored in a separate Parquet file.

Here's a brief overview of each table:

| Table name  | Row count  | Parquet file size |
|---|--:|--:|
| region | 5 | 1.0 kB |
| nation | 25 | 2.2 kB |
| supplier | 1 000 000 | 80.4 MB |
| customer | 15 000 000 | 1.3 GB |
| part | 20 000 000 | 695.4 MB |
| partsupp | 80 000 000 | 4.5 GB |
| orders | 150 000 000 | 6.8 GB |
| lineitem | 600 037 902 | 27.1 GB |

The `lineitem` table is the largest, with over 600 million rows and a file size of 27.1 GB.

## TPC-H queries

There are 22 queries, specified in the TPC-H benchmark; they may vary a little bit depending on each implemnetation. The queries used in this post can be found [here](https://raw.githubusercontent.com/aetperf/tpch/main/queries/duckdb/queries_native.sql) on Github. Here is the first one, for example:

```sql
SELECT
    --Query01
    l_returnflag,
    l_linestatus,
    SUM(l_quantity) AS sum_qty,
    SUM(l_extendedprice) AS sum_base_price,
    SUM(l_extendedprice * (1 - l_discount)) AS sum_disc_price,
    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge,
    AVG(l_quantity) AS avg_qty,
    AVG(l_extendedprice) AS avg_price,
    AVG(l_discount) AS avg_disc,
    COUNT(*) AS count_order
FROM
    lineitem
WHERE
    l_shipdate <= CAST('1998-09-02' AS date)
GROUP BY
    l_returnflag,
    l_linestatus
ORDER BY
    l_returnflag,
    l_linestatus;
```

## System information

The queries are executed on a laptop with the following features:

CPU : 12th Gen Intel© Core™ i9-12900H, 10 cores  
RAM : 32 GB  
OS : Linux mint 21.1, based on Ubuntu 22.04   
Data disk : Samsung SSD 980 PRO 1TB 

Package versions:
```
Python          : 3.11.0 | packaged by conda-forge  
DuckDB          : 0.7.2-dev982  
TableauHyperAPI : 0.0.16638  
Datafusion      : 20.0.0  
```

### Parquet files attachment and specific parameters

The attachment process is chosen in a way that the data is not scanned, being almost instantaneous.

#### Hyper

We used a specific parameter for the Hyper engine, following a discussion with the Tableau Hyper team on [Slack](tableau-datadev.slack.com):

```python
with HyperProcess(
    telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
    parameters=dict({("external_table_sample_size_factor", "0.005")}),
) as hyper:
```

Without this setting, query 9 would crash with an out-of-memory error. The Parquet files are attached as temporary external tables, e.g.:

```sql
CREATE TEMPORARY EXTERNAL TABLE region FOR './region.parquet'
```

#### DuckDB

The default configuration is used. The Parquet files are attached as views:

```sql
CREATE VIEW region AS SELECT * FROM read_parquet('./region.parquet')
```
Query 21 is crashing with a *cannot allocate memory* error. 

#### DataFusion

We tried a few parameters such as `enable_page_index`, `pushdown_filters`, `reorder_filters` but without success... The default configuration seems to be limited and we did not figure out how to adjust the settings. Queries 7, 17, 18 and 21 are crashing.

The Parquet files are attached using the Python API:

```python
ctx = datafusion.SessionContext()
ctx.register_parquet('region', './region.parquet')
```

## Results

Only the Hyper engine succeed in running all the queries, in a total elapsed time of **63.90 s**, with the connection setup, loop on files etc... Note that this time can be significantly improved by using the respective native storage format of the DuckDB or Hyper engines.

We did not include fetch time in the elapsed time, except for Datafusion. So for DuckDB and Hyper, we only measure the query execution time. The data is fetched in a second step in order to check the number of returned rows.

- DuckDB:


```python
# start timer
conn.execute(query)
# stop timer

result = conn.df()
n_returned_rows = result.shape[0]
```

- Hyper

```python
# start timer
result = conn.execute_query(query)
# stop timer

n_returned_rows = 0
while result.next_row():
    n_returned_rows += 1
result.close()
```

- Datafusion

```python
# start timer
result = ctx.sql(query)
data = result.collect()
# stop timer

n_returned_rows = 0 
for _, item in enumerate(data):
    n_returned_rows += item.num_rows
```

Overall, in this test, Datafusion is behind, while Hyper is a bit more efficient than DuckDB, specifically on some queries. 

|   query |  Hyper |  DuckDB | Datafusion |
|--------:|-------:|--------:|-----------:|
|       1 | 3.626  |   3.793 |     81.668 |
|       2 | 1.006  |   0.849 |     13.944 |
|       3 | 2.619  |   2.854 |     39.433 |
|       4 | 1.896  |   5.010 |     30.512 |
|       5 | 3.521  |   3.131 |     51.143 |
|       6 | 1.221  |   2.162 |     19.821 |
|       7 | 2.660  |   6.300 |        |
|       8 | 2.151  |   3.021 |     49.717 |
|       9 | 6.085  |  10.374 |     65.606 |
|      10 | 2.820  |   3.825 |     41.010 |
|      11 | 0.360  |   0.416 |     14.066 |
|      12 | 1.991  |   2.773 |     38.985 |
|      13 | 7.303  |   4.588 |     56.703 |
|      14 | 1.402  |   1.936 |     18.684 |
|      15 | 1.634  |   4.254 |     27.640 |
|      16 | 0.882  |   0.848 |      6.699 |
|      17 | 1.728  |  12.502 |         |
|      18 | 5.917  |  15.003 |         |
|      19 | 3.250  |   4.018 |     42.300 |
|      20 | 1.124  |   4.919 |     80.375 |
|      21 | 4.430  |      |         |
|      22 | 0.995  |   1.938 |     11.485 |
|     Sum | 58.631 | 94.523  | 689.797 |

At the bottom of this table, we display the sum of the querying time, ignoring the failing queries.

### All three engines

<p align="center">
  <img width="1200" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-03-30_01/figure_01.png" alt="all_engines">
</p>

### DuckDB vs Hyper

<p align="center">
  <img width="1200" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-03-30_01/figure_02.png" alt="duckdb_vs_hyper">
</p>

## Conclusion

In this blog post, we conducted the TPC-H benchmark using three different SQL engines: Tableau Hyper, DuckDB, and Apache Arrow DataFusion, for querying Parquet files. The benchmark comprised 22 queries executed on 8 tables, with a scale factor of 100 used for data generation.

With the default settings, it seems that Datafusion may not be the most suitable option for the specific use case mentioned due to its slow performance and tendency to crash. On the other hand, the Hyper engine was found to be faster than the DuckDB engine. But ultimately, the choice of database engine depends on a variety of factors such as the license, the size and complexity of the dataset, the nature of the queries, the available hardware resources... 

Overall, the benchmark demonstrated the potential of these SQL engines for handling a significant amount of data stored in Parquet files.