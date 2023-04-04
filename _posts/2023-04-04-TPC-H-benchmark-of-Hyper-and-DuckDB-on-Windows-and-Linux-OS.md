---
title: TPC-H benchmark of Hyper and DuckDB on Windows and Linux OS
layout: post
comments: true
author: François Pacull & Romain Ferraton
tags: 
- TPC-H 
- benchmark
- SQL
- Parquet
- DuckDB
- Hyper
- Python
- Windows
- Linux
---


In this blog post, we explore the use of two SQL engines, and specifically their Python API, for querying files. The engines in focus are :
- [Tableau Hyper](https://help.tableau.com/current/api/hyper_api/en-us/index.html) / Proprietary License
- [DuckDB](https://duckdb.org/) / MIT License

They both support SQL queries and prioritize the efficiency of analytical SQL queries. To evaluate their performance, we conducted the widely-used TPC-H benchmark on a laptop with a **dual-boot Linux/Windows setup**. We ran the same queries on the same hardware, applied to the same files, with the same libraries on both operating systems, namely Windows 11 home and Linux Mint 21.1 (based on Ubuntu 22.04 Jammy Jellyfish). Our findings include a comparison of the performance of each engine on Parquet and native files, allowing us to measure the performance gap between the two platforms. 

The chosen scale factor of 10 corresponds to a dataset of modest/medium size. Let us describe data files.

## TPC-H SF10

The TPC-H data used in this benchmark is generated using the DuckDB [TPC-H extension](https://duckdb.org/docs/extensions/overview.html#all-available-extensions) and saved into : 
- Parquet files with default compression "snappy" and row group size 122880
- hyper database file
- duckdb database file

The benchmark comprises 8 tables, with a rather scale factor of 10 used for data generation. Each table is stored in a separate Parquet file.

Here's a brief overview of each table:

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

CPU : 12th Gen Intel© Core™ i9-12900H (10 cores)  
RAM : 32 GB 
Data disk : Samsung SSD 980 PRO 1TB 

Package versions:
```
Python          : 3.11.0 | packaged by conda-forge  
DuckDB          : 0.7.2-dev1381  
TableauHyperAPI : 0.0.16638  
```

## Parquet files attachment and specific parameters

The attachment process is chosen in a way that the data is not scanned, being almost instantaneous.

The Parquet files are attached to the Hyper process as temporary external tables, e.g.:

```sql
CREATE TEMPORARY EXTERNAL TABLE region FOR './region.parquet'
```

The Parquet files are attached to DuckDB as views:

```sql
CREATE VIEW region AS SELECT * FROM read_parquet('./region.parquet')
```

## Query execution time

In the following sections, we report the elapsed time for each of the 22 TPC-H queries. To ensure accuracy and reduce the impact of fluctuations, we executed each query three times and recorded the best elapsed time out of the three runs.

We did not include fetch time in the elapsed time. We only measure the query execution time. The data is fetched in a second step in order to check the number of returned rows.

### DuckDB engine on the duckdb file

<p align="center">
  <img width="1200" src="/img/2023-04-04_01/duckdb_duckdb.png" alt="duckdb_duckdb">
</p>

### DuckDB engine on the Parquet files

<p align="center">
  <img width="1200" src="/img/2023-04-04_01/duckdb_parquet.png" alt="duckdb_parquet">
</p>

### Hyper engine on the hyper file

<p align="center">
  <img width="1200" src="/img/2023-04-04_01/hyper_hyper.png" alt="hyper_hyper">
</p>

### Hyper engine on the Parquet files

<p align="center">
  <img width="1200" src="/img/2023-04-04_01/hyper_parquet.png" alt="hyper_parquet">
</p>

## Total TPC-H run

We executed the entire TPC-H benchmark, consisting of 22 queries, on both operating systems and recorded the best elapsed time out of three runs. The resulting total Elapsed Time (E.T.), in seconds, is shown in the table below:

| Engine - file type |  Linux E.T. (s)|   Windows E.T. (s)|
|:-----------------|--------:|----------:|
| DuckDB - duckdb  |   7.688 |    11.923 |
| DuckDB - parquet |   9.096 |    14.784 |
| Hyper - hyper    |   2.773 |     3.831 |
| Hyper - parquet  |   5.896 |    10.625 |

<p align="center">
  <img width="600" src="/img/2023-04-04_01/total_et.png" alt="total_et">
</p>

## Conlusion

Our results showed that both engines were more efficient on native file formats compared to Parquet files, with Hyper outperforming DuckDB on this specific TPC-H benchmark. Additionally, we observed that the total elapsed time on Windows was on average 40-80% longer than on Linux.


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