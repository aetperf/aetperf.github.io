---
title: TPC-H benchmark of DuckDB and Hyper on native files WIP
layout: post
comments: true
author: François Pacull
tags: 
- TPC-H 
- benchmark
- SQL
- DuckDB
- Hyper
- Python
- Linux
---


In this blog post, we explore the use of two SQL engines, and specifically their Python API, for querying files. The engines in focus are :
- [Tableau Hyper](https://help.tableau.com/current/api/hyper_api/en-us/index.html) / Proprietary License
- [DuckDB](https://duckdb.org/) / MIT License

The [TPC-H](https://www.tpc.org/tpch/) benchmark is a widely-used measure of such systems' performance, consisting of a set of queries that have broad industry-wide relevance. The data can be created using pre-determined database sizes, referred to as *scale factors*. In the following with are going to use these scale factors: 
- 1  
- 3  
- 10  
- 30  
- 100  

All the measurements are performed on the same laptop with a Linux OS. While it is possible to query Parquet files with both engines, we use the native file formats in the following:
- *.duckdb* for DuckDB
- *.hyper* for Tableau Hyper
It is usually more efficient to run the queries on the native file format, matching the engine internals, than on Parquet files.

Note that the DuckDB storage format is not always backward compatible, because is under development and not stabilized yet. It will be though when version 1 is introduced. So basically, a *.duckdb* file written with a given version must be read with the same version.

## Package versions:

    Python          : 3.11.3 | packaged by conda-forge | (main, Apr  6 2023, 08:57:19) [GCC 11.3.0]
    DuckDB          : 0.7.2-dev2144
    TableauHyperAPI : 0.0.16868


## System information

The code is executed on a linux laptop with the following features:

    OS : Linux mint 21.1, based on Ubuntu 22.04  
    CPU : 12th Gen Intel© Core™ i9-12900H (10 cores)    
    RAM : 32 GB  
    Data disk : Samsung SSD 980 PRO 1TB  

## Native file size

| Scale factor | *.duckdb* file size  | *.hyper* file size | Total row count |
|----:|----------:|----------:|------------:|
|   1 |  436.0 MB |  436.5 MB |   8 661 245 |
|   3 |  800.6 MB |    1.3 GB |  25 976 639 |
|  10 |    2.7 GB |    4.5 GB |  86 586 082 |
|  30 |    8.2 GB |   13.6 GB | 259 798 402 |
| 100 |   27.7 GB |   46.3 GB | 866 037 932 |

The total row count corresponds to the sum of 8 table lengths:
- lineitem  
- customer  
- orders  
- supplier  
- region  
- partsupp  
- nation  
- part  

## Results

### Query execution time

W report the combined elapsed time for the 22 TPC-H queries. To ensure accuracy and reduce the impact of fluctuations, we executed each query three times and recorded the best elapsed time out of the three runs. These 22 best elapsed times are then summed. 

We did not include fetch time in the elapsed time. We only measure the query execution time. The data is fetched in a second step in order to check the number of returned rows.

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

| Scale factor| DuckDB (s)  | Hyper (s) |
|---------------:|-------:|------:|
|              1 |   0.75 |  0.29 |
|              3 |   2.04 |  0.79 |
|             10 |   5.95 |  2.46 |
|             30 |  18.99 |  8.02 |
|            100 |    NaN | 34.63 |

On the scale factor 100 data, query 21 is crashing when using DuckDB Engine with a *cannot allocate memory* error. 

<p align="center">
  <img width="800" src="/img/2023-04-18_01/output_6_0.png" alt="linear_scale">
</p>


<p align="center">
  <img width="800" src="/img/2023-04-18_01/output_7_0.png" alt="log-scale">
</p>

### Total running time


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