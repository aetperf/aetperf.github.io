---
title: TPC-H benchmark of DuckDB and Hyper on native files WIP
layout: post
comments: true
author: François Pacull
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

All the measurements are performed on the same laptop with a Linux OS. While it is possible to query Parquet files with both engines, we use the native file formats in the following:
- `.duchdb` for DuckDB
- `.hyper` for Tableau Hyper
It is usually more efficient to run the queries on the native file format, matching the engine internals, than on Parquet files.

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


## Results

Here are the time measurements. We only count the sum of the query execution time of the 22 queries. We do not take into account the data fetching time or the initial connection: 

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
      <th>engine</th>
      <th>DuckDB</th>
      <th>Hyper</th>
    </tr>
    <tr>
      <th>scale_factor</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>SF 1 (s)</th>
      <td>0.908807</td>
      <td>0.353306</td>
    </tr>
    <tr>
      <th>SF 3 (s)</th>
      <td>2.227890</td>
      <td>0.898468</td>
    </tr>
    <tr>
      <th>SF 10 (s)</th>
      <td>5.974569</td>
      <td>2.461864</td>
    </tr>
    <tr>
      <th>SF 30 (s)</th>
      <td>18.998208</td>
      <td>8.201111</td>
    </tr>
    <tr>
      <th>SF 100 (s)</th>
      <td>NaN</td>
      <td>35.263464</td>
    </tr>
  </tbody>
</table>
</div>

```

<p align="center">
  <img width="800" src="/img/2023-04-18_01/output_6_0.png" alt="linear_scale">
</p>


<p align="center">
  <img width="800" src="/img/2023-04-18_01/output_7_0.png" alt="log-scale">
</p>

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