---
title: Loading data from PostgreSQL to Pandas with ConnectorX
layout: post
comments: true
author: François Pacull
tags: Python Pandas PostgreSQL DataFrame Loading data SQLAlchemy
---

[ConnectorX](https://sfu-db.github.io/connector-x/intro.html) is a library written in Rust, which enables fast and memory-efficient data loading from various databases to different dataframes. We refer to an interesting [paper](https://wooya.me/files/ConnectorX.pdf): 

Wang, Xiaoying, et al. *ConnectorX: Accelerating Data Loading From Databases to Dataframes.* 2021

They provide a detailed analysis for the `pandas.read_sql` function, which lead to some surprises:

> A surprising finding is that the majority of the time is actually spent on the client side rather than on the query execution or the data transfer.

Most of the time is spent on deserialization and conversion to dataframe. Besides optimizing this process, they also implement an efficient parallelization based on query partitioning.

In the present post, we want to load some data in Python from PostgreSQL to Pandas. We have a table created in Python with the [Faker](https://github.com/joke2k/faker) package. Initially, this table has been loaded into PostgreSQL:

<p align="center">
  <img width="1000" src="/img/2022-04-20_01/faker.png" alt="faker">
</p>

The table has 1000000 rows and 16 columns. The query used to load the data is rather basic:

```python
QUERY = 'SELECT * FROM "faker_s1000000i"'
```

We are going to compare the loading time using different drivers and methods. We do not measure the peak memory, but just the elapsed time for loading the data into Pandas. Database server and client are on the same machine.

## Imports


```python
import urllib

import connectorx as cx
import pandas as pd
import psycopg2
import pyodbc
import ray
import turbodbc
from sqlalchemy import create_engine

ray.init(ignore_reinit_error=True)
```

```python
pg_username = "****"
pg_password = "****"
pg_server = "localhost"
pg_port = 5432
pg_database = "pgdb"
```

## SQLAlchemy + psycopg2

This is the most common way for loading data from Postgres into Pandas.


```python
connect_string = f"postgresql+psycopg2://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"
engine_psycopg2 = create_engine(connect_string)
```


```python
%%time
with engine_psycopg2.connect() as connection:
    df = pd.read_sql(sql=QUERY, con=connection)
```

    CPU times: user 7.06 s, sys: 755 ms, total: 7.81 s
    Wall time: 8.54 s




```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

## SQLAlchemy + psycopg2 by chunks

Now let's imagine that we want to reduce the memory usage of the previous process by loading the data by chunks.


```python
%%time
chunksize = 1_000
engine = create_engine(connect_string)
connection = engine.connect().execution_options(
    stream_results=True, max_row_buffer=chunksize
)
dfs = []
for df in pd.read_sql(QUERY, connection, chunksize=chunksize):
    dfs.append(df)
df = pd.concat(dfs)
connection.close()
```

    CPU times: user 9.39 s, sys: 428 ms, total: 9.82 s
    Wall time: 11 s


```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

## pyodbc
 
The official PostgreSQL ODBC driver is used here with default settings.


```python
connection = pyodbc.connect(dsn="PostgreSQL")
```


```python
%%time
df = pd.read_sql(sql=QUERY, con=connection)
```

    /home/francois/miniconda3/envs/sql/lib/python3.9/site-packages/pandas/io/sql.py:761: UserWarning: pandas only support SQLAlchemy connectable(engine/connection) ordatabase string URI or sqlite3 DBAPI2 connectionother DBAPI2 objects are not tested, please consider using SQLAlchemy
      warnings.warn(


    CPU times: user 16.9 s, sys: 723 ms, total: 17.6 s
    Wall time: 17.7 s



```python
connection.close()
```


```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

## Turbodbc arrow

Here we do not use Pandas'`read_sql` function but Apache Arrow along with Turbodbc. See this blog post for more details: https://arrow.apache.org/blog/2017/06/16/turbodbc-arrow/


```python
connection = turbodbc.connect(dsn="PostgreSQL")
```


```python
%%time
cursor = connection.cursor()
cursor.execute(QUERY)
data = cursor.fetchallarrow()
df = data.to_pandas(split_blocks=False, date_as_object=True)
```

    CPU times: user 7.17 s, sys: 374 ms, total: 7.55 s
    Wall time: 7.8 s



```python
connection.close()
```


```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

## ConnectorX

We are going to connect the data source with this connection string URI:


```python
connect_string = f"postgres://{pg_username}:{urllib.parse.quote_plus(pg_password)}@{pg_server}:{pg_port}/{pg_database}"
```

### Pandas


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="pandas")
```

    CPU times: user 3.71 s, sys: 278 ms, total: 3.99 s
    Wall time: 3.96 s



```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

### Arrow


```python
%%time
table = cx.read_sql(conn=connect_string, query=QUERY, return_type="arrow")
```

    CPU times: user 3.2 s, sys: 237 ms, total: 3.43 s
    Wall time: 3.35 s



```python
type(table)
```




    pyarrow.lib.Table



Since the returning type is not a Pandas dataframe but an Arrow table, we need to convert it (and take the conversion time into account):


```python
%%time
df = table.to_pandas()
```

    CPU times: user 1.21 s, sys: 209 ms, total: 1.42 s
    Wall time: 1.4 s



```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```


### Modin


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="modin")
```

    UserWarning: The pandas version installed 1.4.2 does not match the supported pandas version in Modin 1.4.1. This may cause undesired side effects!
    UserWarning: Distributing <class 'pandas.core.frame.DataFrame'> object. This may take some time.


    CPU times: user 6.09 s, sys: 629 ms, total: 6.72 s
    Wall time: 6.78 s



```python
type(df)
```




    modin.pandas.dataframe.DataFrame



Here we need to convert from a Modin to a Pandas dataframe:


```python
%%time
df = df._to_pandas()
```

    CPU times: user 1.28 s, sys: 266 ms, total: 1.55 s
    Wall time: 1.53 s




```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```


### Dask


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="dask")
```

    CPU times: user 5.4 s, sys: 528 ms, total: 5.92 s
    Wall time: 5.88 s



```python
type(df)
```




    dask.dataframe.core.DataFrame



Each partition in a Dask DataFrame is a Pandas DataFrame. Since we are using a single partition in the present case, converting to Pandas is almost instentaneous:


```python
%%time
df = df.compute()
```

    CPU times: user 1.53 ms, sys: 86 µs, total: 1.62 ms
    Wall time: 1.15 ms




```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

### Polars


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="polars")
```

    CPU times: user 3.34 s, sys: 255 ms, total: 3.59 s
    Wall time: 3.51 s



```python
type(df)
```




    polars.internals.frame.DataFrame



Again, we need to convert from a Polars to a Pandas dataframe :


```python
%%time
df = df.to_pandas()
```

    CPU times: user 848 ms, sys: 149 ms, total: 997 ms
    Wall time: 982 ms


```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```


### Pandas 2 partitions


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="pandas", partition_on="index", partition_num=2)
```

    CPU times: user 4.77 s, sys: 596 ms, total: 5.37 s
    Wall time: 2.88 s



```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

### Pandas 4 partitions


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="pandas", partition_on="index", partition_num=4)
```

    CPU times: user 5.95 s, sys: 657 ms, total: 6.61 s
    Wall time: 2.2 s




```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```

### Pandas 8 partitions


```python
%%time
df = cx.read_sql(conn=connect_string, query=QUERY, return_type="pandas", partition_on="index", partition_num=8)
```

    CPU times: user 5.72 s, sys: 689 ms, total: 6.41 s
    Wall time: 2.28 s




```python
assert isinstance(df, pd.core.frame.DataFrame)
assert df.shape == (1000000, 16)
```


## Comparison


We did not include here the code associated with time measurement. Basically, for each startegy, the loading process is repeated 5 times and only the best elapsed time is kept.

<p align="center">
  <img width="1000" src="/img/2022-04-20_01/output_92_0.png" alt="Elapsed time">
</p>


We can see that ConnectorX is an efficient tool. We did not measure memory efficiency in this post, but it is supposed to be interesting as well. The supported data sources are the following (from the [documentation](https://sfu-db.github.io/connector-x/intro.html)): Postgres, Mysql, Mariadb (through mysql protocol), Sqlite, Redshift (through postgres protocol), Clickhouse (through mysql protocol), SQL Server, Azure SQL Database (through mssql protocol), Oracle. It would be great if more data sources could be supported, such as SAP Hana for example!

Although we loaded the data into a Pandas dataframe in the present post, we could also imagine to use ConnectorX along with [Polars](https://github.com/pola-rs/polars) dataframes to perform some data analysis tasks.


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