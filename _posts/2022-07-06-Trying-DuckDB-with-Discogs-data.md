---
title: Trying DuckDB with Discogs data
layout: post
comments: true
author: François Pacull & Romain Ferraton
tags: 
- Python
- DuckDB
- PostgreSQL
- SQL
- Pandas
- Arrow
---


<p align="center">
  <img width="400" src="https://github.com/duckdb/duckdb/blob/master/logo/DuckDB_Logo.png?raw=true" alt="DuckDB_Logo">
</p>

This notebook is a small example of using DuckDB with the Python API. 

* What is [DuckDB](https://github.com/duckdb/duckdb)? 

> DuckDB is an in-process SQL OLAP Database Management System

It is a relational DBMS that supports SQL. OLAP stands for *Online analytical processing*, generally associated with complex queries, scanning a lot of data in order to generate some reports, as opposed to processing transactions. DuckDB is speeding up the data analysis process when dealing with rather large datasets. A key factor of this efficiency is the columnar-vectorized query execution engine. *In-process* means that DuckDB does not follow the client-server model and runs entirely within the host process.



DuckDB is released under a MIT License. How do you install DuckDB with the Pyhton API? Well it is as simple as:

```bash
pip install duckdb
```

Voilà!

<p align="center">
  <img width="400" src="https://upload.wikimedia.org/wikipedia/en/thumb/e/e8/Discogs_logo.svg/2560px-Discogs_logo.svg.png" alt="Discogs_Logo">
</p>

* What is [Discogs](https://www.discogs.com/)? Here is a description from [wikipedia](https://en.wikipedia.org/wiki/Discogs):

> Discogs is a website and crowdsourced database of information about audio recordings, including commercial releases, promotional releases, and bootleg or off-label releases.

Monthly dumps of Discogs data (Release, Artist, Label, and Master Release data) can be downloaded from their [website](http://data.discogs.com/?prefix=data/2022/), as compressed XML files. We used the nifty tool [discogs-xml2db](https://github.com/philipmat/discogs-xml2db) to convert the XML files into CSV files, that we are going to load into DuckDB. We also used it to load the same data into PostgreSQL in order to later compare the execution time measures of a query.

In the following notebook, we are going to create a database with the following selection of tables (over 25+ available tables):


```python
SELECTED_TABLE_NAMES = [  # Selection of tables to import into the DB
    "artist",
    "artist_alias",
    "artist_namevariation",
    "release",
    "release_genre",
    "release_artist",
]
```

Let's start with the Python imports.

## Imports


```python
import glob
import os
import urllib
from time import perf_counter

import duckdb
import pandas as pd
import pyarrow.parquet as pq
from sqlalchemy import create_engine
```

Package versions:

    Python          : 3.9.13 
    duckdb          : 0.4.0
    pyarrow         : 8.0.0
    JupyterLab      : 3.4.3
    Pandas          : 1.4.3


We also import some Jupyter extensions to create some SQL cells later:


```python
%load_ext sql
%config SqlMagic.autopandas = True
%config SqlMagic.feedback = False
%config SqlMagic.displaycon = False
```

## Data Loading

In order to batch import the data into DuckDB, we need some CSV or Parquet files. Here we 6 CSV files, corresponding to the `SELECTED_TABLE_NAMES` from the Discogs database:


```python
CSV_FILES = "/home/francois/Data/Disk_1/discogs_data/*.csv"

# list the CSV files located in the data directory
csv_file_paths = glob.glob(CSV_FILES)
csv_file_paths.sort()

# look for the CSV files corresponding to the selected tables
tables = []
for csv_file_path in csv_file_paths:
    csv_file_name = os.path.basename(csv_file_path)
    table_name = os.path.splitext(csv_file_name)[0]
    if table_name in SELECTED_TABLE_NAMES:
        tables.append((table_name, csv_file_path))
        file_size = os.path.getsize(csv_file_path) / 1000000.0
        print(f"Table {table_name:20s} - CSV file size : {file_size:8.2f} MB")
```

    Table artist               - CSV file size :   580.77 MB
    Table artist_alias         - CSV file size :   105.62 MB
    Table artist_namevariation - CSV file size :    93.16 MB
    Table release              - CSV file size :  2540.58 MB
    Table release_artist       - CSV file size :  3973.41 MB
    Table release_genre        - CSV file size :   369.94 MB


First we need to connect to the DuckDB database, specifying a file path:

```Python
conn = duckdb.connect(db_path)
```

If the database file does not already exists, it is created. The data is then loaded from the CSV files into the database, for example using a `CREATE TABLE table_name AS SELECT * FROM 'csv_file_path'` command. The database is saved into disk as a DuckDB file. In the present case, data loading takes around 5 minutes the first time:

    CREATE TABLE artist AS SELECT * FROM '/home/francois/Data/Disk_1/discogs_data/artist.csv' - Elapsed time:   8.33 s
    CREATE TABLE artist_alias AS SELECT * FROM '/home/francois/Data/Disk_1/discogs_data/artist_alias.csv' - Elapsed time:   8.35 s
    CREATE TABLE artist_namevariation AS SELECT * FROM '/home/francois/Data/Disk_1/discogs_data/artist_namevariation.csv' - Elapsed time:   3.85 s
    CREATE TABLE release AS SELECT * FROM '/home/francois/Data/Disk_1/discogs_data/release.csv' - Elapsed time:  38.06 s
    CREATE TABLE release_artist AS SELECT * FROM '/home/francois/Data/Disk_1/discogs_data/release_artist.csv' - Elapsed time: 103.13 s
    CREATE TABLE release_genre AS SELECT * FROM '/home/francois/Data/Disk_1/discogs_data/release_genre.csv' - Elapsed time: 138.63 s
    CPU times: user 4min 47s, sys: 15.9 s, total: 5min 3s
    Wall time: 5min
 
Two files are actually created by DuckDB:
- DiscogsDB : 6.5GB
- DiscogsDB.wal : 319 MB
The `.wal` file is not necessary when we disconnect from the database, as it is a [checkpoint file](https://github.com/duckdb/duckdb/issues/301#issuecomment-1011335087). It is alctually removed when closing the connection, and not created when connecting to the database in `read_only` mode.

Subsequent connections only take a fraction of a second.


```python
db_path = "/home/francois/Data/Disk_1/discogs_data/DiscogsDB"
db_exists = os.path.isfile(db_path)
db_exists
```




    True



From the [documentation](https://duckdb.org/docs/api/python#startup--shutdown):

> If the database file does not exist, it will be created (the file extension may be `.db`, `.duckdb`, or anything else). The special value `:memory:` (the default) can be used to create an in-memory database. Note that for an in-memory database no data is persisted to disk (i.e. all data is lost when you exit the Python process). If you would like to connect to an existing database in read-only mode, you can set the `read_only` flag to `True`. Read-only mode is required if multiple Python processes want to access the same database file at the same time.


```python
%%time
if not db_exists:

    conn = duckdb.connect(db_path, read_only=False)

    # load the CSV files and create the tables
    for table_name, csv_file_path in tables:
        query = f"""CREATE TABLE {table_name} AS SELECT * FROM '{csv_file_path}'"""
        start = perf_counter()
        conn.execute(query)
        elapsed_time_s = perf_counter() - start
        print(f"{query} - Elapsed time: {elapsed_time_s:6.2f} s")

else:

    conn = duckdb.connect(db_path, read_only=True)
```

    CPU times: user 687 ms, sys: 187 ms, total: 873 ms
    Wall time: 869 ms


## Querying

Let's test if the connection is working, by executing a query with the `execute()` method:


```python
query = "SELECT COUNT(*) FROM artist"
conn.execute(query)
conn.fetchone()
```




    (8107524,)



Seems to be OK! 

### Pandas

We can use the `.df()` method if we want the query execution to return a Pandas dataframe:


```python
query = "SELECT * FROM artist LIMIT 5"
df = conn.execute(query).df()
df
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
      <th>id</th>
      <th>name</th>
      <th>realname</th>
      <th>profile</th>
      <th>data_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>The Persuader</td>
      <td>Jesper Dahlbäck</td>
      <td>NaN</td>
      <td>Needs Vote</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Mr. James Barth &amp; A.D.</td>
      <td>Cari Lekebusch &amp; Alexi Delano</td>
      <td>NaN</td>
      <td>Correct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Josh Wink</td>
      <td>Joshua Winkelman</td>
      <td>After forming [l=Ovum Recordings] as an indepe...</td>
      <td>Needs Vote</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Johannes Heil</td>
      <td>Johannes Heil</td>
      <td>Electronic music producer, musician and live p...</td>
      <td>Needs Vote</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Heiko Laux</td>
      <td>Heiko Laux</td>
      <td>German DJ and producer based in Berlin. He is ...</td>
      <td>Needs Vote</td>
    </tr>
  </tbody>
</table>
</div>



We could also create a table from a dataframe, if not in `read_only` mode, or query the dataframe in SQL:


```python
query = "SELECT COUNT(*) FROM df"
conn.execute(query).fetchone()
```




    (5,)



### Arrow

We can export the result of a query as a [PyArrow table](https://arrow.apache.org/docs/python/generated/pyarrow.Table.html):


```python
query = "SELECT * FROM artist LIMIT 1000"
tabl = conn.execute(query).arrow()
```


```python
type(tabl)
```




    pyarrow.lib.Table



And also query this arrow table:


```python
query = "SELECT COUNT(*) FROM tabl"
conn.execute(query).fetchone()
```




    (1000,)



## JupyterLab

We can also create some SQL cells directly using the DuckDB Python client. First we need to connect:


```python
%sql duckdb:///{db_path}
```

Then we can create SQL cells starting with the `%sql` magic command:


```python
%sql SELECT COUNT(*) FROM artist
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
      <th>count_star()</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8107524</td>
    </tr>
  </tbody>
</table>
</div>



What if we want a SQL cell to return a Pandas dataframe? We can assign the result of query to a dataframe using the `<<` operator. 


```python
%sql df << SELECT * FROM release_artist LIMIT 3
```

    Returning data to local variable df



```python
df
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
      <th>release_id</th>
      <th>artist_id</th>
      <th>artist_name</th>
      <th>...</th>
      <th>role</th>
      <th>tracks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>The Persuader</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>507025</td>
      <td>George Cutmaster General</td>
      <td>...</td>
      <td>Lacquer Cut By</td>
      <td>None</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>239</td>
      <td>Jesper Dahlbäck</td>
      <td>...</td>
      <td>Written-By [All Tracks By]</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>




```python
%sql SELECT * FROM df WHERE artist_name = 'The Persuader'
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
      <th>release_id</th>
      <th>artist_id</th>
      <th>artist_name</th>
      <th>...</th>
      <th>role</th>
      <th>tracks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1</td>
      <td>The Persuader</td>
      <td>...</td>
      <td>None</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



## Export

We can export the result of a SQL query into a CSV or a Parquet file.

### CSV export

A `COPY` statement can bu used to export the result of a query as a CSV file:


```python
csv_file_path = "./artist.csv"
```


```python
%sql COPY (SELECT * FROM artist LIMIT 1000) TO '{csv_file}' WITH (HEADER 1, DELIMITER ',');
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
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>



### Parquet export


```python
parquet_file_path = "./artist.parquet"
```


```python
%sql COPY (SELECT * FROM artist LIMIT 1000) TO '{parquet_file_path}' (FORMAT PARQUET);
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
      <th>Count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1000</td>
    </tr>
  </tbody>
</table>
</div>




```python
table = pq.read_table(parquet_file_path)
```


```python
table.shape
```




    (1000, 5)



The Parquet file can also be queried directly:


```python
%sql SELECT * FROM read_parquet('{parquet_file_path}') LIMIT 5;
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
      <th>id</th>
      <th>name</th>
      <th>realname</th>
      <th>profile</th>
      <th>data_quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>The Persuader</td>
      <td>Jesper Dahlbäck</td>
      <td>None</td>
      <td>Needs Vote</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>Mr. James Barth &amp; A.D.</td>
      <td>Cari Lekebusch &amp; Alexi Delano</td>
      <td>None</td>
      <td>Correct</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>Josh Wink</td>
      <td>Joshua Winkelman</td>
      <td>After forming [l=Ovum Recordings] as an indepe...</td>
      <td>Needs Vote</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>Johannes Heil</td>
      <td>Johannes Heil</td>
      <td>Electronic music producer, musician and live p...</td>
      <td>Needs Vote</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>Heiko Laux</td>
      <td>Heiko Laux</td>
      <td>German DJ and producer based in Berlin. He is ...</td>
      <td>Needs Vote</td>
    </tr>
  </tbody>
</table>
</div>



As explained in the [documentation](https://duckdb.org/docs/guides/import/query_parquet):
    
> The Parquet file will be processed in parallel. Filters will be automatically pushed down into the Parquet scan, and only the relevant columns will be read automatically.

There are many many more useful features available with DuckDB and we refer to the [documentation](https://duckdb.org/docs/) for  an overview of all what can be done with DuckDB. Now let's focus on a simple test case.

## A More complex query

Now we are going to create a little more complex query. We want to compute the number of releases per year and per music genre. So we only need to use 2 distinct tables: `release` and `release_genre`. One of the issues is that the `year_released` column from the contains a string entered by the Discogs contributors, that may vary in its form, but usually looks like `yyyy-mm-dd`. In order to get the release year, we are going to take the first 4 characters from the `year_released` field and assume it's a numeric year. 


```python
query = """SELECT se.year_released,
       se.genre,
       Count(*) AS release_count
FROM   (SELECT Substr(re.released, 1, 4) AS YEAR_RELEASED,
               re.id,
               re_g.genre
        FROM   release AS re
               LEFT JOIN release_genre AS re_g
                      ON re.id = re_g.release_id
        WHERE  re.released IS NOT NULL
               AND re.released NOT IN ( '?' )) AS se
GROUP  BY se.year_released,
          se.genre
ORDER  BY release_count DESC"""
```

### DuckDB


```python
start = perf_counter()
df1 = conn.execute(query).df()
elapsed_time_s_1 = perf_counter() - start
print(f"Elapsed time: {elapsed_time_s_1:6.2f} s")
df1.head()
```

    Elapsed time:   1.81 s





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
      <th>YEAR_RELEASED</th>
      <th>genre</th>
      <th>release_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>Electronic</td>
      <td>162166</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>Electronic</td>
      <td>156421</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>Electronic</td>
      <td>152393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>Electronic</td>
      <td>151043</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016</td>
      <td>Electronic</td>
      <td>148127</td>
    </tr>
  </tbody>
</table>
</div>




```python
df1.shape
```




    (1647, 3)



### PostgreSQL

PostgreSQL client and server are installed on the same computer as DuckDB.


```python
PG_USERNAME = "discogs_user"
PG_PASSWORD = "****"
PG_SERVER = "localhost"
PG_PORT = 5432
PG_DATABASE = "discogsdb"
CONNECT_STRING = (
    f"postgresql+psycopg2://{PG_USERNAME}:"
    + f"{urllib.parse.quote_plus(PG_PASSWORD)}@{PG_SERVER}:{PG_PORT}/{PG_DATABASE}"
)
```


```python
engine = create_engine(CONNECT_STRING)
```


```python
start = perf_counter()
df2 = pd.read_sql(query, engine)
elapsed_time_s_2 = perf_counter() - start
print(f"Elapsed time: {elapsed_time_s_2:6.2f} s")
df2.head()
```

    Elapsed time:  21.51 s





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
      <th>year_released</th>
      <th>genre</th>
      <th>release_count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020</td>
      <td>Electronic</td>
      <td>162166</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2018</td>
      <td>Electronic</td>
      <td>156421</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2019</td>
      <td>Electronic</td>
      <td>152393</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017</td>
      <td>Electronic</td>
      <td>151043</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2016</td>
      <td>Electronic</td>
      <td>148127</td>
    </tr>
  </tbody>
</table>
</div>




```python
df2.shape
```




    (1647, 3)




```python
df = pd.DataFrame(
    data={"DuckDB": [elapsed_time_s_1], "PostGreSQL": [elapsed_time_s_2]},
    index=[""],
)
ax = df.plot.bar(figsize=(6, 6))
_ = ax.set(title="Query execution time", xlabel="DB", ylabel="Elapsed time (s)")
```


<p align="center">
  <img width="500" src="/img/2022-07-06_01/output_61_0.png" alt="Query execution time">
</p>

So the same query on the same computer is executed about 10 times faster with DuckDB than with PostgreSQL (both DB with default settings). 

To conclude this post, let's plot the result of the previous query with Pandas/Matplotlib.

### Evolution of the 10 more popular genres ever

The first job is to convert the year to `int` and filter out entries such as `197`, `197?`, `70's`... Then we pivot the table in order to have each genre in a distinct column. We select years between 1940 and 2020 (data might be missing for the recent releases, in 2021 and 2022). 


```python
df1 = df1[df1.YEAR_RELEASED.str.isnumeric() & (df1.YEAR_RELEASED.str.len() == 4)].copy(
    deep=True
)
df1.YEAR_RELEASED = df1.YEAR_RELEASED.astype(int)
df1.rename(columns={"YEAR_RELEASED": "year"}, inplace=True)
df1 = df1.pivot_table(index="year", columns="genre", values="release_count")
df1 = df1.fillna(0)
df1 = df1.astype(int)
df1 = df1[(df1.index >= 1940) & (df1.index <= 2020)]
df1.sort_index(inplace=True)
order_max = df1.max().sort_values(ascending=False).index
df1 = df1[order_max[:10]]
df1.tail(3)
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
      <th>genre</th>
      <th>Electronic</th>
      <th>Rock</th>
      <th>Pop</th>
      <th>...</th>
      <th>Latin</th>
      <th>Reggae</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018</th>
      <td>156421</td>
      <td>124208</td>
      <td>41438</td>
      <td>...</td>
      <td>4106</td>
      <td>4321</td>
    </tr>
    <tr>
      <th>2019</th>
      <td>152393</td>
      <td>118760</td>
      <td>39830</td>
      <td>...</td>
      <td>4039</td>
      <td>4189</td>
    </tr>
    <tr>
      <th>2020</th>
      <td>162166</td>
      <td>115612</td>
      <td>37814</td>
      <td>...</td>
      <td>3280</td>
      <td>3813</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = df1.plot(figsize=(16, 10), legend=True, grid=True)
_ = ax.set(
    title=f"Number of annual releases by genre",
    xlabel="Year",
    ylabel="Number of releases",
)
```


<p align="center">
  <img width="800" src="/img/2022-07-06_01/output_64_0.png" alt="Number of annual releases by genre">
</p>


Kind of sad to see Jazz and Reggae so low.

### Close the DuckDB connection


```python
%%time
conn.close()
```

    CPU times: user 26.3 ms, sys: 4.55 ms, total: 30.8 ms
    Wall time: 29.1 ms

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