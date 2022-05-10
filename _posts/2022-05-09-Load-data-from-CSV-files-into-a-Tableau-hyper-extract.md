---
title: Load data from CSV files into a Tableau Hyper extract
layout: post
comments: true
author: François Pacull
tags: Python Tableau Hyper CSV
---

<p align="center">
  <img width="600" src="/img/2022-05-09_01/hyper_logo_1.jpg" alt="Hyper">
</p>


Hyper is Tableau’s *in-memory data engine technology, designed for fast data ingest and analytical query processing on large or complex data sets*. In the present notebook, we are going to create a Tableau Hyper extract from CSV files in Python. The goal is to compare the efficiency of different possible data ingestion techniques.

We are using [Tableau Hyper Python API](https://help.tableau.com/current/api/hyper_api/en-us/reference/py/index.html). The Hyper API is a toolbox to deal with Tableau extract (.hyper) files, and "automate the boring stuff". As described in the [Hyper SQL documentation](https://help.tableau.com/current/api/hyper_api/en-us/reference/sql/external-data-in-sql.html), Hyper has three different options to read external data in SQL:
> 1 - External data can be copied into a Hyper table with the COPY SQL command.  
> 2 - External data can be read directly in a SQL query using the set returning function external. In this case, no Hyper table is involved, so such a query can even be used if no database is attached to the current session.  
> 3 - External data can be exposed as if it was a table using the CREATE TEMPORARY EXTERNAL TABLE SQL command. It can then subsequently be queried using the name of the external table. Again, no Hyper table is involved; querying an external table will instead result in the data being read from the external source directly. 

Let's try the three strategies and apply them on a set of 4 CSV files with 1 million rows each. The tables in the CSV files have been created in Python with the [Faker](https://faker.readthedocs.io/en/master/) package, and written into CSV files with Pandas.

## Imports


```python
from time import perf_counter

from tableauhyperapi import (
    Connection,
    CreateMode,
    HyperProcess,
    Nullability,
    SqlType,
    TableDefinition,
    TableName,
    Telemetry,
)

DATABASE = "./test.hyper"  # hyper file database

# CSV file list
CSV_FILES = ["./test_01.csv", "./test_02.csv", "./test_03.csv", "./test_04.csv"]
csv_array = ", ".join(["'" + f + "'" for f in CSV_FILES])
```

## Create a connection


```python
hyper = HyperProcess(
    telemetry=Telemetry.DO_NOT_SEND_USAGE_DATA_TO_TABLEAU,
    parameters={"default_database_version": "2"},
)
connection = Connection(
        endpoint=hyper.endpoint,
        create_mode=CreateMode.CREATE_AND_REPLACE,
        database=DATABASE,
    )
```

## Create table definition


```python
# create schema
connection.catalog.create_schema("extract")

# create table
columns = []
data_types = {
    "name": SqlType.varchar(100),
    "job": SqlType.varchar(200),
    "birthdate": SqlType.date(),
    "email": SqlType.varchar(40),
    "last_connect": SqlType.timestamp(),
    "company": SqlType.varchar(150),
    "industry": SqlType.varchar(150),
    "city": SqlType.varchar(50),
    "state": SqlType.varchar(50),
    "zipcode": SqlType.varchar(15),
    "netNew": SqlType.bool(),
    "sales1_rounded": SqlType.int(),
    "sales2_decimal": SqlType.double(),
    "priority": SqlType.small_int(),
    "sales2_rounded": SqlType.int(),
}
is_nullable = Nullability.NOT_NULLABLE  # all columns are not nullable here
for column_name, dtype in data_types.items():
    columns.append(TableDefinition.Column(column_name, dtype, is_nullable))
table = TableName("extract", "faker")
table_def = TableDefinition(table_name=table, columns=columns)
connection.catalog.create_table(table_def)
```

## COPY

Here we loop on the 4 CSV files and insert them sequentially. However, each CSV reading step seems to be multithreaded.


```python
start = perf_counter()

for csv_file in CSV_FILES:
    copy_command = f"""COPY "extract"."faker"
    FROM '{csv_file}' WITH (FORMAT CSV, DELIMITER ',')"""
    _ = connection.execute_command(copy_command)

end = perf_counter()
elapsed_time = end - start
print(f"Elapsed time: {elapsed_time} s")
```

    Elapsed time: 8.200508020999678 s



```python
connection.execute_scalar_query("""SELECT COUNT(*) FROM  "extract"."faker" """)
```




    4000000




```python
# Cleanup
_ = connection.execute_command("""TRUNCATE TABLE "extract"."faker" """)
```

## INSERT SELECT FROM EXTERNAL TABLE


```python
start = perf_counter()
sql_command = f"""INSERT INTO "extract"."faker"
    SELECT * FROM external(
    ARRAY[{csv_array}],
    COLUMNS => DESCRIPTOR(
        name             varchar(100),
        job              varchar(200),
        birthdate        DATE,
        email            varchar(40),
        last_connect     timestamp,
        company          varchar(150),
        industry         varchar(150),
        city             varchar(50),
        state            varchar(50),
        zipcode          varchar(15),
        netNew           bool,
        sales1_rounded   int,
        sales2_decimal   double precision,
        priority         smallint,
        sales2_rounded   int
    ),
    FORMAT => 'csv', DELIMITER => ',')"""
_ = connection.execute_command(sql_command)
end = perf_counter()
elapsed_time = end - start
print(f"Elapsed time: {elapsed_time} s")
```

    Elapsed time: 11.32282723299977 s



```python
connection.execute_scalar_query("""SELECT COUNT(*) FROM  "extract"."faker" """)
```




    4000000




```python
# Cleanup
_ = connection.execute_command("""TRUNCATE TABLE "extract"."faker" """)
```

## CREATE EXTERNAL TABLE & INSERT SELECT


```python
start = perf_counter()
sql_command = f"""CREATE TEMP EXTERNAL TABLE faker (
    name             varchar(100),
    job              varchar(200),
    birthdate        DATE,
    email            varchar(40),
    last_connect     timestamp,
    company          varchar(150),
    industry         varchar(150),
    city             varchar(50),
    state            varchar(50),
    zipcode          varchar(15),
    netNew           bool,
    sales1_rounded   int,
    sales2_decimal   double precision,
    priority         smallint,
    sales2_rounded   int)
FOR ARRAY[{csv_array}]
WITH ( FORMAT => 'csv', DELIMITER => ',')"""
_ = connection.execute_command(sql_command)

sql_command = """INSERT INTO "extract"."faker" SELECT * FROM faker"""
_ =  connection.execute_command(sql_command)
end = perf_counter()
elapsed_time = end - start
print(f"Elapsed time: {elapsed_time} s")

```

    Elapsed time: 10.639697628000249 s



```python
connection.execute_scalar_query("""SELECT COUNT(*) FROM  "extract"."faker" """)
```




    4000000



## Close the connection & hyperprocess


```python
connection.close()
hyper.close()
```

## Conclusion

The COPY method is the fastest method to load CSV files into an Hyper extract.


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
