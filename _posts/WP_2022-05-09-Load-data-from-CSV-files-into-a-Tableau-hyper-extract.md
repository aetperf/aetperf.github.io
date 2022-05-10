# Loading data from CSV files into a Tableau Hyper extract

<p align="center">
  <img width="300" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2022-05-09_01/hyper_logo_1.jpg" alt="Hyper">
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
csv_array_str = ", ".join(["'" + f + "'" for f in CSV_FILES])
```

## Create a connection

We start a local Hyper server instance first, and create a connection. We could also use a context manager here, so that we wouldn't have to close them explicitly at the end.


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

We create a table named `faker` in the `extract` schema, with 15 columns of various types. 

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
is_nullable = Nullability.NOT_NULLABLE
for column_name, dtype in data_types.items():
    columns.append(TableDefinition.Column(column_name, dtype, is_nullable))
table = TableName("extract", "faker")
table_def = TableDefinition(table_name=table, columns=columns)
connection.catalog.create_table(table_def)
```

## 1 - COPY

Here we loop on the 4 CSV files and insert them sequentially.

```python
start = perf_counter()

for csv_file in CSV_FILES:
    copy_command = f"""COPY "extract"."faker"
    FROM '{csv_file}' WITH (FORMAT CSV, DELIMITER ',')"""
    _ = connection.execute_command(copy_command)

end = perf_counter()
elapsed_time = end - start
print(f"Elapsed time: {elapsed_time:6.2f} s")
```

    Elapsed time:   7.90 s


```python
connection.execute_scalar_query("""SELECT COUNT(*) FROM  "extract"."faker" """)
```


    4000000



```python
# Cleanup
_ = connection.execute_command("""TRUNCATE TABLE "extract"."faker" """)
```

## 2 - INSERT SELECT FROM EXTERNAL TABLE


```python
start = perf_counter()
sql_command = f"""INSERT INTO "extract"."faker"
    SELECT * FROM external(
    ARRAY[{csv_array_str}],
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
print(f"Elapsed time: {elapsed_time:6.2f} s")
```

    Elapsed time:  11.35 s



```python
connection.execute_scalar_query("""SELECT COUNT(*) FROM  "extract"."faker" """)
```


    4000000



```python
# Cleanup
_ = connection.execute_command("""TRUNCATE TABLE "extract"."faker" """)
```

## 3 - CREATE EXTERNAL TABLE & INSERT SELECT


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
FOR ARRAY[{csv_array_str}]
WITH ( FORMAT => 'csv', DELIMITER => ',')"""
_ = connection.execute_command(sql_command)

sql_command = """INSERT INTO "extract"."faker" SELECT * FROM faker"""
_ =  connection.execute_command(sql_command)
end = perf_counter()
elapsed_time = end - start
print(f"Elapsed time: {elapsed_time:6.2f} s")

```

    Elapsed time: 12.88 s



```python
connection.execute_scalar_query("""SELECT COUNT(*) FROM  "extract"."faker" """)
```




    4000000



## Close the connection & hyperprocess


```python
connection.close()
hyper.close()
```

The COPY method seems to be the most efficient for loading data from CSV files into Hyper extracts. It benefits from some multi-threading while the `INSERT` techniques appear to be single-theaded all the way.
