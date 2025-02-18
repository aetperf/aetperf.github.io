
<p align="center">
  <img width="900" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2025-02-18_01/duck_01.png" alt="duck">
</p>

[DuckDB](https://duckdb.org/) is an in-process SQL database that allows you to query data from various sources, including private AWS s3 buckets. This Python notebook demonstrates how to use DuckDB's "secrets" feature to manage authentication credentials for querying private s3 data.

Imagine you need to access a private compressed CSV file located at the following path:

```python
s3_FILE_PATH = "s3://ap-dvf-data/dvf_zip/full2014.csv.gz"
```

Here's how you can do it.

## Imports

This section imports the necessary libraries, including [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) for AWS interactions.

```python
import os

import boto3
import duckdb
```

Package versions:

    Python               : 3.13.0  
    duckdb               : 1.2.0
    boto3                : 1.36.22

## Recover AWS credentials programmatically

The credentials are sourced from the AWS profile configured on your computer. Ensure that the user profile has the necessary permissions [s3:GetObject", "s3:ListBucket", ...] on the targeted s3 bucket. We can check the ListBucket permission with `aws cli`:

```bash
aws s3 ls s3://ap-dvf-data/dvf_zip/
```
    2025-02-12 11:25:45   71732630 full2014.csv.gz
    2025-02-12 11:25:53   77832978 full2015.csv.gz

So let's fetch the credentials:

```python
session = boto3.Session()
credentials = session.get_credentials()
current_credentials = credentials.get_frozen_credentials()
aws_access_key_id = current_credentials.access_key
aws_secret_access_key = current_credentials.secret_key
```

We could have chosen another AWS profile than the default one. This can be done with the `--profile` argument in aws cli, for example:

```bash
aws s3 ls s3://ap-dvf-data/dvf_zip/ --profile john_doe
```

Then we would also create a boto3 session using the specified profile:

```python
session = boto3.Session(profile_name=john_doe)
```

## Create and use a DuckDB secret

We establish a connection to a new DuckDB database in the *in-memory* mode:

```python
con = duckdb.connect()
```

First let's attempt to query the file without any secret:

```python
read_test_query = (
    f"SELECT * FROM read_csv('{s3_FILE_PATH}', delim=',', strict_mode=false) LIMIT 3;"
)
try:
    con.sql(read_test_query)
except duckdb.HTTPException as e:
    print(f"HTTPException - {e}")
```

    HTTPException - HTTP Error: HTTP GET error on 'https://ap-dvf-data.s3.amazonaws.com/dvf_zip/full2014.csv.gz' (HTTP 403)

So let's create the secret in the DuckDB database for accessing the S3 bucket:

```python
sql_create_secret = f"""
CREATE SECRET s3_dvf (
    TYPE s3,
    KEY_ID '{aws_access_key_id}',
    SECRET '{aws_secret_access_key}',
    SCOPE 's3://ap-dvf-data',
    REGION 'eu-west-1'
);"""
con.sql(sql_create_secret)
```

    ┌─────────┐
    │ Success │
    │ boolean │
    ├─────────┤
    │ true    │
    └─────────┘

By default the secret is temporary. `CREATE SECRET` is equivalent to `CREATE TEMPORARY SECRET`. It is stored within the DuckDB in-memory database session. We could also use a `PERSISTENT` secret with `CREATE PERSISTENT SECRET`, which would allow us to query the s3 bucket without recreating the secret in a follow-up session. However, this persistent secret would be stored in unencrypted binary format on the disk, in the default `.duckdb` folder. This permanent secret directory could also be changed with `SET secret_directory=...`

<p align="center">
  <img width="900" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2025-02-18_01/create_secret.png" alt="create_secret">
</p>
Credits: figure from DuckDB's [documentation](https://duckdb.org/docs/sql/statements/create_secret.html)

Secret Provider needs to be provided with the `TYPE` argument. In the present case this is `S3`, but others are supported, for example `GCS`, `R2`, or `AZURE`. The `REGION` argument is important for `S3`. 

Note that the `SCOPE` is optional. Here is its description from DuckDB's [documentation](https://duckdb.org/docs/configuration/secrets_manager.html#types-of-secrets):

> Secrets can also have an optional scope, which is a file path prefix that the secret applies to. When fetching a secret for a path, the secret scopes are compared to the path, returning the matching secret for the path. In the case of multiple matching secrets, the longest prefix is chosen.

We can list all the secrets currently configured in DuckDB :

```python
sql_list_secrets = (
    "SELECT name, type, persistent, storage, scope FROM duckdb_secrets();"
)
con.sql(sql_list_secrets)
```


    ┌─────────┬─────────┬───────────┬─────────┬────────────────────┐
    │  name   │  type   │persistent │ storage │       scope        │
    │ varchar │ varchar │ boolean   │ varchar │     varchar[]      │
    ├─────────┼─────────┼───────────┼─────────┼────────────────────┤
    │ s3_dvf  │ s3      │false      │ memory  │ [s3://ap-dvf-data] │
    └─────────┴─────────┴───────────┴─────────┴────────────────────┘


We can also retrieve the secret associated with a specific s3 file path.:

```python
sql_which_secret = f"FROM which_secret('{s3_FILE_PATH}', 's3');"
con.sql(sql_which_secret)
```


    ┌─────────┬────────────┬─────────┐
    │  name   │ persistent │ storage │
    │ varchar │  varchar   │ varchar │
    ├─────────┼────────────┼─────────┤
    │ s3_dvf  │ TEMPORARY  │ memory  │
    └─────────┴────────────┴─────────┘


Now let's execute the query again and store the result in a Pandas DataFrame, now with the secret in place :


```python
%%time
read_test_query = f"""SELECT 
          id_mutation, 
          date_mutation, 
          nature_mutation, 
          valeur_fonciere, 
          type_local 
        FROM read_csv('{s3_FILE_PATH}', delim=',', strict_mode=false) 
        LIMIT 3;"""
df = con.sql(read_test_query).df()
df.head()
```

    CPU times: user 2.45 s, sys: 147 ms, total: 2.59 s
    Wall time: 3.43 s



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id_mutation</th>
      <th>date_mutation</th>
      <th>nature_mutation</th>
      <th>valeur_fonciere</th>
      <th>type_local</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2014-1</td>
      <td>2014-01-09</td>
      <td>Vente</td>
      <td>251500.0</td>
      <td>Maison</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2014-2</td>
      <td>2014-01-09</td>
      <td>Vente</td>
      <td>174500.0</td>
      <td>Appartement</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2014-2</td>
      <td>2014-01-09</td>
      <td>Vente</td>
      <td>174500.0</td>
      <td>Dépendance</td>
    </tr>
  </tbody>
</table>
</div>



The response time is kind of large because we are querying a rather large CSV file, it would be more efficient to query a Parquet file.

For some reason, we could also remove the secret from DuckDB to revoke access in the following way:

```python
sql_drop_secret = "DROP SECRET s3_dvf;"
con.sql(sql_drop_secret)
```

```python
con.sql(sql_list_secrets)
```

    ┌─────────┬─────────┬──────────┬────────────┬─────────┬───────────┐
    │  name   │  type   │ provider │ persistent │ storage │   scope   │
    │ varchar │ varchar │ varchar  │  boolean   │ varchar │ varchar[] │
    ├─────────┴─────────┴──────────┴────────────┴─────────┴───────────┤
    │                             0 rows                              │
    └─────────────────────────────────────────────────────────────────┘


```python
con.close()
```