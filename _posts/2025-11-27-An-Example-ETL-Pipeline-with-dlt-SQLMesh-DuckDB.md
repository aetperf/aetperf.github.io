---
title: An example ETL Pipeline with dlt + SQLMesh + DuckDB
layout: post
comments: true
author: François Pacull
categories: [Data Engineering, Python]
tags:
- dlt
- SQLMesh
- DuckDB
- ETL
- data pipeline
- Python
- data engineering
- data transformation
- SQL
- open source
---

<p align="center">
  <img width="600" src="/img/2025-11-27_01/postmodern.jpg" alt="A Typical postmodern painting, AI generated">
</p>

In this post, we walk through building a basic **ETL (Extract-Transform-Load)** pipeline, which is a common pattern for moving and transforming data between systems. This is a toy example, intentionally over-simplistic, but it helped us explore how three modern Python tools work together.

The stack we used:

- **[dlt](https://dlthub.com/product/dlt) (data load tool)**: Handles both extraction, pulling data from Yahoo Finance into DuckDB, and loading, pushing transformed data to SQL Server
- **[SQLMesh](https://www.tobikodata.com/sqlmesh)**: Manages SQL transformations with helpful features like version control, column-level lineage, and incremental processing
- **[DuckDB](https://duckdb.org/)**: Serves as our in-process analytical database, no server setup required

**A note before we begin**: I'm familiar with DuckDB and SQLGlot, the SQL parser that SQLMesh is using under the hood I guess, but I'm a newcomer to both dlt and SQLMesh, so take this as a learning notebook.

**Outline**
- [About the Tools](#about_the_tools)
- [Setup & Configuration](#setup_configuration)
- [Extract: dlt + yfinance](#extract)
- [SQLMesh Project Setup](#sqlmesh_project_setup)
- [Transform: Run SQLMesh Pipeline](#transform)
- [Verify: Query Transformed Data](#verify)
- [Lineage](#lineage)
- [Load: dlt to SQL Server](#load)
- [Production Deployment: Logging & Return Codes](#production_deployment)
- [Further Engineering Considerations](#further_engineering)
- [References](#references)

Here is a basic diagram showing the pipeline architecture

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────┐
│    EXTRACT      │      │    TRANSFORM     │      │      LOAD       │
│                 │      │                  │      │                 │
│  dlt pipeline   │──▶───│  SQLMesh Models  │──▶───│  dlt pipeline   │
│  (yfinance      │      │  (DuckDB Engine) │      │  (DuckDB →      │
│   → DuckDB)     │      │                  │      │   SQL Server)   │
│                 │      │  raw → marts     │      │                 │
│                 │      │  (50-day SMA)    │      │                 │
└─────────────────┘      └──────────────────┘      └─────────────────┘
```

## About the Tools<a name="about_the_tools"></a>

Here's a brief overview of each tool.

### dlt (data load tool)
dlt is an open-source Python library designed to simplify data loading. What we found particularly useful is how it automates schema inference, data normalization, and incremental loading. Some key features:
- **Declarative pipelines**: You define sources as Python generators using the `@dlt.resource` decorator
- **Schema evolution**: It automatically handles schema changes as your data evolves
- **Multiple connectors**: Works with DuckDB, SQL Server, BigQuery, Snowflake, and many others

### SQLMesh
SQLMesh is a data transformation framework that brings software engineering practices to SQL development. We appreciated how it helped us think about transformations more systematically:
- **Version control**: Track changes to your SQL models over time
- **Column-level lineage**: See exactly which source columns affect which outputs, helpful for debugging
- **Virtual Data Environments**: Test changes without duplicating data

### DuckDB
DuckDB is an embedded analytical database optimized for OLAP (Online Analytical Processing) workloads, for analytics queries over large datasets rather than transactional operations:
- **In-process**: No separate server to install or manage
- **Fast**: Uses columnar storage with vectorized execution
- **Handles relatively large data**: Out-of-core processing lets it work with datasets larger than available RAM

### Python Requirements

```
duckdb
sqlmesh
polars
pandas  # for Yahoo Finance data ressource
yfinance
pyodbc
sqlalchemy
dlt[duckdb,mssql]
duckdb-engine
```

## Setup & Configuration<a name="setup_configuration"></a>

Let's start by importing the necessary libraries and defining the configuration. We are using stock market data as the example dataset; it's freely available via Yahoo Finance and has enough complexity to demonstrate the transformation capabilities.


```python
import json
import os
import subprocess
import warnings
from datetime import date

import dlt
import duckdb
import pandas as pd
import polars as pl
import yfinance as yf
from sqlglot import lineage
from sqlmesh import Context

warnings.filterwarnings("ignore", category=DeprecationWarning)

# Configuration
DUCKDB_FILE = "./financial_etl_dlt.duckdb"
SQLMESH_PROJECT_DIR = "./dlt_sqlmesh_project"
CREDENTIALS_FILE = "./credentials.json"

# Stock configuration
TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
START_DATE = "2020-01-01"
END_DATE = date.today().isoformat()

# SQL Server target configuration
TARGET_CONN_KEY = "ms_target_01"  # key in credentials JSON file
TARGET_SCHEMA = "dbo"
TARGET_TABLE = "Dim_Stock_Metrics"

print("Configuration loaded successfully!")
print(f"  - DuckDB file: {DUCKDB_FILE}")
print(f"  - SQLMesh project: {SQLMESH_PROJECT_DIR}")
print(f"  - Tickers: {TICKERS}")
print(f"  - Date range: {START_DATE} to {END_DATE}")
```

    Configuration loaded successfully!
      - DuckDB file: ./financial_etl_dlt.duckdb
      - SQLMesh project: ./dlt_sqlmesh_project
      - Tickers: ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA']
      - Date range: 2020-01-01 to 2025-11-27


## Extract: dlt + yfinance<a name="extract"></a>

With the configuration in place, we can move to the **Extract** phase. Here, we use dlt to pull stock data from Yahoo Finance and load it directly into DuckDB. The data includes **OHLCV** values: Open, High, Low, Close (prices), and Volume.

### Creating a Custom dlt Resource

dlt uses the `@dlt.resource` decorator to define data sources. Each resource is a Python generator that yields data rows:

```python
@dlt.resource(table_name="my_table")
def my_source():
    yield {"col1": "value1", "col2": 123}
```

Key parameters:
- `table_name`: Name of the destination table
- `write_disposition`: How to handle existing data (`replace`, `append`, `merge`)
- `columns`: Optional schema hints for data types


```python
# Define a custom dlt resource for Yahoo Finance data
@dlt.resource(
    table_name="eod_prices_raw",
    write_disposition="replace",
    columns={
        "ticker": {"data_type": "text"},
        "date": {"data_type": "date"},
        "open_price": {"data_type": "double"},
        "high_price": {"data_type": "double"},
        "low_price": {"data_type": "double"},
        "close_price": {"data_type": "double"},
        "volume": {"data_type": "bigint"},
    },
)
def yfinance_eod_prices(tickers: list[str], start_date: str, end_date: str):
    """
    Extract End-of-Day stock prices from Yahoo Finance.

    This is a custom dlt resource that yields OHLCV data for multiple tickers.
    dlt will automatically:
    - Infer the schema from the yielded dictionaries
    - Handle batching and loading to the destination
    - Track load metadata (_dlt_load_id, _dlt_id)
    """
    print(f"Downloading OHLCV data for {len(tickers)} tickers...")

    # Download data for all tickers at once
    data = yf.download(
        tickers, start=start_date, end=end_date, progress=False, auto_adjust=True
    )

    if data.empty:
        raise ValueError("No data returned from Yahoo Finance")

    # Process each ticker
    rows_yielded = 0
    for ticker in tickers:
        # Handle multi-ticker vs single-ticker response format
        if len(tickers) > 1:
            ticker_data = data.xs(ticker, level=1, axis=1)
        else:
            ticker_data = data

        # Yield each row with OHLCV data
        for date_idx, row in ticker_data.iterrows():
            if pd.notna(row["Close"]) and pd.notna(row["Volume"]):
                yield {
                    "ticker": ticker,
                    "date": date_idx.date(),
                    "open_price": float(row["Open"]),
                    "high_price": float(row["High"]),
                    "low_price": float(row["Low"]),
                    "close_price": float(row["Close"]),
                    "volume": int(row["Volume"]),
                }
                rows_yielded += 1

    print(f"Yielded {rows_yielded} rows with OHLCV data")
```



```python
# Create and run the dlt extract pipeline
print("Running dlt extract pipeline...")
print("=" * 50)

# Create pipeline
extract_pipeline = dlt.pipeline(
    pipeline_name="financial_extract",
    destination=dlt.destinations.duckdb(DUCKDB_FILE),
    dataset_name="raw",
)

# Run the pipeline
load_info = extract_pipeline.run(
    yfinance_eod_prices(tickers=TICKERS, start_date=START_DATE, end_date=END_DATE)
)

print(f"\nLoad completed!")
print(f"  - Pipeline: {extract_pipeline.pipeline_name}")
print(f"  - Destination: {DUCKDB_FILE}")
print(f"  - Dataset/Schema: raw")
print(f"\nLoad info:")
print(load_info)
```

    Running dlt extract pipeline...
    ==================================================
    Downloading OHLCV data for 7 tickers...
    Yielded 10395 rows with OHLCV data
    
    Load completed!
      - Pipeline: financial_extract
      - Destination: ./financial_etl_dlt.duckdb
      - Dataset/Schema: raw
    
    Load info:
    Pipeline financial_extract load step completed in 0.53 seconds
    1 load package(s) were loaded to destination duckdb and into dataset raw
    The duckdb destination used duckdb:////home/francois/Workspace/posts/pipeline_duckdb/./financial_etl_dlt.duckdb location to store data
    Load package 1764251419.3941712 is LOADED and contains no failed jobs


```python
# Verify the extracted data in DuckDB
print("Verifying extracted OHLCV data in DuckDB...")
print("=" * 50)

with duckdb.connect(DUCKDB_FILE) as con:
    # Check what tables exist
    tables = con.execute(
        """
        SELECT table_schema, table_name 
        FROM information_schema.tables 
        WHERE table_schema = 'raw'
    """
    ).fetchall()
    print(f"Tables in 'raw' schema: {tables}")

    # Query the data with all OHLCV columns
    df_raw = con.execute(
        """
        SELECT ticker, date, open_price, high_price, low_price, close_price, volume
        FROM raw.eod_prices_raw
        ORDER BY ticker, date
        LIMIT 10
    """
    ).pl()

print("\nFirst 10 rows of extracted OHLCV data:")
df_raw
```

    Verifying extracted OHLCV data in DuckDB...
    ==================================================
    Tables in 'raw' schema: [('raw', 'eod_prices_raw'), ('raw', '_dlt_loads'), ('raw', '_dlt_pipeline_state'), ('raw', '_dlt_version')]
    
    First 10 rows of extracted OHLCV data:





<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<table border="1" class="dataframe"><thead><tr><th>ticker</th><th>date</th><th>open_price</th><th>high_price</th><th>low_price</th><th>close_price</th><th>volume</th></tr><tr><td>str</td><td>date</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>i64</td></tr></thead><tbody><tr><td>&quot;AAPL&quot;</td><td>2020-01-02</td><td>71.476615</td><td>72.528597</td><td>71.223274</td><td>72.468277</td><td>135480400</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-03</td><td>71.69616</td><td>72.523746</td><td>71.53933</td><td>71.763718</td><td>146322800</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-06</td><td>70.885487</td><td>72.374177</td><td>70.634554</td><td>72.335571</td><td>118387200</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-07</td><td>72.34522</td><td>72.600975</td><td>71.775804</td><td>71.995369</td><td>108872000</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-08</td><td>71.698574</td><td>73.455087</td><td>71.698574</td><td>73.153488</td><td>132079200</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-09</td><td>74.13066</td><td>74.900342</td><td>73.879735</td><td>74.707321</td><td>170108400</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-10</td><td>74.941371</td><td>75.440821</td><td>74.374363</td><td>74.876221</td><td>140644800</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-13</td><td>75.192313</td><td>76.502459</td><td>75.074081</td><td>76.475914</td><td>121532000</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-14</td><td>76.41317</td><td>76.623082</td><td>75.320175</td><td>75.443222</td><td>161954400</td></tr><tr><td>&quot;AAPL&quot;</td><td>2020-01-15</td><td>75.242981</td><td>76.12365</td><td>74.688034</td><td>75.119926</td><td>121923600</td></tr></tbody></table></div>



### dlt Metadata Tables

One nice aspect of dlt is that it automatically creates metadata tables to track loads:

| Table | Purpose |
|-------|--------|
| `_dlt_loads` | Tracks each pipeline run with timestamps and status |
| `_dlt_version` | Schema version information |

Each data row also gets:
- `_dlt_load_id`: Unique identifier for the load batch
- `_dlt_id`: Unique identifier for each row


```python
# Explore dlt metadata
print("dlt Load Metadata:")
print("=" * 50)

with duckdb.connect(DUCKDB_FILE) as con:
    # Show load history
    loads = con.execute(
        """
        SELECT load_id, schema_name, status, inserted_at
        FROM raw._dlt_loads
        ORDER BY inserted_at DESC
        LIMIT 5
    """
    ).pl()

print("Recent loads:")
loads
```

    dlt Load Metadata:
    ==================================================
    Recent loads:



<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<table border="1" class="dataframe"><thead><tr><th>load_id</th><th>schema_name</th><th>status</th><th>inserted_at</th></tr><tr><td>str</td><td>str</td><td>i64</td><td>datetime[μs, Europe/Paris]</td></tr></thead><tbody><tr><td>&quot;1764251419.3941712&quot;</td><td>&quot;financial_extract&quot;</td><td>0</td><td>2025-11-27 14:50:21.208674 CET</td></tr><tr><td>&quot;1764250753.3403895&quot;</td><td>&quot;financial_extract&quot;</td><td>0</td><td>2025-11-27 14:39:14.771508 CET</td></tr><tr><td>&quot;1764249716.0756714&quot;</td><td>&quot;financial_extract&quot;</td><td>0</td><td>2025-11-27 14:21:57.290876 CET</td></tr><tr><td>&quot;1764249510.5065134&quot;</td><td>&quot;financial_extract&quot;</td><td>0</td><td>2025-11-27 14:18:31.900797 CET</td></tr><tr><td>&quot;1764248385.3157668&quot;</td><td>&quot;financial_extract&quot;</td><td>0</td><td>2025-11-27 13:59:46.452635 CET</td></tr></tbody></table></div>


## SQLMesh Project Setup<a name="sqlmesh_project_setup"></a>

Now that we have raw data in DuckDB, we need to set up SQLMesh for the transformation step. Before running any transformations, you need to initialize a **SQLMesh project**. This is typically a one-time setup using the `sqlmesh init` command:

```bash
sqlmesh init duckdb
```

This creates a project directory with the following structure:

```
dlt_sqlmesh_project/
├── config.py              # Database connection & project settings
├── external_models.yaml   # Schema definitions for external tables (dlt tables)
├── models/                # SQL transformation models
│   └── marts/
│       └── stock_metrics.sql
├── seeds/                 # Static CSV data (optional)
├── audits/                # Custom audit definitions (optional)
├── macros/                # Reusable SQL macros (optional)
├── tests/                 # Unit tests (optional)
├── logs/                  # Execution logs
└── .cache/                # SQLMesh internal cache
```

### Key Configuration Files

Let's examine the three essential files that define our SQLMesh project.

#### 1. `config.py` - Database connection configuration

**FILE:** dlt_sqlmesh_project/config.py

```python
from sqlmesh.core.config import Config, DuckDBConnectionConfig, ModelDefaultsConfig

config = Config(
    gateways={
        "duckdb_local": {
            "connection": DuckDBConnectionConfig(database="../financial_etl_dlt.duckdb"),
            "state_connection": DuckDBConnectionConfig(database="../financial_etl_dlt.duckdb"),
        }
    },
    default_gateway="duckdb_local",
    model_defaults=ModelDefaultsConfig(dialect="duckdb"),
)
```

Key settings:
- connection: Points to the DuckDB file created by dlt
- state_connection: Where SQLMesh stores its metadata (same file)
- dialect: SQL dialect for parsing/generating queries

#### 2. `external_models.yaml` - Schema for tables NOT managed by SQLMesh (i.e., dlt tables)

**FILE:** dlt_sqlmesh_project/external_models.yaml

```yaml
- name: raw.eod_prices_raw
  description: Raw OHLCV stock prices loaded by dlt from Yahoo Finance
  columns:
    ticker: text
    date: date
    open_price: double
    high_price: double
    low_price: double
    close_price: double
    volume: bigint
    _dlt_load_id: text
    _dlt_id: text
```

Purpose:
- Tells SQLMesh about tables it doesn't manage (external sources)
- dlt creates `raw.eod_prices_raw` : SQLMesh needs to know its schema
- OHLCV = Open, High, Low, Close, Volume (standard financial data)
- Includes dlt metadata columns (_dlt_load_id, _dlt_id)
- Equivalent to `sources` in dbt

#### 3. `models/marts/stock_metrics.sql` - The transformation model with technical indicators

**FILE:** dlt_sqlmesh_project/models/marts/stock_metrics.sql

```sql
MODEL (
  name marts.stock_metrics,
  kind INCREMENTAL_BY_TIME_RANGE (time_column trade_date, batch_size 30),
  start '2020-01-01',
  cron '@daily',
  grain [ticker, trade_date],
  audits (...)
);

-- Technical indicators calculated using SQL window functions:

-- 1. SIMPLE MOVING AVERAGES (single column: close_price)
sma_20_day = AVG(close_price) OVER (... ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
sma_50_day = AVG(close_price) OVER (... ROWS BETWEEN 49 PRECEDING AND CURRENT ROW)

-- 2. BOLLINGER BANDS (single column: close_price)
bollinger_upper = SMA20 + 2 * STDDEV(close_price)
bollinger_lower = SMA20 - 2 * STDDEV(close_price)

-- 3. RSI - Relative Strength Index (single column: close_price)
rsi_14_day = 100 - (100 / (1 + avg_gain / avg_loss))

-- 4. MACD - Moving Average Convergence Divergence (single column: close_price)
macd_line = SMA_12_day - SMA_26_day

-- 5. ATR - Average True Range [MULTI-COLUMN: high, low, close]
true_range = GREATEST(
    high - low,                    -- Intraday range
    ABS(high - prev_close),        -- Gap up
    ABS(low - prev_close)          -- Gap down
)
atr_14_day = AVG(true_range) OVER 14 days

-- 6. DAILY METRICS [MULTI-COLUMN]
daily_return_pct  = (close - prev_close) / prev_close * 100
price_range_pct   = (high - low) / close * 100           -- Uses high, low, close
volume_ratio      = volume / AVG(volume) OVER 20 days
```

Key features:
- ATR uses 3 columns (high_price, low_price, close_price)
- price_range_pct also uses multiple columns
- WINDOW clauses for efficient computation
- CTEs for intermediate calculations (gains, losses, true_range)

## Transform: Run SQLMesh Pipeline<a name="transform"></a>

With the project configured, we can now run **SQLMesh** to transform the raw data. This is where we found SQLMesh particularly helpful: it handles the complexity of incremental processing and keeps track of what has already been computed.

The `sqlmesh plan --auto-apply` command will:
1. Detect the external table (`raw.eod_prices_raw`) created by dlt
2. Execute our transformation model (`marts.stock_metrics`)
3. Create the output table with technical indicators like SMA (Simple Moving Average, an average of prices over a rolling window, commonly used in financial analysis)
4. Run data quality audits automatically


```python
# Run SQLMesh to apply transformations
print("Running SQLMesh transformation plan...")
print("=" * 50)

try:
    # Run sqlmesh plan with auto-apply
    result = subprocess.run(
        ["sqlmesh", "plan", "--auto-apply"],
        cwd=SQLMESH_PROJECT_DIR,
        capture_output=True,
        text=True,
        check=True,
    )
    print(result.stdout)
    if result.stderr:
        # Filter out deprecation warnings
        stderr_lines = [
            l
            for l in result.stderr.split("\n")
            if "DeprecationWarning" not in l and l.strip()
        ]
        if stderr_lines:
            print("Info:")
            print("\n".join(stderr_lines))
    print("\nSQLMesh transformation completed successfully!")
except subprocess.CalledProcessError as e:
    print("SQLMesh execution failed!")
    print(f"stdout: {e.stdout}")
    print(f"stderr: {e.stderr}")
    raise
```

    Running SQLMesh transformation plan...
    ==================================================

    Summary of differences from `prod`:

    Metadata Updated:
    - marts.stock_metrics

      audits (
        NOT_NULL('columns' = (ticker, trade_date, close_price, volume)),
    -   UNIQUE_COMBINATION_OF_COLUMNS('columns' = (ticker, trade_date))
    +   UNIQUE_COMBINATION_OF_COLUMNS('columns' = (ticker, trade_date)),
    +   VALID_RSI_RANGE(),
    +   VALID_OHLC_PRICES(),
    +   POSITIVE_VOLUME(),
    +   VALID_ATR()
      ),

    Metadata Updated: marts.stock_metrics

    SKIP: No physical layer updates to perform

    [ 1/72] marts.stock_metrics   [insert 2020-01-01 - 2020-01-30, audits passed 6] 0.04s
    ...
    [72/72] marts.stock_metrics   [insert 2025-10-31 - 2025-11-26, audits passed 6] 0.03s

    Auditing models ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% • 72/72 • 0:00:02

    Model batches executed

    SKIP: No model batches to execute

    Updating virtual layer ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 100.0% • 1/1 • 0:00:00

    Virtual layer updated

    SQLMesh transformation completed successfully!


**A note on the output:** The `prod` environment represents the current "deployed" state. When we run the plan, it compares our model code against `prod` and shows a diff of what changed — similar to how Git works. In this case, we added new audits, so SQLMesh detected a metadata change but recognized that the underlying transformation logic was unchanged. That's why it skipped reprocessing the data (`No physical layer updates`) and only ran the new audits against the existing data.

One concept that helped us understand SQLMesh better is its two-layer architecture:

```
Physical Layer (Versioned Tables)          Virtual Layer (Views)
┌────────────────────────────────────┐    ┌─────────────────────────┐
│ sqlmesh__marts.marts__stock_metrics│    │ marts.stock_metrics     │
│        __[fingerprint]             │◄───│ (VIEW pointing to       │
└────────────────────────────────────┘    │  physical table)        │
                                          └─────────────────────────┘
```

Benefits:
- **Instant environment creation**: Just create views, no data copy
- **Instant promotion**: Swap view pointers, no recomputation
- **Easy rollbacks**: Point to previous version

## Verify: Query Transformed Data<a name="verify"></a>

Before loading the data to its final destination, it's worth verifying that the transformation worked as expected. Let's query the transformed data from DuckDB and look at some of the technical indicators.


```python
# Connect to DuckDB and query the transformed data with technical indicators
print("Querying transformed data with technical indicators...")
print("=" * 50)

with duckdb.connect(DUCKDB_FILE) as con:
    # Query the marts table - show key technical indicators
    query = """
    SELECT 
        ticker,
        trade_date,
        ROUND(close_price, 2) AS close,
        ROUND(sma_20_day, 2) AS sma20,
        ROUND(sma_50_day, 2) AS sma50,
        ROUND(rsi_14_day, 1) AS rsi,
        ROUND(macd_line, 2) AS macd,
        ROUND(atr_14_day, 2) AS atr,
        ROUND(bollinger_upper, 2) AS bb_upper,
        ROUND(bollinger_lower, 2) AS bb_lower
    FROM marts.stock_metrics
    WHERE ticker = 'AAPL'
    ORDER BY trade_date DESC
    LIMIT 15
    """
    df_transformed = con.execute(query).pl()

print("Sample AAPL data with technical indicators (most recent 15 rows):")
df_transformed
```

    Querying transformed data with technical indicators...
    ==================================================
    Sample AAPL data with technical indicators (most recent 15 rows):

<div><style>
.dataframe > thead > tr,
.dataframe > tbody > tr {
  text-align: right;
  white-space: pre-wrap;
}
</style>
<table border="1" class="dataframe"><thead><tr><th>ticker</th><th>trade_date</th><th>close</th><th>sma20</th><th>sma50</th><th>rsi</th><th>macd</th><th>atr</th><th>bb_upper</th><th>bb_lower</th></tr><tr><td>str</td><td>date</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td><td>f64</td></tr></thead><tbody><tr><td>&quot;AAPL&quot;</td><td>2025-11-26</td><td>277.55</td><td>271.13</td><td>271.13</td><td>63.0</td><td>1.02</td><td>5.95</td><td>277.91</td><td>264.34</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-25</td><td>276.97</td><td>270.77</td><td>270.77</td><td>61.6</td><td>0.7</td><td>6.14</td><td>276.98</td><td>264.56</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-24</td><td>275.92</td><td>270.41</td><td>270.41</td><td>60.3</td><td>0.33</td><td>6.11</td><td>275.95</td><td>264.86</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-21</td><td>271.49</td><td>270.06</td><td>270.06</td><td>55.1</td><td>0.14</td><td>5.95</td><td>274.98</td><td>265.14</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-20</td><td>266.25</td><td>269.97</td><td>269.97</td><td>41.4</td><td>0.1</td><td>5.73</td><td>275.0</td><td>264.94</td></tr><tr><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td><td>&hellip;</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-12</td><td>273.47</td><td>270.49</td><td>270.49</td><td>63.0</td><td>0.0</td><td>5.46</td><td>275.11</td><td>265.87</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-11</td><td>275.25</td><td>270.12</td><td>270.12</td><td>73.1</td><td>0.0</td><td>5.64</td><td>274.45</td><td>265.79</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-10</td><td>269.43</td><td>269.39</td><td>269.39</td><td>43.6</td><td>0.0</td><td>5.52</td><td>270.72</td><td>268.05</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-07</td><td>268.21</td><td>269.38</td><td>269.38</td><td>26.7</td><td>0.0</td><td>5.4</td><td>270.84</td><td>267.92</td></tr><tr><td>&quot;AAPL&quot;</td><td>2025-11-06</td><td>269.51</td><td>269.61</td><td>269.61</td><td>39.2</td><td>0.0</td><td>5.38</td><td>270.63</td><td>268.6</td></tr></tbody></table></div>

## Lineage<a name="lineage"></a>

We can also explore column lineage using SQLMesh Python API:

```python
def get_source_columns(node):
    """Recursively extract source column names from lineage node."""
    sources = []
    for downstream in node.downstream:
        if not downstream.downstream:
            # Leaf node - actual source column
            sources.append(downstream.name)
        else:
            sources.extend(get_source_columns(downstream))
    return sources


# Initialize SQLMesh context
ctx = Context(paths=[SQLMESH_PROJECT_DIR])

# Get the stock_metrics model
model = ctx.get_model("marts.stock_metrics")

print("Column Lineage for marts.stock_metrics")
print("=" * 60)

# Compute actual lineage using SQLGlot
print(f"\n{'Output Column':20} {'Type':10} {'Source Columns'}")
print("-" * 60)

for col_name, col_type in model.columns_to_types.items():
    result = lineage.lineage(col_name, model.query)
    sources = list(dict.fromkeys(get_source_columns(result)))  # dedupe, preserve order
    source_str = ", ".join(sources) if sources else "N/A"
    print(f"{col_name:20} {str(col_type):10} {source_str}")
```

    Column Lineage for marts.stock_metrics
    ============================================================

    Output Column        Type       Source Columns
    ------------------------------------------------------------
    ticker               TEXT       eod_prices_raw.ticker
    trade_date           DATE       eod_prices_raw.date
    open_price           DOUBLE     eod_prices_raw.open_price
    high_price           DOUBLE     eod_prices_raw.high_price
    low_price            DOUBLE     eod_prices_raw.low_price
    close_price          DOUBLE     eod_prices_raw.close_price
    volume               BIGINT     eod_prices_raw.volume
    sma_20_day           DOUBLE     eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker
    sma_50_day           DOUBLE     eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker
    bollinger_upper      DOUBLE     eod_prices_raw.close_price
    bollinger_lower      DOUBLE     eod_prices_raw.close_price
    rsi_14_day           DOUBLE     eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker
    macd_line            DOUBLE     eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker
    sma_9_day            DOUBLE     eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker
    atr_14_day           DOUBLE     eod_prices_raw.low_price, eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker, eod_prices_raw.high_price
    daily_return_pct     DOUBLE     eod_prices_raw.close_price, eod_prices_raw.date, eod_prices_raw.ticker
    price_range_pct      DOUBLE     eod_prices_raw.high_price, eod_prices_raw.close_price, eod_prices_raw.low_price
    volume_ratio         DOUBLE     eod_prices_raw.volume
    volume_sma_20        BIGINT     eod_prices_raw.volume


### SQLMesh Audits

One feature we found particularly useful is SQLMesh's audits : these are data quality checks that run automatically after model execution. SQLMesh supports two types:

**1. Built-in Audits** (defined inline in the MODEL):
```sql
audits (
  not_null(columns := (ticker, trade_date)),
  unique_combination_of_columns(columns := (ticker, trade_date))
)
```

**2. Custom Audits** (defined in `audits/` directory):
```
dlt_sqlmesh_project/
└── audits/
    ├── valid_rsi_range.sql      # RSI must be 0-100
    ├── valid_ohlc_prices.sql    # High >= Low, etc.
    ├── positive_volume.sql      # Volume > 0
    └── valid_atr.sql            # ATR >= 0
```

Custom audits return rows that **violate** the condition. If any rows are returned, the audit fails.

Example Custom Audit: valid_ohlc_prices.sql

```sql
AUDIT (
  name valid_ohlc_prices,
  dialect duckdb
);

/*
 * OHLC Price Sanity Check (Multi-Column Audit)
 * Validates relationships between Open, High, Low, Close prices:
 *   - High must be >= Low
 *   - High must be >= Open and Close
 *   - Low must be <= Open and Close
 * Returns rows that violate these constraints.
 */

SELECT
  ticker,
  trade_date,
  open_price,
  high_price,
  low_price,
  close_price,
  CASE
    WHEN high_price < low_price THEN 'high < low'
    WHEN high_price < open_price THEN 'high < open'
    WHEN high_price < close_price THEN 'high < close'
    WHEN low_price > open_price THEN 'low > open'
    WHEN low_price > close_price THEN 'low > close'
  END AS violation_reason
FROM @this_model
WHERE high_price < low_price
   OR high_price < open_price
   OR high_price < close_price
   OR low_price > open_price
   OR low_price > close_price

```

Note: `@this_model` is a macro that refers to the model being audited.

Let's run all the audits:

```python
# Run SQLMesh audits to validate data quality
print("Running data quality audits...")
print("=" * 50)

result = subprocess.run(
    ["sqlmesh", "audit"],
    cwd=SQLMESH_PROJECT_DIR,
    capture_output=True,
    text=True,
)

print(result.stdout)
if result.returncode == 0:
    print("\nAll audits passed!")
else:
    print("\nAudit failures detected:")
    print(result.stderr)
```

    Running data quality audits...
    ==================================================
    Found 6 audits.
    not_null on model marts.stock_metrics ✅ PASS.
    unique_combination_of_columns on model marts.stock_metrics ✅ PASS.
    valid_rsi_range on model marts.stock_metrics ✅ PASS.
    valid_ohlc_prices on model marts.stock_metrics ✅ PASS.
    positive_volume on model marts.stock_metrics ✅ PASS.
    valid_atr on model marts.stock_metrics ✅ PASS.
    
    Finished with 0 audit errors and 0 audits skipped.
    Done.
    
    
    All audits passed!


## Load: dlt to SQL Server<a name="load"></a>

With our data transformed and validated, we can move to the final **Load** phase. Here, we use dlt again, this time to export the transformed data from DuckDB to SQL Server. This demonstrates how dlt can work bidirectionally, not just for ingestion but also for exporting data to downstream systems.

### dlt sql_database Source

dlt provides a `sql_database` source that can read from any SQLAlchemy-compatible database, including DuckDB:

```python
from dlt.sources.sql_database import sql_database

source = sql_database(
    credentials="duckdb:///path/to/db.duckdb",
    schema="marts",
    table_names=["stock_metrics"]
)
```

### dlt mssql Destination

dlt supports SQL Server as a destination:

```python
pipeline = dlt.pipeline(
    destination=dlt.destinations.mssql(credentials={...})
)
```

Prerequisites:
- Microsoft ODBC Driver 17 or 18 for SQL Server
- Valid SQL Server credentials


```python
# Load credentials
def load_credentials(creds_file: str, conn_key: str) -> dict:
    """Load SQL Server credentials from JSON file."""
    with open(creds_file, "r") as f:
        creds = json.load(f)
    return creds[conn_key]["info"]


print("Loading SQL Server credentials...")
try:
    creds = load_credentials(CREDENTIALS_FILE, TARGET_CONN_KEY)
    print(f"  Server: {creds['server']}:{creds['port']}")
    print(f"  Database: {creds['database']}")
    print(f"  Target: {TARGET_SCHEMA}.{TARGET_TABLE}")
    creds_loaded = True
except Exception as e:
    print(f"Warning: Could not load credentials - {e}")
    print("\nSQL Server load will be skipped.")
    creds_loaded = False
```

    Loading SQL Server credentials...
      Server: localhost:1433
      Database: FINANCIAL_MART
      Target: dbo.Dim_Stock_Metrics



```python
# Create and run the dlt load pipeline (DuckDB -> SQL Server)
if creds_loaded:
    print("Running dlt load pipeline (DuckDB -> SQL Server)...")
    print("=" * 50)

    try:
        # Import the sql_database source
        from dlt.sources.sql_database import sql_database

        # Create source from DuckDB
        # Read from the marts.stock_metrics table
        duckdb_source = sql_database(
            credentials=f"duckdb:///{DUCKDB_FILE}",
            schema="marts",
            table_names=["stock_metrics"],
        )

        # Create pipeline to SQL Server
        # Note: We need to pass driver options for self-signed certificates
        load_pipeline = dlt.pipeline(
            pipeline_name="financial_load",
            destination=dlt.destinations.mssql(
                credentials={
                    "database": creds["database"],
                    "username": creds["username"],
                    "password": creds["password"],
                    "host": creds["server"],
                    "port": creds["port"],
                    "driver": "ODBC Driver 18 for SQL Server",
                    "query": {"TrustServerCertificate": "yes"},
                }
            ),
            dataset_name=TARGET_SCHEMA,
        )

        # Run the pipeline
        load_info = load_pipeline.run(duckdb_source, write_disposition="replace")

        print(f"\nLoad completed!")
        print(f"  - Source: {DUCKDB_FILE} (marts.stock_metrics)")
        print(f"  - Destination: SQL Server ({creds['database']})")
        print(f"\nLoad info:")
        print(load_info)

    except Exception as e:
        print(f"\nLoad failed: {e}")
        print("\nTroubleshooting tips:")
        print("  1. Verify SQL Server is running and accessible")
        print("  2. Check ODBC Driver 17/18 is installed")
        print("  3. Verify the target database exists")
else:
    print("\nSkipping SQL Server load (no valid credentials).")
    print("The transformed data is available in DuckDB at:")
    print(f"  {DUCKDB_FILE} -> marts.stock_metrics")
```

    Running dlt load pipeline (DuckDB -> SQL Server)...
    ==================================================


    /home/francois/miniconda3/envs/pipeline313/lib/python3.13/site-packages/duckdb_engine/__init__.py:184: DuckDBEngineWarning: duckdb-engine doesn't yet support reflection on indices
      warnings.warn(


    
    Load completed!
      - Source: ./financial_etl_dlt.duckdb (marts.stock_metrics)
      - Destination: SQL Server (FINANCIAL_MART)
    
    Load info:
    Pipeline financial_load load step completed in 6.88 seconds
    1 load package(s) were loaded to destination mssql and into dataset dbo
    The mssql destination used mssql://migadmin:***@localhost:1433/financial_mart location to store data
    Load package 1764251428.6438282 is LOADED and contains no failed jobs


We can check that the data is in SQL Server:

```sql
SELECT TOP 10 * FROM dbo.stock_metrics ORDER BY trade_date DESC",
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
      <th>ticker</th>
      <th>trade_date</th>
      <th>close_price</th>
      <th>volume</th>
      <th>sma_50_day</th>
      <th>_dlt_load_id</th>
      <th>_dlt_id</th>
      <th>open_price</th>
      <th>high_price</th>
      <th>low_price</th>
      <th>...</th>
      <th>bollinger_upper</th>
      <th>bollinger_lower</th>
      <th>rsi_14_day</th>
      <th>macd_line</th>
      <th>sma_9_day</th>
      <th>atr_14_day</th>
      <th>daily_return_pct</th>
      <th>price_range_pct</th>
      <th>volume_ratio</th>
      <th>volume_sma_20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>TSLA</td>
      <td>2025-11-26</td>
      <td>426.579987</td>
      <td>63299400</td>
      <td>425.9326</td>
      <td>1764251428.6438282</td>
      <td>dp7lDd+JOoBe9g</td>
      <td>423.950012</td>
      <td>426.940002</td>
      <td>416.890015</td>
      <td>...</td>
      <td>473.2135</td>
      <td>378.6517</td>
      <td>43.15</td>
      <td>-14.1993</td>
      <td>407.6211</td>
      <td>21.3321</td>
      <td>1.7120</td>
      <td>2.3559</td>
      <td>0.72</td>
      <td>88165984</td>
    </tr>
    <tr>
      <th>1</th>
      <td>GOOGL</td>
      <td>2025-11-26</td>
      <td>319.950012</td>
      <td>51290800</td>
      <td>290.8753</td>
      <td>1764251428.6438282</td>
      <td>DfvXqKHwDRHjuQ</td>
      <td>320.679993</td>
      <td>324.500000</td>
      <td>316.790009</td>
      <td>...</td>
      <td>319.7867</td>
      <td>261.9638</td>
      <td>69.13</td>
      <td>4.6406</td>
      <td>298.8444</td>
      <td>12.1707</td>
      <td>-1.0790</td>
      <td>2.4097</td>
      <td>1.12</td>
      <td>45707632</td>
    </tr>
    <tr>
      <th>2</th>
      <td>META</td>
      <td>2025-11-26</td>
      <td>633.609985</td>
      <td>15186100</td>
      <td>617.5516</td>
      <td>1764251428.6438282</td>
      <td>j/1Q7cXe0XMQ9Q</td>
      <td>637.690002</td>
      <td>638.359985</td>
      <td>631.630005</td>
      <td>...</td>
      <td>652.9907</td>
      <td>582.1125</td>
      <td>56.86</td>
      <td>-8.2399</td>
      <td>607.3067</td>
      <td>17.8543</td>
      <td>-0.4102</td>
      <td>1.0622</td>
      <td>0.62</td>
      <td>24326500</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MSFT</td>
      <td>2025-11-26</td>
      <td>485.500000</td>
      <td>25697600</td>
      <td>497.4609</td>
      <td>1764251428.6438282</td>
      <td>w+FErTh2jGGz1g</td>
      <td>486.309998</td>
      <td>488.309998</td>
      <td>481.200012</td>
      <td>...</td>
      <td>526.8620</td>
      <td>468.0599</td>
      <td>43.31</td>
      <td>-5.6150</td>
      <td>486.8763</td>
      <td>11.8285</td>
      <td>1.7841</td>
      <td>1.4645</td>
      <td>0.98</td>
      <td>26267837</td>
    </tr>
    <tr>
      <th>4</th>
      <td>AMZN</td>
      <td>2025-11-26</td>
      <td>229.160004</td>
      <td>38435400</td>
      <td>236.8532</td>
      <td>1764251428.6438282</td>
      <td>idQBkg7Rc/3mEw</td>
      <td>230.740005</td>
      <td>231.750000</td>
      <td>228.770004</td>
      <td>...</td>
      <td>260.0106</td>
      <td>213.6957</td>
      <td>36.48</td>
      <td>-6.3015</td>
      <td>226.1933</td>
      <td>6.5457</td>
      <td>-0.2221</td>
      <td>1.3004</td>
      <td>0.70</td>
      <td>55171516</td>
    </tr>
    <tr>
      <th>5</th>
      <td>NVDA</td>
      <td>2025-11-26</td>
      <td>180.259995</td>
      <td>183181100</td>
      <td>189.3247</td>
      <td>1764251428.6438282</td>
      <td>hNEoZkDqEOIfQA</td>
      <td>181.630005</td>
      <td>182.910004</td>
      <td>178.240005</td>
      <td>...</td>
      <td>206.1015</td>
      <td>172.5480</td>
      <td>43.50</td>
      <td>-4.4397</td>
      <td>182.7556</td>
      <td>9.0200</td>
      <td>1.3722</td>
      <td>2.5907</td>
      <td>0.83</td>
      <td>221997200</td>
    </tr>
    <tr>
      <th>6</th>
      <td>AAPL</td>
      <td>2025-11-26</td>
      <td>277.549988</td>
      <td>33413600</td>
      <td>271.1275</td>
      <td>1764251428.6438282</td>
      <td>KXA3vllpset5WQ</td>
      <td>276.959991</td>
      <td>279.529999</td>
      <td>276.630005</td>
      <td>...</td>
      <td>277.9141</td>
      <td>264.3409</td>
      <td>63.02</td>
      <td>1.0158</td>
      <td>271.5611</td>
      <td>5.9532</td>
      <td>0.2094</td>
      <td>1.0449</td>
      <td>0.67</td>
      <td>49662732</td>
    </tr>
    <tr>
      <th>7</th>
      <td>TSLA</td>
      <td>2025-11-25</td>
      <td>419.399994</td>
      <td>71915600</td>
      <td>425.8967</td>
      <td>1764251428.6438282</td>
      <td>h4GeCQBQU+CrHg</td>
      <td>414.420013</td>
      <td>420.480011</td>
      <td>405.950012</td>
      <td>...</td>
      <td>474.5472</td>
      <td>377.2461</td>
      <td>35.78</td>
      <td>-12.6092</td>
      <td>404.8889</td>
      <td>22.9257</td>
      <td>0.3878</td>
      <td>3.4645</td>
      <td>0.80</td>
      <td>89547461</td>
    </tr>
    <tr>
      <th>8</th>
      <td>GOOGL</td>
      <td>2025-11-25</td>
      <td>323.440002</td>
      <td>88632100</td>
      <td>289.2600</td>
      <td>1764251428.6438282</td>
      <td>frSIcFsmz+UMgw</td>
      <td>326.209991</td>
      <td>328.829987</td>
      <td>317.649994</td>
      <td>...</td>
      <td>315.2424</td>
      <td>263.2776</td>
      <td>71.99</td>
      <td>3.7683</td>
      <td>294.2467</td>
      <td>12.1350</td>
      <td>1.5255</td>
      <td>3.4566</td>
      <td>1.95</td>
      <td>45397456</td>
    </tr>
    <tr>
      <th>9</th>
      <td>META</td>
      <td>2025-11-25</td>
      <td>636.219971</td>
      <td>25213000</td>
      <td>616.6595</td>
      <td>1764251428.6438282</td>
      <td>0bqKg0SIKNKX/g</td>
      <td>624.000000</td>
      <td>637.049988</td>
      <td>618.299988</td>
      <td>...</td>
      <td>652.2370</td>
      <td>581.0819</td>
      <td>50.11</td>
      <td>-7.5019</td>
      <td>604.6711</td>
      <td>18.6593</td>
      <td>3.7795</td>
      <td>2.9471</td>
      <td>1.02</td>
      <td>24834300</td>
    </tr>
  </tbody>
</table>
<p>10 rows × 21 columns</p>
</div>

## Production Deployment: Logging & Return Codes<a name="production_deployment"></a>

The pipeline above works well for development, but deploying to production with a scheduler (cron, Airflow, Prefect, etc.) requires a few additional considerations. We found that proper logging and return codes are essential for monitoring and debugging issues.

### Why This Matters

| Aspect | Development | Production |
|--------|-------------|------------|
| **Output** | `print()` statements | Structured log files |
| **Exit behavior** | Exceptions crash the script | Return codes signal status |
| **Monitoring** | Manual inspection | Automated alerts on failures |

### Logging Setup

Replace `print()` with Python's `logging` module for production, or [loguru](https://loguru.readthedocs.io/en/stable/):

```python
import logging
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Usage
logger.info("Starting ETL pipeline...")
logger.warning("Missing data for ticker: XYZ")
logger.error("Failed to connect to SQL Server")
```

**Log Levels:**
- `DEBUG`: Detailed diagnostic info (row counts, query times)
- `INFO`: General progress updates ("Loaded 2012 rows")
- `WARNING`: Non-critical issues ("Retrying connection...")
- `ERROR`: Failures that stop the pipeline
- `CRITICAL`: System-level failures

### Return Codes for Schedulers

Schedulers use exit codes to determine success/failure. Use `sys.exit()` with meaningful codes:

```python
import sys

# Define exit codes
EXIT_SUCCESS = 0
EXIT_GENERAL_ERROR = 1
EXIT_CONFIG_ERROR = 2
EXIT_DATA_QUALITY_ERROR = 3
EXIT_CONNECTION_ERROR = 4

def main():
    try:
        # Load configuration
        if not os.path.exists('credentials.json'):
            logger.error("Missing credentials.json")
            sys.exit(EXIT_CONFIG_ERROR)
        
        # Extract
        logger.info("Extracting data...")
        df = extract_data()
        
        # Validate
        if len(df) == 0:
            logger.error("No data extracted - aborting")
            sys.exit(EXIT_DATA_QUALITY_ERROR)
        
        # Transform
        logger.info("Running SQLMesh transformations...")
        run_sqlmesh()
        
        # Load
        logger.info("Loading to SQL Server...")
        load_to_sql_server()
        
        logger.info("Pipeline completed successfully")
        sys.exit(EXIT_SUCCESS)
        
    except ConnectionError as e:
        logger.error(f"Connection failed: {e}")
        sys.exit(EXIT_CONNECTION_ERROR)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(EXIT_GENERAL_ERROR)

if __name__ == "__main__":
    main()
```

### Sample Scheduler Configuration (cron)

```bash
# Run daily at 6 AM, log output, alert on failure
0 6 * * * /usr/bin/python3 /path/to/run_pipeline.py >> /var/log/etl.log 2>&1 || mail -s "ETL Failed" alerts@company.com
```

### Sample Log Output

```
2024-01-15 06:00:01 - INFO - Starting ETL pipeline...
2024-01-15 06:00:02 - INFO - Extracting data from yfinance...
2024-01-15 06:00:05 - INFO - Extracted 2012 rows for ['AAPL', 'MSFT']
2024-01-15 06:00:06 - INFO - Running SQLMesh transformations...
2024-01-15 06:00:08 - INFO - SQLMesh plan applied successfully
2024-01-15 06:00:09 - INFO - Loading to SQL Server...
2024-01-15 06:00:12 - INFO - Loaded 2012 rows to dbo.Dim_Stock_Metrics
2024-01-15 06:00:12 - INFO - Pipeline completed successfully
```

## Further Engineering Considerations<a name="further_engineering"></a>

There are a few more topics worth considering as you move toward production. These go beyond the scope of this example, but we wanted to mention them briefly.

### 1. Error Handling and Observability

While dlt and SQLMesh provide solid foundations, production-grade pipelines typically need more comprehensive error handling and observability.

-   **Centralized Logging**: Beyond basic file logging, integrate with centralized logging systems for easier searching, aggregation, and analysis of logs across multiple pipeline runs.
-   **Alerting**: Configure alerts based on log errors, audit failures, or unexpected data volumes.
-   **Monitoring Dashboards**: Build dashboards to visualize pipeline health, execution times, data volumes, and data quality metrics over time. This helps in proactive identification of issues and performance bottlenecks.
-   **Idempotency and Retries**: Design pipelines to be idempotent where possible, allowing safe retries without duplicating data or side effects. `dlt`'s `write_disposition` and `SQLMesh`'s transactional updates aid this. Implement retry mechanisms with exponential backoff for transient errors (e.g., network issues, API rate limits).

### 2. Schema Evolution Strategy

dlt's automatic schema evolution is a helpful feature, but managing it thoughtfully becomes important when transformations and downstream consumers are involved.

-   **Graceful Changes**: While `dlt` handles schema additions, consider how schema changes (e.g., column renaming, type changes) in the raw layer propagate. `SQLMesh`'s column-level lineage can help identify affected downstream models.
-   **Versioned External Models**: For significant schema changes in external sources, `SQLMesh` allows versioning of external model definitions. This lets you gracefully transition dependents without breaking existing queries.
-   **Impact Analysis**: Before deploying a schema change, use `SQLMesh`'s `plan` command to visualize the impact on dependent models. This helps prevent unexpected issues in the transformation layer.

### 3. Cost/Resource Management

While DuckDB is efficient for local development and moderate data volumes, larger-scale deployments may require thinking about cost and resource tradeoffs.

-   **Compute vs. Storage**: For cloud environments, understand the trade-offs between compute and storage costs. DuckDB is compute-bound locally; in a cloud data warehouse, query complexity and data scanned directly impact costs.
-   **Incremental Processing**: `SQLMesh`'s incremental models are critical for cost optimization. By only processing new or changed data, you significantly reduce compute resources and execution time compared to full table rebuilds.
-   **Resource Allocation**: Fine-tune resource allocation (CPU, memory) for pipeline execution environments, especially when running on orchestrators like Airflow.
-   **Cloud-Native Alternatives**: If the DuckDB file grows excessively large or requires distributed processing, consider migrating to cloud-native data warehouses or data lake solutions that offer scalable compute and storage.

### 4. Testability

Beyond SQLMesh audits, it is recommended to consider a broader testing strategy for production pipelines:

-   **Unit Tests for dlt Resources**: Write Python unit tests for your custom `dlt` resources (`yfinance_eod_prices` in this example). Mock external dependencies (like the `yfinance` API) to ensure the resource logic works correctly under various conditions.
-   **SQLMesh Unit Tests**: `SQLMesh supports SQL unit tests for models. These tests define expected outputs for specific input data, ensuring transformation logic is correct and remains so after changes.
-   **Integration Tests**: Test the full pipeline flow (Extract -> Transform -> Load) in a controlled environment with representative data. This ensures all components work together seamlessly.
-   **Virtual Data Environments (VDEs) for Development**: Leverage `SQLMesh` VDEs to create isolated environments for feature development. This allows developers to test changes to models without impacting production data or other developers' work. Changes can be validated in a VDE before merging to a shared environment.

By considering these aspects, you can evolve a pipeline like this from a working example into something more suitable for production use.

## References<a name="references"></a>

- [dlt Documentation](https://dlthub.com/docs/intro)
- [SQLMesh Documentation](https://sqlmesh.readthedocs.io/)
- [SQLMesh + dlt Integration](https://sqlmesh.readthedocs.io/en/stable/integrations/dlt/)
- [dlt MSSQL Destination](https://dlthub.com/docs/dlt-ecosystem/destinations/mssql)
- [DuckDB Documentation](https://duckdb.org/docs/)

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