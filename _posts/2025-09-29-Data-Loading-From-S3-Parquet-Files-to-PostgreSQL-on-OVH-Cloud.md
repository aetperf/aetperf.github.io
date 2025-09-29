---
title: Data Loading from S3 Parquet Files to PostgreSQL on OVH Cloud
layout: post
comments: true
author: François Pacull
categories: [database, performance]
tags:
- FastTransfer
- S3 Parquet files
- PostgreSQL bulk loading
- OVH Object Storage
- DuckDB
- Parallel data ingestion
- Cloud database performance
---

## Introduction

Loading data efficiently from cloud storage into databases remains a critical challenge for data engineers. [FastTransfer](https://aetperf.github.io/FastTransfer-Documentation/) addresses this challenge through parallel data loading capabilities and flexible configuration options. This post demonstrates FastTransfer's performance when loading Parquet files from OVH's S3-compatible object storage into PostgreSQL on OVH public cloud infrastructure.

## The Setup: Cloud Infrastructure That Makes Sense

For our testing environment, we deployed PostgreSQL on an OVH c3-64 instance in their Gravelines datacenter. Here's what we're working with:

### Software Versions
- **FastTransfer**: Version 0.13.10.0 (X64 architecture, .NET 8.0.19)
- **Operating System**: Ubuntu 24.04.3 LTS
- **Source Engine**: DuckDB v1.3.2 (for Parquet reading and streaming)
- **Target Database**: PostgreSQL 16.10

These versions represent current stable releases, with PostgreSQL 16 bringing improved parallel query performance and DuckDB offering exceptional Parquet reading capabilities.

### Hardware Configuration
- **Compute**: 32 vCores @ 2.3 GHz with 64 GB RAM
- **Storage**: 400 GB local NVMe for the OS, plus a 1TB high-speed Gen2 block storage volume where PostgreSQL's data directory resides
- **Network**: 4 Gbps bandwidth - which we'll see becomes a limiting factor at higher parallelism
- **Location**: Gravelines (GRA11) datacenter

Note that the PostgreSQL data directory is specifically configured on the 1TB block storage volume, not the local NVMe. While marketed as "high-speed," this virtual block storage attached to the instance typically performs slower than local SSDs but offers advantages in flexibility, snapshots, and volume management.

This configuration represents a practical mid-range setup - not the smallest instance that would struggle with parallel workloads, nor an oversized machine that would mask performance characteristics. In my experience, this sweet spot helps identify real bottlenecks rather than just throwing hardware at the problem.

### The Data: TPC-H Orders Table

We're using the TPC-H benchmark's orders table at scale factor 10, which gives us:
- 16 Parquet files, evenly distributed at 29.2 MiB each
- Total dataset size: 467.8 MiB
- 15 million rows with mixed data types (integers, decimals, dates, and varchar)

The data resides in an OVH S3-compatible object storage bucket, and each file contains roughly 937,500 rows. This distribution allows us to test parallel loading strategies effectively - there's enough data to see performance differences, but not so much that individual test runs become impractical.

## FastTransfer in Action: The Command That Does the Heavy Lifting

Here's the actual command we use to load data:

```bash
./FastTransfer \
  --sourceconnectiontype "duckdbstream" \
  --sourceserver ":memory:" \
  --query "SELECT * exclude filename from read_parquet('s3://arpeiofastbcp/tpch/sf10/orders/*.parquet', filename=true) t" \
  --targetconnectiontype "pgcopy" \
  --targetserver "localhost:5432" \
  --targetuser "fasttransfer" \
  --targetpassword "********" \
  --targetdatabase "tpch" \
  --targetschema "tpch_10_test" \
  --targettable "orders" \
  --method "DataDriven" \
  --distributeKeyColumn "filename" \
  --datadrivenquery "select file from glob('s3://arpeiofastbcp/tpch/sf10/orders/*.parquet')" \
  --loadmode "Truncate" \
  --mapmethod "Name" \
  --batchsize 10000 \
  --degree 16
```

Let's break down the key components and understand what each parameter does:

### Source Configuration
- **`--sourceconnectiontype "duckdbstream"`**: Uses DuckDB's memory-efficient streaming connection, optimized for large datasets and better memory management compared to the standard `duckdb` connection type
- **`--sourceserver ":memory:"`**: Runs DuckDB in-memory mode for temporary data processing without persisting to disk
- **`--query`**: The DuckDB SQL that leverages the `read_parquet()` function to directly access Parquet files from S3, with `filename=true` to capture file origins for distribution

### Target Configuration
- **`--targetconnectiontype "pgcopy"`**: Uses PostgreSQL's native COPY protocol, the fastest method for bulk loading data into PostgreSQL
- **`--targetserver "localhost:5432"`**: Standard PostgreSQL connection details
- **`--targetuser` and `--targetpassword`**: Database authentication credentials

### Parallelization Strategy
- **`--method "DataDriven"`**: Distributes work based on distinct values in a specified column - in our case, each worker processes specific files
- **`--distributeKeyColumn "filename"`**: Uses the filename column to assign work to workers, ensuring each file is processed by exactly one worker
- **`--datadrivenquery`**: Overrides the default distinct value selection with an explicit file list using `glob()`, giving us precise control over work distribution
- **`--degree 16`**: Creates 16 parallel workers. FastTransfer supports 1-1024 workers, or negative values for CPU-adaptive scaling (e.g., `-2` uses half available CPUs)

### Loading Configuration
- **`--loadmode "Truncate"`**: Clears the target table before loading, ensuring a clean slate (alternative is `"Append"` for adding to existing data)
- **`--mapmethod "Name"`**: Maps source to target columns by name rather than position, providing flexibility when column orders differ
- **`--batchsize 10000`**: Processes 10,000 rows per bulk copy operation (default is 1,048,576). Smaller batches can reduce memory usage but may impact throughput

### About FastTransfer

FastTransfer is designed specifically for efficient data movement between different database systems, particularly excelling with large datasets (>1 million cells). The tool requires the target table to pre-exist and supports various database types including ClickHouse, MySQL, Oracle, PostgreSQL, and SQL Server. Its strength lies in intelligent work distribution - whether using file-based distribution like our DataDriven approach, or other methods like CTID (PostgreSQL-specific), RangeId (numeric ranges), or Random (modulo-based distribution).

## Performance Analysis: Where Theory Meets Reality

We tested three different table configurations to understand how PostgreSQL constraints affect loading performance. Each test was run four times, and we report the best result to minimize noise from network variability or system background tasks.

### Configuration 1: Table with Primary Key (LOGGED)

Starting with a standard production table - a primary key on `o_orderkey` with full durability (WAL logging enabled):

| Degree of Parallelism | Load Time (seconds) | Speedup |
|----------------------|---------------------|---------|
| 1                    | 52.9                | 1.0x    |
| 2                    | 29.9                | 1.8x    |
| 4                    | 19.7                | 2.7x    |
| 8                    | 18.9                | 2.8x    |
| 16                   | 24.7                | 2.1x    |

Performance peaks at 8 parallel workers, achieving a 2.8x speedup. Beyond this point, we hit diminishing returns due to PostgreSQL's constraint checking overhead combined with WAL logging contention. With multiple workers trying to update the primary key index and write to the WAL simultaneously, lock contention becomes the bottleneck rather than network throughput.

### Configuration 2: Table without Primary Key (LOGGED)

Removing the primary key constraint while keeping WAL logging enabled shows what happens when we eliminate index maintenance overhead but retain durability:

| Degree of Parallelism | Load Time (seconds) | Speedup |
|----------------------|---------------------|---------|
| 1                    | 42.0                | 1.0x    |
| 2                    | 23.0                | 1.8x    |
| 4                    | 13.0                | 3.2x    |
| 8                    | 11.1                | 3.8x    |
| 16                   | 12.4                | 3.4x    |

The sweet spot remains at 8 workers, but now we achieve 3.8x speedup and reduce the absolute load time to just 11.1 seconds - a 41% improvement over the primary key configuration. The plateau beyond 8 workers is due to WAL logging contention - even without constraint checking, parallel workers still compete for sequential WAL writes, limiting further scaling.

### Configuration 3: UNLOGGED Table without Primary Key

For scenarios where we can rebuild from source if needed (think staging environments or temporary transformations), UNLOGGED tables without constraints offer compelling performance by completely bypassing WAL writes:

| Degree of Parallelism | Load Time (seconds) | Speedup |
|----------------------|---------------------|---------|
| 1                    | 42.5                | 1.0x    |
| 2                    | 23.1                | 1.8x    |
| 4                    | 12.2                | 3.5x    |
| 8                    | 7.4                 | 5.7x    |
| 16                   | 5.5                 | 7.7x    |

This configuration scales remarkably well, achieving 7.7x speedup at 16 workers. With both constraint checking and WAL logging eliminated, PostgreSQL's write path is no longer the bottleneck. Instead, we finally hit the network bandwidth limit of our 4 Gbps connection. This explains the continued scaling to 16 workers - we're now purely limited by network throughput rather than database overhead.

## Visual Performance Comparison

<img src="/img/2025-09-29_02/transfer_s3_to_postgres_comparison.jpg" alt="Performance Comparison." width="900">

The bar chart clearly shows how constraints and logging impact loading performance. Notice how the gap widens as parallelism increases - at degree 16, UNLOGGED tables load 4.5x faster than tables with primary keys. The performance characteristics reveal different bottlenecks: tables with primary keys hit a wall around 2.8x speedup due to constraint checking and WAL contention, tables without primary keys achieve better scaling but plateau around 4x due to WAL logging overhead, while UNLOGGED tables show nearly linear scaling up to 16 workers where they finally hit the network bandwidth limit.

## Practical Insights and Trade-offs

### When to Use Each Configuration

**Tables with Primary Keys**
- Production tables requiring data integrity
- When duplicate prevention is critical
- Accept the performance trade-off for data consistency

**Tables without Primary Keys**
- Initial staging tables where you'll add constraints after loading
- Append-only log tables where duplicates are handled elsewhere
- When you can guarantee uniqueness at the source

**UNLOGGED Tables**
- Staging environments where data can be reloaded if lost
- Temporary transformation tables
- Development and testing environments

Consider that UNLOGGED tables provide exceptional performance but lose all data if PostgreSQL crashes. In my experience, they work brilliantly for ETL pipelines where the source data remains available.

### Network and I/O Considerations

With our 4 Gbps network connection, theoretical maximum throughput is about 500 MB/s. However, the bottleneck shifts depending on PostgreSQL configuration:

**Configurations 1 & 2 (LOGGED tables):**
These are limited by PostgreSQL's internal overhead, not network bandwidth:
- **Configuration 1 (WITH PK)**: Constraint checking + WAL contention limits scaling
- **Configuration 2 (WITHOUT PK)**: WAL logging contention limits scaling beyond 8 workers

**Configuration 3 (UNLOGGED):**
Only here do we hit network bandwidth limits:
- **At degree=16 UNLOGGED**: 467.8 MiB in 5.5 seconds = ~85 MB/s per worker × 16 workers = ~1360 MB/s theoretical demand
- Since 4 Gbps = 500 MB/s theoretical maximum, we're clearly network-bound in this configuration

This explains why UNLOGGED tables continue scaling to 16 workers while LOGGED configurations plateau at 8 workers. The proximity of compute and storage within OVH's Gravelines datacenter ensures minimal latency for object storage access, but bandwidth limitations only become apparent when PostgreSQL's internal bottlenecks are removed.

### Optimal Parallelism Settings

Our testing reveals that optimal parallelism depends on your constraints:
- **With constraints**: Use degree 4-8 (more workers just add contention)
- **Without constraints**: Use degree 8-16 depending on your hardware
- **For UNLOGGED tables**: Scale up to 16 or even higher on larger machines

One approach is to start with degree 8 as a baseline and adjust based on your specific workload and hardware.

## Key Takeaways

1. **Different bottlenecks emerge at each configuration level** - Primary keys + WAL logging create the most contention, WAL logging alone limits scaling, and only UNLOGGED tables reveal network bandwidth limits

2. **UNLOGGED tables offer exceptional performance for appropriate use cases** - If you can rebuild from source, the 7.7x speedup is compelling and finally utilizes available network bandwidth

3. **PostgreSQL's internal overhead often masks infrastructure limits** - Network bandwidth only becomes the bottleneck when you eliminate database-level contention

4. **FastTransfer's DataDriven distribution effectively prevents worker contention** - Each worker processes distinct files, ensuring the bottleneck is PostgreSQL internals, not work distribution

5. **Cloud infrastructure can deliver solid performance** - Our mid-range OVH instance handles parallel loads effectively, with network bandwidth being the ultimate constraint at maximum performance

## Conclusion

FastTransfer achieves sub-6-second load times for 467.8 MiB of Parquet data from OVH S3 to PostgreSQL, reaching 870 MB/s throughput with optimal parallelization. The DataDriven distribution method effectively prevents worker contention while PostgreSQL's UNLOGGED tables provide maximum performance. These results demonstrate that cloud storage to database pipelines can achieve enterprise-grade performance with the right tooling.

---

## About FastTransfer

FastTransfer is a commercial high-performance data migration tool developed by [arpe.io](https://arpe.io). It provides parallel data transfer capabilities across multiple database platforms including PostgreSQL, MySQL, Oracle, SQL Server, ClickHouse, and DuckDB.

**Key Features:**
- Advanced parallelization strategies for optimal performance
- Cross-platform compatibility with major databases
- Flexible configuration for various data migration scenarios
- Production-ready with comprehensive logging and monitoring

For licensing information, support options, and to request a trial, visit the [official documentation](https://aetperf.github.io/FastTransfer-Documentation/).