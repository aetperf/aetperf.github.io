---
title: High-Speed PostgreSQL Replication on OVH with FastTransfer WIP
layout: post
comments: true
author: François Pacull
categories: [database, performance]
tags:
- FastTransfer
- PostgreSQL replication
- OVH 
- High-performance
- Database migration speed
- TPC-H benchmark
- 20 Gbps network
- c3-256
- PostgreSQL parallel transfer
---


## Introduction

PostgreSQL-to-PostgreSQL replication at scale requires tools that can fully leverage modern cloud infrastructure and network capabilities. [FastTransfer](https://aetperf.github.io/FastTransfer-Documentation/) is a commercial data migration tool designed to maximize throughput through advanced parallelization. This post demonstrates FastTransfer's performance transferring 113GB of TPC-H data between OVH c3-256 instances over a 20 Gbit/s network.

## Infrastructure Setup

- **OVH Instances**: c3-256 (256GB RAM, 128 vCores @2.3GHz, 400GB NVMe)
- **Network**: 20 Gbit/s vrack, Paris datacenter (eu-west-par-c)
- **OS**: Ubuntu 24
- **PostgreSQL**: Version 16
- **Dataset**: TPC-H SF100 lineitem table (~600M rows, ~113GB)

## PostgreSQL Configuration

Optimized for bulk operations: 80GB shared_buffers, 128 parallel workers, minimal WAL logging. Target tables: UNLOGGED, no primary keys.

## Target Database Disk Performance

The target PostgreSQL instance uses the native 400GB NVMe instance disk (not block storage) for database storage. This provides excellent I/O performance crucial for high-speed data ingestion:

### FIO Benchmark Command
```bash
fio --name=seqwrite --filename=/tmp/fio-test --rw=write \
    --bs=1M --size=8G --direct=1 --numjobs=1 --runtime=30 --group_reporting
```

### Results
```
Sequential Write Performance (8GB test, 1MB blocks):
- Throughput: 1,260 MB/s (1.26 GB/s)
- IOPS: 1,259
- Average latency: 787 microseconds
- 95th percentile: 1.5ms
- 99th percentile: 2.3ms
```

The native NVMe storage delivers consistent low-latency writes with over 1.2 GB/s throughput, ensuring disk I/O is not a bottleneck for the PostgreSQL COPY operations even at peak network transfer rates.

## Network Performance

The private network connection between source and target instances was tested using iperf3 to verify bandwidth capacity:

### iperf3 Benchmark Command
```bash
# On target instance
iperf3 -s

# On source instance
iperf3 -c 10.10.0.50 -P 64 -t 30
```

### Results
```
Network Throughput Test (64 parallel streams, 30 seconds):
- Average throughput: 20.5 Gbit/s
- Total data transferred: 71.7 GB
- Consistent performance across all streams
```

The network delivers full line-rate performance, slightly exceeding the nominal 20 Gbit/s specification. With 64 parallel TCP streams, the network provides ample bandwidth for FastTransfer's parallel data transfer operations.

## FastTransfer Command

FastTransfer version: 0.13.12

```bash
./FastTransfer \
  --sourceconnectiontype "pgcopy" \
  --sourceconnectstring "Host=localhost;Port=5432;Database=tpch;Trust Server Certificate=True;Application Name=FastTransfer;Maximum Pool Size=150;Timeout=15;Command Timeout=10800;Username=fasttransfer;Password=******" \
  --sourceschema "tpch_100" --sourcetable "lineitem" \
  --targetconnectiontype "pgcopy" \
  --targetconnectstring "Host=10.10.0.50;Port=5432;Database=tpch;Trust Server Certificate=True;Application Name=FastTransfer;Maximum Pool Size=150;Timeout=15;Command Timeout=10800;Username=fasttransfer;Password=******" \
  --targetschema "tpch_100" --targettable "lineitem" \
  --loadmode "Truncate" --method "Ctid" --degree 128
```

Note the `Maximum Pool Size`=150 in the connection string, increased from the default 100 to support 128 parallel threads.

## Performance Results

### Transfer Time

<img src="/img/2025-09-29_03/lineitem_elapsed_time.jpg" alt="Transfer Time." width="900">

Transfer time: 749s (single thread) → 70s (128 threads)

### Throughput Scaling

<img src="/img/2025-09-29_03/lineitem_throughput.jpg" alt="Throughput." width="900">

Throughput: 145 MB/s → 1,880 MB/s (75% of 20 Gbit/s link capacity)


## Results Summary

- **113GB transferred in 70 seconds** (degree=128)
- **1.88 GB/s peak throughput** achieved
- **10.7x speedup** with 128 parallel connections
- **Optimal range**: 32-64 threads for best efficiency/performance balance

## Conclusion

FastTransfer achieves 1.88 GB/s throughput when transferring 113GB of data between PostgreSQL instances, utilizing 75% of the available 20 Gbit/s network capacity. The 10.7x speedup with 128 parallel connections demonstrates excellent scalability on OVH's high-end infrastructure. These results confirm that FastTransfer can effectively saturate modern cloud networking for PostgreSQL-to-PostgreSQL migrations.

---

## About FastTransfer

FastTransfer is a commercial high-performance data migration tool developed by [arpe.io](https://arpe.io). It provides parallel data transfer capabilities across multiple database platforms including PostgreSQL, MySQL, Oracle, SQL Server, ClickHouse, and DuckDB.

**Key Features:**
- Advanced parallelization strategies for optimal performance
- Cross-platform compatibility with major databases
- Flexible configuration for various data migration scenarios
- Production-ready with comprehensive logging and monitoring

For licensing information, support options, and to request a trial, visit the [official documentation](https://aetperf.github.io/FastTransfer-Documentation/).