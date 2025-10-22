---
title: Performance Analysis of Parallel Data Replication Between Two PostgreSQL 18 Instances on OVH WIP
layout: post
comments: true
author: François Pacull
categories: [database, performance]
tags:
- FastTransfer
- PostgreSQL 18
- Performance analysis
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

Parallel data replication between PostgreSQL instances presents unique challenges at scale, particularly when attempting to maximize throughput on high-performance cloud infrastructure. [FastTransfer](https://aetperf.github.io/FastTransfer-Documentation/) is a commercial data migration tool designed to leverage advanced parallelization strategies for efficient data movement. This post provides an in-depth performance analysis of FastTransfer transferring 113GB of data between two PostgreSQL 18 instances on OVH c3-256 servers, examining CPU, disk I/O, and network bottlenecks across parallelism degrees from 1 to 128.

## Test Configuration

The test dataset consists of the TPC-H SF100 lineitem table (~600M rows, ~113GB), configured as an UNLOGGED table without indexes, constraints, or triggers. Testing was performed at eight parallelism degrees: 1, 2, 4, 8, 16, 32, 64, and 128.

Both instances were tuned for bulk loading operations, with all durability features disabled, large memory allocations, and PostgreSQL 18's io_uring support enabled (configuration details in Appendix A). Despite this comprehensive optimization, it appears that lock contention emerges at high parallelism degrees, fundamentally limiting scalability.

**Note on Statistical Rigor:** Each configuration was run only once rather than following standard statistical practice : running each configuration several times minimum, reporting mean, standard deviation, and confidence intervals. This decision was made because preliminary observations showed very small variations between successive runs, suggesting the results are stable and reproducible under these controlled conditions.

## OVH Infrastructure Setup

The test environment consists of two identical OVH cloud instances designed for high-performance workloads:

<img src="/img/2025-10-13_01/architecture.png" alt="Architecture diagram." width="900">

**Figure 1: OVH Infrastructure Architecture** - The test setup consists of two identical c3-256 instances (128 vCores, 256GB RAM, 400GB NVMe) running PostgreSQL 18 on Ubuntu 24.04. The source instance contains the TPC-H SF100 lineitem table (~600M rows, 113GB). FastTransfer orchestrates parallel data replication across a 20 Gbit/s vrack private network connection to the target instance. Both instances are located in the Paris datacenter (eu-west-par-c) for minimal network latency.

**Hardware Configuration:**

- **Instance Type**: OVH c3-256
- **Memory**: 256GB RAM
- **CPU**: 128 vCores @ 2.3 GHz
- **Storage**:
  - **Target**: 400GB local NVMe SSD
  - **Source**: OVH Block Storage (high-speed-gen2, 1.95 TiB, 30 IOPS/GB up to 20,000 IOPS max, 0.5 MB/s/GB up to 1 GB/s max)
- **Network**: 20 Gbit/s vrack (2.5 GB/s)

**Note on Storage Configuration**: The source instance PostgreSQL data directory resides on attached OVH Block Storage rather than local NVMe. This asymmetric storage configuration does not affect the analysis conclusions, as the source PostgreSQL instance exhibits backpressure behavior (low CPU utilization at 0.11 cores/process at degree 128) rather than storage-limited performance, confirming that the target instance lock contention remains the primary bottleneck.

**Software Stack:**

- **OS**: Ubuntu 24.04.3 LTS with Linux kernel 6.8
- **PostgreSQL**: Version 18.0, with io_uring, huge pages (vm.nr_hugepages=45000)
- **FastTransfer**: Version 0.13.12

**Infrastructure Performance Baseline:**

- **Network**: 20.5 Gbit/s (2.56 GB/s) verified with iperf3
- **Disk Sequential Write**: 3,741 MB/s (FIO benchmark with 128K blocks)
- **Disk Random Read**: 313 MB/s, 80,200 IOPS (FIO, 4K blocks)
- **Disk Random Write**: 88.2 MB/s, 22,600 IOPS (FIO, 4K blocks)

## Executive Summary

FastTransfer achieves strong absolute performance, transferring 113GB in just 67 seconds at degree 128, equivalent to **1.69 GB/s sustained throughput**. The parallel replication process scales continuously across all tested degrees, with total elapsed time decreasing from 878 seconds (degree 1) to 67 seconds (degree 128), representing a **13.1x speedup**. While this represents 10.2% efficiency relative to the theoretical 128x maximum, the system delivers consistent real-world performance improvements even at extreme parallelism levels, though lock contention on the target PostgreSQL instance increasingly limits scaling efficiency beyond degree 32.

<img src="/img/2025-10-13_01/elapsed_time_by_degree.png" alt="Elapsed time by degree." width="900">

**Figure 2: Total Elapsed Time by Degree of Parallelism** - Wall-clock time improves continuously across all tested degrees, from 878 seconds (degree 1) to 67 seconds (degree 128). Performance gains remain positive throughout, though the rate of improvement diminishes beyond degree 32 due to increasing lock contention. 

**Key Findings:**

1. **Overall Performance**: The system achieves consistent performance improvements across all parallelism degrees, with the fastest transfer time of 67 seconds (1.69 GB/s) at degree 128. This represents practical value for production workloads, reducing transfer time from ~15 minutes to just over 1 minute.

2. **Target PostgreSQL**: Appears to be the primary scaling limitation. System CPU reaches 83.9% at degree 64, meaning only 16.2% of CPU time performs productive work. Mean CPU decreases from 4,596% (degree 64) to 4,230% (degree 128) despite doubling parallelism, though the reason for this unexpected improvement remains unclear.

3. **FastTransfer**: Does not appear to be a bottleneck. Operates with binary COPY protocol (`pgcopy` mode for both source and target), batch size 1,048,576 rows. Achieves 20.2x speedup with 15.8% efficiency, the best scaling efficiency among all components.

4. **Source PostgreSQL**: Appears to be a victim of backpressure, not an independent bottleneck. At degree 128, 105 processes use only 11.7 cores (0.11 cores/process), suggesting they're blocked waiting for target acknowledgments rather than actively contending for resources.

5. **Network**: Saturates at ~2,450 MB/s (98% of capacity) only at degree 128 during active bursts. Degrees 1-64 operate well below capacity, so network doesn't appear to explain scaling behavior across most of the tested range.

6. **Disk**: Does not appear to be a bottleneck. Average utilization is only 24.3% at degree 128, with 76% idle capacity remaining.

## 1. CPU Usage Analysis

### 1.1 Mean and Peak CPU Usage

<img src="/img/2025-10-13_01/plot_01_mean_cpu.png" alt="Plot 01: mean CPU." width="900">


**Figure 3: Mean CPU Usage by Component** - Target PostgreSQL (red) dominates resource consumption at high parallelism, while source PostgreSQL (blue) reaches around 12 cores.

<img src="/img/2025-10-13_01/plot_02_peak_cpu.png" alt="Plot 02: peak CPU." width="900">


**Figure 4: Peak CPU Usage by Component** - Target PostgreSQL exhibits extremely high peak values (~6,969% at degree 128). The large spikes combined with relatively lower mean values indicate high variance, characteristic of processes alternating between lock contention and productive work. The variance between peak (~6,969%) and mean (~3,294%) at degree 128 suggests lock queue buildup: processes stall waiting in queues, then burst with intense CPU activity when they finally acquire locks.

**Component Scaling Summary:**

| Component         | Degree 1         | Degree 128          | Speedup | Efficiency |
| ----------------- | ---------------- | ------------------- | ------- | ---------- |
| Source PostgreSQL | 93% (0.93 cores) | 1,110% (11.1 cores) | 11.9x   | 9.3%       |
| FastTransfer      | 31% (0.31 cores) | 631% (6.3 cores)    | 20.1x   | 15.7%      |
| Target PostgreSQL | 98% (0.98 cores) | 3,294% (32.9 cores) | 33.6x   | 26.3%      |

Source PostgreSQL's poor scaling appears to stem from backpressure: FastTransfer's batch-and-wait protocol (1,048,576 rows/batch) means source processes send a batch, then block waiting for target acknowledgment. When the target cannot consume data quickly due to lock contention, this delay propagates backward. At degree 128, 105 source processes collectively use only 11.7 cores (0.11 cores/process), suggesting they're waiting rather than actively working.

### 1.2 FastTransfer Architecture

<img src="/img/2025-10-13_01/plot_03_fasttransfer_user_system.png" alt="Plot 3: FastTransfer User vs System CPU." width="900">

**Figure 5: FastTransfer User vs System CPU** - At degree 128, FastTransfer uses 419% user CPU (66%) and 212% system CPU (34%). The system CPU proportion is appropriate for network I/O intensive applications.

FastTransfer uses PostgreSQL's binary COPY protocol for both source and target (`--sourceconnectiontype "pgcopy"` and `--targetconnectiontype "pgcopy"`). Data flows directly from source PostgreSQL's COPY TO BINARY through FastTransfer to target PostgreSQL's COPY FROM BINARY without data transformation. FastTransfer acts as an intelligent network proxy coordinating parallel streams and batch acknowledgments, explaining its relatively low CPU usage.

### 1.3 Process Counts and CPU Efficiency

<img src="/img/2025-10-13_01/plot_04_thread_process_counts.png" alt="Plot 4: Thread/Process Counts." width="900">

**Figure 6: Thread/Process Counts** - FastTransfer (green) maintains 1 process across all degrees using internal threading. PostgreSQL components (blue=source, red=target) scale linearly with their process-per-connection model. At degree 128, source spawns 105 processes, target spawns 88 processes.

<img src="/img/2025-10-13_01/plot_05_cpu_efficiency.png" alt="Plot 5: CPU Efficiency." width="900">

**Figure 7: CPU Efficiency (CPU per Degree)** - Lower values indicate better scaling. Source PostgreSQL (blue) drops significantly from 93% at degree 1 to 8.7% at degree 128, indicating processes spend most time waiting rather than working due to backpressure. Target PostgreSQL (red) drops from 69% at degree 64 to 26% at degree 128, reflecting reduced CPU utilization per worker despite achieving the best absolute performance (67s elapsed time). 

## 2. The Lock Contention Problem: System CPU Analysis

### 2.1 System CPU

<img src="/img/2025-10-13_01/plot_06_system_cpu_percentage.png" alt="Plot 6: System CPU as % of Total CPU." width="900">

**Figure 8: System CPU as % of Total CPU** - Target PostgreSQL (red line) crosses the 50% warning threshold at degree 16, exceeds 70% at degree 32, and peaks at 83.9% at degree 64. At this maximum, only 16.2% of CPU time performs productive work while 83.9% appears spent on lock contention and kernel overhead.

CPU time divides into two categories: **User CPU** (application code performing actual data insertion) and **System CPU** (kernel operations handling locks, synchronization, context switches, I/O). A healthy system maintains system CPU below 30%.

**System CPU Progression:**

| Degree | Total CPU | User CPU | System CPU | System % | Productive Work           |
| ------ | --------- | -------- | ---------- | -------- | ------------------------- |
| 1      | 98%       | 80%      | 18%        | 18.2%    | Healthy baseline          |
| 16     | 1,342%    | 496%     | 846%       | 63.0%    | Warning threshold crossed |
| 32     | 2,436%    | 602%     | 1,834%     | 75.3%    | High contention           |
| 64     | 4,596%    | 743%     | 3,854%     | 83.9%    | **Maximum contention**    |
| 128    | 4,230%    | 1,248%   | 2,982%     | 70.5%    | Reduced contention        |

At degree 64, processes appear to spend 83.9% of time managing locks rather than inserting data. By degree 128, system CPU percentage unexpectedly decreases to 70.5% for unclear reasons, though absolute performance continues to improve.

### 2.2 Root Causes of Lock Contention

The target table was already optimized for bulk loading (UNLOGGED, no indexes, no constraints, no triggers), eliminating all standard overhead sources. The remaining contention appears to stem from PostgreSQL's fundamental architecture:

1. **Shared Buffer Pool Locks**: All 128 parallel connections compete for buffer pool partition locks to read/modify/write pages. PostgreSQL's buffer manager has inherent limitations for concurrent write parallelism.

2. **Relation Extension Locks**: When the table grows, PostgreSQL requires an exclusive lock (only one process at a time). 

3. **Free Space Map (FSM) Locks**: All 128 writers query and update the FSM to find pages with free space, creating constant FSM thrashing.

## 3. Distribution and Time Series Analysis

### 3.1 CPU Distribution

<img src="/img/2025-10-13_01/plot_7_distribution_degree_4.png" alt="Plot 7: CPU Distribution at Degree 4." width="900">

**Figure 9: CPU Distribution at Degree 4** - Tight, healthy distributions with small standard deviations. All components operate consistently without significant contention.

<img src="/img/2025-10-13_01/plot_8_distribution_degree_32.png" alt="Plot 8: CPU Distribution at Degree 32." width="900">

**Figure 10: CPU Distribution at Degree 32** - Target PostgreSQL (red) becomes bimodal with wide spread (1000-3000% range), indicating some samples capture waiting processes while others capture active processes. Source (blue) remains relatively tight.

<img src="/img/2025-10-13_01/plot_9_distribution_degree_128.png" alt="Plot 9: CPU Distribution at Degree 128." width="900">

**Figure 11: CPU Distribution at Degree 128** - Target PostgreSQL (red) spans nearly 0-10000%, indicating highly variable behavior. Some processes are nearly starved (near 0%) while others burn high CPU on lock spinning (>8000%). This wide distribution suggests lock thrashing.

### 3.2 CPU Time Series

<img src="/img/2025-10-13_01/plot_10_timeseries_degree_4.png" alt="Plot 10: Time Series at Degree 4." width="900">

**Figure 12: CPU Over Time at Degree 4** - All components show stable, smooth CPU usage with minimal oscillations throughout the test duration.

<img src="/img/2025-10-13_01/plot_11_timeseries_degree_32.png" alt="Plot 11: Time Series at Degree 32." width="900">

**Figure 13: CPU Over Time at Degree 32** - Target PostgreSQL (red) shows increasing variability and oscillations, indicating periods of successful lock acquisition alternating with blocking periods.

<img src="/img/2025-10-13_01/plot_12_timeseries_degree_128.png" alt="Plot 12: Time Series at Degree 128." width="900">

**Figure 14: CPU Over Time at Degree 128** - Target PostgreSQL (red) exhibits oscillations with wild CPU swings, suggesting significant lock thrashing. Source (blue) and FastTransfer (green) show variability reflecting downstream backpressure.

## 4. Performance Scaling Analysis: Degrees 64 to 128

### 4.1 Continued Performance Improvement at Extreme Parallelism

Degree 128 achieves the best absolute performance in the test suite, completing the transfer in 67 seconds compared to 92 seconds at degree 64, a meaningful **1.37x speedup** that brings total throughput to 1.69 GB/s. While this represents 68.7% efficiency for the doubling operation (rather than the theoretical 2x), the continued improvement demonstrates that the system remains functional and beneficial even at extreme parallelism levels.

Total CPU decreases 8.0% (4,596% → 4,230%) despite doubling parallelism, while system CPU percentage improves from 83.9% to 70.5%. The reason for this unexpected efficiency improvement remains unclear, though it allows the system to maintain productivity despite the coordination challenges inherent in managing 128 parallel streams.

### 4.2 Unexpected Efficiency Improvements at Degree 128

Degree 128 exhibits a counterintuitive result: **lower system CPU overhead** (70.5%) than degree 64 (83.9%) despite doubling parallelism, while total CPU actually **decreases by 8.0%** (4,596% → 4,230%). User CPU efficiency improves by 82.1% (16.2% → 29.5% of total CPU), meaning nearly double the proportion of CPU time goes to productive work rather than lock contention. The reason for these improvements remains unclear.

**The Comparative Analysis:**

| Metric                  | Degree 64            | Degree 128           | Change               |
| ----------------------- | -------------------- | -------------------- | -------------------- |
| Elapsed Time            | 92s                  | 67s                  | 1.37x speedup        |
| Total CPU               | 4,596% (46.0 cores)  | 4,230% (42.3 cores)  | **-8.0%**            |
| User CPU                | 743% (16.2% of total)| 1,248% (29.5% of total) | **+68.0%, +82.1% efficiency** |
| System CPU              | 3,854% (83.9% of total) | 2,982% (70.5% of total) | **-22.6%**       |
| Network Throughput      | 1,033 MB/s mean      | 1,088 MB/s mean      | +5.3%                |
| Network Peak            | 2,335 MB/s (93.4%)   | 2,904 MB/s (116.2%)  | **Saturation**       |
| Disk Throughput         | 759 MB/s             | 1,099 MB/s           | +44.8%               |

**The Counterintuitive Result:**

Doubling parallelism from 64 to 128 processes produces unexpected improvements:

1. **Reduced Total CPU Usage**: Despite increasing from ~64 to 88 processes, total CPU decreased by 8.0%
2. **Improved Efficiency**: User CPU as % of total increased from 16.2% to 29.5% (82.1% relative improvement)
3. **Reduced Lock Contention**: System CPU decreased from 83.9% to 70.5% (13.4 percentage point reduction) despite more processes competing for locks
4. **Increased Throughput**: Network +5.3%, disk +44.8%, achieving 1.37x speedup with less CPU

**Open Question: Why Does Efficiency Improve at Degree 128?**

The improvement from degree 64 to 128 is puzzling for several reasons:

1. **Why does network bandwidth increase by 5.3%** (1,033 MB/s → 1,088 MB/s) when adding more parallelism to an already saturated network? At degree 128, network peaks at 2,904 MB/s (116.2% of capacity), yet mean throughput still increases.

2. **Why does system CPU overhead decrease** from 83.9% to 70.5% despite doubling parallelism? More processes should create more lock contention, not less.

3. **Why does user CPU efficiency nearly double** (16.2% → 29.5% of total) when adding 64 more processes competing for the same resources?

One hypothesis is that network saturation at degree 128 acts as a pacing mechanism, rate-limiting data delivery and preventing all 128 processes from simultaneously contending for locks. However, this doesn't fully explain why network throughput itself increases, nor why the efficiency gains are so substantial. The interaction between network saturation, lock contention, and process scheduling appears more complex than initially understood.

**Note:** Network saturation occurs **only at degree 128** and doesn't explain poor scaling from degree 1-64. The primary bottleneck causing poor scaling efficiency (13.1x instead of 128x) remains target CPU lock contention across the entire tested range. The degree 128 improvements, while beneficial, represent an unexplained anomaly rather than the dominant scaling pattern.

## 5. Disk I/O and Network Analysis

### 5.1 FIO Disk Benchmarks vs PostgreSQL Performance

<img src="/img/2025-10-13_01/disk_fio_vs_postgresql_bandwidth.png" alt="Disk Bandwidth Comparison." width="900">

**Figure 15a: FIO Benchmark vs PostgreSQL Actual Performance (Bandwidth)** - PostgreSQL achieves 3,759 MB/s peak (100.5% of FIO's 3,741 MB/s), demonstrating it can saturate disk during bursts. However, average is only 170 MB/s (4.5% of peak), revealing highly bursty behavior with long idle periods.

<img src="/img/2025-10-13_01/disk_fio_vs_postgresql_iops.png" alt="Disk IOPS Comparison." width="900">

**Figure 15b: FIO Benchmark vs PostgreSQL Actual Performance (IOPS)** - Peak IOPS reaches 55,501 operations per second during intensive bursts, but average IOPS is only 207 operations per second, further confirming the bursty pattern with most time spent idle or at low activity levels.

**FIO Results:**

- Sequential Write (128K): 3,741 MB/s, 29,900 IOPS
- Random Read (4K): 313 MB/s, 80,200 IOPS
- Random Write (4K): 88.2 MB/s, 22,600 IOPS

**PostgreSQL Actual:**

- Peak Write: 3,759 MB/s, 55,501 IOPS (matches FIO sequential)
- Mean Write: 170 MB/s, 207 IOPS (only 4.5% of peak)
- Mean Disk Utilization: 14.6% (disk idle 85% of the time)

### 5.2 Source Disk I/O Analysis

This section examines source PostgreSQL disk I/O behavior to provide concrete evidence that the source is experiencing backpressure from the target bottleneck rather than being disk-limited.

#### The Cache Effect and First-Run Impact

The source instance has 256GB RAM, and the lineitem table is ~113GB. An important detail explains the disk behavior across test runs:

**Degree 1 was the first test run** with no prior warm-up or cold run to pre-load the table into cache. During this first run:

**At degree 1 (first ~10 seconds)**: Heavy disk activity (500 MB/s, 100% utilization) loads the table into memory (shared_buffers + OS page cache).

**At degree 1 (remaining ~860 seconds)**: Near-zero disk activity—the table is fully cached in RAM, no disk reads needed.

**At degrees 2-128**: Essentially zero disk activity—the entire table remains cached in memory from the initial degree 1 load.

**This explains why degree 2 is more than twice as fast as degree 1**: The degree 1 run includes the initial table-loading overhead (~10 seconds of intensive disk I/O), while degree 2 benefits from the already-cached table with no disk loading required. The speedup from degree 1 to 2 reflects both the doubling of parallelism AND the elimination of the initial cache-loading penalty.

<img src="/img/2025-10-13_01/source_disk_utilization_timeseries.png" alt="Source Disk Utilization Time Series." width="900">

**Figure 16: Source Disk Utilization Over Time** - Shows disk utilization across all test runs (vertical lines mark test boundaries for degrees 1, 2, 4, 8, 16, 32, 64, 128). At degree 1, utilization peaks at ~50% during the initial table load, then drops to near-zero. At higher degrees (2-128), utilization remains below 1% throughout, confirming the disk is idle and not limiting performance.

#### The Evidence Chain

The source disk analysis reveals the interplay between caching and backpressure:

```
Table Cached in RAM (256GB RAM, 113GB table)
    ↓
Source Disk Becomes Idle (0.1 MB/s, 0.0% utilization at degrees 2-128)
    ↓
Meanwhile: Target Lock Contention (83.9% system CPU at degree 64)
    ↓
Target Can't Consume Data Fast Enough (1,088 MB/s vs 1,684 MB/s source TX)
    ↓
FastTransfer Batches Block (waiting for target acknowledgment)
    ↓
Source Processes Sleep (0.11 cores/process, blocked in system calls)
```

#### Conclusion

**Source disk I/O is NOT a bottleneck at any parallelism degree.** The source exhibits different behavior depending on parallelism:

- **Degree 1**: Brief initial disk load (~10 seconds), then reads from RAM cache
- **Degrees 2-128**: Table fully cached in memory (256GB available, 113GB table)

The near-zero disk utilization (<1%) at high parallelism degrees confirms the disk is idle and not limiting performance. This indicates that source processes are constrained by downstream backpressure rather than local disk I/O capacity. Resolving target CPU lock contention would automatically improve source performance, as the source has substantial unused capacity waiting to be unlocked.

### 5.3 Target Disk I/O Time Series

<img src="/img/2025-10-13_01/target_disk_write_throughput_timeseries.png" alt="Target Disk Write Throughput Time Series." width="900">

**Figure 17: Target Disk Write Throughput Over Time** - Vertical lines mark test boundaries (degrees 1, 2, 4, 8, 16, 32, 64, 128). Throughput exhibits bursty behavior with spikes to 2000-3759 MB/s followed by drops to near zero. Sustained baseline varies from ~100 MB/s (low degrees) to ~300 MB/s (degree 128) but never sustains disk capacity.

<img src="/img/2025-10-13_01/target_disk_utilization_timeseries.png" alt="Target Disk Utilization Time Series." width="900">

**Figure 18: Target Disk Utilization Over Time** - Mean utilization remains below 25% across all degrees. Spikes reach 70-90% during bursts but quickly return to low baseline. This strongly suggests disk I/O is not the bottleneck.

### 5.4 Network Throughput Analysis

<img src="/img/2025-10-13_01/target_network_rx_timeseries.png" alt="Target Network RX Time Series." width="900">

**Figure 19: Target Network Ingress Over Time** - At degree 128, throughput plateaus at ~2,450 MB/s (98% of capacity) during active bursts, but averages only 1,088 MB/s (43.5%) due to alternating active/idle periods. At degrees 1-64, network remains well below capacity.

**Network Scaling Summary:**

| Degree | Mean RX    | % of 2.5 GB/s Capacity | Active Burst Plateau                |
| ------ | ---------- | ---------------------- | ----------------------------------- |
| 1      | 122 MB/s   | 4.9%                   | Well below capacity                 |
| 8      | 631 MB/s   | 25.3%                  | Well below capacity                 |
| 16     | 769 MB/s   | 30.8%                  | Well below capacity                 |
| 32     | 911 MB/s   | 36.4%                  | Well below capacity                 |
| 64     | 1,033 MB/s | 41.3%                  | Well below capacity                 |
| 128    | 1,088 MB/s | 43.5%                  | **~2,450 MB/s (98%) during bursts** |

Network saturation occurs **only at degree 128** during active bursts. Therefore, network doesn't explain poor scaling from degree 1 through 64, target CPU lock contention remains the primary bottleneck.

### 5.5 Cross-Degree Scaling Analysis

<img src="/img/2025-10-13_01/cross_degree_disk_write_mean.png" alt="Cross Degree Mean Disk Write." width="900">

**Figure 20: Mean Disk Write Throughput by Degree** - Scales from 90 MB/s (degree 1) to 1,099 MB/s (degree 128), only 12.3x improvement for 128x parallelism (9.6% efficiency).

<img src="/img/2025-10-13_01/cross_degree_network_comparison.png" alt="Cross Degree Network Comparison." width="900">

**Figure 21: Network Throughput Comparison: Source TX vs Target RX** - At degree 128, source transmits 1,684 MB/s while target receives only 1,088 MB/s, creating a 596 MB/s (35%) deficit. This suggests the target cannot keep pace with source data production, likely due to CPU lock contention.

**Technical Note on TX/RX Discrepancy:** The apparent 35% violation of flow conservation is explained by TCP retransmissions. The source TX counter (measured via `sar -n DEV`) counts both original packets and retransmitted packets, while the target RX counter only counts successfully received unique packets. When the target is overloaded with CPU lock contention (83.9% system CPU at degree 64), it cannot drain receive buffers fast enough, causing packet drops that trigger TCP retransmissions. The 596 MB/s "deficit" is actually retransmitted data counted twice at the source but only once at the target, providing quantitative evidence of the target's inability to keep pace with source data production.

<img src="/img/2025-10-13_01/cross_degree_disk_utilization.png" alt="Cross Degree Disk Utilization." width="900">


**Figure 22: Disk Utilization by Degree** - Mean utilization increases from 2.2% (degree 1) to only 24.3% (degree 128), remaining far below the 80% saturation threshold at all degrees. This strongly indicates disk I/O is not the bottleneck.

### 5.6 I/O Analysis Conclusions

1. **Disk does not appear to be the bottleneck**: 24% average utilization at degree 128 with 76% idle capacity. PostgreSQL matches FIO peak (3,759 MB/s) but sustains only 170 MB/s average.

2. **Network does not appear to be the bottleneck for degrees 1-64**: Utilization remains below 42% through degree 64. Saturation occurs only at degree 128 during active bursts (~2,450 MB/s plateau).

3. **Target CPU lock contention appears to be the root cause**: Low disk utilization + network saturation only at degree 128 + poor scaling efficiency throughout + high system CPU percentage (83.9% at degree 64) all point to the same conclusion.

4. **Backpressure suggests target bottleneck**: Source can produce 1,684 MB/s but target can only consume 1,088 MB/s. Source processes use only 0.11 cores/process, suggesting they're blocked waiting for target acknowledgments.

## 6. Conclusions

### 6.1 Performance Achievement and Bottleneck Analysis

FastTransfer successfully demonstrates strong absolute performance, achieving a 13.1x speedup that reduces 113GB transfer time from approximately 15 minutes (878s) to just over 1 minute (67s). This represents practical, production-ready performance with sustained throughput of 1.69 GB/s at degree 128. The system delivers continuous performance improvements across all tested parallelism degrees, confirming that parallel replication provides meaningful benefits even when facing coordination challenges.

The primary scaling limitation appears to be target PostgreSQL lock contention beyond degree 32. System CPU grows to 83.9% at degree 64, meaning only 16.2% of CPU performs productive work. Degree 128 continues to improve absolute performance (67s vs 92s) even as total CPU decreases from 4,596% to 4,230%, though the reason for this unexpected efficiency improvement remains unclear.

Source PostgreSQL and FastTransfer appear to be victims of backpressure rather than independent bottlenecks. FastTransfer demonstrates the best scaling efficiency (20.2x speedup, 15.8% efficiency), while source processes spend most time waiting for target acknowledgments. Resolving target lock contention would likely improve their performance further.

### 6.2 Why Additional Tuning Cannot Help

The target table is already optimally configured (UNLOGGED, no indexes, no constraints, no triggers). PostgreSQL configuration includes all recommended bulk loading optimizations (80GB shared_buffers, huge pages, io_uring, fsync=off). Despite this, system CPU remains at 70-84% at high degrees.

The bottleneck appears to be **architectural**, not configurational:

- Buffer pool partition locks are hardcoded, not tunable
- Relation extension lock is a single exclusive lock per table by design
- FSM access requires serialization to maintain consistency

No configuration parameter appears able to eliminate these fundamental coordination requirements.

## Appendix A: PostgreSQL Configuration

Both PostgreSQL 18 instances were aggressively tuned for maximum bulk loading performance. The configuration represents state-of-the-art optimization with every available parameter tuned for performance.

### Target PostgreSQL Configuration (Key Settings)

```ini
# Memory allocation
shared_buffers = 80GB              # 31% of 256GB RAM
huge_pages = on                    # vm.nr_hugepages=45000
work_mem = 256MB
maintenance_work_mem = 16GB

# Durability disabled (benchmark only, NOT production)
synchronous_commit = off
fsync = off
full_page_writes = off

# WAL configuration (minimal for UNLOGGED)
wal_level = minimal
wal_buffers = 128MB
max_wal_size = 128GB
checkpoint_timeout = 15min
checkpoint_completion_target = 0.5

# Background writer (AGGRESSIVE)
bgwriter_delay = 10ms              # Down from default 200ms
bgwriter_lru_maxpages = 2000       # 2x default
bgwriter_lru_multiplier = 8.0      # 2x default
bgwriter_flush_after = 0

# I/O configuration (PostgreSQL 18 optimizations)
backend_flush_after = 0
effective_io_concurrency = 400     # Optimized for NVMe
maintenance_io_concurrency = 400
io_method = io_uring               # NEW PG18: async I/O
io_max_concurrency = 512           # NEW PG18
io_workers = 8                     # NEW PG18: up from default 3

# Worker processes
max_worker_processes = 128
max_parallel_workers = 128

# Autovacuum (PostgreSQL 18)
autovacuum = on
autovacuum_worker_slots = 32       # NEW PG18: runtime adjustment
autovacuum_max_workers = 16
autovacuum_vacuum_cost_delay = 0   # No throttling

# Query tuning
enable_partitionwise_join = on
enable_partitionwise_aggregate = on
random_page_cost = 1.1             # NVMe SSD
effective_cache_size = 192GB       # ~75% of RAM
```

### Source PostgreSQL Configuration (Key Settings)

The source instance is optimized for fast parallel reads to support high-throughput data extraction:

```ini
# Memory allocation
shared_buffers = 80GB              # ~31% of 256GB RAM
huge_pages = on                    # vm.nr_hugepages=45000
work_mem = 256MB
maintenance_work_mem = 4GB         # Lower than target (16GB)

# Durability disabled (benchmark only, NOT production)
synchronous_commit = off
fsync = off
full_page_writes = off

# WAL configuration
wal_level = minimal
wal_buffers = -1                   # Auto-sized
max_wal_size = 32GB                # Smaller than target (128GB)
checkpoint_timeout = 60min         # Longer than target (15min)
checkpoint_completion_target = 0.9

# Background writer
bgwriter_delay = 50ms              # Less aggressive than target (10ms)
bgwriter_lru_maxpages = 1000       # Half of target (2000)
bgwriter_lru_multiplier = 4.0      # Half of target (8.0)
bgwriter_flush_after = 2MB

# I/O configuration (PostgreSQL 18 optimizations)
backend_flush_after = 0
effective_io_concurrency = 400     # Identical to target
maintenance_io_concurrency = 400
io_method = io_uring               # NEW PG18: async I/O
io_max_concurrency = 512           # NEW PG18
io_workers = 8                     # NEW PG18

# Worker processes
max_connections = 500              # Higher than target for parallel readers
max_worker_processes = 128
max_parallel_workers_per_gather = 64
max_parallel_workers = 128

# Query tuning (optimized for parallel reads)
enable_partitionwise_join = on
enable_partitionwise_aggregate = on
random_page_cost = 1.1             # Block Storage (not NVMe)
effective_cache_size = 192GB       # ~75% of RAM
default_statistics_target = 500

# Autovacuum (PostgreSQL 18)
autovacuum = on
autovacuum_worker_slots = 32       # NEW PG18: runtime adjustment
autovacuum_max_workers = 16
autovacuum_vacuum_cost_delay = 0   # No throttling
```

**Key Differences from Target:**
- **Connection limit**: 500 vs target's default, accommodating parallel reader connections
- **Background writer**: Less aggressive settings since source focuses on reads, not writes
- **Checkpoint timeout**: 60 minutes vs target's 15 minutes, reducing checkpoint overhead during reads
- **Storage**: Block Storage (random_page_cost = 1.1) vs target's NVMe

### Table Configuration

The target table eliminates all overhead sources:

- **UNLOGGED**: No WAL write, flush, or archival overhead
- **No indexes**: Eliminates 50-80% of bulk load cost
- **No primary key**: No index maintenance or uniqueness checking
- **No constraints**: No foreign key, check, or unique validation
- **No triggers**: No trigger execution overhead

This represents the absolute minimum overhead possible. The fact that lock contention persists suggests the bottleneck lies in PostgreSQL's buffer management and relation extension architecture rather than higher-level features.

### PostgreSQL 18 New Features Utilized

- **io_uring**: Improved async I/O on Linux kernel 5.1+ (Ubuntu 24.04 ships with kernel 6.8)
- **io_max_concurrency**: Fine-grained I/O parallelism control, utilizing all 128 vCPUs
- **io_workers**: Increased from default 3 to 8 for better NVMe parallelism
- **autovacuum_worker_slots**: Dynamic autovacuum worker management without restart

These PostgreSQL 18 enhancements provide measurable I/O efficiency improvements, but the fundamental architectural limitation of concurrent writes to a single table persists.

---

## Future Work: PostgreSQL Instrumentation Analysis

While this analysis relied on system-level metrics (CPU, disk, network via `sar`, `iostat`, `pidstat`), a follow-up study will use PostgreSQL's internal instrumentation to definitively identify bottlenecks at the database engine level. This will provide direct evidence of lock contention and wait events rather than inferring them from system CPU percentages.

**Planned Instrumentation:**

- **`pg_stat_activity` sampling**: Capture every 1 second during tests to track process states (`state`, `wait_event_type`, `wait_event`) in real-time
- **Wait event analysis**: Log and aggregate wait events to quantify time spent in different wait states (Lock, LWLock, BufferPin, IO, etc.)
- **`pg_stat_io` statistics**: Monitor I/O operations at the PostgreSQL level (shared buffer hits/misses, relation extension operations, FSM access patterns)
- **`pg_stat_database` metrics**: Track transaction commits, buffer operations, and temporary file usage across parallelism degrees
- **`pg_locks` monitoring**: Capture actual lock acquisition and contention events to identify specific lock types (relation extension, buffer content, etc.)

This instrumentation will validate the lock contention hypothesis presented in this analysis and provide quantitative breakdowns of where PostgreSQL processes spend their time. The results will be the subject of a future blog post.

---

## About FastTransfer

FastTransfer is a commercial high-performance data migration tool developed by [arpe.io](https://arpe.io). It provides parallel data transfer capabilities across multiple database platforms including PostgreSQL, MySQL, Oracle, SQL Server, ClickHouse, and DuckDB.

**Key Features:**

- Advanced parallelization strategies for optimal performance
- Cross-platform compatibility with major databases
- Flexible configuration for various data migration scenarios
- Production-ready with comprehensive logging and monitoring

For licensing information, support options, and to request a trial, visit the [official documentation](https://aetperf.github.io/FastTransfer-Documentation/).
