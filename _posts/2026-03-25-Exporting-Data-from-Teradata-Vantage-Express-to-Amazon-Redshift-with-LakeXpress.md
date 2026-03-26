---
title: Exporting Data from Teradata Vantage Express to Amazon Redshift with LakeXpress WIP
layout: post
tags:
  - LakeXpress
  - Teradata
  - Amazon Redshift
  - Parquet
  - AWS
  - data engineering
---

[LakeXpress](https://lakexpress-docs.arpe.io/latest/) is a command-line tool that exports data from relational databases to Parquet files and publishes them to cloud data platforms. In this post, we move a healthcare dataset from a local Teradata Vantage Express instance to Amazon Redshift, using S3 as intermediate storage. We then measure the effect of adding time-based partitioning and scaling the target cluster from 1 to 4 nodes. The source runs in a VirtualBox VM on a Linux laptop; the target is a single-node `ra3.large` provisioned cluster in `eu-west-1`.

The pipeline uses:

- [Teradata Vantage](https://www.teradata.com/platform/vantagecloud) -- a commercial MPP (massively parallel processing database) analytics database. **Vantage Express** is the free single-node edition for development and testing.
- [Amazon Redshift](https://aws.amazon.com/redshift/) -- AWS's cloud data warehouse, originally forked from PostgreSQL 8.0. It uses columnar storage and an MPP architecture while retaining the PostgreSQL wire protocol and most of the SQL dialect.
- [Amazon S3](https://aws.amazon.com/s3/) -- object storage, used here as the intermediate landing zone for exported Parquet files.
- [Amazon Redshift Spectrum](https://docs.aws.amazon.com/redshift/latest/dg/c-using-spectrum.html) -- a Redshift feature that queries data directly in S3 via external tables, without loading it into the cluster.
- [AWS Glue Data Catalog](https://docs.aws.amazon.com/glue/latest/dg/catalog-and-crawler.html) -- the metastore that Spectrum uses for external table definitions.
- [Apache Parquet](https://parquet.apache.org/) -- a columnar file format common in analytics workloads.
- [VirtualBox](https://www.virtualbox.org/) -- a free hypervisor for running virtual machines.

**Outline**

- [Teradata Vantage Express in VirtualBox](#teradata-vantage-express-in-virtualbox)
- [The SynPUF dataset](#the-synpuf-dataset)
- [SQL Server as configuration and log database](#sql-server-as-configuration-and-log-database)
- [S3 bucket](#s3-bucket)
- [Amazon Redshift cluster](#amazon-redshift-cluster)
- [Credentials file](#credentials-file)
- [LakeXpress configuration](#lakexpress-configuration)
- [Running the sync](#running-the-sync)
- [Improving publish performance with partitioned exports](#improving-publish-performance-with-partitioned-exports)
- [Shutdown](#shutdown)

## Teradata Vantage Express in VirtualBox

[Teradata Vantage Express](https://www.teradata.com/getting-started/vantage-express) is a free, single-node edition of Teradata Vantage intended for development and testing. We run version 20.00.28.81 (SLES 15) inside [VirtualBox](https://www.virtualbox.org/) on a Linux host. The VM is configured with 4 CPUs and 8 GB of RAM, using both host-only and NAT networking -- the host-only adapter places it at `192.168.56.101` on port 1025.

Starting the VM headless from the command line:

```bash
VBoxManage startvm "VantageExpress_20.00.28.81_SLES15_20251203053450" --type headless
```

```
Waiting for VM "VantageExpress_20.00.28.81_SLES15_20251203053450" to power on...
VM "VantageExpress_20.00.28.81_SLES15_20251203053450" has been successfully started.
```

To check if the VM is already running:

```bash
VBoxManage list runningvms
```

### VM resources and parallelism

Vantage Express runs on a single AMP (Access Module Processor), so it does not support intra-table parallelism the way PostgreSQL (`Ctid`) or Oracle (`Rowid`) do. LakeXpress can still export multiple tables concurrently with `--n_jobs`, but each table is read through a single Teradata session. The host has 20 cores, but giving the VM more than 4 CPUs would not help much given the single-AMP constraint.

### Spool space

Teradata uses SPOOL space as working memory for query processing. On Vantage Express, the default is 512 MB per user, shared across all concurrent sessions. With `--n_jobs 4`, four sessions run simultaneously, each getting roughly 128 MB -- tight enough to cause spool-space errors on larger tables. We set spool space to 4 GB on both the database and the user:

| Setting        | Default | Configured | Reason                                   |
|----------------|--------:|-----------:|------------------------------------------|
| User SPOOL     | 512 MB  | 4 GB       | Shared across all concurrent sessions    |
| Database SPOOL | 512 MB  | 4 GB       | Governs sessions defaulting to this DB   |
| TEMPORARY      | not set | 512 MB     | Needed if export uses volatile/temp tables |

The `NO FALLBACK` and `NO JOURNAL` settings on the database reduce I/O overhead, appropriate for a single-node test instance with no redundancy requirements.

## The SynPUF dataset

The source data is the [CMS Synthetic Public Use Files (SynPUF)](https://www.cms.gov/data-research/statistics-trends-and-reports/medicare-claims-synthetic-public-use-files/cms-2008-2010-data-entrepreneurs-synthetic-public-use-file-de-synpuf), a synthetic Medicare claims dataset covering 2008--2010. It was loaded into the `SYNPUF` database from 8 CSV files (362 MB compressed), totaling 11,494,963 rows across 5 tables and about 1.4 GB on disk:

| Table                    | Type                          |      Rows |     Size |
|--------------------------|-------------------------------:|----------:|---------:|
| BENEFICIARY_SUMMARY      | SET, PPI on birth date        |   343,644 |  21.5 MB |
| INPATIENT_CLAIMS         | MULTISET, SI on CLM_FROM_DT   |    66,773 |  14.9 MB |
| OUTPATIENT_CLAIMS        | SET                           |   790,790 | 134.5 MB |
| CARRIER_CLAIMS           | SET, NUPI                     | 4,741,335 | 841.3 MB |
| PRESCRIPTION_DRUG_EVENTS | MULTISET, PPI on service date | 5,552,421 | 416.6 MB |

The schema uses Teradata-specific features: SET and MULTISET tables, Partitioned Primary Indexes (PPI), a Secondary Index, and a Non-Unique Primary Index (NUPI). The database also includes a view (`V_BENEFICIARY_CLAIMS_SUMMARY`), a macro (`CLAIMS_SUMMARY_BY_YEAR`), and a stored procedure (`CALC_MEMBER_RISK_SCORES`). LakeXpress exports tables only.

## SQL Server as configuration and log database

LakeXpress uses a relational database — here SQL Server 2022 running in a Docker container — to store both the export configuration and the job tracking information. When we create a configuration, it is persisted in this database, and LakeXpress reads it back at runtime to drive the export. In that sense, LakeXpress together with its database acts as an orchestrator: the database holds the what and the how, while the LakeXpress engine executes accordingly and logs progress back into the same database.

## S3 bucket

The exported Parquet files land in `s3://aetplakexpress/lakexpress/`. To clear any previous export before a fresh run:

```bash
aws s3 rm s3://aetplakexpress/lakexpress/ --recursive
```

## Amazon Redshift cluster

We use a provisioned cluster -- a single `ra3.large` node in `eu-west-1` at $0.60/hour. The cluster requires an IAM role (`LakeXpressRedshiftRole`) with read access to the S3 bucket so that Redshift can run `COPY` from S3. The security group must allow inbound TCP on port 5439.

LakeXpress supports two publish methods for Redshift:

- **internal**: `COPY` from S3 Parquet into native Redshift tables. Data is loaded into the cluster. Only S3 read access is needed on the IAM role.
- **external**: creates Redshift Spectrum external tables backed by the AWS Glue Data Catalog. Data stays in S3 and is read at query time. The IAM role needs both S3 and Glue catalog access.

We use `internal` in this post.

The cluster can be managed from the command line with the [AWS CLI](https://aws.amazon.com/cli/). Checking the cluster status:

```bash
aws redshift describe-clusters --cluster-identifier lakexpress-test --region eu-west-1 \
    --query "Clusters[0].ClusterStatus"
```

Returns `"available"`, `"paused"`, `"resuming"`, etc. Resuming a paused cluster:

```bash
aws redshift resume-cluster --cluster-identifier lakexpress-test --region eu-west-1
```

Pausing the cluster to stop billing:

```bash
aws redshift pause-cluster --cluster-identifier lakexpress-test --region eu-west-1
```

Deleting the cluster entirely:

```bash
aws redshift delete-cluster --cluster-identifier lakexpress-test \
    --skip-final-cluster-snapshot --region eu-west-1
```

## Credentials file

LakeXpress reads connection details from a JSON file (`data/ds_credentials.json`). Each entry is referenced by key in the `config create` command. The Redshift entry includes `iam_role` and `region` -- not used for the connection itself (standard PostgreSQL wire protocol) but required by the `COPY` command to locate the S3 bucket and assume the IAM role.

```json
{
  "log_db_ms_02": {
    "ds_type": "mssql",
    "auth_mode": "classic",
    "info": {
      "username": "$env{LX_LXDB_USER}",
      "password": "$env{LX_LXDB_PASSWORD}",
      "server": "localhost",
      "port": 1433,
      "database": "lakexpress_log"
    }
  },
  "source_teradata": {
    "ds_type": "teradata",
    "auth_mode": "classic",
    "info": {
      "server": "192.168.56.101",
      "port": 1025,
      "database": "SYNPUF",
      "username": "$env{LX_TERADATA_USER}",
      "password": "$env{LX_TERADATA_PASSWORD}"
    }
  },
  "aws_s3_01": {
    "ds_type": "s3",
    "auth_mode": "profile",
    "info": {
      "directory": "s3://your-bucket/lakexpress/",
      "profile": "your-aws-profile"
    }
  },
  "redshift_classic": {
    "ds_type": "redshift",
    "auth_mode": "classic",
    "info": {
      "host": "your-cluster.eu-west-1.redshift.amazonaws.com",
      "port": 5439,
      "database": "dev",
      "username": "$env{LX_REDSHIFT_USER}",
      "password": "$env{LX_REDSHIFT_PASSWORD}",
      "iam_role": "arn:aws:iam::123456789012:role/YourRedshiftRole",
      "region": "eu-west-1"
    }
  }
}
```

The `$env{...}` syntax substitutes environment variables at runtime, keeping secrets out of the file.

## LakeXpress configuration

The configuration below exports from the Teradata `SYNPUF` schema, writes Parquet files to S3, and publishes to Redshift using the internal method. The target schema name follows a date pattern (`INT_synpuf_YYYYMMDD`):

```bash
./LakeXpress config create \
    -a data/ds_credentials.json \
    --lxdb_auth_id log_db_ms_02 \
    --source_db_auth_id source_teradata \
    --source_db_name SYNPUF \
    --source_schema_name SYNPUF \
    --n_jobs 4 \
    --target_storage_id aws_s3_01 \
    --sub_path synpuf \
    --publish_method internal \
    --publish_schema_pattern "INT_{subpath}_{date}" \
    --publish_table_pattern "{schema}_{table}" \
    --publish_target redshift_classic
```

Here is a breakdown of each argument:

| Argument | Description |
|---|---|
| `-a` | Path to the JSON credentials file containing connection details for all components (LakeXpress DB, source database, storage, and publishing targets). |
| `--lxdb_auth_id` | Identifier in the credentials file for the LakeXpress database, where configuration and job tracking information are stored. |
| `--source_db_auth_id` | Identifier in the credentials file for the source database to export from. |
| `--source_db_name` | Name of the source database. |
| `--source_schema_name` | Source schema name to export. Supports SQL `LIKE` patterns (e.g. `prod_%`). |
| `--n_jobs` | Number of parallel table export jobs. |
| `--target_storage_id` | Identifier in the credentials file for the cloud storage destination. |
| `--sub_path` | Sub-path inserted between the base storage path and the schema directory, also available as the `{subpath}` token in naming patterns. |
| `--publish_method` | `internal` loads data into the target database; `external` registers tables pointing at cloud storage. |
| `--publish_schema_pattern` | Dynamic schema naming using tokens such as `{schema}`, `{subpath}`, `{date}`, `{timestamp}`. |
| `--publish_table_pattern` | Dynamic table naming using the same tokens. Must include `{table}`. |
| `--publish_target` | Identifier in the credentials file for the target data platform where tables are published. |

This returns a `sync_id` that identifies the configuration for subsequent operations.

## Running the sync

The `sync` command runs schema discovery, parallel export to Parquet, upload to S3, and `COPY` into Redshift in a single pass:

```bash
./LakeXpress sync --sync_id sync-20260325-c793c6 --quiet_fbcp
```

The `--quiet_fbcp` flag suppresses the per-row progress output from the underlying FastBCP export engine.

### Export phase

LakeXpress discovers the 5 tables, sorts them by estimated row count, and dispatches them across 4 parallel workers. Each worker runs a FastBCP process that reads from Teradata via `teradatasql`, writes a Zstd-compressed Parquet file, and uploads it directly to S3:

```
Discovered 5 tables in schema SYNPUF
Tables sorted by estimated row count (largest first) across all schemas:
  1. SYNPUF.PRESCRIPTION_DRUG_EVENTS (22,209,684 rows)
  2. SYNPUF.CARRIER_CLAIMS (14,224,005 rows)
  3. SYNPUF.OUTPATIENT_CLAIMS (3,163,160 rows)
  4. SYNPUF.BENEFICIARY_SUMMARY (1,374,576 rows)
  5. SYNPUF.INPATIENT_CLAIMS (267,092 rows)
```

The estimated row counts from Teradata's catalog (`DBC.TablesV`) are inflated relative to the actual counts when statistics have not been recently collected.

BENEFICIARY_SUMMARY and INPATIENT_CLAIMS complete in under 10 seconds, freeing workers for the remaining tables. CARRIER_CLAIMS is the bottleneck at 1:46, despite having fewer rows than PRESCRIPTION_DRUG_EVENTS -- it has more columns (81 vs. 12) and wider rows.

```
Export: 5/5 tables succeeded
Total Parquet export - time (s) : 110.67
```

### Publish phase

LakeXpress connects to Redshift, creates the target schema `INT_synpuf_20260325`, generates DDL with type mappings from Teradata to Redshift, and runs `COPY` commands to load each table from S3. Table creation is parallelized across 4 workers, each with its own Redshift connection:

```
PUBLISH SUMMARY
============================================================
  Schemas created: 1
  Tables created:  5
  Tables failed:   0
  Rows loaded:     11,494,963
```

```
Total publication - time (s) :  98.11
Total elapsed     - time (s) : 208.81
```

The full pipeline completed in 3 minutes 29 seconds for 11.5 million rows across 5 tables.

| Table                    |      Rows | Export (s) | Publish (s) |
|--------------------------|----------:|-----------:|------------:|
| BENEFICIARY_SUMMARY      |   343,644 |          4 |           8 |
| INPATIENT_CLAIMS         |    66,773 |          3 |          10 |
| OUTPATIENT_CLAIMS        |   790,790 |         15 |          22 |
| PRESCRIPTION_DRUG_EVENTS | 5,552,421 |         28 |          22 |
| CARRIER_CLAIMS           | 4,741,335 |        106 |          97 |

CARRIER_CLAIMS dominates both phases. The 97s publish time for a single 841 MB file on a single-node cluster is the natural target for improvement.

## Improving publish performance with partitioned exports

The Redshift `COPY` command scans S3 paths recursively and loads multiple Parquet files in parallel. Exporting large tables into multiple files partitioned by time gives Redshift more work units to distribute across its compute slices. To take full advantage of this, we also resize the cluster from 1 to 4 nodes. A single-node cluster cannot elastic resize, so we use a classic resize:

```bash
aws redshift resize-cluster --cluster-identifier lakexpress-test \
    --node-type ra3.large --number-of-nodes 4 --classic --region eu-west-1
```

Classic resize can take 30 minutes or more depending on data volume. Monitor the status with:

```bash
aws redshift describe-clusters --cluster-identifier lakexpress-test --region eu-west-1 \
    --query "Clusters[0].ClusterStatus"
```

With 4 `ra3.large` nodes ($2.40/hour total), the cluster has more compute slices to distribute `COPY` work.

Clean up the previous export:

```bash
aws s3 rm s3://aetplakexpress/lakexpress/ --recursive
```

Then create a new configuration with `--fastbcp_table_config` to partition the two largest tables by date column using the `Timepartition` method with 4 parallel readers:

```bash
./LakeXpress config create \
    -a data/ds_credentials.json \
    --lxdb_auth_id log_db_ms_02 \
    --source_db_auth_id source_teradata \
    --source_db_name SYNPUF \
    --source_schema_name SYNPUF \
    --fastbcp_table_config "CARRIER_CLAIMS:Timepartition:(CLM_FROM_DT,year,month):4;PRESCRIPTION_DRUG_EVENTS:Timepartition:(SRVC_DT,year,month):4" \
    --n_jobs 4 \
    --target_storage_id aws_s3_01 \
    --sub_path synpuf \
    --publish_method internal \
    --publish_schema_pattern "INT_{subpath}_{date}" \
    --publish_table_pattern "{schema}_{table}" \
    --publish_target redshift_classic
```

Multiple table configs can also be passed as separate arguments instead of semicolon-separated in a single value:

```bash
    --fastbcp_table_config "CARRIER_CLAIMS:Timepartition:(CLM_FROM_DT,year,month):4" \
    --fastbcp_table_config "PRESCRIPTION_DRUG_EVENTS:Timepartition:(SRVC_DT,year,month):4"
```

The `Timepartition:(CLM_FROM_DT,year,month):4` syntax tells [FastBCP](https://fastbcp-docs.arpe.io/latest/documentation/cli/parallel_parameters#available-methods) to split CARRIER_CLAIMS by year and month on `CLM_FROM_DT` using 4 parallel processes. Unlike database-specific partitioning methods such as `Ctid` (PostgreSQL), `Physloc` (SQL Server), or `Rowid` (Oracle), `Timepartition` works with any source database. Each year/month combination produces a separate Parquet file under Hive-style paths (`year=YYYY/month=MM/`), which Redshift then loads in parallel across its compute slices.

With `--n_jobs 4`, both partitioned tables run simultaneously alongside two smaller tables, reaching up to 10 concurrent Teradata sessions at peak (4 + 4 + 1 + 1). The 4 GB spool space configured earlier is shared across all sessions.

### Results

With both tables partitioned into 36 files each (2008--2010, 12 months per year) and a 4-node cluster, the full pipeline completes in 2 minutes 1 second -- down from 3 minutes 29 seconds. The S3 layout after export (truncated):

```
lakexpress/synpuf/SYNPUF/BENEFICIARY_SUMMARY/BENEFICIARY_SUMMARY.parquet
lakexpress/synpuf/SYNPUF/INPATIENT_CLAIMS/INPATIENT_CLAIMS.parquet
lakexpress/synpuf/SYNPUF/OUTPATIENT_CLAIMS/OUTPATIENT_CLAIMS.parquet
lakexpress/synpuf/SYNPUF/CARRIER_CLAIMS/year=2008/month=01/CARRIER_CLAIMS.parquet
lakexpress/synpuf/SYNPUF/CARRIER_CLAIMS/year=2008/month=02/CARRIER_CLAIMS.parquet
...
lakexpress/synpuf/SYNPUF/CARRIER_CLAIMS/year=2010/month=12/CARRIER_CLAIMS.parquet
lakexpress/synpuf/SYNPUF/PRESCRIPTION_DRUG_EVENTS/year=2008/month=01/PRESCRIPTION_DRUG_EVENTS.parquet
lakexpress/synpuf/SYNPUF/PRESCRIPTION_DRUG_EVENTS/year=2008/month=02/PRESCRIPTION_DRUG_EVENTS.parquet
...
lakexpress/synpuf/SYNPUF/PRESCRIPTION_DRUG_EVENTS/year=2010/month=12/PRESCRIPTION_DRUG_EVENTS.parquet
```

```
Export: 5/5 tables succeeded
Total Parquet export - time (s) :  81.54

PUBLISH SUMMARY
============================================================
  Schemas created: 1
  Tables created:  5
  Tables failed:   0
  Rows loaded:     11,494,963

Total publication - time (s) :  39.79
Total elapsed     - time (s) : 121.35
```

The export also benefits from Timepartition: FastBCP splits each large table across 4 parallel Teradata readers, reducing export time from 111s to 82s.

Comparison of per-table publish times (1 node, single file vs. 4 nodes, 36 partitioned files):

| Table                    |      Rows | 1 node (s) | 4 nodes + partitioned (s) | Speedup |
|--------------------------|----------:|-----------:|--------------------------:|--------:|
| BENEFICIARY_SUMMARY      |   343,644 |          8 |                         7 |    1.1x |
| INPATIENT_CLAIMS         |    66,773 |         10 |                         5 |    2.0x |
| OUTPATIENT_CLAIMS        |   790,790 |         22 |                        18 |    1.2x |
| PRESCRIPTION_DRUG_EVENTS | 5,552,421 |         22 |                         9 |    2.4x |
| CARRIER_CLAIMS           | 4,741,335 |         97 |                        38 |    2.6x |

The largest gains are on the two partitioned tables. CARRIER_CLAIMS drops from 97s to 38s: with 36 Parquet files, all 4 Redshift nodes participate in the `COPY` instead of a single node reading a single file.

End-to-end summary:

| Phase     | 1 node, no partition | 4 nodes, partitioned | Speedup  |
|-----------|---------------------:|---------------------:|---------:|
| Export    |                 111s |                  82s |     1.4x |
| Publish   |                  98s |                  40s |     2.5x |
| **Total** |             **209s** |             **121s** | **1.7x** |

## Shutdown

Gracefully shutting down the Teradata VM:

```bash
VBoxManage controlvm "VantageExpress_20.00.28.81_SLES15_20251203053450" acpipowerbutton
```

The Redshift cluster can be paused or deleted with the AWS CLI commands in the [Amazon Redshift cluster](#amazon-redshift-cluster) section.
