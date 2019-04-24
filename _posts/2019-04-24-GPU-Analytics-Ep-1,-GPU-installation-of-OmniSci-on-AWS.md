---
title: GPU Analytics Ep 1, GPU installation of OmniSci on AWS
layout: post
author: François Pacull
tags: GPU database OmniSci AWS analytics
---

In this post, we are going to install the OmniSci GPU database on an Ubuntu 18.04 AWS instance. These are the actual command lines I entered when performing the installation. 

But let's start by introducing the motivation behind GPU databases: **GPUs allow fast data analysis**. GPU computing should provide a [1,000X speed-up in 15 or 20 years]((https://www.nvidia.com/en-us/about-nvidia/ai-computing/)) and also offers the ability to visualize the data efficiently. Since the size of the global data-sphere will double between 2019 and 2022 ([from 40 to 80 Zetabytes](https://www.datanami.com/2018/11/27/global-datasphere-to-hit-175-zettabytes-by-2025-idc-says/)), increasing the speed of data ingestion, data analysis, data visualization is a necessity.

## About OmniSci

OmniSci is the company behind the open-source [MapD Core](https://github.com/omnisci/mapd-core) database, which is (quote from their github repository):
> an in-memory, column store, SQL relational database designed from the ground up to run on GPUs. MapD Core is licensed under the Apache License, Version 2.0. 

OmniSci is actually their new brand name, replacing MapD. OmniSci is part of the [GPU Open Analytivs Initiative](http://gpuopenanalytics.com).

## Launch Amazon EC2 Instance

From the AWS console, I launch a new instance (in Ireland in our case).

- Step 1: Choose an Amazon Machine Image (AMI)  
64-bit (x86) Ubuntu Server 18.04 LTS (HVM), EBS General Purpose (SSD) Volume Type

- Step 2: Choose an Instance Type  
GPU instance p3.2xlarge, 8 vCPUs, 61 GiB, 1 GPU NVIDIA Tensor Core V100 with 16 GB of memory.

This GPU has a Volta architecture. The successive NVIDIA generations are the following ones, from oldest to newest: Kepler, Maxwell, Pascal, Volta, Turing. It is important to note that the libraries from [RAPIDS](https://rapids.ai/), the Open GPU Data Science suite, require at least a Pascal architecture, which prevents from choosing a p2 instance (NVIDIA K80 of the Kepler generation), although it is cheaper (0.90 USD per hour instead of 3.06 USD for the p3.2xlarge instance). Also, note that you need to increase the initial AWS limit of p3 instances, which is 0 by default, using the AWS Support Center (it took a few hours in my case).

Regarding the storage size, I asked for 100 GB (we can increase it later if ever we need to).

The next steps deal with network and security... After that, I can eventually start the instance. Once it is running, I can connect via `ssh`:
```bash
$ ssh -i ~/keys/EC2_nopass.pem ubuntu@ec2-34-243-80-83.eu-west-1.compute.amazonaws.com
The authenticity of host 'ec2-34-243-80-83.eu-west-1.compute.amazonaws.com (34.243.80.83)' can't be established.
ECDSA key fingerprint is SHA256:xxxxxxxxxxxxxxxxxxx
Are you sure you want to continue connecting (yes/no)? yes
Warning: Permanently added 'ec2-34-243-80-83.eu-west-1.compute.amazonaws.com,34.243.80.83' (ECDSA) to the list of known hosts.
Welcome to Ubuntu 18.04.2 LTS (GNU/Linux 4.15.0-1032-aws x86_64)
[...]
  System information as of Wed Apr 24 08:20:10 UTC 2019
  System load:  0.0               Processes:           142
  Usage of /:   1.1% of 96.88GB   Users logged in:     0
  Memory usage: 0%                IP address for ens3: 10.0.10.124
  Swap usage:   0%
[...]
ubuntu@ip-10-0-10-124:~$ 
```

## Install NVIDIA drivers

The following installation procedure is taken from the [OmniSci website](https://www.omnisci.com/docs/latest/4_ubuntu-apt-gpu-os-recipe.html).

- Update the entire system

```bash
ubuntu@ip-10-0-10-124:~$ sudo apt update
ubuntu@ip-10-0-10-124:~$ sudo apt upgrade
```

Choose the default options when prompted.


- Install a "headless" Java runtime environment

```bash
ubuntu@ip-10-0-10-124:~$ sudo apt install default-jre-headless
```


- Verify that the apt-transport-https utility is installed

```bash
ubuntu@ip-10-0-10-124:~$ sudo apt install apt-transport-https
```


- Reboot and reconnect
```bash
ubuntu@ip-10-0-10-124:~$ sudo reboot
```


- Create a group called omnisci and a user named omnisci, who will be the owner of the OmniSci database

```bash
ubuntu@ip-10-0-10-124:~$ sudo useradd -U -m omnisci
```


- Verify the System has the correct kernel headers and development packages installed ([NVIDIA reference](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#verify-kernel-packages))

```bash
ubuntu@ip-10-0-10-124:~$ sudo apt-get install linux-headers-$(uname -r)
Reading package lists... Done
Building dependency tree       
Reading state information... Done
linux-headers-4.15.0-1035-aws is already the newest version (4.15.0-1035.37).
linux-headers-4.15.0-1035-aws set to manually installed.
0 upgraded, 0 newly installed, 0 to remove and 0 not upgraded.
```

Eveything is up-to-date.

- Install CUDA Toolkit 10.1 from NVIDIA's [website](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&target_distro=Ubuntu&target_version=1804&target_type=deblocal)

```bash
ubuntu@ip-10-0-10-124:~$ wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
ubuntu@ip-10-0-10-124:~$ sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.105-418.39_1.0-1_amd64.deb
ubuntu@ip-10-0-10-124:~$ sudo apt-key add /var/cuda-repo-10-1-local-10.1.105-418.39/7fa2af80.pub
ubuntu@ip-10-0-10-124:~$ sudo apt-get update
ubuntu@ip-10-0-10-124:~$ sudo apt-get install cuda
ubuntu@ip-10-0-10-124:~$ sudo apt install cuda-drivers linux-image-extra-virtual
```

Choose the default options when prompted.

- Reboot and reconnect

```bash
ubuntu@ip-10-0-10-124:~$ sudo reboot
```


After this step, we can check that the GPU device is ready with the System Management Interface:

```bash
ubuntu@ip-10-0-10-124:~$ nvidia-smi 
Wed Apr 24 09:20:38 2019       
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 418.39       Driver Version: 418.39       CUDA Version: 10.1     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  Tesla V100-SXM2...  On   | 00000000:00:1E.0 Off |                    0 |
| N/A   36C    P0    37W / 300W |      0MiB / 16130MiB |      0%      Default |
+-------------------------------+----------------------+----------------------+
                                                                               
+-----------------------------------------------------------------------------+
| Processes:                                                       GPU Memory |
|  GPU       PID   Type   Process name                             Usage      |
|=============================================================================|
|  No running processes found                                                 |
+-----------------------------------------------------------------------------+
```

## Install OmniSci

- Download and add a GPG key to apt

```bash
ubuntu@ip-10-0-10-124:~$ curl https://releases.omnisci.com/GPG-KEY-omnisci | sudo apt-key add -
  % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                 Dload  Upload   Total   Spent    Left  Speed
100  5290  100  5290    0     0   4588      0  0:00:01  0:00:01 --:--:--  4588
OK
```


- Download the OmniSci list file

```bash
ubuntu@ip-10-0-10-124:~$ echo "deb https://releases.omnisci.com/os/apt/ stable cuda" | sudo tee /etc/apt/sources.list.d/omnisci.list
deb https://releases.omnisci.com/os/apt/ stable cuda
```

- update the packages and install OmniSci

```bash
sudo apt update
sudo apt install omnisci
```

- Set Environment Variables

```bash
ubuntu@ip-10-0-10-124:~$ echo 'export OMNISCI_USER=omnisci' >> ~/.bashrc 
ubuntu@ip-10-0-10-124:~$ echo 'export OMNISCI_GROUP=omnisci' >> ~/.bashrc 
ubuntu@ip-10-0-10-124:~$ echo 'export OMNISCI_STORAGE=/var/lib/omnisci' >> ~/.bashrc 
ubuntu@ip-10-0-10-124:~$ echo 'export OMNISCI_PATH=/opt/omnisci' >> ~/.bashrc 
ubuntu@ip-10-0-10-124:~$ echo 'export OMNISCI_LOG=/var/lib/omnisci/data/mapd_log' >> ~/.bashrc 
ubuntu@ip-10-0-10-124:~$ source ~/.bash
```


Or you can edit the `.bashrc` file manually with your favorite terminal editor.

The `$OMNISCI_STORAGE` directory is where (large) files are going to be stored. It must be dedicated to OmniSci only.

- Initilize OmniSci

```bash
ubuntu@ip-10-0-10-124:~$ cd $OMNISCI_PATH/systemd
ubuntu@ip-10-0-10-124:/opt/omnisci/systemd$ sudo ./install_omnisci_systemd.sh
OMNISCI_PATH: OmniSci install directory
[/opt/omnisci]: 

OMNISCI_STORAGE: OmniSci data and configuration storage directory
[/var/lib/omnisci]: 

OMNISCI_USER: user OmniSci will be run as
[root]: 

OMNISCI_GROUP: group OmniSci will be run as
[root]: 

OMNISCI_PATH:    /opt/omnisci
OMNISCI_STORAGE:    /var/lib/omnisci
OMNISCI_USER:    root
OMNISCI_GROUP:    root
```

- Activate OmniSci

```bash
ubuntu@ip-10-0-10-124:/opt/omnisci/systemd$ cd $OMNISCI_PATH
ubuntu@ip-10-0-10-124:/opt/omnisci$ sudo systemctl start omnisci_server
ubuntu@ip-10-0-10-124:/opt/omnisci$ systemctl status omnisci_server
● omnisci_server.service - OmniSci database server
   Loaded: loaded (/lib/systemd/system/omnisci_server.service; disabled; vendor preset: enabled)
   Active: active (running) since Wed 2019-04-24 09:36:49 UTC; 13s ago
 Main PID: 2840 (omnisci_server)
    Tasks: 25 (limit: 4915)
   CGroup: /system.slice/omnisci_server.service
           ├─2840 /opt/omnisci/bin/omnisci_server --config /var/lib/omnisci/omnisci.conf
           └─2862 -Xmx1024m -DMAPD_LOG_DIR=/var/lib/omnisci/data -jar /opt/omnisci/bin/calcite-1.0-SNAPSHOT-jar-with-dependencies.jar -e /opt/omnisci/QueryEngine/ -d /var/lib/omni

Apr 24 09:36:49 ip-10-0-10-124 systemd[1]: Started OmniSci database server.
lines 1-10/10 (END)
ubuntu@ip-10-0-10-124:/opt/omnisci$  $OMNISCI_PATH/bin/omnisql --version
OmniSQL Version: 4.6.0-20190415-38f897c50e
```

## Test OmniSci

Let's insert some data to check that everything is working well! A convenient script has been installed in order to load some data into the database:


```bash
ubuntu@ip-10-0-10-124:/opt/omnisci$ cd $OMNISCI_PATH
ubuntu@ip-10-0-10-124:/opt/omnisci$ sudo ./insert_sample_data
/opt/omnisci/sample_datasets /opt/omnisci
--2019-04-24 09:52:42--  https://data.mapd.com/manifest.tsv
Resolving data.mapd.com (data.mapd.com)... 72.28.97.165
Connecting to data.mapd.com (data.mapd.com)|72.28.97.165|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 192 [application/octet-stream]
Saving to: ‘manifest.tsv’

manifest.tsv                                 100%[=============================================================================================>]     192  --.-KB/s    in 0s      

2019-04-24 09:52:43 (34.6 MB/s) - ‘manifest.tsv’ saved [192/192]

/opt/omnisci
Enter dataset number to download, or 'q' to quit:
 #     Dataset                   Rows    Table Name             File Name
 1)    Flights (2008)            7M      flights_2008_7M        flights_2008_7M.tar.gz
 2)    Flights (2008)            10k     flights_2008_10k       flights_2008_10k.tar.gz
 3)    NYC Tree Census (2015)    683k    nyc_trees_2015_683k    nyc_trees_2015_683k.tar.gz
1
/opt/omnisci/sample_datasets /opt/omnisci
- downloading and extracting flights_2008_7M.tar.gz
--2019-04-24 09:54:21--  https://data.mapd.com/flights_2008_7M.tar.gz
Resolving data.mapd.com (data.mapd.com)... 72.28.97.165
Connecting to data.mapd.com (data.mapd.com)|72.28.97.165|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 377039522 (360M) [application/octet-stream]
Saving to: ‘flights_2008_7M.tar.gz’

flights_2008_7M.tar.gz                       100%[=============================================================================================>] 359.57M  3.49MB/s    in 93s     

2019-04-24 09:55:54 (3.88 MB/s) - ‘flights_2008_7M.tar.gz’ saved [377039522/377039522]

flights_2008_7M/
flights_2008_7M/flights_2008_7M.csv
flights_2008_7M/flights_2008_7M.sql
/opt/omnisci
- adding schema
User mapd connected to database mapd
User mapd disconnected from database mapd
- inserting file: /opt/omnisci/sample_datasets/flights_2008_7M/flights_2008_7M.csv
User mapd connected to database mapd
Result
Loaded: 7009728 recs, Rejected: 0 recs in 46.885000 secs
User mapd disconnected from database mapd

```


We now connect to OmniSci Core (default password is `HyperInteractive`): 

```bash
ubuntu@ip-10-0-10-124:/opt/omnisci$ $OMNISCI_PATH/bin/omnisql
password: HyperInteractive
User mapd connected to database mapd
omnisql> SELECT * FROM flights_2008_7M LIMIT 2;
flight_year|flight_month|flight_dayofmonth|flight_dayofweek|deptime|crsdeptime|arrtime|crsarrtime|uniquecarrier|flightnum|tailnum|actualelapsedtime|crselapsedtime|airtime|arrdelay|depdelay|origin|dest|distance|taxiin|taxiout|cancelled|cancellationcode|diverted|carrierdelay|weatherdelay|nasdelay|securitydelay|lateaircraftdelay|dep_timestamp|arr_timestamp|carrier_name|plane_type|plane_manufacturer|plane_issue_date|plane_model|plane_status|plane_aircraft_type|plane_engine_type|plane_year|origin_name|origin_city|origin_state|origin_country|origin_lat|origin_lon|dest_name|dest_city|dest_state|dest_country|dest_lat|dest_lon|origin_merc_x|origin_merc_y|dest_merc_x|dest_merc_y
2008|1|10|4|702|700|758|800|WN|1|N505SW|56|60|46|-2|2|DAL|HOU|239|2|8|0|NULL|0|NULL|NULL|NULL|NULL|NULL|2008-01-10 07:02:00|2008-01-10 07:58:00|Southwest Airlines|Corporation|BOEING|1990-05-14|737-5H4|Valid|Fixed Wing Multi-Engine|Turbo-Jet|1990|Dallas Love|Dallas|TX|USA|32.84711|-96.85177|William P Hobby|Houston|TX|USA|29.64542|-95.27889|-1.078149e+07|3875028|-1.06064e+07|3458053
2008|1|10|4|730|730|831|830|WN|3|N642WN|61|60|48|1|0|DAL|HOU|239|3|10|0|NULL|0|NULL|NULL|NULL|NULL|NULL|2008-01-10 07:30:00|2008-01-10 08:31:00|Southwest Airlines|Corporation|BOEING|1997-02-11|737-3H4|Valid|Fixed Wing Multi-Engine|Turbo-Fan|1997|Dallas Love|Dallas|TX|USA|32.84711|-96.85177|William P Hobby|Houston|TX|USA|29.64542|-95.27889|-1.078149e+07|3875028|-1.06064e+07|3458053
omnisql> SELECT origin_city AS "Origin", dest_city AS "Destination", AVG(airtime) AS "Average Airtime" FROM flights_2008_7M WHERE distance < 200 GROUP BY origin_city, dest_city LIMIT 10;
Origin|Destination|Average Airtime
Jacksonville|Tampa|37.80867850098619
Orlando|Tampa|19
West Palm Beach|Tampa|35.51176470588236
Ft. Lauderdale|Tampa|38.19741636303412
Sarasota|Tampa|19.5
Norfolk|Baltimore|37.0005980861244
Philadelphia|Baltimore|24.73684210526316
Newark|Baltimore|37.52857142857143
New York|Baltimore|39.21726755218216
Harrisburg|Baltimore|25

```

Well, it seems that the database is working. Next we will see how to install the RAPIDS suite and load some data from OmniSci to dataframes.
