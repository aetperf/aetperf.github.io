---
title: Uncompress a 7-zip file dropped on S3 with a Lambda function IN PROGRESS
layout: post
comments: true
author: François Pacull
tags: Python AWS Lambda S3 CloudWatch 7-Zip Uncompress
---


We want an AWS Lambda function to be triggered when a file is dropped on a given S3 bucket, and this function to uncompress the dropped file if it is a 7-Zip archive.

## Create a S3 bucket

Firstly, let's go to the AWS management console (I used my default profile), look for the S3 service and create a bucket. We create a bucket called *lambda-7z-test*:

![General configuration](./img/2021-03-25_01/2021-03-25_11-45_1.png)

We block all public access to this bucket, disable bucket versioning, and configure some kind of encryption:

![Default encryption](./img/2021-03-25_01/2021-03-25_11-46_2.png)

Note that we will be using the same bucket as input and output. This is not recommended when using a S3 "create file" trigger for the lambda function. However, we will only trigger the lambda function on files with a specific *7z* extension and be careful not to output 7zip files from the lambda function.

## Create a role

Now we look for the IAM service, click on Roles and Create a new role. We choose the Lambda use case:

![Create role](./img/2021-03-25_01/2021-03-25_14-45.png)

We give a name to this role, *lambda_s3_7z* and configure some polices:

- S3 full access
- Lambda full access
- CloudWatch full access (used to monitor the lambda function executions)

![Create role review](./img/2021-03-25_01/2021-03-25_14-49.png)

## Create a Lambda function

We now look for the Lambda service on the management console and choose to create a new lambda, in Python 3.7 with the *lambda_s3_7z* existing role:

![Create lambda](./img/2021-03-25_01/2021-03-25_14-50.png)

We call this lambda function *uncompress_7z*, and create a S3 trigger:

![Add trigger](./img/2021-03-25_01/2021-03-25_14-53.png)

The lambda function will be triggered when an object ending with *7z*  is created in the *lambda-7z-test* bucket. Also, it's important to configure the lambda function in order to increase the default timeout (3 s) and RAM (128 MB):

![Default config](./img/2021-03-25_01/2021-03-25_16-58_1.png)

You can edit these parameters, increase the RAM up to 10240 MB. Increasing the timeout is important if you provide an archive with all the Python virtual environment, and is the process is the function is a long one. Also make sure to change the handler in the Runtime settings if ever you change it; this is the entry point in the Python code:

 ![Default config](./img/2021-03-25_01/2021-03-25_17-16.png)

As a first version, we use this function:

```python
import os
import sys
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def lambda_handler(event, context):
    logger.info('lambda_handler IN')
    records = event.get('Records', [])
    for record in records:
        bucket = record['s3']['bucket']['name']
        print("bucket : ", bucket)
    logger.info('lambda_handler OUT')
```

This only uses the standard Python library, so does not need to install any package. In order to uncompress the *7z* file, we will need to install some packages, later on.


## Test the trigger

First we create a working directory:

```bash
(lambdas) $  mkdir lambda-7z-s3
(lambdas) $  cd lambda-7z-s3
(lambdas) lambda-7z-s3/ $
```

We download [this](https://www.gutenberg.org/files/2488/2488-h/) directory from the [gutenberg project](https://www.gutenberg.org/ebooks/2488), created a *tar.7z* file and uploaded it to the s3 bucket :

```bash 
lambda-7z-s3/ $ ls sample_7z_file 
2488-h.tar.7z
lambda-7z-s3/ $ aws s3 cp ./sample_7z_file/2488-h.tar.7z s3://lambda-7z-test/
upload: sample_7z_file/2488-h.tar.7z to s3://lambda-7z-test/2488-h.tar.7z
```

We can check that the 7z file is now on the s3 bucket:

```bash 
lambda-7z-s3/ $ aws s3 ls s3://lambda-7z-test/                 
2021-03-25 17:21:55    8702187 2488-h.tar.7z
```

Also, we can check that the Lambda function was triggered on CloudWatch events (CloudWatch > CloudWatch Logs > Log groups > /aws/lambda/uncompress_7z):

```
START RequestId: 57544b05-6a85-43fb-9268-c1e5c3209867 Version: $LATEST
[INFO]	2021-03-25T16:28:31.564Z	57544b05-6a85-43fb-9268-c1e5c3209867	lambda_handler IN
bucket :  lambda-7z-test
[INFO]	2021-03-25T16:28:31.564Z	57544b05-6a85-43fb-9268-c1e5c3209867	lambda_handler OUT
END RequestId: 57544b05-6a85-43fb-9268-c1e5c3209867
REPORT RequestId: 57544b05-6a85-43fb-9268-c1e5c3209867	Duration: 1.50 ms	Billed Duration: 2 ms	Memory Size: 1024 MB	Max Memory Used: 49 MB	Init Duration: 110.67 ms	
```


Here the kind of record received by the Lambda function:

```json
{
   "eventVersion":"2.1",
   "eventSource":"aws:s3",
   "awsRegion":"eu-west-1",
   "eventTime":"2021-03-25T15:27:23.331Z",
   "eventName":"ObjectCreated:CompleteMultipartUpload",
   "userIdentity":{
      "principalId":"AWS:AIDA****4IS5Q"
   },
   "requestParameters":{
      "sourceIPAddress":"78.203.80.175"
   },
   "responseElements":{
      "x-amz-request-id":"8Z7AZ**JT7",
      "x-amz-id-2":"7ldXArajlyXM3jTX6****O7OA/V6QyG4IQi"
   },
   "s3":{
      "s3SchemaVersion":"1.0",
      "configurationId":"9809f6b5-f4c0**ae12d5",
      "bucket":{
         "name":"lambda-7z-test",
         "ownerIdentity":{
            "principalId":"A18LC**NEB"
         },
         "arn":"arn:aws:s3:::lambda-7z-test"
      },
      "object":{
         "key":"2488-h.tar.7z",
         "size":8702187,
         "eTag":"1d5cbc**f512f0-2",
         "sequencer":"0060**795"
      }
   }
}
```

We can remove the *7z* file from S3 from the command line:

```bash
lambda-7z-s3/ $ aws s3 rm s3://lambda-7z-test/2488-h.tar.7z                  
delete: s3://lambda-7z-test/2488-h.tar.7z
```


References:

- https://github.com/pimlock/s3-uncompressor-sam  
- https://www.tutorialspoint.com/aws_lambda/aws_lambda_using_lambda_function_with_amazon_s3.htm