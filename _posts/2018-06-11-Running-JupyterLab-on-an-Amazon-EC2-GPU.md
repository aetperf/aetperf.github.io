Following [this](https://blog.keras.io/running-jupyter-notebooks-on-gpu-on-aws-a-starter-guide.html) very usefull Keras blog entry.

* Create an Amazon account if you don't already have one
* Navigate to the EC2 control panel and follow the `launch instance` link
* Select the official `Deep Learning AMI (Ubuntu) Version 10.0`
* Select the `p2.xlarge` instance (Ohio)
* Configure instance details / Configure Security Group : create a custom TCP rule to allow port 8888 and your IP
* If you don't have any connection keys, create some new ones (e.g. `AmazonEC2.pem`) and download them
* Launch your instance by clicking on the appropriate button

Here I had a maximum limit of 0 instance by default, which make it difficult to launch even a single one... I had to request a limit increase from 0 to 1 (this only took a few minutes):

```
Hello,

We have approved and processed your limit increase request(s). It can sometimes take up to 30 minutes for this to propagate and become available for use. I hope this helps, but please reopen this case if you encounter any issues.

Summary of limit(s) requested for increase: [US East (Ohio)]: EC2 Instances / Instance Limit (p2.xlarge), New Limit = 1

Best regards, Amazon Web Services
```

* Once the instance is running, connect to it:
```bash
> ssh -i "AmazonEC2.pem" ubuntu@ec2-18-***-175-210.us-east-2.compute.amazonaws.com
=============================================================================
       __|  __|_  )
       _|  (     /   Deep Learning AMI 10.0 (Ubuntu)
      ___|\___|___|
=============================================================================

Welcome to Ubuntu 16.04.4 LTS (GNU/Linux 4.4.0-1060-aws x86_64v)

Please use one of the following commands to start the required environment with the framework of your choice:
for MXNet(+Keras1) with Python3 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p36
for MXNet(+Keras1) with Python2 (CUDA 9.0 and Intel MKL-DNN) _______________________________ source activate mxnet_p27
for TensorFlow(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p36
for TensorFlow(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _____________________ source activate tensorflow_p27
for Theano(+Keras2) with Python3 (CUDA 9.0) _______________________________________________ source activate theano_p36
for Theano(+Keras2) with Python2 (CUDA 9.0) _______________________________________________ source activate theano_p27
for PyTorch with Python3 (CUDA 9.0 and Intel MKL) ________________________________________ source activate pytorch_p36
for PyTorch with Python2 (CUDA 9.0 and Intel MKL) ________________________________________ source activate pytorch_p27
for CNTK(+Keras2) with Python3 (CUDA 9.0 and Intel MKL-DNN) _________________________________ source activate cntk_p36
for CNTK(+Keras2) with Python2 (CUDA 9.0 and Intel MKL-DNN) _________________________________ source activate cntk_p27
for Caffe2 with Python2 (CUDA 9.0) ________________________________________________________ source activate caffe2_p27
for Caffe with Python2 (CUDA 8.0) __________________________________________________________ source activate caffe_p27
for Caffe with Python3 (CUDA 8.0) __________________________________________________________ source activate caffe_p35
for Chainer with Python2 (CUDA 9.0 and Intel iDeep) ______________________________________ source activate chainer_p27
for Chainer with Python3 (CUDA 9.0 and Intel iDeep) ______________________________________ source activate chainer_p36
for base Python2 (CUDA 9.0) __________________________________________________________________ source activate python2
for base Python3 (CUDA 9.0) __________________________________________________________________ source activate python3
```
* Set up SSL certificates:

```bash
$ mkdir ssl
$ cd ssl
ssl $ sudo openssl req -x509 -nodes -days 365 -newkey rsa:1024 -keyout "cert.key" -out "cert.pem" -batch
ssl $ ls
cert.key  cert.pem
ssl $ cd ..
```

* Generate a password

```python
$ ipython
Python 3.6.4 |Anaconda, Inc.| (default, Jan 16 2018, 18:10:19)
Type 'copyright', 'credits' or 'license' for more information
IPython 6.4.0 -- An enhanced Interactive Python. Type '?' for help.

In [1]: from IPython.lib import passwd

In [2]: passwd()
Enter password:
Verify password:
Out[2]: 'sha1:dbd263***0ae4'

In [3]: exit
```
You need to save the hash of the entered password for later.


* Configure JupyterLab

```bash
$ jupyter lab --generate-config
Overwrite /home/ubuntu/.jupyter/jupyter_notebook_config.py with default config? [y/N]y
Writing default config to: /home/ubuntu/.jupyter/jupyter_notebook_config.py
```
Jupyter config file, where everything is commented out
```bash
$ vi ~/.jupyter/jupyter_notebook_config.py
```
insert the following lines of Python code:
```python
L = get_config()  # get the config object
c.NotebookApp.certfile = u'/home/ubuntu/ssl/cert.pem' # path to the certificate we generated
c.NotebookApp.keyfile = u'/home/ubuntu/ssl/cert.key' # path to the certificate key we generated
c.IPKernelApp.pylab = 'inline'  # in-line figure when using Matplotlib
c.NotebookApp.ip = '*'  # serve the notebooks locally
c.NotebookApp.open_browser = False  # do not open a browser window by default when using notebooks
c.NotebookApp.password = 'sha1:dbd263***0ae4'  #  hash of the password created above
```

* Install whatever you want on the remote machine (or use a predefined environment):

Create a `conda env` and `conda install tensorflow_gpu`, ...

* Set up local port forwarding on your local machine:

```bash
sudo ssh -i AmazonEC2.pem -L 443:127.0.0.1:8888 ubuntu@ec2-18-***-175-210.us-east-2.compute.amazonaws.com
```

* then on the remote machine, start JupyterLab:

```bash
$ mkdir notebooks
$ cd notenooks
$ /notebooks$ jupyter lab
```

* On your local browser, enter the address `https://127.0.0.1/`. You will have a safety warning:

> This warning is simply due to the fact that the SSL certificate we generated isn't verified by any trusted authority (obviously: we just generated our own). Click "advanced" and proceed to navigate, which is safe.

and then Hallelujah!!!

![alt text](/img/2018-06-11_01/jupyter_01.jpg "")


* To stop the instance: go to the `EC2 Management console`, select the instance and then `Actions`, `Instance State`, then `Stop`.
