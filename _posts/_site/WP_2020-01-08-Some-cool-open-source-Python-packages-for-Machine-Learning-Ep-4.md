
<p align="center">
  <img width="750" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2020-01-08_01/moebius_4.jpg" alt="Moebius">
</p>

There is a very rich ecosystem of Python libraries related to ML. Here is a list of some “active”, open-source packages that may be useful for ML day-to-day activities.

This post is following these ones:

* [Some cool open-source Python packages for Machine Learning EP 1](https://www.architecture-performance.fr/ap_blog/some-cool-open-source-python-packages-for-machine-learning-ep-1/) (2019/07/11)
* [Some cool open-source Python packages for Machine Learning EP 2](https://www.architecture-performance.fr/ap_blog/some-cool-open-source-python-packages-for-machine-learning-ep-2/) (2019/08/08)
* [Some cool open-source Python packages for Machine Learning EP 3](https://www.architecture-performance.fr/ap_blog/some-cool-open-source-python-packages-for-machine-learning-ep-3/) (2019/10/23)

(☞ﾟヮﾟ)☞

## Online learning

* [creme](https://github.com/creme-ml/creme) - a library for online machine learning, also known as incremental learning. Online learning is a machine learning regime where a model learns one observation at a time. This is in contrast to batch learning where all the data is processed in one go. Incremental learning is desirable when the data is too big to fit in memory, or simply when you want to handle streaming data. In addition to many online machine learning algorithms, creme provides utilities for extracting features from a stream of data.

## Auto-ML

* [AutoGluon](https://github.com/awslabs/autogluon) - enables easy-to-use and easy-to-extend AutoML with a focus on deep learning and real-world applications spanning image, text, or tabular data.

## Time series

* [GluonTS](https://github.com/awslabs/gluon-ts/) - a toolkit for probabilistic time series modeling, built around Apache MXNet. GluonTS provides utilities for loading and iterating over time series datasets, state of the art models ready to be trained, and building blocks to define your own models and quickly experiment with different solutions.
* [ADTK](https://github.com/arundo/adtk) - Anomaly Detection Toolkit (ADTK) is a package for unsupervised / rule-based time series anomaly detection.
* [Pmdarima](https://github.com/alkaline-ml/pmdarima) - a statistical library designed to fill the void in Python's time series analysis capabilities, including the equivalent of R's auto.arima function. .

## General machine learning & statistics

* [pomegranate](https://github.com/jmschrei/pomegranate) - pomegranate is a package for building probabilistic models in Python that is implemented in Cython for speed. A primary focus of pomegranate is to merge the easy-to-use API of scikit-learn with the modularity of probabilistic modeling to allow users to specify complicated models without needing to worry about implementation details. The models implemented here are built from the ground up with big data processing in mind and so natively support features like multi-threaded parallelism and out-of-core processing.
* [Mlxtend](https://github.com/rasbt/mlxtend) - a library of extension and helper modules for Python's data analysis and machine learning libraries (initiated and maintained by [Sebastian Raschka](https://sebastianraschka.com/)).
* [Pyglmnet](https://github.com/glm-tools/pyglmnet/) - a python implementation of elastic-net regularized generalized linear models.

## REST API endpoints

* [FastAPI](https://github.com/tiangolo/fastapi) - a modern, fast (high-performance), web framework for building APIs with Python 3.6+ based on standard Python type hints.

## Conversational framework

* [Rasa](https://github.com/RasaHQ/rasa) - a machine learning framework to automate text-and voice-based conversations. With Rasa, you can build contexual assistants on Facebook Messenger, Slack, Microsoft Bot Framework, Rocket.Chat, Mattermost, Telegram, Twilio, your own custom conversational channels... or voice assistants as: Alexa Skills, Google Home Actions.

## NLP

* [Flair](https://github.com/flairNLP/flair) - a very simple framework for state-of-the-art NLP. Developed by Zalando Research.
* [Snips NLU](https://github.com/snipsco/snips-nlu) - Snips NLU (Natural Language Understanding) is a Python library that allows to extract structured information from sentences written in natural language.
* [FastText](https://github.com/facebookresearch/fastText) - an open-source, free, lightweight library that allows users to learn text representations and text classifiers. It works on standard, generic hardware. Models can later be reduced in size to even fit on mobile devices.
* [Kashgari](https://github.com/BrikerMan/Kashgari/) - a simple and powerful NLP Transfer learning framework, build a state-of-art model in 5 minutes for named entity recognition (NER), part-of-speech tagging (PoS), and text classification tasks.

## Graphs

* [DGL](https://github.com/dmlc/dgl) - a package built to ease deep learning on graph, on top of existing DL frameworks.
* [Higra](https://github.com/higra/Higra) - a C++/Python library for efficient sparse graph analysis with a special focus on hierarchical methods.

## Data summarization

* [apricot](https://github.com/jmschrei/apricot) - a package for the greedy selection of diverse subsets of data from massive data sets using submodular selection. 

## Music

* [Spleeter](https://github.com/deezer/spleeter) - Spleeter is the Deezer source separation library with pretrained models written in Python and uses Tensorflow. It makes it easy to train source separation model (assuming you have a dataset of isolated sources), and provides already trained state of the art model for performing various flavour of separation.

## Workflows

* [Metaflow](https://github.com/Netflix/metaflow) - a human-friendly Python library that helps scientists and engineers build and manage real-life data science projects. Metaflow was originally developed at Netflix to boost productivity of data scientists who work on a wide variety of projects from classical statistics to state-of-the-art deep learning.

I like this figure from MetaFlow's [documentation](https://docs.metaflow.org/introduction/what-is-metaflow) about its purpose:

<p align="center">
  <img width="600" src="https://lh5.googleusercontent.com/f68UAR7UxClKR6XvHXq-ZeepDXLHISoVboj6amF9f0cv6jCxqTsjyYnm-isKIHdINMl_gdlPVbR3wJ9vIDwipjMIglHwwetL3-rVNhUTRN4a7UYXmVZ78IqeI2fGJ2OA2Silk5ZS" alt="MetaFlow">
</p>

## Containers

* [repo2docker](https://repo2docker.readthedocs.io/en/latest/) - fetches a git repository and builds a container image based on the configuration files found in the repository.

## Privacy

* [CrypTen](https://github.com/facebookresearch/CrypTen) - a framework for Privacy Preserving Machine Learning built on PyTorch. Its goal is to make secure computing techniques accessible to Machine Learning practitioners.
* [PySyft](https://github.com/OpenMined/PySyft) - a library for secure and private Deep Learning. PySyft decouples private data from model training, using Federated Learning, Differential Privacy, and Multi-Party Computation (MPC) within the main Deep Learning frameworks like PyTorch and TensorFlow. 

## Satellite imagery

* [Raster Vision](https://github.com/azavea/raster-vision) - a framework for building computer vision models on satellite, aerial, and other large imagery sets (including oblique drone imagery).
* [eo-learn](https://github.com/sentinel-hub/eo-learn) - earth observation processing framework for machine learning in Python.
* [sentinelhub](https://github.com/sentinel-hub/sentinelhub-py/) - download and process satellite imagery in Python scripts using Sentinel Hub services.
* [sentinelsat](https://github.com/sentinelsat/sentinelsat) - makes searching, downloading and retrieving the metadata of Sentinel satellite images from the Copernicus Open Access Hub easy.

## Scraper

* [twitterscraper](https://github.com/taspinar/twitterscraper) - a simple script to scrape for Tweets using the Python package requests to retrieve the content and Beautifulsoup4 to parse the retrieved content..

## JupyterLab

* [nbgather](https://github.com/microsoft/gather) - tools for cleaning code, recovering lost code, and comparing versions of code in Jupyter Lab (this is actually not a Python package but a JupyterLab extension).

## Miscellaneous tools

* [Knock Knock](https://github.com/huggingface/knockknock) - a small library to get a notification when your training is complete or when it crashes during the process with two additional lines of code.
* [perfplot](https://github.com/nschloe/perfplot) - perfplot extends Python's `timeit` by testing snippets with input parameters (e.g., the size of an array) and plotting the results.
