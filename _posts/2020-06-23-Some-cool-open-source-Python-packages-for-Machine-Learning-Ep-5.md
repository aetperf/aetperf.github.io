---
title: Some cool open-source Python packages for Machine Learning Ep 5
layout: post
comments: true
author: François Pacull
tags: Python open source machine learning packages
---


<p align="center">
  <img width="800" src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Sombrero_Galaxy_in_infrared_light_%28Hubble_Space_Telescope_and_Spitzer_Space_Telescope%29.jpg/1920px-Sombrero_Galaxy_in_infrared_light_%28Hubble_Space_Telescope_and_Spitzer_Space_Telescope%29.jpg" alt="Moebius">    
</p>

*Image credit: NASA/JPL-Caltech and The Hubble Heritage Team (STScI/AURA)*


_  


There is a very rich ecosystem of Python libraries related to ML. Here is a list of some “active”, open-source packages that may be useful for ML day-to-day activities.

Previous post list:

* [Some cool open-source Python packages for Machine Learning EP 1](https://aetperf.github.io/2019/07/11/Some-cool-open-source-Python-packages-for-Machine-Learning.html) (2019/07/11)
* [Some cool open-source Python packages for Machine Learning EP 2](https://aetperf.github.io/2019/08/08/Some-cool-open-source-Python-packages-for-Machine-Learning-Ep-2.html) (2019/08/08)
* [Some cool open-source Python packages for Machine Learning EP 3](https://aetperf.github.io/2019/10/23/Some-cool-open-source-Python-packages-for-Machine-Learning-Ep-3.html) (2019/10/23)
* [Some cool open-source Python packages for Machine Learning EP 4](https://aetperf.github.io/2020/01/08/Some-cool-open-source-Python-packages-for-Machine-Learning-Ep-4.html) (2020/01/08)

(☞ﾟヮﾟ)☞

_


## Recommender systems

* [Surprise](https://github.com/NicolasHug/Surprise) - a scikit building and analyzing recommender systems that deal with explicit rating data.

## Chat-bots

* [DeepPavlov](https://github.com/deepmipt/DeepPavlov) - conversational AI library built on TensorFlow and Keras. DeepPavlov is designed for
    - development of production ready chat-bots and complex conversational systems,
    - research in the area of NLP and, particularly, of dialog systems.   
  
* [ParlAI](https://github.com/facebookresearch/ParlAI) - a framework for training and evaluating AI models on a variety of openly available dialogue datasets. 

## Causal Inference, probabilistic ML

* [Causal ML](https://github.com/uber/causalml) - a package that provides a suite of uplift modeling and causal inference methods using machine learning algorithms based on recent research. It provides a standard interface that allows user to estimate the Conditional Average Treatment Effect (CATE) or Individual Treatment Effect (ITE) from experimental or observational data. Essentially, it estimates the causal impact of intervention T on outcome Y for users with observed features X, without strong assumptions on the model form. Here is a quote from [Uber Engineering Blog](https://eng.uber.com/causal-inference-at-uber/):

> At a more granular level, causal inference enables data scientists and product analysts to answer causal questions based on observational data, especially when A/B testing is not possible, or gain additional insights from a well-designed experiment. For example, we may launch an email campaign that is open for participation to all customers in a market. In this case, since we don’t have a randomized control group, how do we measure the campaign’s effect? In another example, suppose we have a randomized, controlled A/B test experiment but not everyone in the treatment group actually receive the treatment (i.e., if they don’t open the email). How do we estimate the treatment effect for the treated? Causal inference enables us to answer these types of questions, leading to better user experiences on our platform.

* [CausalNex](https://github.com/quantumblacklabs/causalnex) - a library that helps data scientists to infer causation rather than observing correlation. 

* [PyMC3](https://github.com/pymc-devs/pymc3) - a package for Bayesian statistical modeling and Probabilistic Machine Learning focusing on advanced Markov Chain Monte Carlo (MCMC) and Variational Inference (VI) algorithms. 

## Data engineering

* [Great Expectations](https://github.com/great-expectations/great_expectations) - Great Expectations helps data teams eliminate pipeline debt, through data testing, documentation, and profiling. Software developers have long known that testing and documentation are essential for managing complex codebases. Great Expectations brings the same confidence, integrity, and acceleration to data science and data engineering teams.

## Feature engineering

* [Feature-engine](https://github.com/solegalli/feature_engine) - a library with multiple transformers to engineer features for use in machine learning models. Feature-engine's transformers follow Scikit-learn functionality with fit() and transform() methods to first learn the transforming paramenters from data and then transform the data.

* [LOFO Importance](https://github.com/aerdem4/lofo-importance) - LOFO (Leave One Feature Out) Importance calculates the importances of a set of features based on a metric of choice, for a model of choice, by iteratively removing each feature from the set, and evaluating the performance of the model, with a validation scheme of choice, based on the chosen metric.

* [xfeat](https://github.com/pfnet-research/xfeat) - flexible Feature Engineering & Exploration Library using GPUs and [Optuna](https://github.com/optuna/optuna).

## Data Annotation

* [PigeonXT](https://github.com/dennisbakhuis/pigeonXT) -  an extention to the original [Pigeon](https://github.com/agermanidis/pigeon), created by [Anastasis Germanidis](https://pypi.org/user/agermanidis/). PigeonXT is a simple widget that lets you quickly annotate a dataset of unlabeled examples from the comfort of your Jupyterlab notebook. 

## Data visualization

* [missingno](https://github.com/ResidentMario/missingno) - a small toolset of flexible and easy-to-use missing data visualizations and utilities that allows you to get a quick visual summary of the completeness (or lack thereof) of your dataset.

## Privacy & Security

* [FATE](https://github.com/FederatedAI/FATE) - a project initiated by Webank's AI Department to provide a secure computing framework to support the federated AI ecosystem. It implements secure computation protocols based on homomorphic encryption and multi-party computation (MPC). It supports federated learning architectures and secure computation of various machine learning algorithms, including logistic regression, tree-based algorithms, deep learning and transfer learning. Definition of federated learning from [Wikipedia](https://en.wikipedia.org/wiki/Federated_learning):

> Federated learning (aka collaborative learning) is a machine learning technique that trains an algorithm across multiple decentralized edge devices or servers holding local data samples, without exchanging their data samples. This approach stands in contrast to traditional centralized machine learning techniques where all data samples are uploaded to one server, as well as to more classical decentralized approaches which assume that local data samples are identically distributed.

> Federated learning enables multiple actors to build a common, robust machine learning model without sharing data, thus addressing critical issues such as data privacy, data security, data access rights and access to heterogeneous data. Its applications are spread over a number of industries including defense, telecommunications, IoT, or pharmaceutics. 

## Ensemble learning

* [DESlib](https://github.com/scikit-learn-contrib/DESlib) - an easy-to-use ensemble learning library focused on the implementation of the state-of-the-art techniques for dynamic classifier and ensemble selection. The library is is based on scikit-learn, using the same method signatures: fit, predict, predict_proba and score. 

* [pycobra](https://github.com/bhargavvader/pycobra) - a library for ensemble learning. It serves as a toolkit for regression and classification using these ensembled machines, and also for visualisation of the performance of the new machine and constituent machines. 

* [combo](https://github.com/yzhao062/combo) - a toolbox for machine Learning model combination. Model combination can be considered as a subtask of ensemble learning, and has been widely used in real-world tasks and data science competitions like Kaggle.

## NLP

* [AdaptNLP](https://github.com/Novetta/adaptnlp) - a high level framework and library for running, training, and deploying state-of-the-art Natural Language Processing (NLP) models for end to end tasks.

* [LexNLP](https://github.com/LexPredict/lexpredict-lexnlp) - a library for working with real, unstructured legal text, including contracts, plans, policies, procedures, and other material.

* [AllenNLP](https://github.com/allenai/allennlp) - a NLP research library, built on PyTorch, for developing state-of-the-art deep learning models on a wide variety of linguistic tasks. 

* [Tokenizers](https://github.com/huggingface/tokenizers) - an implementation of today's most used tokenizers, with a focus on performance and versatility.

* [Emot](https://github.com/NeelShah18/emot) - a library to extract the emojis and emoticons from a text (string).

* [FlashText](https://github.com/vi3k6i5/flashtext) - this package can be used to replace keywords in sentences or extract keywords from sentences. It is based on the [FlashText](https://arxiv.org/abs/1711.00046) algorithm.

* [aitextgen](https://github.com/minimaxir/aitextgen) - a robust tool for text-based AI training and generation using GPT-2, based on [PyTorch](https://pytorch.org/), [Hugging Face Transformers](https://github.com/huggingface/transformers) and [pytorch-lightning](https://github.com/PyTorchLightning/pytorch-lightning).

* [numerizer](https://github.com/jaidevd/numerizer) - a module to convert natural language numerics into ints and floats.

## Computer vision

* [Kornia](https://github.com/kornia/kornia) - Kornia is a differentiable computer vision library for PyTorch. It consists of a set of routines and differentiable modules to solve generic computer vision problems. At its core, the package uses PyTorch as its main backend both for efficiency and to take advantage of the reverse-mode auto-differentiation to define and compute the gradient of complex functions.

## Time series

* [tslearn](https://github.com/tslearn-team/tslearn) - a package that provides machine learning tools for the analysis of time series.

* [Seglearn](https://github.com/dmbee/seglearn) - a package for machine learning time series or sequences. It provides an integrated pipeline for segmentation, feature extraction, feature processing, and final estimator. 

## Pipeline frameworks

* [DBND](https://github.com/databand-ai/dbnd) - a framework for building and tracking data pipelines. DBND is used for processes ranging from data ingestion, preparation, machine learning model training and production.

## Deep learning frameworks

* [Haste](https://github.com/lmnt-com/haste) - a CUDA implementation of fused RNN layers with built-in DropConnect and Zoneout regularization. Haste includes a standalone C++ API, a TensorFlow Python API, and a PyTorch API .

* [MindSpore](https://github.com/mindspore-ai/mindspore) - a deep learning training/inference framework that could be used for mobile, edge and cloud scenarios. MindSpore is designed to provide development experience with friendly design and efficient execution for the data scientists and algorithmic engineers, native support for Ascend AI processor, and software hardware co-optimization.

## Framework wrapper

* [PyTorch Lightning](https://github.com/PyTorchLightning/pytorch-lightning) - lightweight PyTorch wrapper for ML researchers, high-level interface.

* [Sonnet](https://github.com/deepmind/sonnet) - a library built on top of TensorFlow 2 designed to provide simple, composable abstractions for machine learning research.

* [Thinc](https://github.com/explosion/thinc) - a lightweight deep learning library that offers an elegant, type-checked, functional-programming API for composing models, with support for layers defined in other frameworks such as PyTorch, TensorFlow and MXNet.

## Hyper-parameter optimization

* [Nevergrad](https://github.com/facebookresearch/nevergrad) - a toolbox for performing gradient-free optimization. The library includes a wide range of optimizers, such as:
    - Differential evolution,
    - Sequential quadratic programming,
    - FastGA,
    - Covariance matrix adaptation,
    - Population control methods for noise management,
    - Particle swarm optimization.  
  
* [HiPlot](https://github.com/facebookresearch/hiplot) - a lightweight interactive visualization tool to help AI researchers discover correlations and patterns in high-dimensional data using parallel plots and other graphical ways to represent information.

## Auto-ML

* [NNI](https://github.com/microsoft/nni) - a toolkit for automate machine learning lifecycle, including feature engineering, neural architecture search, model compression and hyper-parameter tuning.

## Tensor computing

* [Hummingbird](https://github.com/microsoft/hummingbird) - a library for compiling trained traditional ML models into tensor computations. Hummingbird allows users to seamlessly leverage neural network frameworks (such as [PyTorch](https://pytorch.org/)) to accelerate traditional ML models

## Automatic differentiation

* [JAX](https://github.com/google/jax) - JAX is [Autograd](https://github.com/hips/autograd) and [XLA](https://www.tensorflow.org/xla), brought together for high-performance machine learning research. With its updated version of [Autograd](https://github.com/hips/autograd), JAX can automatically differentiate native Python and NumPy functions. It can differentiate through loops, branches, recursion, and closures, and it can take derivatives of derivatives of derivatives. It supports reverse-mode differentiation (a.k.a. backpropagation) via grad as well as forward-mode differentiation, and the two can be composed arbitrarily to any order. What’s new is that JAX uses [XLA](https://www.tensorflow.org/xla) to compile and run your NumPy programs on GPUs and TPUs. 

## Open science collaboration

* [OpenML-Python](https://github.com/openml/openml-python) - an interface for [OpenML](https://www.openml.org/), an online platform for open science collaboration in machine learning. It can be used to download or upload OpenML data such as datasets and machine learning experiment results.


## ------ NOT PYTHON, but with Python support ------

## Workflows

* [Flyte](https://github.com/lyft/flyte) - a container-native, type-safe workflow and pipelines platform optimized for large scale processing and machine learning written in Golang. Workflows can be written in any language, with out of the box support for Python.

## Experimentation frameworks

* [Katib](https://github.com/kubeflow/katib) - a Kubernetes-based system for Hyperparameter Tuning and Neural Architecture Search. Katib supports a number of ML frameworks, including TensorFlow, Apache MXNet, PyTorch, XGBoost, and others. Note that there is a Python [SDK](https://github.com/kubeflow/katib/tree/master/sdk/python).


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
