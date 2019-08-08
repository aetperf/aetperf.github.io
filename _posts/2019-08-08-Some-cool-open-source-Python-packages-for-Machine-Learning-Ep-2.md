---
title: Some cool open-source Python packages for Machine Learning Ep 2
layout: post
comments: true
author: François Pacull
tags: Python open source machine learning packages
---

![moebius](/img/2019-08-08_01/moebius.jpg)

There is a very rich ecosystem of Python libraries related to ML. Here is a list of some “active”, open-source packages that may be useful for ML day-to-day activities.

This post is following that one:

* [Some cool open-source Python packages for Machine Learning EP 1](https://aetperf.github.io/2019/07/11/Some-cool-open-source-Python-packages-for-Machine-Learning.html) (2019/07/11)

(☞ﾟヮﾟ)☞

## Database connectivity

* [Turbodbc](https://github.com/blue-yonder/turbodbc) - a module to access relational databases via the Open Database Connectivity (ODBC) interface.  
* [ibis](https://github.com/ibis-project/ibis) - a toolbox to bridge the gap between local Python environments, remote storage, execution systems like Hadoop components (HDFS, Impala, Hive, Spark) and SQL databases. 

## Data description

* [Pandas Profiling](https://github.com/pandas-profiling/pandas-profiling) - Generates profile reports from a pandas DataFrame.

## Data preparation

* [Snorkel](https://github.com/HazyResearch/snorkel) - a system for quickly generating training data with weak supervision.  
* [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) - a package to Tackle the Curse of Imbalanced Datasets in Machine Learning

## Feature engineering

* [dirty_cat](https://github.com/dirty-cat/dirty_cat/) - dirty cat helps with machine-learning on non-curated categories, by providing encoders that are robust to morphological variants, such as typos, in the category strings.

## Dimension reduction

* [ivis](https://github.com/beringresearch/ivis) - a machine learning algorithm for reducing dimensionality of very large datasets. 

## Auto-ML

* [auto-sklearn](https://github.com/automl/auto-sklearn) - an automated machine learning toolkit and a drop-in replacement for a scikit-learn estimator.
* [Auto-Keras](https://github.com/keras-team/autokeras) - an open source software library for automated machine learning.  
* [Keras Tuner](https://github.com/keras-team/keras-tuner) - An hyperparameter tuner for Keras.

## Model analysis

* [Skater](https://github.com/oracle/Skater) - a unified framework to enable Model Interpretation for all forms of model.

## Workflow management

* [prefect](https://github.com/PrefectHQ/prefect) - a workflow management system, designed for modern infrastructure and powered by the open-source Prefect Core workflow engine. 
* [papermill](https://github.com/nteract/papermill) - a tool for parameterizing, executing, and analyzing Jupyter Notebooks.

## Model management

* [Studio](https://github.com/studioml/studio) - a model management framework written in Python to help simplify and expedite your model building experience. 

## Data visualization

* [kepler.gl](https://github.com/keplergl/kepler.gl) - a powerful open source geospatial analysis tool for large-scale data sets **with a jupyter widget** to render large-scale interactive maps in Jupyter Notebook.  
* [glue](https://glueviz.org/) - a library to explore relationships within and among related datasets.
* [KeplerMapper](https://github.com/scikit-tda/kepler-mapper) - an implementation of the TDA Mapper algorithm for visualization of high-dimensional data.

## Models

* [pytorch-transformers](https://github.com/huggingface/pytorch-transformers) - a library of state-of-the-art pre-trained models for Natural Language Processing (NLP).  
* [spacy-pytorch-transformers](https://github.com/explosion/spacy-pytorch-transformers) - provides spaCy model pipelines that wrap Hugging Face's pytorch-transformers package, so you can use them in spaCy.

## Time series

* [STUMPY](https://github.com/TDAmeritrade/stumpy) - a powerful and scalable library that can be used for a variety of time series data mining tasks.


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