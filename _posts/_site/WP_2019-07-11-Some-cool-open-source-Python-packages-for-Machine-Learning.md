
<p align="center">
  <img width="600" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2019-07-11_01/moebius_01.jpg" alt="Moebius">
</p>


There is a very rich ecosystem of Python libraries related to ML. Here is a list of some "active", open-source packages that may be useful for ML day-to-day activities. Of course, this list is far from being exhaustive and should evolve as fast as the Python ecosystem does. Also, we exclude from the current list: 
* main ML algorithm frameworks ([scikit-learn](https://github.com/scikit-learn/scikit-learn), [LightGBM](https://github.com/microsoft/LightGBM), [PyTorch](https://github.com/pytorch/pytorch), ...), 
* famous user-friendly libraries built on top of deep-learning libraries ([fastai](https://github.com/fastai/fastai), [Keras](https://github.com/keras-team/keras), ...), 
* specific application-oriented libraries ([spaCy](https://github.com/explosion/spaCy), [scikit-image](https://github.com/scikit-image/scikit-image), [StellarGraph](https://github.com/stellargraph/stellargraph), ...),
* packages dealing with the general data/analytics environment ([JupyterLab](https://github.com/jupyterlab/jupyterlab), [Pandas](https://github.com/pandas-dev/pandas), [Dask](https://github.com/dask/dask), [Conda](https://github.com/conda/conda), ...) that are also used in many other domains, even if some of the following tools are more on the data-engineering side than on the ML one.

We hope you will find this list informative!

## Data cleaning

* [Pyjanitor](https://github.com/ericmjl/pyjanitor) - a clean API for cleaning data.

## Auto-ML

* [Featuretools](https://github.com/Featuretools/featuretools) - a library for automated feature engineering.
* [TPOT](https://github.com/EpistasisLab/tpot) - an automated tool that optimizes ML pipelines using genetic programming.
* [Scikit-Optimize](https://github.com/scikit-optimize/scikit-optimize) - a simple and efficient library to minimize expensive and noisy black-box functions.
* [Randopt](https://github.com/seba-1511/randopt) - a package for ML experiment management, hyper-parameter optimization, and results visualization.
* [Optuna](https://github.com/pfnet/optuna/) - an automatic hyper-parameter optimization software framework, particularly designed for ML.

## Dimension reduction and visualization

* [UMAP](https://github.com/lmcinnes/umap) - Uniform Manifold Approximation and Projection is a dimension reduction technique that can be used for visualization similarly to t-SNE, but also for general non-linear dimension reduction.

## Model analysis

* [ELI5](https://github.com/TeamHG-Memex/eli5) - a library which allows to visualize and debug various ML models using unified API.
* [Yellowbrick](https://github.com/DistrictDataLabs/yellowbrick) - a suite of visual diagnostic tools called "Visualizers" that extend the scikit-learn API to allow human steering of the model selection process.
* [SHAP](https://github.com/slundberg/shap) - SHapley Additive exPlanations is a unified approach to explain the output of any ML model.

## Experimentation frameworks and tools

* [Guild AI](https://github.com/guildai/guildai) - a toolkit that automates and optimizes ML experiments.
* [ModelChimp](https://github.com/ModelChimp/modelchimp) - an experiment tracker for Deep Learning and ML experiments.
* [Sacred](https://github.com/IDSIA/sacred) - a tool to help you configure, organize, log and reproduce experiments.
* [SKLL](https://github.com/EducationalTestingService/skll) - SciKit-Learn Laboratory provides command-line utilities to make it easier to run ML experiments with scikit-learn.
* [DVC](https://github.com/iterative/dvc) - Data Version Control is a tool for data science and ML projects.

## Model export

* [ONNXMLTools](https://github.com/onnx/onnxmltools) - enables you to convert models from different ML toolkits into [ONNX](https://onnx.ai/) (Open Neural Network Exchange)

## Worflows

* [MLflow](https://github.com/mlflow/mlflow/) - a platform to streamline ML development, including tracking experiments, packaging code into reproducible runs, and sharing and deploying models.
* [Kubeflow](https://github.com/kubeflow/kubeflow) - a Cloud Native platform for ML based on Google's internal ML pipelines.

## Data pipelines

* [Kedro](https://github.com/quantumblacklabs/kedro) - a workflow development tool that helps you build data pipelines that are robust, scalable, deployable, reproducible and versioned.
* [Dagster](https://github.com/dagster-io/dagster) - a system for building modern data applications.
