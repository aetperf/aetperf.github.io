
<p align="center">
  <img width="600" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2019-08-09_01/moebius.jpg" alt="Moebius">
</p>

Since we are big users of *scikit-learn* and *XGBoost*, we wanted to try a package that would automate the process of building a machine learning model with these tools. Here is the introduction to *auto-sklearn* from its [github.io](https://automl.github.io/auto-sklearn/master/index.html) website:

> *auto-sklearn* is an automated machine learning toolkit and a drop-in replacement for a *scikit-learn* estimator. It leverages recent advantages in Bayesian optimization, meta-learning and ensemble construction. 

Here is the wikipedia definition of [AutoML](https://en.wikipedia.org/wiki/Automated_machine_learning):

> Automated machine learning (AutoML) is the process of automating end-to-end the process of applying machine learning to real-world problems. In a typical machine learning application, practitioners have a dataset consisting of input data points to train on. The raw data itself may not be in a form that all algorithms may be applicable to it out of the box. An expert may have to apply the appropriate data pre-processing, feature engineering, feature extraction, and feature selection methods that make the dataset amenable for machine learning. Following those preprocessing steps, practitioners must then perform algorithm selection and hyperparameter optimization to maximize the predictive performance of their final machine learning model. As many of these steps are often beyond the abilities of non-experts, AutoML was proposed as an artificial intelligence-based solution to the ever-growing challenge of applying machine learning. 


The theory behind *auto-sklearn* package is presented in the paper published at [NIPS2015](http://papers.nips.cc/paper/5872-efficient-and-robust-automated-machine-learning.pdf) (to be honest, I did not read the article because I just wanted to give `auto-sklearn` a very quick try with some very basic examples).

Here are the main features of the API. You can:
* set time and memory limits
* restrict the searchspace by selecting or excluding some preprocessing methods or some estimators
* specify some resampling strategies (e.g. 5-fold cv)
* perform some parallel computation with the SMAC algorithm (sequential model-based algorithm configuration) that stores some data on a shared file system
* save your model as you would do with *scikit-learn* (`pickle`)

We are going to try it on two toy datasets for **classification** algorithms:  
* hand-written digits  
* breast cancer  

## Installation

At first we had a *Segmentation fault* when running the model. We found some useful information on [github](https://github.com/automl/auto-sklearn/issues/688), and here are the steps performed to install `auto-sklearn` (0.5.2) on an Ubuntu 18.04 OS, in a python 3.7 `conda` environment:

```bash
$ conda create -n autosklearn python=3.7
$ source activate autosklearn
(autosklearn) $ conda install -c conda-forge 'swig<4'
(autosklearn) $ conda install gxx_linux-64 gcc_linux-64
(autosklearn) $ curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install
(autosklearn) $ pip install 'pyrfr>=0.8'
(autosklearn) $ pip install auto-sklearn
```

Note that the *scikit-learn* version associated with *auto-sklearn* is 0.19.2 (latest is 0.21.3).

## First data set

[This](https://archive.ics.uci.edu/ml/datasets/optical+recognition+of+handwritten+digits) is a description of the UCI ML hand-written digits dataset: each datapoint is a 8x8 image of a digit. However we only have a subset (copy of the test set) included in *scikit-learn*. So we can say that it is rather small (table from *scikit-lean*'s documentation):

| Characteristics | |
|----------|:-------------:|
| Classes | 10 |
| Samples per class | ~180 |
| Samples total | 1797 |
| Dimensionality | 64 |
| Features | integers 0-16 |

### Basic algorithm

Let's run a very simple algorithm (taken from this *scikit-learn* [example](https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py)) in order to compare elapsed time and accuracy with *auto-sklearn*:

```python
%%time
import sklearn.datasets
from sklearn.metrics import accuracy_score
from sklearn import svm

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
# Create a classifier: a support vector classifier
classifier = svm.SVC(gamma=0.001)
classifier.fit(X_train, y_train)
y_hat = classifier.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, y_hat): 6.3f}")
```
Accuracy score: 0.991  
CPU times: user 262 ms, sys: 0 ns, total: 262 ms  
Wall time: 316 ms  


### Auto-sklearn

This is the first python script given in the *auto-sklearn* [documentation](https://automl.github.io/auto-sklearn/master/index.html#example):
```python
import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
from sklearn.metrics import accuracy_score

X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=1)
automl = autosklearn.classification.AutoSklearnClassifier()
automl.fit(X_train, y_train)
y_hat = automl.predict(X_test)
print(f"Accuracy score: {accuracy_score(y_test, y_hat): 6.3f})
```
We just made one small change; since we want to make sure to use several cores,  we instanciate the `autosklearn` model with this argument:
```python
automl = autosklearn.classification.AutoSklearnClassifier(n_jobs=4)
```
And let's run it.................................................................................................................. 

Accuracy score: 0.993  
14020,06s user 3161,23s system 477% cpu 1:00:01,78 total

The first remark is that it takes a very long time to run, about 53 minutes. However, we did not set any constraint on the algorithms to try, or set a time limit. 

The accuracy score is not so bad: *0.993*! The *auto-sklearn* algorithm did not see the test set at all, so there is no data leakage.  

| Algorithm | Elapsed time (s) | Accuracy score |
|-----------|-------------:|---------------:|
| Basic | 0.32 | 0.991 |
| *auto-sklearn* | 3601.78 | 0.993 |


## Second data set

[This](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic)) is a description of the Breast Cancer Wisconsin (Diagnostic) Data Set: each datapoint is a collection of geometric measurements of a breast mass computed from a digitized image. Here is description table from *scikit-lean*'s documentation:

| Characteristics | |
|----------|:-------------:|
| Classes | 2 |
| Samples per class | 212(M),357(B) |
| Samples total |569 |
| Dimensionality | 30 |
| Features | real, positive |

M: malignant  
B: benign  

Again, this is a very small dataset.

### Basic algorithm

We run a logistic regression classifier with default settings, and without any preprocessing except scaling:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import make_pipeline

RS = 124  # random seed

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.target = 1 - df.target
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)

rsc = RobustScaler()
lrc = LogisticRegression(random_state=RS, n_jobs=-1)
pipeline = make_pipeline(rsc, lrc)
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

print(f"Accuracy score: {accuracy_score(y_test, predictions): 6.3f}")
print(f"Balanced accuracy score: {roc_auc_score(y_test, predictions): 6.3f}")
```
Accuracy score:  0.971  
Balanced accuracy score:  0.964  
python breast_cancer_01.py  0,74s user 0,27s system 167% cpu 0,603 total

We compute the balanced accuracy score since the classes are a little imbalanced. We could not find `sklearn.metrics.balanced_accuracy_score` in the 0.19.2 release, so used `roc_auc_score` instead, which is equivalent in the case of a binary classification.

### Auto-sklearn

The *auto-sklearn* code is very similar to the previous one:

```python
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import autosklearn.classification
import sklearn.datasets
from sklearn.metrics import accuracy_score, roc_auc_score

RS = 124  # random seed

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target
df.target = 1 - df.target
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=RS)

automl = autosklearn.classification.AutoSklearnClassifier(n_jobs=4)
automl.fit(X_train, y_train)

predictions = automl.predict(X_test)
print(f"Accuracy score : {accuracy_score(y_test, predictions): 6.3f}")
print(f"Balanced accuracy score : {roc_auc_score(y_test, predictions): 6.3f}")
```
python autosklearn_02.py  15038,88s user 923,99s system 443% cpu 1:00:01,35 total  
Accuracy score :  0.947  
Balanced accuracy score :  0.936  


| Algorithm | Elapsed time (s) | Balanced accuracy score |
|-----------|-------------:|---------------:|
| Basic | 0.60 | 0.964 |
| *auto-sklearn* | 3601.35 | 0.936 |


## Final thoughts

It it a little weird that the elapsed time for each dataset is about 1 hour with *auto-sklearn*, maybe there is a default time limit for small datasets?

Anyway, in both case *auto-sklearn* gives some decent accuracy results, but with a very large amount of computation with respect to the problem given.

We should try some other examples and maybe reduce the search space and give some hints to the algorithms... 

Although we are often using some automatic hyperparameter tuning tools, it seems that automating the end-to-end process is not very efficient, energy-wise :) However, we are still interested in trying *auto-sklearn* on some other test cases and some other AutoML packages as well.