---
title: Calculating daily mean temperatures with scikit-learn WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
- supervised machine learning
- regression
- scikit-learn
- optuna
- pandas
---


The goal is of this post is to compute the daily mean air temperature TAVG from other values in the daily summaries: maximum and minimum daily temperatures and daily precipitation. The data comes from the [Climate Data Online Search](https://www.ncei.noaa.gov/cdo-web/search) of the National Centers for Environmental Information (NCEI). NCEI is a U.S. government agency that manages one of the world’s largest archives of atmospheric, coastal, geophysical, and oceanic data. We downloaded the Daily Sumaries on all the available time range, from 1920 to today, for the Lyon' station. This station is actually located at the St Exupery airport, rather outside of the city. Here is a description of the variables from their [documentation](https://www.ncei.noaa.gov/pub/data/ghcn/daily/readme.txt):

    PRCP = Precipitation (tenths of mm)
    TMAX = Maximum temperature (tenths of degrees C)
    TMIN = Minimum temperature (tenths of degrees C)
    TAVG = Average temperature (tenths of degrees C)
      [Note that TAVG from source 'S' corresponds
       to an average for the period ending at
       2400 UTC rather than local midnight]

The dataset has 37614 rows and 5 columns: DATE, PRCP, TAVG, TMAX and TMIN. However, TAVG is missing for more that half of the time range:

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/missingno.png" alt="missingno">
</p>

An obvious approach would be to compute TAVG as the arithmetic mean of TMAX and TMIN, and this is going to be our first approach. Because TAVG is available on half of the table, will use this part without missing data to create our dataset, split it into train/test sets and try out different approaches available in [scikit-learn](https://scikit-learn.org/stable/).

## Imports


```python
import warnings

import matplotlib.pyplot as plt
import missingno as msno
import numpy as np
import pandas as pd
from astral import Observer
from astral.sun import daylight
from optuna import create_study
from optuna.exceptions import ExperimentalWarning
from optuna.samplers import NSGAIISampler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklego.preprocessing import RepeatingBasisFunction

warnings.filterwarnings("ignore", category=ExperimentalWarning)
plt.style.use("fivethirtyeight")

temperature_fp = "./3441363.csv"
RS = 124  # random seed
FS = (9, 6)  # figure size
```

    Python version       : 3.11.5  
    OS                   : Linux  
    Machine              : x86_64  
    matplotlib           : 3.7.2  
    numpy                : 1.24.4  
    pandas               : 2.1.0  
    astral               : 3.2  
    optuna               : 3.3.0  
    sklearn              : 1.3.0  

## Load the data


```python
df = pd.read_csv(temperature_fp)
station = df[["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION"]].iloc[0]
df = df[["DATE", "PRCP", "TAVG", "TMIN", "TMAX"]].dropna(how="all")
df.DATE = pd.to_datetime(df.DATE)
df.set_index("DATE", inplace=True)
df = df.asfreq("D")
df.sort_index(inplace=True)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRCP</th>
      <th>TAVG</th>
      <th>TMIN</th>
      <th>TMAX</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1920-09-01</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>8.8</td>
      <td>19.7</td>
    </tr>
    <tr>
      <th>1920-09-02</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>9.1</td>
      <td>20.1</td>
    </tr>
    <tr>
      <th>1920-09-03</th>
      <td>1.5</td>
      <td>NaN</td>
      <td>9.3</td>
      <td>16.8</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (37617, 4)




```python
station
```




    STATION              FR069029001
    NAME         LYON ST EXUPERY, FR
    LATITUDE                 45.7264
    LONGITUDE                 5.0778
    ELEVATION                  235.0
    Name: 0, dtype: object



A surprising fact about these data set is that the daily mean temperature can be slightly smaller than the daily min:


```python
len(df[df.TAVG < df.TMIN])
```




    31



and larger that the daily max:


```python
len(df[df.TAVG > df.TMAX])
```




    235



I guess that it comes from the measurement methods and also from different daily time ranges:
> TAVG corresponds to an average for the period ending at 2400 UTC rather than local midnight

## Missing values


```python
ax = (100 * df.isna().sum(axis=0) / len(df)).plot.bar(alpha=0.7, figsize=(6, 3))
_ = ax.set_ylim(0, 60)
_ = ax.set(
    title="Ratio of missing values per column", xlabel="Column name", ylabel="Ratio (%)"
)
```


<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_16_0.png" alt="output_16_0">
</p>



## First approach : TAVG_am, arithmetic mean of TMIN and TMAX


```python
df["TAVG_am"] = 0.5 * (df["TMIN"] + df["TMAX"])
ax = df[["TAVG", "TAVG_am"]].plot.scatter(
    x="TAVG", y="TAVG_am", alpha=0.2, figsize=(6, 6)
)

df_tmp = df[["TAVG", "TAVG_am"]].dropna(how="any")
X = df_tmp["TAVG"].values[:, np.newaxis]
y = df_tmp["TAVG_am"].values
lr = LinearRegression()
lr.fit(X, y)

x_min, x_max = -11, 36
points = np.linspace(x_min, x_max, 100)
_ = ax.plot(points, points, color="r")
_ = plt.plot(X, lr.predict(X), color="k", linewidth=1)
_ = ax.set_xlim(x_min, x_max)
_ = ax.set_ylim(x_min, x_max)
_ = ax.legend(["TAVG, TAVG_am couples", "y=x", "Linear regression"])
_ = ax.set(title="Correlation between TAVG and TAVG_am")
```

    
<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_18_0.png" alt="output_18_0">
</p>


Although both values are clearly correlated, we see that TAVG_am may be off by 3 or 4 degrees on warm days.


```python
diff = (df.TAVG - df.TAVG_am).dropna()
avg = diff.mean()
std = diff.std()
ax = diff.plot.hist(bins=50, alpha=0.7, figsize=(6, 6))
_ = plt.axvline(x=avg, color="b", linewidth=1)
_ = plt.axvline(x=avg - 2 * std, color="r", linewidth=1)
_ = plt.axvline(x=avg + 2 * std, color="r", linewidth=1)
ax.legend(["Distribution", "$\\bar{x}$", "$\\bar{x} \pm 2 \sigma$"])
_ = ax.set(title="Distribution of the error of TAVG_am", xlabel="TAVG - TAVG_am")
```


<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_20_0.png" alt="output_20_0">
</p>


## Model evaluation


```python
def evaluate(y_true, y_pred, return_score=False):
    d = {}
    d["mae"] = mean_absolute_error(y_true, y_pred)
    d["rmse"] = mean_squared_error(y_true, y_pred, squared=False)
    if return_score:
        return d
    else:
        print(
            f'Mean absolute error : {d["mae"]:8.4f}, Mean squared error : {d["rmse"]:8.4f}'
        )
```


```python
y_pred = df.loc[~df.TAVG.isna() & ~df.TAVG_am.isna(), "TAVG_am"].values
y_true = df.loc[~df.TAVG.isna() & ~df.TAVG_am.isna(), "TAVG"].values
evaluate(y_true, y_pred)
```

    Mean absolute error :   0.9098, Mean squared error :   1.2088



```python
# cleanup
df.drop("TAVG_am", axis=1, inplace=True)
```

## Feature engineering

- encode cyclic time data  
- diurnal temperature range (DTR)  
- daylight  
- TMIN and TMAX delta with the day before

### Temporal features


```python
df.reset_index(drop=False, inplace=True)
df["year"] = df["DATE"].dt.year
df["month"] = df["DATE"].dt.month
df["dayofyear"] = df["DATE"].dt.dayofyear
df.set_index("DATE", inplace=True)
df.head(3)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PRCP</th>
      <th>TAVG</th>
      <th>TMIN</th>
      <th>TMAX</th>
      <th>year</th>
      <th>month</th>
      <th>dayofyear</th>
    </tr>
    <tr>
      <th>DATE</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1920-09-01</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>8.8</td>
      <td>19.7</td>
      <td>1920</td>
      <td>9</td>
      <td>245</td>
    </tr>
    <tr>
      <th>1920-09-02</th>
      <td>0.0</td>
      <td>NaN</td>
      <td>9.1</td>
      <td>20.1</td>
      <td>1920</td>
      <td>9</td>
      <td>246</td>
    </tr>
    <tr>
      <th>1920-09-03</th>
      <td>1.5</td>
      <td>NaN</td>
      <td>9.3</td>
      <td>16.8</td>
      <td>1920</td>
      <td>9</td>
      <td>247</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["daycountinyear"] = df["year"].map(lambda y: pd.Timestamp(y, 12, 31).dayofyear)
df["sin_encoded_year"] = np.sin(
    2.0 * np.pi * (df["dayofyear"] - 1) / df["daycountinyear"]
)
df["cos_encoded_year"] = np.cos(
    2.0 * np.pi * (df["dayofyear"] - 1) / df["daycountinyear"]
)
df.drop(["year", "daycountinyear"], axis=1, inplace=True)
```


```python
ax = df["2020":"2020"][["sin_encoded_year", "cos_encoded_year"]].plot(figsize=(6, 3))
_ = ax.set(title="Cyclical encoded day-of-year", xlabel="Day-of-year")
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_29_0.png" alt="output_29_0">
</p>



Radial basis functions


```python
rbf = RepeatingBasisFunction(
    n_periods=6, column="dayofyear", input_range=(1, 366), remainder="drop"
)
rbf.fit(df)
RBFs = pd.DataFrame(index=df.index, data=rbf.transform(df))
RBFs = RBFs.add_prefix("rbf_")
axs = RBFs[:400].plot(
    subplots=True,
    figsize=(6, 6),
    sharex=True,
    title="Radial Basis Functions",
    legend=False,
    rot=90,
)
for ax in axs:
    ax.yaxis.set_tick_params(labelleft=False)
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_31_0.png" alt="output_31_0">
</p>



```python
df = pd.concat((df, RBFs), axis=1)
```

### Temperature ranges, deltas with previous and next days

Daily temperature range:


```python
df["DTR"] = df["TMAX"] - df["TMIN"]
```

TMIN and TMAX deltas with previous and next days:


```python
df["TMAX_day_before"] = df["TMAX"].shift(+1)
df["TMIN_day_before"] = df["TMIN"].shift(+1)
df["TMAX_day_after"] = df["TMAX"].shift(-1)
df["TMIN_day_after"] = df["TMIN"].shift(-1)
df["TMAX_delta_db"] = df["TMAX"] - df["TMAX_day_before"]
df["TMIN_delta_db"] = df["TMIN"] - df["TMIN_day_before"]
df["TMAX_delta_da"] = df["TMAX"] - df["TMAX_day_after"]
df["TMIN_delta_da"] = df["TMIN"] - df["TMIN_day_after"]
df.drop(
    ["TMIN_day_before", "TMAX_day_before", "TMIN_day_after", "TMAX_day_after"],
    axis=1,
    inplace=True,
)
```

### Daytime length


```python
observer = Observer(latitude=station.LATITUDE, longitude=station.LONGITUDE)


def compute_daytime(date):
    sr, ss = daylight(observer, date=date)
    return (ss - sr).total_seconds()


df["datetime"] = df.index
df["daytime"] = df["datetime"].map(lambda d: compute_daytime(d))
df.drop("datetime", axis=1, inplace=True)
```

## Train/test split


```python
target = "TAVG"
features = [c for c in df.columns if c != target]
features
```




    ['PRCP',
     'TMIN',
     'TMAX',
     'month',
     'dayofyear',
     'sin_encoded_year',
     'cos_encoded_year',
     'rbf_0',
     'rbf_1',
     'rbf_2',
     'rbf_3',
     'rbf_4',
     'rbf_5',
     'DTR',
     'TMAX_delta_db',
     'TMIN_delta_db',
     'TMAX_delta_da',
     'TMIN_delta_da',
     'daytime']




```python
dataset = df[~df.TAVG.isna()].copy(deep=True)
dataset.dropna(how="any", inplace=True)
X, y = dataset[features].copy(deep=True), dataset[target].copy(deep=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RS, shuffle=True
)
```

## Baseline : arithmetic mean


```python
class BaselineModel:
    def fit(self, X_train):
        pass

    def predict(self, X_test):
        if "TAVG_am" in X_test:
            return X_test["TAVG_am"].values
        else:
            return 0.5 * (X_test["TMIN"].values + X_test["TMAX"].values)
```


```python
baseline = BaselineModel()
y_pred = baseline.predict(X_test)
```


```python
results = pd.DataFrame()
results = pd.concat(
    [
        results,
        pd.Series(evaluate(y_test, y_pred, return_score=True)).to_frame("baseline").T,
    ],
    axis=0,
)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mae</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.916753</td>
      <td>1.214852</td>
    </tr>
  </tbody>
</table>
</div>



## Ridge


```python
ridge_reg = Pipeline(
    [
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ]
)
ridge_reg.fit(X_train, y_train)
y_pred = ridge_reg.predict(X_test)
```


```python
results = pd.concat(
    [
        results,
        pd.Series(evaluate(y_test, y_pred, return_score=True)).to_frame("ridge").T,
    ],
    axis=0,
)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mae</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.916753</td>
      <td>1.214852</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>0.740136</td>
      <td>0.984712</td>
    </tr>
  </tbody>
</table>
</div>




```python
fi = pd.DataFrame(data={"importance": ridge_reg["ridge"].coef_}, index=X_train.columns)
fi.sort_values(by="importance", ascending=False, inplace=True)
ax = fi.plot.barh(figsize=(7, 5), alpha=0.7, legend=False)
ax.invert_yaxis()
```

    
<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_49_0.png" alt="output_49_0">
</p>


```python
drop_features = ["dayofyear", "month", "TMIN_delta_db"]
X_train.drop(drop_features, axis=1, inplace=True)
X_test.drop(drop_features, axis=1, inplace=True)
X.drop(drop_features, axis=1, inplace=True)
for f in drop_features:
    features.remove(f)
```

## DecisionTreeRegressor


```python
dtr = DecisionTreeRegressor(random_state=RS)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
evaluate(y_test, y_pred)
```

    Mean absolute error :   0.9473, Mean squared error :   1.2625



```python
mae_train_score = []
mae_test_score = []
for md in range(2, 21):
    dtr = DecisionTreeRegressor(max_depth=md, random_state=RS)
    dtr.fit(X_train, y_train)

    y_pred = dtr.predict(X_train)
    d_train = evaluate(y_train, y_pred, return_score=True)
    mae_train_score.append(d_train["mae"])

    y_pred = dtr.predict(X_test)
    d_test = evaluate(y_test, y_pred, return_score=True)
    mae_test_score.append(d_test["mae"])
```


```python
plt.figure(figsize=(7, 5))
_ = plt.plot(range(2, 21), mae_train_score)
_ = plt.plot(range(2, 21), mae_test_score)
ax = plt.gca()
_ = ax.legend(["train", "test"])
_ = ax.set(title="Train/test error vs max_depth", xlabel="max_depth", ylabel="MAE")
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_54_0.png" alt="output_54_0">
</p>



```python
dtr = DecisionTreeRegressor(max_depth=8, random_state=RS)
dtr.fit(X_train, y_train)
y_pred = dtr.predict(X_test)
```


```python
results = pd.concat(
    [
        results,
        pd.Series(evaluate(y_test, y_pred, return_score=True))
        .to_frame("decision tree")
        .T,
    ],
    axis=0,
)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mae</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.916753</td>
      <td>1.214852</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>0.740136</td>
      <td>0.984712</td>
    </tr>
    <tr>
      <th>decision tree</th>
      <td>0.782112</td>
      <td>1.049301</td>
    </tr>
  </tbody>
</table>
</div>




```python
fi = pd.DataFrame(data={"importance": dtr.feature_importances_}, index=X_train.columns)
fi.sort_values(by="importance", ascending=False, inplace=True)
ax = fi[:30].plot.barh(figsize=(7, 5), logx=True, alpha=0.7, legend=False)
ax.invert_yaxis()
_ = ax.set(
    title="Decision tree feature importance",
    xlabel="Importance (Log scale)",
    ylabel="Feature",
)
```
    
<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_57_0.png" alt="output_57_0">
</p>

Feature selection:


```python
drop_features = ["rbf_2", "rbf_4", "sin_encoded_year"]
X_train.drop(drop_features, axis=1, inplace=True)
X_test.drop(drop_features, axis=1, inplace=True)
X.drop(drop_features, axis=1, inplace=True)
for f in drop_features:
    features.remove(f)
```

## RandomForestRegressor


```python
rfr = RandomForestRegressor(
    n_estimators=250,
    max_depth=13,
    min_samples_split=10,
    max_features=0.9,
    n_jobs=12,
    random_state=RS,
)
rfr.fit(X_train, y_train)
y_pred = rfr.predict(X_test)
evaluate(y_test, y_pred)
```

    Mean absolute error :   0.6746, Mean squared error :   0.9102



```python
results = pd.concat(
    [
        results,
        pd.Series(evaluate(y_test, y_pred, return_score=True))
        .to_frame("random forest")
        .T,
    ],
    axis=0,
)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mae</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.916753</td>
      <td>1.214852</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>0.740136</td>
      <td>0.984712</td>
    </tr>
    <tr>
      <th>decision tree</th>
      <td>0.782112</td>
      <td>1.049301</td>
    </tr>
    <tr>
      <th>random forest</th>
      <td>0.674629</td>
      <td>0.910177</td>
    </tr>
  </tbody>
</table>
</div>




```python
fi = pd.DataFrame(data={"importance": rfr.feature_importances_}, index=X_train.columns)
fi.sort_values(by="importance", ascending=False, inplace=True)
ax = fi[:25].plot.barh(figsize=(7, 5), logx=True, alpha=0.6, legend=False)
ax.invert_yaxis()
_ = ax.set(
    title="Random forest feature importance",
    xlabel="Importance (Log scale)",
    ylabel="Feature",
)
```


<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_63_0.png" alt="output_63_0">
</p>



```python
drop_features = ["rbf_5", "rbf_1"]
X_train.drop(drop_features, axis=1, inplace=True)
X_test.drop(drop_features, axis=1, inplace=True)
X.drop(drop_features, axis=1, inplace=True)
for f in drop_features:
    features.remove(f)
```

## HistGradientBoostingRegressor


```python
hgbr = HistGradientBoostingRegressor(random_state=RS)
hgbr.fit(X_train, y_train)
y_pred = hgbr.predict(X_test)
evaluate(y_test, y_pred)
```

    Mean absolute error :   0.6692, Mean squared error :   0.9037



```python
results = pd.concat(
    [
        results,
        pd.Series(evaluate(y_test, y_pred, return_score=True))
        .to_frame("hist gradient boosting")
        .T,
    ],
    axis=0,
)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mae</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.916753</td>
      <td>1.214852</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>0.740136</td>
      <td>0.984712</td>
    </tr>
    <tr>
      <th>decision tree</th>
      <td>0.782112</td>
      <td>1.049301</td>
    </tr>
    <tr>
      <th>random forest</th>
      <td>0.674629</td>
      <td>0.910177</td>
    </tr>
    <tr>
      <th>hist gradient boosting</th>
      <td>0.669222</td>
      <td>0.903695</td>
    </tr>
  </tbody>
</table>
</div>




```python
FIXED_PARAMS = {}
FIXED_PARAMS["loss"] = "squared_error"
FIXED_PARAMS["monotonic_cst"] = {"TMIN": 1, "TMAX": 1}
FIXED_PARAMS["validation_fraction"] = 0.15
FIXED_PARAMS["n_iter_no_change"] = 15
FIXED_PARAMS["early_stopping"] = True
FIXED_PARAMS["random_state"] = RS


class Objective(object):
    def __init__(self, X, y, fixed_params=FIXED_PARAMS):
        self.X = X
        self.y = y
        self.fixed_params = fixed_params

    def __call__(self, trial):
        learning_rate = trial.suggest_float("learning_rate", 0.02, 0.2, log=True)
        max_iter = trial.suggest_categorical("max_iter", [50, 100, 150, 200, 250])
        max_leaf_nodes = trial.suggest_categorical(
            "max_leaf_nodes", [21, 26, 31, 36, 41]
        )
        max_depth = trial.suggest_int("max_depth", 6, 32, log=False)
        min_samples_leaf = trial.suggest_categorical(
            "min_samples_leaf", [10, 15, 20, 25, 30]
        )
        l2_regularization = trial.suggest_float(
            "l2_regularization", 1e-8, 1e-1, log=True
        )
        max_bins = trial.suggest_categorical("max_bins", [205, 225, 255])

        hgbr = HistGradientBoostingRegressor(
            learning_rate=learning_rate,
            max_iter=max_iter,
            max_leaf_nodes=max_leaf_nodes,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            l2_regularization=l2_regularization,
            max_bins=max_bins,
            **self.fixed_params
        )

        score = cross_val_score(
            hgbr, self.X, self.y, n_jobs=3, cv=3, scoring="neg_mean_absolute_error"
        )
        mae = -1.0 * score.mean()
        return mae
```


```python
objective = Objective(X_train, y_train)

sampler = NSGAIISampler(
    population_size=100,
    seed=RS,
)
study = create_study(direction="minimize", sampler=sampler)
study.optimize(
    objective,
    n_trials=1000,
    n_jobs=1,
)
```

    [I 2023-09-06 13:52:45,182] A new study created in memory with name: no-name-d32e6de4-7cae-43d5-8f5e-afe0d72605b9
    [I 2023-09-06 13:52:45,989] Trial 0 finished with value: 1.8808348240559607 and parameters: {'learning_rate': 0.025532591774822706, 'max_iter': 50, 'max_leaf_nodes': 36, 'max_depth': 9, 'min_samples_leaf': 15, 'l2_regularization': 0.0012473085120470293, 'max_bins': 225}. Best is trial 0 with value: 
    ...
    [I 2023-09-06 14:01:29,745] Trial 999 finished with value: 0.6788856968370406 and parameters: {'learning_rate': 0.04112414787841993, 'max_iter': 250, 'max_leaf_nodes': 36, 'max_depth': 22, 'min_samples_leaf': 30, 'l2_regularization': 0.0024433893814918765, 'max_bins': 225}. Best is trial 868 with value: 0.6764808748435326.



```python
hp = study.best_params.copy()
hp.update(FIXED_PARAMS)
for key, value in hp.items():
    print(f"{key:>20s} : {value}")
print(f"{'best objective value':>20s} : {study.best_value}")
```

           learning_rate : 0.04403581826578526
                max_iter : 200
          max_leaf_nodes : 36
               max_depth : 28
        min_samples_leaf : 25
       l2_regularization : 0.017899518645399605
                max_bins : 255
                    loss : squared_error
           monotonic_cst : {'TMIN': 1, 'TMAX': 1}
     validation_fraction : 0.15
        n_iter_no_change : 15
          early_stopping : True
            random_state : 124
    best objective value : 0.6764808748435326



```python
trials = study.trials_dataframe()
score = np.ndarray(len(trials), dtype=np.float64)
i = 0
value_min = 1.0e15
for row in trials.itertuples():
    value_min = np.amin([value_min, row.value])
    score[row.number] = value_min
    i += 1
trials["best_value"] = score
ax = trials[["number", "best_value"]].plot(
    x="number", y="best_value", logy=True, color="r", label="Best value"
)
ax = trials[["number", "value"]].plot.scatter(
    x="number", y="value", logy=True, color="b", ax=ax, label="Trial value"
)
```


<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_71_0.png" alt="output_71_0">
</p>



Eval on train set:


```python
n_splits = 3
kf = KFold(n_splits=n_splits, random_state=RS, shuffle=True)
y_pred = np.zeros_like(y_train)
hgbrs = []
for train_index, test_index in kf.split(X_train):
    hgbr = HistGradientBoostingRegressor(**hp)
    hgbr.fit(X_train.iloc[train_index], y_train.iloc[train_index])
    hgbrs.append(hgbr)
    y_pred[test_index] = hgbr.predict(X_train.iloc[test_index])
evaluate(y_train, y_pred)
```

    Mean absolute error :   0.6767, Mean squared error :   0.9047


Eval on test set:


```python
y_pred = np.zeros((X_test.shape[0],), dtype=np.float64)
for model in hgbrs:
    y_pred += model.predict(X_test)
y_pred /= len(hgbrs)
evaluate(y_test, y_pred)
```

    Mean absolute error :   0.6673, Mean squared error :   0.9018



```python
results = pd.concat(
    [
        results,
        pd.Series(evaluate(y_test, y_pred, return_score=True))
        .to_frame("hist gradient boosting w HPO")
        .T,
    ],
    axis=0,
)
results
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mae</th>
      <th>rmse</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>baseline</th>
      <td>0.916753</td>
      <td>1.214852</td>
    </tr>
    <tr>
      <th>ridge</th>
      <td>0.740136</td>
      <td>0.984712</td>
    </tr>
    <tr>
      <th>decision tree</th>
      <td>0.782112</td>
      <td>1.049301</td>
    </tr>
    <tr>
      <th>random forest</th>
      <td>0.674629</td>
      <td>0.910177</td>
    </tr>
    <tr>
      <th>hist gradient boosting</th>
      <td>0.669222</td>
      <td>0.903695</td>
    </tr>
    <tr>
      <th>hist gradient boosting w HPO</th>
      <td>0.667300</td>
      <td>0.901843</td>
    </tr>
  </tbody>
</table>
</div>



## Fill the missing TAVG values


```python
X_mv = df[df.TAVG.isna()][features]  # dataset with missing TAVG values
y_pred = np.zeros((X_mv.shape[0],), dtype=np.float64)
for model in hgbrs:
    y_pred += model.predict(X_mv)
y_pred /= len(hgbrs)
TAVG_pred = pd.DataFrame(data=y_pred, index=X_mv.index, columns=["TAVG_pred"])
df = pd.concat((df, TAVG_pred), axis=1)
df.TAVG = df.TAVG.fillna(df.TAVG_pred)
df.drop("TAVG_pred", axis=1, inplace=True)
df = df[["PRCP", "TAVG", "TMIN", "TMAX"]]
```


```python
_ = msno.matrix(df)
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_79_0.png" alt="output_79_0">
</p>



```python
ax = (
    df[["TMIN", "TMAX", "TAVG"]]["1921":"2022"]
    .resample("Y")
    .mean()
    .plot(figsize=(6, 4))
)
_ = ax.set(
    title="Yearly average temperatures in Lyon",
    xlabel="Year",
    ylabel="Temperature (°C)",
)
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_80_0.png" alt="output_80_0">
</p>



```python
df.to_parquet("./lyon_historical_temperatures.parquet")
```

    /home/francois/miniconda3/envs/py311/lib/python3.11/site-packages/pyarrow/pandas_compat.py:373: FutureWarning: is_sparse is deprecated and will be removed in a future version. Check `isinstance(dtype, pd.SparseDtype)` instead.
      if _pandas_api.is_sparse(col):


## Mean June Temperatures


```python
T_month = (
    df[["TMIN", "TAVG", "TMAX"]]
    .resample("M")
    .agg({"TMIN": "min", "TAVG": "mean", "TMAX": "max"})
)
T_month["month"] = T_month.index.month
```


```python
T_june = T_month[T_month.month == 6].copy(deep=True)
ax = T_june[["TMIN", "TAVG", "TMAX"]].plot(
    style="-", linewidth=1, drawstyle="steps-mid", figsize=(9, 6), legend=False
)
_ = ax.set(
    title="June temperatures\n in Lyon Saint-Exupéry",
    xlabel="Year",
    ylabel="Temperature (°C)",
)
_ = ax.legend(("min(TMIN)", "mean(TAVG)", "max(TMAX)"))
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_84_0.png" alt="output_84_0">
</p>



## Summer months temperature anomalies


```python
start, end = "1920", "1980"

df_summer = T_month.loc[T_month.month.isin([6, 7, 8])].TAVG.resample("Y").mean()
t_summer = df_summer[start:end].mean()

anomaly = (df_summer - t_summer).copy(deep=True).to_frame("anomaly")
anomaly["label"] = anomaly.index.year.astype(str)
anomaly.loc[anomaly["label"].str[-1] != "0", "label"] = ""
rw = anomaly.anomaly.rolling(10, center=True).mean().to_frame("window")

fig, ax = plt.subplots(figsize=FS)
mask = anomaly[start:].anomaly > 0
colors = np.array(["dodgerblue"] * len(anomaly[start:]))
colors[mask.values] = "tomato"
plt.bar(anomaly[start:].index.year, anomaly[start:].anomaly, color=colors)
plt.plot(
    rw[start:].index.year.tolist(),
    rw[start:].window.values,
    "k",
    linewidth=3,
    alpha=0.5,
    label="10-year moving average",
)

ax.legend()
ax.set_xlabel("Date")
ax.set_ylabel(f"Degrees (°C) +/- from {start}-{end} average")
plt.autoscale(enable=True, axis="x", tight=True)
ax.set_title(
    f"June-July-August temperature anomalies\n in Lyon-France w.r.t {start}-{end} mean"
);
```

<p align="center">
  <img width="1000" src="/img/2023-09-06_01/output_86_0.png" alt="output_86_0">
</p>




## References

https://etda.libraries.psu.edu/catalog/13504jeb5249

https://climate.rutgers.edu/stateclim/?section=menu&%20target=calculating_daily_mean_temperature

https://www.sciencedirect.com/science/article/abs/pii/S0168192304002199

https://www.njweather.org/content/better-approach-calculating-daily-mean-temperature-0


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