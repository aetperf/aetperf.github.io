
## Monthly averaged historical temperatures in France and over the global land surface

The aim of this notebook is just to play with time series along with a couple of statistical and plotting libraries.

### Imports


```python
%matplotlib inline
import pandas as pd  # 0.23.0
import numpy as np
import matplotlib.pyplot as plt  # 2.2.2
import seaborn as sns  # 0.8.1
from statsmodels.tsa.stattools import adfuller  # 0.9.0
import statsmodels.api as sm
```

### Loading the data

Historical monthly average temperature in France from the [World Bank Group](http://sdwebx.worldbank.org/climateportal/index.cfm).

> Historical data to understand the seasonal CYCLE: This gridded historical dataset is derived from observational data, and provides quality controlled temperature and rainfall values from thousands of weather stations worldwide, as well as derivative products including monthly climatologies and long term historical climatologies. The dataset is produced by the Climatic Research Unit (CRU) of University of East Anglia (UEA), and reformatted by International Water Management Institute (IWMI). CRU-(Gridded Product). CRU data can be mapped to show the baseline climate and seasonality by month, for specific years, and for rainfall and temperature.

The data was loaded into a spreadsheet and then exported as a csv file, after changing decimal separator from comma to point and removing useless columns (country code).


```python
temperature = pd.read_csv('data/tas_1901_2015_France.csv')
temperature.head()
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
      <th>tas</th>
      <th>Year</th>
      <th>Month</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.25950</td>
      <td>1901</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.32148</td>
      <td>1901</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.82373</td>
      <td>1901</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.38568</td>
      <td>1901</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.69960</td>
      <td>1901</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(len(temperature), "rows")
print("column types :\n", temperature.dtypes)
```

    1380 rows
    column types :
     tas      float64
    Year       int64
    Month      int64
    dtype: object


### Creating a DatetimeIndex

Pandas'`to_datetime()` method requires at least the `year`, `month` and `day` columns (in lower case). So we rename the `Year` and `Month` columns and create a `day` column arbitrarily set to the 15th of each month.


```python
temperature.rename(columns={'Year': 'year', 'Month': 'month'}, inplace=True)
temperature["day"] = 15
temperature.head()
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
      <th>tas</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.25950</td>
      <td>1901</td>
      <td>1</td>
      <td>15</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.32148</td>
      <td>1901</td>
      <td>2</td>
      <td>15</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.82373</td>
      <td>1901</td>
      <td>3</td>
      <td>15</td>
    </tr>
    <tr>
      <th>3</th>
      <td>9.38568</td>
      <td>1901</td>
      <td>4</td>
      <td>15</td>
    </tr>
    <tr>
      <th>4</th>
      <td>12.69960</td>
      <td>1901</td>
      <td>5</td>
      <td>15</td>
    </tr>
  </tbody>
</table>
</div>



Now we can apply the `to_datetime()` method, change the index and clean the dataframe.


```python
temperature["Date"] = pd.to_datetime(temperature[['year', 'month', 'day']])
temperature.set_index("Date", inplace=True)
temperature.drop(['year', 'month', 'day'], axis=1, inplace=True)
temperature.head()
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
      <th>tas</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1901-01-15</th>
      <td>3.25950</td>
    </tr>
    <tr>
      <th>1901-02-15</th>
      <td>0.32148</td>
    </tr>
    <tr>
      <th>1901-03-15</th>
      <td>4.82373</td>
    </tr>
    <tr>
      <th>1901-04-15</th>
      <td>9.38568</td>
    </tr>
    <tr>
      <th>1901-05-15</th>
      <td>12.69960</td>
    </tr>
  </tbody>
</table>
</div>



Plotting these monthly averaged temperatures is rather messy because of seasonal fluctuations.


```python
temperature.plot(figsize=(12, 6));
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_10_1.png">
</p>

Note that we could also have created a `PeriodIndex` with a monthy frequency:
```python
temperature = temperature.to_period(freq='M')
```

However this seems to be less handy for resampling operations.

### Resampling

First thing we can do is to compute the annual `mean`, `min` and `max`. The `resample()` method is used with the 'AS' offset, which corresponds to the year start frequency (see [this](http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases) following documentation for the whole list of rules).


```python
temp_annual = temperature.resample("AS").agg(['mean', 'min', 'max'])
```


```python
temp_annual.head(5)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead tr th {
        text-align: left;
    }

    .dataframe thead tr:last-of-type th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr>
      <th></th>
      <th colspan="3" halign="left">tas</th>
    </tr>
    <tr>
      <th></th>
      <th>mean</th>
      <th>min</th>
      <th>max</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1901-01-01</th>
      <td>10.177163</td>
      <td>0.32148</td>
      <td>20.1921</td>
    </tr>
    <tr>
      <th>1902-01-01</th>
      <td>9.991733</td>
      <td>2.26860</td>
      <td>18.1870</td>
    </tr>
    <tr>
      <th>1903-01-01</th>
      <td>10.070277</td>
      <td>2.95230</td>
      <td>17.7461</td>
    </tr>
    <tr>
      <th>1904-01-01</th>
      <td>10.307586</td>
      <td>2.31437</td>
      <td>18.9831</td>
    </tr>
    <tr>
      <th>1905-01-01</th>
      <td>9.982693</td>
      <td>0.49251</td>
      <td>18.8757</td>
    </tr>
  </tbody>
</table>
</div>




```python
temp_annual.tas.plot(figsize=(12, 6))
ax = plt.gca()
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.5)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("Temperature (°C)")
sns.despine()
```


<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_15_0.png">
</p>


### Rolling window mean

Next we compute a centered rolling window mean with 10 years windows and an exponentially-weighted moving average with a 2 years half life. 


```python
temp_annual[('tas', 'mean')].plot(label='yearly temperature', figsize=(12, 6))
temp_annual[('tas', 'mean')].rolling(10, center=True).mean().plot(label='10 years rolling window mean')
temp_annual[('tas', 'mean')].ewm(halflife=2).mean().plot(label='EWMA(halflife=2)')  # Exponentially-weighted moving average
ax = plt.gca()
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.5)
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.ylabel("Temperature (°C)")
sns.despine()
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_17_0.png">
</p>


We can also compute and plot the standard deviation of the data over the rolling window.


```python
m = temp_annual[('tas', 'mean')].rolling(10, center=True).agg(['mean', 'std'])
ax = m['mean'].plot(label='10 years rolling window mean', figsize=(12, 6))
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.5)
ax.fill_between(m.index, m['mean'] - m['std'], m['mean'] + m['std'], alpha=.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("Temperature (°C)")
sns.despine()
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_19_0.png">
</p>


### Checking for Stationarity

Well I am not very familiar with Statistics related with time series, so I just grabbed something that looks fancy on [this](https://www.analyticsvidhya.com/blog/2016/02/time-series-forecasting-codes-python/) page :) This is the [Dickey-Fuller test](https://en.wikipedia.org/wiki/Dickey%E2%80%93Fuller_test) from [StatsModels](https://www.statsmodels.org/stable/index.html).

Let us apply this test to the original monthly temperature dataframe. First we convert it into a time `Series`.


```python
ts = pd.Series(temperature.tas, index=temperature.index)
ts.head()
```




    Date
    1901-01-15     3.25950
    1901-02-15     0.32148
    1901-03-15     4.82373
    1901-04-15     9.38568
    1901-05-15    12.69960
    Name: tas, dtype: float64



Then we perform the test:


```python
dftest = adfuller(ts, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
```

    Test Statistic                   -1.651531
    p-value                           0.456242
    #Lags Used                       24.000000
    Number of Observations Used    1355.000000
    Critical Value (1%)              -3.435185
    Critical Value (5%)              -2.863675
    Critical Value (10%)             -2.567907
    dtype: float64


We already saw that the mean is clearly increasing starting in the 80s while standard deviation seems to be uniform. Now we can observe that the Test Statistic is significantly greater than the critical values and that the p-value is not so small, which also seems to indicate that the data is not stationary.

Let us try to make this time series artificially stationary by removing the rolling mean from the data and run the test again. We start by computing the mean on a 120 months rolling window.


```python
ts_mean = ts.rolling(window=120).mean().rename("t_mean")
ts_mean.plot();
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_25_1.png">
</p>


Note that there are `NaN` values at the begining of the serie because the there is not enough data in the rolling window.


```python
ts_mean.head()
```




    Date
    1901-01-15   NaN
    1901-02-15   NaN
    1901-03-15   NaN
    1901-04-15   NaN
    1901-05-15   NaN
    Name: t_mean, dtype: float64



Now we merge the mean temperature with the original monthly temperature and compute the "corrected" temperature `t_modified`.


```python
stat_temp = pd.merge(temperature, ts_mean.to_frame(), 
                     left_index=True, right_index=True, how='inner').dropna()
stat_temp["t_modified"] = stat_temp["tas"]-stat_temp["t_mean"]+stat_temp["t_mean"][0]
```

Let us re-compute the rolling mean for original and modified data.


```python
stat_temp.tas.rolling(window=120).mean().plot(label='original', figsize=(12, 6))
stat_temp.t_modified.rolling(window=120).mean().plot(label='modified')
ax = plt.gca()
ax.grid(color=(0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.5)
plt.autoscale(enable=True, axis='x', tight=True)
plt.legend()
plt.ylabel("Temperature (°C)")
sns.despine()
```


<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_31_0.png">
</p>


Now we run the Dickey–Fuller test again, on the modified data.


```python
ts2 = pd.Series(stat_temp.t_modified, index=stat_temp.index)
ts2.head()
```




    Date
    1910-12-15    4.385380
    1911-01-15    2.021325
    1911-02-15    3.879265
    1911-03-15    6.911458
    1911-04-15    8.259301
    Name: t_modified, dtype: float64




```python
dftest = adfuller(ts2, autolag='AIC')
dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
for key, value in dftest[4].items():
    dfoutput['Critical Value (%s)'%key] = value
print(dfoutput)
```

    Test Statistic                -5.913709e+00
    p-value                        2.602534e-07
    #Lags Used                     2.300000e+01
    Number of Observations Used    1.237000e+03
    Critical Value (1%)           -3.435647e+00
    Critical Value (5%)           -2.863879e+00
    Critical Value (10%)          -2.568015e+00
    dtype: float64


This time the data is found to be stationary (Test Statistic is lower than the critical values and small p_value), which is not a big surprise...

We should probably consider a longer time span when testing for the unstationarity of temperature.

### Comparison with global land temperature

We are now going to compare France against global land's temperatures, taken from Berkeley Earth's [website](http://berkeleyearth.lbl.gov/auto/Global/Complete_TAVG_complete.txt).


```python
glt = pd.read_csv('data/Complete_TAVG_complete.txt', delim_whitespace=True, skiprows=33).iloc[:,0:4]
glt.columns = ['year', 'month', 'Anomaly', 'Unc'] 
glt.head()
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
      <th>year</th>
      <th>month</th>
      <th>Anomaly</th>
      <th>Unc</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1750</td>
      <td>1</td>
      <td>0.382</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1750</td>
      <td>2</td>
      <td>0.539</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1750</td>
      <td>3</td>
      <td>0.574</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1750</td>
      <td>4</td>
      <td>0.382</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1750</td>
      <td>5</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



Let us list the years with `NaN` values in the Anomaly column.


```python
gby = glt[['year', 'Anomaly']].groupby('year').count()
gby[gby.Anomaly<12]
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
      <th>Anomaly</th>
    </tr>
    <tr>
      <th>year</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1750</th>
      <td>4</td>
    </tr>
    <tr>
      <th>1751</th>
      <td>1</td>
    </tr>
    <tr>
      <th>1752</th>
      <td>6</td>
    </tr>
    <tr>
      <th>2017</th>
      <td>7</td>
    </tr>
  </tbody>
</table>
</div>



As we can see monthly Anomaly data are complete between 1753 and 2016.


```python
glt = glt.loc[(glt.year >= 1753) & (glt.year <= 2016)]
```

Estimated Jan 1951-Dec 1980 monthly absolute temperature, copied from the file header:


```python
mat = pd.DataFrame({
    'month': np.arange(1, 13),
    't_abs': [2.62, 3.23, 5.33, 8.36, 11.37, 13.53, 14.41, 13.93, 12.12, 9.26, 6.12, 3.67]})
#    'unc': [0.08, 0.07, 0.06, 0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.06, 0.07, 0.08]})
```

Let us merge the absolute temperature with the Anomaly.


```python
glt = pd.merge(glt, mat, on='month', how='left')
glt['t'] = glt.t_abs + glt.Anomaly
```

And create a `DatetimeIndex`.


```python
glt['day'] = 15
glt["Date"] = pd.to_datetime(glt[['year', 'month', 'day']])
glt.set_index("Date", inplace=True)
glt.drop(['year', 'month', 'day'], axis=1, inplace=True)
glt.head()
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
      <th>Anomaly</th>
      <th>Unc</th>
      <th>t_abs</th>
      <th>t</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1753-01-15</th>
      <td>-1.108</td>
      <td>3.646</td>
      <td>2.62</td>
      <td>1.512</td>
    </tr>
    <tr>
      <th>1753-02-15</th>
      <td>-1.652</td>
      <td>4.461</td>
      <td>3.23</td>
      <td>1.578</td>
    </tr>
    <tr>
      <th>1753-03-15</th>
      <td>1.020</td>
      <td>3.623</td>
      <td>5.33</td>
      <td>6.350</td>
    </tr>
    <tr>
      <th>1753-04-15</th>
      <td>-0.557</td>
      <td>3.898</td>
      <td>8.36</td>
      <td>7.803</td>
    </tr>
    <tr>
      <th>1753-05-15</th>
      <td>0.399</td>
      <td>1.652</td>
      <td>11.37</td>
      <td>11.769</td>
    </tr>
  </tbody>
</table>
</div>




```python
glt.t.plot(figsize=(12, 6), linewidth=0.5)
ax = plt.gca()
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("Temperature (°C)")
sns.despine()
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_48_0.png">
</p>


Now we resample the data to a yearly frequency.


```python
glty = glt.resample("AS").agg(['mean', 'min', 'max'])
```

Let us plot these yearly averaged temperature along with the 95% confidence interval.


```python
glty[('t', 'mean')].plot(figsize=(12, 6), label='Yearly mean')
ax = plt.gca()
ax.fill_between(glty.index, glty[('t', 'mean')] - 0.95*glty[('Unc', 'mean')], 
                            glty[('t', 'mean')] +  0.95*glty[('Unc', 'mean')], alpha=.25, label='95% confidence interval')
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("Temperature (°C)")
plt.title('Global land temperature')
plt.legend()
sns.despine()
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_52_0.png">
</p>


Now we plot both France's and global land's yearly mean.


```python
temp_annual[('tas', 'mean')].plot(figsize=(12, 6), label='France')
glty.loc[glty.index >= temp_annual.index.min(), ('t', 'mean')].plot(label='Global land')
ax = plt.gca()
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("Temperature (°C)")
plt.legend()
sns.despine()
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_54_0.png">
</p>


Well it seems like France is getting warmer faster than the average land surface.

### Average increasing rates

Now let's try to perform a linear regression on both temperature data between 1975 and 2015-2016.
In order to use OLS from statsmodels, we need to convert the `datetime` objects into real numbers. We do so using the `to_julian_date` tool. We also add a constant term since we need an intercept.


```python
df_france = temp_annual.loc[temp_annual.index >= pd.Timestamp('1975-01-01'), ('tas', 'mean')].copy(deep=True).reset_index()
df_france.columns = ['Date', 't']
df_france.set_index('Date', drop=False, inplace=True)
df_france['jDate'] = df_france['Date'].map(pd.Timestamp.to_julian_date)
df_france['const'] = 1.0
#epoch = pd.to_datetime(0, unit='d').to_julian_date()
#df_france['Date2'] = pd.to_datetime(df_france['jDate']- epoch, unit='D')

df_land = glty.loc[glty.index >= pd.Timestamp('1975-01-01'), ('t', 'mean')].copy(deep=True).reset_index()
df_land.columns = ['Date', 't']
df_land.set_index('Date', drop=False, inplace=True)
df_land['jDate'] = df_land['Date'].map(pd.Timestamp.to_julian_date)
df_land['const'] = 1.0
```


```python
mod = sm.OLS(df_france['t'], df_france[['jDate', 'const']])
res = mod.fit()
print(res.summary())
df_france['pred'] = res.predict(df_france[['jDate', 'const']])
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      t   R-squared:                       0.761
    Model:                            OLS   Adj. R-squared:                  0.755
    Method:                 Least Squares   F-statistic:                     124.4
    Date:                Thu, 24 May 2018   Prob (F-statistic):           1.07e-13
    Time:                        16:31:45   Log-Likelihood:                -24.189
    No. Observations:                  41   AIC:                             52.38
    Df Residuals:                      39   BIC:                             55.81
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    jDate          0.0002   1.62e-05     11.152      0.000       0.000       0.000
    const       -430.3375     39.620    -10.861      0.000    -510.477    -350.198
    ==============================================================================
    Omnibus:                        3.133   Durbin-Watson:                   1.971
    Prob(Omnibus):                  0.209   Jarque-Bera (JB):                1.820
    Skew:                          -0.254   Prob(JB):                        0.403
    Kurtosis:                       2.101   Cond. No.                     1.39e+09
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.39e+09. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
print('Temperature change per year: ', res.params[0]*365.0)
```

    Temperature change per year:  0.06583231229893599



```python
mod = sm.OLS(df_land['t'], df_land[['jDate', 'const']])
res = mod.fit()
print(res.summary())
df_land['pred'] = res.predict(df_land[['jDate', 'const']])
```

                                OLS Regression Results                            
    ==============================================================================
    Dep. Variable:                      t   R-squared:                       0.793
    Model:                            OLS   Adj. R-squared:                  0.788
    Method:                 Least Squares   F-statistic:                     153.0
    Date:                Thu, 24 May 2018   Prob (F-statistic):           3.00e-15
    Time:                        16:31:45   Log-Likelihood:                 14.544
    No. Observations:                  42   AIC:                            -25.09
    Df Residuals:                      40   BIC:                            -21.61
    Df Model:                           1                                         
    Covariance Type:            nonrobust                                         
    ==============================================================================
                     coef    std err          t      P>|t|      [0.025      0.975]
    ------------------------------------------------------------------------------
    jDate        7.56e-05   6.11e-06     12.368      0.000    6.32e-05     8.8e-05
    const       -176.0015     14.975    -11.753      0.000    -206.267    -145.736
    ==============================================================================
    Omnibus:                        4.505   Durbin-Watson:                   2.032
    Prob(Omnibus):                  0.105   Jarque-Bera (JB):                1.852
    Skew:                           0.062   Prob(JB):                        0.396
    Kurtosis:                       1.979   Cond. No.                     1.36e+09
    ==============================================================================
    
    Warnings:
    [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
    [2] The condition number is large, 1.36e+09. This might indicate that there are
    strong multicollinearity or other numerical problems.



```python
print('Temperature change per year: ', res.params[0]*365.0)
```

    Temperature change per year:  0.027594017104916324



```python
df_france.t.plot(figsize=(12, 6), label='France')
df_france.pred.plot(label='France linear regression')
df_land.t.plot(label='Land')
df_land.pred.plot(label='Land linear regression')
ax = plt.gca()
ax.grid(color= (0.1, 0.1, 0.1), linestyle='--', linewidth=1, alpha=0.25)
plt.autoscale(enable=True, axis='x', tight=True)
plt.ylabel("Temperature (°C)")
plt.legend()
sns.despine()
```

<p align="center">
  <img src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_files/2018-05-24-Pandas-Time-Series-example-with-some-historical-land-temperatures_61_0.png">
</p>


So over the last 40 years, France is getter hotter by 0.06583 degree per year in average, against 0.02759 degree for the global land.
