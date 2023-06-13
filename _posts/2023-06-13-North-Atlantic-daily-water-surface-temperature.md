---
title: North Atlantic daily water surface temperature
layout: post
comments: true
author: François Pacull
tags: 
- Python
- Pandas
---

Data source : [https://climatereanalyzer.org](https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json) (NOAA Optimum Interpolation SST (OISST) dataset version 2.1)


```python
import matplotlib.pyplot as plt
import pandas as pd
plt.style.use("fivethirtyeight")

FS = (12, 7)  # figure size
```

## Load the Data


```python
df = pd.read_json(
    path_or_buf="https://climatereanalyzer.org/clim/sst_daily/json/oisst2.1_natlan1_sst_day.json"
)
df.set_index("name", inplace=True)
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
      <th>data</th>
    </tr>
    <tr>
      <th>name</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1981</th>
      <td>[None, None, None, None, None, None, None, Non...</td>
    </tr>
    <tr>
      <th>1982</th>
      <td>[20.13, 20.06, 20.0, 20.01, 19.99, 19.98, 19.9...</td>
    </tr>
    <tr>
      <th>1983</th>
      <td>[19.76, 19.77, 19.77, 19.75, 19.7, 19.68, 19.6...</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["data"].map(len).unique()
```




    array([366])




```python
df = pd.DataFrame(df["data"].to_list(), columns=list(range(1, 367)), index=df.index)
df = df.T
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
      <th>name</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>1988</th>
      <th>1989</th>
      <th>1990</th>
      <th>...</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
      <th>1982-2011 mean</th>
      <th>plus 2σ</th>
      <th>minus 2σ</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>20.13</td>
      <td>19.76</td>
      <td>19.90</td>
      <td>19.76</td>
      <td>19.57</td>
      <td>19.72</td>
      <td>20.05</td>
      <td>19.82</td>
      <td>19.84</td>
      <td>...</td>
      <td>20.56</td>
      <td>20.77</td>
      <td>20.30</td>
      <td>20.60</td>
      <td>20.66</td>
      <td>20.85</td>
      <td>20.78</td>
      <td>20.09</td>
      <td>20.66</td>
      <td>19.53</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>20.06</td>
      <td>19.77</td>
      <td>19.92</td>
      <td>19.74</td>
      <td>19.53</td>
      <td>19.71</td>
      <td>20.03</td>
      <td>19.80</td>
      <td>19.80</td>
      <td>...</td>
      <td>20.56</td>
      <td>20.71</td>
      <td>20.27</td>
      <td>20.58</td>
      <td>20.63</td>
      <td>20.82</td>
      <td>20.76</td>
      <td>20.07</td>
      <td>20.63</td>
      <td>19.51</td>
    </tr>
    <tr>
      <th>3</th>
      <td>NaN</td>
      <td>20.00</td>
      <td>19.77</td>
      <td>19.93</td>
      <td>19.72</td>
      <td>19.51</td>
      <td>19.71</td>
      <td>20.00</td>
      <td>19.78</td>
      <td>19.76</td>
      <td>...</td>
      <td>20.54</td>
      <td>20.66</td>
      <td>20.22</td>
      <td>20.55</td>
      <td>20.61</td>
      <td>20.80</td>
      <td>20.73</td>
      <td>20.05</td>
      <td>20.62</td>
      <td>19.49</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 46 columns</p>
</div>



## Long time range daily mean


```python
year_start = 1982
year_end = 2012
df[f"{year_start}-{year_end} mean"] = df[
    [str(y) for y in range(year_start, year_end + 1)]
].mean(axis=1)
```

## Figure


```python
alpha_1 = 0.2
color_1 = "silver"
alpha_2 = 0.4
color_2 = "grey"
color_3 = "black"

ax = (df["1982"] - df["1982-2012 mean"]).plot(figsize=FS, color=color_1, alpha=alpha_1)
for year in [str(y) for y in range(1983, 2012)]:
    ax = (df[year] - df["1982-2012 mean"]).plot(ax=ax, color=color_1, alpha=alpha_1)
ax = (df["2012"] - df["1982-2012 mean"]).plot(
    ax=ax, color=color_1, label="1982-2012", alpha=alpha_1
)

ax = (df["2013"] - df["1982-2012 mean"]).plot(ax=ax, color=color_2, alpha=alpha_2)
for year in [str(y) for y in range(2014, 2022)]:
    ax = (df[year] - df["1982-2012 mean"]).plot(ax=ax, color=color_2, alpha=alpha_2)
ax = (df["2022"] - df["1982-2012 mean"]).plot(
    ax=ax, color=color_2, label="2013-2022", alpha=alpha_2
)

ax = (df["2023"] - df["1982-2012 mean"]).plot(
    ax=ax, color=color_3, label="2023", alpha=1.0
)
_ = ax.set_ylim(-1.0, +1.5)
_ = ax.legend()
_ = ax.set(
    title="North Atlantic daily water surface temperature anomaly\n w.r.t. 1982-2012 mean",
    xlabel="Day of year",
    ylabel="Tempetature anomaly (°C)",
)
```


<p align="center">
  <img width="1000" src="/img/2023-06-13_01/output_9_0.png" alt="output_9_0">
</p>