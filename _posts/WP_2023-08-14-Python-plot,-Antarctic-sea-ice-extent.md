# Python plot - Antarctic sea ice extent

Data source : https://ads.nipr.ac.jp/vishop/#/extent   
REGION SELECTOR = Antarctic  
At the bottom of the page : *Download the sea ice extent (CSV file) - seasonal dataset*  

From the [National Institute of Polar Research (Japan) website](https://ads.nipr.ac.jp/vishop/#/extent):

> The sea-ice extent is calculated as the areal sum of sea ice covering the ocean where sea-ice concentration [SIC] exceeds a threshold [15% for AMSR-E]. SICs are derived from various satellite-borne passive microwave radiometer [PMR] sensors using the algorithm developed and provided by Dr. Comiso of NASA GSFC through a cooperative relationship between NASA and JAXA. The following sensor's data were used;  
•	Jan. 1980 ～ Jul. 1987	：	SMMR  
•	Jul. 1987 ～ Jun. 2002	：	SSM/I  
•	Jun. 2002 ～ Oct. 2011	：	AMSR-E  
•	Oct. 2011 ～ Jul. 2012	：	WindSat  
•	Jul. 2012 ～ the present	：	AMSR2  


```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
plt.style.use("fivethirtyeight")

CSV_FILE_PATH = "./VISHOP_EXTENT_GRAPH_Antarctic.csv"
FS = (12, 7)  # figure size
```

## Load the Data


```python
df = pd.read_csv(CSV_FILE_PATH)
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>#num</th>
      <th>month</th>
      <th>date</th>
      <th>...</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>...</td>
      <td>6568300.0</td>
      <td>5778923.0</td>
      <td>4777328.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>...</td>
      <td>6451907.0</td>
      <td>5570948.0</td>
      <td>4644691.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>...</td>
      <td>6325097.0</td>
      <td>5411740.0</td>
      <td>4485865.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 54 columns</p>
</div>



We remove columns that are not specific years [`#num`, `month`, `date`, `time[second]`, `1980's Average`, ...]


```python
cols = df.columns
cols = [c for c in cols if (len(c) == 4) and c.isnumeric() and (c.startswith("19") or c.startswith("20"))]
df = df[cols]
df = df.astype(float)
df = df.replace(-9999.0, np.nan)
df.head(3)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>...</th>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5966499.0</td>
      <td>...</td>
      <td>6568300.0</td>
      <td>5778923.0</td>
      <td>4777328.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>6988174.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>6451907.0</td>
      <td>5570948.0</td>
      <td>4644691.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5855460.0</td>
      <td>...</td>
      <td>6325097.0</td>
      <td>5411740.0</td>
      <td>4485865.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 46 columns</p>
</div>



## Non-leap years

The dataframe columns correspond to years and rows to [month , day] combinations. Thus, February 29th has missing values on non-leap years. We shift the values on these years in order to have a day-of-year row index without missing values on the 29th of February:


```python
df.iloc[58:61][[str(y) for y in range(2014, 2024)]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <td>...</td>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>3753923.0</td>
      <td>3800984.0</td>
      <td>2857127.0</td>
      <td>...</td>
      <td>3197871.0</td>
      <td>2211479.0</td>
      <td>2063912.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2853039.0</td>
      <td>...</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60</th>
      <td>3772864.0</td>
      <td>3807337.0</td>
      <td>2863943.0</td>
      <td>...</td>
      <td>3336462.0</td>
      <td>2231068.0</td>
      <td>2095439.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
for year in range(1978, 2024):
    if (year - 1972) % 4 != 0:
        year_str = str(year)
        if year_str in df:
            df.loc[59:365, year_str] = df.loc[59:365, year_str].shift(-1)
```


```python
df.iloc[58:61][[str(y) for y in range(2014, 2024)]]
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <td>...</td>
      <th>2021</th>
      <th>2022</th>
      <th>2023</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>58</th>
      <td>3753923.0</td>
      <td>3800984.0</td>
      <td>2857127.0</td>
      <td>...</td>
      <td>3197871.0</td>
      <td>2211479.0</td>
      <td>2063912.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3772864.0</td>
      <td>3807337.0</td>
      <td>2853039.0</td>
      <td>...</td>
      <td>3336462.0</td>
      <td>2231068.0</td>
      <td>2095439.0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>3805451.0</td>
      <td>3803617.0</td>
      <td>2863943.0</td>
      <td>...</td>
      <td>3474391.0</td>
      <td>2263092.0</td>
      <td>2095754.0</td>
    </tr>
  </tbody>
</table>
</div>



## Long time range daily mean


```python
year_start = 1978
year_end = 2012
current_year = 2023
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

ax = (df[f"{year_start}"] - df[f"{year_start}-{year_end} mean"]).plot(
    figsize=FS, color=color_1, alpha=alpha_1
)
for year in [str(y) for y in range(year_start + 1, year_end)]:
    ax = (df[year] - df[f"{year_start}-{year_end} mean"]).plot(
        ax=ax, color=color_1, alpha=alpha_1
    )
ax = (df[f"{year_end}"] - df[f"{year_start}-{year_end} mean"]).plot(
    ax=ax, color=color_1, label=f"{year_start}-{year_end}", alpha=alpha_1
)

ax = (df[f"{year_end+1}"] - df[f"{year_start}-{year_end} mean"]).plot(
    ax=ax, color=color_2, alpha=alpha_2
)
for year in [str(y) for y in range(year_end + 2, current_year - 1)]:
    ax = (df[year] - df[f"{year_start}-{year_end} mean"]).plot(
        ax=ax, color=color_2, alpha=alpha_2
    )
ax = (df[f"{current_year-1}"] - df[f"{year_start}-{year_end} mean"]).plot(
    ax=ax, color=color_2, label=f"{year_end+1}-{current_year-1}", alpha=alpha_2
)

ax = (df[f"{current_year}"] - df[f"{year_start}-{year_end} mean"]).plot(
    ax=ax, color=color_3, label=f"{current_year}", alpha=1.0
)

plt.hlines(y = 0, xmin = 0, xmax = 365, alpha=0.7, linewidth=0.5)
_ = ax.set_ylim(-3e6, +3e6)
_ = ax.legend()
_ = ax.set(
    title=f"Antarctic sea ice extent anomaly \n w.r.t. {year_start}-{year_end} mean",
    xlabel="Day of year",
    ylabel="Sea ice extent anomaly (million $km^2$)",
)
```


<p align="center">
  <img width="1000" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2023-08-14_01/output_13_0.png" alt="output_13_0">
</p>
