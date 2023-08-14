---
title: Python plot - Antarctic Sea Ice Extent
layout: post
comments: true
author: François Pacull
tags: 
- Python
- Pandas
- Matplotlib
---

Data source : https://ads.nipr.ac.jp/vishop/#/extent   
REGION SELECTOR = Antarctic  
At the bottom of the page : *Download the sea ice extent (CSV file) - seasonal dataset*  

From the [National Institute of Polar Research (Japan) website](https://ads.nipr.ac.jp/vishop/#/extent):

> The sea-ice extent is calculated as the areal sum of sea ice covering the ocean where sea-ice concentration (SIC) exceeds a threshold (15% for AMSR-E). SICs are derived from various satellite-borne passive microwave radiometer (PMR) sensors using the algorithm developed and provided by Dr. Comiso of NASA GSFC through a cooperative relationship between NASA and JAXA. The following sensor's data were used;  
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
      <th>#num</th>
      <th>month</th>
      <th>date</th>
      <th>time[second]</th>
      <th>1980's Average</th>
      <th>1990's Average</th>
      <th>2000's Average</th>
      <th>2010's Average</th>
      <th>1978</th>
      <th>1979</th>
      <th>...</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
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
      <td>0</td>
      <td>6803077</td>
      <td>6673152</td>
      <td>7042962</td>
      <td>6995424.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>8360560.0</td>
      <td>9219873.0</td>
      <td>6942134</td>
      <td>5373844.0</td>
      <td>6351252.0</td>
      <td>5263487.0</td>
      <td>6068206</td>
      <td>6568300.0</td>
      <td>5778923.0</td>
      <td>4777328.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>86400</td>
      <td>6623463</td>
      <td>6524885</td>
      <td>6870609</td>
      <td>6820823.4</td>
      <td>NaN</td>
      <td>6988174.0</td>
      <td>...</td>
      <td>8220524.0</td>
      <td>9116202.0</td>
      <td>6696112</td>
      <td>5223903.0</td>
      <td>6190309.0</td>
      <td>5115518.0</td>
      <td>5966632</td>
      <td>6451907.0</td>
      <td>5570948.0</td>
      <td>4644691.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>1</td>
      <td>3</td>
      <td>172800</td>
      <td>6433937</td>
      <td>6381587</td>
      <td>6686831</td>
      <td>6668285.5</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>...</td>
      <td>8094706.0</td>
      <td>9022589.0</td>
      <td>6491773</td>
      <td>5148723.0</td>
      <td>6046952.0</td>
      <td>4977912.0</td>
      <td>5858135</td>
      <td>6325097.0</td>
      <td>5411740.0</td>
      <td>4485865.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 54 columns</p>
</div>



We remove columns that are not specific years (`#num`, `month`, `date`, `time[second]`, `1980's Average`, ...)


```python
cols = df.columns
cols = [c for c in cols if (len(c) == 4) and c.isnumeric() and (c.startswith("19") or c.startswith("20"))]
df = df[cols]
df = df.astype(float)
df = df.replace(-9999.0, np.nan)
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
      <th>1978</th>
      <th>1979</th>
      <th>1980</th>
      <th>1981</th>
      <th>1982</th>
      <th>1983</th>
      <th>1984</th>
      <th>1985</th>
      <th>1986</th>
      <th>1987</th>
      <th>...</th>
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
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
      <td>6199770.0</td>
      <td>NaN</td>
      <td>6510515.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7703813.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>8360560.0</td>
      <td>9219873.0</td>
      <td>6942134.0</td>
      <td>5373844.0</td>
      <td>6351252.0</td>
      <td>5263487.0</td>
      <td>6068206.0</td>
      <td>6568300.0</td>
      <td>5778923.0</td>
      <td>4777328.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>6988174.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7038279.0</td>
      <td>NaN</td>
      <td>6860933.0</td>
      <td>6402066.0</td>
      <td>NaN</td>
      <td>6819079.0</td>
      <td>...</td>
      <td>8220524.0</td>
      <td>9116202.0</td>
      <td>6696112.0</td>
      <td>5223903.0</td>
      <td>6190309.0</td>
      <td>5115518.0</td>
      <td>5966632.0</td>
      <td>6451907.0</td>
      <td>5570948.0</td>
      <td>4644691.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>5855460.0</td>
      <td>5733736.0</td>
      <td>NaN</td>
      <td>6174902.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>7377712.0</td>
      <td>NaN</td>
      <td>...</td>
      <td>8094706.0</td>
      <td>9022589.0</td>
      <td>6491773.0</td>
      <td>5148723.0</td>
      <td>6046952.0</td>
      <td>4977912.0</td>
      <td>5858135.0</td>
      <td>6325097.0</td>
      <td>5411740.0</td>
      <td>4485865.0</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 46 columns</p>
</div>



## Non-leap years

The dataframe columns correspond to years and rows to (month , day) combinations. Thus, February 29th has missing values on non-leap years. We shift the values on these years in order to have a day-of-year row index without missing values on the 29th of February:


```python
df.iloc[58:61][[str(y) for y in range(2014, 2024)]]
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
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
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
      <td>2151409.0</td>
      <td>2407226.0</td>
      <td>2483470.0</td>
      <td>2792735.0</td>
      <td>3197871.0</td>
      <td>2211479.0</td>
      <td>2063912.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>2853039.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2786252.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>60</th>
      <td>3772864.0</td>
      <td>3807337.0</td>
      <td>2863943.0</td>
      <td>2147345.0</td>
      <td>2452571.0</td>
      <td>2496269.0</td>
      <td>2780405.0</td>
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
      <th>2014</th>
      <th>2015</th>
      <th>2016</th>
      <th>2017</th>
      <th>2018</th>
      <th>2019</th>
      <th>2020</th>
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
      <td>2151409.0</td>
      <td>2407226.0</td>
      <td>2483470.0</td>
      <td>2792735.0</td>
      <td>3197871.0</td>
      <td>2211479.0</td>
      <td>2063912.0</td>
    </tr>
    <tr>
      <th>59</th>
      <td>3772864.0</td>
      <td>3807337.0</td>
      <td>2853039.0</td>
      <td>2147345.0</td>
      <td>2452571.0</td>
      <td>2496269.0</td>
      <td>2786252.0</td>
      <td>3336462.0</td>
      <td>2231068.0</td>
      <td>2095439.0</td>
    </tr>
    <tr>
      <th>60</th>
      <td>3805451.0</td>
      <td>3803617.0</td>
      <td>2863943.0</td>
      <td>2149610.0</td>
      <td>2457657.0</td>
      <td>2543193.0</td>
      <td>2780405.0</td>
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
  <img width="1000" src="/img/2023-06-13_01/output_13_0.png" alt="output_13_0">
</p>


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