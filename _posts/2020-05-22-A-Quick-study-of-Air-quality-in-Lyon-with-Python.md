---
title: A Quick study of air quality in Lyon with Python
layout: post
comments: true
author: François Pacull
tags: Python Pandas GET API
---

The aim of this post is to use Python to fetch air quality data from a web service and make some plots. 

We are going to look at some data from earlier this year, before and after the lockdown and eventually compare 2020 with some from previous years, in the city of Lyon, France. 

The data is provided by an institute called [Atmo](https://www.atmo-auvergnerhonealpes.fr/) monitoring air quality over the Auvergne-Rhône-Alpes region in France. They also come up with an API (HTTP GET method). An API token is required to use it, that you get from their website after registration.



## Imports


```python
import requests
import pandas as pd
from datetime import datetime
from fastprogress.fastprogress import progress_bar
from matplotlib import pyplot as plt

plt.style.use("seaborn")
%load_ext lab_black

FS = (16, 9)  # figure size

# location: LYON 4EME ARRONDISSEMENT (4th district)
CODE_INSEE = 69384

# api-token
API_TOKEN = "8ff64072eafffb9f02d20db396001d07"

# api-endpoint
URL = f"https://api.atmo-aura.fr/communes/{CODE_INSEE}/indices?api_token={API_TOKEN}"
```

## First try of the API 


```python
PARAMS = {"date": "2020-05-24"}
r = requests.get(url=URL, params=PARAMS)
data = r.json()
data
```




    {'licence': 'https://opendatacommons.org/licenses/odbl/',
     'commune': 'LYON-4E-ARRONDISSEMENT',
     'code_insee': '69384',
     'indices': {'date': '2020-05-24',
      'valeur': '37.5560109609897',
      'couleur_html': '#99E600',
      'qualificatif': 'Bon',
      'type_valeur': 'prévision'}}



Since we will not change the location (4th district in Lyon), we only need to keep the `valeur` key from this dictionary, which is the air pollution index value. It usually ranges from 0 to 100, but can also exceeds 100 in case of warning threshold is exceeded, which represents a risk to the health of the whole population or to environmental degradation justifying the implementation of emergency measures.

Note that you can get historic values but also forecasts if you enter a future date in the `get` params (2 day horizon at most whe I tried that).

So let's fetch the daily pollution index on a temporal range starting at the begining of 2020 and ending today.

## Air Quality in 2020 so far

First we create a date range:


```python
start = datetime(2020, 1, 1).date()
end = datetime.now().date()
day_range = pd.date_range(start=start, end=end, freq="D")
```

Now we can loop on these days and fetch the air quality data for each day. Since each item in `day_range` is a `Timestamp` object, we convert them to `str` with `strftime`:


```python
daily_pollution_index = []
for day in progress_bar(day_range):
    day_str = day.strftime("%Y-%m-%d")
    params = {"date": day_str}
    r = requests.get(url=URL, params=params)
    data = r.json()
    daily_pollution_index.append(data["indices"])
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='143' class='' max='143' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [143/143 00:21<00:00]
</div>



The list is given to Pandas to create a dataframe:


```python
df = pd.DataFrame(daily_pollution_index)
df.head(2)
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
      <th>date</th>
      <th>valeur</th>
      <th>couleur_html</th>
      <th>qualificatif</th>
      <th>type_valeur</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2020-01-01</td>
      <td>50.4253716534476</td>
      <td>#FFFF00</td>
      <td>Moyen</td>
      <td>réelle</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2020-01-02</td>
      <td>48.5803761861679</td>
      <td>#C3F000</td>
      <td>Bon</td>
      <td>réelle</td>
    </tr>
  </tbody>
</table>
</div>



We clean up the dataframe a little bit, and create a `DatetimeIndex`:


```python
df.rename(
    columns={"valeur": "value",}, inplace=True,
)
df.drop(["type_valeur", "couleur_html", "qualificatif"], axis=1, inplace=True)
df["value"] = df["value"].astype(float)
df["date"] = pd.to_datetime(df["date"])
df.set_index("date", inplace=True)
df.head(2)
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>50.425372</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>48.580376</td>
    </tr>
  </tbody>
</table>
</div>



And we can now plot the daily historical air pollution index.


```python
ax = df["value"].plot(
    ms=10, linewidth=3, alpha=0.7, style="o-", figsize=FS, c="#1f77b4",
)
_ = plt.vlines("2020-03-17", 0, 100, colors="r")  # lockdown start
_ = plt.vlines("2020-05-11", 0, 100, colors="r")  # lockdown end
_ = ax.set_ylim(0, 100)
_ = ax.set(
    title=f"Daily air quality in Lyon 4E",
    xlabel="Date",
    ylabel="Air pollution index (smaller is better)",
)
```


![png](/img/2020-05-22_01/output_16_0.png)


Well we do not see a significant drop in air pollution during the lockdown that took place between march 17 and may 11. The reason is that the air pollution index is combining 3 different kinds of pollutants:
- NO2: Nitrogen dioxide,
- O3: Ozone, 
- PM10: particulate matter, which are coarse particles with a diameter between 2.5 and 10 micrometers (μm) 

From what I understand, an index is computed for each of these three pollutants, and the air pollution index corresponds a combination of these three values. So the pollution index will keep being large if any of these pollutant concentrations remain large.

Now theses pollutants have very different origins: NO2 is mainly related to road traffic and other fossil fuel combustion processes, PM10 to wood heating (in the suburb) and agriculture. O3 is more complex; here is a quote from [wikipedia](https://en.wikipedia.org/wiki/Ozone#Ozone_air_pollution):

> Ozone precursors are a group of pollutants, predominantly those emitted during the combustion of fossil fuels. Ground-level ozone pollution (tropospheric ozone) is created near the Earth's surface by the action of daylight UV rays on these precursors. The ozone at ground level is primarily from fossil fuel precursors, but methane is a natural precursor, and the very low natural background level of ozone at ground level is considered safe.

Also, these levels of pollution also depends on the weather. A cold dry weather may yield more PM10 pollution due to heating. A sunny weather without wind may increase the level of ozone pollution, while some rain may "clean" the particles.



Concerning the first part of 2020, let's try to compare the above pollution values with the ones recorded in the 2 previous years.

## Air Quality from 2018 to 2020

We already have the data for 2020, so we now fetch the data for the same range of days (january 1 to may 22), but for 2018 and 2019. The data from previous years (<2018) does not seem to be available on the web service.


```python
daily_pollution_index = []
for year in range(2018, 2020):
    for day in progress_bar(day_range):
        try:
            day = day.replace(year=year).date()
            day_str = day.strftime("%Y-%m-%d")
            params = {"date": day_str}
            r = requests.get(url=URL, params=params)
            data = r.json()
            if data is not None:
                if data["indices"] is not None:
                    daily_pollution_index.append(data["indices"])
        except:
            print(f"No data for the date: {day.date()}")
```



<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='143' class='' max='143' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [143/143 00:22<00:00]
</div>



    No data for the date: 2020-02-29




<div>
    <style>
        /* Turns off some styling */
        progress {
            /* gets rid of default border in Firefox and Opera. */
            border: none;
            /* Needs to be in here for Safari polyfill so background images work as expected. */
            background-size: auto;
        }
        .progress-bar-interrupted, .progress-bar-interrupted::-webkit-progress-bar {
            background: #F44336;
        }
    </style>
  <progress value='143' class='' max='143' style='width:300px; height:20px; vertical-align: middle;'></progress>
  100.00% [143/143 00:21<00:00]
</div>



    No data for the date: 2020-02-29


Similarly to what we did before, we create a dataframe with a `DatetimeIndex`:


```python
df_hist = pd.DataFrame(daily_pollution_index)
df_hist.rename(
    columns={"valeur": "value",}, inplace=True,
)
df_hist.drop(["type_valeur", "couleur_html", "qualificatif"], axis=1, inplace=True)
df_hist["value"] = df_hist["value"].astype(float)
df_hist["date"] = pd.to_datetime(df_hist["date"])
df_hist.set_index("date", inplace=True)
df_hist.head(2)
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
      <th>value</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2018-01-01</th>
      <td>32.346865</td>
    </tr>
    <tr>
      <th>2018-01-02</th>
      <td>31.257633</td>
    </tr>
  </tbody>
</table>
</div>



Now we merge the data corresponding to the 3 different years:


```python
df.rename(
    columns={"value": "2020",}, inplace=True,
)
df["month"] = df.index.month
df["day"] = df.index.day
df.reset_index(drop=False, inplace=True)
df_hist["month"] = df_hist.index.month
df_hist["day"] = df_hist.index.day
df_all = pd.merge(df, df_hist["2018"], on=["month", "day"], how="left").rename(
    columns={"value": "2018"}
)
df_all = pd.merge(df_all, df_hist["2019"], on=["month", "day"], how="left").rename(
    columns={"value": "2019"}
)
df_all.drop(["month", "day"], axis=1, inplace=True)
df_all.set_index("date", inplace=True)
df_all.head(2)
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
      <th>2020</th>
      <th>2018</th>
      <th>2019</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>50.425372</td>
      <td>32.346865</td>
      <td>21.538545</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>48.580376</td>
      <td>31.257633</td>
      <td>31.926375</td>
    </tr>
  </tbody>
</table>
</div>



Let's plot the monthly average:


```python
monthly = df_all.resample("M").mean()
monthly["month"] = monthly.index.month_name()
monthly.set_index("month", inplace=True, drop=True)
ax = monthly.plot.bar(figsize=FS, color=colors, rot=10)
_ = ax.set_ylim(0, 100)
_ = ax.set(
    title=f"Daily air quality in Lyon 4E",
    xlabel="Date",
    ylabel="Air pollution index (smaller is better)",
)
```


![png](/img/2020-05-22_01/output_25_0.png)


When there is  no significant difference between the different years.

Unfortunately the web service does not offer separate data for each pollutant. However these data are available from their website [here](https://www.atmo-auvergnerhonealpes.fr/donnees/telecharger), as CSV files.

## Evolution of each pollutant

We download the data for the 3 pollutants seen earlier, on the same time range and with the same frequency. Let's look at the first file:


```python
NO2 = pd.read_csv("NO2.csv", sep=";")
NO2
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
      <th>Station</th>
      <th>Polluant</th>
      <th>Mesure</th>
      <th>Unité</th>
      <th>01/01/2020</th>
      <th>02/01/2020</th>
      <th>03/01/2020</th>
      <th>04/01/2020</th>
      <th>05/01/2020</th>
      <th>06/01/2020</th>
      <th>...</th>
      <th>13/05/2020</th>
      <th>14/05/2020</th>
      <th>15/05/2020</th>
      <th>16/05/2020</th>
      <th>17/05/2020</th>
      <th>18/05/2020</th>
      <th>19/05/2020</th>
      <th>20/05/2020</th>
      <th>21/05/2020</th>
      <th>22/05/2020</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Lyon Centre</td>
      <td>Dioxyde d'azote</td>
      <td>Dioxyde d'azote</td>
      <td>microg/m3</td>
      <td>38,2</td>
      <td>42,6</td>
      <td>64,4</td>
      <td>34,5</td>
      <td>25,4</td>
      <td>54,1</td>
      <td>...</td>
      <td>17</td>
      <td>8,3</td>
      <td>10,6</td>
      <td>5,3</td>
      <td>4,4</td>
      <td>7,8</td>
      <td>4,7</td>
      <td>6,1</td>
      <td>6,4</td>
      <td>18,3</td>
    </tr>
  </tbody>
</table>
<p>1 rows × 147 columns</p>
</div>



We need to transpose the dataframe in order to create a `DatetimeIndex`, and take care of the french number and date formats:


```python
NO2.drop(["Station", "Polluant", "Mesure", "Unité"], axis=1, inplace=True)
NO2 = NO2.T
NO2.index = pd.to_datetime(NO2.index, format="%d/%m/%Y")
NO2.columns = ["NO2"]
NO2["NO2"] = NO2["NO2"].astype(str).map(lambda s: s.replace(",", ".")).astype(float)
NO2.head(2)
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
      <th>NO2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>38.2</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>42.6</td>
    </tr>
  </tbody>
</table>
</div>



Now we do the same with the two other files and concatenate them:


```python
O3 = pd.read_csv("O3.csv", sep=";")
O3.drop(["Station", "Polluant", "Mesure", "Unité"], axis=1, inplace=True)
O3 = O3.T
O3.index = pd.to_datetime(O3.index, format="%d/%m/%Y")
O3.columns = ["O3"]
O3["O3"] = O3["O3"].astype(str).map(lambda s: s.replace(",", ".")).astype(float)
PM10 = pd.read_csv("PM10.csv", sep=";")
PM10.drop(["Station", "Polluant", "Mesure", "Unité"], axis=1, inplace=True)
PM10 = PM10.T
PM10.index = pd.to_datetime(PM10.index, format="%d/%m/%Y")
PM10.columns = ["PM10"]
PM10["PM10"] = PM10["PM10"].astype(str).map(lambda s: s.replace(",", ".")).astype(float)
pol = pd.concat([NO2, O3, PM10], axis=1)
pol.head(2)
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
      <th>NO2</th>
      <th>O3</th>
      <th>PM10</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-01-01</th>
      <td>38.2</td>
      <td>0.8</td>
      <td>38.2</td>
    </tr>
    <tr>
      <th>2020-01-02</th>
      <td>42.6</td>
      <td>0.9</td>
      <td>35.1</td>
    </tr>
  </tbody>
</table>
</div>



Now here is a little function to plot each pollutant:


```python
def plot_pollutant(df, pollutant, unit):
    ax = pol[pollutant].plot(
        ms=10,
        linewidth=3,
        alpha=0.6,
        style="-o",
        legend=False,
        color=colors[0],
        figsize=FS,
    )
    max_val = pol[pollutant].max()
    _ = plt.vlines("2020-03-17", 0, max_val, colors="r")  # lockdown start
    _ = plt.vlines("2020-05-11", 0, max_val, colors="r")  # lockdown end
    _ = ax.set(
        title=f"Daily {pollutant} cencentration in Lyon centre",
        xlabel="Date",
        ylabel=f"{pollutant} cencentration ({unit})",
    )
```


```python
plot_pollutant(pol, "NO2", "microg/m3")
```


![png](/img/2020-05-22_01/output_35_0.png)



```python
plot_pollutant(pol, "O3", "microg/m3")
```


![png](/img/2020-05-22_01/output_36_0.png)



```python
plot_pollutant(pol, "PM10", "microg/m3")
```


![png](/img/2020-05-22_01/output_37_0.png)


So clearly the NO2 concentration dropped at the begining of the lockdown alongside the intensity of road traffic.


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