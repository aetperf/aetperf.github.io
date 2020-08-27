---
title: Lunch break, ridge plots with Bokeh
layout: post
comments: true
author: François Pacull
tags: Python Bokeh visualization Ridge
---

[Bokeh](https://bokeh.org/) is a great visualization Python library. In this short post, we are going to use it to create a ridge plot.

<p align="center">
  <img width="750" src="/img/2020-08-26_01/closeup.jpg" alt="closeup">
</p>

For that purpose, we use the [COVID-19 death data](https://github.com/CSSEGISandData/COVID-19) from Johns Hopkins University, and plot the daily normalized death rate (100000 * number of daily deaths / population) per EU(+UK) country.

## Imports



```python
import colorcet as cc
import numpy as np
import pandas as pd
from bokeh.io import show, output_notebook
from bokeh.models import ColumnDataSource, DatetimeTickFormatter
from bokeh.plotting import figure

output_notebook()

# Johns Hopkins University data url
URL = "https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv"
```



<div class="bk-root">
    <a href="https://bokeh.org" target="_blank" class="bk-logo bk-logo-small bk-logo-notebook"></a>
    <span id="1001">Loading BokehJS ...</span>
</div>




## Load and prepare the data

Load the COVID-19 data into a dataframe:


```python
deaths = pd.read_csv(URL)
deaths.head(2)
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
      <th>Province/State</th>
      <th>Country/Region</th>
      <th>Lat</th>
      <th>...</th>
      <th>8/24/20</th>
      <th>8/25/20</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>NaN</td>
      <td>Afghanistan</td>
      <td>33.93911</td>
      <td>...</td>
      <td>1389</td>
      <td>1397</td>
    </tr>
    <tr>
      <th>1</th>
      <td>NaN</td>
      <td>Albania</td>
      <td>41.15330</td>
      <td>...</td>
      <td>254</td>
      <td>259</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 221 columns</p>
</div>



Also load a list of EU countries:


```python
countries = (
    pd.read_csv(
        "https://pkgstore.datahub.io/opendatafortaxjustice/listofeucountries/listofeucountries_csv/data/5ab24e62d2ad8f06b59a0e7ffd7cb556/listofeucountries_csv.csv"
    )
    .values[:, 0]
    .tolist()
)

# Match country names
countries = [c if c != "Czech Republic" else "Czechia" for c in countries]
countries = [c if c != "Slovak Republic" else "Slovakia" for c in countries]

n_countries = len(countries)
print(countries)
```

    ['Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia', 'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Ireland', 'Italy', 'Latvia', 'Lithuania', 'Luxembourg', 'Malta', 'Netherlands', 'Poland', 'Portugal', 'Romania', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'United Kingdom']


We select EU countries in the COVID-19 data:


```python
deaths_eu = deaths.loc[deaths["Country/Region"].isin(countries)].copy(deep=True)

# cleanup
deaths_eu.drop(["Province/State", "Lat", "Long"], axis=1, inplace=True)
deaths_eu = deaths_eu.groupby("Country/Region").sum()  # with overseas territories
deaths_eu.index.name = "Country"
assert len(deaths_eu) == n_countries
deaths_eu.head(2)
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>...</th>
      <th>8/24/20</th>
      <th>8/25/20</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austria</th>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>733</td>
      <td>733</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>9996</td>
      <td>9996</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 217 columns</p>
</div>



Now we load the population count by country into a dataframe. The CSV file comes from [this](https://datahub.io/JohnSnowLabs/population-figures-by-country) website.


```python
pop = pd.read_csv(
    "./data/population-figures-by-country-csv_csv.csv",
    usecols=["Country", "Country_Code", "Year_2016"],
)
pop.loc[pop.Country == "Czech Republic", "Country"] = "Czechia"
pop.loc[pop.Country == "Slovak Republic", "Country"] = "Slovakia"
```

And select EU countries:


```python
pop_eu = pop[pop.Country.isin(countries)].copy(deep=True)
pop_eu.drop("Country_Code", axis=1, inplace=True)
pop_eu.set_index("Country", drop=True, inplace=True)
assert len(pop_eu) == n_countries
pop_eu.head(2)
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
      <th>Year_2016</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austria</th>
      <td>8747358.0</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>11348159.0</td>
    </tr>
  </tbody>
</table>
</div>



This population data date back to 2016, but it is recent enough for this blog post...

We compute the death density as the number of deaths per 100000 inhabitants for each country:


```python
dd_eu = deaths_eu.div(pop_eu.Year_2016, axis=0) * 100000
dd_eu.head(2)
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
      <th>1/22/20</th>
      <th>1/23/20</th>
      <th>...</th>
      <th>8/24/20</th>
      <th>8/25/20</th>
    </tr>
    <tr>
      <th>Country</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Austria</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>8.379673</td>
      <td>8.379673</td>
    </tr>
    <tr>
      <th>Belgium</th>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>88.084772</td>
      <td>88.084772</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 217 columns</p>
</div>



Now we pivot the dataframe, convert the index into a `DatetimeIndex`: 


```python
dd_eu = dd_eu.T
dd_eu.index = pd.to_datetime(dd_eu.index)
dd_eu.tail(2)
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
      <th>Country</th>
      <th>Austria</th>
      <th>Belgium</th>
      <th>...</th>
      <th>Sweden</th>
      <th>United Kingdom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-24</th>
      <td>8.379673</td>
      <td>88.084772</td>
      <td>...</td>
      <td>58.698661</td>
      <td>63.255251</td>
    </tr>
    <tr>
      <th>2020-08-25</th>
      <td>8.379673</td>
      <td>88.084772</td>
      <td>...</td>
      <td>58.708759</td>
      <td>63.279627</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 28 columns</p>
</div>



and compute a smoothed daily count of deaths per 100000 inhabitants:


```python
nd = 5
rate = (
    dd_eu.diff()
    .rolling(nd, center=True)
    .median()
    .rolling(3 * nd, center=False)
    .mean()
    .dropna()
)

rate.tail(2)
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
      <th>Country</th>
      <th>Austria</th>
      <th>Belgium</th>
      <th>...</th>
      <th>Sweden</th>
      <th>United Kingdom</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-22</th>
      <td>0.006859</td>
      <td>0.066971</td>
      <td>...</td>
      <td>0.026928</td>
      <td>0.015032</td>
    </tr>
    <tr>
      <th>2020-08-23</th>
      <td>0.006859</td>
      <td>0.066971</td>
      <td>...</td>
      <td>0.027601</td>
      <td>0.014423</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 28 columns</p>
</div>



Let's reorder the countries from lowest to highest maximum daily death rate:


```python
order = rate.max(axis=0).sort_values().index.values.tolist()
rate = rate[order]
rate.tail(2)
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
      <th>Country</th>
      <th>Latvia</th>
      <th>Slovakia</th>
      <th>...</th>
      <th>Spain</th>
      <th>Belgium</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-08-22</th>
      <td>0.0</td>
      <td>3.700743e-18</td>
      <td>...</td>
      <td>0.025694</td>
      <td>0.066971</td>
    </tr>
    <tr>
      <th>2020-08-23</th>
      <td>0.0</td>
      <td>3.700743e-18</td>
      <td>...</td>
      <td>0.029139</td>
      <td>0.066971</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 28 columns</p>
</div>



Here we duplicate the last row in order to later create nice looking Bokeh `Patches` (with a vertical line on the right side):


```python
rate = pd.concat([rate, rate.tail(1)], axis=0)
rate.iloc[-1] = 0.0
```

We choose a color palette (linear sampling):


```python
palette = [cc.rainbow[int(i * 9)] for i in range(len(order))]
```

Finally we can create the ridge plot.

## Plot

Most of the following code comes from Bokeh's [documentation](https://docs.bokeh.org/en/latest/docs/gallery/ridgeplot.html).

```python
def ridge(category, data, scale=5):
    return list(zip([category] * len(data), scale * data))


source = ColumnDataSource(data=dict(x=rate.index.values))
p = figure(
    y_range=order,
    plot_height=900,
    plot_width=900,
    toolbar_location=None,
    title="Daily normalized rate of COVID-19 deaths per EU(+UK) country",
)
p.title.text_font_size = "15pt"
p.xaxis.major_label_text_font_size = "10pt"
p.yaxis.major_label_text_font_size = "10pt"

for i, country in enumerate(order):
    y = ridge(country, rate[country])
    source.add(y, country)
    p.patch(
        "x",
        country,
        color=palette[i],
        alpha=0.25,
        line_color="black",
        line_alpha=0.5,
        source=source,
    )

p.outline_line_color = None
p.background_fill_color = "#efefef"

p.xaxis.formatter = DatetimeTickFormatter(days="%m/%d")

p.ygrid.grid_line_color = None
p.xgrid.grid_line_color = "#dddddd"
p.xgrid.ticker = p.xaxis.ticker

p.axis.minor_tick_line_color = None
p.axis.major_tick_line_color = None
p.axis.axis_line_color = None

p.y_range.range_padding = 0.85

show(p)
```





<p align="center">
  <img width="750" src="/img/2020-08-26_01/output_01.jpg" alt="Ridge plot">
</p>





The highest rate in this plot was reached in Belgium:


```python
rate["Belgium"].max()
```




    2.4268253555488593




```python
str(rate["Belgium"].idxmax().date())
```




    '2020-04-21'


