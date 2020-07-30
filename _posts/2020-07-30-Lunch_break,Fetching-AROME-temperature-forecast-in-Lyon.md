---
title: Lunch break, fetching AROME temperature forecast in Lyon
layout: post
comments: true
author: François Pacull
tags: Python time series temperature AROME web services météo france
---


Since a "small" heat wave is coming, I would like to get some temperature forecast in my neighborhood for the next hours, from my [JupyterLab](https://github.com/jupyterlab/jupyterlab) notebook. We are going to fetch the results from the Météo-France AROME 0.01 model. Here is the AROME item listed in the Météo-France web service documentation: 

> Results from the French high resolution atmospheric forecast model (called AROME) on a grid with a resolution of 0°01 or 0°025 for France. Data is updated every 3 hours and available up to 42 hours, with a temporal resolution of 1 hour.

The weather forecast data are under an [ETALAB](https://www.etalab.gouv.fr/licence-ouverte-open-licence) open license. Note that an access request must be made to support.inspire@meteo.fr in order to get some credentials for these web services.

We are going to use the [PyMeteoFr](https://github.com/aetperf/pymeteofr) package, which is a Python wrapper of the Météo-France web services. Note that this package is young and did not reach yet a stable level.

<p align="center">
  <img width="600" src="/img/2020-07-30_01/Homer.gif" alt="Homer">
</p>

## Imports


```python
from datetime import datetime, timezone

import pytz

from pymeteofr import Fetcher

%load_ext lab_black

TOKEN = "__xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx__"  # API key
```

## Fetching the temperature forecast

Let's instanciate a PyMeteoFr `Fetcher` instance with the API key:


```python
fetcher = Fetcher(token=TOKEN)
```

Now that the token is set, we need to choose which weather-related product we are interested in. This is done using the `select_product` method with appropriate arguments: `dataset`, `area` and `accuracy`:


```python
fetcher.select_product(dataset="arome", area="france", accuracy=0.01)
```

    -- GetCapabilities request --


Then we choose a meteorological variable from AROME: `Temperature at specified height level above ground`, and a forecasting horizon: 35 hours


```python
fetcher.list_titles()
```




    ['Brightness temperature',
     'Convective available potential energy',
     'Geometric height',
     'High cloud cover',
     'Low cloud cover',
     'Medium cloud cover',
     'Pressure',
     'Relative humidity at specified height level above ground',
     'Temperature at specified height level above ground',
     'Wind speed (gust) at specified height level above ground',
     'Wind speed at specified height level above ground',
     'rainfall rate',
     'u component of wind at specified height level above ground',
     'u component of wind gust at specified height level above ground',
     'v component of wind at specified height level above ground',
     'v component of wind gust at specified height level above ground']




```python
fetcher.select_coverage_id(title="Temperature at specified height level above ground")
fetcher.check_run_time(horizon=35)
```

    -- DescribeCoverage request --


The `run_time` is a time stamp identifying when the model was run. The latest available run time (UTC) for our temperature forecast is the following:


```python
fetcher.run_time
```




    '2020-07-30T06:00:00Z'



Finally we need to set a point of interest, which is my street with a longitude and a latitude:


<p align="center">
  <img width="600" src="/img/2020-07-30_01/place_bellevue.jpg" alt="place_bellevue">
</p>


```python
fetcher.set_poi("lyon", 4.835999, 45.774429)
```

Let's fetch the temperature forecast and copy the resulting Pandas object:


```python
%%time
fetcher.create_time_series()
forecast = fetcher.series.copy(deep=True)
```

    -- GetCoverage request 2020-07-30T11:00:00Z --
    -- GetCoverage request 2020-07-30T12:00:00Z --
    -- GetCoverage request 2020-07-30T13:00:00Z --  
...  
    -- GetCoverage request 2020-07-31T19:00:00Z --
    -- GetCoverage request 2020-07-31T20:00:00Z --
    -- GetCoverage request 2020-07-31T21:00:00Z --
    CPU times: user 1.32 s, sys: 56.8 ms, total: 1.38 s
    Wall time: 32 s



```python
forecast.head(2)
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
      <th>lyon</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-30 11:00:00</th>
      <td>32.077190</td>
    </tr>
    <tr>
      <th>2020-07-30 12:00:00</th>
      <td>34.052318</td>
    </tr>
  </tbody>
</table>
</div>



## Plot

Before plotting, we need to change the `DatetimeIndex` from UTC to the local timezone:


```python
forecast.index.name = "Date"
forecast.reset_index(drop=False, inplace=True)
forecast.Date = forecast.Date.dt.tz_localize(pytz.timezone("UTC"))
forecast.Date = forecast.Date.dt.tz_convert(tz=pytz.timezone("Europe/Paris"))
forecast.Date = forecast.Date.dt.tz_localize(None)
forecast.set_index("Date", inplace=True)
forecast.head(2)
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
      <th>lyon</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-07-30 13:00:00</th>
      <td>32.077190</td>
    </tr>
    <tr>
      <th>2020-07-30 14:00:00</th>
      <td>34.052318</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = forecast.plot(style="d-", figsize=(18, 7), legend=False)
ax.grid()
ax.autoscale(enable=True, axis="x", tight=True)
_ = ax.set(title="Temperature forecast in Lyon - AROME 0.01", ylabel="Temperature (°C)")
```


<p align="center">
  <img width="800" src="/img/2020-07-30_01/output_21_0.png" alt="temperature">
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

