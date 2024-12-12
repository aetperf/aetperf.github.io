---
title: Pizzerias Around the World
layout: post
comments: true
author: François Pacull
tags: 
- Python
- datashader
- geospatial
- geopandas
- pizza
---

In November 2024, Foursquare released a new dataset called [Foursquare Open Source Places](https://location.foursquare.com/resources/blog/products/foursquare-open-source-places-a-new-foundational-dataset-for-the-geospatial-community/). This dataset is a comprehensive global collection of Points of Interest (POIs), providing detailed information about venues, including their categories, attributes, and geospatial details. 

In this post, we are going to play a little bit with this dataset by fetching and plotting all the pizzerias of the world.

## Imports

```python
import os

import contextily as cx
import datashader as ds
import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import xyzservices.providers as xyz
from colorcet import palette
from datashader import transfer_functions as tf

PARQUET_FP = "./pizzerias.parquet"  # file path
FS = (12, 7)
```

## Querying and Visualizing Pizzerias


Using DuckDB, we analyze the FSQ OS Places dataset stored on Amazon s3 to extract worldwide pizzeria locations.

```python
%%time
conn = duckdb.connect()
query = "SELECT COUNT(*) FROM 's3://fsq-os-places-us-east-1/release/dt=2024-11-19/places/parquet/places-*.snappy.parquet';"
conn.sql(query)
```

    CPU times: user 351 ms, sys: 25.7 ms, total: 376 ms
    Wall time: 1.16 s

    ┌──────────────┐
    │ count_star() │
    │    int64     │
    ├──────────────┤
    │    104511073 │
    └──────────────┘


Inspired by a LinkedIn post about [Belgian fritteries](https://www.linkedin.com/posts/jcolot_pas-toutes-ses-frites-dans-le-m%C3%AAme-sachet-activity-7268154014310010880--eXt?utm_source=share&utm_medium=member_desktop) by [
Julien Colot](https://www.linkedin.com/in/jcolot/), we query pizzeria locations using the category pizzeria : ID `4bf58dd8d48988d1ca941735`. Given that the dataset organizes POI categories hierarchically, we filter the data using the `list_contains` function to directly target the relevant category ID, avoiding the need to unnest the entire dataset. Although this method is painly slow, it is a one-time operation. We save the data to a Parquet file for further reuse:

```python
if not os.path.isfile(PARQUET_FP):
    query = f"""
    COPY (
      SELECT * FROM read_parquet('s3://fsq-os-places-us-east-1/release/dt=2024-11-19/places/parquet/places-*.snappy.parquet') AS t
      WHERE list_contains(t.fsq_category_ids, '4bf58dd8d48988d1ca941735')
    ) TO '{PARQUET_FP}';"""
    conn.sql(query)
df = pd.read_parquet(PARQUET_FP)
conn.close()
```
    CPU times: user 6min 31s, sys: 6min 6s, total: 12min 38s
    Wall time: 40min 35s



```python
df.shape
```




    (650873, 24)


We have 650873 pizzerias listed, one pizzeria for each row.

```python
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
      <th>fsq_place_id</th>
      <th>name</th>
      <th>...</th>
      <th>fsq_category_labels</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>4b4a33e9f964a5207b7e26e3</td>
      <td>Pizza Vesuvio</td>
      <td>...</td>
      <td>[Dining and Drinking &gt; Restaurant &gt; Pizzeria, ...</td>
      <td>2024-11-19</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4d0ff14e38bb6ea88488c1aa</td>
      <td>Pizzaville</td>
      <td>...</td>
      <td>[Dining and Drinking &gt; Restaurant &gt; Pizzeria]</td>
      <td>2024-11-19</td>
    </tr>
    <tr>
      <th>2</th>
      <td>520a776111d2995bb605242b</td>
      <td>Pizzaria jet pizza</td>
      <td>...</td>
      <td>[Dining and Drinking &gt; Restaurant &gt; Pizzeria]</td>
      <td>2024-11-19</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>


Let's have a look at the columns we got:

```python
df.columns
```

    Index(['fsq_place_id', 'name', 'latitude', 'longitude', 'address', 'locality',
           'region', 'postcode', 'admin_region', 'post_town', 'po_box', 'country',
           'date_created', 'date_refreshed', 'date_closed', 'tel', 'website',
           'email', 'facebook_id', 'instagram', 'twitter', 'fsq_category_ids',
           'fsq_category_labels', 'dt'],
          dtype='object')

We see that there is a `date_closed` column:

```python
df.date_closed.isna().sum(axis=0)
```
    590316

We select pizzerias that are still open, without a `date_closed` value:

```python
df = df.loc[df.date_closed.isna()]
```

We have 590316 open pizzerias. Let's transform the pandas DataFrame into a geopandas GeoDataFrame:

```python
gdf = gpd.GeoDataFrame(
    df, geometry=gpd.points_from_xy(df.longitude, df.latitude), crs="EPSG:4326"
)
```

To get a quick overview of the spatial distribution of the pizzerias, we can visualize the GeoDataFrame:

```python
ax = gdf.plot(markersize=4, alpha=0.5, figsize=FS)
ax.grid()
```

<p align="center">
  <img width="900" src="/img/2024-12-12_01/output_10_0.png" alt="quick">
</p>
    
## Plot all the pizzerias with Datashader

### World view

We visualize the global distribution of pizzerias using [Datashader](https://datashader.org/). We start by filtering the dataset to exclude extreme latitudes, focusing on points between -75 and 80 degrees latitude.

```python
df_world = df.loc[(df.latitude > -75) & (df.latitude < 80)]
cmap = palette["fire"]
bg_col = "black"
size_x, size_y = 1200, 600
cvs = ds.Canvas(plot_width=size_x, plot_height=size_y)
agg = cvs.points(df_world, "longitude", "latitude")
img = tf.shade(agg, cmap=cmap)
img = tf.set_background(img, bg_col)
img
```

<p align="center">
  <img width="900" src="/img/2024-12-12_01/output_18_0.png" alt="World">
</p>

A large number of pizzerias are located in Europe and the US, as well as in Uruguay and Argentina, regions with a significant Italian diaspora.

### Europe view

```python
bbox = -13.876174, 35.238452, 40.253716, 60.966935
df_europe = df.loc[
    (df.longitude > bbox[0])
    & (df.longitude < bbox[2])
    & (df.latitude > bbox[1])
    & (df.latitude < bbox[3])
]
size_x, size_y = 1200, 800
cvs = ds.Canvas(plot_width=size_x, plot_height=size_y)
agg = cvs.points(df_europe, "longitude", "latitude")
img = tf.shade(agg, cmap=cmap)
img = tf.set_background(img, bg_col)
img
```

<p align="center">
  <img width="900" src="/img/2024-12-12_01/output_20_0.png" alt="Europe">
</p>

Many pizzerias are concentrated along the Mediterranean and Adriatic coasts.

### Italy

```python
bbox = 6.492820,36.389024,18.663258,47.206285
df_italy = df.loc[
    (df.longitude > bbox[0])
    & (df.longitude < bbox[2])
    & (df.latitude > bbox[1])
    & (df.latitude < bbox[3])
]
size_x, size_y = 1000, 1200
cvs = ds.Canvas(plot_width=size_x, plot_height=size_y)
agg = cvs.points(df_italy, "longitude", "latitude")
img = tf.shade(agg, cmap=cmap)
img = tf.set_background(img, bg_col)
img
```

<p align="center">
  <img width="900" src="/img/2024-12-12_01/output_23_0.png" alt="Italy">
</p>

Major italian cities like Naples, Rome, Milan, and Turin stand out as pizzeria hotspots.

### Lyon Croix-Rousse

Let’s examine my district, Lyon Croix-Rousse, to validate the dataset in an area where I am familiar with all the pizzerias.

```python
bbox = 4.815526,45.765160,4.845303,45.780993
df_lyon = df.loc[
    (df.longitude > bbox[0])
    & (df.longitude < bbox[2])
    & (df.latitude > bbox[1])
    & (df.latitude < bbox[3])
]
df_lyon.head(3)
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
      <th>fsq_place_id</th>
      <th>name</th>
      <th>...</th>
      <th>fsq_category_labels</th>
      <th>dt</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8822</th>
      <td>4cc1df5b01fb236a265a9bba</td>
      <td>Pizza Des Canuts</td>
      <td>...</td>
      <td>[Dining and Drinking &gt; Restaurant &gt; Pizzeria]</td>
      <td>2024-11-19</td>
    </tr>
    <tr>
      <th>11326</th>
      <td>553f6623498e53d4de32f6de</td>
      <td>Caffe San Remo</td>
      <td>...</td>
      <td>[Dining and Drinking &gt; Restaurant &gt; Pizzeria]</td>
      <td>2024-11-19</td>
    </tr>
    <tr>
      <th>18889</th>
      <td>de751b48f717441a2c7c1768</td>
      <td>Djama Badis</td>
      <td>...</td>
      <td>[Dining and Drinking &gt; Bar, Dining and Drinkin...</td>
      <td>2024-11-19</td>
    </tr>
  </tbody>
</table>
<p>3 rows × 24 columns</p>
</div>




```python
gdf_lyon = gpd.GeoDataFrame(
    df_lyon,
    geometry=gpd.points_from_xy(df_lyon.longitude, df_lyon.latitude),
    crs="EPSG:4326",
)
```


```python
ax = gdf_lyon.plot(markersize=4, figsize=(12, 8))
cx.add_basemap(
    ax,
    source=xyz.CartoDB.VoyagerNoLabels,
    crs=gdf_lyon.crs.to_string(),
    alpha=0.8,
)
_ = plt.axis("off")
```

    
<p align="center">
  <img width="900" src="/img/2024-12-12_01/output_27_0.png" alt="Lyon">
</p>

We can create an interactive plot with the GeoDataFrame `.explore()` method:


```python
gdf_lyon[["name", "address", "geometry"]].explore()
```

<p align="center">
  <img width="900" src="/img/2024-12-12_01/Lyon.png" alt="Lyon">
</p>

The dataset for Lyon Croix-Rousse appears reasonably accurate, but there are some discrepancies:

- Incorrect location: one pizzeria is mapped to the wrong place.
- Fictional entry: at least one pizzeria seems entirely fictional.
- Missing entries: two pizzerias are missing.

Overall, I estimate the dataset's accuracy at around 90% for downtown area. However, it’s possible that the data is more reliable in other regions, or countries.