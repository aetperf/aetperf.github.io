
In this short post, we are using the [Global Human Settlement Layer](https://ghsl.jrc.ec.europa.eu/ghs_pop2019.php) from the European Commission:

> This spatial raster dataset depicts the distribution of population, expressed as the number of people per cell.

The downloaded file has a worldwide resolution of 250m, with a World Mollweide coordinates reference system. Values are expressed as decimals (float32) and represent the absolute number of inhabitants of the cell. A value of -200 is found whenever there is no data (e.g. in the oceans). Also, it corresponds to the 2015 population estimates.

We are going to load the data into a [xarray](https://github.com/pydata/xarray) DataArray and make some plots with [Datashader](https://github.com/holoviz/datashader). 

> Datashader is a graphics pipeline system for creating meaningful representations of large datasets quickly and flexibly.

## Imports


```python
import rioxarray
import xarray as xr
import datashader as ds
from datashader import transfer_functions as tf
from colorcet import palette

FP = "GHS_POP_E2015_GLOBE_R2019A_54009_250_V1_0.tif"  # file path
```

# Loading the dataset

Let's start by opening the file using [rioxarray](https://github.com/corteva/rioxarray), and [dask](https://dask.org/) as backend. rioxarray is a *geospatial xarray extension powered by [rasterio](https://github.com/rasterio/rasterio)*.


```python
da = rioxarray.open_rasterio(
    FP,
    chunks=True,
    lock=False,
)
```


```python
type(da)
```




    xarray.core.dataarray.DataArray




## Total population

Let's compute the total population count:


```python
%%time
total_pop = da.where(da[0] > 0).sum().compute()
total_pop = float(total_pop.values)
```

    CPU times: user 4min 45s, sys: 22.5 s, total: 5min 8s
    Wall time: 40.8 s



```python
print(f"Total population : {total_pop}")
```

    Total population : 7349329920.0


World population was indeed around 7.35 billion in 2015.

### Europe

Let's focus on Europe with a bounding box in World_Mollweide coordinates:


```python
minx = float(da.x.min().values)
maxx = float(da.x.max().values)
miny = float(da.y.min().values)
maxy = float(da.y.max().values)
print(f"minx : {minx}, maxx : {maxx}, miny : {miny}, maxy : {maxy}")
```

    minx : -18040875.0, maxx : 18040875.0, miny : -8999875.0, maxy : 8999875.0


So let's clip the data array using a bounding box:


```python
dac = da.rio.clip_box(
    minx=-1_000_000.0,
    miny=4_250_000.0,
    maxx=2_500_000.0,
    maxy=7_750_000.0,
)
```

And plot this selection:


```python
dac0 = xr.DataArray(dac)[0]
dac0 = dac0.where(dac0 > 0)
dac0 = dac0.fillna(0.0).compute()
```


```python
size = 1200
cvs = ds.Canvas(plot_width=size, plot_height=size)
raster = cvs.raster(dac0)
```

We are using the default `mean` downsampling operation to produce the image.

```python
cmap = palette["fire"]
img = tf.shade(
    raster, how="eq_hist", cmap=cmap
)
img
```

<p align="center">
  <img width="1200" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2022-03-18_01/output_19_0.png" alt="Europe">
</p>


## France

We are now going to focus on France, by cliping /re-projecting/re-cliping the data:


```python
dac = da.rio.clip_box(
    minx=-450_000.0,
    miny=5_000_000.0,
    maxx=600_000.0,
    maxy=6_000_000.0,
)
```


```python
dacr = dac.rio.reproject("EPSG:2154")
```


```python
minx = float(dacr.x.min().values)
maxx = float(dacr.x.max().values)
miny = float(dacr.y.min().values)
maxy = float(dacr.y.max().values)
print(f"minx : {minx}, maxx : {maxx}, miny : {miny}, maxy : {maxy}")
```

    minx : 3238.8963631442175, maxx : 1051199.0429940927, miny : 6088320.296559229, maxy : 7160193.962105454



```python
dacrc = dacr.rio.clip_box(
    minx=80_000,
    miny=6_150_000,
    maxx=1_100_000,
    maxy=7_100_000,
)
```


```python
dac0 = xr.DataArray(dacrc)[0]
dac0 = dac0.where(dac0 > 0)
dac0 = dac0.fillna(0.0).compute()
```


```python
cvs = ds.Canvas(plot_width=size, plot_height=size)
raster = cvs.raster(dac0)
```


```python
cmap = palette["fire"]
img = tf.shade(raster, how="eq_hist", cmap=cmap)
img
```

<p align="center">
  <img width="1200" src="https://github.com/aetperf/aetperf.github.io/blob/master/img/2022-03-18_01/output_27_0.png" alt="France">
</p>




We can notice that some areas are not detailed up to the 250m accuracy, but rather averaged over larger regions, exhibiting a uniform color (e.g. in the southern Alps).