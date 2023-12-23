---
title: Lyon's Digital Terrain Model with IGN Data WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
---

The data comes from this IGN page : https://geoservices.ign.fr/rgealti

This Digital Terrain Model (DTM) provides a detailed grid-based depiction of the topography of the entire French territory on a large scale. Regular updates are obtained from surveys conducted using airborne LIDAR or aerial image correlation. For our purposes, we will be working with the 5-meter resolution option, although a 1-meter resolution is also available.

OS and package versions:

    Python version       : 3.11.6  
    OS                   : Linux  
    Machine              : x86_64  
    contextily           : 1.4.0
    geopandas            : 0.14.1
    matplotlib           : 3.8.2
    numpy                : 1.26.2
    pandas               : 2.1.3
    py7zr                : 0.20.6
    pyproj               : 3.6.1
    rasterio             : 1.3.9
    requests             : 2.31.0
    bs4                  : 4.12.2
    osgeo                : 3.7.3
    shapely              : 2.0.2
    tqdm                 : 4.66.1






```python
import glob
import os
import re

import contextily as cx
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import py7zr
import pyproj
import rasterio as rio
import requests
from bs4 import BeautifulSoup
from osgeo import gdal, osr
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from rasterio.plot import show
from shapely.geometry import Point, Polygon
from shapely.ops import transform
from tqdm import tqdm

DATA_DIR_PATH = "/home/francois/Data/RGE_ALTI/5M/"
IGN_URL = "https://geoservices.ign.fr/rgealti#telechargement5m"  # 5 m precision
tif_dir_path = os.path.join(DATA_DIR_PATH, "tif_files")
deps_shp_fp = "/home/francois/Data/RGE_ALTI/departements-20140306-50m-shp/departements-20140306-50m.shp"
```

## Parse the HTML from the IGN web page using BeautifulSoup

We specifically target and extract download links related to the 5-meter resolution, filtering out any links associated with the 1-meter resolution.

```python
# Retrieve HTML content from IGN web page
response = requests.get(IGN_URL)

# Extract download links using BeautifulSoup
download_links = None
if response.status_code == 200:
    html_content = response.text
    soup = BeautifulSoup(html_content, "html.parser")
    pattern = re.compile(
        r"https://wxs\.ign\.fr/[^/]+/telechargement/prepackage/RGEALTI-5M_PACK_[^$]+\$[^/]+/file/[^/]+\.7z"
    )
    links = soup.find_all("a", href=pattern)
    download_links = [link["href"] for link in links]
else:
    print(f"Failed to retrieve the page. Status code: {response.status_code}")

# Display the number of download links found
if download_links:
    link_count = len(download_links)
else:
    link_count = 0
print(f"Found {link_count} download link(s)")
```

    Found 105 download link(s)

## Organize and Validate Data with Pandas

```python
# Create Pandas DataFrame with download links
df = pd.DataFrame(data={"download_link": download_links})
df["compressed_file_name"] = df["download_link"].map(os.path.basename)
```

```python
# Extract components and assert data integrity

def extract_components(s):
    base_name = s.split(".7z")[0]
    keys = ["dataset", "version", "res", "filetype", "crs", "dep", "date"]
    d = dict(zip(keys, base_name.split("_")))
    return pd.Series(d)


df[["dataset", "version", "res", "filetype", "crs", "dep", "date"]] = df[
    "compressed_file_name"
].apply(extract_components)
```


```python
assert np.array_equal(df["dataset"].unique(), np.array(["RGEALTI"]))
assert np.array_equal(df["res"].unique(), np.array(["5M"]))
assert np.array_equal(df["filetype"].unique(), np.array(["ASC"]))
```


```python
# Extract department codes and refine DataFrame

def get_dep_code(s):
    dep_code = s.removeprefix("D")
    if dep_code[0] == "0":
        dep_code = dep_code.removeprefix("0")
    return dep_code


df["dep_code"] = df["dep"].map(get_dep_code)
df = df.drop(["dataset", "version", "res", "dep", "filetype", "date"], axis=1)
```


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
      <th>download_link</th>
      <th>compressed_file_name</th>
      <th>crs</th>
      <th>dep_code</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>https://wxs.ign.fr/9u5z4x13jqu05fb3o9cux5e1/te...</td>
      <td>RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D001_2023-08-0...</td>
      <td>LAMB93-IGN69</td>
      <td>01</td>
    </tr>
    <tr>
      <th>1</th>
      <td>https://wxs.ign.fr/9u5z4x13jqu05fb3o9cux5e1/te...</td>
      <td>RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D002_2020-09-0...</td>
      <td>LAMB93-IGN69</td>
      <td>02</td>
    </tr>
    <tr>
      <th>2</th>
      <td>https://wxs.ign.fr/9u5z4x13jqu05fb3o9cux5e1/te...</td>
      <td>RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D003_2023-08-1...</td>
      <td>LAMB93-IGN69</td>
      <td>03</td>
    </tr>
  </tbody>
</table>
</div>

Our Python script efficiently retrieves, organizes, and validates the IGN DTM download links.

## Bounding box definition


```python
# Define bounding box coordinates
bbox = (4.346466, 45.463020, 5.340729, 46.030342)

# Define points to create a polygon
point1 = (bbox[0], bbox[1])
point2 = (bbox[0], bbox[3])
point3 = (bbox[2], bbox[3])
point4 = (bbox[2], bbox[1])
```


```python
# Create a polygon from the points
polygon = Polygon([point1, point2, point3, point4])

# Display the polygon
polygon
```

    
<p align="center">
  <img width="400" src="/img/2023-12-12_01/output_11_0.svg" alt="Bounding box">
</p>



```python
bbox_gdf = gpd.GeoSeries([polygon], crs="EPSG:4326")
fig, ax = plt.subplots(1, figsize=(12, 12))
ax = bbox_gdf.plot(alpha=0.3, ax=ax)
cx.add_basemap(ax, crs=bbox_gdf.crs.to_string(), source=cx.providers.CartoDB.Positron)
ax.set_axis_off()
```


<p align="center">
  <img width="800" src="/img/2023-12-12_01/output_12_0.png" alt="Bounding box with map tiles">
</p>    

This bounding box serves as the spatial extent for our analysis, encompassing the specific region of interest within Lyon. 


## Intersection with "department" zones

zones shapefile downloaded from [this](https://www.data.gouv.fr/fr/datasets/contours-des-departements-francais-issus-d-openstreetmap/) page.  

[Export de mars 2014 - vérifié et simplifié à 50m](https://www.data.gouv.fr/fr/datasets/r/6e53bca5-1153-49d4-bff5-dd69f39369b5)


```python
zones = gpd.read_file(deps_shp_fp)[["code_insee", "geometry"]]
zones.head(2)
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
      <th>code_insee</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01</td>
      <td>POLYGON ((5.25559 45.78459, 5.23987 45.77758, ...</td>
    </tr>
    <tr>
      <th>1</th>
      <td>02</td>
      <td>POLYGON ((3.48175 48.86640, 3.48647 48.85768, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
intersected_zones = zones.loc[zones.intersects(polygon)]
intersected_zones
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
      <th>code_insee</th>
      <th>geometry</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01</td>
      <td>POLYGON ((5.25559 45.78459, 5.23987 45.77758, ...</td>
    </tr>
    <tr>
      <th>38</th>
      <td>38</td>
      <td>POLYGON ((5.44781 45.07178, 5.44982 45.07078, ...</td>
    </tr>
    <tr>
      <th>42</th>
      <td>42</td>
      <td>POLYGON ((4.02072 45.32827, 4.01866 45.32764, ...</td>
    </tr>
    <tr>
      <th>69</th>
      <td>69</td>
      <td>POLYGON ((4.87189 45.52749, 4.86173 45.51689, ...</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = intersected_zones.boundary.plot(alpha=0.2, label="insee_code")
ax = bbox_gdf.plot(alpha=0.3, ax=ax)
```

<p align="center">
  <img width="600" src="/img/2023-12-12_01/output_16_0.png" alt="Intersected zones">
</p>    



```python
zone_codes = intersected_zones.code_insee.values.tolist()
zone_codes
```




    ['01', '38', '42', '69']



## Data download


```python
download_items = []
for zone_code in zone_codes:
    download_items.append(
        df.loc[
            df.dep_code == zone_code,
            ["download_link", "compressed_file_name", "crs"],
        ].to_dict(orient="records")[0]
    )
```


```python
dem_dir_paths = []
for download_item in download_items:
    crs = download_item["crs"]

    # url, paths, file name
    dem_file_url = download_item["download_link"]
    dem_file_name = download_item["compressed_file_name"]
    dem_file_path = os.path.join(DATA_DIR_PATH, dem_file_name)
    dem_dir_path = os.path.splitext(dem_file_path)[0]

    # download
    is_downloaded = os.path.isfile(dem_file_path) | os.path.isdir(dem_dir_path)
    if not is_downloaded:
        print(f"downloading file {dem_file_name}")
        r = requests.get(dem_file_url, stream=True)
        with open(dem_file_path, "wb") as fd:
            for chunk in r.iter_content(chunk_size=512):
                fd.write(chunk)

    # uncompress
    dem_dir_paths.append(dem_dir_path)
    is_uncompressed = os.path.isdir(dem_dir_path)
    if not is_uncompressed:
        print(f"uncompressing file {dem_file_name}")
        with py7zr.SevenZipFile(dem_file_path, "r") as archive:
            archive.extractall(path=DATA_DIR_PATH)
        if os.path.exists(dem_file_path):
            os.remove(dem_file_path)
```


```python
dem_dir_paths
```




    ['/home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D001_2023-08-08',
     '/home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D038_2020-11-13',
     '/home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D042_2023-08-10',
     '/home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D069_2023-08-10']



## Convert `.asc` files to `.tif` files


```python
if not os.path.exists(tif_dir_path):
    os.makedirs(tif_dir_path)
    print(f"tif files directory : {tif_dir_path}")
```

The following 2 functions and 1 class are taken straight from the great blog post [1] by Guillaume Attard:


```python
def get_header_asc(filepath):
    """
    This function reads the header of an asc file and returns
    the data into a dictionnary
    """
    file = open(filepath)
    content = file.readlines()[:6]
    content = [item.split() for item in content]
    return dict(content)


class RGEitem:
    """
    This class is used to handle RGE items.
    """

    def __init__(self, filepath):
        self.filename = os.path.basename(filepath)
        self.dir = os.path.dirname(filepath)
        self.data = np.loadtxt(filepath, skiprows=6)
        self.header = get_header_asc(filepath)
        self.ncols = int(self.header["ncols"])
        self.nrows = int(self.header["nrows"])
        self.xllc = float(self.header["xllcorner"])
        self.yllc = float(self.header["yllcorner"])
        self.res = float(self.header["cellsize"])
        self.zmin = float(self.data.min())
        self.zmax = float(self.data.max())
        self.novalue = -99999.0


def asc_to_tif(file, output_rasterpath, epsg):
    """
    Transforms an .asc file into a geoTIFF.

    Params:
    -------
    file: an RGEitem
    output_raster_path (str): path of the tif file that will be saved.
    epsg (int): projection system.

    Returns:
    --------
    output_rasterpath (str): name of the output geoTIFF
    """
    xmin = file.xllc
    ymax = file.yllc + file.nrows * file.res
    geotransform = (xmin, file.res, 0, ymax, 0, -file.res)

    # Open the file
    output_raster = gdal.GetDriverByName("GTiff").Create(
        output_rasterpath, file.ncols, file.nrows, 1, gdal.GDT_Float32
    )
    # Specify the coordinates.
    output_raster.SetGeoTransform(geotransform)
    # Establish the coordinate encoding.
    srs = osr.SpatialReference()
    # Specify the projection.
    srs.ImportFromEPSG(epsg)
    # Export the coordinate system to the file.
    output_raster.SetProjection(srs.ExportToWkt())
    # Writes the array.
    output_raster.GetRasterBand(1).WriteArray(file.data)
    # Set nodata value.
    output_raster.GetRasterBand(1).SetNoDataValue(file.novalue)
    output_raster.FlushCache()
```


```python
def find_asc_files(directory):
    directory = os.path.join(directory, "**/*.asc")
    asc_file_paths = glob.glob(directory, recursive=True)
    return asc_file_paths


all_tif_file_paths = []
for dem_dir_path in dem_dir_paths:
    print(f"top directory : {dem_dir_path}")
    asc_file_paths = find_asc_files(dem_dir_path)
    asc_file_count = len(asc_file_paths)
    print(f"{asc_file_count} asc files")
    for asc_file_path in tqdm(asc_file_paths):
        asc_file_name = os.path.basename(asc_file_path)
        asc_file_name, asc_file_extension = os.path.splitext(
            os.path.basename(asc_file_name)
        )
        tif_file_path = os.path.join(tif_dir_path, asc_file_name + ".tif")
        all_tif_file_paths.append(tif_file_path)
        if not os.path.isfile(tif_file_path):
            file = RGEitem(asc_file_path)
            asc_to_tif(file, tif_file_path, epsg=2154)
```

    top directory : /home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D001_2023-08-08
    291 asc files


    100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 291/291 [00:00<00:00, 56083.37it/s]


    top directory : /home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D038_2020-11-13
    383 asc files


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 383/383 [00:00<00:00, 206321.40it/s]


    top directory : /home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D042_2023-08-10
    248 asc files


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 248/248 [00:00<00:00, 197042.51it/s]


    top directory : /home/francois/Data/RGE_ALTI/5M/RGEALTI_2-0_5M_ASC_LAMB93-IGN69_D069_2023-08-10
    174 asc files


    100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 174/174 [00:00<00:00, 231685.36it/s]



```python
fig, ax = plt.subplots(1, figsize=(6, 6))

with rio.open(tif_file_path) as img:
    show(img, cmap="terrain", ax=ax)

ax.set_axis_off()
```

<p align="center">
  <img width="300" src="/img/2023-12-12_01/output_27_0.png" alt="A small tif file">
</p>   
    


```python
len(all_tif_file_paths)
```




    1096



## Raster Mosaic with Python


```python
output_mosaic_path = os.path.join(tif_dir_path, "lyon_mosaic.tif")

raster_to_mosaic = []
for p in all_tif_file_paths:
    raster = rio.open(p)
    raster_to_mosaic.append(raster)
```


```python
mosaic, output = merge(raster_to_mosaic)
```


```python
output_meta = raster.meta.copy()
output_meta.update(
    {
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": output,
    }
)
```


```python
with rio.open(output_mosaic_path, "w", **output_meta) as m:
    m.write(mosaic)

del mosaic
```

Let's downsample this mosaic and plot it:

```python
upscale_factor = 0.125

with rio.open(output_mosaic_path) as dataset:
    # resample data to target shape
    data = dataset.read(
        out_shape=(
            dataset.count,
            int(dataset.height * upscale_factor),
            int(dataset.width * upscale_factor),
        ),
        resampling=Resampling.bilinear,
    )
    data = np.where(data == -99999.0, np.nan, data)
```


```python
fig, ax = plt.subplots(1, figsize=(16, 16))

show(data, cmap="terrain", ax=ax)
ax.set_axis_off()
```



<p align="center">
  <img width="800" src="/img/2023-12-12_01/output_37_0.png" alt="Mosaic tif file">
</p>   
    


## Raster Bounding Box Cropping with Rasterio


```python
wgs84 = pyproj.CRS("EPSG:4326")
lam93 = pyproj.CRS("EPSG:2154")
```


```python
project = pyproj.Transformer.from_crs(wgs84, lam93, always_xy=True).transform
polygon_lam93 = transform(project, polygon)
```


```python
polygon_lam93
```

<p align="center">
  <img width="300" src="/img/2023-12-12_01/output_41_0.svg" alt="Bounding box on projected CRS">
</p>   
    


```python
mosaic_raster = rio.open(output_mosaic_path)
```


```python
mosaic_raster.crs
```




    CRS.from_epsg(2154)




```python
out_image, out_transform = mask(mosaic_raster, [polygon_lam93], crop=True, filled=False)
out_meta = mosaic_raster.meta
```


```python
out_meta.update(
    {
        "driver": "GTiff",
        "height": out_image.shape[1],
        "width": out_image.shape[2],
        "transform": out_transform,
    }
)
```

## Save and plot


```python
with rio.open("lyon_dem.tif", "w", **out_meta) as dest:
    dest.write(out_image)
```


```python
dem = rio.open("lyon_dem.tif")
```


```python
fig, ax = plt.subplots(1, figsize=(16, 16))
_ = show(dem, cmap="terrain", ax=ax)
ax.set_axis_off()
```


<p align="center">
  <img width="800" src="/img/2023-12-12_01/output_49_0.png" alt="Cropped mosaic">
</p>   
    


## Point queries


```python
rng = np.random.default_rng(421)
```


```python
# Set the number of random coordinates you want
num_points = 1000

# Generate random x, y coordinates within the bounding box
random_x = rng.uniform(bbox[0], bbox[2], num_points)
random_y = rng.uniform(bbox[1], bbox[3], num_points)
point_coords = [transform(project, Point(x, y)) for x, y in zip(random_x, random_y)]
point_coords = [(p.x, p.y) for p in point_coords]
```


```python
points = pd.DataFrame(point_coords, columns=["x_lam93", "y_lamb93"])
points["x_wgs84"] = random_x
points["y_wgs84"] = random_y
points_gdf = gpd.GeoDataFrame(
    points,
    geometry=gpd.points_from_xy(points.x_lam93, points.y_lamb93),
    crs="EPSG:2154",
)
```


```python
fig, ax = plt.subplots(1, figsize=(10, 8))
figure = show(dem, cmap="terrain", ax=ax)
im = figure.get_images()[0]
clb = fig.colorbar(im, ax=ax)
clb.ax.set_title("Elevation (m)", fontsize=8)
ax = points_gdf.plot(ax=ax, c="k", markersize=3, alpha=0.4)
_ = plt.axis("off")
_ = ax.set(title="Elevation map of the region of Lyon, FR")
```

    
<p align="center">
  <img width="800" src="/img/2023-12-12_01/output_54_0.png" alt="Sample points queries mosaic">
</p>   


```python
%%time
points["elevation"] = [x[0] for x in dem.sample(point_coords)]
points.loc[points["elevation"] < 0.0, "elevation"] = np.nan
points.head(3)
```

    CPU times: user 15.8 ms, sys: 2.31 ms, total: 18.1 ms
    Wall time: 17.2 ms





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
      <th>x_lam93</th>
      <th>y_lamb93</th>
      <th>x_wgs84</th>
      <th>y_wgs84</th>
      <th>elevation</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>818322.302210</td>
      <td>6.540550e+06</td>
      <td>4.527680</td>
      <td>45.954361</td>
      <td>297.980011</td>
    </tr>
    <tr>
      <th>1</th>
      <td>820552.295444</td>
      <td>6.510972e+06</td>
      <td>4.548974</td>
      <td>45.687670</td>
      <td>709.400024</td>
    </tr>
    <tr>
      <th>2</th>
      <td>826049.449585</td>
      <td>6.494117e+06</td>
      <td>4.615158</td>
      <td>45.534934</td>
      <td>353.480011</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(points["elevation"].max(), points["elevation"].min())
```

    839.79 150.5


## References

[1] Guillaume Attard (2022) https://medium.com/@gui.attard/pre-processing-the-dem-of-france-rge-alti-5m-for-implementation-into-earth-engine-de9a0778e0d9



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