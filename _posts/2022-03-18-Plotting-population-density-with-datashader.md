---
title: Plotting population density with datashader
layout: post
comments: true
author: François Pacull
tags: Python raster datashader population dataviz
---

In this short post, we are using the [Global Human Settlement Layer](https://ghsl.jrc.ec.europa.eu/ghs_pop2019.php):

> This spatial raster dataset depicts the distribution of population, expressed as the number of people per cell.

The file we used has a resolution of 250m, with a World Mollweide coordinates reference system. Values are expressed as decimals (float32) and represent the absolute number of inhabitants of the cell. A value of -200 is found whenever there is no data (e.g. in the oceans). We downloaded the file corresponding to the 2015 population estimates.

[Datashader](https://github.com/holoviz/datashader) is a graphics pipeline system for creating meaningful representations of large datasets quickly and flexibly.

## Imports


```python
import rioxarray
import xarray as xr
import datashader as ds
from datashader import transfer_functions as tf
from colorcet import palette

%load_ext lab_black

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




```python
da
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray (band: 1, y: 72000, x: 144328)&gt;
dask.array&lt;open_rasterio-95bb17bcac9421dbddd850e4aabcecb4&lt;this-array&gt;, shape=(1, 72000, 144328), dtype=float32, chunksize=(1, 5632, 5632), chunktype=numpy.ndarray&gt;
Coordinates:
  * band         (band) int64 1
  * x            (x) float64 -1.804e+07 -1.804e+07 ... 1.804e+07 1.804e+07
  * y            (y) float64 9e+06 9e+06 8.999e+06 ... -8.999e+06 -9e+06 -9e+06
    spatial_ref  int64 0
Attributes:
    STATISTICS_COVARIANCES:  6079.753709536097
    STATISTICS_MAXIMUM:      442590.9375
    STATISTICS_MEAN:         3.402693912331
    STATISTICS_MINIMUM:      0
    STATISTICS_SKIPFACTORX:  1
    STATISTICS_SKIPFACTORY:  1
    STATISTICS_STDDEV:       77.972775438201
    _FillValue:              -200.0
    scale_factor:            1.0
    add_offset:              0.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'></div><ul class='xr-dim-list'><li><span class='xr-has-index'>band</span>: 1</li><li><span class='xr-has-index'>y</span>: 72000</li><li><span class='xr-has-index'>x</span>: 144328</li></ul></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-3c0359c1-2827-4e69-80e3-498284a78fad' class='xr-array-in' type='checkbox' checked><label for='section-3c0359c1-2827-4e69-80e3-498284a78fad' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>dask.array&lt;chunksize=(1, 5632, 5632), meta=np.ndarray&gt;</span></div><div class='xr-array-data'><table>
    <tr>
        <td>
            <table>
                <thead>
                    <tr>
                        <td> </td>
                        <th> Array </th>
                        <th> Chunk </th>
                    </tr>
                </thead>
                <tbody>

                    <tr>
                        <th> Bytes </th>
                        <td> 38.71 GiB </td>
                        <td> 121.00 MiB </td>
                    </tr>

                    <tr>
                        <th> Shape </th>
                        <td> (1, 72000, 144328) </td>
                        <td> (1, 5632, 5632) </td>
                    </tr>
                    <tr>
                        <th> Count </th>
                        <td> 339 Tasks </td>
                        <td> 338 Chunks </td>
                    </tr>
                    <tr>
                    <th> Type </th>
                    <td> float32 </td>
                    <td> numpy.ndarray </td>
                    </tr>
                </tbody>
            </table>
        </td>
        <td>
        <svg width="194" height="124" style="stroke:rgb(0,0,0);stroke-width:1" >

  <!-- Horizontal lines -->
  <line x1="10" y1="0" x2="24" y2="14" style="stroke-width:2" />
  <line x1="10" y1="4" x2="24" y2="19" />
  <line x1="10" y1="9" x2="24" y2="24" />
  <line x1="10" y1="14" x2="24" y2="28" />
  <line x1="10" y1="18" x2="24" y2="33" />
  <line x1="10" y1="23" x2="24" y2="38" />
  <line x1="10" y1="28" x2="24" y2="43" />
  <line x1="10" y1="32" x2="24" y2="47" />
  <line x1="10" y1="37" x2="24" y2="52" />
  <line x1="10" y1="42" x2="24" y2="57" />
  <line x1="10" y1="46" x2="24" y2="61" />
  <line x1="10" y1="51" x2="24" y2="66" />
  <line x1="10" y1="56" x2="24" y2="71" />
  <line x1="10" y1="59" x2="24" y2="74" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="10" y1="0" x2="10" y2="59" style="stroke-width:2" />
  <line x1="24" y1="14" x2="24" y2="74" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="10.0,0.0 24.9485979497544,14.948597949754403 24.9485979497544,74.81224187193166 10.0,59.86364392217726" style="fill:#ECB172A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="10" y1="0" x2="130" y2="0" style="stroke-width:2" />
  <line x1="24" y1="14" x2="144" y2="14" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="10" y1="0" x2="24" y2="14" style="stroke-width:2" />
  <line x1="14" y1="0" x2="29" y2="14" />
  <line x1="19" y1="0" x2="34" y2="14" />
  <line x1="28" y1="0" x2="43" y2="14" />
  <line x1="33" y1="0" x2="48" y2="14" />
  <line x1="38" y1="0" x2="53" y2="14" />
  <line x1="47" y1="0" x2="62" y2="14" />
  <line x1="52" y1="0" x2="67" y2="14" />
  <line x1="56" y1="0" x2="71" y2="14" />
  <line x1="66" y1="0" x2="81" y2="14" />
  <line x1="70" y1="0" x2="85" y2="14" />
  <line x1="80" y1="0" x2="95" y2="14" />
  <line x1="84" y1="0" x2="99" y2="14" />
  <line x1="89" y1="0" x2="104" y2="14" />
  <line x1="98" y1="0" x2="113" y2="14" />
  <line x1="103" y1="0" x2="118" y2="14" />
  <line x1="108" y1="0" x2="123" y2="14" />
  <line x1="117" y1="0" x2="132" y2="14" />
  <line x1="122" y1="0" x2="137" y2="14" />
  <line x1="130" y1="0" x2="144" y2="14" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="10.0,0.0 130.0,0.0 144.9485979497544,14.948597949754403 24.9485979497544,14.948597949754403" style="fill:#8B4903A0;stroke-width:0"/>

  <!-- Horizontal lines -->
  <line x1="24" y1="14" x2="144" y2="14" style="stroke-width:2" />
  <line x1="24" y1="19" x2="144" y2="19" />
  <line x1="24" y1="24" x2="144" y2="24" />
  <line x1="24" y1="28" x2="144" y2="28" />
  <line x1="24" y1="33" x2="144" y2="33" />
  <line x1="24" y1="38" x2="144" y2="38" />
  <line x1="24" y1="43" x2="144" y2="43" />
  <line x1="24" y1="47" x2="144" y2="47" />
  <line x1="24" y1="52" x2="144" y2="52" />
  <line x1="24" y1="57" x2="144" y2="57" />
  <line x1="24" y1="61" x2="144" y2="61" />
  <line x1="24" y1="66" x2="144" y2="66" />
  <line x1="24" y1="71" x2="144" y2="71" />
  <line x1="24" y1="74" x2="144" y2="74" style="stroke-width:2" />

  <!-- Vertical lines -->
  <line x1="24" y1="14" x2="24" y2="74" style="stroke-width:2" />
  <line x1="29" y1="14" x2="29" y2="74" />
  <line x1="34" y1="14" x2="34" y2="74" />
  <line x1="43" y1="14" x2="43" y2="74" />
  <line x1="48" y1="14" x2="48" y2="74" />
  <line x1="53" y1="14" x2="53" y2="74" />
  <line x1="62" y1="14" x2="62" y2="74" />
  <line x1="67" y1="14" x2="67" y2="74" />
  <line x1="71" y1="14" x2="71" y2="74" />
  <line x1="81" y1="14" x2="81" y2="74" />
  <line x1="85" y1="14" x2="85" y2="74" />
  <line x1="95" y1="14" x2="95" y2="74" />
  <line x1="99" y1="14" x2="99" y2="74" />
  <line x1="104" y1="14" x2="104" y2="74" />
  <line x1="113" y1="14" x2="113" y2="74" />
  <line x1="118" y1="14" x2="118" y2="74" />
  <line x1="123" y1="14" x2="123" y2="74" />
  <line x1="132" y1="14" x2="132" y2="74" />
  <line x1="137" y1="14" x2="137" y2="74" />
  <line x1="144" y1="14" x2="144" y2="74" style="stroke-width:2" />

  <!-- Colored Rectangle -->
  <polygon points="24.9485979497544,14.948597949754403 144.9485979497544,14.948597949754403 144.9485979497544,74.81224187193166 24.9485979497544,74.81224187193166" style="fill:#8B4903A0;stroke-width:0"/>

  <!-- Text -->
  <text x="84.948598" y="94.812242" font-size="1.0rem" font-weight="100" text-anchor="middle" >144328</text>
  <text x="164.948598" y="44.880420" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(-90,164.948598,44.880420)">72000</text>
  <text x="7.474299" y="87.337943" font-size="1.0rem" font-weight="100" text-anchor="middle" transform="rotate(45,7.474299,87.337943)">1</text>
</svg>
        </td>
    </tr>
</table></div></div></li><li class='xr-section-item'><input id='section-28f4629a-9b4e-4556-92f8-73f01d6a60f2' class='xr-section-summary-in' type='checkbox'  checked><label for='section-28f4629a-9b4e-4556-92f8-73f01d6a60f2' class='xr-section-summary' >Coordinates: <span>(4)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>band</span></div><div class='xr-var-dims'>(band)</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>1</div><input id='attrs-9163ac78-cd1b-4a78-b8bf-9baecb81ff5e' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-9163ac78-cd1b-4a78-b8bf-9baecb81ff5e' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-eb6c91d6-d3fd-4f9d-b88a-b1674bce7792' class='xr-var-data-in' type='checkbox'><label for='data-eb6c91d6-d3fd-4f9d-b88a-b1674bce7792' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([1])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>x</span></div><div class='xr-var-dims'>(x)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>-1.804e+07 -1.804e+07 ... 1.804e+07</div><input id='attrs-ab31ddac-cabc-445f-b666-3603417760fe' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-ab31ddac-cabc-445f-b666-3603417760fe' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-6b077c70-a7b6-4ed3-9322-da93e69a1a6c' class='xr-var-data-in' type='checkbox'><label for='data-6b077c70-a7b6-4ed3-9322-da93e69a1a6c' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([-18040875., -18040625., -18040375., ...,  18040375.,  18040625.,
        18040875.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span class='xr-has-index'>y</span></div><div class='xr-var-dims'>(y)</div><div class='xr-var-dtype'>float64</div><div class='xr-var-preview xr-preview'>9e+06 9e+06 ... -9e+06 -9e+06</div><input id='attrs-33569cc0-5c41-482c-afa7-c7c4d563a6cb' class='xr-var-attrs-in' type='checkbox' disabled><label for='attrs-33569cc0-5c41-482c-afa7-c7c4d563a6cb' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-f1da22f4-d7be-444d-838f-b45380aeae13' class='xr-var-data-in' type='checkbox'><label for='data-f1da22f4-d7be-444d-838f-b45380aeae13' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'></dl></div><div class='xr-var-data'><pre>array([ 8999875.,  8999625.,  8999375., ..., -8999375., -8999625., -8999875.])</pre></div></li><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-ed3336fc-8064-4890-b065-7cf3b04d98c9' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-ed3336fc-8064-4890-b065-7cf3b04d98c9' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-5d100c41-7bcb-477d-acab-2dfce247eb0b' class='xr-var-data-in' type='checkbox'><label for='data-5d100c41-7bcb-477d-acab-2dfce247eb0b' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>spatial_ref :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>GeoTransform :</span></dt><dd>-18041000.0 250.0 0.0 9000000.0 0.0 -250.0</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-dc7ee665-c7a9-4177-bfb3-1c7082e3b309' class='xr-section-summary-in' type='checkbox'  ><label for='section-dc7ee665-c7a9-4177-bfb3-1c7082e3b309' class='xr-section-summary' >Attributes: <span>(10)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>STATISTICS_COVARIANCES :</span></dt><dd>6079.753709536097</dd><dt><span>STATISTICS_MAXIMUM :</span></dt><dd>442590.9375</dd><dt><span>STATISTICS_MEAN :</span></dt><dd>3.402693912331</dd><dt><span>STATISTICS_MINIMUM :</span></dt><dd>0</dd><dt><span>STATISTICS_SKIPFACTORX :</span></dt><dd>1</dd><dt><span>STATISTICS_SKIPFACTORY :</span></dt><dd>1</dd><dt><span>STATISTICS_STDDEV :</span></dt><dd>77.972775438201</dd><dt><span>_FillValue :</span></dt><dd>-200.0</dd><dt><span>scale_factor :</span></dt><dd>1.0</dd><dt><span>add_offset :</span></dt><dd>0.0</dd></dl></div></li></ul></div></div>




```python
da.spatial_ref
```




<div><svg style="position: absolute; width: 0; height: 0; overflow: hidden">
<defs>
<symbol id="icon-database" viewBox="0 0 32 32">
<path d="M16 0c-8.837 0-16 2.239-16 5v4c0 2.761 7.163 5 16 5s16-2.239 16-5v-4c0-2.761-7.163-5-16-5z"></path>
<path d="M16 17c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
<path d="M16 26c-8.837 0-16-2.239-16-5v6c0 2.761 7.163 5 16 5s16-2.239 16-5v-6c0 2.761-7.163 5-16 5z"></path>
</symbol>
<symbol id="icon-file-text2" viewBox="0 0 32 32">
<path d="M28.681 7.159c-0.694-0.947-1.662-2.053-2.724-3.116s-2.169-2.030-3.116-2.724c-1.612-1.182-2.393-1.319-2.841-1.319h-15.5c-1.378 0-2.5 1.121-2.5 2.5v27c0 1.378 1.122 2.5 2.5 2.5h23c1.378 0 2.5-1.122 2.5-2.5v-19.5c0-0.448-0.137-1.23-1.319-2.841zM24.543 5.457c0.959 0.959 1.712 1.825 2.268 2.543h-4.811v-4.811c0.718 0.556 1.584 1.309 2.543 2.268zM28 29.5c0 0.271-0.229 0.5-0.5 0.5h-23c-0.271 0-0.5-0.229-0.5-0.5v-27c0-0.271 0.229-0.5 0.5-0.5 0 0 15.499-0 15.5 0v7c0 0.552 0.448 1 1 1h7v19.5z"></path>
<path d="M23 26h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 22h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
<path d="M23 18h-14c-0.552 0-1-0.448-1-1s0.448-1 1-1h14c0.552 0 1 0.448 1 1s-0.448 1-1 1z"></path>
</symbol>
</defs>
</svg>
<style>/* CSS stylesheet for displaying xarray objects in jupyterlab.
 *
 */

:root {
  --xr-font-color0: var(--jp-content-font-color0, rgba(0, 0, 0, 1));
  --xr-font-color2: var(--jp-content-font-color2, rgba(0, 0, 0, 0.54));
  --xr-font-color3: var(--jp-content-font-color3, rgba(0, 0, 0, 0.38));
  --xr-border-color: var(--jp-border-color2, #e0e0e0);
  --xr-disabled-color: var(--jp-layout-color3, #bdbdbd);
  --xr-background-color: var(--jp-layout-color0, white);
  --xr-background-color-row-even: var(--jp-layout-color1, white);
  --xr-background-color-row-odd: var(--jp-layout-color2, #eeeeee);
}

html[theme=dark],
body.vscode-dark {
  --xr-font-color0: rgba(255, 255, 255, 1);
  --xr-font-color2: rgba(255, 255, 255, 0.54);
  --xr-font-color3: rgba(255, 255, 255, 0.38);
  --xr-border-color: #1F1F1F;
  --xr-disabled-color: #515151;
  --xr-background-color: #111111;
  --xr-background-color-row-even: #111111;
  --xr-background-color-row-odd: #313131;
}

.xr-wrap {
  display: block !important;
  min-width: 300px;
  max-width: 700px;
}

.xr-text-repr-fallback {
  /* fallback to plain text repr when CSS is not injected (untrusted notebook) */
  display: none;
}

.xr-header {
  padding-top: 6px;
  padding-bottom: 6px;
  margin-bottom: 4px;
  border-bottom: solid 1px var(--xr-border-color);
}

.xr-header > div,
.xr-header > ul {
  display: inline;
  margin-top: 0;
  margin-bottom: 0;
}

.xr-obj-type,
.xr-array-name {
  margin-left: 2px;
  margin-right: 10px;
}

.xr-obj-type {
  color: var(--xr-font-color2);
}

.xr-sections {
  padding-left: 0 !important;
  display: grid;
  grid-template-columns: 150px auto auto 1fr 20px 20px;
}

.xr-section-item {
  display: contents;
}

.xr-section-item input {
  display: none;
}

.xr-section-item input + label {
  color: var(--xr-disabled-color);
}

.xr-section-item input:enabled + label {
  cursor: pointer;
  color: var(--xr-font-color2);
}

.xr-section-item input:enabled + label:hover {
  color: var(--xr-font-color0);
}

.xr-section-summary {
  grid-column: 1;
  color: var(--xr-font-color2);
  font-weight: 500;
}

.xr-section-summary > span {
  display: inline-block;
  padding-left: 0.5em;
}

.xr-section-summary-in:disabled + label {
  color: var(--xr-font-color2);
}

.xr-section-summary-in + label:before {
  display: inline-block;
  content: '►';
  font-size: 11px;
  width: 15px;
  text-align: center;
}

.xr-section-summary-in:disabled + label:before {
  color: var(--xr-disabled-color);
}

.xr-section-summary-in:checked + label:before {
  content: '▼';
}

.xr-section-summary-in:checked + label > span {
  display: none;
}

.xr-section-summary,
.xr-section-inline-details {
  padding-top: 4px;
  padding-bottom: 4px;
}

.xr-section-inline-details {
  grid-column: 2 / -1;
}

.xr-section-details {
  display: none;
  grid-column: 1 / -1;
  margin-bottom: 5px;
}

.xr-section-summary-in:checked ~ .xr-section-details {
  display: contents;
}

.xr-array-wrap {
  grid-column: 1 / -1;
  display: grid;
  grid-template-columns: 20px auto;
}

.xr-array-wrap > label {
  grid-column: 1;
  vertical-align: top;
}

.xr-preview {
  color: var(--xr-font-color3);
}

.xr-array-preview,
.xr-array-data {
  padding: 0 5px !important;
  grid-column: 2;
}

.xr-array-data,
.xr-array-in:checked ~ .xr-array-preview {
  display: none;
}

.xr-array-in:checked ~ .xr-array-data,
.xr-array-preview {
  display: inline-block;
}

.xr-dim-list {
  display: inline-block !important;
  list-style: none;
  padding: 0 !important;
  margin: 0;
}

.xr-dim-list li {
  display: inline-block;
  padding: 0;
  margin: 0;
}

.xr-dim-list:before {
  content: '(';
}

.xr-dim-list:after {
  content: ')';
}

.xr-dim-list li:not(:last-child):after {
  content: ',';
  padding-right: 5px;
}

.xr-has-index {
  font-weight: bold;
}

.xr-var-list,
.xr-var-item {
  display: contents;
}

.xr-var-item > div,
.xr-var-item label,
.xr-var-item > .xr-var-name span {
  background-color: var(--xr-background-color-row-even);
  margin-bottom: 0;
}

.xr-var-item > .xr-var-name:hover span {
  padding-right: 5px;
}

.xr-var-list > li:nth-child(odd) > div,
.xr-var-list > li:nth-child(odd) > label,
.xr-var-list > li:nth-child(odd) > .xr-var-name span {
  background-color: var(--xr-background-color-row-odd);
}

.xr-var-name {
  grid-column: 1;
}

.xr-var-dims {
  grid-column: 2;
}

.xr-var-dtype {
  grid-column: 3;
  text-align: right;
  color: var(--xr-font-color2);
}

.xr-var-preview {
  grid-column: 4;
}

.xr-var-name,
.xr-var-dims,
.xr-var-dtype,
.xr-preview,
.xr-attrs dt {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  padding-right: 10px;
}

.xr-var-name:hover,
.xr-var-dims:hover,
.xr-var-dtype:hover,
.xr-attrs dt:hover {
  overflow: visible;
  width: auto;
  z-index: 1;
}

.xr-var-attrs,
.xr-var-data {
  display: none;
  background-color: var(--xr-background-color) !important;
  padding-bottom: 5px !important;
}

.xr-var-attrs-in:checked ~ .xr-var-attrs,
.xr-var-data-in:checked ~ .xr-var-data {
  display: block;
}

.xr-var-data > table {
  float: right;
}

.xr-var-name span,
.xr-var-data,
.xr-attrs {
  padding-left: 25px !important;
}

.xr-attrs,
.xr-var-attrs,
.xr-var-data {
  grid-column: 1 / -1;
}

dl.xr-attrs {
  padding: 0;
  margin: 0;
  display: grid;
  grid-template-columns: 125px auto;
}

.xr-attrs dt,
.xr-attrs dd {
  padding: 0;
  margin: 0;
  float: left;
  padding-right: 10px;
  width: auto;
}

.xr-attrs dt {
  font-weight: normal;
  grid-column: 1;
}

.xr-attrs dt:hover span {
  display: inline-block;
  background: var(--xr-background-color);
  padding-right: 10px;
}

.xr-attrs dd {
  grid-column: 2;
  white-space: pre-wrap;
  word-break: break-all;
}

.xr-icon-database,
.xr-icon-file-text2 {
  display: inline-block;
  vertical-align: middle;
  width: 1em;
  height: 1.5em !important;
  stroke-width: 0;
  stroke: currentColor;
  fill: currentColor;
}
</style><pre class='xr-text-repr-fallback'>&lt;xarray.DataArray &#x27;spatial_ref&#x27; ()&gt;
array(0)
Coordinates:
    spatial_ref  int64 0
Attributes:
    crs_wkt:       PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,...
    spatial_ref:   PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,...
    GeoTransform:  -18041000.0 250.0 0.0 9000000.0 0.0 -250.0</pre><div class='xr-wrap' style='display:none'><div class='xr-header'><div class='xr-obj-type'>xarray.DataArray</div><div class='xr-array-name'>'spatial_ref'</div></div><ul class='xr-sections'><li class='xr-section-item'><div class='xr-array-wrap'><input id='section-84be34f1-9f52-4908-9ee2-954bfe8c9d8c' class='xr-array-in' type='checkbox' checked><label for='section-84be34f1-9f52-4908-9ee2-954bfe8c9d8c' title='Show/hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-array-preview xr-preview'><span>0</span></div><div class='xr-array-data'><pre>array(0)</pre></div></div></li><li class='xr-section-item'><input id='section-948a6128-e4ac-4179-9d6f-c3beb7b11a0a' class='xr-section-summary-in' type='checkbox'  checked><label for='section-948a6128-e4ac-4179-9d6f-c3beb7b11a0a' class='xr-section-summary' >Coordinates: <span>(1)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><ul class='xr-var-list'><li class='xr-var-item'><div class='xr-var-name'><span>spatial_ref</span></div><div class='xr-var-dims'>()</div><div class='xr-var-dtype'>int64</div><div class='xr-var-preview xr-preview'>0</div><input id='attrs-efc52fb7-1ac1-4324-bb7b-034a211c00d6' class='xr-var-attrs-in' type='checkbox' ><label for='attrs-efc52fb7-1ac1-4324-bb7b-034a211c00d6' title='Show/Hide attributes'><svg class='icon xr-icon-file-text2'><use xlink:href='#icon-file-text2'></use></svg></label><input id='data-de48efd8-2206-4302-918b-616efd505178' class='xr-var-data-in' type='checkbox'><label for='data-de48efd8-2206-4302-918b-616efd505178' title='Show/Hide data repr'><svg class='icon xr-icon-database'><use xlink:href='#icon-database'></use></svg></label><div class='xr-var-attrs'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>spatial_ref :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>GeoTransform :</span></dt><dd>-18041000.0 250.0 0.0 9000000.0 0.0 -250.0</dd></dl></div><div class='xr-var-data'><pre>array(0)</pre></div></li></ul></div></li><li class='xr-section-item'><input id='section-b2b5c848-424b-4996-9e48-50f7c5e42a57' class='xr-section-summary-in' type='checkbox'  checked><label for='section-b2b5c848-424b-4996-9e48-50f7c5e42a57' class='xr-section-summary' >Attributes: <span>(3)</span></label><div class='xr-section-inline-details'></div><div class='xr-section-details'><dl class='xr-attrs'><dt><span>crs_wkt :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>spatial_ref :</span></dt><dd>PROJCS[&quot;World_Mollweide&quot;,GEOGCS[&quot;WGS 84&quot;,DATUM[&quot;WGS_1984&quot;,SPHEROID[&quot;WGS 84&quot;,6378137,298.257223563,AUTHORITY[&quot;EPSG&quot;,&quot;7030&quot;]],AUTHORITY[&quot;EPSG&quot;,&quot;6326&quot;]],PRIMEM[&quot;Greenwich&quot;,0],UNIT[&quot;Degree&quot;,0.0174532925199433]],PROJECTION[&quot;Mollweide&quot;],PARAMETER[&quot;central_meridian&quot;,0],PARAMETER[&quot;false_easting&quot;,0],PARAMETER[&quot;false_northing&quot;,0],UNIT[&quot;metre&quot;,1,AUTHORITY[&quot;EPSG&quot;,&quot;9001&quot;]],AXIS[&quot;Easting&quot;,EAST],AXIS[&quot;Northing&quot;,NORTH]]</dd><dt><span>GeoTransform :</span></dt><dd>-18041000.0 250.0 0.0 9000000.0 0.0 -250.0</dd></dl></div></li></ul></div></div>



## Total population

Let's compute the total population:


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


```python
cmap = palette["fire"]
img = tf.shade(
    raster, how="eq_hist", cmap=cmap
)
img
```

<p align="center">
  <img width="1200" src="/img/2022-03-18_01/output_19_0.png" alt="Europe">
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
  <img width="1200" src="/img/2022-03-18_01/output_27_0.png" alt="France">
</p>




We can notice that some areas are not detailed up to the 250m accuracy, but rather averaged over larger regions, exhibiting a uniform color (e.g. in the southern Alps).


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