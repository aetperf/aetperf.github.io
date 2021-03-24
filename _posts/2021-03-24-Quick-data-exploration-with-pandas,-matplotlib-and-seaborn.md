---
title: Quick data exploration with pandas, matplotlib and seaborn
layout: post
comments: true
author: François Pacull
tags: Python pandas matplotlib seaborn
---

<p align="center">
  <img width="300" src="https://pandas.pydata.org/static/img/pandas_secondary.svg" alt="Pandas">
</p>

In this JupyterLab Python notebook we are going to look at the rate of coronavirus (COVID-19) cases in french departments (administrative divisions of France). The data source is the french government's [open data](https://www.data.gouv.fr/fr/datasets/taux-dincidence-de-lepidemie-de-covid-19/#_).

We are going to perform a few operations, such has filtering some data, pivoting some tables, smoothing time series with a rolling window or plotting an heatmap.

**Disclaimer** : although we are going to use some COVID-19 data in this notebook, I want the reader to know that I have ABSOLUTELY no knowledge in epidemiology or any medicine-related subject. The point of this post is not COVID-19 at all but only to show an application of the Python data stack.

## Imports


```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import colorcet as cc

FS = (16, 8)  # figure size
```

## Loading the data

We load the data from an URL straight to a pandas DataFrame:


```python
tests = pd.read_csv(
    "https://www.data.gouv.fr/fr/datasets/r/406c6a23-e283-4300-9484-54e78c8ae675",
    sep=";",
    low_memory=False,
)
tests.head(2)
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
      <th>dep</th>
      <th>jour</th>
      <th>P</th>
      <th>T</th>
      <th>cl_age90</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>01</td>
      <td>2020-05-13</td>
      <td>0</td>
      <td>16</td>
      <td>9</td>
      <td>83001.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>01</td>
      <td>2020-05-13</td>
      <td>1</td>
      <td>17</td>
      <td>19</td>
      <td>84665.0</td>
    </tr>
  </tbody>
</table>
</div>



We have 6 columns here:  
- `dep` : "department"'s code  
- `jour` :  date  
- `P` : number of positive tests per day   
- `T` : number of tests per day  
- `cl_age90` : age group   
- `pop` : population corresponding to an age group and a department

We have 11 age group values, however 0 gather all age groups:


```python
tests.cl_age90.unique().tolist()
```


    [9, 19, 29, 39, 49, 59, 69, 79, 89, 90, 0]



For example in Paris, we have:


```python
depnum = "75"
pop_paris = (
    tests[tests.dep == depnum][["cl_age90", "pop"]]
    .drop_duplicates()
    .set_index("cl_age90")
)
ax = pop_paris.plot.bar(figsize=FS, alpha=0.6)
ax.grid()
_ = ax.set(
    title=f"Population in department {depnum} per age group", ylabel="Population",
)
```

   
<p align="center">
  <img width="800" src="/img/2021-03-24_01/output_7_0.png" alt="">
</p>
    



```python
assert (
    pop_paris[pop_paris.index > 0].sum().values[0]
    == pop_paris[pop_paris.index == 0].values[0][0]
)
```

We start by creating a `DatetimeIndex`:


```python
tests.jour = pd.to_datetime(tests.jour, format="%Y-%m-%d")
tests.set_index("jour", inplace=True)
tests.index.name = "Date"
tests.head(2)
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
      <th>dep</th>
      <th>P</th>
      <th>T</th>
      <th>cl_age90</th>
      <th>pop</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-13</th>
      <td>01</td>
      <td>0</td>
      <td>16</td>
      <td>9</td>
      <td>83001.0</td>
    </tr>
    <tr>
      <th>2020-05-13</th>
      <td>01</td>
      <td>1</td>
      <td>17</td>
      <td>19</td>
      <td>84665.0</td>
    </tr>
  </tbody>
</table>
</div>



## COVID-19 and test rates in the Rhône department

Now we select a department (Rhône department with code 69):


```python
depnum = "69"
dep_tot = tests[(tests.dep == depnum) & (tests.cl_age90 == 0)].copy(deep=True)
dep_tot.drop(["dep", "cl_age90"], axis=1, inplace=True)
dep_tot.head(2)
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
      <th>P</th>
      <th>T</th>
      <th>pop</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-13</th>
      <td>20</td>
      <td>1468</td>
      <td>1876051.0</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>41</td>
      <td>1531</td>
      <td>1876051.0</td>
    </tr>
  </tbody>
</table>
</div>



We can now compute and plot the COVID-19 rate for all age groups in this department:


```python
ax = (
    (100000 * dep_tot.P / dep_tot["pop"])
    .rolling(7, center=True)
    .mean()
    .plot(style="-", figsize=FS, logy=True, alpha=0.6)
)
ax = (
    (100000 * dep_tot["T"] / dep_tot["pop"])
    .rolling(7, center=True)
    .mean()
    .plot(style="-", ax=ax, logy=True, alpha=0.6)
)
ax.grid()
_ = ax.set(
    title=f"Daily COVID-19 rate (per 100000) in department {depnum} (log scale)",
    ylabel="log scale",
)
_ = ax.legend(["Daily COVID-19 rate", "Daily test rate"])
ax.autoscale(enable=True, axis="x", tight=True)
```


<p align="center">
  <img width="800" src="/img/2021-03-24_01/output_14_0.png" alt="">
</p>
    


We can also show the positivity rate:


```python
ax = (100 * dep_tot.P / dep_tot["T"]).rolling(7, center=True).mean().plot(figsize=FS)
ax.grid()
_ = ax.set(
    title=f"Positivity rate in department {depnum}", ylabel="Positivity rate (%)",
)
ax.autoscale(enable=True, axis="x", tight=True)
```

<p align="center">
  <img width="800" src="/img/2021-03-24_01/output_16_0.png" alt="">
</p>


## Departement with the worst COVID-19 rate

First we need to select departments with a rather large population size (at least 50000 inhabitants):


```python
pop = (
    tests[tests.cl_age90 == 0][["dep", "pop"]]
    .drop_duplicates()
    .sort_values(by="pop")
    .reset_index(drop=True)
)
pop.head()
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
      <th>dep</th>
      <th>pop</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>975</td>
      <td>5997.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>977</td>
      <td>9961.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>978</td>
      <td>35334.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>48</td>
      <td>76286.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>116270.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
pop_th = 50000
large_deps = pop[pop["pop"] > pop_th].dep.values.tolist()
```



Now we pivot the table such that each column corresponds to a department:


```python
cr_alldep = tests[tests.cl_age90 == 0][["dep", "P", "pop"]]
cr_alldep["ir"] = 100000 * cr_alldep.P / cr_alldep["pop"]
cr_alldep.drop(["pop", "P"], axis=1, inplace=True)
cr_alldep = cr_alldep.pivot_table(index="Date", columns="dep", values="ir")
cr_alldep = cr_alldep[
    large_deps
]  # Here we remove the smallest department regarding population
cr_alldep.head(2)
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
      <th>dep</th>
      <th>48</th>
      <th>23</th>
      <th>...</th>
      <th>75</th>
      <th>59</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-13</th>
      <td>0.0</td>
      <td>0.860067</td>
      <td>...</td>
      <td>1.768864</td>
      <td>1.390505</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>0.0</td>
      <td>0.860067</td>
      <td>...</td>
      <td>2.606747</td>
      <td>1.622255</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 102 columns</p>
</div>



Let's look at the 5 departments with the highest COVID-19 rate in the most recent days:


```python
n_deps = 5
deps = (
    cr_alldep.rolling(7, center=True)
    .mean()
    .dropna()
    .iloc[-1]
    .sort_values(ascending=False)[:n_deps]
    .index.values.tolist()
)
deps
```



    ['93', '95', '94', '77', '75']



We can now plot the evolution of the COVID-19 rate in these 5 most affected departments:


```python
highest_cr = cr_alldep[deps]
ax = highest_cr.rolling(7, center=True).mean().plot(figsize=FS, alpha=0.6)
ax.grid()
_ = ax.set(
    title="Daily COVID-19 rate (per 100000) in the most affected departments",
    ylabel="COVID-19 rate",
)
ax.autoscale(enable=True, axis="x", tight=True)
```


    
<p align="center">
  <img width="800" src="/img/2021-03-24_01/output_25_0.png" alt="">
</p>
    


Now we are going to focus on the department with highest COVID-19 rate.

## Heatmap of the COVID-19 rate by age group in the most affected department

We start by pivoting the table such that each column corresponds to an age group:


```python
depnum = deps[0]
dep_ag = tests[(tests.dep == depnum) & (tests.cl_age90 != 0)].copy(deep=True)
dep_ag["ir"] = 100000 * dep_ag.P / dep_ag["pop"]
dep_ag.drop(["dep", "P", "T", "pop"], axis=1, inplace=True)
dep_ag = dep_ag.pivot_table(index="Date", columns="cl_age90", values="ir")
dep_ag.head(2)
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
      <th>cl_age90</th>
      <th>9</th>
      <th>19</th>
      <td>...</td>
      <th>89</th>
      <th>90</th>
    </tr>
    <tr>
      <th>Date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2020-05-13</th>
      <td>1.160501</td>
      <td>1.324369</td>
      <td>...</td>
      <td>19.272929</td>
      <td>96.936797</td>
    </tr>
    <tr>
      <th>2020-05-14</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>26.500277</td>
      <td>9.693680</td>
    </tr>
  </tbody>
</table>
</div>



Also, we compute the weakly average and transpose the table:


```python
cr_smooth = dep_ag.resample("W").mean().T
cr_smooth = cr_smooth.sort_index(ascending=False)
cr_smooth.columns = [t.date() for t in cr_smooth.columns]
cr_smooth.head(2)
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
      <th>2020-05-17</th>
      <th>2020-05-24</th>
      <th>...</th>
      <th>2021-03-14</th>
      <th>2021-03-21</th>
    </tr>
    <tr>
      <th>cl_age90</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>90</th>
      <td>23.264831</td>
      <td>24.926605</td>
      <td>...</td>
      <td>54.007644</td>
      <td>74.318211</td>
    </tr>
    <tr>
      <th>89</th>
      <td>12.527404</td>
      <td>7.915667</td>
      <td>...</td>
      <td>60.916221</td>
      <td>63.038538</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 45 columns</p>
</div>



We can now plot the heatmap:


```python
fig, ax = plt.subplots(figsize=(25, 8))
ax = sns.heatmap(
    cr_smooth.astype(int), ax=ax, annot=True, cbar=False, fmt="d", cmap=cc.fire[::-1]
)
_ = ax.set(
    title=f"Daily COVID-19 rate in department {depnum}",
    xlabel="Date",
    ylabel="Age group",
)
```

    
<p align="center">
  <img width="1600" src="/img/2021-03-24_01/output_31_0.png" alt="">
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