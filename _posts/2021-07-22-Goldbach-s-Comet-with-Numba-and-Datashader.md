---
title: Goldbach's Comet with Numba and Datashader
layout: post
comments: true
author: François Pacull
tags: Python Goldbach primes Python Numba Datashader visualization
---

This Python post is about computing and plotting Goldbach function. However, it is based on some very basic mathematical knowledge, nothing fancy! The point is to see how [Numba](http://numba.pydata.org/) can easily accelerate some computations. We also use [Datashader](https://datashader.org/) to perform some efficient plotting.

Here is the definition of the Goldbach function from [wikipedia](https://en.wikipedia.org/wiki/Goldbach%27s_comet):

> The function $g ( E )$  is defined for all even integers $E > 2$ to be the number of different ways in which E can be expressed as the sum of two primes. For example, $g ( 22 ) = 3$  since 22 can be expressed as the sum of two primes in three different ways ( 22 = 11 + 11 = 5 + 17 = 3 + 19).

Note that for Goldbach's conjecture to be false, there must be $g(E) = 0$ somewhere.

Anyway, here are the steps used in this post to compute Golbach function: 
- Define a maximum positive integer $n$.
- For each natural number smaller or equal to $n$, build a quick way to check if it is a prime or not. In order to do that, we are going to create a boolean vector using the sieve of Eratosthenes.

```python
is_prime_vec = generate_is_prime_vector(n)  
```

- for each even number $E$ smaller or equal to $n$, compute $g(E)$ by counting the number of cases where $E-p$ is prime for all primes $p$ not larger than $E/2$.

```python
g_vec = compute_g_vector(is_prime_vec)
```

## Imports


```python
import numpy as np
import pandas as pd
import primesieve
from numba import jit, njit, prange
import matplotlib.pyplot as plt
import perfplot
import datashader as ds
from datashader import transfer_functions as tf
from colorcet import palette
```

Package versions:

```python
    Python    : 3.9.6
    pandas    : 1.3.0
    datashader: 0.13.0
    primesieve: 2.3.0
    matplotlib: 3.4.2
    numpy     : 1.21.1
    colorcet  : 2.0.6
    perfplot  : 0.9.6
    numba     : 0.53.1
```


## Sieve of Eratosthenes with Numba

This code is inspired from [wikipedia](https://en.wikipedia.org/wiki/Sieve_of_Eratosthenes#Pseudocode) and [geeksforgeeks](https://www.geeksforgeeks.org/sieve-of-eratosthenes/). It flags all primes not greater than $n$. The algorithm includes a common optimization, which is to start enumerating the multiples of each prime $p$ from $p^2$.


```python
@jit(nopython=True)
def generate_is_prime_vector(n: int) -> np.ndarray:
    """Sieve of Eratosthenes.

    Parameters
    ----------
    n : int
        `n` is the largest number that we want to flag ad prime or not.

    Returns
    -------
    is_prime_vec : ndarray
        an array of type bool (`i` is prime if `is_prime_vec[i]` is True).
    """

    # initialize all entries as True, except 0 and 1
    is_prime_vec = np.ones(n + 1, dtype=np.bool_)
    is_prime_vec[0:2] = False

    # loop on prime numbers
    p = 2
    while p * p <= n:
        if is_prime_vec[p] == True:
            # Update all multiples of p, starting from p * p
            for i in range(p * p, n + 1, p):
                is_prime_vec[i] = False
        p += 1

    return is_prime_vec
```

Let's run it with a small value of `n`:


```python
n = 11
generate_is_prime_vector(n)
```




    array([False, False,  True,  True, False,  True, False,  True, False,
           False, False,  True])



We can get the list of primes from the `is_prime` vector using `np.nonzero`: 


```python
np.nonzero(generate_is_prime_vector(n))[0]
```




    array([ 2,  3,  5,  7, 11])



The list of primes not greater then `n` can also be computed using the `primesieve` library:


```python
primesieve.primes(n)
```




    array('Q', [2, 3, 5, 7, 11])



Let's check that the list of primes found is correct with a larger value of `n`:


```python
%%time
n = 1_000_000
is_prime_vec = generate_is_prime_vector(n)
prime_list_1 = list(np.nonzero(is_prime_vec)[0])
prime_list_2 = primesieve.primes(n)
np.testing.assert_array_equal(prime_list_1, prime_list_2)
```

    CPU times: user 37.1 ms, sys: 69 µs, total: 37.2 ms
    Wall time: 36.5 ms


### Elapsed time

This `generate_is_prime_vector` function is rather efficient compared to the next step of computing $g(E)$ for all $E$s. This classical implementation of the sieve of Eratosthenes is supposed to be $O(n \; log \; log \; n)$.


```python
out = perfplot.bench(
    setup=lambda n: n,
    kernels=[
        lambda n: generate_is_prime_vector(n),
    ],
    labels=["generate_is_prime_vector(n)"],
    n_range=[10 ** k for k in range(1, 10)],
)
```


```python
out
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> n          </span>┃<span style="font-weight: bold"> generate_is_prime_vector(n) </span>┃
┡━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 10         │ 6.3e-07                     │
│ 100        │ 7.070000000000001e-07       │
│ 1000       │ 1.7830000000000001e-06      │
│ 10000      │ 1.2794000000000002e-05      │
│ 100000     │ 0.000149319                 │
│ 1000000    │ 0.002291595                 │
│ 10000000   │ 0.081507009                 │
│ 100000000  │ 1.119336045                 │
│ 1000000000 │ 11.769184520000001          │
└────────────┴─────────────────────────────┘
</pre>






    




```python
ms = 10
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plt.loglog(
    out.n_range,
    1.0e-9 * out.n_range * np.log(np.log(out.n_range)),
    "-",
    label="$O(n \; log \; log \; n)$",
)
plt.loglog(
    out.n_range, out.timings_s[0], "v-", ms=ms, label="generate_is_prime_vector(n)"
)
plt.legend()
plt.grid(True)
_ = ax.set_ylabel("Runtime [s]")
_ = ax.set_xlabel("n")
_ = ax.set_title("Timings of generate_is_prime_vector")
```

<p align="center">
  <img width="600" src="/img/2021-07-22_01/output_18_0.png" alt="Timings of generate_is_prime_vector">
</p>
 


## Find the number of prime pairs for a given even number E

Now we show how $g(E)$ can be computed for a given value of $E$:


```python
# E is the even number that we want to decompose into prime pairs
E = 22
assert E % 2 == 0

E_half = int(0.5 * E)

# we generate the boolean prime vector for all the integer up E
is_prime_vec = generate_is_prime_vector(E)

# initialization
i = 2  # loop counter (we know that 0 and 1 are not primes)
count = 0  # number of prime pairs

# we loop over all the prime numbers smaller than or equal to half of num
while i <= E_half:
    if is_prime_vec[i] and is_prime_vec[E - i]:
        print(f"({i:2d}, {E- i:2d})")
        count += 1
    i += 1
print(f"{count} prime pairs")
```

    ( 3, 19)
    ( 5, 17)
    (11, 11)
    3 prime pairs


## Loop over all even numbers E not larger than n

Now we apply the previous process to all $E$s smaller or equal to $n$, which is related to the length of `is_prime_vec`. We only compute the latter once and use it for all the evaluations of $g(E)$. Note that in the following `compute_g_vector` function, the outer loop has a constant step size of 1, in order to later use Numba `prange`, which only supports this unit step size. This means that we loop on contiguous integer values of $E/2$ instead of even values of $E$.


```python
@jit(nopython=True)
def compute_g_vector(is_prime_vec: np.ndarray) -> np.ndarray:
    """Evaluate the Golbach function.

    This evatuates the Golbach function for all even integers E not
    larger than the largest integer from the argument is_prime.

    Parameters
    ----------
    is_prime_vec : ndarray
        an array of type bool (`i` is prime iif `is_prime[i]` is True).

    Returns
    -------
    g_vec : ndarray
        an array of type uint (`g_vec[i]` corresponds to g(E) where E = 2i).
    """

    n_max = len(is_prime_vec) - 1
    n_max_half = int(0.5 * n_max) + 1

    g_vec = np.empty(n_max_half, dtype=np.uint)

    for i in range(n_max_half):
        count = 0
        E = 2 * i
        for j in range(2, i + 1):
            if is_prime_vec[j] and is_prime_vec[E - j]:
                count += 1
        g_vec[i] = count

    return g_vec
```

The $i$-th value of `g_vec` correponds to $g(2 \, i)$ with $i \geq 0 $:


| i |  E  | g_vec[i] |
|--:|----:|---------:|
| 0 |  0  |        0 |
| 1 |  2  |        0 |
| 2 |  4  |        1 |
| 3 |  6  |        1 |
| 4 |  8  |        1 |
| 5 | 10  |        2 |


and so on... We can check the values of $g$ at least for some for some small values of $E$:


```python
n = 56
is_prime_vec = generate_is_prime_vector(n)
g_vec = compute_g_vector(is_prime_vec)
g_vec_ref = [0, 0, 1, 1, 1, 2, 1, 2, 2, 2, 2, 3, 3, 3, 2, 3, 2, 4, 4, 2, 3, 4, 
    3, 4, 5, 4, 3, 5, 3]
np.testing.assert_array_equal(g_vec, g_vec_ref)
```

## First plot of the comet


```python
%%time
n = 10_000
is_prime_vec = generate_is_prime_vector(n)
g_vec = compute_g_vector(is_prime_vec)
g_df = pd.DataFrame(data={"g": g_vec}, index=2 * np.arange(len(g_vec)))
g_df = g_df[g_df.index > 2]  # The function g(E) is defined for all even integers E>2
g_df.head(2)
```

    CPU times: user 59.9 ms, sys: 150 µs, total: 60.1 ms
    Wall time: 60.5 ms





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
      <th>g</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4</th>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>




```python
ax = g_df.plot(style=".", ms=5, alpha=0.5, legend=False, figsize=FS)
_ = ax.set(
    title="Goldbach's comet",
    xlabel="E",
    ylabel="g(E)",
)
ax.autoscale(enable=True, axis="x", tight=True)
ax.grid(True)
```


<p align="center">
  <img width="800" src="/img/2021-07-22_01/output_27_0.png" alt="Goldbach's comet n=1e4">
</p>
    


## Parallel loop

We are now going to parallelize `compute_g_vector` just by using the Numba `njit` decorator and `prange` on the outer loop. The vector `g_vec` is shared across threads but each thread is only writing into its own partition.


```python
@njit(parallel=True)
def compute_g_vector_par(is_prime_vec: np.ndarray) -> np.ndarray:
    """Evaluate the Golbach function.

    This evatuates the Golbach function for all even integers E not
    larger than the largest integer from the argument is_prime.

    Parameters
    ----------
    is_prime_vec : ndarray
        an array of type bool (`i` is prime iif `is_prime[i]` is True).

    Returns
    -------
    g_vec : ndarray
        an array of type uint (g_vec[i] corresponds to g(E) where E = 2i).
    """

    n_max = len(is_prime_vec) - 1
    n_max_half = int(0.5 * n_max) + 1

    g_vec = np.empty(n_max_half, dtype=np.uint)

    for i in prange(n_max_half):
        count = 0
        E = 2 * i
        for j in range(2, i + 1):
            if is_prime_vec[j] and is_prime_vec[E - j]:
                count += 1
        g_vec[i] = count

    return g_vec
```

Here we check that the sequential and parallel versions return the same values:

```python
g_vec_par = compute_g_vector_par(is_prime_vec)
np.testing.assert_array_equal(g_vec, g_vec_par)
```

### Elapsed time

Now we compare the running time of the sequential and parallel versions of `compute_g_vector`. The code is running on a laptop with 8 cores (Intel(R) i7-7700HQ CPU @ 2.80GHz).


```python
out = perfplot.bench(
    setup=lambda n: generate_is_prime_vector(n),
    kernels=[
        lambda is_prime_vec: compute_g_vector(is_prime_vec),
        lambda is_prime_vec: compute_g_vector_par(is_prime_vec),
    ],
    labels=["jit", "njit"],
    n_range=[10 ** k for k in range(1, 6)],
)
```


    Output()



```python
out
```


<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">┏━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┓
┃<span style="font-weight: bold"> n      </span>┃<span style="font-weight: bold"> jit                   </span>┃<span style="font-weight: bold"> njit                   </span>┃
┡━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━┩
│ 10     │ 8.480000000000001e-07 │ 6.506e-06              │
│ 100    │ 2.271e-06             │ 7.0370000000000006e-06 │
│ 1000   │ 0.000224464           │ 6.1258e-05             │
│ 10000  │ 0.033075403           │ 0.010347824            │
│ 100000 │ 2.923607557           │ 1.1248999480000001     │
└────────┴───────────────────────┴────────────────────────┘
</pre>



Basically, running the parallel version on 8 cores divides the running time by 3.


```python
ms = 10
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(1, 1, 1)
plt.loglog(
    out.n_range,
    1.0e-9 * np.power(out.n_range, 2),
    "-",
    label="$O(N^2)$",
)
plt.loglog(out.n_range, out.timings_s[0], "v-", ms=ms, label="jit")
plt.loglog(out.n_range, out.timings_s[1], "^-", ms=ms, label="njit")
plt.legend()
plt.grid("on")
_ = ax.set_ylabel("Runtime [s]")
_ = ax.set_xlabel("N")
_ = ax.set_title("Timings of find_distinct_prime_pairs_count")
```


<p align="center">
  <img width="600" src="/img/2021-07-22_01/output_35_0.png" alt="Timings of find_distinct_prime_pairs_count">
</p>

    
If we run the parallel version with $n=1e6$, we might estimate the running time of $n=1e7$ and see if this is affordable.


```python
%%time
n = 1_000_000
is_prime_vec = generate_is_prime_vector(n)
g_vec_par = compute_g_vector_par(is_prime_vec)
```

    CPU times: user 6min 9s, sys: 428 ms, total: 6min 9s
    Wall time: 1min 28s


So computing $g$ with $n = 1e7$ should take one or a few hours on the laptop executing `compute_g_vector_par`... Let's do that next.

## Compute more data


```python
%%time
n = 10_000_000
is_prime_vec = generate_is_prime_vector(n)
```

    CPU times: user 162 ms, sys: 0 ns, total: 162 ms
    Wall time: 68.7 ms



```python
%%time
g_vec_par = compute_g_vector_par(is_prime_vec)
g_df = pd.DataFrame(data={"E": 2 * np.arange(len(g_vec_par)), "g": g_vec_par})
g_df = g_df[g_df.E > 2]  # The function g(E) is defined for all even integers E>2
```

    CPU times: user 8h 48min 44s, sys: 558 ms, total: 8h 48min 44s  
    Wall time: 1h 49min 9s  

A bit less than 2 hours...
        

## Plotting g with Datashader


```python
cmap = palette["kbc"]
bg_col = "white"
height = 800
width = int(np.round(1.6180 * height))
```


```python
%%time
cvs = ds.Canvas(plot_width=width, plot_height=height)
agg = cvs.points(g_df, "E", "g")
img = tf.shade(agg, cmap=cmap)
img = tf.set_background(img, bg_col)
img
```

    CPU times: user 809 ms, sys: 23.9 ms, total: 833 ms
    Wall time: 826 ms


<p align="center">
  <img width="800" src="/img/2021-07-22_01/output_45_1.png" alt="Goldbach's comet n=1e7">
</p>


We can clearly observe some dense lines in this "comet tail". In order to visualize this vertical distribution of prime pairs, we are are going to normalize $g$. As explained on the wikipedia [page](https://en.wikipedia.org/wiki/Goldbach%27s_comet):

> An illuminating way of presenting the comet data is as a histogram. The function g(E) can be normalized by dividing by the locally averaged value of g, gav, taken over perhaps 1000 neighboring values of the even number E. The histogram can then be accumulated over a range of up to about 10% either side of a central E. 


```python
%%time
g_df["g_av"] = g_df["g"].rolling(window=1000, center=True).mean()
g_df["g_norm"] = g_df["g"] / g_df["g_av"]
```

    CPU times: user 170 ms, sys: 67.8 ms, total: 238 ms
    Wall time: 230 ms



```python
%%time
cvs = ds.Canvas(plot_width=width, plot_height=height)
agg = cvs.points(g_df, "E", "g_norm")
img = tf.shade(agg, cmap=cmap)
img = tf.set_background(img, bg_col)
img
```

    CPU times: user 644 ms, sys: 27.9 ms, total: 672 ms
    Wall time: 674 ms


<p align="center">
  <img width="800" src="/img/2021-07-22_01/output_48_1.png" alt="Normalized Goldbach function">
</p>


We can now proceed to plot the histogram of the comet data, which will lead to some kind of cross section of the above plot:


```python
ax = g_df["g_norm"].hist(bins=1000, alpha=0.5, figsize=(20, 10))
ax.grid(True)
_ = ax.set(
    title="Histogram of the normalized Goldbach function",
    xlabel="Normalized number of prime pairs",
    ylabel="Number of occurrences",
)
_ = plt.xticks(np.arange(0.5, 3, 0.1))
```

<p align="center">
  <img width="800" src="/img/2021-07-22_01/output_50_0.png" alt="Histogram of the normalized Goldbach function">
</p>
    


## Prime E/2 values only


Finally, we are going to isolate a part of the most dense line from the comet tail (for a normalized number of prim pairs around 0.66$). As explained in [wikipedia](https://en.wikipedia.org/wiki/Goldbach%27s_comet):
    
> Of particular interest is the peak formed by selecting only values of E/2 that are prime. [...] The peak is very close to a Gaussian form. 


```python
g_df["E_half"] = (0.5 * g_df["E"]).astype(int)
```


```python
g_df_primes = g_df[g_df["E_half"].isin(np.nonzero(is_prime_vec)[0])]
```


```python
%%time
cvs = ds.Canvas(plot_width=width, plot_height=height)
agg = cvs.points(g_df_primes, "E", "g_norm")
img = tf.shade(agg, cmap=cmap)
img = tf.set_background(img, bg_col)
img
```

    CPU times: user 79 ms, sys: 0 ns, total: 79 ms
    Wall time: 75.9 ms


<p align="center">
  <img width="800" src="/img/2021-07-22_01/output_55_1.png" alt="Normalized Goldbach function for prime E/2 values">
</p>



```python
ax = g_df_primes["g_norm"].hist(bins=500, alpha=0.5, figsize=(20, 10))
ax.grid(True)
_ = ax.set(
    title="Histogram of the normalized Goldbach function for prime E/2 values only",
    xlabel="Normalized number of prime pairs",
    ylabel="Number of occurrences",
)
```

<p align="center">
  <img width="800" src="/img/2021-07-22_01/output_56_0.png" alt="Histogram of the normalized Goldbach function for prime E/2 values only">
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