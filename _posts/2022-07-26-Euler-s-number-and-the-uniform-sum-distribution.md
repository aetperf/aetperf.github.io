---
title: Euler's number and the uniform sum distribution
layout: post
comments: true
author: François Pacull
tags: 
- Python
- Euler's number
- Numba
- Irwin-Wall distribution
- Uniform sum distribution
- Numba
---

Last year I stumbled upon this tweet from *[@fermatslibrary](https://twitter.com/fermatslibrary)* [[1]](https://twitter.com/fermatslibrary/status/1388491536640487428?s=20): 

<p align="center">
  <img width="300" src="/img/2022-07-28_01/fermatslibrarys_tweet.png" alt="tweet">
</p>

I find it a little bit intriguing for Euler's number $e$ to appear here! But actually, it is not uncommon to encounter $e$ in probability theory. As explained by Stefanie Reichert in the article *e is everywhere* [2]:

> Imagine you are in Monte Carlo enjoying a few games of roulette, which is a Bernoulli trial process. If you place a bet on a single number, your chances are 1/37 to win that game. For 37 games, the probability that you will lose every single time is — maybe surprisingly — close to 1/e. Or, pretend you are at the theatre, where you — along with everybody else — leave your coat in the cloak room, which has one hook per guest, and receive a number. However, your coat is placed on a random hook. The probability that none of the coats are on the correct hook for a large number of guests approaches, again, 1/e. The number of practical examples is endless.


Let's derive mathematically the above statement from *Fermat's Library* and perform random experiments in Python and [Numba](https://numba.pydata.org/) to speed up the computations.

## A First Python experiment

We start with a very simple experiment to evaluate the average number of random numbers required to go over 1 when summing them:


```python
%%time
import numpy as np


def eval_N():
    """N is the number of random samples drawn from a uniform
    distribution over [0, 1) that we need to pick so that their
    sum is strictly greater than 1.

    Returns:
        float: The value of N.
    """
    s, N = 0.0, 0
    while s <= 1.0:
        s += np.random.rand()
        N += 1
    return N


RS = 124  # random seed
np.random.seed(RS)
n_eval = 1_000_000  # number of evaluations
m = 0.0  # m is the average value of n
for i in range(n_eval):
    m += eval_N()
m /= n_eval
print(f"m = {m:6.4f}")
```

    m = 2.7179
    CPU times: user 1.71 s, sys: 326 ms, total: 2.03 s
    Wall time: 1.51 s


Indeed, not so far from $e$!


```python
np.e
```




    2.718281828459045



Before we dive into some probability distributions, let's import all the other packages required for this notebook.


```python
from functools import partial
from math import comb, factorial

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from numba import jit, njit, prange

FS = (15, 8)  # figure size
CMAP = "plasma"  # color map
```

## Mathematical formulation

First of all, we consider some independent and identically distributed (i.i.d) random variables $ \left\\{ U_k \right\\}\_\{ k \geq 1\}$, each having standard uniform distribution  $\mathbf{U}(0,1)$. For $x > 0$, we define $N(x)$ as the following: 

$$N(x) \equiv \min \left\{ n \in \mathbb{N}^* \; s.t. \; \sum_{k=1}^n U_k > x \right\}$$

We are actually interested in the expected value of $N$:

$$m(x) \equiv E \left[ N(x) \right]$$


The above statement in *Fermat's Library*'s [tweet](https://twitter.com/fermatslibrary/status/1388491536640487428?s=20) can be expressed like this: 

> $m(1)$ is equal to $e$.

First of all, let's look at the distribution of $\sum_{k=1}^n U_k > x$, which is referred to as the *Irwin-Hall distribution*.

## The Irwin-Hall distribution

The *Irwin-Hall distribution* is the continuous probability distribution of the sum of $n$ i.i.d random variables, each having standard uniform distribution $\mathbf{U}(0,1)$:

$$X_n = \sum_{k=1}^n U_k$$

It is also called the *uniform sum distribution*. $X_n$ has a continuous distribution with support $[0, n]$ for $n \in \mathbb{N}^* $.

### Cumulative ditribution function of Irwin-Wall distribution

For $x > 0$ and $n \in \mathbb{N}^* $, the Cumulative Distribution Function (CDF) of $X_n$ is the following one:

$$F_{X_n}(x)= \frac{1}{n!} \sum_{k=0}^{\lfloor x \rfloor} (-1)^k  { {n}\choose{k} } (x-k)^n $$

See [[3]](https://www.randomservices.org/random/special/IrwinHall.html) for a complete derivation of this formulae. The CDF corresponds to the probability that the variable, $X_n$ in our case, takes on a value less than or equal to $x$:

$$P \left[ X_n \leq x \right] = F_{X_n}(x)$$



Let's write a function `cdf_th` to compute this CDF:


```python
def cdf_th(x, n=2):
    """Analytical cumulative distribution of the Irwin-Wall distribution.

    Args:
        x (float): input of the cumulative distribution function.
        n (int): number of uniform random variables in the summation.

    Returns:
        float: The function value at x.
    """
    if x <= 0.0:
        return 0.0
    elif x >= n:
        return 1.0
    else:
        x_floor = int(np.floor(x))
        cdf = 0
        for k in range(x_floor + 1):
            cdf += np.power(-1.0, k) * comb(n, k) * np.power(x - k, n)
        cdf /= factorial(n)
        return cdf
```

We can evaluate and plot this CDF for various values of $n$:


```python
x = np.linspace(-1, 11, 1000)
cdf_th_df = pd.DataFrame(data={"x": x})
for n in range(1, 11):
    y = list(map(partial(cdf_th, n=n), x))
    cdf_th_df[str(n)] = y

ax = cdf_th_df.plot(x="x", figsize=FS, grid=True, cmap=CMAP)
_ = ax.legend(title="n")
_ = ax.set(title="CDF of $X_n$", xlabel="x", ylabel="y")
_ = ax.set_xlim(-1, 11)
```


<p align="center">
  <img width="800" src="/img/2022-07-28_01/output_10_0.png" alt="CDF of $X_n$$">
</p>
    


But we can also try to approach the analytical CDF with some observations, adding scalar drawn from the uniform distribution in the interval [0, 1).

### The Empirical CDF

This part is inspired by a great blog post [[4]](https://www.rdatagen.net/post/a-fun-example-to-explore-probability/) by Keith Goldfeld, with some R code. We are going to use `np.random.rand()` to build an empirical CDF. 


```python
def cdf_emp(n=5, s=100_000, rs=124):
    """Empirical cumulative distribution of the Irwin-Wall distribution.

    Args:
        n (int): number of uniform random variables in the summation.
        s (int): number of samples used to evaluate the distribution.
        rs (int): random seed.

    Returns:
        (float, float): the x and y data points of the approached CDF.
    """
    arr = np.empty(s, dtype=np.float64)
    np.random.seed(rs)
    for i in prange(s):
        c = 0.0
        for k in range(n):
            c += np.random.rand()
        arr[i] = c
    x = np.sort(arr)
    y = np.arange(s) / np.float64(s)
    return x, y
```


```python
%%time
x, y = cdf_emp(n=10, rs=RS)
```

    CPU times: user 461 ms, sys: 3.7 ms, total: 465 ms
    Wall time: 463 ms



```python
cmap = matplotlib.cm.get_cmap(CMAP)
ax = cdf_th_df.plot(
    x="x", y="10", figsize=FS, alpha=0.6, label="Theoretical", c=cmap(0)
)
_ = ax.scatter(x, y, label="Empirical", alpha=0.6, s=50, color=cmap(0.8))
_ = ax.legend()
_ = ax.set(title="Theoretical and empirical CDF of $X_{10}$", xlabel="x", ylabel="y")
```


<p align="center">
  <img width="800" src="/img/2022-07-28_01/output_14_0.png" alt="Theoretical and empirical CDF of $X_{10}$">
</p>
    

Actually we can do that for several values of $n$, as we did precedently with the analytical CDF:


```python
cdf_emp_df = pd.DataFrame()
for n in range(1, 11):
    x, y = cdf_emp(n=n)
    cdf_emp_df["x_" + str(n)] = x
    cdf_emp_df["y_" + str(n)] = y
```


```python
ax = cdf_emp_df.plot(x="x_1", y="y_1", figsize=FS, label="1", c=cmap(0))
for n in range(2, 11):
    ax = cdf_emp_df.plot(
        x="x_" + str(n), y="y_" + str(n), ax=ax, label=str(n), c=cmap((n - 1) / 9)
    )
_ = ax.legend(title="n")
_ = ax.set(
    title="Empirical CDF of $X_n$",
    xlabel="x",
    ylabel="y",
)
_ = ax.set_xlim(-1, 11)
```


<p align="center">
  <img width="800" src="/img/2022-07-28_01/output_17_0.png" alt="Empirical CDF of $X_n$">
</p>
    


Now that we went over the CDF of the Irwin-Wall distribution, we can proceed to derive a formulae for $m(x)$.

## Mathematical derivation of $m(x)$

For $n \in \mathbb{N}^* $, we have:

$$
\begin{align*} 
P \left[ N(x) = n \right] &= P \left[ (X_{n-1} \leq x) \; \& \; (X_n > x) \right] \\
&= P\left[ (X_n > x) \; | \; (X_{n-1} \leq x) \right] P\left[ X_{n-1} \leq x \right] \\
&= \left( 1 - P\left[ (X_n \leq x) \; | \; (X_{n-1} \leq x) \right] \right) P\left[ X_{n-1} \leq x \right] \\
&= \left( 1 - \frac{P\left[ (X_{n-1} \leq x) \; \& \; (X_n \leq x) \right]}{P\left[ X_{n-1} \leq x \right]} \right) P\left[ X_{n-1} \leq x \right] \\
&= \left( 1 - \frac{P\left[ X_n \leq x \right]}{P\left[ X_{n-1} \leq x \right]} \right) P\left[ X_{n-1} \leq x \right] \\
&= P\left[ X_{n-1} \leq x \right] -  P\left[ X_n \leq x) \right]
\end{align*}
$$

A derivation for $x=1$ can be also found in the blog post [[4]](https://www.rdatagen.net/post/a-fun-example-to-explore-probability/) by Keith Goldfeld.

Because we know the CDF for $X_n$, and thus for $X_{n-1}$, we can write:

$$
\begin{align*} 
P \left[ N(x) = n \right] &= F_{X_{n-1}}(x) - F_{X_n}(x) \\
&= \frac{1}{(n-1)!} \sum_{k=0}^{\lfloor x \rfloor} (-1)^k { {n-1}\choose{k} } (x-k)^{n-1} \\
&- \frac{1}{n!} \sum_{k=0}^{\lfloor x \rfloor} (-1)^k { {n}\choose{k} } (x-k)^{n} \\
&= \frac{1}{(n-1)!} \sum_{k=0}^{\lfloor x \rfloor} \frac{(-1)^k (x-k)^{n-1}}{n} { {n}\choose{k} } (n-x)
\end{align*}
$$

Also, we can observe that $P \left[ N(x) = n \right] = 0$ if $ n < \lceil x \rceil$. This is because: 

$$\forall n \in \mathbb{N}^* , \; \sum_{k=1}^n U_k \leq n \; \Rightarrow \; N(x) \geq \lceil x \rceil$$

We can know derive a formulae for $m(x)$:

$$
\begin{align*}
m(x) &=  E\left[ N(x) \right] \\
&= \sum_{n=0}^{\infty} n P \left[ N(x) = n \right] \\
&= \sum_{n=\lceil x \rceil}^{\infty} \frac{n}{(n-1)!} \sum_{k=0}^{\lfloor x \rfloor} \frac{(-1)^k (x-k)^{n-1}}{n} { {n}\choose{k} } (n-x) \\
&= \sum_{n=\lceil x \rceil}^{\infty} n (n-x) \sum_{k=0}^{\lfloor x \rfloor} \frac{(-1)^k (x-k)^{n-1}}{k! (n-k)!} 
\end{align*}
$$

So let's find the exact value of $m(1)$ using the power series expansion $e^x = \sum_{n=0}^{\infty} \frac{x^n}{n!}$:

$$
\begin{align*}
m(1) &= \sum_{n=2}^{\infty} n (n-1) \sum_{k=0}^1 \frac{(-1)^k (1-k)^{n-1}}{k! (n-k)!} \\
&= \sum_{n=2}^{\infty} n (n-1) \left[ \frac{1}{n!} + 0 \right] \\
&= \sum_{n=2}^{\infty} \frac{1}{(n-2)!} = \sum_{n=0}^{\infty} \frac{1}{n!} = e
\end{align*}
$$


So we finally got our result: 

$$ m(1) = e $$

<p align="center">
  <img width="300" src="https://i.kym-cdn.com/photos/images/newsfeed/000/011/296/success_baby.jpg" alt="success">
</p>

With a little bit longer derivation, (this is why we are not going to display it here) we can show that: $$m(2) = e^2 -e$$

Wolfram MathWorld's web page about the *Uniform Sum Distribution* [[5]](https://mathworld.wolfram.com/UniformSumDistribution.html), also lists the next integer values for $x$:

$$
\begin{align*}
m(3) &= \frac{1}{2} \left( 2 e^3 -4 e^2 + e\right) \\
m(4) &= \frac{1}{6} \left( 6 e^4 -18 e^3 + 12 e^2 - e\right) \\
m(5) &= \frac{1}{24} \left( 24 e^5 - 96 e^4 + 108 e^3 - 32 e^2 + e\right) \\
\end{align*}
$$

We can write a function that approaches the $m(x)$ formulae. This is an *approximation* because we try to compute an infinite sum with a finite loop.


```python
def analytical_m(x, n_max=150):
    """Approximation of the analytical formulae for m(x).

    Args:
        x (float): input value of m.
        n_max (int): end of loop.

    Returns:
        float: the computed value for m(x).
    """
    x = float(x)
    m = 0.0
    for n in range(int(np.ceil(x)), n_max):
        s = 0.0
        for k in range(int(np.ceil(x))):
            s += (
                np.power(-1, k)
                * np.power(x - k, n - 1)
                / (np.math.factorial(k) * np.math.factorial(n - k))
            )
        m += n * (n - x) * s
    return m
```

Comparing the approximated analytical values of $m(x)$ with the exact values from Wolfram MathWorld [[5]](https://mathworld.wolfram.com/UniformSumDistribution.html), we get:


```python
m = {}
m[1] = np.e
m[2] = np.exp(2) - np.e
m[3] = 0.5 * (2 * np.exp(3) - 4 * np.exp(2) + np.e)
m[4] = (6 * np.exp(4) - 18 * np.exp(3) + 12 * np.exp(2) - np.e) / 6.0
m[5] = (
    24 * np.exp(5) - 96 * np.exp(4) + 108 * np.exp(3) - 32 * np.exp(2) + np.e
) / 24.0

for x in range(1, 6):
    print(
        f"x={x}, m({x}) = {m[x]:12.10f}, approximated m({x}) = {analytical_m(x):12.10f}"
    )
```

    x=1, m(1) = 2.7182818285, approximated m(1) = 2.7182818285
    x=2, m(2) = 4.6707742705, approximated m(2) = 4.6707742705
    x=3, m(3) = 6.6665656396, approximated m(3) = 6.6665656396
    x=4, m(4) = 8.6666044900, approximated m(4) = 8.6666044900
    x=5, m(5) = 10.6666620686, approximated m(5) = 10.6666620686