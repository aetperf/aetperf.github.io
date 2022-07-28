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

$$N(x) \equiv \min \left\\{ n \in \mathbb{N}^+ \; s.t. \; \sum\_{k=1}^n U_k > x \right\\}$$

$$N(x) \equiv \min \left\\{ n \in \mathbb{N}^* \; s.t. \; \sum\_{k=1}^n U_k > x \right\\}$$

$$N(x) \equiv \min \left\\{ n \in \mathbb{N}^* \; s.t. \; \sum\_{k=1}^n U_k > x \right\\}$$

$$N(x) \equiv \min \left\\{ n \in \mathbb{N}^* \\; s.t. \\; \sum\_{k=1}^n U_k > x \right\\}$$

We are actually interested in the expected value of $N$:

$$m(x) \equiv E \left[ N(x) \right]$$
