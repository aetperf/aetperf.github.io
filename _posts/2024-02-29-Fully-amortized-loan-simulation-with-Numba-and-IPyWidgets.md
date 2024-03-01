---
title: Fully Amortized Loan simulation with Numba and IPyWidgets
layout: post
comments: true
author: François Pacull
tags: 
- Python
- Numba
- IPyWidgets
---


In this blog post, we will show how to use Python to simulate the amortization of a fully amortized loan, such as a mortgage or a car loan. We will derive the formula for calculating the monthly payments and the outstanding balance, and implement them in Python using [Numba](https://numba.pydata.org/). We will also use the [IPyWidgets](https://ipywidgets.readthedocs.io/en/stable/) library to create interactive widgets that will allow us to explore the effects of different loan parameters on the amortization schedule.

## Amortization formulas

When you take out a loan, you typically agree to repay the loan in equal monthly installments over a fixed period of time. This type of loan is known as a fully amortized loan. We will use the following variables in our derivation:

$A$ : the principal amount borrowed  
$M$ : the total number of monthly payments  
$r$ : the annual interest rate  
$i$ : the monthy interest rate  
$P_m$ : the principal part of monthly payment $m$  
$I_m$ : the interest part of monthly payment $m$  
$T$ : the constant monthly payment  
$B_m$ : the balance (principal still due) after the $m$-th payment

We have the following identities:
- $i = r / 12$
- $I_1 = i A$
- $\sum_{m=1}^M P_m = A$
- $B_M = 0$

$\forall \; 1 \leq m \leq M:$
- $T = P_m + I_m$
- $B_m =A - (P_1 + ... + P_m)$
  
$\forall \; 2 \leq m \leq M:$
- $I_m = i B_{m-1}$
- $B_m = B_{m-1} - P_m$

Since $A$, $r$ and $M$ are known, it is easy to compute $I_1$. However, we need to compute eather $T$ or $P_1$ in order to be able to perform the simulation. Here is the derivation of the $P_1$ formula. 

It is easy to show by recursion that:

$$P_m=(T-iA)(1+i)^{m-1}$$

Indeed, we have $P_1=T-I_1=T-iA$, and:

$$
\begin{align*}
P_{m+1} &= T - I_{m+1} \\
        &= T - i B_m \\
        &= T - i (B_{m-1} - P_m) \\
        &= (T-I_m) + i P_m \\
        &= P_m + i P_m \\
        &= P_m (1+i)
\end{align*}
$$

By summing up all the $P_m$ terms, we get:

$$\sum_{m=1}^M P_m = A = (T-iA) \sum_{m=1}^M (1+i)^{m-1}$$

And so:

$$T= \frac{A}{\sum_{m=0}^{M-1} (1+i)^m} + i A$$

For $i>0$, the denominator of the fraction can be simplified in the following way:

$$
\begin{align*}
\sum_{m=0}^{M-1} (1+i)^m &= \frac{(1+i)-1}{i} \sum_{m=0}^{M-1} (1+i)^m \\
&= \frac{1}{i}  \left( \sum_{m=1}^{M} (1+i)^m - \sum_{m=0}^{M-1} (1+i)^m  \right) \\
&= \frac{1}{i}  \left( (1+i)^M - 1  \right)
\end{align*}
$$

This leads to: 

$$T = \frac{i A}{(1+i)^M - 1} + i A$$

And eventually: 

$$P_1 = \frac{i A}{(1+i)^m - 1}$$

## Imports

In the next section, we will implement these formulas in Python. First, let's import the necessary libraries:

```python
import ipywidgets as widgets
import numpy as np
import pandas as pd
from ipywidgets import interact
from numba import jit
```

We are operating on Python version 3.11.5 and running on a Linux x86_64 machine.


    ipywidgets             : 8.0.7
    numpy                  : 1.24.4
    pandas                 : 2.1.3
    numba                  : 0.57.1


## Implementing the Amortization Formulas in Python

```python
@jit(nopython=True)
def _compute_amortized_loan_inner(a, r, mc):
    """
    a: amount, r: annual interest rate, mc: month count
    """
    P = np.empty(mc, dtype=np.float64)  # principal part
    I = np.empty(mc, dtype=np.float64)  # interest part
    B = np.empty(mc, dtype=np.float64)  # balance

    # init
    if r > 0.0:
        P[0] = r * a / (12.0 * ((1.0 + r / 12.0) ** mc - 1.0))
    else:
        P[0] = a / mc
    I[0] = r * a / 12.0
    t = I[0] + P[0]
    B[0] = a - P[0]

    # loop on months
    for m in range(1, mc):
        I[m] = r * B[m - 1] / 12.0
        P[m] = t - I[m]
        B[m] = B[m - 1] - P[m]

    return P, I, B, t


def compute_amortized_loan(amount=200_000, interest_rate_pc=4.0, period_m=180):
    """Compute the amortization schedule for a fully amortized loan.

    Parameters
    ----------
    amount : float
        The principal amount borrowed.
    interest_rate_pc : float
        The annual interest rate, in percentage points.
    period_m : int
        The total number of monthly payments.

    Returns
    -------
    df : pandas.DataFrame
        A DataFrame containing the amortization schedule, with one row for
        each monthly payment. The columns of the DataFrame are:

        * `principal`: the principal part of the monthly payment
        * `interest`: the interest part of the monthly payment
        * `balance`: the outstanding balance after the monthly payment
        * `total`: the total monthly payment
    """

    r = 0.01 * interest_rate_pc

    P, I, B, t = _compute_amortized_loan_inner(amount, r, period_m)

    df = pd.DataFrame(
        data={"principal": P, "interest": I, "balance": B},
        index=np.arange(1, period_m + 1),
    )
    df.rename_axis("month", inplace=True)
    df["total"] = t

    return df
```

Now we can use the compute_amortized_loan function to calculate the amortization schedule for a loan of $200,000 with an annual interest rate of 3.5% over a period of 180 months (15 years).

```python
%%time
df = compute_amortized_loan(amount=200_000, interest_rate_pc=3.5, period_m=180)
```

    CPU times: user 709 ms, sys: 1.31 s, total: 2.02 s
    Wall time: 316 ms



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
      <th>principal</th>
      <th>interest</th>
      <th>balance</th>
      <th>total</th>
    </tr>
    <tr>
      <th>month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>846.431749</td>
      <td>583.333333</td>
      <td>199153.568251</td>
      <td>1429.765083</td>
    </tr>
    <tr>
      <th>2</th>
      <td>848.900509</td>
      <td>580.864574</td>
      <td>198304.667742</td>
      <td>1429.765083</td>
    </tr>
    <tr>
      <th>3</th>
      <td>851.376468</td>
      <td>578.388614</td>
      <td>197453.291274</td>
      <td>1429.765083</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(3)
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
      <th>principal</th>
      <th>interest</th>
      <th>balance</th>
      <th>total</th>
    </tr>
    <tr>
      <th>month</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>178</th>
      <td>1417.327263</td>
      <td>12.437820</td>
      <td>2.847068e+03</td>
      <td>1429.765083</td>
    </tr>
    <tr>
      <th>179</th>
      <td>1421.461134</td>
      <td>8.303949</td>
      <td>1.425607e+03</td>
      <td>1429.765083</td>
    </tr>
    <tr>
      <th>180</th>
      <td>1425.607062</td>
      <td>4.158021</td>
      <td>-5.029506e-10</td>
      <td>1429.765083</td>
    </tr>
  </tbody>
</table>
</div>



## Plot

We will now create static and interactive visualizations of the amortization schedule.

## Static


```python
def plot_amortized_loan(amount=100_000, interest_rate_pc=3.5, period_m=180):
    df = compute_amortized_loan(amount, interest_rate_pc, period_m)
    ax = df[["principal", "interest"]].plot.area(
        stacked=True, alpha=0.6, figsize=(10, 6)
    )
    cost = (df.total - df.principal).sum()
    total = df.total.values[0]
    ax.legend(loc="center right")
    _ = ax.set_xlim(1, period_m)
    _ = plt.text(
        x=0.05 * period_m, y=0.15 * total, s=f"Monthly payment = {total:10.2f}"
    )
    _ = plt.text(
        x=0.05 * period_m,
        y=0.05 * total,
        s=f"Total cost = {cost:10.2f} ({100.*cost/amount:.2f} %)",
    )
    _ = ax.set(
        title=f"Fully amortized loan simulation\namount={amount:.0f}, rate={interest_rate_pc:.2f} %, months={period_m}",
        xlabel="Month",
        ylabel="Monthly payment",
    )
```


```python
plot_amortized_loan(amount=110_000, interest_rate_pc=3.78, period_m=180)
```

<p align="center">
  <img width="800" src="/img/2024-02-29_01/output_11_0.png" alt="static">
</p> 
    


## Interactive

We will now use the [IPyWidgets](https://ipywidgets.readthedocs.io/en/stable/) library to create interactive widgets that will allow us to explore the effects of different loan parameters on the amortization schedule.

```python
_ = interact(
    plot_amortized_loan,
    amount=widgets.FloatSlider(
        value=200000,
        min=10000,
        max=250000,
        step=1000,
        description="Amount",
        continuous_update=False,
        readout_format=".0f",
    ),
    interest_rate_pc=widgets.FloatSlider(
        value=3.5,
        min=0.0,
        max=7.5,
        step=0.01,
        description="Rate (%)",
        continuous_update=False,
        readout_format=".2f",
    ),
    period_m=widgets.IntSlider(
        value=120,
        min=2,
        max=360,
        step=1,
        description="Months",
        continuous_update=False,
        readout_format="d",
    ),
)
```

<p align="center">
  <img width="800" src="/img/2024-02-29_01/ipywidget.png" alt="ipywidget">
</p> 

Thanks to Numba, each adjustment made using the slider widget triggers a swift response, leading to an almost instantaneous update of the figure.

## References

[1] Bret D. Whissel, A Derivation of Amortization, [https://www.bretwhissel.net/amortization/amortize2col.pdf](https://www.bretwhissel.net/amortization/amortize2col.pdf)
