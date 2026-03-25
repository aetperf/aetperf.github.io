---
title: Summation of Floating-Point Numbers in Cython
layout: post
comments: true
author: François Pacull
categories: [Python, Numerical Computing]
tags:
- Python
- Cython
- floating-point
- summation
- Kahan
- numerical accuracy
- IEEE 754
- NumPy
- performance
image: /img/2026-03-25_01/Kahan.png
---

Adding up a list of floating-point numbers:

$$S_n = \sum_{i=0}^{n-1} x_i$$

seems simple. [IEEE 754](https://en.wikipedia.org/wiki/IEEE_754) double-precision floats (`float64`) carry about 15 to 17 significant decimal digits, and each individual addition is faithfully rounded, but rounding errors accumulate across successive operations. How fast they accumulate depends on the summation algorithm.

This post implements seven algorithms in [Cython](https://cython.org/), tests them on inputs with condition numbers ranging from 1 to $10^{300}$, and benchmarks them on a billion elements. Most of the algorithms come from the Wikipedia page on Kahan summation [[1]](#ref1), with error bounds drawn from Higham [[2]](#ref2). These bounds involve the *unit roundoff*, the maximum relative error introduced by rounding a real number to its nearest float64 representation: $u = 2^{-53} \approx 1.1 \times 10^{-16}$.

[William Kahan](https://en.wikipedia.org/wiki/William_Kahan), the principal architect of IEEE 754, published one of the first compensated summation algorithms in 1965, around the same time as Ivo Babuška, Ole Møller, or Jack M. Wolfe. Several of the methods below build on his ideas. These techniques have found use, for example, in celestial mechanics, where numerical integrators must sum billions of small force increments over large timescales without letting roundoff drift dominate the physics. 

<p align="center">
  <img src="/img/2026-03-25_01/Kahan.png" alt="William Kahan" width="400" /><br>
  <b>William Kahan (born 1933)</b> — <a href="https://www.heidelberg-laureate-forum.org/laureate/william-morton-kahan/">source</a>
</p>

**Outline**
- [Imports](#imports)
- [Recursive summation](#recursive)
- [Pairwise summation](#pairwise)
- [Kahan summation](#kahan)
- [Kahan-Babuška-Neumaier summation](#neumaier)
- [Kahan-Babuška-Klein summation](#klein)
- [Shewchuk summation](#shewchuk)
- [Accuracy comparison](#accuracy)
- [Scaling up with Accupy](#accupy)
- [Ill condition numbers](#condition)
- [Performance](#performance)
- [References](#references)

## Imports<a name="imports"></a>

All implementations below are written in Cython, a typed superset of Python that compiles to C. Each algorithm's inner loop is a `cdef` function callable only from C, wrapped by a thin `cpdef` function exposed to Python.

The Python package [Accupy](https://github.com/sigma-py/accupy/) by Nico Schlömer, generates ill-conditioned sums with known exact results, computed with arbitrary-precision arithmetic via [mpmath](https://mpmath.org/), which makes it useful for testing summation accuracy. Note that Accupy does not appear to be maintained anymore and requires NumPy < 2 (the latest version is 2.4.3 at the time of writing), so a dedicated environment is needed.

```python
import math
from time import perf_counter

import accupy
import numpy as np
import pandas as pd

%load_ext cython

DTYPE = np.float64  # 64 bit floating point data type used hereafter
```

Package versions:

    Python               : 3.13.11
    OS                   : Linux
    accupy               : 0.3.6
    numpy                : 1.26.4
    pandas               : 2.3.3

## Recursive summation<a name="recursive"></a>

The simplest approach: loop through the array and accumulate into a single variable. The relative error grows as $O(n \cdot u \cdot \text{cond})$, where the condition number of a sum is defined as:

$$\text{cond} = \frac{\sum_{i=0}^{n-1} |x_i|}{\left|\sum_{i=0}^{n-1} x_i\right|}$$

When all terms share the same sign the condition number is 1 and the error stays modest. Heavy cancellation, i.e. $\|\sum x_i \| \ll \sum \|x_i\|$, drives it up, amplifying rounding errors accordingly. As Goldberg puts it [[3]](#ref3):

> "The evaluation of any expression containing a subtraction (or an addition of quantities with opposite signs) could result in a relative error so large that *all* the digits are meaningless."

The error bound also depends on the order in which terms are added, since accumulating a large partial sum before adding small terms loses more low-order bits than the reverse. From Higham [[2]](#ref2):

> "When summing nonnegative numbers by recursive summation the increasing ordering is the best ordering, in the sense of having the smallest a priori forward error bound."

```cython
%%cython --compile-args=-Ofast

# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t recursive_cy(DTYPE_t[:] a) noexcept nogil:
    cdef:
        DTYPE_t s = 0.0
        size_t i, n = <size_t> a.shape[0]

    for i in range(n):
        s += a[i]

    return s

cpdef recursive_sum(a):
    return recursive_cy(a)
```

> **Compiler caveat.** The cell above is compiled with `-Ofast`, which implies `-ffast-math` and in particular `-funsafe-math-optimizations`. The compiler is then free to reorder floating-point additions, for instance by auto-vectorizing the loop with multiple accumulators, which can change the numerical result. For recursive summation this is acceptable: there is no compensation logic to protect. Starting from the pairwise cells below, all Cython code adds `-fno-unsafe-math-optimizations` to preserve the intended evaluation order. Without it, the compiler could optimize away expressions that are algebraically zero but matter in finite-precision arithmetic.

## Pairwise summation<a name="pairwise"></a>

Pairwise (or cascade) summation splits the array in half, sums each half recursively, and adds the two results. The computation forms a balanced binary tree of depth $\lceil \log_2 n \rceil$, so each input passes through at most that many additions. The error bound drops from $O(n \cdot u \cdot \text{cond})$ to $O(u \cdot \log_2 n \cdot \text{cond})$. For a million elements, that is roughly a 50 000-fold improvement over the recursive approach. NumPy uses a variant of this strategy with 8 accumulators and an unroll factor of 128 for SIMD vectorization.

Two implementations follow: a simple recursive split (`pairwise1`) and a version closer to NumPy's inner loop [[4]](#ref4) with 8 accumulators (`pairwise2`).

```cython
%%cython --compile-args=-Ofast --compile-args=-fno-unsafe-math-optimizations

# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t pairwise1_cy(DTYPE_t[:] a) noexcept nogil:
    cdef:
        DTYPE_t s
        size_t i, m, n_block = 64, n = <size_t> a.shape[0]

    if n < n_block:
        s = 0.0
        for i in range(n):
            s += a[i]
    else:
        m = <size_t> n // 2
        s = pairwise1_cy(a[:m]) + pairwise1_cy(a[m:])

    return s

cpdef pairwise1_sum(a):
    return pairwise1_cy(a)
```

The second version is closer to what `np.sum` uses internally, with 8 accumulators and a block size of 128:

```cython
%%cython --compile-args=-Ofast --compile-args=-fno-unsafe-math-optimizations

# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t

cdef ssize_t PW_BLOCKSIZE = 128

cdef DTYPE_t pairwise2_cy(DTYPE_t[:] a) noexcept nogil:
    cdef:
        DTYPE_t s
        size_t i, m, n = <size_t> a.shape[0]
        cdef DTYPE_t r[8]


    if n < 8:
        s = 0.0
        for i in range(n):
            s += a[i]
        return s
    elif n <= PW_BLOCKSIZE:
        #
        # sum a block with 8 accumulators
        # 8 times unroll reduces blocksize to 16 and allows vectorization with
        # avx without changing summation ordering
        #
        r[0] = a[0]
        r[1] = a[1]
        r[2] = a[2]
        r[3] = a[3]
        r[4] = a[4]
        r[5] = a[5]
        r[6] = a[6]
        r[7] = a[7]

        i = 8
        while i < n - (n % 8):
            r[0] += a[i + 0]
            r[1] += a[i + 1]
            r[2] += a[i + 2]
            r[3] += a[i + 3]
            r[4] += a[i + 4]
            r[5] += a[i + 5]
            r[6] += a[i + 6]
            r[7] += a[i + 7]
            i += 8

        # accumulate now to avoid stack spills for single peel loop
        s = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]))

        # do non multiple of 8 rest
        for i in range(i, n):
            s += a[i]

        return s
    else:
        # divide by two but avoid non-multiples of unroll factor
        m = n // 2
        m -= m % 8
        return pairwise2_cy(a[:m]) + pairwise2_cy(a[m:])

cpdef pairwise2_sum(a):
    return pairwise2_cy(a)
```

## Kahan summation<a name="kahan"></a>

The next three methods, Kahan, Neumaier, and Klein, are *compensated summation* algorithms. They track rounding errors in a separate variable and fold them back into the sum.

Kahan's original algorithm keeps a running compensation variable that accumulates the low-order bits lost at each step. The residual from one addition is folded back into the next, so the absolute error is bounded by $(2u + O(n u^2)) \sum \|x_i\|$, independent of $n$ to first order.

```cython
%%cython --compile-args=-Ofast --compile-args=-fno-unsafe-math-optimizations

# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp

ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t kahan_cy(DTYPE_t[:] a) noexcept nogil:
    """
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """
    cdef:
        DTYPE_t s = 0.0, c = 0.0, t, y  # c is a running compensation for lost low-order bits
        size_t i, n = <size_t> a.shape[0]

    for i in range(n):
        # c is zero the first time around.
        y = a[i] - c
        # Alas, s is big, y small, so low-order digits of y are lost.
        t = s + y
        # (t - s) cancels the high-order part of y; subtracting y recovers negative (low part of y).
        c = (t - s) - y
        # Algebraically, c should always be zero. Beware overly-aggressive optimizing compilers!
        s = t
        # Next time around, the lost low part will be added to y in a fresh attempt.

    return s

cpdef kahan_sum(a):
    return kahan_cy(a)
```

## Kahan-Babuška-Neumaier summation<a name="neumaier"></a>

Kahan's compensation captures the wrong residual when the incoming element is larger in magnitude than the running sum, exactly the situation in the first accuracy test below. Neumaier's variant checks which operand is larger and adjusts accordingly, fixing this weakness at no extra cost.

```cython
%%cython --compile-args=-Ofast --compile-args=-fno-unsafe-math-optimizations

# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp
from libc.math cimport fabs

ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t neumaier_cy(DTYPE_t[:] a) noexcept nogil:
    """
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """
    cdef:
        DTYPE_t s = 0.0, t, c = 0.0  # c is a running compensation for lost low-order bits
        size_t i, n = <size_t> a.shape[0]

    for i in range(n):
        t = s + a[i]
        if fabs(s) >= fabs(a[i]):
            c += (s - t) + a[i]  # If s is bigger, low-order digits of a[i] are lost.
        else:
            c += (a[i] - t) + s  # Else low-order digits of s are lost.
        s = t

    return s + c  # Correction only applied once in the very end.

cpdef neumaier_sum(a):
    return neumaier_cy(a)
```

## Kahan-Babuška-Klein summation<a name="klein"></a>

Klein's method is a *doubly compensated* algorithm: it applies the Neumaier correction a second time, compensating the compensation variable itself. This handles cases where the first-order correction term also loses significant digits, pushing the absolute error coefficient down from $O(u)$ to $O(u^2)$.

```cython
%%cython --compile-args=-Ofast --compile-args=-fno-unsafe-math-optimizations

# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp
from libc.math cimport fabs

ctypedef cnp.float64_t DTYPE_t

cdef DTYPE_t klein_cy(DTYPE_t[:] a) noexcept nogil:
    """
    https://en.wikipedia.org/wiki/Kahan_summation_algorithm
    """
    cdef:
        DTYPE_t s = 0.0, t, cs = 0.0, ccs = 0.0, c = 0.0, cc = 0.0
        size_t i, n = <size_t> a.shape[0]

    for i in range(n):
        t = s + a[i]
        if fabs(s) >= fabs(a[i]):
            c = (s - t) + a[i]
        else:
            c = (a[i] - t) + s
        s = t
        t = cs + c
        if fabs(cs) >= fabs(c):
            cc = (cs - t) + c
        else:
            cc = (c - t) + cs
        cs = t
        ccs += cc

    return s + cs + ccs

cpdef klein_sum(a):
    return klein_cy(a)
```

## Shewchuk summation<a name="shewchuk"></a>

All previous algorithms use a fixed number of accumulators and therefore have finite precision. Shewchuk's algorithm [[5]](#ref5) takes a different approach: it maintains a growable list of *non-overlapping partials* whose exact sum equals the true running total. Each new element is folded in via the fast two-sum primitive; only non-zero residuals are retained. The final result is the partials summed from largest to smallest. This is the algorithm behind Python's `math.fsum` [[6]](#ref6), and it is exact up to the final rounding to `float64`.

```cython
%%cython --compile-args=-Ofast --compile-args=-fno-unsafe-math-optimizations
# cython: language_level=3, boundscheck=False, wraparound=False, embedsignature=False, initializedcheck=False
# distutils: define_macros=NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION

cimport numpy as cnp
from libc.math cimport fabs
from libc.stdio cimport printf
from libc.stdlib cimport free, malloc, realloc
from libc.string cimport memset

ctypedef cnp.float64_t DTYPE_t

cdef ssize_t NUM_PARTIALS = 32

cdef DTYPE_t* reallocate_array(DTYPE_t* p, size_t old_size, size_t new_size) noexcept nogil:
    cdef DTYPE_t* new_p = <DTYPE_t*>realloc(p, new_size * sizeof(DTYPE_t))

    if new_p == NULL:
        # Handle realloc failure
        free(p)
        return NULL

    # Zero-initialize the new portion of the array
    # assert(new_size > old_size):
    memset(new_p + old_size, 0, (new_size - old_size) * sizeof(DTYPE_t))

    return new_p


cdef DTYPE_t shewchuk_cy(DTYPE_t[:] a) noexcept nogil:

    cdef:
        ssize_t i, j, m = 0, n = <size_t>a.shape[0] , s = NUM_PARTIALS, s_new
        DTYPE_t x, y, t, hi, yr, lo = 0.0
        DTYPE_t* p

    if n < NUM_PARTIALS:
        s = n

    p = <DTYPE_t*>malloc(s * sizeof(DTYPE_t))

    for idx in range(n):
        x = a[idx]
        i = 0
        for j in range(m):
            y = p[j]
            if fabs(x) < fabs(y):
               t = x 
               x = y
               y = t
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                p[i] = lo
                i += 1
            x = hi

        m = i  # ps[i:] = [x]
        if x != 0.0:
            if m >= s:
                # realloc
                s_new = s + s
                p = reallocate_array(p, s, s_new)
                s = s_new
            p[m] = x
            m += 1

    hi = 0.0
    if m > 0:
        m -= 1
        hi = p[m]
        # sum_exact(p, hi) from the top, stop when the sum becomes
        # inexact.
        while m > 0:
            x = hi
            m -= 1
            y = p[m]
            hi = x + y
            yr = hi - x
            lo = y - yr
            if lo != 0.0:
                break

        # Make half-even rounding work across multiple partials.
        # Needed so that sum([1e-16, 1, 1e16]) will round-up the last
        # digit to two instead of down to zero (the 1e-16 makes the 1
        # slightly closer to two). With a potential 1 ULP rounding
        # error fixed-up, math.fsum() can guarantee commutativity.
        if (m > 0) & (((lo < 0.0) & (p[m-1] < 0.0)) |
                      ((lo > 0.0) & (p[m-1] > 0.0))):
            y = lo * 2.0
            x = hi + y
            yr = x - hi
            if y == yr:
                hi = x

    free(p)

    return hi

cpdef shewchuk_sum(a):
    return shewchuk_cy(a)
```

The final block handles half-even rounding across multiple partials. After the top-down summation, if the residual `lo` and the next remaining partial share the same sign, `2*lo` should round the result up rather than down. Without this fix, `sum([1e-16, 1, 1e16])` would round the last digit to zero instead of two, and reordering the inputs would change the answer. With it, `math.fsum` guarantees commutativity.

## Accuracy comparison<a name="accuracy"></a>

The helper below runs every algorithm on the same array and prints `computed - true`. A value of `0.0` means the algorithm returned the exact `float64` answer.

```python
def compute_and_print(a, s_true):
    print(f"{'recursive':>15s} : {recursive_sum(a) - s_true}")
    print(f"{'pairwise1':>15s} : {pairwise1_sum(a) - s_true}")
    print(f"{'pairwise2':>15s} : {pairwise2_sum(a) - s_true}")
    print(f"{'numpy.sum':>15s} : {np.sum(a) - s_true}")
    print(f"{'Kahan':>15s} : {kahan_sum(a) - s_true}")
    print(f"{'Neumaier':>15s} : {neumaier_sum(a) - s_true}")
    print(f"{'Klein':>15s} : {klein_sum(a) - s_true}")
    print(f"{'Shewchuk':>15s} : {shewchuk_sum(a) - s_true}")
    print(f"{'math.fsum':>15s} : {math.fsum(a) - s_true}")
```

The first test is a 3-element array from Accupy's README: `[1e16, 1, -1e16]`, true sum `1.0`. The two large terms cancel, leaving only the middle element:

```python
a = np.array([1.0e16, 1.0, -1.0e16], dtype=DTYPE)
s_true = 1.0
compute_and_print(a, s_true)
```

          recursive : -1.0
          pairwise1 : -1.0
          pairwise2 : -1.0
          numpy.sum : -1.0
              Kahan : -1.0
           Neumaier : 0.0
              Klein : 0.0
           Shewchuk : 0.0
          math.fsum : 0.0

Recursive, pairwise, NumPy, and Kahan all lose the `1.0` entirely. Tracing Kahan step by step: after adding `1e16` the sum is exact (`s = 1e16`, `c = 0`). Adding `1` next, the compensation correctly captures the lost digit (`c = -1`). But at step 3 the compensation is destroyed: `y = a[2] - c = -1e16 + 1` rounds to `-1e16` in `float64` because the `+1` falls below the unit in the last place of `1e16`. The accumulated correction is lost. Neumaier's variant avoids this by never folding the compensation back into the next element; it keeps a separate running correction and only applies it at the very end.

The next example comes from the `math.fsum` source code:

```python
a = np.array([1e-16, 1, 1e16], dtype=DTYPE)
s_true = 1.0000000000000002e16
compute_and_print(a, s_true)
```

          recursive : -2.0
          pairwise1 : -2.0
          pairwise2 : -2.0
          numpy.sum : -2.0
              Kahan : -2.0
           Neumaier : -2.0
              Klein : -2.0
           Shewchuk : 0.0
          math.fsum : 0.0

Only Shewchuk and `math.fsum` return the correct result.

## Scaling up with Accupy<a name="accupy"></a>

[Accupy](https://github.com/sigma-py/accupy/) can generate arrays of arbitrary length with a prescribed condition number via `generate_ill_conditioned_sum`, and returns the exact sum computed with arbitrary precision:

```python
a, exact, cond = accupy.generate_ill_conditioned_sum(1000, 1.0e20)
a
```

    array([ 5.25331133e+06,  1.72398320e+09, -9.93291138e+04, ...,
           -7.60090448e+08,  3.76657913e-16,  5.99954380e+09])

```python
s_true = np.float64(exact)
compute_and_print(a, s_true)
```

          recursive : -182500.43325268468
          pairwise1 : 32768.29809249201
          pairwise2 : 98304.29809249201
          numpy.sum : 98304.29809249201
              Kahan : 14043.566748268988
           Neumaier : 2.7694735393879455e-11
              Klein : 0.0
           Shewchuk : 0.0
          math.fsum : 0.0

Recursive and pairwise sums are off by tens of thousands. Kahan reduces the error but is still off by thousands. Neumaier is accurate to 12 digits, while Klein, Shewchuk, and `math.fsum` return the exact `float64` result.

## Ill condition numbers<a name="condition"></a>

Accupy also returns the condition number as an mpmath `mpf` (arbitrary-precision float):

```python
cond
```

    mpf('1617919375038351020341.180641601355421988297017104159911557044663132765633303554530009976717849894093269')

Converted to `float64`:

```python
np.float64(cond)
```

    1.6179193750383511e+21

The condition number measures the intrinsic sensitivity of the summation problem to rounding errors. As noted in [[1]](#ref1), the relative error of every fixed-precision summation algorithm is proportional to this quantity. It is a property of the data, not of the method.

We can compute it ourselves for arbitrary arrays. The accuracy here comes from `math.fsum`, which tracks exact residuals for `float64` inputs, not from extended-precision types.

```python
def compute_cond(a):
    return math.fsum(np.abs(a)) / np.abs(math.fsum(a))
```

```python
compute_cond(a)
```

    1.617919375038351e+21

Pushing the condition number higher, to $\approx 10^{30}$:

```python
a, exact, cond = accupy.generate_ill_conditioned_sum(15, 1.0e30)
s_true = np.float64(exact)
compute_and_print(a, s_true)
```

          recursive : -252381674628155.3
          pairwise1 : -252381674628155.3
          pairwise2 : -252381674628155.3
          numpy.sum : -252381674628155.3
              Kahan : -252381674628155.3
           Neumaier : -0.02264500586761986
              Klein : 0.0
           Shewchuk : 0.0
          math.fsum : 0.0

```python
np.float64(cond)
```

    8.258292664776317e+30

```python
compute_cond(a)
```

    8.258292664776316e+30

We can also construct pathological patterns by hand. The repeating pattern `[1, 1e17, 1, -1e17]` tiled 10 000 times has a true sum of 20 000:

```python
n = 10000
a = np.array([1, 1e17, 1, -1e17] * n, dtype=DTYPE)
s_true = 2.0 * n
compute_and_print(a, s_true)
```

          recursive : -20000.0
          pairwise1 : -20000.0
          pairwise2 : -20000.0
          numpy.sum : -20000.0
              Kahan : -20000.0
           Neumaier : 0.0
              Klein : 0.0
           Shewchuk : 0.0
          math.fsum : 0.0

The next example is a 7-element array spanning 200 orders of magnitude, with a true sum of $10^{-100}$. Even Neumaier fails here; only Klein and Shewchuk recover the correct result:

```python
a = np.array([1e100, 1, -1e100, 1e-100, 1e50, -1, -1e50], dtype=DTYPE)
s_true = 1e-100
compute_and_print(a, s_true)
```

          recursive : -1e-100
          pairwise1 : -1e-100
          pairwise2 : -1e-100
          numpy.sum : -1e-100
              Kahan : -1e-100
           Neumaier : -1e-100
              Klein : 0.0
           Shewchuk : 0.0
          math.fsum : 0.0

```python
accupy.cond(a)
```

    2e+200

```python
compute_cond(a)
```

    2e+200

Finally, a 9-million-element array built from a repeating 9-element pattern spanning $10^{-100}$ to $10^{200}$:

```python
n = 1_000_000
a = np.array(
    [1.0e200, 1.0e-1, 1.0, -1.0e200, -1.0e-1, 1.0e100, 1.0e-100, -1.0, -1.0e100] * n,
    dtype=DTYPE,
)
s_true = n * 1e-100
compute_and_print(a, s_true)
```

          recursive : -1e-94
          pairwise1 : 6.4e+102
          pairwise2 : -5.600000000000001e+102
          numpy.sum : -1.6e+101
              Kahan : -1e-94
           Neumaier : -1e-94
              Klein : -9.99999e-95
           Shewchuk : 0.0
          math.fsum : 0.0

Most failing methods return 0 (error of `-1e-94`), but all three pairwise variants (`pairwise1`, `pairwise2`, `numpy.sum`) blow up to errors of order `1e101`–`1e102`. The cause is block-pattern misalignment. The input repeats a 9-element cycle, but pairwise summation splits the array into blocks whose size is not a multiple of 9. Block boundaries cut across the pattern: a block contains several complete cycles plus a few leftover elements from the next cycle, e.g. `1e200` and `1e-1`. The complete cycles nearly cancel, but the leftover leaves the block sum around `1e200`. These massive intermediate partial sums then cancel against each other higher up in the binary tree, and catastrophic cancellation at that scale produces errors far larger than the true sum. Recursive summation avoids this because it processes elements one by one: `+1e200` is followed by `-1e200` just 3 positions later, so the running sum never stays at `1e200` for long. The result is still wrong, but the error stays at the scale of the true sum rather than exploding.

The three pairwise variants differ in magnitude because their block sizes and splitting strategies are not identical.

```python
compute_cond(a)
```

    2e+300

At condition number $2 \times 10^{300}$, every algorithm except Shewchuk and `math.fsum` fails. Klein's second-order compensation still has finite precision; at some point the accumulated corrections themselves lose significant digits.

## Performance<a name="performance"></a>

Each algorithm is now timed on the first billion terms of the Basel series ($\sum 1/k^2$), a well-conditioned input where the bottleneck is pure summation throughput rather than numerical difficulty.

A sanity check on 1M terms first. The errors are all around $10^{-6}$, reflecting the truncation error from stopping the series early, not a summation accuracy issue:

```python
%%time
n = 1_000_000
a = np.power(1.0 / np.arange(1, n + 1, dtype=DTYPE), 2)
s_true = np.power(np.pi, 2) / 6.0
compute_and_print(a, s_true)
```

          recursive : -9.999994563525405e-07
          pairwise1 : -9.999995005394169e-07
          pairwise2 : -9.999995000953277e-07
          numpy.sum : -9.999994996512385e-07
              Kahan : -9.99999499873283e-07
           Neumaier : -9.99999499873283e-07
              Klein : -9.99999499873283e-07
           Shewchuk : -9.99999499873283e-07
          math.fsum : -9.99999499873283e-07
    CPU times: user 44.4 ms, sys: 4.1 ms, total: 48.5 ms
    Wall time: 47.7 ms

```python
%%time
n = 1_000_000_000
a = np.power(1.0 / np.arange(1, n + 1, dtype=DTYPE), 2)

funcs = {
    "recursive": recursive_sum,
    "pairwise1": pairwise1_sum,
    "pairwise2": pairwise2_sum,
    "numpy.sum": np.sum,
    "Kahan": kahan_sum,
    "Neumaier": neumaier_sum,
    "Klein": klein_sum,
    "Shewchuk": shewchuk_sum,
    "math.fsum": math.fsum,
}

timings = []
for func_name, func in funcs.items():
    start = perf_counter()
    _ = func(a)
    et_s = perf_counter() - start
    timings.append({"summation": func_name, "elapsed_time_s": et_s})
```

    CPU times: user 45.7 s, sys: 4.32 s, total: 50 s
    Wall time: 49.8 s

```python
df = pd.DataFrame(timings)
df = df.sort_values(by="elapsed_time_s", ascending=True)
df = df.set_index("summation")
```

```python
ax = df.plot.barh(legend=False, grid=True, alpha=0.7)
_ = ax.set(
    title="Timings of various summation techniques",
    xlabel="Elapsed time (s)",
    ylabel="Summation",
)
```

<p align="center">
  <img width="600" src="/img/2026-03-25_01/output_58_0.png" alt="Timings of various summation techniques">
</p>

Shewchuk and `math.fsum` are roughly 10x slower than recursive summation and compress the rest of the chart. Both use the same algorithm, and `math.fsum` is implemented in C [[6]](#ref6), but it accepts any Python iterable and calls `PyFloat_AsDouble` on each element. Our Cython version uses a typed memoryview with direct pointer access to the array memory, avoiding the per-element Python object overhead. Zooming in on the remaining algorithms:

```python
ax = df.iloc[:-2].plot.barh(legend=False, grid=True, alpha=0.7)
_ = ax.set(
    title="Timings of various summation techniques",
    xlabel="Elapsed time (s)",
    ylabel="Summation",
)
```

<p align="center">
  <img width="600" src="/img/2026-03-25_01/output_60_0.png" alt="Timings without Shewchuk and math.fsum">
</p>

Pairwise summation achieves $O(u \log n)$ accuracy at nearly the same speed as recursive summation, thanks to SIMD-friendly memory access patterns. The compensated methods (Kahan, Neumaier, Klein) cost more. Shewchuk's growable partials list involves branching and dynamic memory management on every element, which accounts for the large gap.

For well-conditioned sums, pairwise summation gives the best accuracy per cycle. When condition numbers go above $\approx 10^{15}$, only Shewchuk / `math.fsum` can be trusted. But a wise advice from Higham [[2]](#ref2) is:

> "If high accuracy is important, consider implementing recursive summation in higher precision; if feasible this may be less expensive (and more accurate) than using one of the alternative methods at the working precision."

## References<a name="references"></a>

<a name="ref1"></a>[1] Wikipedia, [Kahan summation algorithm](https://en.wikipedia.org/wiki/Kahan_summation_algorithm)

<a name="ref2"></a>[2] N.J. Higham, *Accuracy and Stability of Numerical Algorithms*, 2nd edition, SIAM, 2002

<a name="ref3"></a>[3] D. Goldberg, [What Every Computer Scientist Should Know About Floating-Point Arithmetic](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html), *ACM Computing Surveys* 23(1), 1991

<a name="ref4"></a>[4] NumPy, [pairwise summation inner loop](https://github.com/numpy/numpy/blob/d6bfeb02241c8307e64e2f419821b2fc9b021862/numpy/_core/src/umath/loops_utils.h.src#L81), source code

<a name="ref5"></a>[5] J.R. Shewchuk, Adaptive Precision Floating-Point Arithmetic and Fast Robust Geometric Predicates, *Discrete & Computational Geometry* 18(3), 1997

<a name="ref6"></a>[6] CPython, [`mathmodule.c`](https://github.com/python/cpython/blob/main/Modules/mathmodule.c), `math.fsum` implementation
