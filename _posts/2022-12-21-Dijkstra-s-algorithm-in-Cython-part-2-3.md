---
title: Dijkstra's algorithm in Cython, part 2/3 WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
- Dijkstra
- Shortest path
- Cython
- Priority queue
- Binary heap
- Graphs
- Path algorithms
---

This post is the second part of a three-part series. In the [first part](https://aetperf.github.io/2022/12/21/Dijkstra-s-algorithm-in-Cython-part-1-3.html), we looked at the Cython implementation of Dijkstra's algorithm. In the current post, we are going to compare different priority queue implementations, using Dijkstra's algorithm on some road networks.

We apply the shortest path algorithm to the DIMACS road networks that we downloaded in a previous post: [Download some benchmark road networks for Shortest Paths algorithms
](https://aetperf.github.io/2022/09/22/Download-some-benchmark-road-networks-for-Shortest-Paths-algorithms.html).

The implementation of Dijkstra's algorithm is the one described in the first part: [Dijkstra's algorithm in Cython, part 1/3](https://aetperf.github.io/2022/12/21/Dijkstra-s-algorithm-in-Cython-part-1-3.html). The second approach is used, in which the priority queue is initialized with only the source vertex element.

Each priority queue is based on a slightly different heap. Here is a table with the different heaps that we are going to compare:

| priority queue label | heap | optimization | *d*-ary heap |
|----------------------|------|--------------|--------------|
| bin_basic | binary | no | yes |
| bin | binary | yes | yes |
| 3-ary | 3-ary | yes | yes |
| 4-ary | 4-ary | yes | yes |
| fib | Fibonacci | N.A. | no |


Here is a definition of *d*-ary heaps from [wikipedia](https://en.wikipedia.org/wiki/D-ary_heap):

> The *d*-ary heap or *d*-heap is a priority queue data structure, a generalization of the binary heap in which the nodes have *d* children instead of 2. Thus, a binary heap is a 2-heap, and a ternary heap is a 3-heap

So we refer to the binary heap as one of the *d*-arry heaps.

The optimization designates a small change in the `_min_heapify` part of the *d*-ary heaps. We will describe it in the second section of this post.

All these priority queues derive from the one described in the previous post: [A Cython implementation of a priority queue](https://aetperf.github.io/2022/11/23/A-Cython-implementation-of-a-min-priority-queue.html), except the one based on a Fibonacci heap. The priority queue based on a Fibonacci heap is taken from the [AequilibraE repository](https://github.com/AequilibraE/aequilibrae). AequilibraE is a Python package for transportation modeling. This priority queue was originally developed by [Jake VanderPlas](http://vanderplas.com/), and is part of the [SciPy](https://github.com/scipy/scipy) library.

The *bin_basic* priority queue is exacly the one presented in the post: [A Cython implementation of a priority queue](https://aetperf.github.io/2022/11/23/A-Cython-implementation-of-a-min-priority-queue.html).

## *d*-ary heaps

We implement the *d*-ary heaps as a slight modification of the binary heap. The only lines that are changed are located in the `_min_heapify` and `_decrease_key_from_node_index` functions. In a *d*-ary heap, a parent node has *d* children, so we need to change:
- the access to a parent from a child (`_decrease_key_from_node_index`)
- the loop on child nodes from a parent node (`_min_heapify`). We need to check *d* child nodes, and not just 2, in order find the child node with min key value.

For a node with index `i`, the parent is found at index `(i - 1) // d`. The child nodes of `i` are the nodes `d * i + 1`, ..., `d * i + d`.

## The `_min_heapify` optimization

Let's start by showing the code of the `_min_heapify` function in the `bin_basic` priority queue:

```cython
cdef inline void _min_heapify(
    PriorityQueue* pqueue,
    size_t node_idx) nogil:
    """Re-order sub-tree under a given node (given its node index) 
    until it satisfies the heap property.

    input
    =====
    * PriorityQueue* pqueue : priority queue
    * size_t node_idx : node index
    """
    cdef: 
        size_t l, r, i = node_idx, s

    while True:

        l =  2 * i + 1  # left child
        r = l + 1       # right child
        
        if (
            (l < pqueue.size) and 
            (pqueue.Elements[pqueue.A[l]].key < pqueue.Elements[pqueue.A[i]].key)
        ):
            s = l
        else:
            s = i

        if (
            (r < pqueue.size) and 
            (pqueue.Elements[pqueue.A[r]].key < pqueue.Elements[pqueue.A[s]].key)
        ):
            s = r

        if s != i:
            _exchange_nodes(pqueue, i, s)
            i = s
        else:
            break
```

We can observe that we always have `r > l`, since `r = l + 1`. So we do not need to check if `l < pqueue.size` if we know that `r < pqueue.size`. This saves us from a few `if` statements. This means that the `while` loop can be rewritten in the following way:

```cython
    while True:

        l =  2 * i + 1  # left child
        r = l + 1       # right child

        s = i
        val_min = pqueue.Elements[pqueue.A[s]].key
        if (r < pqueue.size):
            val_tmp = pqueue.Elements[pqueue.A[r]].key
            if val_tmp < val_min:
                s = r
                val_min = val_tmp
            val_tmp = pqueue.Elements[pqueue.A[l]].key
            if val_tmp < val_min:
                s = l
        else:
            if (l < pqueue.size):
                val_tmp = pqueue.Elements[pqueue.A[l]].key
                if val_tmp < val_min:
                    s = l

```

Also, we are using two `DTYPE_t` variables: `val_min` and `val_tmp`. A similar optimization is applied to the different *d*-ary heaps.

## Results

Package versions:

    Python version       : 3.10.8
    cython               : 0.29.32
    numpy                : 1.23.5

Computations are performed on a laptop with an 8 cores Intel i7-7700HQ CPU @ 2.80GHz, running Linux. Similarly to the first part of the post series, we checked the result against SciPy, only measured the execution time of the `run` phase (not the `setup` phase), and use the best time over 3 runs. We used the 3 largest DIMACS networks in order to get some significant elapsed time. Here are the features of these 3 networks:

| Network | vertex count | edge count |
|---------|-------------:|-----------:|
| W | 6262104 | 15119284 |
| CTR | 14081816 | 33866826 |
| USA | 23947347 | 57708624 |

<p align="center">
  <img width="600" src="/img/2022-12-21_02/heap_comparison.jpg" alt="heap comparison">
</p>

We see that the optimization of the `_min_heapify` only brings small benefits. Improvements due to the *3*-ary or *4*-ary heaps as compared to the binary heap is also of small magnitude, but still significant. Finally, we observe a clear advantage of the *d*-ary heaps over the Fibonacci one for this kind of networks. This might be due to a different underlying data structure?

Other improvements that we did not check in this post series could be:
- 0-based indexing in the heap tree for *d*-arry heaps
- a [monotone priority queue](https://en.wikipedia.org/wiki/Monotone_priority_queue)
- "lighter" priority queues without the `decrease_key` operation


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