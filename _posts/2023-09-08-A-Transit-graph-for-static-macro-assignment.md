---
title: A Transit graph for static macro assignment WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
- static macro assignment
- Hyperpath
- Transit network
---


This post is a description of a graph structure for a *transit network*, used for *static*, *link-based*, *frequency-based* *assignment*, such as the classic algorithm "Optimal strategies" by Spiess & Florian [1]. This assignment algorithm has been implemented in [AequilibraE](http://www.aequilibrae.com/python/latest/) package, a comprehensive Python package for transportation modeling that offers various functionalities and features.

Let's start by giving a few short definitions:

- *transit* definition from [Wikipedia](https://en.wikipedia.org/wiki/Public_transport):

> system of transport for passengers by group travel systems available for use by the general public unlike private transport, typically managed on a schedule, operated on established routes, and that charge a posted fee for each trip.

- *transit network*: a set of transit lines and stops, where passengers can board, alight or change vehicles. 

- *assignment*: distribution of the passengers (demand) on the network (supply), knowing that transit users attempt to minimize total travel time, time or distance walking, time waiting, number of transfers, fares, etc...

- *static* assignment : assignment without time evolution. Dynamic properties of the flows, such as congestion, are not well described, unlike with dynamic assignment models. 

- *frequency-based* (or *headway-based*) as opposed to schedule-based : schedules are averaged in order to get line frequencies. In the schedule-based approach, distinct vehicle trips are represented by distinct links. We can see the associated network as a time-expanded network, where the third dimension would be time. 

- *link-based*: the assignment algorithm is not evaluating paths, or any aggregated information besides attributes stored by nodes and links. In the present case, each link has an associated cost (travel time) `c` [s] and frequency `f` [1/s].

We are going at first to describe the input transit network, which is mostly composed of stops, lines and zones.

## Transit stops and stations

Transit stops are points where passenger can board, alight or change vehicles. Also, they can be part of larger stations, where stops are connected by transfer links.

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208088240-38e72a88-569e-4b12-a3df-9bc1fb1a4e62.jpg" alt="208088240-38e72a88-569e-4b12-a3df-9bc1fb1a4e62">
</p>

In this figure, we have two stops : A and B, which belong to the same station (in red).

## Transit lines

A transit line is a set of services that may use different routes. 

### Transit routes

A routes is described by a sequence of stop nodes. We assume here the routes to be directed. For example, we can take a simple case with 3 stops:

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208088742-2c51a9f5-298c-4a1a-af10-edcde6437d0c.jpg" alt="208088742-2c51a9f5-298c-4a1a-af10-edcde6437d0c">
</p>

In this case, the `L1` line is made of two distinct routes: 
- ABC 
- CBA.  

But we can have many different configurations:
- a partial route at a given moment of the day: AB, 
- a route with an additional stop : ABDC
- a route that does not stop at a given stop: AC

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208088800-5f65028f-8040-49bb-89d7-ad81b013538b.jpg" alt="208088800-5f65028f-8040-49bb-89d7-ad81b013538b">
</p>


So lines can be decomposed into multiple "sub-lines" depending on the distinct routes, with distinct elements being part of the same commercial line. In the above case, we would have for example:

|line id|commercial name|stop sequence| headway (s) |
|-------|---------------|-------------|-------------|
|L1_a1  | L1            | ABC         | 600 |
|L1_a2  | L1            | ABDC        | 3600 |
|L1_a3  | L1            | AB          | 3600 |
|L1_a4  | L1            | AC          | 3600 |
|L1_b1  | L1            | CBA         | 600 |

The headway is associated to each sub-line and corresponds to the mean time range between consecutive vehicles. It is related to the inverse of the line frequency. The frequency is what is used as link attribute in the assignment algorithm.

### Line segments

A line segment is a portion of a transit line between two consecutive stops. With the previous example line `L1_a1`, we would get two distinct line segments:

|line id|segment index| origin stop | destination stop | travel_time (s) |
|----------|---------|-------------|------------------|--------------|
| L1_a1 | 1 | A | B | 300 |
| L1_a1 | 2 | B | C | 600 |

Note that we included a travel time for each line segment. This is another link attribute used by the assignment algorithm.

## Transit assignment zones and connectors

In order to assign the passengers on the network, we also need to express the demand in the different regions of the network. This is why the network area is decomposed into a partition of transit assignment zones, for example into 4 non-overlapping zones:

<p align="center">
  <img width="600" src="/img/2023-09-08_01/208088960-bd088858-ef7b-43e5-86bc-341eb8f6e7b0.jpg" alt="208088960-bd088858-ef7b-43e5-86bc-341eb8f6e7b0">
</p>


Then the demand is express as a number of trips from each zone to each zone: a 4 by 4 Origin/Destination (OD) matrix in this case. 

Also, each zone centroid is connected to some network nodes, in order to connect the supply and demand. These are the *connectors*.

<p align="center">
  <img width="600" src="/img/2023-09-08_01/208089058-a735d969-5f13-4ab4-b983-2c637e865aa4.jpg" alt="208089058-a735d969-5f13-4ab4-b983-2c637e865aa4">
</p>

We now have all the elements required to describe the assignment graph.

## The Assignment graph

### Link and node types

The transit network is used to generate a graph with specific nodes and links used to model the transit process. Links can be of different types:
- *on-board*
- *boarding*
- *alighting*
- *dwell*
- *transfer* 
- *connector*
- *walking*

Nodes can be of the following types:
- *stop*
- *boarding*
- *alighting*
- *od* 
- *walking* 

Here is a figure showing how a simple stop is described:

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208089118-72766743-ce62-4f25-8296-026a8f9657b5.jpg" alt="208089118-72766743-ce62-4f25-8296-026a8f9657b5">
</p>


The waiting links are the *boarding* and *transfer* links. Basically, each line segment is associated with a *boarding*, an *on-board* and an *alighting* link. 

*Transfer* links appear between distinct lines at the same stop:

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208089209-885dd6ac-f3e6-43e0-b8ff-f548a375aec9.jpg" alt="208089209-885dd6ac-f3e6-43e0-b8ff-f548a375aec9">
</p>

They can also be added between all the lines of a station if increasing the number of links is not an issue.

*walking* links connect *stop* nodes, while *connector* links connect the zone centroids (*od* nodes) to *stop* nodes:

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208089273-6ab4c267-7591-4f77-a1c1-88d072927061.jpg" alt="208089273-6ab4c267-7591-4f77-a1c1-88d072927061">
</p>

Connectors that connect *od* to *stop* nodes allow passengers to access the network, while connectors in the opposite direction allow them to egress. Walking nodes/links may be used to connect stops from distant stations.

### Link attributes

Here is a table that summarize the link characteristics/attributes depending on the link types:

| link type | from node type | to node type | cost | frequency |
|-----------|----------------|--------------|------|-----------|
|*on-board*|*boarding*|*alighting*| trav. time | $\infty$ |
|*boarding*|*stop*|*boarding*| const. | line freq. |
|*alighting*|*alighting*|*stop*| const. | $\infty$ |
|*dwell*|*alighting*|*boarding*| const. | $\infty$ |
|*transfer*|*alighting*|*boarding*| const. + trav. time | dest. line freq. |
|*connector*|*od* or *stop*|*od* or *stop*| trav. time | $\infty$ |
|*walking*|*stop* or *walking*|*stop* or *walking*| trav. time | $\infty$ |

The travel time is specific to each line segment or walking time. For example, there can be 10 minutes connection between stops in a large transit station. A constant boarding and alighting time is used all over the network. The *dwell* links have constant cost equal to the sum of the alighting and boarding constants.

We can use more attributes for specific link types, e.g.:
- *line_id*: for *on-board*, *boarding*, *alighting* and *dwell* links.
- *line_seg_idx*: the line segment index for *boarding*, *on-board* and *alighting* links.
- *stop_id*: for *alighting*, *dwell* and *boarding* links. This can also apply to *transfer* links for inner stop transfers.
- *o_line_id*: origin line id for *transfer* links
- *d_line_id*: destination line id for *transfer* links

Next, we are going see a classic transit network example with only four stops and four lines.

## A Small example : Spiess and Florian

This example is taken from *Spiess and Florian* [1]:

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208089367-5e636a8e-c133-425d-bc7c-0c9b4af7a038.jpg" alt="208089367-5e636a8e-c133-425d-bc7c-0c9b4af7a038">
</p>

Travel time is indicated on the figure. We have the following four line characteristics:

|line id|route|headway (min)| frequency (1/s) |
|-------|-----|------------:|-----------------|
|L1|AB|12|0.001388889|
|L2|AXY|12|0.001388889|
|L3|XYB|30|0.000555556|
|L4|YB|6|0.002777778|

Passengers want to go from A to B, so we can divide the network area into two distinct zones: TAZ 1 and TAZ 2. The assignment graph associated to this network has 26 links:

<p align="center">
  <img width="800" src="/img/2023-09-08_01/208089460-913526d1-fd40-4ed8-b1a3-65cf264de336.jpg" alt="208089460-913526d1-fd40-4ed8-b1a3-65cf264de336">
</p>


Here is a table listing all links :

| link id | link type | line id | cost | frequency |
|---------|-----------|---------|------|-----------|
|1|*connector*||0|$\infty$|
|2|*boading*|L1|0|0.001388889|
|3|*boading*|L2|0|0.001388889|
|4|*on-board*|L1|1500|$\infty$|
|5|*on-board*|L2|420|$\infty$|
|6|*alighting*|L2|0|$\infty$|
|7|*dwell*|L2|0|$\infty$|
|8|*transfer*||0|0.000555556|
|9|*boarding*|L2|0|0.001388889|
|10|*boarding*|L3|0|0.000555556|
|11|*on-board*|L2|360|$\infty$|
|12|*on-board*|L3|240|$\infty$|
|13|*alighting*|L3|0|$\infty$|
|14|*alighting*|L2|0|$\infty$|
|15|*transfer*|L3|0|0.000555556|
|16|*transfer*||0|0.002777778|
|17|*dwell*|L3|0|$\infty$|
|18|*transfer*||0|0.002777778|
|19|*boarding*|L3|0|0.000555556|
|20|*boarding*|L4|0|0.002777778|
|21|*on-board*|L3|240|$\infty$|
|22|*on-board*|L4|600|$\infty$|
|23|*alighting*|L4|0|$\infty$|
|24|*alighting*|L3|0|$\infty$|
|25|*alighting*|L1|0|$\infty$|
|26|*connector*||0|$\infty$|

## Transit graph in AequilibraE

A few more edges types have been introduced in AequilibraE. Mainly we differentiate the connectors directed from the demand to the supply (*access connectors*) from the ones in the opposite direction (*egress connectors*). Also, we differentiate the transfer edges connecting lines within the same stop (*inner transfer*) from the ones connecting lines between distinct stops from the same station (*outer transfer*).
- on-board
- boarding
- alighting
- dwell
- access_connector
- egress_connector
- inner_transfer
- outer_transfer
- walking

If we buit the graph for the city of Lypn France (GTFS files from 2022), we get 20196 vertices and 91107 edges. Here is the distribution of edge types:

| Edge type        |   Count |
|:-----------------|--------:|
| outer_transfer   |   27287 |
| inner_transfer   |   10721 |
| walking          |    9140 |
| on-board         |    7590 |
| boarding         |    7590 |
| alighting        |    7590 |
| dwell            |    7231 |
| access_connector |    6979 |
| egress_connector |    6979 |


and vertex types:

| Vertex type   |   Count |
|:------------|--------:|
| alighting   |    7590 |
| boarding    |    7590 |
| stop        |    4499 |
| od          |     517 |


## References 

[1] Heinz Spiess, Michael Florian, *Optimal strategies: A new assignment model for transit networks,* Transportation Research Part B: Methodological, Volume 23, Issue 2, 1989, Pages 83-102, ISSN 0191-2615, https://doi.org/10.1016/0191-2615(89)90034-9.


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