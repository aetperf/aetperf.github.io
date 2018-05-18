---
---
title: Lyon DataVis and AI mini-conference
layout: post
author: François Pacull
tags: Conference DataViz DeepLearning AI
---

Yesterday I went to [this](http://data.em-lyon.com/2018/04/26/mini-conference-on-data-visualization-and-ai/) mini-conference at ENS Lyon and enjoyed it very much. Here is a very short description of the speakers/subjects:

## [Lane Harrison](https://web.cs.wpi.edu/~ltharrison/)

Lane is studying how people interact with data visualization: how do they explore a complex visualization, where do they click, how many times... how to improve their understanding of complex data with visualization tools such as a simple search functionality, and more generally user-centered tools.

## [Yannick Assogba](http://clome.info/)

This was really interesting to me. He is one of the developer of [TensorFlow.js](https://js.tensorflow.org/). He mentioned several things such as:

  * [TensorFlow.js implementation of PoseNet](https://github.com/tensorflow/tfjs-models/tree/master/posenet): PoseNet is based on a convolutional neural network and allow to detect a person in a video in real time with an accurate prediction of the body position. You can try a live demo [here](https://storage.googleapis.com/tfjs-models/demos/posenet/camera.html).

 ![](/img/posenet.jpg)

  * [FACETS](https://pair-code.github.io/facets/): Facets is a visualization tool to aid in understanding and analyzing machine learning datasets.

  * [tensorboard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard): a suite of web applications for inspecting and understanding your TensorFlow runs and graphs.

## [Jim Vallandingham](http://vallandingham.me/)

Great presentation about the [quick draw game](https://quickdraw.withgoogle.com/#) dataset, which can be found [here](https://github.com/googlecreativelab/quickdraw-dataset): 50 million drawings across 345 categories.

![](/img/qdgame.jpg)

An old version of his slides can be found [here](http://vallandingham.me/quickdraw_talk.html) on his website.

## [Gabriel Florit](https://gabrielflorit.github.io/)

He is a graphics reporter on the investigative team at the Washington Post. He talked about many cool things: augmented reality, his project [SCRIPT-8](https://script-8.github.io/): a JavaScript fantasy console ("Make pixel-art games with a livecoding browser editor"), ...

## [Arvind Satyanarayan](http://arvindsatya.com/)

He is one of authors of [Vega](https://vega.github.io/vega/), which is a visualization grammar: a declarative language for creating, saving (in JSON), and sharing interactive visualization designs, and [Vega-Lite](https://vega.github.io/vega-lite/). Here is a little example of an interactive moving average in Vega-Lite 2:

![](/img/vega_lite_01.gif)

You can check the corresponding JSON file and play with this interactive plot  [here](https://vega.github.io/editor/#/examples/vega-lite/selection_layer_bar_month). This Vega-Lite visualization grammar is very powerful, and is also used in the Python [Altair](https://altair-viz.github.io/) visualization library, created by [Jake Vanderplas](http://www.vanderplas.com).

Here is a quote from the Vega website page describing the project:

> Over years working in data visualization, we’ve sought to build tools that help designers craft sophisticated graphics, including systems such as Prefuse, Protovis and D3.js. However, in the grand scheme of things, “artisanal” visualizations hand-coded by skilled designers are the exception, not the rule. The vast majority of the world’s visualizations instead are produced using end-user applications such as spreadsheets and business intelligence tools. While valuable, these tools often fall short of fully supporting the iterative, interactive process of data analysis. Improved tools could help a larger swath of people create effective visualizations and better understand their data.

He was here to talk about deep learning and his work on [The Building Blocks of Interpretability](https://distill.pub/2018/building-blocks/). This was fun!

![](/img/mixed4d.jpeg)

Note that this last research work is presented on the [Distill](https://distill.pub/) platform, which is a very powerful web medium to present ML top research.
