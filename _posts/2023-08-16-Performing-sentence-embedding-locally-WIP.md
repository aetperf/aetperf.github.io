---
title: Performing sentence embedding locally WIP
layout: post
comments: true
author: François Pacull
tags: 
- Python
- sentence embedding
- sentence_transformers
---


What is sentence embedding? Sentence embedding refers to the process of representing a sentence as a fixed-length numerical vector in a high-dimensional space. The goal of sentence embedding is to capture the semantic and contextual information of a sentence in a way that allows for various natural language processing (NLP) tasks to be performed more effectively. Sentence embedding techniques map the variable-length sentence input into a continuous vector space where semantically similar sentences are represented by vectors that are close to each other. 

The simple use case in the following is computing similarity between a few sentences. The similarity between two vectors measures their relatedness.

We are going to use a recent text embedding model provided by the Beijing Academy of Artificial Intelligence (BAAI): [`BAAI/bge-base-en`](https://huggingface.co/BAAI/bge-base-en). BGE is short for *BAAI General Embedding*. This model has an embedding dimension of 768 and an input sequence length of 512 tokens. This means that longer sequences will be truncated. As a comparison, OpenAi's embedding model [`text-embedding-ada-002`](https://platform.openai.com/docs/guides/embeddings/second-generation-models) has an embedding dimension of 8191 and an input sequence length of 1536 tokens. However, the latter model cannot be run locally. For the reference, here is a table comparing some of the features of both embedding models:

| model | can run locally | multilingual | input sequence length | embedding dimension |
|---|---:|---:|---:|---:|
| `BAAI/bge-base-en`        | yes |  no |  512 |  768 |
| `text-embedding-ada-002`  |  no | yes | 1536 | 8191 |

Note that we could have used the larger instance of the BAAI/bge model, `BAAI/bge-large-en`, however we wanted to use a relatively small model, 0.44GB, that can run fast on a regular machine. Despite its small size, `BAAI/bge-base-en` does rank well on the Massive Text Embedding Benchmark leaderboard (MTEB): [https://huggingface.co/spaces/mteb/leaderboard](https://huggingface.co/spaces/mteb/leaderboard), 2nd rank at the time of writing this post.

We are going to run this embedding model using the great Python library: [sentence_transformers](https://www.sbert.net/). We could also have used the other Python libraries giving access to this model: [FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding/tree/master) or [LangChain](https://python.langchain.com/docs/integrations/text_embedding/bge_huggingface) through Hugging Face. It is also possible to use Hugging face's library [transformers](https://github.com/huggingface/transformers) along PyTorch, but it is less straightforward.

Let's start with the imports.

## Imports

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sentence_transformers import SentenceTransformer
```

System information and package versions:

    Python version       : 3.11.4
    OS                   : Linux
    Machine              : x86_64
    matplotlib           : 3.7.1
    numpy                : 1.24.4
    pandas               : 2.0.3
    seaborn              : 0.12.2
    sentence_transformers: 2.2.2

## Download and load the embedding model

```python
model = SentenceTransformer('BAAI/bge-base-en')
```

The very first time this model is instanciated as above, a bunch of files are downloaded:

<p align="center">
  <img width="1000" src="/img/2023-08-16_01/Selection_103.png" alt="Selection_103">
</p>

The model artifact is cached is some hidden home directory:

```bash
$ tree .cache/torch/sentence_transformers/BAAI_bge-base-en
.cache/torch/sentence_transformers/BAAI_bge-base-en
├── 1_Pooling
│   └── config.json
├── config.json
├── config_sentence_transformers.json
├── model.safetensors
├── modules.json
├── pytorch_model.bin
├── README.md
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt

1 directory, 12 files
```

We are now ready to encode our first sentence:

```python
emb = model.encode("It’s not an easy thing to meet your maker", normalize_embeddings=True)
emb[:5]
```

    array([-0.01139987,  0.00527102, -0.00226131, -0.01054202,  0.04873622],
          dtype=float32)

We can check the embedding dimension and that the resulting vector has been normalized:

```python
emb.shape
```


    (768,)





```python
np.linalg.norm(emb, ord=2)
```


    0.99999994



## Compute some embeddings

We use a little helper function that could be enriched to clean the input text. In the present version we only remove carriage return and line feed.

```python
def get_embedding(text, normalize=True):
    text = text.replace("\r", " ").replace("\n", " ")
    return model.encode(text, normalize_embeddings=normalize)
```


```python
embs = []
```


```python
# first sentence
sen_1 = "Berlin is the capital of Germany"
emb_1 = get_embedding(sen_1)
embs.append((sen_1, emb_1))
```


```python
emb_1[:5]
```


    array([-0.02016803, -0.00276157,  0.03377605,  0.00311152,  0.02026351],
          dtype=float32)


```python
emb_1.shape
```

    (768,)


```python
# second sentence 
sen_2 = "Yaoundé is the capital of Cameroon"
emb_2 = get_embedding(sen_2)
embs.append((sen_2, emb_2))
```

### cosine similarity $S$

We can check that these two previous sentences are related by computing their cosine similarity:

$$S(u,v) = cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$


```python
def cosine_similarity(vec1, vec2, normalized=True):
    if normalized:
        return np.dot(vec1, vec2)
    else:
        return np.dot(vec1, vec2) / (
            np.linalg.norm(vec1, ord=2) * np.linalg.norm(vec2, ord=2)
        )
```

The value of "S" is bounded between -1 and 1: $S(u,u)=1$ and $S(u,-u)=S(-u,u)=-1$:

```python
cosine_similarity(emb_1, emb_1)
```

    1.0000001

```python
cosine_similarity(-emb_1, emb_1)
```

    -1.0000001

A zero value indicates that the two vectors are orthogonal:

```python
cosine_similarity(emb_1, emb_2 - np.dot(emb_1, emb_2) * emb_1)
```

    -4.4703484e-08

So let's compute the similarity the two sentences dealing with capitals:


```python
cosine_similarity(emb_1, emb_2)
```


    0.8196905

## Cosine similarity heatmap

Now we compute 3 more embeddings and a 5-by-5 similarity matrix:

```python
sen_3 = "Esplanade MRT station is an underground Mass Rapid Transit station on the Circle line in Singapore"
emb_3 = get_embedding(sen_3)
embs.append((sen_3, emb_3))
sen_4 = "Remove jalapeños from surface and stir in the additional chili powder, ground cumin and onion powder."
emb_4 = get_embedding(sen_4)
embs.append((sen_4, emb_4))
sen_5 = "Fideua is a traditional dish, similar in style to paella but made with short spaghetti-like pasta."
emb_5 = get_embedding(sen_5)
embs.append((sen_5, emb_5))

def compute_cosine_similarity_matrix(embs, label_size=30):
    l = len(embs)
    cds = np.zeros((l, l), dtype=np.float64)
    for i, (_, emb) in enumerate(embs):
        cds[i, i] = 0.5
        for j in range(i + 1, l):
            cs = cosine_similarity(emb, embs[j][1], normalized=True)
            cds[i, j] = cs
    cds += np.transpose(cds)
    labels = [t[0][:label_size] + "..." for t in embs]
    df = pd.DataFrame(data=cds, index=labels, columns=labels)
    return df


df = compute_cosine_similarity_matrix(embs)
df
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
      <th>Berlin is the capital of Germa...</th>
      <th>Yaoundé is the capital of Came...</th>
      <th>Esplanade MRT station is an un...</th>
      <th>Remove jalapeños from surface ...</th>
      <th>Fideua is a traditional dish, ...</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Berlin is the capital of Germa...</th>
      <td>1.000000</td>
      <td>0.819691</td>
      <td>0.711520</td>
      <td>0.685216</td>
      <td>0.701959</td>
    </tr>
    <tr>
      <th>Yaoundé is the capital of Came...</th>
      <td>0.819691</td>
      <td>1.000000</td>
      <td>0.702962</td>
      <td>0.673310</td>
      <td>0.714926</td>
    </tr>
    <tr>
      <th>Esplanade MRT station is an un...</th>
      <td>0.711520</td>
      <td>0.702962</td>
      <td>1.000000</td>
      <td>0.655668</td>
      <td>0.625339</td>
    </tr>
    <tr>
      <th>Remove jalapeños from surface ...</th>
      <td>0.685216</td>
      <td>0.673310</td>
      <td>0.655668</td>
      <td>1.000000</td>
      <td>0.757284</td>
    </tr>
    <tr>
      <th>Fideua is a traditional dish, ...</th>
      <td>0.701959</td>
      <td>0.714926</td>
      <td>0.625339</td>
      <td>0.757284</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>


Finally we plot the heatmap associated with the similarity matrix:


```python
fig, ax = plt.subplots(figsize=(6, 5))
ax = sns.heatmap(df, cmap="YlGnBu")
ax.xaxis.tick_top()
for label in ax.get_xticklabels():
    label.set_rotation(90)
```

    

<p align="center">
  <img width="1000" src="/img/2023-08-16_01/output_24_0.png" alt="output_24_0">
</p>