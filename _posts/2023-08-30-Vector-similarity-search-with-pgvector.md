---
title: Vector similarity search with pgvector
layout: post
comments: true
author: François Pacull
tags: 
- Python
- vector database
- pgvector
- Postgres
- SQL
- sentence embedding
- text-embedding-ada-002
- openai
- cosine similarity
- langchain
- pandas
- question answering over documents
---


In the realm of vector databases, [*pgvector*](https://github.com/pgvector/pgvector) emerges as a noteworthy open-source extension tailored for Postgres databases. This extension equips Postgres with the capability to efficiently perform vector similarity searches, a powerful technique with applications ranging from recommendation systems to semantic search.

To illustrate the practical implementation of *pgvector*, we'll delve into a specific use case involving the "Simple English Wikipedia" dataset. This dataset, available [here](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip) from OpenAI, contains vector embeddings for Wikipedia articles.

We'll guide you through the process of setting up and utilizing *pgvector* for vector similarity search within a Postgres database. The blog post covers essential steps, including table creation, loading the dataset, and performing queries to find nearest neighbors based on cosine similarity. Additionally, we'll demonstrate how to integrate the Langchain vectorstore [*PGVector*](https://python.langchain.com/docs/integrations/vectorstores/pgvector) to streamline embedding storage and retrieval. We will finally perform question answering over the documents stored in Postgres with [LangChain](https://python.langchain.com/docs/get_started/introduction.html).

## The *Simple English Wikipedia* dataset

The *Simple English Wikipedia dataset* is a substantial resource provided by OpenAI. This dataset is accessible through [this link](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip
) and weighs approximately 700MB when compressed, expanding to around 1.7GB when in CSV format. 

The dataset comprises a collection of Wikipedia articles, each equipped with associated vector embeddings. The embeddings are constructed using OpenAI's [*text-embedding-ada-002*](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) model, yielding vectors with 1536 elements.

The dataset's columns of significance include `content_vector` and `title_vector`, representing the vector embeddings for the article's title and content. 

```python
!head -n 1 /tmp/vector_database_wikipedia_articles_embedded.csv
```
    id,url,title,text,title_vector,content_vector,vector_id

7 columns

Let’s get started by importing the necessary libraries.

## Imports


```python
import ast
import json
import os
import warnings
from time import perf_counter

import numpy as np
import openai
import pandas as pd
import psycopg2
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector

openai_json_fp = "./openai.json"
postgres_json_fp = "./postgres.json"

# the dataset file is located in the /tmp dir
# postgres must have read access on all directories above where the file is located
dataset_fp = "/tmp/vector_database_wikipedia_articles_embedded.csv"  
```

System information and package versions:
    
    Python implementation: CPython
    Python version       : 3.11.5
    OS                   : Linux
    Machine              : x86_64
    numpy                : 1.25.2
    openai               : 0.27.8
    pandas               : 2.0.3
    psycopg2             : 2.9.6
    langchain            : 0.0.271
    PostgreSQL           : 15.4
    pgvector             : 0.4.4 
    

Before delving into the practical aspects, it's imperative to configure the OpenAI API key. This ensures seamless interaction with OpenAI services for embedding generation and more. 

## OpenAI API key

There are two approaches to handle this key in the following:
- Environment Variable 
- JSON File  

```python
if os.getenv("OPENAI_API_KEY") is not None:
    print("OPENAI_API_KEY is ready")
    openai.api_key = os.getenv("OPENAI_API_KEY")
elif os.path.exists(openai_json_fp):
    with open(openai_json_fp, "r") as j:
        content = json.loads(j.read())
    openai.api_key = content["api_key"]
else:
    raise RuntimeError("OpenAI API key not found")
```

Now we test the API with the `get_embedding` function, which utilizes OpenAI's API to generate a semantic embedding for the input text using the specified model. The function returns either a list or an array representation of the embedding, as shown by the example output:

```python
def get_embedding(text, model="text-embedding-ada-002", out_type="array"):
    text = text.replace("\r", " ").replace("\n", " ")
    embedding = openai.Embedding.create(input=[text], model=model)["data"][0][
        "embedding"
    ]
    if out_type == "list":
        return embedding
    elif out_type == "array":
        return np.array(embedding)
```

```python
get_embedding("Yeah man!!")
```

    array([ 0.00981273, -0.00974305,  0.0144562 , ..., -0.00974305,
           -0.00406699, -0.02893774])


## Postgres Credentials

```python
if os.path.exists(postgres_json_fp):
    with open(postgres_json_fp, "r") as j:
        pg_credentials = json.loads(j.read())
else:
    raise RuntimeError("Postgres credentials not found")
```
Once the Postgres credentials are acquired from a JSON file, the following step involves establishing a connection to the database with the `psycopg2` package:

```python
conn = psycopg2.connect(**pg_credentials)
```

## Installing *pgvector*

Here are the official installation notes: [https://github.com/pgvector/pgvector#installation-notes](https://github.com/pgvector/pgvector#installation-notes). The process of installing *pgvector* is relatively straightforward on Linux systems:

```bash
cd /tmp
git clone --branch v0.4.4 https://github.com/pgvector/pgvector.git
cd pgvector
make
make install
```

We also had to specify the path to `pg_config` before the installation. 

## Loading the Dataset into Postgres

To efficiently load the provided dataset into your Postgres database, the following code sections illustrate each step of the process:

- Dropping Existing Table (if exists):

```python
sql = "DROP TABLE IF EXISTS wikipedia_articles;"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

- Enabling the *pgvector* Extension:

```python
sql = "CREATE EXTENSION IF NOT EXISTS vector;"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

```python
sql = "SELECT extname, extversion FROM pg_extension WHERE extname='vector';"
pd.read_sql(sql=sql, con=conn, index_col="extname")
```


| extname   | extversion   |
|----------:|-------------:|
| vector    | 0.4.4        |


- Creating the Table:


```python
sql = "CREATE TABLE wikipedia_articles (id int PRIMARY KEY, url varchar(1000), title varchar(1000), text varchar, title_vector vector(1536), content_vector vector(1536), vector_id int);"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

This step involves creating the table `wikipedia_articles` with various columns: `id`, `url`, `title`, `text`, `title_vector`, `content_vector`, and `vector_id`. The column types are defined, including the `vector` columns with 1536 dimensions.

- Loading the Dataset:


```python
%%time
sql = f"""COPY wikipedia_articles(id,url,title,text,title_vector,content_vector,vector_id)
FROM '{dataset_fp}'
DELIMITER ','
CSV HEADER;
"""
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

    CPU times: user 1.86 ms, sys: 0 ns, total: 1.86 ms
    Wall time: 13.9 s


Here, the dataset is loaded from the provided CSV file (`dataset_fp`) into the `wikipedia_articles` table using the `COPY` command. This is done in a batched manner to efficiently process and insert the data.

- Verifying the number of rows:


```python
sql = "SELECT COUNT(*) FROM wikipedia_articles"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        n_rows = cur.fetchone()[0]
        print(f"n_rows = {n_rows}")
```

    n_rows = 25000

- Displaying the first 3 rows of the table:

```python
n = 3
sql = f"SELECT * FROM wikipedia_articles ORDER BY id ASC LIMIT {n}"
df = pd.read_sql(sql=sql, con=conn)
df.head(n)
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
      <th>id</th>
      <th>url</th>
      <th>title</th>
      <th>text</th>
      <th>...</th>
      <th>vector_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>https://simple.wikipedia.org/wiki/April</td>
      <td>April</td>
      <td>April is the fourth month of the year in the J...</td>
      <td>...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://simple.wikipedia.org/wiki/August</td>
      <td>August</td>
      <td>August (Aug.) is the eighth month of the year ...</td>
      <td>...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>https://simple.wikipedia.org/wiki/Art</td>
      <td>Art</td>
      <td>Art is a creative activity that expresses imag...</td>
      <td>...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


## First query on the vector table

We look for the nearest neighbors to the first article of the table, comparing article contents using cosine similarity $S$:

$$S(u,v) = cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

Note that we have $S(u,v)=u \cdot v$ in case of normalized vectors. In SQL we are going to use the `<#>` operator, that returns the negative inner product. From *pgvector*'s [documentation](https://github.com/pgvector/pgvector#installation-notes):  
> <#> returns the negative inner product since Postgres only supports ASC order index scans on operators


```python
%%time
sql = """SELECT b.title , (a.content_vector <#> b.content_vector) * -1 as similarity
FROM wikipedia_articles a cross join wikipedia_articles b 
WHERE b.id != 1 and a.id=1
ORDER BY similarity desc 
LIMIT 5;"""
pd.read_sql(sql=sql, con=conn)
```

    CPU times: user 703 µs, sys: 987 µs, total: 1.69 ms
    Wall time: 172 ms




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
      <th>title</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>May</td>
      <td>0.924446</td>
    </tr>
    <tr>
      <th>1</th>
      <td>March</td>
      <td>0.913814</td>
    </tr>
    <tr>
      <th>2</th>
      <td>January</td>
      <td>0.902376</td>
    </tr>
    <tr>
      <th>3</th>
      <td>February</td>
      <td>0.900264</td>
    </tr>
    <tr>
      <th>4</th>
      <td>July</td>
      <td>0.885249</td>
    </tr>
  </tbody>
</table>
</div>

The first article of the table is about the month of April. We can see that similar articles in the table are also about months: May, March, ...


## Querying with Text Input

In this section, we provide a set of functions that allow you to perform a similarity search based on text input, enabling you to find relevant articles from the dataset that are similar to the provided input. Here's a breakdown of the components:

The following function takes an embedding (`emb`) and performs a similarity search using a Common Table Expression (CTE). It calculates the similarity between the provided embedding and the content vectors of articles in the dataset. The articles are ordered by ascending similarity and limited to a specified count (`match_count`). The function returns a DataFrame containing the article IDs, titles, and their similarity scores.

```python
def similarity_search_from emb(emb, conn, match_threshold=0.75, match_count=10):
    sql = f"""WITH cte AS (SELECT id, title, (content_vector <#> '{emb}') as similarity 
    FROM wikipedia_articles
    ORDER BY similarity asc
    LIMIT {match_count})
    SELECT * FROM cte
    WHERE similarity < -{match_threshold}"""

    df = pd.read_sql(sql=sql, con=conn)
    df.similarity *= -1.0
    return df
```

This higher-level function `similarity_search` combines the embedding generation and similarity search steps. It takes a text input, generates an embedding for that input using the `get_embedding` function, and then uses the `similarity_search_from_emb` function to perform the similarity search. The matching articles with similarity scores above a specified threshold (`match_threshold`) are returned in a DataFrame.

```python
def similarity_search(text, match_threshold=0.75, match_count=10):
    start = perf_counter()
    emb = get_embedding(text, out_type="list")
    end = perf_counter()
    elapsed_time = end - start
    print(f"get embedding : {elapsed_time:5.3f}")

    start = perf_counter()
    df = similarity_search_from_emb(emb, conn, match_threshold=0.75, match_count=match_count)
    end = perf_counter()
    elapsed_time = end - start
    print(f"similarity search : {elapsed_time:5.3f}")

    return df
```

This example demonstrates how to use the functions to find articles related to "The Foundation series by Isaac Asimov" with their corresponding similarity scores:

```python
%%time
similarity_search("The Foundation series by Isaac Asimov")
```

    get embedding : 0.382
    similarity search : 0.120
    CPU times: user 8.86 ms, sys: 1.16 ms, total: 10 ms
    Wall time: 502 ms



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
      <th>id</th>
      <th>title</th>
      <th>similarity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>8698</td>
      <td>Isaac Asimov</td>
      <td>0.861273</td>
    </tr>
    <tr>
      <th>1</th>
      <td>60390</td>
      <td>Three Laws of Robotics</td>
      <td>0.813722</td>
    </tr>
    <tr>
      <th>2</th>
      <td>12399</td>
      <td>The Hitchhiker's Guide to the Galaxy</td>
      <td>0.808487</td>
    </tr>
    <tr>
      <th>3</th>
      <td>17651</td>
      <td>Science fiction</td>
      <td>0.807418</td>
    </tr>
    <tr>
      <th>4</th>
      <td>84847</td>
      <td>Philosophiæ Naturalis Principia Mathematica</td>
      <td>0.806654</td>
    </tr>
    <tr>
      <th>5</th>
      <td>4570</td>
      <td>A Brief History of Time</td>
      <td>0.804930</td>
    </tr>
    <tr>
      <th>6</th>
      <td>55040</td>
      <td>Robert A. Heinlein</td>
      <td>0.802639</td>
    </tr>
    <tr>
      <th>7</th>
      <td>66541</td>
      <td>Artemis Fowl</td>
      <td>0.791295</td>
    </tr>
    <tr>
      <th>8</th>
      <td>56319</td>
      <td>The Dark Tower (series)</td>
      <td>0.791016</td>
    </tr>
    <tr>
      <th>9</th>
      <td>23889</td>
      <td>Stanislav Lem</td>
      <td>0.788935</td>
    </tr>
  </tbody>
</table>
</div>

Now we are going to use the `PGVector` vectorstore from the [LangChain package](https://python.langchain.com/docs/get_started/introduction.html).

# LangChain vectorstore `PGVector`

Unfortunatly we cannot query the previous `wikipedia_articles` table with LangChain. The `PGVector` vectorstore creates two tables into Postgres:
- `langchain_pg_collection` listing the different collections
- `langchain_pg_embedding` storing texts, embeddings, metadata and collection name


```python
CONNECTION_STRING = PGVector.connection_string_from_db_params(
    driver="psycopg2",
    host=pg_credentials.get("host", "localhost"),
    port=int(pg_credentials.get("port", "5432")),
    database=pg_credentials.get("database", "postgres"),
    user=pg_credentials.get("user", "postgres"),
    password=pg_credentials.get("password", "postgres"),
)
```

Create the store:


```python
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
store = PGVector(
    collection_name="wikipedia_articles",
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    pre_delete_collection=True,
)
```

The following step in not so efficient, since we are going to fetch all the data from the `wikipedia_articles` into a Pandas dataframe in order to load it into a the `wikipedia_articles` `PGVector` collection just created:

```python
%%time
sql = f"SELECT id, url, title, text, content_vector, vector_id FROM wikipedia_articles"
df = pd.read_sql(sql=sql, con=conn)
```

    CPU times: user 251 ms, sys: 194 ms, total: 444 ms
    Wall time: 1.33 s


However, the embedding vectors from this dataframe are stored as long strings while LangChain expect a list of floats. Let's transform each string as a list of floats:


```python
%%time
df.content_vector = df.content_vector.map(ast.literal_eval)
```

    CPU times: user 1min 11s, sys: 236 ms, total: 1min 11s
    Wall time: 1min 11s

We can now load the data into the `PGVector` collection:

```python
%%time
_ = store.add_embeddings(
    texts=df.text.values.tolist(),
    embeddings=df.content_vector.values.tolist(),
    metadata=df[["url", "title", "vector_id"]].to_dict("records"),
    ids=df.id.values.tolist(),
)
```

    CPU times: user 9.29 s, sys: 180 ms, total: 9.47 s
    Wall time: 15.4 s

```python
sql = "SELECT * FROM langchain_pg_collection"
pd.read_sql(sql, conn)
```


|    | name               | cmetadata   | uuid                                 |
|---:|:-------------------|:------------|:-------------------------------------|
|  0 | wikipedia_articles |             | 6fb17082-8a07-4c51-907c-5d3712359876 |


```python
sql = "SELECT COUNT(*) FROM langchain_pg_embedding"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        size = cur.fetchone()[0]
print(size)
```

    25000

And we can perform a first query over the vector store:


```python
query = "Tell me about Hip Hop?"

# Fetch the k=3 most similar documents
docs = store.similarity_search(query, k=3)
```


```python
docs
```


    [Document(page_content='Hip hop is a type of culture/art style that started in the 1970s in the Bronx. [...]', metadata={}),
     Document(page_content='Rapping is a type of vocals, like singing. [...]', metadata={}),
     Document(page_content="Breakdance (also called breaking, b-boying or b-girling) is a type of dance that is done by people who are part of the hip hop culture. [...] ", metadata={})]


However, this is just returning articles with similar contents to the question text, but not really answering the question. So let's use a ChatLLM to build a Q&A bot to query the vector store easily using questions. 

Before that, we are going to add an fake entry to the wikipedia articles in order to check that the bot is using the vector store.

## Adding a fake wikipedia article


```python
with conn:
    with conn.cursor() as cur:
        sql = f"SELECT MAX(id) FROM wikipedia_articles"
        cur.execute(sql)
        max_id = cur.fetchone()[0]
        sql = f"SELECT MAX(vector_id) FROM wikipedia_articles"
        cur.execute(sql)
        max_vector_id = cur.fetchone()[0]

article_id = max_id + 1
url = "https://simple.wikipedia.org/wiki/FrancoisPacull"
title = "François Pacull"
text = """François Pacull

François Pacull is a Python developer, known for his unbridled passion for pizza. Born on an undisclosed date in a parallel universe, Pacull's journey from a curious coder to a pizza-loving programmer has left an indelible mark on both the digital and culinary realms.

Early Life and Education

Details about François Pacull's early life remain shrouded in mystery, much like the inner workings of a black box algorithm. It is said that he demonstrated an uncanny knack for deciphering complex problems from a young age, often using pizza slices as visual aids in his learning process.

Pacull's formal education reportedly includes degrees in Mathematic.

Pythonic Pizzas and Pizza-Infused Python

François Pacull's coding prowess was not confined to the digital realm alone; he ventured into the culinary sphere as well. Armed with his coding skills, he created a Python program that generated pizza recipes based on mathematical parameters. Whether it was a Fibonacci-inspired topping arrangement or a Pi-themed crust, François Pacull's pizzas transcended taste to become mathematical masterpieces.

Conversely, he harnessed his love for pizza to inspire Python projects. The "PizzaSort" algorithm, for instance, sorted pizza slices based on their toppings' complexity, much like sorting elements in an array."""
title_vector = get_embedding(title)
content_vector = get_embedding(text)
vector_id = max_vector_id + 1
df_1 = pd.DataFrame(
    data={
        "id": [article_id],
        "url": [url],
        "title": [title],
        "text": [text],
        "content_vector": [content_vector],
        "vector_id": [vector_id],
    }
)
```


```python
%%time
_ = store.add_embeddings(
    texts=df_1.text.values.tolist(),
    embeddings=df_1.content_vector.values.tolist(),
    metadata=df_1[["url", "title", "vector_id"]].to_dict("records"),
    ids=df_1.id.values.tolist(),
)
```

    CPU times: user 2.18 ms, sys: 3.84 ms, total: 6.02 ms
    Wall time: 9.55 ms


## Documents Q&A bot example with LangChain


```python
retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
```


```python
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai.api_key, temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
```


```python
query = "When was written the Foundations series by Isaac Asimov"
answer = qa({"query": query})
print(answer["result"])
```

    The Foundation series by Isaac Asimov was written between 1942 and 1993. The first book in the series, "Foundation," was published in 1951, and the final book, "Forward the Foundation," was published posthumously in 1993.



```python
len(answer["source_documents"])
```




    3




```python
query = "What are the three Laws of Robotics"
answer = qa({"query": query})
print(answer["result"])
```

    The Three Laws of Robotics are as follows:
    
    1. A robot may not injure a human being or, by failing to act, allow a human being to come to harm.
    
    2. A robot must obey orders given to it by human beings, except where carrying out those orders would conflict with the First Law.
    
    3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.



```python
query = "Who is François Pacull ?"
answer = qa({"query": query})
print(answer["result"])
```

    François Pacull is a Python developer known for his passion for pizza. He is also known for creating a Python program that generates pizza recipes based on mathematical parameters. However, beyond this information, there is not much else known about François Pacull.



```python
conn.close()
```


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