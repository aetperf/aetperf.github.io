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
- environment variable 
- JSON file  

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


## Postgres credentials

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

## Loading the dataset into Postgres

To efficiently load the provided dataset into the Postgres database, the following code sections illustrate each step of the process:

- Dropping existing table, if exists:

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

This step involves creating the table `wikipedia_articles` with various columns: `id`, `url`, `title`, `text`, `title_vector`, `content_vector`, and `vector_id`. The column types are defined, including the `vector` columns with 1536 dimensions.


```python
sql = "CREATE TABLE wikipedia_articles (id int PRIMARY KEY, url varchar(1000), title varchar(1000), text varchar, title_vector vector(1536), content_vector vector(1536), vector_id int);"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

- Loading the Dataset:

Here, the dataset is loaded from the provided CSV file into the `wikipedia_articles` table using the `COPY` command. This is done in a batched manner to efficiently process and insert the data.


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


## Querying with text input

In this section, we provide a set of functions that allow you to perform a similarity search based on text input, enabling you to find relevant articles from the dataset that are similar to the provided input.

The following function takes an embedding `emb` and performs a similarity search using a Common Table Expression. It calculates the similarity between the provided embedding and the content vectors of articles in the dataset. The articles are ordered by ascending similarity and limited to a specified count `match_count`. The function returns a DataFrame containing the article IDs, titles, and their similarity scores.

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

This higher-level function `similarity_search` combines the embedding generation and similarity search steps. It takes a text input, generates an embedding for that input using the `get_embedding` function, and then uses the `similarity_search_from_emb` function to perform the similarity search. The matching articles with similarity scores above a specified threshold `match_threshold` are returned in a DataFrame.

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

## LangChain vectorstore PGVector integration

Unfortunatly we cannot query the previous `wikipedia_articles` table with LangChain. So in this section, we load the `wikipedia_articles` into the LangChain [`PGVector`](https://python.langchain.com/docs/integrations/vectorstores/pgvector) vectorstore. `PGVector` is LangChain interface with *pgvector*.

The `PGVector` vectorstore creates two tables into Postgres:
- `langchain_pg_collection` listing the different collections
- `langchain_pg_embedding` storing texts, embeddings, metadata and collection name

Below is a step-by-step explanation of the process:

- Connection String Setup:

The `CONNECTION_STRING` is generated using the Postgres database credentials provided:

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

- Creating the PGVector Store:

We create a `PGVector` store named `wikipedia_articles` using the connection string and an instance of the `OpenAIEmbeddings` class to generate embeddings. The `pre_delete_collection` flag indicates that any existing collection with the same name should be deleted before creating the new collection:

```python
embeddings = OpenAIEmbeddings(openai_api_key=openai.api_key)
store = PGVector(
    collection_name="wikipedia_articles",
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
    pre_delete_collection=True,
)
```

- Fetching Data from PostgreSQL:

Data from the `wikipedia_articles` table is fetched into a Pandas DataFrame using SQL queries. This data will be used to load embeddings into the `PGVector` collection. This step in not so efficient, since we are going to fetch all the data from the `wikipedia_articles` in memory...

```python
%%time
sql = f"SELECT id, url, title, text, content_vector, vector_id FROM wikipedia_articles"
df = pd.read_sql(sql=sql, con=conn)
```

    CPU times: user 251 ms, sys: 194 ms, total: 444 ms
    Wall time: 1.33 s

- Converting Embedding Strings to Lists:

The embedding vectors in the DataFrame are stored as strings. We convert these strings to lists of floats using the `ast.literal_eval` function, enabling compatibility with the `PGVector` store:


```python
%%time
df.content_vector = df.content_vector.map(ast.literal_eval)
```

    CPU times: user 1min 11s, sys: 236 ms, total: 1min 11s
    Wall time: 1min 11s


- Loading Data into PGVector Collection:

The embeddings, texts, metadata, and IDs are loaded into the `PGVector` collection using the `add_embeddings` method. This step makes the dataset's embeddings available for similarity search:

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

Also we can close the psycopg2 now, since it is no longer needed:

```python
conn.close()
```

- Querying with PGVector Store:

An example query is demonstrated using the `store.similarity_search` method. Given a query "Tell me about Hip Hop?", the method retrieves the `k=3` most similar documents from the `PGVector` collection. In this case, the returned documents are those that have similar content to the query.

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


It's important to note that while the PGVector similarity search retrieves documents with similar content, it may not provide answers to specific questions. To address this, we are going to use a Chat Language Model to build a Q&A bot that leverages the vector store for efficient querying and integrates natural language understanding to answer questions more effectively.

Before building the Q&A bot using a ChatLLM and querying the vectorstore, a fake Wikipedia article is added to ensure that the bot is utilizing the vectorstore and not its own memory.

## Adding a fake wikipedia article

- Fetching maximum IDs:

These values will be used to assign new IDs for the fake article:

```python
article_id = df["id"].max() + 1
vector_id = df["vector_id"].max() + 1
```

- Creating Fake Article Data:

A fake Wikipedia article named "François Pacull" is created with a fictitious biography. The article's title, URL, text, and content are defined, and vector embeddings are generated for the title and content using the get_embedding function.


```python
url = "https://simple.wikipedia.org/wiki/FrancoisPacull"
title = "François Pacull"
text = """François Pacull

François Pacull is a Python developer, known for his unbridled passion for pizza. Born on an undisclosed date in a parallel universe, Pacull's journey from a curious coder to a pizza-loving programmer has left an indelible mark on both the digital and culinary realms.

Early Life and Education

Details about François Pacull's early life remain shrouded in mystery, much like the inner workings of a black box algorithm. It is said that he demonstrated an uncanny knack for deciphering complex problems from a young age, often using pizza slices as visual aids in his learning process.
"""
title_vector = get_embedding(title)
content_vector = get_embedding(text)

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

- Adding the record to the `PGVector` collection:

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

The process of adding this fake article demonstrates how to incorporate additional data into the `PGVector` collection. Let's create the Q&A bot.

## Documents Q&A bot example with LangChain

The following code demonstrates an example of using the LangChain framework to build a Question-Answering (QA) bot that retrieves answers from documents stored in the `PGVector` collection. Here's how the example works:

- Creating the Retriever:

The `store` object is used to create a `retriever` using the `as_retriever` method:

```python
retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
```

The default similarity metric is cosine. We retrive the 3 most similar entries for each query.

- Creating the QA Model:

The `RetrievalQA` class is instantiated using the OpenAI's *gpt-3.5-turbo* chat model, the previously created `retriever`, and `return_source_documents` set to `True`. This configuration enables the bot to return the source documents that contributed to its answer:


```python
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai.api_key, model_name="gpt-3.5-turbo", temperature=0),
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True,
)
```

- Querying the QA Bot:

Queries are posed to the QA bot using the `qa` instance. For each query, the bot generates an answer and returns the result along with the source documents that were used to derive the answer:

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

Let's try another question:


```python
query = "What are the three Laws of Robotics"
answer = qa({"query": query})
print(answer["result"])
```

    The Three Laws of Robotics are as follows:
    
    1. A robot may not injure a human being or, by failing to act, allow a human being to come to harm.
    
    2. A robot must obey orders given to it by human beings, except where carrying out those orders would conflict with the First Law.
    
    3. A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.


Finally, let's ask a question about something that cannot be found outside the vectorstore:


```python
query = "Who is François Pacull ?"
answer = qa({"query": query})
print(answer["result"])
```

    François Pacull is a Python developer known for his passion for pizza. However, beyond this information, there is not much else known about François Pacull.


The example showcases how the LangChain-based QA bot can retrieve answers from the `PGVector` collection based on queries. The bot provides accurate answers along with the relevant source documents, making it a useful tool for extracting information from the stored documents.


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