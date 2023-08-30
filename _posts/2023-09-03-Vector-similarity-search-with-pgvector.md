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

In the realm of vector databases, [pgvector](https://github.com/pgvector/pgvector) emerges as a noteworthy open-source extension tailored for Postgres databases. This extension equips Postgres with the capability to efficiently perform vector similarity searches, a powerful technique with applications ranging from recommendation systems to semantic search.

To illustrate the practical implementation of Pgvector, we'll delve into a specific use case involving the "Simple English Wikipedia" dataset. This dataset, [available from OpenAI](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip), contains vector embeddings for Wikipedia articles.

We'll guide you through the process of setting up and utilizing Pgvector for vector similarity search within a Postgres database. The blog post covers essential steps, including table creation, loading the dataset, and performing queries to find nearest neighbors based on cosine similarity. Additionally, we'll demonstrate how to integrate the [Langchain vectorstore PGVector](https://python.langchain.com/docs/integrations/vectorstores/pgvector) to streamline embedding storage and retrieval. We will finally perform question answering over documents.


# The Simple English Wikipedia dataset

The *Simple English Wikipedia dataset* is a substantial resource provided by OpenAI. This dataset is accessible through [this link](https://cdn.openai.com/API/examples/data/vector_database_wikipedia_articles_embedded.zip
) and weighs approximately 700MB when compressed, expanding to around 1.7GB when in CSV format. 

The dataset comprises a collection of Wikipedia articles, each equipped with associated vector embeddings. The embeddings are constructed using [OpenAI's *text-embedding-ada-002*](https://platform.openai.com/docs/guides/embeddings/what-are-embeddings) model, yielding vectors with 1536 elements or dimensions.

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

# the dataset file is licated in the tmp dir
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
Once the Postgres credentials are acquired fro a JSON file, the following step involves establishing a connection to the database with the `psycopg2` package:

```python
conn = psycopg2.connect(**pg_credentials)
```

## Load the dataset into Postgres


```python
sql = "DROP TABLE IF EXISTS wikipedia_articles;"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

Enable the extension:


```python
sql = "CREATE EXTENSION IF NOT EXISTS vector;"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

Create a vector column with 7 dimensions:


```python
sql = "CREATE TABLE wikipedia_articles (id int PRIMARY KEY, url varchar(1000), title varchar(1000), text varchar, title_vector vector(1536), content_vector vector(1536), vector_id int);"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        conn.commit()
```

Load the dataset (from the OpenAI website):


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



```python
sql = "SELECT COUNT(*) FROM wikipedia_articles"
with conn:
    with conn.cursor() as cur:
        cur.execute(sql)
        n_rows = cur.fetchone()[0]
        print(f"n_rows = {n_rows}")
```

    n_rows = 25000



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
      <th>title_vector</th>
      <th>content_vector</th>
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
      <td>[0.0010094646,-0.020700546,-0.012527447,-0.043...</td>
      <td>[-0.011253941,-0.013491976,-0.016845843,-0.039...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://simple.wikipedia.org/wiki/August</td>
      <td>August</td>
      <td>August (Aug.) is the eighth month of the year ...</td>
      <td>[0.0009286514,0.000820168,-0.0042785504,-0.015...</td>
      <td>[0.00036099547,0.007262262,0.0018810921,-0.027...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>https://simple.wikipedia.org/wiki/Art</td>
      <td>Art</td>
      <td>Art is a creative activity that expresses imag...</td>
      <td>[0.0033937139,0.0061537535,0.011738217,-0.0080...</td>
      <td>[-0.0049596895,0.015772194,0.006536909,-0.0027...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



## First query on the vector table

Nearest neighbor to the first article content, using cosine similarity $S$.

$$S(u,v) = cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$$

Note that we have $S(u,v)=u \cdot v$ in case of normalized vectors.

In SQL we are going to use the `<#>` operator, that returns the negative inner product.

From pgvector's [documentation](https://github.com/pgvector/pgvector#installation-notes):  
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



## Second query with text input


```python
def similarity_search(emb, conn, match_threshold=0.75, match_count=10):
    """WITH a command table expression"""
    sql = f"""WITH cte AS (SELECT id, title, (content_vector <#> '{emb}') as similarity 
    FROM wikipedia_articles
    ORDER BY similarity asc
    LIMIT {match_count})
    SELECT * FROM cte
    WHERE similarity < -{match_threshold}"""

    df = pd.read_sql(sql=sql, con=conn)
    df.similarity *= -1.0
    return df


def simple_search(text, match_threshold=0.75, match_count=10):
    start = perf_counter()
    emb = get_embedding(text, out_type="list")
    end = perf_counter()
    elapsed_time = end - start
    print(f"get embedding : {elapsed_time:5.3f}")

    start = perf_counter()
    df = similarity_search(emb, conn, match_threshold=0.75, match_count=match_count)
    end = perf_counter()
    elapsed_time = end - start
    print(f"similarity search : {elapsed_time:5.3f}")

    return df
```


```python
%%time
simple_search("The Foundation series by Isaac Asimov")
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



# Langchain vectorstore PGVector


```python

```


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

Fetch the data into a dataframe:


```python
%%time
sql = f"SELECT id, url, title, text, content_vector, vector_id FROM wikipedia_articles"
df = pd.read_sql(sql=sql, con=conn)
df.head(3)
```

    CPU times: user 251 ms, sys: 194 ms, total: 444 ms
    Wall time: 1.33 s





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
      <th>content_vector</th>
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
      <td>[-0.011253941,-0.013491976,-0.016845843,-0.039...</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>https://simple.wikipedia.org/wiki/August</td>
      <td>August</td>
      <td>August (Aug.) is the eighth month of the year ...</td>
      <td>[0.00036099547,0.007262262,0.0018810921,-0.027...</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6</td>
      <td>https://simple.wikipedia.org/wiki/Art</td>
      <td>Art</td>
      <td>Art is a creative activity that expresses imag...</td>
      <td>[-0.0049596895,0.015772194,0.006536909,-0.0027...</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>



The embedding vectors are in a stores as a list of strings. Let's transform each string as a list of floats:


```python
%%time
df.content_vector = df.content_vector.map(ast.literal_eval)
```

    CPU times: user 1min 11s, sys: 236 ms, total: 1min 11s
    Wall time: 1min 11s



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

https://lancedb.github.io/lancedb/notebooks/code_qa_bot/

https://www.timescale.com/blog/how-to-build-llm-applications-with-pgvector-vector-store-in-langchain/


```python

```

work with an existing vectorstore:


```python
query = "Tell me about Hip Hop?"

# Fetch the k=3 most similar documents
docs = store.similarity_search(query, k=3)
```


```python
docs
```




    [Document(page_content='Hip hop is a type of culture/art style that started in the 1970s in the Bronx. It began in Jamaican American, African American, and Puerto Rican/Hispanic and Latino American urban areas in some of the larger cities of the United States. Hip hop uses drum beats produced by a drum machine, and rapping, where the rapper or group chants or says words with a rhythm that rhymes. The lyrics of hip hop songs are often about the life of urban people in the big cities. Hip hop music also uses musical styles from pop music such as disco and reggae. Rap and hip hop music have become successful music genres.\n\nHip hop as a culture involves the music as well as a style of dressing called "urban" clothes (baggy pants, Timberland leather work boots, and oversize shirts); a dancing style called breakdancing or "B-boying"; and graffiti, a street art in which people paint pictures or words on walls. In the 2000s, hip hop music and hip hop culture are very popular in the United States and Canada. Hip hop musicians usually use nicknames. Many of the popular hip hop musicians from the 2000s use nicknames, such as Snoop Dogg, Jay-Z, Eminem, Lil\' Wayne, and 50 Cent. Hip hop is sometimes fused with other genres such as country music and rock music.\n\nRapping\nRapping is a form of singing. It is a mix between singing and talking. The words are spoken with rhythm and in the text there are rhymes. The urban youth made rhyming games based on rap. The beat in the background is a simple loop that is sometimes made by the rapper themself or sometimes copied from a sample CD. The simple loop carries out through the entire song usually, except for the chorus. It developed in the ethnic minority urban (city) areas, as an American form of Jamaican "toasting" (chanting and rhyming with a microphone).\n\nRun DMC and The Sugarhill Gang were early popular hip hop groups in the 1980s. When rappers began to use violent language and gestures, the music was then liked by gangsters. This kind of music was called "gangsta rap". Gangsta rap often has lyrics which are about guns, drug dealing and life as a thug on the street. This genre also began in the 1980s and is still produced.\n\nSome well known early rappers include: Tupac Shakur, Snoop Dogg, The Notorious B.I.G., Eminem, and Sean "P-Diddy" Combs. During the 1990s there was a rivalry between the two big record labels "Death Row Records" and "Bad Boy Records". The rappers Tupac Shakur and Notorious B.I.G. were murdered. Later, the two record labels stopped the rivalry. Because most of the rappers who rapped for "Death Row Records" were from the West Coast of the US and most of the rappers who rapped for "Bad Boy Records" were from the East Coast, this rivalry was called "the West Coast – East Coast beef".\n\nThe fastest rapper according to Guinness World Records is Twista. In 1992, he rapped 11 syllables in one second. In 2013, the song Rap God by Eminem took the record for most words in a song; 1,560 in a little over 6 minutes, which is about 4 words per second.\n\nRelated pages\n Rhythm and blues\n Rock and roll\n Soul music\n Contemporary R&B\n Jazz\n Funk\n Pop music\n Country music\n Breakdance\n Reggaeton\n West Coast hip hop\n Comedy hip hop\n Old-school hip hop\n Political hip hop\n Underground hip hop\n Dirty rap\n Hip hop soul\n Rap metal\n Horrorcore\n Battle rap\n\nReferences \n\nHip hop\nMusic genres', metadata={}),
     Document(page_content='Rapping is a type of vocals, like singing. It is different to singing because it is more like talking, but timed with rhythm over music. Someone who raps is called a rapper, or sometimes an MC. That stands for Master of Ceremonies.\n\nRapping can be done over music of many types (or genres), including jazz, house, reggae and many more. One genre of music that includes a lot of rapping is hip hop. What people think of as rapping today, was started by African Americans in New York City, USA, in the 1970s. People would talk over disco music DJs at parties, and this gradually evolved into rapping. But, the start of the art of rapping is even older. Reggae artists in Jamaica used a similar style to rapping from the 1950s. Going back further than that, the West African Griots (travelling musicians and poets) would also rap over tribal drums in the 1400s.\n\nToday rapping is a very popular style of vocals. Many of the best selling artists in the world use it in their music.\n\nReferences \n\nMusic', metadata={}),
     Document(page_content="Breakdance (also called breaking, b-boying or b-girling) is a type of dance that is done by people who are part of the hip hop culture. B-boy means boy who dances on breaks (breakbeats). Breakdancing was invented in the early 1970s by African American and Latino American inner-city youth in the South Bronx in New York City.The dance style evolved during the 70s and 80s in big cities of the United States.\n\nBreakdancing uses different body movements, spins, arm movements, leg movements, all of which are done to the rhythm of hip hop music. Breakdancing was most popular in the 1980s but continues to be common today.\n\nThere are four categories in breakdance. They are power moves (windmill, tomas, airtrax and so on), style moves, toprock, downrock (footwork), and freezes (chair, airchair and so on). Many of moves come from gymnastics and kung-fu.\n\nBreakdancers dance with breakbeats. The difficulty of their skills decides the better b-boy. One of the biggest breakdance contests in the world is Battle of the Year (BOTY). It has several different types of contests. There are one-on-one battles, team battle, contest of showcase and so on. B-boy battle means dancing on random music. In 2013, the team coming from South Korea, Fusion MC, won the championship. Floorriorz coming from Japan got the award of best show.Good behavior with the sub\n\nHistory\nBreakdance occurred around a time where there was a lot of violence, on the streets of New York\n\nImportant Movements\nA\xa0freeze\xa0is a\xa0technique where the dancer suddenly stops, often in an interesting or balance-intensive position. Freezes often incorporate various twists of the body into stylish and often difficult positions.\nThe two-step move sets up the direction of movement and builds up momentum when dancing. This move allows the dancer to stay low and in contact with the ground, which places him in a good position for performing other\xa0dance\xa0moves.\xa0 As such, the two-step is often one of the first moves a break-dancer learns and it leads onto the 6-step.\nA kick in breakdance is a one-handed handstand, with often an impressive leg position and the free arm in some stylish position. They are often executed quickly to impress.\n\nLiterature\n-Guillaume Éradel, C'est quoi le breakdance? Saint-Denis, Edilivre, 2015 ()\n\nReferences\n\nDance\nHip hop", metadata={})]




```python
retriever = store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3},
)
```


```python
qa = RetrievalQA.from_chain_type(
    # llm=OpenAI(openai_api_key=openai.api_key),
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