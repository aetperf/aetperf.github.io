---
title: Built-in Expectations in Great Expectations
layout: post
comments: true
author: François Pacull
tags: Python MLOps data quality testing
---


<p align="center">
  <img width="200" src="https://github.com/great-expectations/great_expectations/blob/develop/generic_dickens_protagonist.png?raw=true" alt="Great Expectations">
</p>

[Great expectation](https://github.com/great-expectations/great_expectations) is a Python tool for data testing, documentation, and profiling. Here is a figure from the [documentation](https://docs.greatexpectations.io/docs/) describing its purpose:

<p align="center">
  <img width="600" src="https://docs.greatexpectations.io/assets/images/ge_overview-12fb8aa5caade62567e21be108526231.png" alt="Data validation with Great Expectations">
</p>

Great Expectations makes it easy to include data testing in your ML pipeline, when dealing with tabular data. Data testing is similar to software testing. You launch a test suite to check various assumptions on a given dataset: column names, types, min-max values, distributions, proportion of missing values, categories, string content... These are called Expectations. A set of declared Expectations is called an Expectation Suite. Once a Suite has been created, you can use it to validate any new/modified data. If ever it fails, you can adapt the pipeline depending on the type of data failure. The library also allows the automatic creation of data quality reports and documentation. 

Data checking makes it easier to identify the source of an error in an ML model, but also helps identifying some drifts or other problems with the input data. So, more generally it helps build trust around data. 

Great Expectations has many components and many features. In this post, we are going to focus on a single subject: Expectations, which is a central element of the library. We are going to list built-in Expectations, and apply them to an example dataset: the ubiquitous Titanic dataset. We are NOT going to deal with other subjects such as Expectations Suites, Datasources, Checkpoints, Stores, Data Contexts, CLI, deployment, metrics...

## Expectations

As it is written in the [documentation](https://docs.greatexpectations.io/docs/reference/core_concepts/#key-ideas), "It all starts with Expectations. An Expectation is how we communicate the way data should appear." From the [API reference](https://legacy.docs.greatexpectations.io/en/latest/reference/core_concepts/expectations/expectations.html#expectations):

> An Expectation is a statement describing a verifiable property of data. Like assertions in traditional python unit tests, Expectations provide a flexible, declarative language for describing expected behavior. Unlike traditional unit tests, Great Expectations applies Expectations to data instead of code.


Note that it is also possible to create custom Expectations. Here we are going to look at the built-in Expectations for the Pandas backend. As explained in the [documentation](https://docs.greatexpectations.io/docs/reference/expectations/implemented_expectations/): "Because Great Expectations can run against different platforms, not all Expectations have been implemented for all platforms." The different backends are Pandas, SQL and Spark.

### Imports


```python
import string
from datetime import datetime

import pandas as pd
import numpy as np
import great_expectations as ge
```

Versions:  
- Python 3.9.7  
- great_expectations 0.13.31   
- pandas 1.3.2  


### Load the data into a dataframe

We load the data with Great Expectations:


```python
df = ge.read_csv(
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
```


Here is a small description of each feature:  
- Survival - Survival (0 = No; 1 = Yes). Not included in test.csv file.  
- Pclass - Passenger Class (1 = 1st; 2 = 2nd; 3 = 3rd)  
- Name - Name  
- Sex - Sex  ("male or "female")
- Age - Age  
- SibSp - Number of Siblings/Spouses Aboard  
- Parch - Number of Parents/Children Aboard  
- Ticket - Ticket Number  
- Fare - Passenger Fare  
- Cabin - Cabin  
- Embarked - Port of Embarkation (C = Cherbourg; Q = Queenstown; S = Southampton)  


```python
df.shape
```




    (891, 12)




```python
df.isna().sum()
```




    PassengerId      0
    Survived         0
    Pclass           0
    Name             0
    Sex              0
    Age            177
    SibSp            0
    Parch            0
    Ticket           0
    Fare             0
    Cabin          687
    Embarked         2
    dtype: int64




```python
df.nunique()
```




    PassengerId    891
    Survived         2
    Pclass           3
    Name           891
    Sex              2
    Age             88
    SibSp            7
    Parch            7
    Ticket         681
    Fare           248
    Cabin          147
    Embarked         3
    dtype: int64




```python
df.dtypes
```




    PassengerId      int64
    Survived         int64
    Pclass           int64
    Name            object
    Sex             object
    Age            float64
    SibSp            int64
    Parch            int64
    Ticket          object
    Fare           float64
    Cabin           object
    Embarked        object
    dtype: object



### Testing an Expectation on the dataset

The Expectation is seen as a DataFrame method. Let's try the `expect_column_to_exist` Expectation:


```python
expect = df.expect_column_to_exist(column="PassengerId")
expect
```




    {
      "exception_info": {
        "raised_exception": false,
        "exception_traceback": null,
        "exception_message": null
      },
      "success": true,
      "result": {},
      "meta": {}
    }



It returns a dict-like validation report. The level of verbosity can be changed when creating reports. But anyway, in the present post, we are only interested in the "success" key:


```python
expect.success
```




    True



So let's start with the available Expectations. Note that some Expectations have not yet been migrated to the v3 (Batch Request) API, so we did not include them in this post. Also, we did not investigate [distributional Expectations](https://docs.greatexpectations.io/docs/reference/expectations/distributional_expectations), which are not so easy to use.



## Expectations : Table shape


```python
columns = df.columns
for column in columns:
    expect = df.expect_column_to_exist(column=column)
    assert expect.success
```


```python
expect = df.expect_table_columns_to_match_ordered_list(column_list=columns)
assert expect.success
```


```python
expect = df.expect_table_columns_to_match_ordered_list(column_list=columns[::-1])
assert not expect.success
```


```python
expect = df.expect_table_columns_to_match_set(column_set=columns)
assert expect.success
```


```python
expect = df.expect_table_columns_to_match_set(column_set=columns[:-1])
expect.result["details"]
```




    {'mismatched': {'unexpected': ['Embarked']}}



When the `exact_match` boolean parameter is set to `False`, we get a failure if a given column is not found in the dataset, but not if columns from the dataset are not given:


```python
expect = df.expect_table_columns_to_match_set(
    column_set=["PassengerId", "toto"], exact_match=False
)
assert not expect.success
```


```python
expect = df.expect_table_columns_to_match_set(column_set=[], exact_match=False)
assert expect.success
```


```python
expect = df.expect_table_row_count_to_be_between(min_value=700, max_value=1000)
assert expect.success
```

One can skip one of the arguments `min_value` or `max_value`. The default value is `None`:


```python
expect = df.expect_table_row_count_to_be_between(max_value=1000)
assert expect.success
```


```python
expect = df.expect_table_row_count_to_equal(value=891)
assert expect.success
```

## Expectations : Missing values, unique values, and types

All the Expectations in this category support the `mostly` argument, used when only a fraction of the column values is supposed to match the expectation.


```python
expect = df.expect_column_values_to_be_unique(column="PassengerId")
assert expect.success
```


```python
expect = df.expect_column_values_to_be_unique(column="Ticket", mostly=0.6)
assert expect.success
```


```python
for column in ["PassengerId", "Survived"]:
    expect = df.expect_column_values_to_not_be_null(column=column)
    assert expect.success
```


```python
expect = df.expect_column_values_to_not_be_null(column="Age", mostly=0.8)
assert expect.success
```


```python
expect = df.expect_column_values_to_not_be_null(column="Age", mostly=0.9)
assert not expect.success
```


```python
tmp_col = "NullCol"
df[tmp_col] = np.NaN
expect = df.expect_column_values_to_be_null(column="NullCol")
assert expect.success
df.drop(tmp_col, axis=1, inplace=True)
```


```python
expect = df.expect_column_values_to_be_null(column="Cabin", mostly=0.5)
assert expect.success
```

For the expected types (`type_` parameter), we use regular Python types. With the Pandas backend, we could also use NumPy dtypes. From the [documentation](https://legacy.docs.greatexpectations.io/en/latest/autoapi/great_expectations/expectations/core/expect_column_values_to_be_of_type/index.html?highlight=expect_column_values_to_be_of_type#great_expectations.expectations.core.expect_column_values_to_be_of_type.ExpectColumnValuesToBeOfType):

> Valid types are defined by the current backend implementation and are dynamically loaded. For example, valid types for PandasDataset include any numpy dtype values (such as ‘int64’) or native python types (such as ‘int’), whereas valid types for a SqlAlchemyDataset include types named by the current driver such as ‘INTEGER’ in most SQL dialects and ‘TEXT’ in dialects such as postgresql. Valid types for SparkDFDataset include ‘StringType’, ‘BooleanType’ and other pyspark-defined type names.


```python
types = {
    "PassengerId": "int",
    "Survived": "int",
    "Pclass": "int",
    "Name": "str",
    "Sex": "str",
    "Age": "float",
    "SibSp": "int",
    "Parch": "int",
    "Ticket": "str",
    "Fare": "float",
    "Cabin": "str",
    "Embarked": "str",
}
for column, type_ in types.items():
    expect = df.expect_column_values_to_be_of_type(column=column, type_=type_)
    assert expect.success
```

We could also use the `mostly` argument in case where most of the values are of the same type:


```python
tmp_col = "mixedTypes"
df[tmp_col] = 1
df.loc[df[:3].index, tmp_col] = "a"
```


```python
expect = df.expect_column_values_to_be_of_type(column=tmp_col, type_="int", mostly=0.9)
assert expect.success
```


```python
expect = df.expect_column_values_to_be_in_type_list(
    column=tmp_col, type_list=["int", "str"]
)
assert expect.success
```

Let's insert a `float` type item in the temporary column and use the `mostly` argument:


```python
df.loc[df[:1].index, tmp_col] = np.pi
```


```python
expect = df.expect_column_values_to_be_in_type_list(
    column=tmp_col, type_list=["int", "str"]
)
assert not expect.success
```


```python
expect = df.expect_column_values_to_be_in_type_list(
    column=tmp_col, type_list=["int", "str"], mostly=0.99
)
assert expect.success
df.drop(tmp_col, axis=1, inplace=True)
```

## Expectations : Sets and ranges

All the Expectations in this category support the `mostly` argument.


```python
expect = df.expect_column_values_to_be_in_set(column="Survived", value_set=[0, 1])
assert expect.success
```


```python
expect = df.expect_column_values_to_be_in_set(column="Pclass", value_set=[1, 2, 3])
assert expect.success
```


```python
expect = df.expect_column_values_to_be_in_set(
    column="Sex",
    value_set=["female", "male"],
)
assert expect.success
```


```python
expect = df.expect_column_values_to_be_in_set(
    column="Embarked",
    value_set=["C", "Q", "S"],
)
assert expect.success
```

```python
expect = df.expect_column_values_to_be_in_set(
    column="Embarked", value_set=["C", "S"], mostly=0.9
)
assert expect.success
```

Another possible argument is the boolean `parse_strings_as_datetimes`.


```python
date_col = "datetimeCol"
date = "1912-04-15"
df[date_col] = date
df[date_col] = pd.to_datetime(df[date_col])
```



```python
expect = df.expect_column_values_to_be_in_set(
    column=date_col,
    value_set=["1912-04-14", date, 2],
    parse_strings_as_datetimes=False,
)
assert not expect.success
```


```python
expect = df.expect_column_values_to_be_in_set(
    column=date_col,
    value_set=["1912-04-14", date, 2],
    parse_strings_as_datetimes=True,
)
assert expect.success
```


```python
alphabet_string = string.ascii_uppercase
value_set = [l for l in alphabet_string if l not in ["C", "Q", "S"]]

expect = df.expect_column_values_to_not_be_in_set(
    column="Embarked",
    value_set=value_set,
)
assert expect.success
```

With `mostly`:


```python
expect = df.expect_column_values_to_not_be_in_set(
    column="Embarked", value_set=["Q", "X", "Y", "Z"], mostly=0.9
)
assert expect.success
```


```python
expect = df.expect_column_values_to_be_between(column="Age", min_value=0, max_value=120)
assert expect.success
```


One can also use `strict_min` and `strict_max` to include the bounds or not (`default=False`). 


```python
df.Parch.max()
```




    6




```python
expect = df.expect_column_values_to_be_between(
    column="Parch", min_value=0, max_value=6, strict_max=True
)
assert not expect.success
```


```python
expect = df.expect_column_values_to_be_between(column="Parch", min_value=0, max_value=6)
assert expect.success
```

Other possible arguments are:
- `parse_strings_as_datetimes` (boolean or None) – If True, parse min_value, max_value, and all non-null column values to datetimes before making comparisons.
- `output_strftime_format` (str or None) – A valid strfime format for datetime output. Only used if parse_strings_as_datetimes=True.
- `mostly`

Let's try an example with `parse_strings_as_datetimes` and `output_strftime_format`:


```python
df[date_col] = "1912/04/15"
expect = df.expect_column_values_to_be_between(
    column=date_col,
    min_value="1912-04-14",
    max_value="1912-04-16",
    parse_strings_as_datetimes=True,
    output_strftime_format="%Y/%m/%d",
)
assert expect.success
```


```python
df.sort_values(by="PassengerId", inplace=True, ascending=True)
expect = df.expect_column_values_to_be_increasing(column="PassengerId")
assert expect.success
```

Other possible arguments are:
- `strictly` (Boolean or None) – If True, values must be strictly greater than previous values
- `parse_strings_as_datetimes` (boolean or None) – If True, all non-null column values to datetimes before making comparisons
- `mostly`


```python
df.sort_values(by="PassengerId", inplace=True, ascending=True)
expect = df.expect_column_values_to_be_increasing(column="PassengerId", strictly=True)
assert expect.success
```

In order to use `parse_strings_as_datetimes`, we create a column with an increasing dates as strings:


```python
df[date_col] = pd.date_range(
    start=datetime(1912, 1, 1), periods=df.shape[0], freq="D"
).astype(str)
```



```python
expect = df.expect_column_values_to_be_increasing(
    column=date_col, parse_strings_as_datetimes=True, strictly=True
)
assert expect.success
df.drop(date_col, axis=1, inplace=True)
```


```python
df.sort_values(by="PassengerId", inplace=True, ascending=False)
expect = df.expect_column_values_to_be_decreasing(column="PassengerId")
assert expect.success
```

Of course, `expect_column_values_to_be_decreasing` takes the same arguments as `expect_column_values_to_be_increasing`.

## Expectations : String matching

All the Expectations in this category support the `mostly` argument.


```python
expect = df.expect_column_value_lengths_to_be_between(
    column="Name", min_value=10, max_value=200
)
assert expect.success
```


```python
expect = df.expect_column_value_lengths_to_be_between(
    column="Name", min_value=10, max_value=50, mostly=0.9
)
assert expect.success
```


```python
expect = df.expect_column_value_lengths_to_equal(column="Embarked", value=1)
assert expect.success
```


```python
expect = df.expect_column_value_lengths_to_equal(column="Cabin", value=3, mostly=0.3)
assert expect.success
```

We are going to check this`expect_column_values_to_match_regex` expectation on the `Cabin` feature. Here are some example values:
```
'B57 B59 B63 B66', 'C7', 'E34', 'C32', 'B18', 'C124', 'C91', 'E40', 'T'
```


```python
expect = df.expect_column_values_to_match_regex(
    column="Cabin",
    regex="(?:[A-Z]\\d{0,3}\\s?)+",  # at least 1 group of 1 uppercase letter followed by 0 or more digits and 0 or 1 white space
)
assert expect.success
```


```python
expect = df.expect_column_values_to_match_regex_list(
    column="Cabin", regex_list=["[A-Z]\\d{0,3}", "(?:[A-Z]\\d{0,3}\\s?){2,100}"]
)
assert expect.success
```


```python
expect = df.expect_column_values_to_not_match_regex_list(
    column="Cabin", regex_list=["[a-z]\\d{1,3}", "[a-z]\\d{1,3}[a-z]"]
)
assert expect.success
```

## Expectations : Datetime and JSON parsing

All the Expectations in this category support the `mostly` argument.


```python
df["date_example"] = "1912/04/15"
expect = df.expect_column_values_to_match_strftime_format(
    column="date_example", strftime_format="%Y/%m/%d"
)
assert expect.success
df.drop("date_example", axis=1, inplace=True)
```


```python
df["date_example"] = "2012-01-19 17:21:00"
expect = df.expect_column_values_to_be_dateutil_parseable(column="date_example")
assert expect.success
df.drop("date_example", axis=1, inplace=True)
```


```python
df[
    "json_example"
] = """{
    "menu": {
        "id": "file",
        "value": "File",
        "popup": {
            "menuitem": [
                { "value": "New", "onclick": "CreateNewDoc()" },
                { "value": "Open", "onclick": "OpenDoc()" },
                { "value": "Close", "onclick": "CloseDoc()" }
            ]
        }
    }
}"""
expect = df.expect_column_values_to_be_json_parseable(column="json_example")
assert expect.success
df.drop("json_example", axis=1, inplace=True)
```


```python
df["latitude"] = 180.0 * (np.random.rand(len(df)) - 0.5)
df["longitude"] = 360.0 * (np.random.rand(len(df)) - 0.5)
df["json_example"] = (
    """{"latitude": """
    + df.latitude.astype(str)
    + """, "longitude": """
    + df.latitude.astype(str)
    + "}"
)

json_schema = {
    "title": "Longitude and Latitude",
    "description": "A geographical coordinate on a planet (most commonly Earth).",
    "required": ["latitude", "longitude"],
    "type": "object",
    "properties": {
        "latitude": {"type": "number", "minimum": -90, "maximum": 90},
        "longitude": {"type": "number", "minimum": -180, "maximum": 180},
    },
}

expect = df.expect_column_values_to_match_json_schema(
    column="json_example", json_schema=json_schema
)
assert expect.success
df.drop(["latitude", "longitude", "json_example"], axis=1, inplace=True)
```

## Expectations : Aggregate functions

`expect_column_distinct_values_to_be_in_set` is not really different from `expect_column_values_to_be_in_set`. Note that it also supporst the `parse_strings_as_datetimes` argument.


```python
expect = df.expect_column_distinct_values_to_be_in_set(
    column="Survived", value_set=[0, 1]
)
assert expect.success
```


```python
expect = df.expect_column_distinct_values_to_contain_set(
    column="Embarked",
    value_set=["C", "Q"],
)
assert expect.success
```


```python
expect = df.expect_column_distinct_values_to_equal_set(
    column="Embarked",
    value_set=["C", "Q", "S"],
)
assert expect.success
```


```python
expect = df.expect_column_mean_to_be_between(column="Age", min_value=20, max_value=40)
assert expect.success
```


```python
df.Age.median()
```




    28.0




```python
expect = df.expect_column_median_to_be_between(column="Age", min_value=20, max_value=40)
assert expect.success
```


```python
expect = df.expect_column_quantile_values_to_be_between(
    column="Age",
    quantile_ranges={
        "quantiles": [0.25, 0.5, 0.75],
        "value_ranges": [[15, 25], [23, 33], [35, 43]],
    },
)
assert expect.success
```


```python
expect = df.expect_column_mean_to_be_between(
    column="Age", min_value=0.0, strict_max=100.0
)
assert expect.success
```


```python
expect = df.expect_column_stdev_to_be_between(
    column="Age", min_value=10.0, strict_max=20.0
)
assert expect.success
```


```python
expect = df.expect_column_unique_value_count_to_be_between(
    column="Pclass", min_value=1, max_value=3
)
assert expect.success
```

`expect_column_proportion_of_unique_values_to_be_between`, from  the [API reference](https://legacy.docs.greatexpectations.io/en/latest/autoapi/great_expectations/expectations/core/expect_column_proportion_of_unique_values_to_be_between/index.html?highlight=expect_column_proportion_of_unique_values_to_be_between):

> Expect the proportion of unique values to be between a minimum value and a maximum value.

> For example, in a column containing [1, 2, 2, 3, 3, 3, 4, 4, 4, 4], there are 4 unique values and 10 total values for a proportion of 0.4.



```python
expect = df.expect_column_proportion_of_unique_values_to_be_between(
    column="Embarked", strict_min=0.9, strict_max=0.99
)
assert expect.success
```


```python
expect = df.expect_column_most_common_value_to_be_in_set(
    column="Embarked", value_set=("S", "C")
)
assert expect.success
```


```python
expect = df.expect_column_sum_to_be_between(
    column="Survived", min_value=300, max_value=400
)
assert expect.success
```


```python
expect = df.expect_column_min_to_be_between(column="Age", min_value=0, max_value=10)
assert expect.success
```


```python
expect = df.expect_column_max_to_be_between(column="Age", min_value=40, max_value=100)
assert expect.success
```


```python
df.Pclass.value_counts()
```




    3    491
    1    216
    2    184
    Name: Pclass, dtype: int64



For the `expect_column_kl_divergence_to_be_less_than` Expectation, we used some coefficients found on the [github repository](https://github.com/great-expectations/great_expectations/blob/4a0aa8c67420aadcafeff7a45fbbe54943be543a/tests/profile/fixtures/expected_evrs_BasicSuiteBuilderProfiler_on_titanic_demo_mode.json):


```python
df.expect_column_kl_divergence_to_be_less_than(
    column="Pclass",
    partition_object={
        "values": ["*", 1, 2, 3],
        "weights": [
            0.0007616146230007616,
            0.24523990860624523,
            0.2124904798172125,
            0.5415079969535415,
        ],
    },
    threshold=0.6,
)
assert expect.success
```

## Conditional Expectations

Finally we are going to have a look at conditional Expectations with the `row_condition` argument. As explained in the [documentation](https://docs.greatexpectations.io/docs/reference/expectations/conditional_expectations):

> Sometimes one may hold an Expectation not for a dataset in its entirety but only for a particular subset. Alternatively, what one expects of some variable may depend on the value of another. One may, for example, expect a column that holds the country of origin to not be null only for people of foreign descent.


```python
df.expect_column_values_to_be_in_set(
    column="Sex",
    value_set=["male", "female"],
    row_condition='Embarked=="S"',
    condition_parser="pandas",
)
assert expect.success
```

Note that some of the Expectations do not take the `row_condition` argument:
- `expect_column_to_exist`
- `expect_table_columns_to_match_ordered_list`
- `expect_table_column_count_to_be_between`
- `expect_table_column_count_to_equal`


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