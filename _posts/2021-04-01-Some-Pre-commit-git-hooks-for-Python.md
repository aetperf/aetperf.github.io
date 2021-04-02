---
title: Some Pre-commit git hooks for Python
layout: post
comments: true
author: François Pacull
tags: Python Git Hooks Black ISort Pycln Mypy Linter
---

<p align="center">
  <img width="400" src="/img/2021-04-01_01/pre-commit.png" alt="Pre-commit">
</p>


Pre-commit hooks are a great way to automatically check and clean the code. They are executed when committing changes to git . This can be useful when several people are working on the same package with different code styles, but also to help finding some typos, mistakes, etc... 

In this post we are dealing with git pre-commit hooks for Python code, with the [`pre-commit`](https://github.com/pre-commit/pre-commit) package. We will not go into much details regarding the different configurations of all the possible hooks. [Here](https://pre-commit.com/hooks.html) is a list of some supported hooks. What we want is basically to format the code, remove unused imports and sort and classify these imports (standard library < external libraries < local imports).

First we need to install [`pre-commit`](https://github.com/pre-commit/pre-commit), which is a framework for managing and maintaining multi-language pre-commit hooks:

```python
$ pip install pre-commit
$ pre-commit --version 
pre-commit 2.11.1
```

Next we need to go to our git repository of interest and create a config file for `pre-commit`. Here is the initial YAML config file that we are going to complete later:

```
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
```

It is called `.pre-commit-config.yaml`. So far, it deals with trailing white spaces and end of files. What we want to use as pre-commit hooks, in this post, are the following tools:

- [black](https://github.com/psf/black): a Python code formatter

<p align="center">
  <img width="250" src="/img/2021-04-01_01/black.png" alt="Black">
</p>

- [pycln](https://github.com/hadialqattan/pycln): a formatter for finding and removing unused import statements

<p align="center">
  <img width="250" src="/img/2021-04-01_01/pycln.png" alt="Pycln">
</p>

- [isort](https://github.com/PyCQA/isort): a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type

<p align="center">
  <img width="250" src="/img/2021-04-01_01/isort.png" alt="Isort">
</p>

So we need to add the following lines to the pre-commit config file:

```yaml
-   repo: https://github.com/psf/black
    rev: 20.8b1
    hooks:
    -   id: black
        args: [--config=pyproject.toml]
-   repo: https://github.com/hadialqattan/pycln
    rev: v0.0.1-beta.3
    hooks:
    -   id: pycln
        args: [--config=pyproject.toml]
-   repo: https://github.com/pycqa/isort
    rev: 5.5.4
    hooks:
    -   id: isort
        files: "\\.(py)$"
        args: [--settings-path=pyproject.toml]
```

Also, we need to create a TOML settings file for theses tools, named `pyproject.toml`:

```toml
[tool.black]
line-length = 79

[tool.pycln]
all = true

[tool.isort]
line_length = 79
multi_line_output = 3
include_trailing_comma = true
force_grid_wrap = 0
use_parentheses = true
ensure_newline_before_comments = true
```

A lot of settings can be specified... A concern might be the line length and to make Black compatible with isort. Now we add `pre-commit` and the different formatting tools to the `requirements-dev.txt` file:

```python 
$ cat requirements-dev.txt
pre-commit
black
pycln
isort
$ pip install -r requirements-dev.txt
```

Also we need to "register" `pre-commit`:

```python 
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

We can run the hooks manually in order to check that everything is working and to initialize the whole process:

```bash
$ pre-commit run --all-files
[INFO] Initializing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Initializing environment for https://github.com/psf/black.
[INFO] Initializing environment for https://github.com/hadialqattan/pycln.
[INFO] Initializing environment for https://github.com/pycqa/isort.
[INFO] Installing environment for https://github.com/pre-commit/pre-commit-hooks.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/psf/black.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/hadialqattan/pycln.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
[INFO] Installing environment for https://github.com/pycqa/isort.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Failed
- hook id: end-of-file-fixer
- exit code: 1
- files were modified by this hook

Fixing README.md
Fixing requirements.txt

black....................................................................Passed
pycln....................................................................Passed
isort....................................................................Failed
- hook id: isort
- files were modified by this hook

Fixing /home/francois/Workspace/PCH/src/model_tls.py
Fixing /home/francois/Workspace/PCH/src/train.py
```

Now everything runs smoothly when we commit some changes:

```bash
$ git commit -m "pre-commit hook test"
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
black....................................................................Passed
pycln....................................................................Passed
isort....................................................................Passed
```

We can test these hooks with a "dirty" piece of code. Here we are just looking at some imports part:

```python
from sklearn.metrics import max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import os, sys
import pandas as pd
import model_cfg, model_tls
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
```

These imports are a little bit messy. Also, the `sys` and `sklearn.linear.LinearRegression` imports are not used later in the code. After committing this source code, it looks quite better:

```python
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    max_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import KFold

import model_cfg
import model_tls
```

What if we want to add a hook and update the process? Let's say we want to add this tool:

- [Mypy](https://github.com/python/mypy): optional static type checker for Python

We just update the YAML pre-commit config file `.pre-commit-config.yaml` by adding the Mypy hook:

```yaml
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.812
    hooks:
    -   id: mypy
```

Then we add `mypy` to the `requirements-dev.txt` file. We do not specify any specific setting for Mypy in the  `pyproject.toml`. Let's run the pre-commit hooks:

```bash
$ pre-commit run --all-files
[INFO] Initializing environment for https://github.com/pre-commit/mirrors-mypy.
[INFO] Installing environment for https://github.com/pre-commit/mirrors-mypy.
[INFO] Once installed this environment will be reused.
[INFO] This may take a few minutes...
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
black....................................................................Passed
pycln....................................................................Passed
isort....................................................................Passed
mypy.....................................................................Passed
```

This seems to be OK! Let's not forget to commit the YAML file change:

```bash
$ git add .pre-commit-config.yaml
$ git commit -m "added the mypy hook "
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
black....................................................................Passed
pycln....................................................................Passed
isort....................................................................Passed
mypy.....................................................................Passed
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