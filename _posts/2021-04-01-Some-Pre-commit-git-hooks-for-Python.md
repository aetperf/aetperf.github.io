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


The pre-commit hooks are a great way to check and clean the code with no pain. They are executed automatically when committing changes. This can be usefull when several people are working on the some package with different code styles, but also can help finding some mistakes, typos, etc... 

In this post we are dealing with git pre-commit hooks for Python code. We are going to use the `pre-commit` package. We are not gonna go into much details regarding all the different possible configurations of the many hooks available. What we want is basically to format the code, to remove unused imports and to sort and classify these imports (standard library, external libraries, and local imports).

First we need to install [`pre-commit`](https://github.com/pre-commit/pre-commit), which is a framework for managing and maintaining multi-language pre-commit hooks:

```python
$ pip install pre-commit
$ pre-commit --version 
pre-commit 2.11.1
```

Next we need to go to our git repository of interest and create a config file for `pre-commit`. So we create this initial YAML config file that we are going to complete later:

```
repos:
-   repo: https://github.com/pre-commit/pre-commit-hooks
    rev: v3.4.0
    hooks:
    -   id: trailing-whitespace
    -   id: end-of-file-fixer
```

It needs to be called ``.pre-commit-config.yaml`. But what we also want to use as pre-commit hooks, are the following tools:

- [black](https://github.com/psf/black): a Python code formatter
- [pycln](https://github.com/hadialqattan/pycln): a formatter for finding and removing unused import statements
- [isort](https://github.com/PyCQA/isort): a Python utility / library to sort imports alphabetically, and automatically separated into sections and by type

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
target-version = ['py36', 'py37', 'py38']
include = '\.pyi?$'

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

Now we add pre-commit to the requirements-dev.txt file. We assume here that it has already been created:

```python 
$ echo "pre-commit" >> requirements-dev.txt
```

Also we need to register the git hook scripts:

```python 
$ pre-commit install
pre-commit installed at .git/hooks/pre-commit
```

Now we can try to tun the hooks to check that everything is working and to initialize the hooks:

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

And now everything runs smoothly:

```bash
$ git commit -m "pre-commit hook test"
Trim Trailing Whitespace.................................................Passed
Fix End of Files.........................................................Passed
black....................................................................Passed
pycln....................................................................Passed
isort....................................................................Passed
```


We can test these hooks with a piece of code. Here we are just looking at the imports part:

```python
from sklearn.metrics import max_error, mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
import os, sys
import pandas as pd
import model_cfg, model_tls
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model importLinearRegression
from sklearn.model_selection import KFold
```

They are rather messy. Also, the sys, sklearn.linear imports are not used later in the code. After committing this source code, it looks quite better:


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

What if we want to add a hook and update pre-commit? Let's say we want to add this tool:

- [Mypy](https://github.com/python/mypy): optional static type checker for Python


We need to update the YAML pre-commit config file ``.pre-commit-config.yaml`by adding the Mypy hook:

```yaml
-   repo: https://github.com/pre-commit/mirrors-mypy
    rev: v0.812
    hooks:
    -   id: mypy
```

Now let's run the pre-commit hooks:

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

This seems to be OK. Let's not forget to commit the YAML file change:

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