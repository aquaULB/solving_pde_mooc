---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#First-order-derivative" data-toc-modified-id="First-order-derivative-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>First-order derivative</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Summary</a></span></li></ul></div>
<!-- #endregion -->

<h1 style="text-align: center">Partial differential Equation I</h1>

## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In Chapter 2 and 3 of this course, we described respectively the time integration of ordinary differential equations and the discretization of differential operators using finite difference formulas.

Here we combine the tools learned in these two chapters to address the numerical solution of partial differential equations. We mainly focus on the first order wave equation (all symbols are properly defined in the corresponding sections of the notebooks),

$$
   \frac{\partial u}{\partial t}+c\frac{\partial u}{\partial x}=0,
$$

and the heat equation,

$$
   \partial_t T(x,t) = \alpha \frac{d^2 T} {dx^2}(x,t) + \sigma (x,t).
$$

But before we start digging into the theory, we begin this chapter by introducing the concept of Python modules.


## Python modules

...

## Advection equation

...

## Summary

...

```python
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

