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

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>
<h1 style="text-align: center">Finite Differences I</h1>


<h2 class="nocount">Contents</h2>

1. [Introduction](#Introduction)
2. [Summary](#Summary)
3. [Exercises](#Exercises)

<!-- #region -->
## Introduction

In this part of the course we describe how to compute derivatives of functions such as,

$$
f(x)=e^x sin(x)
$$

There as several conceptually different ways to do this. Following the same approach as for time integration, we can rely on Taylor's theorem to use the value of $f(x)$ at some neighbouring points of $x$. This approach relies on what are known as finite differences. Another way to compute derivatives relies on decomposing the function $f$ on a basis of functions $T_k(x)$ and computing the derivatives of $f$ from the known derivatives of $T_k(x)$. This method is known as the spectral method and will be described later on in the course.

## First-order derivative

We first consider the first-order derivative of $f(x)$. According to Taylor's theorem, we can approximate $f(x+\Delta x)$ as,

\begin{equation}
f(x+\Delta x)= f(x)+f'(x)\Delta x+O(\Delta x^2)
\end{equation}

and then get the following expression for the derivative of $f$ at point $x$:

\begin{equation}
f'(x) = \frac{f(x+\Delta x) - f(x)}{\Delta x}+O(\Delta x) \label{forwardTaylorDiff1}
\end{equation}

This expression is the usual left derivative of $f$. 

In any numerical problem, we have to limit the number of points at which we store the values of $f$ because the random access memory (RAM) of our computers is finite. We therefore need to introduce a grid along with a set of grid points at which we evaluate all physical quantities. For simplicity we consider a uniform grid in which the $n+1$ grid points are evenly distributed. The coordinates of the grid points are therefore,

\begin{equation}
 x_i = i \Delta x, \; \; 0\leq i \leq n
\end{equation}

with the endpoints of the grid located respectively at $x_0$ and $x_n$.

From eq. \ref{forwardTaylorDiff1} we can then define the following first-oder accurate approximation of the first-order derivative of $f$ at $x_i$:

\begin{equation}
f'(x_i) = \frac{f(x_{i+1}) - f(x_i)}{\Delta x}, \;\; \hbox{forward finite difference}\label{forwardDiff1}. 
\end{equation}

Schematically, we represent this expression through *a stencil* that indicates which grid points are involved in the computation:

<img width="600px" src="../figures/forwardDiff1.png">

In the above stencil, we use two grid points - indicated in red - to obtain a first-order accurate expression.

Using two grid points, we can actually achieve second-order accuracy. We resort again to Taylor's theorem to write:

\begin{equation}
f(x+\Delta x)= f(x)+f'(x)\Delta x+\frac12 f''(x)\Delta x^2+O(\Delta x^3) \\
f(x-\Delta x)= f(x)-f'(x)\Delta x+\frac12 f''(x)\Delta x^2+O(\Delta x^3) \\
\end{equation}

If we substract these two equations, we get:

\begin{equation}
f'(x) = \frac{f(x+\Delta x) - f(x-\Delta x)}{2\Delta x}+O(\Delta x^2) \label{forwardTaylorDiff2}
\end{equation}

We can then define the following second-oder accurate approximation of the first-order derivative of $f$ at $x_i$:

\begin{equation}
f'(x_i) = \frac{f(x_{i+1}) - f(x_{i-1})}{2\Delta x},\;\; \hbox{centered finite difference} \label{forwardDiff2}.
\end{equation}

This expression is called the centered finite difference first-order derivative and its stencil looks like this:

<img width="600px" src="../figures/centeredDiff1.png">


## Summary

In this notebook we have described...

## Exercises

**Exercise 1.** 
<!-- #endregion -->

```python
from IPython.core.display import HTML
css_file = '../Styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```
