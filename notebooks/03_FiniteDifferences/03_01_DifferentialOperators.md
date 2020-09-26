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


## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In this part of the course we describe how to compute numerically derivatives of functions such as,

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

In the above stencil, we use two grid points - indicated in red - to obtain a first-order accurate expression. In an identical manner, we can define another first-order estimate for the first-order derivative using a backward finite difference,

\begin{equation}
f'(x_i) = \frac{f(x_{i}) - f(x_{i-1})}{\Delta x}, \;\; \hbox{backward finite difference}\label{backwardDiff1}. 
\end{equation}

It is based on the right derivative of $f$ and its stencil is,

<img width="600px" src="../figures/backwardDiff1.png">

Using two grid points, we can actually do better than first-order accuracy. Resorting again to Taylor's theorem we write:

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

Using just two grid points, it's not possible to achieve an accuracy of higher order.

```python
nx = 20
lx = 2
dx = lx / (nx-1)
x = np.linspace(0, lx, nx)
print(x)
f = np.exp(x)*np.sin(10*x)
dfdx = np.exp(x)*(np.sin(10*x) + 10*np.cos(10*x))

fp = np.zeros(nx)
fp[1: -1]= (f[2:] - f[0:-2])/(2*dx)

fig, ax = plt.subplots(figsize=(8, 5))

ax.plot(x, dfdx, '*')
ax.plot(x, fp, '+')

ax.set_xlabel('$x$')
ax.set_ylabel('$f,f\'$')
#ax.set_title('Speed vs time (m/s)')

```

## Summary

In this notebook we have described...

## Exercises

**Exercise 1.** 

```python
from IPython.core.display import HTML
css_file = '../Styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```
