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
<div class="toc"><ul class="toc-item"><li><span><a href="#Introduction" data-toc-modified-id="Introduction-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Introduction</a></span></li><li><span><a href="#Python-modules" data-toc-modified-id="Python-modules-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Python modules</a></span></li><li><span><a href="#Advection-equation" data-toc-modified-id="Advection-equation-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Advection equation</a></span></li><li><span><a href="#Summary" data-toc-modified-id="Summary-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Summary</a></span></li></ul></div>
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

```python
import sys
sys.path.insert(0, '../modules')
```

```python
from steppers import euler_step
```

```python
def rhs(u, dx, c):
    
    nx = u.shape[0]
    f = np.empty(nx)
    f[1:-1] = -c*(u[2:]-u[:-2]) / (2*dx)
    f[0] = 0
    f[-1] = 0
    
    return f
```

```python
t_final = 0.4
dt = 0.01
nt = int(t_final / dt)
c = 1.
nx = 101
lx = 1.
dx = lx / (nx-1)
x = np.linspace(0., 1., nx)
```

```python
# create the initial condition and plot it
u0 = np.exp(-200 * (x-0.25)**2)
```

```python
u = np.empty((nt+1, nx))
u[0] = u0.copy()
```

```python
for n in range(nt):
    t=t+dt
    u[n+1] = euler_step(u[n], rhs, dt, dx, c)
```

```python
fig, ax = plt.subplots(figsize=(10, 7))

ax.plot(x, u[0], label='Initial condition')
ax.plot(x, u[int(0.12 / dt)], lw=1.5, color='green', label='t=0.12')
ax.plot(x, u[int(0.25 / dt)], lw=1.5, color='indigo', label='t=0.25')
ax.plot(x, u[int(0.38 / dt)], lw=1.5, color='brown', label='t=0.38')

ax.set_xlabel('$x$')
ax.set_ylabel('$u$')
ax.set_title('Initial condition for the first order wave equation')
ax.legend()
```

<!-- #region cell_style="center" -->
## Summary

...
<!-- #endregion -->

```python
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

