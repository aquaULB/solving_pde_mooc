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
<h1 style="text-align: center">Finite Differences II</h1>


<h2 class="nocount">Contents</h2>

1. [Introduction](#Introduction)
2. [Higher order derivative](#Second-order-derivative)
3. [Functions](#Functions)
4. [Matrix formulation](#Matrix-formulation)
5. [Summary](#Summary)


## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In this notebook we extend the concept of finite differences to higher order derivatives. We also discuss the use of `functions` and finally we describe how to construct matrices corresponding to the finite difference operators. The latter are very useful when solving boundary value problems or eigenvalue problems.

## Higher order derivative

Using finite differences, we can construct derivatives up to any order. Before we discuss this, we first explicitly describe in detail the second-order derivative that is later used extensively in the course.


### Second-order derivative

Using Taylor's theorem we can write:

\begin{align}
& f(x+\Delta x) = f(x)+f'(x)\Delta x+\frac12 f''(x)\Delta x^2+\frac16 f'''(x)\Delta x^3+O(\Delta^4)\label{eq:leftTaylor2} \\
& f(x-\Delta x) = f(x)-f'(x)\Delta x+\frac12 f''(x)\Delta x^2-\frac16 f'''(x)\Delta x^3+O(\Delta^4)\label{eq:rightTaylor2}.
\end{align}

If we add these two equations, we can define a centered second-order accurate formula for $f''$ at grid point $x_i$:

\begin{equation}
f''_i=\frac{f_{i-1}-2f_i+f_{i+1}}{\Delta^2}\label{eq:centeredDiff2}
\end{equation}

The stencil for this expression is the sequence $[-1,0,1]$ and we represent it as:

<img width="600px" src="../figures/centeredDiff2.png">

The centered second-order derivative cannot be used at the boundary nodes. Some one-sided formulas are needed at those locations.

Let us write a Python code to check that expression \ref{eq:centeredDiff2} works as expected. We use the same test function as in the previous notebook - $f(x)=e^x \sin(3\pi x)$ - and first create a fine representation for it in the interval $x\in [O, \pi]$.

```python
pi = np.pi       # 3.14...
nx = 200         # number of grid points (fine grid)
lx = pi          # length of the interval
dx = lx / (nx-1) # grid spacing
```

```python
x = np.linspace(0, lx, nx)   # coordinates in the fine grid
f = np.exp(x)*np.sin(3*pi*x) # function in the fine grid

# Let us build a numpy array for the exact repre-
# sentation of the first-order derivative of f(x).
ddf = np.exp(x)*(np.sin(3*pi*x) + 6*pi*np.cos(3*pi*x)-9*pi**2*np.sin(3*pi*x))
```

We now build a coarse grid with 80 points, and evaluate the second-order derivative using the centered finite difference formula; note how we use the slicing technique we described in the previous notebook.

```python
nx = 80 # number of grid points (coarse grid)
lx = pi # length of invertal
dx = lx / (nx-1) # grid spacing
x_c = np.linspace(0, lx, nx) # coordinates of points on the coarse grid

f_c = np.exp(x_c)*np.sin(3*pi*x_c) # function on the coarse grid

ddf_c = np.empty(nx) 
ddf_c[1:-1] = (f_c[2:] -2*f_c[1:-1] +f_c[:-2]) / dx**2 # boundary nodes are included
```

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x[1:-1], ddf[1:-1])
ax.plot(x_c[1:-1], ddf_c[1:-1], '^g')
ax.set_xlabel('$x$')
ax.set_ylabel('$f\'$')
```

As the centered formula for $f''$ is not defined at the boundary nodes, these have been excluded in the computation. But in the next section, we will provide information on how to cope with this issue.


### Higher order derivatives and one-side stencils


It should now be clear that the construction of finite difference formulas to compute differential operators can be done using Taylor's theorem. In general, as we increase the order of the derivative, we have to increase the number of points in the corresponding stencil. The construction of these stencils is not complicated and is well documented in several places so we will not repeat it here. Instead we refer to the very detailed [Wikipedia][1] page that contains finite difference formulas for all the cases we use in this course, and many more.

To make this notebook sef contained, we list here some of the formulas we will/might need later on.

[1]: <https://en.wikipedia.org/wiki/Finite_difference_coefficient> "list of finite difference formulas"


We begin we some **centered finite difference** expressions:

<br>
<table class="wikitable" style="text-align:center">
	<tbody>
		<tr>
			<th>Derivative</th>
			<th>Accuracy</th>
			<th>&minus;3</th>
			<th>&minus;2</th>
			<th>&minus;1</th>
			<th>0</th>
			<th>1</th>
			<th>2</th>
			<th>3</th>
		</tr>
		<tr>
			<td rowspan="2">1</td>
			<td>2</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&minus;1/2</td>
			<td>0</td>
			<td>1/2</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>4</td>
			<td>&nbsp;</td>
			<td>1/12</td>
			<td>&minus;2/3</td>
			<td>0</td>
			<td>2/3</td>
			<td>&minus;1/12</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td rowspan="2">2</td>
			<td>2</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>1</td>
			<td>&minus;2</td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>4</td>
			<td>&nbsp;</td>
			<td>&minus;1/12</td>
			<td>4/3</td>
			<td>&minus;5/2</td>
			<td>4/3</td>
			<td>&minus;1/12</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td rowspan="2">3</td>
			<td>2</td>
			<td>&nbsp;</td>
			<td>&minus;1/2</td>
			<td>1</td>
			<td>0</td>
			<td>&minus;1</td>
			<td>1/2</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>4</td>
			<td>1/8</td>
			<td>&minus;1</td>
			<td>13/8</td>
			<td>0</td>
			<td>&minus;13/8</td>
			<td>1</td>
			<td>&minus;1/8</td>
		</tr>
		<tr>
			<td rowspan="2">4</td>
			<td>2</td>
			<td>&nbsp;</td>
			<td>1</td>
			<td>&minus;4</td>
			<td>6</td>
			<td>&minus;4</td>
			<td>1</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>4</td>
			<td>&minus;1/6</td>
			<td>2</td>
			<td>&minus;13/2</td>
			<td>28/3</td>
			<td>&minus;13/2</td>
			<td>2</td>
			<td>&minus;1/6</td>
		</tr>
	</tbody>
</table>


You should recognize the centered difference stencils we have already discussed for the first- and second-order derivatives. Each lines contains the coefficients $c_j$ to be applied at the corresponding stencil point; to complete the finite difference formula, we also need to divide the finite difference with $\Delta^k$ where $k$ is the order derivative. For example, the second-order accurate formula for the fourth-order derivative is:

\begin{equation}
f''''_i = \frac{f_{i-2}-4f_{i-1}+6f_{i+1}-4f_{i+1}+f_{i+2}}{\Delta^4}
\end{equation}

Graphically we have:

<img width="800px" src="../figures/centeredDiff4.png">

<!-- #region -->
We have seen that centered stencils are usually not applicable at boundary nodes. Thankfully, alternate finite difference formulas can be constructed for these nodes and the [Wikipedia][1]
pages also lists a large collections of such **one-sided formulas**. Here we limit our attention to first- and second-order accurate expressions.



[1]: <https://en.wikipedia.org/wiki/Finite_difference_coefficient> "list of finite difference formulas"
<!-- #endregion -->

<!-- #region -->
**Forward one-sided finite difference formulas:**


<table class="wikitable" style="text-align:center">
	<tbody>
		<tr>
			<th>Derivative </th>
			<th>Accuracy </th>
			<th>0 </th>
			<th>1 </th>
			<th>2 </th>
			<th>3 </th>
			<th>4 </th>
			<th>5 </th>
		</tr>
		<tr>
			<td rowspan="2">1 </td>
			<td>1</td>
			<td>−1</td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>2</td>
			<td>−3/2</td>
			<td>2</td>
			<td>−1/2</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td rowspan="2">2 </td>
			<td>1</td>
			<td>1</td>
			<td>−2</td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>2</td>
			<td>2</td>
			<td>−5</td>
			<td>4</td>
			<td>−1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td rowspan="2">3 </td>
			<td>1</td>
			<td>−1</td>
			<td>3</td>
			<td>−3</td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>2</td>
			<td>−5/2</td>
			<td>9</td>
			<td>−12</td>
			<td>7</td>
			<td>−3/2</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td rowspan="2">4 </td>
			<td>1</td>
			<td>1</td>
			<td>−4</td>
			<td>6</td>
			<td>−4</td>
			<td>1</td>
			<td>&nbsp;</td>
		</tr>
		<tr>
			<td>2</td>
			<td>3</td>
			<td>−14</td>
			<td>26</td>
			<td>−24</td>
			<td>11</td>
			<td>−2</td>
		</tr>
	</tbody>
</table>

<!-- #endregion -->

**Backward one-sided finite difference formulas:**

<table class="wikitable" style="text-align:center">
	<tbody>
		<tr>
			<th>Derivative </th>
			<th>Accuracy </th>
			<th>−5 </th>
			<th>−4 </th>
			<th>−3 </th>
			<th>−2 </th>
			<th>−1 </th>
			<th>0 </th>
		</tr>
		<tr>
			<td rowspan="2">1 </td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>−1</td>
			<td>1 </td>
		</tr>
		<tr>
			<td>2</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>1/2</td>
			<td>−2</td>
			<td>3/2 </td>
		</tr>
		<tr>
			<td rowspan="2">2 </td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>1</td>
			<td>−2</td>
			<td>1 </td>
		</tr>
		<tr>
			<td>2</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>−1</td>
			<td>4</td>
			<td>−5</td>
			<td>2 </td>
		</tr>
		<tr>
			<td rowspan="2">3 </td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>&nbsp;</td>
			<td>−1</td>
			<td>3</td>
			<td>−3</td>
			<td>1 </td>
		</tr>
		<tr>
			<td>2</td>
			<td>&nbsp;</td>
			<td>3/2</td>
			<td>−7</td>
			<td>12</td>
			<td>−9</td>
			<td>5/2 </td>
		</tr>
		<tr>
			<td rowspan="2">4 </td>
			<td>1</td>
			<td>&nbsp;</td>
			<td>1</td>
			<td>−4</td>
			<td>6</td>
			<td>−4</td>
			<td>1 </td>
		</tr>
		<tr>
			<td>2</td>
			<td>−2</td>
			<td>11</td>
			<td>−24</td>
			<td>26</td>
			<td>−14</td>
			<td>3 </td>
		</tr>
	</tbody>
</table>



Again, you should recognize the one-sided formulas we described in the previous notebook for the first-order derivative.


## Functions

...


## Matrix formulation

...

## Summary


...

```python
from IPython.core.display import HTML
css_file = '../Styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```python

```
