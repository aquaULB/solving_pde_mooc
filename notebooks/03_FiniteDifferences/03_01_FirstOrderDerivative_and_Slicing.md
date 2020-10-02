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
2. [First-order derivative](#First-order-derivative)
3. [Python slicing](#Python-slicing)
4. [One-sided finite differences](#One-sided-finite-differences)
5. [Summary](#Summary)


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
f(x)=e^x \sin(3\pi x) \label{eq:testfunc}
$$

There as several conceptually different ways to do this. Following the same approach as for time integration, we can rely on Taylor's theorem to use the value of $f(x)$ at some neighbouring points of $x$. This approach relies on what are known as finite differences. Another way to compute derivatives relies on decomposing the function $f(x)$ on a basis of functions $T_k(x)$ and computing the derivatives of $f(x)$ from the known derivatives of $T_k(x)$. This method is known as the spectral method and will be described later on in the course.

Let $x$ be the continuous variable defined in the interval $x\in[x_0,x_n]$. In any numerical problem, we have to limit the number of points at which we store the values of $f(x)$ because the random access memory (RAM) of our computers is limited. We therefore need to introduce an approximation of our continuous interval - the numerical grid. It is a set of grid points at which we evaluate all physical quantities.

For simplicity consider a uniform grid in which the $n+1$ grid points are evenly distributed. Therefore the coordinates of the grid points are:

\begin{equation}
 x_i = i \Delta x, \; \; 0\leq i \leq n
\end{equation}

with the endpoints of the grid located respectively at $x_0$ and $x_n$.

We will refer to the continuous variable defined in $[x_0,x_n]$ by $x$, and to its discrete representation by $x_i$. $x_i$ is called a *grid point* or sometimes a *grid node*. The value of some function $f(x)$ at the grid point $x_i$ is then denoted as $f_i$.

Using these notations, the approximation of a derivative through a finite-difference expression is generically given by:

\begin{equation}
f^{(k)}_i = \frac{1}{(\Delta x)^k}\sum_{j\in\mathcal{J}}c_j f_i,\label{eq:generic}
\end{equation}

where $k$ represents the order of derivative, $\mathcal{J}$ is called the *stencil* - the group of points used to build the finite-difference approximation - and $c_j$ is the finite-difference coefficient defined at the grid point $j$ of the stencil.

## First-order derivative

As an example of how the finite-difference approximation for the derivative of particular order can be derived, let us consider first-order derivative of $f(x)$. According to Taylor's theorem, we can approximate $f(x+\Delta x)$ as follows:

\begin{equation}
f(x+\Delta x)= f(x)+f'(x)\Delta x+O(\Delta x^2)\label{TaylorSer}.
\end{equation}

Expression \ref{TaylorSer} is exact. We assume that $\Delta x \to 0$, and cut the terms that are of higher order in $\Delta x$ than $1$:

\begin{equation}
f(x+\Delta x)= f(x)+f'(x)\Delta x.
\end{equation}

We can express the first-order derivative of $f(x)$ at point $x$:

\begin{equation}
f'(x) = \frac{f(x+\Delta x) - f(x)}{\Delta x} \label{eq:forwardTaylorDiff1}.
\end{equation}

This expression is the usual left derivative of $f(x)$.

Let us now approximate \ref{eq:forwardTaylorDiff1} in the grid $x_0, x_1,\dots, x_i,\dots x_{n-1}, x_n$. In the nodal notations it reads:

\begin{equation}
f^{' \rm f}_i = \frac{f_{i+1} - f_i}{\Delta x},\;\; \hbox{forward finite difference} \label{eq:forwardNodal}.
\end{equation}

\ref{eq:forwardNodal} represents first-order accurate finite-difference approximation of $f'(x)$ at $x_i$. The stencil is given by the sequence $[0, 1]$, where 0 stands for the point at which derivative is being evaluated, and the corresponding finite-difference coefficients are $[-1, 1]$ (see \ref{eq:generic}).

In the following figure we illustrate the stencil points and mark those involved in the computation of $f'(x_i)$:

<img width="600px" src="../figures/forwardDiff1.png">

The indices below the grid axis stand for the indices of the grid nodes, and the indices above the grid axis stand for the stencil indices.

It is important to highlight that *enumeration of stencil has nothing to do with enumeration of the grid nodes*. Enumeration of grid nodes normally starts at one of the grid boundaries ($x_0$ in our cases) and ends at another boundary. Enumeration of stencil always defined for each particular approximation. Zero stencil point usually refers to that grid node at which the derivative is being approximated. The stencils indices then decrease to the left of this point and increase to its right.

In the above stencil, we use two grid points - $x_i$ and $x_{i+1}$ - to obtain a first-order accurate expression for the first-order derivative at $x_i$. It is obvious that the forward finite difference formula \ref{eq:forwardNodal} cannot be used at the right boundary node $x_n$. In section [One-sided finite differences](#One-sided-finite-differences), we discuss how the boundary nodes can be handled when the derivatives are being evaluated using finite differences.

Let us now define the backward finite-differences scheme in the identical manner. As Taylor's theorem is valid for $x$ in the interval $a < x-\Delta x \le x \le x+\Delta x < b$, we can approximate $f(x-\Delta x)$ as follows:

\begin{equation}
f(x-\Delta x) \approx f(x) - f'(x)\Delta x. 
\end{equation}

As a consequence, we get first-order accurate backward finite-differences formula for the first order derivative of $f(x)$ at $x_i$:

\begin{equation}
f^{'\rm b}_i = \frac{f_i - f_{i-1}}{\Delta x}, \;\; \hbox{backward finite difference}\label{eq:backwardNodal}. 
\end{equation}

It is based on the right derivative $f'(x)$. We indicate the stencil points used to build \ref{eq:backwardNodal} in red:

<img width="600px" src="../figures/backwardDiff1.png">

As the forward finite-difference approximation could not be used at the right boundary node $x_n$, the backward finite-difference approximation cannot be used at the left boundary node $x_0$. We also note that $f^{'\rm b}_{i+1} = f^{'\rm f}_i$.

Let us now derive higher order accurate approximation for $f'(x)$. Resorting again to Taylor's theorem we write:

\begin{align}
& f(x+\Delta x) \approx f(x)+f'(x)\Delta x+\frac12 f''(x)\Delta x^2\label{eq:leftTaylor2} \\
& f(x-\Delta x) \approx f(x)-f'(x)\Delta x+\frac12 f''(x)\Delta x^2\label{eq:rightTaylor2}.
\end{align}

We substract equations \ref{eq:leftTaylor2} and \ref{eq:rightTaylor2} and get:

\begin{equation}
f'(x) \approx \frac{f(x+\Delta x) - f(x-\Delta x)}{2\Delta x}+O(\Delta x^2), \label{eq:centeredTaylorDiff}
\end{equation}

which leads us to the second-oder accurate approximation of $f'(x)$ at $x_i$:

\begin{equation}
f^{' \rm c}_i = \frac{f_{i+1} - f_{i-1}}{2\Delta x},\;\; \hbox{centered finite difference} \label{eq:centeredDiff}.
\end{equation}

This expression is called the centered finite difference first-order derivative and its stencil looks like this:

<img width="600px" src="../figures/centeredDiff1.png">

Using just two grid points, it's not possible to achieve an accuracy of higher order than $2$. The centered finite-difference scheme cannot be used both at the left or right boundary nodes of the grid.

Let us check that our formulas are correct. We first create a fine grid to accurately represent the function \ref{eq:testfunc} and its derivative in the interval $x\in [O, \pi]$.

```python
pi = np.pi       # 3.14...
nx = 200         # number of grid points (fine grid)
lx = pi          # length of the invertal
dx = lx / (nx-1) # grid spacing
```

```python
x = np.linspace(0, lx, nx)   # coordinates in the fine grid
f = np.exp(x)*np.sin(3*pi*x) # function in the fine grid

# Let us build numpy array for the exact repre-
# sentation of first-order derivative of f(x).
dfdx = np.exp(x)*(np.sin(3*pi*x) + 3*pi*np.cos(3*pi*x))
```

We have built the numpy array for the exact expression for the first-order derivative of $f(x)$ from its analytical expression. But what if we worked with a function too compicated to derive analytically, or required expression for the higher order derivatives? It is useful to keep in mind the there is Python package just for that - for symbolic computations - [SymPy][20]. We won't get into details and leave it to you, to explore SymPy. Note that SymPy is not the part of basic distribution of Anaconda, you would have to install it. 

Sympy supports [basic symbolic calculus][21], and provides [tools][22] to transform symbolic data to numerical representation. 

[20]: <https://docs.sympy.org/latest/index.html> "SymPy"
[21]: <https://docs.sympy.org/latest/tutorial/calculus.html> "Basic calculus"
[22]: <https://docs.sympy.org/latest/modules/utilities/lambdify.html?highlight=lambdify> "From symbolic to numerical data"


For the lower cost of computations of finite-difference approximations, we build the coarse grid with $80$ points, and evaluated the derivative:

```python
# We don't care about overwriting grid variables,
# as we are not using them further than for con-
# struction of x and f(x) arrays.
nx = 80          # number of grid points (coarse grid)
lx = pi          # length of invertal
dx = lx / (nx-1) # grid spacing
```

```python
x_c = np.linspace(0, lx, nx)       # coordinates in the coarse grid
f_c = np.exp(x_c)*np.sin(3*pi*x_c) # function in the coarse grid

# We create containers for the forward, backward
# and centered finite difference points.
df_forward = np.empty(nx)
df_backward = np.empty(nx)
df_centered = np.empty(nx)

# We fill arrays in the Python loops, so that
# you can explicitely see what grid nodes have
# been taken into account.
for i in range(0, nx-1): # last grid point is omitted
    df_forward[i] = (f_c[i+1] - f_c[i]) / dx
    
for i in range(1, nx): # first grid point is omitted
    df_backward[i] = (f_c[i] - f_c[i-1]) / dx

for i in range(1, nx-1): # first and last grid points are omitted
    df_centered[i] = (f_c[i+1] - f_c[i-1]) / (2*dx)
```

Let us now plot the forward, backward and centered finite-difference approximations of first-order derivative of $f(x)$ against the curve obtained from the exact expression:

```python
fig, ax = plt.subplots(1, 3, figsize=(12, 5), tight_layout=True)

fig.suptitle('Forward, backward and centered finite differences vs exact derivative')

for axis in ax:
    axis.set_xlim(x[0], x[-1])
    
    axis.set_xlabel('$x$')
    axis.set_ylabel("f'")

ax[0].plot(x, dfdx)
ax[0].plot(x_c[0:nx-1], df_forward[0: nx-1], '^g')

ax[1].plot(x, dfdx)
ax[1].plot(x_c[1:nx], df_backward[1: nx], '^m')

ax[2].plot(x, dfdx)
ax[2].plot(x_c[1:nx-1], df_centered[1: nx-1], '^c')
```

What do you think about the agreement? What happens when you increase the number of points in the coarse grid?

In the above cell, we have used the slicing of numpy arrays to extract the relevant entries from our arrays. For example, for the forward finite difference, the expression is not defined at the last grid point. Therefore, the relevant grid coordinates are not the complete `x_c` array but the *slice* `x_c[0:nx-1]`. For the centered finite differences we must exclude the first and last grid points. The appropriate coordinate array slice is then `x_c[1:nx-1]`. The notions of slicing of Python sequences are described in much more detail in the next section.


## Python slicing


We already mentioned the powerful tool of Python - negative indexing. When the programmer tries to access elements of the sequence by referring to the negative index, the enumeration of the elements starts from the tail of the sequence. Let's say we have a Python list:

```python
a = [
    'first',
    'second',
    'third'
]
```

And we want to iterate through its elements starting from `a[2]` to `a[0]`. It is a valid and even preferrable approach to do it using the negative indexing:

```python
for i in range(-1, -4, -1):
    print(a[i])
```

Now that we are fully uquipped in terms of knoweledge about Python indexing, let's proceed to the Python slicing. Python slicing provides simple access to subsequences in Python sequences and spares programmers the neccessity to loop explicitely, as we would do in C++, for example. Moreover, Python slicing [is implemented in C][30], so, it's it considerably faster than the respective code implemented with a Python loop.

Syntax for the python slicing is the following `start:stop+step:step`, where `stop` indicates the last element of the sequence you want to "grasp".

Consider the following demo. First, we create large Python list.

[30]: <https://github.com/python/cpython/blob/master/Objects/sliceobject.c> "Slicing source"

```python
large_sequence = [i for i in range(10**5)]
```

Suppose we want to extract the sublist with the first element equal to `large_sequence[5]` and the last element equal to `large_sequence[99994]`. We could just create an empty list and fill it in a loop.

```python
%%timeit

i_was_filled_in_a_loop = []

for i in range(5, 99995):
    i_was_filled_in_a_loop.append(large_sequence[i])
```

But the more efficient way is to apply Python slicing:

```python
%timeit slice_of_it = large_sequence[5:-5]
```

Python slicing obviously executes faster. Try to avoid Python loops when you can slice instead.

When slicing as we did: `large_sequence[5:-5]`, we use Python slicing together with negative indexing. `5` stands for `start` and `-6` stands for `stop`. `Step` is not specified, so it is set to the default value - `1`. What if we have not specified `start` or `stop`, or if we have even omitted *any* indexing when slicing?

```python
# I'll add few elements to the a list for the
# sake of doing a demo.
#
# list.extend method differs from list.append.
# list.append adds an item as the last element
# of the list, while list.extend expects a se-
# quence as an argument and appends all its ele-
# ments to the list one by one.
#
# For more info:
# https://docs.python.org/3/tutorial/datastructures.html
a.extend(['fourth', 'fifth', 'sixth'])
```

* `start` is omitted

```python
# We pass few arguments to the print function.
# When they are outputted, they are separated by
# the value passed to the sep argument with de-
# fault value set to ' '.
print(a[:3], a[:5], sep='\n')
```

## One-sided finite differences

In section [First-order derivative](#First-order-derivative) we mentionned that the finite difference expressions provided cannot be applied at all grid points. The centered finite difference formula is not valid at both endpoints of the domain as it requires at those locations extra points not included in the domain. At the left boundary - $x_0$ - we could compute the first order derivative using the forward finite difference formula and at the right boundary - $x_n$ - we could use the backward finite difference formula. Throughout the domain we would then have:

\begin{cases}
& \displaystyle f'(x_0) = \frac{f(x_{1}) - f(x_0)}{\Delta x}, \\
& \displaystyle f'(x_i) = \frac{f(x_{i+1}) - f(x_{i-1})}{2\Delta x},\; \; 1 \leq i \leq n-1 \\
& \displaystyle f'(x_n) = \frac{f(x_{n}) - f(x_{n-1})}{\Delta x}.
\end{cases}

The inconvenience of this formulation is that it is second-order accurate for interior grid points but only first order at boundary nodes. It might not seem a big issue, but for certain problems the overall accuracy of the solution will be first order throughout the domain and not second order as we might have wished. We will give an example of this behaviour later on.

To improve our discretised operator, we have to find second-order accurate expressions for the boundary nodes and we can use Taylor's theorem to achieve this goal. At the left boundary node we have:

\begin{align*}
f(x_0 + \Delta) = f(x_0) +f'(x_0)\Delta x+\frac12 f''(x_0)\Delta x^2+O(\Delta x^3)\\
f(x_0 + 2\Delta) = f(x_0) +2f'(x_0)\Delta x+4\frac12 f''(x_0)\Delta x^2+O(\Delta x^3)
\end{align*}

If we multiply the first equation by two and then substract the second one we get:

\begin{equation*}
4 f(x_0 + \Delta) - f(x_0 + 2\Delta) = 3 f(x_0) + 2f'(x_0)\Delta x + O(\Delta x^3)
\end{equation*}

We can then define the following second-oder accurate approximation of $f'$ at $x_0$:

\begin{equation*}
f'_0 = \frac{-\frac32 f_0 + 2f_1 - \frac12 f_2}{\Delta x}
\end{equation*}

The stencil for this expression is represented as:

<img width="600px" src="../figures/onesideDiff1.png">

Similarly, the following expression constitutes a second-oder accurate approximation of $f'$ at $x_n$,

\begin{equation*}
f'(x_n) = \frac{\frac32 f_n - 2f_{n-1} + \frac12 f_{n-2}}{\Delta x}
\end{equation*}

and its stencil is:

<img width="600px" src="../figures/onesideDiff1_2.png">

We can now construct a second order discretized operator throughout the domain by using the above two expressions at the boundary nodes. Our complete computation of the second-order accurate first-order derivative then looks like (for the sake of completeness, we repeat the whole code here):

```python
nx = 80 # number of grid points (coarse grid)
lx = pi # length of invertal
dx = lx / (nx-1) # grid spacing
x_c = np.linspace(0, lx, nx) # coordinates of points on the coarse grid

f_c = np.exp(x_c)*np.sin(3*pi*x_c) # function on the coarse grid

df_2 = np.empty(nx) 
df_2[0] = (-3./2*f_c[0] + 2*f_c[1] - 1./2.*f_c[2]) / dx
df_2[-1] = (3./2*f_c[-1] - 2*f_c[-2] + 1./2.*f_c[-3]) / dx
df_2[1:-1] = (f_c[2:] - f_c[:-2]) / (2*dx)
```

```python
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(x, dfdx)
ax.plot(x_c, df_2, '^g')
ax.set_xlabel('$x$')
ax.set_ylabel('$f\'$')
```

## Summary


...

```python
from IPython.core.display import HTML
css_file = '../Styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```

```python

```
