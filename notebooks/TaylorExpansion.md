---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.0
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Approximations and Taylor expansion

In general, the resolution of numerical problems require some approximations.

The first one is related to the fact that most real numbers need an infinite numbers of digits after the decimal point to be properly represented. To store these numbers in the memory of a computer one therefore needs to cut their representation at some point beyond the decimal point. The number of digits kept is called the precision of the representation. For example, in *single precision* and *double precision*, $\pi$ is given by the following approximations:

\begin{align}
\pi &= 3.1415927 \quad\quad &\text{(single precision)}\\
\pi &= 3.141592653589793 \quad\quad &\text{(double precision)}\\
\end{align}

In this course, we always use double precision for real numbers as this is the default precision used by Python. Such a precision is large enough for the kind of numerical problems we consider, but the reader should still be aware that rounding off errors can cause some difficulties as they can get amplified when certain operations are performed or when some iterative procedures are used. Two good references to get started on the subject are:

- https://docs.python.org/3/tutorial/floatingpoint.html
- https://floating-point-gui.de

In the context of the numerical discretisation of ordinary or partial diffential equations, the more significant limitation in precision usually comes from the limited computer resources available to solve a problem or the time needed to get the solution. Indeed, from the physical point of view, both time and space are continuous variables.

## Taylor's theorem

In order to estimate the accuracy of discretized differential operators or time integration schemes, Taylor's theorem provides a very convenient tool. Let $x$ be any point in the interval $[a\ b]$ and $\Delta x$ a small positive real number:

<img src="figures/taylor.png">

Any *well behaved* function can then be approximated using the following expression (Taylor's expansion):

\begin{align}
f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)+\dots + \frac{f^{(k)}(a)}{k!}(x-a)^k + O(\vert x-a \vert^{k+1})
\end{align}

In the above formula, $f^{(k)}$ denotes the $k$-th derivative of $f$. The symbol $o(\vert x-a \vert^k)$ stands for 

* mention how the error is reduced if one divides de grid size by 2 etc.

<img src="figures/sample.png">
