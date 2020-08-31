---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.5.2
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Time integration - Part 2

In the previous notebook we have considered the forward Euler scheme to time march ordinary differential equations. We have discussed its accuracy and stability with respect to the size of the time step.

Here we introduce some more accurate methods with larger domains of stability that are therefore applicable in more complex situations. We only consider systems of first order differential equations as problems containing higher order derivatives may be reduced to such systems. Our system of equation thus reads,

\begin{align}
    y'(t)=f(t,y)
\end{align}

where $y$ and $f$ are vector valued functions.

## A two stage Runge-Kutta scheme

The forward Euler method is defined through:

\begin{align}
    & y^{n+1} \equiv y^n + f(t,y^n) dt & (\textrm{Forward Euler method}) \label{eq:ForwardEuler3},
\end{align}

with all the intermediate times denoted $t^n = t+ndt$ and the corresponding values of $y(t)$ as $y^n = y(t^n)$.

Graphically, we see that $y^{n+1}$ is evaluated using the value $y^n$ and the slope (derivative) of $y$ at time $t^n$:

<img src="figures/Euler.png" align="center" width="500">

Runge-Kutta schemes increase the accuracy of the estimated value $y^{n+1}$ by introducing intermediate instants between $t^n$ and $t^{n+1}$ at which the derivative of $y$ is evaluated and by making use of this information.

The following two stage Runge-Kutta method is the simplest of such schemes. Graphically, this scheme is defined as follows:

<img src="figures/RK2.png" align="center" width="500">

so that,

\begin{align}
    y^* = y^n +\frac{dt}{2}f(t^n, y^n) \\
    y^{n+1} = y^n + dt f(t+\frac{dt}{2},y^*) \label{RK2ynp1}
\end{align}




In notebook 1.1, we discussed the fact that the forward Euler method is second order accurate for one time step and first order accurate for a complete time interval. Here, we definitely hope to improve these orders of accuracy. Let's check that this is indeed the case by using the Taylor expansion of $y^{n+1}$ in $\eqref{RK2ynp1}$.

\begin{align}
 y^{n+1} & = y^n + dt f(t^n+\frac{dt}{2},y^n + \frac{dt}{2}f(y^n)) \\
         & = y^n + dt \left[ f(t^n, y^n) + \frac{dt}{2} \partial_t f(t^n, y^n) + \frac{dt}{2} f(t^n, y^n)\partial_y f(t^n,y^n) + O(dt^2) \right] \\
         & = y^n + dt y^{'n} + \frac{dt^2}{2}y^{''n} + O(dt^3), \label{TaylorRK2}
\end{align}

where we have used the property: $y''=\partial_t f + f\partial_y f$. Eq. \ref{TaylorRK2} proves that the two stage Runge-Kutta method is third-order for one time step and as a consequence it is second-order accurate for a complete time interval (we leave it as an exercise to show that this two stage Runge-Kutta scheme does not match further the Taylor expansion of $y^{n+1}$ and is therefore not of higher order accuracy). 


Let us now discuss the stability of this two stage Runge-Kutta method for a general linear system of equations. As usual, we may diagonalize the system defined through the matrix $f$ and write:

\begin{align}
    z' = \Lambda z,
\end{align}

where $\Lambda$ is the diagonal matrix composed of the eigenvalues $\lambda_k$ of $f$.

Using the two stage Runge-Kutta scheme we then have,

\begin{align}
    z^{n} = (I+ dt \Lambda + \frac{dt^2}{2}\Lambda^2) z^{n-1} \; \Leftrightarrow \; z^{n} = (I+ dt \Lambda + \frac{dt^2}{2}\Lambda^2)^n z^0.
\end{align}

All the components of $z^{n}$ will remain finite for $n\rightarrow \infty$ as long as the following relation is satisified for all the eigenvalues $\lambda_k$:

\begin{align}
    \vert 1+\lambda_k dt + \frac{\lambda_k^2 dt^2}{2} \vert < 1.
\end{align}




## General Runge-Kutta schemes


General Runge-Kutta schemes are defined as follows \cite{Hairer1987}:

\begin{align}
 y^{n+1} &= y^n + dt(b_1 k_1 + \cdots + b_s k_s) \nonumber \\
 k_1 & = f(t^n, y^n) \nonumber \\
 k_2 & = f(t^n + c_2 dt, y^n + dta_{21}k_1) \nonumber \\
 k_3 & = f(t^n + c_3 dt, y^n + dt(a_{31}k_1+a_{32}k_2)) \nonumber \\
 & \cdots \nonumber \\
 k_s & = f(t^n + c_s dt, y^n + dt(a_{s1}k_1+\cdots + a_{s,s-1}k_{s-1})) \nonumber
\end{align}

Some constraints are then put on all the coefficients to achieve a given order accuracy $O(dt^p)$ for $y^{n+1}$. One then says that the $s$-stage Runge-Kutta method is of order $p$.


The construction of higher order Runge-Kutta schemes is in fact quite complicated and has been the subject of a vast literature (some in depth review of the Runge-Kutta methods may be found in \cite{Butcher2008} or \cite{Hairer1987}). There are in fact no systematic ways of obtaining order $p$ methods with a minimum number of stages $s$. Up to $p=4$ one can achieve order $p$ with $s=p$. For $p=5$ and $p=6$ one needs at least $s=p+1$ stages. For $p=7$ and $p=9$ the minimum number of stages are respectively $s=9$ and $s=11$. "*Above this, very little is know*" \cite{Butcher1996}.

Here we therefore focus our attention on a general purpose fourth order Runge-Kutta scheme that is accurate and stable enough for all the problems we consider in the rest of this course (from now on we refer to it as RK4). It was introduced in 1901 by W. Kutta and reads \cite{Kutta1901}:

\begin{align}
    y^{n+1} & = y^n + \frac16 k_1 + \frac13(k_2 + k_3) + \frac16 k_4 \nonumber \\
    k_1 & = dtf(t^n,y^n) \\
    k_2 & = dtf(t^n+\frac{dt}{2},y^n+\frac12 k_1) \\
    k_3 & = dtf(t^n+\frac{dt}{2},y^n+\frac12 k_2) \\
    k_4 & = dtf(t^n+dt,y^n+k_3)
\end{align}

For an autonomous linear system, it is straightforward to prove that this method is indeed fourth order accurate. After diagonalisation, we have:

\begin{align}
  z^{n+1} & = z^n + \frac16 dt \Lambda z^n + \frac13 dt \Lambda (z^n + \frac12 dt \Lambda z^n ) + \frac13 dt \Lambda (z^n + \frac12 dt \Lambda (z^n + \frac12 dt \Lambda z^n)) \nonumber \\
  & \;\; \;+\frac16 dt \Lambda (z^n + dt \Lambda (z^n + \frac12 dt \Lambda (z^n + \frac12 dt \Lambda z^n))) \nonumber \\
  & = z^n + dt \Lambda z^n + \frac{dt^2}{2}\Lambda^2 z^n + \frac{dt^3}6 \Lambda^3 z^n + \frac{dt^4}{24} \Lambda^4 z^n
\end{align}

The last expression coincides with the Taylor expansion of $z^{n+1}$ up to fourth-order.

In terms of stability, we also see that the RK4 method is stable for a general n autonomous linear system as long as all the eigenvalues of the operator $f$ satisfy,

\begin{align}
    \vert 1+\lambda_k dt + \frac{\lambda_k^2 dt^2}{2} + \frac{\lambda_k^3 dt^3}{6} + \frac{\lambda_k^4 dt^4}{24}\vert < 1.
\end{align}

In the following plot, we compare the regions of stability for the various schemes we have already discussed:

```python
# Plot to do
from scipy import optimize
import matplotlib.pyplot as plt
```

```python

def f(x):
    return (x**2-1)
```

```python
root = optimize.newton(f, 1.5)
```

```python
root
```

```python

```
