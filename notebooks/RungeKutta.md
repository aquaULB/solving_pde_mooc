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

<!-- #region -->
# Time integration - Part 2

In the previous notebook we have considered the forward Euler scheme to time march ordinary differential equations. We have discussed its accuracy and stability with respect to the size of the time step.

Here we introduce some more accurate methods with larger domains of stability that are therefore applicable in more complex situations.


## Runge-Kutta schemes

Denoting our unknown function as $y(t)$, the forward Euler method is defined through the following relation:

\begin{align}
    & y^{n+1} \equiv y^n + y^{'n} dt & (\textrm{Forward Euler method}) \label{eq:ForwardEuler3},
\end{align}

with all the intermediate times denoted $t^n = t+ndt$ and the corresponding values of $y(t)$ as $y^n = y(t^n)$.

Graphically, we see $y^{n+1}$ is evaluated using the value $y^n$ and the slope (derivative) of $y$ at time $t^n$:


Runge-Kutta schemes increase the accuracy of the estimated value $y^{n+1}$ by introducing intermediate instants between $t^n$ and $t^{n+1}$ at which the derivative of $y$ is evaluated and by making use of this information.

The Runge-Kutta method of order 2 is the simplest of such schemes. Graphically, this scheme is designed as follows:

<img src="figures/RK2.png" width="600">

