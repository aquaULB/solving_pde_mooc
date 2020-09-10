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
# Introduction

## Philosophy of the course

Numerous scientific problems can only be addressed through modeling and computer analysis. This stems from the fact that a lot of these problems are complex and we cannot solve the corresponding models using only pen-and-paper. 

Many models appearing in engineering or physical applications are mathematically described by partial differential equations (PDE) and the aim of this course is to provide a practical introduction to the relevant tools to solve them. By practical we mean that we will not develop the theory behind PDEs and also avoid complex mathematical derivations of the numerical methods we describe below. Although these are beautiful subjects, they are often characterized by a steep learning curve and our objective is to get our hands on solving real problems as quickly as possible. In some places we nevertheless provide some mathematical background about the techniques we describe because they cannot be properly understood otherwise. In each case, we try to keep things simple and written in such a way that the corresponding section may be skipped at first reading.

## Outline
By the end of the course, the student should have acquired the necessary skills to solve numerically equations such as the heat equation or Schrödinger's equation:

\begin{align*}
\frac{\partial T(\boldsymbol r,t)}{\partial t}  & = \alpha \Delta T(\boldsymbol r,t), &\hbox{Heat equation}\\
i\hbar \frac{\partial \Psi(\boldsymbol r,t)}{\partial t} & = \left[ \frac{-\hbar^2}{2m}\Delta + V(\boldsymbol r,t) \right]\Psi(\boldsymbol r,t), &\hbox{Schrödinger equation}
\end{align*}

We don't further describing those equations here (it's not a problem if you have never heard about them), but we just emphasises that they both contain the laplacian operator

\begin{equation*}
 \Delta = \frac{\partial^2}{\partial x^2}+ \frac{\partial^2}{\partial y^2}+ \frac{\partial^2}{\partial z^2}
\end{equation*}

that makes them partial differential equations. Our main objective is then to formulate such operators numerically and to solve the resulting problems with computer algorithms. This also requires to complement the equations with suitable boundary conditions and to implement them consistently with the differential operators.

This course covers the following topics:

- <u>*Integration of ordinary differential equations:*</u>
ordinary differential equations (ODEs), unlike partial differential equations, depend on only one variable. The ability to solve them is essential because we will consider many PDEs that are time dependent and need generalizations of the methods developped for ODEs 


- <u>*Finite difference schemes*:</u>
in this chapter we describe the most intuitive way of computing spatial derivatives. It heavily relies on the Taylor's expansion of functions and how we can use the information in the neighborhood of a point to compute derivatives at that point.


- <u>*Resolution of partial differential equations*:</u> this part of course combines the concepts discussed in the first two chapters and describes through examples how they can be used to solve PDEs.


- <u>*Iterative methods*:</u> 

- Fourier transform
- Chebyshev discretisation





## Python and why Python

All the pieces of code written in this course are written in Python. However, we try to make the required prior knowledge of Python as little as possible and the reader is only expected to have a basic knowledge of any programming language and be familiar with concepts like variables, loops, conditional statements, functions etc. In fact, we also design the course in such a way that it can be viewed as a tutorial of Python's numerical tools like numpy, matplotlib, scipy to name of few. Each time we need a new Python functionality, we try to thoroughly document it so that the reader needs no prior knowledge of the different packages but rather learns to use them when progressing through the notebooks.


<!-- #endregion -->

```python

```
