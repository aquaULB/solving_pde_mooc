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

What is the value of the number $\pi$? By definition, it is the ratio of any circle's circumference to its diameter. Because $\pi$ cannot be written as a simple fraction like $\frac{a}{b}$, one says that it is an irrational number and to write its exact value explicitely, an infinite number of digits after the decimal point is needed. As a computer can only store a finite quantity of information, even this ubiquitus number needs to approximated to fit in its memory: some digits have the decimal point have to be dropped and a finite precision representation of $\pi$ is needed.

## Taylor's theorem

Any (well behaved) function can be approximated using what is known as its Taylor expansion:

\begin{align}
f(x)=f(a)+f'(a)(x-a)+\frac{f''(a)}{2!}(x-a)+\dots + \frac{f^{(k)}(a)}{k!}(x-a)^k + o(\vert x-a \vert^k)
\end{align}

In the above formula, $f^{(k)}$ denotes the $k$-th derivative of $f$.
