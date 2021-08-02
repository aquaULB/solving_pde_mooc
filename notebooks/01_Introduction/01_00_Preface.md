---
jupytext:
  formats: ipynb,md:myst
  notebook_metadata_filter: toc
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.10.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
toc:
  base_numbering: 1
  nav_menu: {}
  number_sections: true
  sideBar: true
  skip_h1_title: true
  title_cell: Table of Contents
  title_sidebar: Contents
  toc_cell: true
  toc_position: {}
  toc_section_display: true
  toc_window_display: false
---

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>

# Numerical methods for partial differential equations

<h1 style="font-size:18pt;  ">by Bernard Knaepen & Yelyzaveta Velizhanina<span class="tocSkip"></span></h1>

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Philosophy-of-the-course" data-toc-modified-id="Philosophy-of-the-course-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Philosophy of the course</a></span></li><li><span><a href="#Outline" data-toc-modified-id="Outline-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Outline</a></span></li><li><span><a href="#Tools" data-toc-modified-id="Tools-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Tools</a></span></li></ul></div>

+++

## Philosophy of the course

+++

Numerous scientific problems can only be addressed through modeling and computer analysis. This stems from the fact that a lot of these problems are complex and we cannot solve the corresponding models using only pen-and-paper.

Many models appearing in engineering or physical applications are mathematically described by partial differential equations (PDEs) and the aim of this course is to provide a practical introduction to the relevant tools to solve them using computer algortihms. By practical we mean that we will not develop the theory behind PDEs and also avoid complex mathematical derivations of the numerical methods we describe below. Although these are beautiful subjects, they are often characterized by a steep learning curve and our objective is to get our hands on solving real problems as quickly as possible. In some places we nevertheless provide some mathematical background about the techniques we describe because they cannot be properly understood otherwise. In each case, we try to keep things simple and written in such a way that the corresponding section may be skipped at first reading.

+++

## Outline

+++

By the end of the course, the student should have acquired the necessary skills to solve numerically equations such as the heat equation or Schrödinger's equation:  

<img style="display: block; margin-left: auto; margin-right: auto;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Balign*%7D%0A%5Cfrac%7B%5Cpartial%20T(%5Cboldsymbol%20r%2Ct)%7D%7B%5Cpartial%20t%7D%20%20%26%20%3D%20%5Calpha%20%5CDelta%20T(%5Cboldsymbol%20r%2Ct)%2C%20%26%5Chbox%7BHeat%20equation%7D%5C%5C%0Ai%5Chbar%20%5Cfrac%7B%5Cpartial%20%5CPsi(%5Cboldsymbol%20r%2Ct)%7D%7B%5Cpartial%20t%7D%20%26%20%3D%20%5Cleft%5B%20%5Cfrac%7B-%5Chbar%5E2%7D%7B2m%7D%5CDelta%20%2B%20V(%5Cboldsymbol%20r%2Ct)%20%5Cright%5D%5CPsi(%5Cboldsymbol%20r%2Ct)%2C%20%26%5Chbox%7BSchr%C3%B6dinger%20equation%7D%0A%5Cend%7Balign*%7D">

We don't further describing those equations here (it's not a problem if you have never heard about them), but we just emphasises that they both contain the laplacian operator

<img style="display: block; margin-left: auto; margin-right: auto;" src="https://render.githubusercontent.com/render/math?math=%5Cbegin%7Bequation*%7D%0A%20%5CDelta%20%3D%20%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20x%5E2%7D%2B%20%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20y%5E2%7D%2B%20%5Cfrac%7B%5Cpartial%5E2%7D%7B%5Cpartial%20z%5E2%7D%0A%5Cend%7Bequation*%7D%0A">

that makes them partial differential equations. Our main objective is then to formulate such operators numerically and to solve the resulting problems with computer algorithms. This also requires to complement the equations with suitable boundary conditions and to implement them consistently with the differential operators.

This course covers the following topics:

* **Integration of ordinary differential equations**

    Ordinary differential equations (ODEs), unlike partial differential equations, depend on only one variable. The ability to solve them is essential because we will consider many PDEs that are time dependent and need generalizations of the methods developped for ODEs.
    
* **Finite difference schemes**

    In this chapter we describe the most intuitive way of computing spatial derivatives. It heavily relies on the Taylor's expansion of functions and how we can use the information in the neighborhood of a point to compute derivatives at that point.
    
* **Resolution of partial differential equations**

    This part of the course combines the concepts discussed in the first two chapters and describes through examples how they can be used to solve PDEs.
    
* **Iterative methods**
    
    Many of the techniques involved in solving ODEs or PDEs require to invert matrices. For low dimensional matrices, this can be done through standard elimination methods. For large matrices, such methods are prohibitively expensive and some approximate inversion techniques are needed. In this chapter we describe some of them belonging to the class of iterative methods.

* **Fourier transform**
    
    In the chapter about finite difference schemes we have described an intuitive way to compute spatial derivatives. Here we introduce a more accurate technique that relies on the expansion of the unknown functions using a basis of functions. We illustrate the concepts introduced to solve problems with periodic boundary conditions.
    
* **Chebyshev discretisation**
    
    We conclude this course by giving a brief introduction on the Chebyshev spectral method. It also relies on the concept of Fourier transforms but the expansion basis is constructed using Chebyshev polynomials. This method is very accurate and well adapted to problems with physical boundaries as opposed to problems with periodic boundary conditions.


<h2 id="Tools">Tools</h2>

+++

The numerical resolution of PDEs obviously requires the use of a programming language and optionally some previously developped software packages. For this course we have decided to use the Python programming language and some of the SciPy (Scientific computing tools for Python) packages. In the next notebook - 00_01_ToolkitSetup - we say a few more words about this and also provide a complete set of instructions to install all the necessary tools required to get started with the course.

```{code-cell} ipython3
from IPython.core.display import HTML
css_file = '../styles/notebookstyle.css'
HTML(open(css_file, 'r').read())
```
