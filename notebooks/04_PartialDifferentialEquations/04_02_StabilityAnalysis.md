---
jupyter:
  jupytext:
    formats: ipynb,md
    notebook_metadata_filter: toc
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.6.0
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

<h1 style="text-align: center">Partial differential Equation I</h1>

## Introduction

For convenience, we start with importing some modules needed below:

```python
import numpy as np
import matplotlib.pyplot as plt

%matplotlib inline

plt.style.use('../styles/mainstyle.use')
```

In the previous notebook, we have shown how to transform a partial differential equation into a system of coupled ordinary differential equations using semi-discretization.
