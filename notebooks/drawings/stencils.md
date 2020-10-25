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
    display_name: 'Python 3.7.7 64-bit (''course'': conda)'
    language: python
    name: python37764bitcoursecondacf09dcbfbf8b41f7b84691ece6539f67
---

<!-- #region toc=true -->
<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Basic-sample" data-toc-modified-id="Basic-sample-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Basic sample</a></span></li><li><span><a href="#Taylor's-theorem" data-toc-modified-id="Taylor's-theorem-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Taylor's theorem</a></span></li><li><span><a href="#Forward-first-order-diff" data-toc-modified-id="Forward-first-order-diff-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Forward first-order diff</a></span></li><li><span><a href="#Backward-first-order-diff" data-toc-modified-id="Backward-first-order-diff-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Backward first-order diff</a></span></li><li><span><a href="#Centered-first-order-diff" data-toc-modified-id="Centered-first-order-diff-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Centered first-order diff</a></span></li><li><span><a href="#One-sides-first-order-diff" data-toc-modified-id="One-sides-first-order-diff-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>One sides first-order diff</a></span></li><li><span><a href="#One-sides-first-order-diff-(backward)" data-toc-modified-id="One-sides-first-order-diff-(backward)-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>One sides first-order diff (backward)</a></span></li><li><span><a href="#Centered-second-order-diff" data-toc-modified-id="Centered-second-order-diff-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Centered second-order diff</a></span></li><li><span><a href="#Centered-fourth-order-diff" data-toc-modified-id="Centered-fourth-order-diff-9"><span class="toc-item-num">9&nbsp;&nbsp;</span>Centered fourth-order diff</a></span></li><li><span><a href="#Backward-first-order-diff---periodic-BC" data-toc-modified-id="Backward-first-order-diff---periodic-BC-10"><span class="toc-item-num">10&nbsp;&nbsp;</span>Backward first order diff - periodic BC</a></span></li></ul></div>
<!-- #endregion -->

```python
import numpy as np

import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('../styles/style.use')
```

## Basic sample

```python
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,-3,-1]], y[[0,-3,-1]])

ax.plot(x[2:8:2], y[2:8:2], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)
ax.text(2, 0.5, r'$i-1$', fontsize=18, horizontalalignment='center')
ax.text(4, 0.5, r'$i$', fontsize=18, horizontalalignment='center')
ax.text(6, 0.5, r'$i+1$', fontsize=18, horizontalalignment='center')

fig.savefig('../figures/sample.png', dpi=300)
```

## Taylor's theorem

```python
fig, ax = plt.subplots(figsize=(10, 2))
ax.set_axis_off()
ax.axis('equal')

# edges
x=np.array([0, 10])
y = 0.0 * x

# edges
ax.plot(x, y, marker='|')
ax.text(0, 0.5, r'$a$', fontsize=18, horizontalalignment='center')
ax.text(10, 0.5, r'$b$', fontsize=18, horizontalalignment='center')

# inside
x=np.array([5, 5.8, 7])
y = 0.0 * x

ax.plot(x, y, linestyle='')
ax.text(5, 0.5, r'$x$', fontsize=18, horizontalalignment='center')
ax.text(5.8, 0.5, r'$\xi$', fontsize=18, horizontalalignment='center')
ax.text(7, 0.5, r'$x+\Delta x$', fontsize=18, horizontalalignment='center')

fig.savefig('../figures/taylor.png', dpi=300)
```

## Forward first-order diff

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,2,-3,-1]], y[[0,2,-3,-1]])

ax.plot(x[4:8:2], y[4:8:2], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(4, 0.5, '$f\'_i$', fontsize=18, horizontalalignment='center')
ax.text(4, -0.6, r'$-f_i$', fontsize=18, horizontalalignment='center')
ax.text(6, -0.6, r'$f_{i+1}$', fontsize=18, horizontalalignment='center')
ax.text(11, -0.6, r'$/\Delta$', fontsize=18, horizontalalignment='center')



fig.savefig('../figures/forwardDiff1.png', dpi=300)
```

## Backward first-order diff

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,6,-3,-1]], y[[0,6,-3,-1]])

ax.plot(x[2:6:2], y[2:6:2], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(4, 0.5, '$f\'_i$', fontsize=18, horizontalalignment='center')
ax.text(4, -0.6, r'$f_i$', fontsize=18, horizontalalignment='center')
ax.text(2, -0.6, r'$-f_{i-1}$', fontsize=18, horizontalalignment='center')
ax.text(11, -0.6, r'$/\Delta$', fontsize=18, horizontalalignment='center')



fig.savefig('../figures/backwardDiff1.png', dpi=300)
```

## Centered first-order diff

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,4,-3,-1]], y[[0,4,-3,-1]])

ax.plot(x[[2,6]], y[[2,6]], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(4, 0.5, '$f\'_i$', fontsize=18, horizontalalignment='center')
ax.text(2, -0.6, r'$-\frac12f_{i-1}$', fontsize=18, horizontalalignment='center')
ax.text(6, -0.6, r'$\frac12f_{i+1}$', fontsize=18, horizontalalignment='center')
ax.text(11, -0.6, r'$/\Delta$', fontsize=18, horizontalalignment='center')

fig.savefig('../figures/centeredDiff1.png', dpi=300)
```

## One sides first-order diff

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,6,8,10]], y[[0,6,8,10]])

ax.plot(x[[0,2,4]], y[[0,2,4]], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(0, 0.5, '$f\'_0$', fontsize=18, horizontalalignment='center')
ax.text(0, -0.6, r'$-\frac32 f_{0}$', fontsize=18, horizontalalignment='center')
ax.text(2, -0.6, r'$2 f_{1}$', fontsize=18, horizontalalignment='center')
ax.text(4, -0.6, r'$-\frac12 f_{2}$', fontsize=18, horizontalalignment='center')
ax.text(11, -0.6, r'$/\Delta$', fontsize=18, horizontalalignment='center')



fig.savefig('../figures/onesideDiff1.png', dpi=300)
```

## One sides first-order diff (backward)

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,2,4,10]], y[[0,2,4,10]])

ax.plot(x[[6,8,10]], y[[6,8,10]], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(10, 0.5, '$f\'_{n}$', fontsize=18, horizontalalignment='center')
ax.text(10, -0.6, r'$\frac32 f_{n}$', fontsize=18, horizontalalignment='center')
ax.text(8, -0.6, r'$-2 f_{n-1}$', fontsize=18, horizontalalignment='center')
ax.text(6, -0.6, r'$\frac12 f_{n-2}$', fontsize=18, horizontalalignment='center')
ax.text(11, -0.6, r'$/\Delta$', fontsize=18, horizontalalignment='center')



fig.savefig('../figures/onesideDiff1_2.png', dpi=300)
```

## Centered second-order diff

```python
fig, ax = plt.subplots(figsize=(10, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[[0,-3,-1]], y[[0,-3,-1]])

ax.plot(x[[2,4,6]], y[[2,4,6]], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(4, 0.5, '$f\'\'_i$', fontsize=18, horizontalalignment='center')
ax.text(2, -0.6, r'$f_{i-1}$', fontsize=18, horizontalalignment='center')
ax.text(4, -0.6, r'$-2f_{i}$', fontsize=18, horizontalalignment='center')
ax.text(6, -0.6, r'$f_{i+1}$', fontsize=18, horizontalalignment='center')
ax.text(11, -0.6, r'$/\Delta^2$', fontsize=18, horizontalalignment='center')



fig.savefig('../figures/centeredDiff2.png', dpi=300)
```

## Centered fourth-order diff

```python
fig, ax = plt.subplots(figsize=(12, 3))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 13)
y = 0.0 * x

ax.plot(x[[0,-1]], y[[0,-1]])

ax.plot(x[[2,4,6,8,10]], y[[2,4,6,8,10]], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(6, 0.5, '$f\'\'\'\'_i$', fontsize=18, horizontalalignment='center')
ax.text(2, -0.6, r'$f_{i-2}$', fontsize=18, horizontalalignment='center')
ax.text(4, -0.6, r'$-4f_{i-1}$', fontsize=18, horizontalalignment='center')
ax.text(6, -0.6, r'$6f_{i}$', fontsize=18, horizontalalignment='center')
ax.text(8, -0.6, r'$-4f_{i+1}$', fontsize=18, horizontalalignment='center')
ax.text(10, -0.6, r'$f_{i+2}$', fontsize=18, horizontalalignment='center')
ax.text(13, -0.6, r'$/\Delta^4$', fontsize=18, horizontalalignment='center')



fig.savefig('../figures/centeredDiff4.png', dpi=300)
```

## Backward first order diff - periodic BC

```python
fig, ax = plt.subplots(figsize=(10, 4))
ax.set_axis_off()
ax.axis('equal')

x=np.arange(0, 11)
y = 0.0 * x

ax.plot(x[2:], y[2:], marker='')
ax.plot(x[4::2], y[4::2], linestyle='')

ax.plot(x[0:4:2], y[2:6:2], linestyle='', markeredgecolor='#cc0000', markeredgewidth=2.5)

ax.text(2, 0.5, '$u\'_0$', fontsize=18, horizontalalignment='center')
ax.text(2, -0.6, r'$u_0$', fontsize=18, horizontalalignment='center')
ax.text(10, -0.6, r'$u_{-1}$', fontsize=18, horizontalalignment='center')
ax.text(8, -0.6, r'$u_{-2}$', fontsize=18, horizontalalignment='center')


ax.text(0, -0.6, r'$u_{ghost}$', fontsize=18, horizontalalignment='center')

ax.axhline(-1.5, 0.05, 0.777, linestyle='-', marker='', color='black')
ax.axvline(0.01, 0.15, 0.175, linestyle='-', marker='', color='black')
ax.axvline(8.05, 0.15, 0.175, linestyle='-', marker='', color='black')

ax.axhline(1.5, 0.23, 0.96, linestyle='-', marker='', color='black')
ax.axvline(2., 0.825, 0.85, linestyle='-', marker='', color='black')
ax.axvline(10.1, 0.825, 0.85, linestyle='-', marker='', color='black')

ax.text(4, -1.2, r'$L$', fontsize=18, horizontalalignment='center')
ax.text(6, 1., r'$L$', fontsize=18, horizontalalignment='center')

fig.savefig('../figures/backwardDiff1periodic.png', dpi=300)
```

```python

```
