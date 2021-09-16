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

+++ {"toc": true}

<h1>Table of Contents<span class="tocSkip"></span></h1>
<div class="toc"><ul class="toc-item"><li><span><a href="#Tools-we-will-rely-on" data-toc-modified-id="Tools-we-will-rely-on-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Tools we will rely on</a></span></li><li><span><a href="#Installation" data-toc-modified-id="Installation-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Installation</a></span><ul class="toc-item"><li><span><a href="#Install-Git" data-toc-modified-id="Install-Git-2.1"><span class="toc-item-num">2.1&nbsp;&nbsp;</span>Install Git</a></span></li><li><span><a href="#Get-the-course-notes" data-toc-modified-id="Get-the-course-notes-2.2"><span class="toc-item-num">2.2&nbsp;&nbsp;</span>Get the course notes</a></span></li><li><span><a href="#Install-Anaconda" data-toc-modified-id="Install-Anaconda-2.3"><span class="toc-item-num">2.3&nbsp;&nbsp;</span>Install Anaconda</a></span></li></ul></li><li><span><a href="#Working-with-Jupyter-Notebook" data-toc-modified-id="Working-with-Jupyter-Notebook-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Working with Jupyter Notebook</a></span></li></ul></div>

+++

<div class="copyright" property="vk:rights">&copy;
  <span property="vk:dateCopyrighted">2020</span>
  <span property="vk:publisher">B. Knaepen & Y. Velizhanina</span>
</div>

# Toolkit Setup

## Tools we will rely on

* **Python 3**
  > ... clear and powerful [object-oriented programming language][1], comparable to Perl, Ruby, Scheme, or Java.

  Python has (relatively) **simple** and **elegant syntax**, is **free** and **open-sourse** and (relatively) **easy to debug** (due to being an [interpreted programming language][4]).

  Python is largely popular but still, it cannot compete in performance with compiled, statically typed programming languages, such as C++ or Fortran, when addressing certain tasks. Nevertheless, it is been widely used by C++ developers for prototyping.

* **Anaconda**
  > ... free, easy-to-install [package manager, environment manager, and Python distribution][2] with a collection of 1,500+ open source packages with free community support.

  Anaconda has been very useful for Python developers. First of all, it comes with a distribution of Python 3 and allows you an easy switch between its version. Second, by installing Anaconda you get a conda package manager. Using conda you can install most of Python packages with a simple one-liner command.
* **Git**
  > ... is a free and open source distributed [version control system][3] designed to handle everything from small to very large projects with speed and efficiency.

  Git is used by developers to easily track and share their work. Whenever you introduce any changes to your project they are tracked by git, and you can revert to any of the previous states at any moment. Moreover, you can create "branches" - different versions of your project at the same time. Branches are commonly used, so that multiple people can work on the same files at the same time. Besides, usage of branches allows you to keep the main workflow *clean* by storing your work in progress separately untill you're sure it can be merged into the project.

  *In this course we use Git to maintain the repository with the course notes.* You will mostly use Git to stay synchronized with the state of the repo. Nevertheless, we encourage you to learn basic usage of Git to work with your own projects, as it would be an extremely useful skill for you to acquire.

## Installation

*In this course you are expected to run basic commands via command line. In case you don't know how to open command prompt, read how to do this on [MacOS, Linux or Windows][6].*

### Install Git

* **MacOS**

  If you are running *10.9 Mavericks or a later version of MacOS*, launch the command line and run:

      git --version
  It will start the installation of Git.

  If you are running an earlier version of MacOS, please visit [official website][7] for instructions.
* **Windows**

  Download Git from the [official website][8] and install it via graphical installer.
* **Linux**

  Open command line and run:

      apt-get install git

To check that Git was properly installed and its version, run from the command line:

    git --version

### Get the course notes

The material of this course is available online and can be alternatively accessed [on GitHub][10] or [on a dedicated webpage][11]. The second option is recommended for an easy navigation through the different chapters. However, the content is static and not suited to experiment with the various concepts we discuss. For that purpose, each chapter of the course is also distributed as a collection of Jupyter notebooks [on GitHub][10].

>The Jupyter Notebook is an open-source web application that allows you to create and share documents that contain live code, equations, visualizations and narrative text.

In a Jupyter Notebook coding is easy and interactive. In a single file you can combine [Markdown][12] *cells* and code *cells*.

*You should download the course notes on your computer by following these few steps:*

1. Create your [GitHub][13] account in case you don't have one.

2. Go to the [GitHub page][10] of the course.

3. Click the green "Code" button and copy the HTTP address of this repository.

<img src="../figures/code_copy.png">

4. Open the command line.

5. **Optional:** if you want the course notes to reside at specific location on your computer, change the directory first.

6. Run from the command line:

        git clone https://github.com/aquaULB/solving_pde_mooc.git

    *You will be asked to enter your GitHub credentials.*

A directory named `solving_pde_mooc` will be created in your current folder. Locally, you'll have access to all of the content that is available on GitHub.

*Note that `git clone` does much more than just copies the files in their current state.* It in fact does what it says - clones, creates exact duplicate of the *remote* repository (that on GitHub). It means that you will have full access to all of the changes that will appear on GitHub. We will guide you through how to download those changes further in the course.

### Install Anaconda

*In this course we ask you to install Anaconda and provide you a recipe to install all the required packages in one click*.

For the installation of Anaconda, we refer you to [Anaconda's website][5].

You will now be guided through the setup of the working environment that you will use throughout the course. To isolate it from your other projects that might also use Anaconda, we will use a dedicated **conda environment** -
> ... [a directory that contains a specific collection of conda packages that you have installed][9]... If you change one environment, your other environments are not affected. You can easily activate or deactivate environments, which is how you switch between them.

Open the command line and make sure you are "located" in the root directory of the course (`path/to/the/course/directory/solving_pde_mooc`). Run the following command:

    conda env create -f environment.yml

This will create a conda environment named "course" and launch the installation of multiple packages. This step only has to be done once. After that, you may activate this specific environment by typing in any terminal window,

```
conda activate course
```

At any time you can deactivate the environment by typing:

```
conda deactivate
```

This will switch you back to the base Anaconda environment but you can return to the "course" environment by activating it again.

## Working with Jupyter Notebook

We recommend that you use Jupyter notebooks to work with the course material, as it includes sample codes along with the relevant explanations.

Before trying to view the notebooks in Jupyter Notebook window, activate the conda environment *course* , as the notebooks rely on some of the newly installed packages. Launch the Jupyter Notebook application with the following command line command:

```
jupyter notebook
```

The course's notebooks require some Jupyter Notebook extensions to display properly. In the extension menu check out the following boxes:

<img src="../figures/jupextensions.png">

After performing all the steps above, you will have a working setup to view and explore all the content provided in this course.

[1]: <https://wiki.python.org/moin/BeginnersGuide/Overview> "What is Python"
[2]: <https://docs.anaconda.com> "Anaconda"
[3]: <https://git-scm.com> "Git"
[4]: <https://www.freecodecamp.org/news/compiled-versus-interpreted-languages/> "Interpreted vs compiled"
[5]: <https://docs.anaconda.com/anaconda/install/> "Anaconda installation"
[6]: <https://towardsdatascience.com/a-quick-guide-to-using-command-line-terminal-96815b97b955> "How to open command line"
[7]: <https://git-scm.com/download/mac> "Git Mac Download"
[8]: <https://git-scm.com/download/win> "Git Win Download"
[9]: <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html> "conda environment"
[10]: <https://github.com/aquaULB/solving_pde_mooc> "GitHub course notes"
[11]: <https://aquaulb.github.io/book_solving_pde_mooc/solving_pde_mooc/notebooks/01_Introduction/01_00_Preface.html> "Webbook course notes"
[12]: <https://en.wikipedia.org/wiki/Markdown> "Markdown"
[13]: <https://github.com> "GitHub"
