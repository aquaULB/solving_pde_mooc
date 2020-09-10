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

# Contents

- [Contents](#contents)
  - [Python and why Python](#python-and-why-python)
- [<a name="GetToKnowAnaconda"> Get to know Anaconda </a>](#-get-to-know-anaconda-)
  - [<a name="InstallingAnaconda"> Installing Anaconda </a>](#-installing-anaconda-)
  - [<a name="SettingEnv"> Setting conda environment </a>](#-setting-conda-environment-)
- [<a name="GetToKnowGit"> Get to know Git </a>](#-get-to-know-git-)
  - [Installing Git](#installing-git)
  - [Basic usage](#basic-usage)
    - [Git repositories](#git-repositories)
      - [Creating local Git repository](#creating-local-git-repository)
      - [Creating remote Git repository](#creating-remote-git-repository)
- [Troubleshooting](#troubleshooting)

## Python and why Python

All the pieces of code written in this course are written in Python. However, we try to make the required prior knowledge of Python as little as possible and the reader is only expected to have a basic knowledge of any programming language and be familiar with concepts like variables, loops, conditional statements, functions etc. In fact, we also design the course in such a way that it can be viewed as a tutorial of Python's numerical tools like numpy, matplotlib, scipy to name of few. Each time we need a new Python functionality, we try to thoroughly document it so that the reader needs no prior knowledge of the different packages but rather learns to use them when progressing through the notebooks.

# <a name="GetToKnowAnaconda"> Get to know Anaconda </a>

## <a name="InstallingAnaconda"> Installing Anaconda </a>

> [Anaconda][1] is a free, easy-to-install package manager, environment manager, and Python distribution with a collection of 1,500+ open source packages with free community support. Anaconda is platform-agnostic, so you can use it whether you are on Windows, macOS, or Linux.

We will install Anaconda3, as it includes Python 3 - the latest stable version of Python. Installing Anaconda will automatically install the most useful Python packages - probably all that we will require throughout the course. Though, there is always a possibility to install additional packages from the Anconda repository.

This installation file has been created in Jupyter Notebook. Jupyter Notebook provides an interactive shell for the Python code, and is also included in basic Anaconda edition.

[1]: <https://docs.anaconda.com> "Anaconda"


Installation instructions are provided on the official [website][2], and are quite detailed. Click on this link and install the distribution on your work machine. This is a prerequisite to follow this course.

[2]: <https://docs.anaconda.com/anaconda/install/> "Installation"


After Anaconda has been installed, we are all set to run Jupyter Notebook. Anaconda comes together with *anaconda-navigator*, which provides a graphical interface to access Jupyter Notebook, QtConsole, Spyder etc. We recommend you learn how to work with Anaconda using terminal, as, first, it will improve your skills of using terminal, and second, allow you to easily use conda package manager, create conda environments etc.

First, let's figure out, how we start Jupyter Notebook through terminal.

If you're on **MacOS or Linux**, the .bashrc file on your machine must be updated with a path to anaconda3/bin during the installation, so that each time you start new terminal session, all the containings of anaconda3/bin are globally accesible. It means, that if you just type `jupyter notebook` in command line, Jupyter Notebook should start running.

If you're on **Windows**, then, after opening START menu after installation, you'll be able to see *Anaconda prompt*. This application will start a terminal session with the containings of anaconda3/bin also globally accesible.

In the case if `jupyter notebook` command is not recogmized after installation, refer to the [Troubleshooting section](#troubleshooting1).

[3]: <https://www.journaldev.com/41479/bashrc-file-in-linux> "Bashrc"


## <a name="SettingEnv"> Setting conda environment </a>


> A [conda environment][3] is a directory that contains a specific collection of conda packages that you have installed. For example, you may have one environment with NumPy 1.7 and its dependencies, and another environment with NumPy 1.6 for legacy testing. If you change one environment, your other environments are not affected. You can easily activate or deactivate environments, which is how you switch between them.

We propose that you create a conda environment for the present course, and start with installing an additional component in this environment, so that you'll have all necessary tools in an isolated space, and also learn about conda environments.

* First, create an environment with a specific version of Python

        conda create -n course python=3.8.5

  `-n` is an argument for `conda create`, which has to be followed by a name of your environment. You can replace `course` with any name you prefer. We also specify, that we want the version of Python 3.8.5.

* Next, activate your virtual environment:

        conda activate course

* Now, as virtual environment has been activated, you can use conda package manager to install necessary packages without specifying, that you want them in `course`.

        conda install numpy scipy matplotlib jupyter
        conda install -c conda-forge jupytext jupyter_contrib_nbextensions

  Here `-c` refers to channel.

  > [Conda channels][4] are the locations where packages are stored. They serve as the base for hosting and managing packages. Conda packages are downloaded from remote channels, which are URLs to directories containing conda packages. The conda command searches a default set of channels, and packages are automatically downloaded and updated from https://repo.anaconda.com/pkgs/.

  conda-forge project provides a remote conda channel distributing conda packages, which you sometimes cannot find in conda `default` channel.

  We install `jupytext` package, as it provides a possibility to save Jupyter Notebook as a simple Markdown file, and we also install `jupyter_contrib_nbextensions`, which contains collection of extension for Jupyter Notebook, in particular, support of LaTeX environments.

* Open Jupyter Notebook and activate necessary extensions.

<img src="../figures/jupextensions.png">

Now we are ready to go - we have a virtual environment, which includes all necessary Python packages, and we even activated extended support for LaTeX for our notebooks.

In order to quit virtual environment, run:

        conda deactivate

[3]: <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html> "conda environment"
[4]: <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html> "conda channels"

# <a name="GetToKnowGit"> Get to know Git </a>

## Installing Git

> [Git][5] is a free and open source distributed version control system designed to handle everything from small to very large projects with speed and efficiency.

This description might be hard to comprehend at the moment, but you will learn better what is Git while working with it. What you have to know at the moment, is that due to Git, you can easily track the changes made to the project at different stages and compare all of them to each other; you can recover any older version of your project; after all, you can have separate *branches*, each of which stores different version of your project without any conflict between them.

Before digging deeper into the Git toolkit, let's install it.

Downloading Git is OS-dependant.

* If you're on Windows, you can download Git from the [official website][6] and install it with graphical installer. If, during installation, you choose all the default options, you will have git available in your Anaconda prompt.

* If you're on MacOS

    * 10.9 Mavericks or later

      Simply run

          git --version

      and it will launch the download and installation.

    * Before 10.9 Mavericks

      You have several [options][7]. As it is stated, if you already have *Xcode* installed, then you already have git. If not, we propose you to stick to the *Homebrew* option, as it's the most practical way to go, probably.

        Homebrew package manager is very easy to use, and provides access to the plenty of useful stuff - Firefox, GNU compiler, Qt and many many others.

        To install Homebrew, visit [official website][8], and then just run:

            brew install git

  Voilà, you are good to go.

* If you are on Linux, you have even [more options][9]. We propose, you simply run

      apt-get install git

Check that Git is installed by running

    git --version

I am, for example, getting the following output:

    git version 2.24.3 (Apple Git-128)

## Basic usage

Now let's duscus the basic concepts of Git. For the better understanding we encourage you to create a temporary empty directory. We will make it our first Git project. Let's call it *test_git*.

So, we run:

    mkdir test_git
    cd test_git

### Git repositories

The whole project with all the history of changes and all the separate branches is stored in so-called *Git repository*. To create a repository, you, having Git installed, simply declare some directory containg your project a Git repository - initialize it. This actions creates a *local* Git repository, meaning that it only exist on your work machine.

#### Creating local Git repository

In order to initialize existing local directory as a Git repository, run:

    git init

You'll get the following output:

    Initialized empty Git repository in /path/to/test_git/.git/

As it is nicely explained in [here][10], once created, each local Git repo contains three abstract zones:

* Working space

  Includes all the components of your project in a current state.

* Staging area

  Imagine now you've introduced some modification to the state of your project, which is last one you saved. The files, which get modified, go to the staging area.

* Commit area

  Commiting your modification is a way to create a snapshot of the current, newly modified state of your project. Once you've commited your changes, the modified files go to the commit area.

Let's add a README.md file to our repo. If you're on MacOS/Linux:

    vi README.md

If you're on Windows:

    type nul > README.md

Now let's learn how to create commits. First, you tell what are the files you want to commit:

    git add README.md

If you run

    git add *

then all files, which are in a staging area, will be added for the further commit. It is not a very clean way to go, as you might commit the files, which get updated at each compilation, and don't need to be tracked, like those stored in .ipynb_checkpoints directory, or object files with extension .o, etc.

We are ready to commit. Each commit has to be supplemented with a message:

    git commit -m "First commit"

#### Creating remote Git repository

> To be able to collaborate on any Git project, you need to know how to manage your remote repositories. [Remote repositories][11] are versions of your project that are hosted on the Internet or network somewhere.

Another reason to keep your data in a remote repo is for the sake of safety. If your computer gets broken, stolen, you accidentally remove the project's folder, then you loose all the local data. But if you kept it in a remote repo and updated it regularly, then you recover your whole progress from any other machine.

There are different services which allow to create and manage remote Git repository. Throughout the course we will use GitHub.com. It is easy to use and free. Generally speaking, GitHub is a Git repository hosting service. Let's create our first remote repo on GitHub.

If you don't have GitHub account, you will have to go to GitHub.com and create one.

[5]: <https://git-scm.com> "Git"
[6]: <https://git-scm.com/download/win> "Git Win Download"
[7]: <https://git-scm.com/download/mac> "Git Mac Download"
[8]: <https://brew.sh> "Homebrew"
[9]: <https://git-scm.com/download/linux> "Git Linux Download"
[10]: <https://www.educative.io/edpresso/what-is-git?aid=5082902844932096&utm_source=google&utm_medium=cpc&utm_campaign=edpresso-dynamic&gclid=Cj0KCQjw-uH6BRDQARIsAI3I-UdhHN9Z0GJzbOHJxNHWZH-F4atUOf6VG4914ZYxmiU0gajSGIjUH8QaAlNhEALw_wcB> "Basic Git Tutorial"
[11]: <https://git-scm.com/book/en/v2/Git-Basics-Working-with-Remotes>

# Troubleshooting

* <a name="troubleshooting1"> `jupyter notebook` command is not recognized</a>

It most surely happens, because `jupyter notebook` command is supposed to run binary file "jupyter-notebook", which intends to run the program, but this binary file is located in a directory different from the one you are trying to open it.

In order to be able to run Jupyter Notebook from any directory, we have to run the following command:

    export PATH=/path/to/anaconda3/bin:$PATH

But this will make the files from anaconda3/bin accesible only in the active terminal window. Once you terminate it, you will have to rerun above command for each new terminal session. The solution is to add the path to anaconda3/bin to your .bashrc file.

> The [.bashrc file][3] is a script file that’s executed when a user logs in. The file itself contains a series of configurations for the terminal session. This includes setting up or enabling: coloring, completion, shell history, command aliases, and more.

So that each time you start new terminal session, this commans will execute. To open .bashrc with in-terminal text editor you can type:

    vi .bashrc

Or, if you prefer different text editor, you can use it.

If .bashrc isn't there, it is possible that it has a different name - .zshrc, for example.

Add `export PATH=/path/to/anaconda3/bin:$PATH` to your .bashrc and save it. To get it working either open new terminal window, or run `source .bashrc` in the one, which is opened.

Now `jupyter notebook` command should work.

```python

```
