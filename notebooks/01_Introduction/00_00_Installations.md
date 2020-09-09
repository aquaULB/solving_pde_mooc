# Installing Anaconda


> [Anaconda][1] is a free, easy-to-install package manager, environment manager, and Python distribution with a collection of 1,500+ open source packages with free community support. Anaconda is platform-agnostic, so you can use it whether you are on Windows, macOS, or Linux.

We will install Anaconda3, as it includes Python 3.7 - the latest stable version of Python. Installing Anaconda will automatically install the most useful Python packages - probably all that we will require throughout the course. Though, there is always a possibility to install additional packages from the Anconda repository.

This installation file has been created in Jupyter Notebook. Jupyter Notebook provides an interactive shell for the Python code, and also included in basic Anaconda edition.

[1]: <https://docs.anaconda.com> "Anaconda0"


Installation instructions are provided on the official [website][2], and are quite detailed.

[2]: <https://docs.anaconda.com/anaconda/install/> "Installation"


After Anaconda has been installed, we are all set to run Jupyter Notebook. Anaconda comes together with *anaconda-navigator*, which provides graphical interface to access Jupyter Notebook, QtConsole, Spyder etc. We recommend you learn how to work with Anaconda using terminal, as, first, it will improve your skills of using terminal, and second, allow you to easily use conda package manager, create conda environments etc.

First, let's figure out, how we start Jupyter Notebook through terminal. Normally, if you just type
`jupyter notebook` in command line, Jupyter Notebook should open. But most probably if you try doing it right away, you'll get the similar error:

    zsh: command not found: jupyter notebook

It happens, because the above command is supposed to run binary file "jupyter-notebook", which intends to run the program, but this binary file is located in a directory different from the one you are trying to open it.

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

[3]: <https://www.journaldev.com/41479/bashrc-file-in-linux> "Bashrc"


# Setting conda environment


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

[3]: <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/environments.html> "conda environment"
[4]: <https://docs.conda.io/projects/conda/en/latest/user-guide/concepts/channels.html> "conda channels"

```python

```
