© B. Knaepen & Y. Velizhanina</span>

# Numerical methods for partial differential equations

by Bernard Knaepen & Yelyzaveta Velizhanina

The notebooks contained in this course can be executed on a local machine after installing the required set of Python packages and the distributed version-control system Git. The full set of instructions to complete these two tasks is available by clicking on the following link:

[01_01_ToolkitSetup.md][00]

The reader familiar with Python environment setup, usage of Git and jupyter extensions may skip reading the above instructions and simply install the Python dependencies listed in [requirements.txt][01] using pip:

    pip install -r requirements.txt

Afterwards, to get a local copy of the *solving_pde_mooc* repository, first copy the URL of the remote repo

<img src="./notebooks/figures/cloneRepo.png" width="500">

On your computer, open a terminal window and navigate to a folder in which you would like to store the course material. In that folder, run the git clone command as follows:

    git clone https://github.com/aquaULB/solving_pde_mooc.git

This will automatically create and initialize a local git repository *solving_pde_mooc*  with the local branch *master* tracking the remote branch *origin/master*.

To see the content of a notebook, open a terminal window in the folder *solving_pde_mooc* and type:

    jupyter-notebook

This will open a window in your web browser in which you can open any notebook by clicking on its name. If the content of the notebook does not display properly, make sure that you have activated the appropriate Python environment as explained in the instructions.

The reader not familiar with the usage of jupyter-notebook should read a tutorial such as this one:

[Jupyter Notebook: An Introduction][02]

[00]: https://github.com/aquaULB/solving_pde_mooc/blob/master/notebooks/01_Introduction/01_01_ToolkitSetup.md

[01]: https://github.com/aquaULB/solving_pde_mooc/blob/master/requirements.txt

[02]: https://realpython.com/jupyter-notebook-introduction/