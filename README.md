# Description
This folder contains data and code to reproduce figures appearing in Nicolas & Boos (2023, J. Climate, in review) - *Understanding the spatiotemporal variability of tropical orographic rainfall using convective plume buoyancy*. In particular, code for the different orographic precipitation models appearing in the paper is available. 
Note: Code to produce Figure 1 is not included (it is a straightforward use of the [DRPy](https://github.com/dopplerchase/DRpy) package which is very well documented).

linear_precip_models_2D.py contains code to solve for all the orographic precipitation models (Smith & Barstad 2004, Nicolas & Boos 2022, upslope model)
mountainUtils.py mostly defines a class that is used to carry all data relative to a given region
tools.py contains several averaging, linear regressions, statistics, etc. utilities.
makeFigures.ipynb leverages all of the above to produce the figures.

# Running the code
A .yml file is included that contains all necessary python packages to run the code and produce the figures. Create a conda environment using conda env create -f environment.yml, then activate with conda activate orogconv, launch a Jupyter notebook and you are hopefully all set!

# Contact
For any questions, contact qnicolas --at-- berkeley --dot-- edu
