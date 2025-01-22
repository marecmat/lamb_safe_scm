# Lamb modes: SAFE vs SCM

## Dependancies
- numpy
- scipy
- matplotlib
- tqdm (can be removed in the codes without affecting computations)

## How to use

Each method has been implemented in separated files
- analytical root-finding: `muller_rootfinding.py`
- SAFE: `safe.py` 
- SCM: `scm.py`

Run `compare_methods.py` to plot the dispersion curves for the selected methods (True/False statements at the beginning of the script). Optionnaly writes data.

The script `convergence.py` computes the convergence curves and writes data that are read by `plot_figure.py` to plot the figure.

Finally, `plot_figure.py` loads all the files and outputs a figure.

## Contact 

Mathieu Maréchal,

mathieu.marechal@univ-lemans.fr