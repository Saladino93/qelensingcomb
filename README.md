# qelensingcomb
Combining quadratic CMB lensing estimators.



## Requirements

mystic, pixell, symlens

## Usage

First, define everything in config.yaml type file.

### Pipeline:

* extract_biases.py , this extracts the biases from the provided simulations
* prepare_and_plot.py, this prepares files for plots and optimisation, taking means and scatters
* lmax_optimize.py, to optimize over a fixed configuration
* execute_opt.py, to optimize over several configurations
* process_results.py, processes results from a fixed optimization
* execute_processing.py, processes results from several optimizations

** Local modules used **

* best.py
* utilities.py

