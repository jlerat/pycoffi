# pycoffi
 [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17247296.svg)](https://doi.org/10.5281/zenodo.17247296) 
 [![Build pycoffi](https://github.com/jlerat/pycoffi/actions/workflows/python-package-conda.yml/badge.svg)](https://github.com/jlerat/pycoffi/actions/workflows/python-package-conda.yml) 

This package contains data and script supporting the CoFFI paper.

# Installation
- Create a suitable python environment. We recommend using [miniconda](https://docs.conda.io/projects/miniconda/en/latest/) combined with the environment specification provided in the [env\_pycoffi.yml] (env_pycoffi.yml) file in this repository.
- Git clone this repository and run `pip install .`

# Data
The data provided in this package is in the [data](data) folder:
* [stations.csv](data/stations.csv): list of stations.

* [covariate data](data/covariate): CSV files containing the annual maximum of
  observed and AWRAL streamflow for each site.

* [CoFFI results](outputs): Results from CoFFI model including
    * [detailed results for the Miva site](outputs/results_138001_v7.3)
    * [posterior predictive checks](outputs/posterior_predictive_checks_v7.3)
    * [discrepancy metrics](outputs/metrics_v7.3)

# Code
The code used to generate all plots can be found in the folder
[scripts](scripts):
* [figure1](scripts/figure1_site_map.py): Generate site map
* [figure2](scripts/figure2_data_timeseries.py): Timeseries of observed and
  AWRA-L data for the Miva site.
* [figure3](scripts/figure3_posterior_predictive_checks.py): Posterior
  predictive checks plot for the Miva site.
* [figure4](scripts/figure4_frequency_plot.py): Frequency plot for the Miva
  site.
* [figure5](scripts/figure5_equivalent_record_length.py): Equivalent record
  length plot for the Miva site.
* [figure6](scripts/figure6_predictive_checks_maps.py): Predictive checks maps.
* [figure7](scripts/figure7_discrepancy_stats.py): Statistics on discrepancy
  metric.
* [figure8](scripts/figure8_explain_gains.py): Plot to explain CoFFI
  performance gains.
* [figure9](scripts/figure9_impact_record_length.py): Impact of record length
  on CoFFI performance.


## Attribution
This project is licensed under the [MIT License](LICENSE), which allows for free use, modification, and distribution of the code under the terms of the license.

For proper citation of this project, please refer to the [CITATION.cff](CITATION.cff) file, which provides guidance on how to cite the software and relevant publications.
