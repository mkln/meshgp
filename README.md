## Meshed GPs for spatial or spatiotemporal regression

Install with `devtools::install_github("mkln/meshgp")`. A full documentation for the code is planned. 
Refer to examples in the `analysis` folder for some simulations and comparisons.

List of models that can be implemented currently:

 - `meshgp::meshgp()`: spatially-varying coefficients regression (SVC) using the `Z` argument to store the dynamic inputs
 - `meshgp::mvmeshgp()`: latent GP regression on multivariate outputs
 - `meshgp::meshgp_dev()`: latent GP regression on univariate output with gridded reference set $\neq$ observed locations
 
**Notes**: I'm always updating the source and some older functions/examples may break. Tested on Ubuntu 18.04 (R-4.0.2 w/ Intel MKL 2019.5 or 2020.1) and CentOS 8.2 (R-4.0.2 w/ OpenBLAS 0.3.10). Not tested on macOS or Windows yet. On CentOS, the default OpenBLAS 0.3.3 shipping with R causes segfaults due to possible conflicts with OpenMP. With OpenBLAS 0.3.10 compiled from source (using `make NO_AFFINITY=1 USE_LOCKING=1 USE_OPENMP=1`) there are no issues.

### [Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains](https://doi.org/10.1080/01621459.2020.1833889)
M. Peruzzi, S. Banerjee and A. O. Finley

[Link to arXiv version with supplement](https://arxiv.org/abs/2003.11208)


<img src="https://raw.githubusercontent.com/mkln/meshgp/master/figures/Figure_NDVI_predict.png" height=300 align=center>


Citation: *Michele Peruzzi, Sudipto Banerjee & Andrew O. Finley (2020) Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains, Journal of the American Statistical Association, in press. DOI: 10.1080/01621459.2020.1833889* 

```
@article{doi:10.1080/01621459.2020.1833889,
author = {Michele Peruzzi and Sudipto Banerjee and Andrew O. Finley},
title = {Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains},
journal = {Journal of the American Statistical Association},
volume = {0},
number = {0},
pages = {1-31},
year  = {2020},
publisher = {Taylor & Francis},
doi = {10.1080/01621459.2020.1833889},
URL = {https://doi.org/10.1080/01621459.2020.1833889},
eprint = {https://doi.org/10.1080/01621459.2020.1833889}
}
```