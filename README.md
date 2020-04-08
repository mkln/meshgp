## Q-MGP for spatial or spatiotemporal data
See folder `analysis` for code to reproduce simulations and comparisons.

Install with `devtools::install_github("mkln/meshgp")`. Tested on Ubuntu 18.04. Not tested on macOS or Windows.

<img src="https://raw.githubusercontent.com/mkln/meshgp/master/figures/Figure_NDVI_predict.png" height=300 align=center>

### [Highly Scalable Bayesian Geostatistical Modeling via Meshed Gaussian Processes on Partitioned Domains](https://arxiv.org/abs/2003.11208)
M. Peruzzi, S. Banerjee and A. O. Finley

*We introduce a class of scalable Bayesian hierarchical models for the analysis of massive geostatistical datasets. The underlying idea combines ideas on high-dimensional geostatistics by partitioning the spatial domain and modeling the regions in the partition using a sparsity-inducing directed acyclic graph (DAG). We extend the model over the DAG to a well-defined spatial process, which we call the Meshed Gaussian Process (MGP). A major contribution is the development of a MGPs on tessellated domains, accompanied by a Gibbs sampler for the efficient recovery of spatial random effects. In particular, the cubic MGP (Q-MGP) can harness high-performance computing resources by executing all large-scale operations in parallel within the Gibbs sampler, improving mixing and computing time compared to sequential updating schemes. Unlike some existing models for large spatial data, a Q-MGP facilitates massive caching of expensive matrix operations, making it particularly apt in dealing with spatiotemporal remote-sensing data. We compare Q-MGPs with large synthetic and real world data against state-of-the-art methods. We also illustrate using Normalized Difference Vegetation Index (NDVI) data from the Serengeti park region to recover latent multivariate spatiotemporal random effects at millions of locations.*