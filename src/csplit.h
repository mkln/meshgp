#include "RcppArmadillo.h"
using namespace std;

// PARTITIONING

arma::uvec relabel(const arma::uvec& x, const arma::uvec& ux, arma::uvec& counts);

//[[Rcpp::export]]
arma::field<arma::uvec> csplit(const arma::uvec& x_orig);

arma::vec cpprange(int n);
arma::umat thresholding(const arma::mat& coords, int ell);

//[[Rcpp::export]]
arma::uvec ms_seq(const arma::mat& coords, int ell);