#include "RcppArmadillo.h"
#include <omp.h>

//[[Rcpp::export]]
arma::mat list_mean(const arma::field<arma::mat>& x){
  // all matrices in x must be the same size.
  int n = x.n_elem;
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
  #pragma omp parallel for
  for(int j=0; j<nrows*ncols; j++){
    //for(int h=0; h<ncols; h++){
    arma::vec slices = arma::zeros(n);
    for(int i=0; i<n; i++){
      slices(i) = x(i)(j);
    }
    result(j) = arma::mean(slices);
    //}
  }
  return result;
}
