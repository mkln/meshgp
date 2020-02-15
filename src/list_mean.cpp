#include "RcppArmadillo.h"


//[[Rcpp::export]]
arma::mat list_mean(const arma::field<arma::mat>& x){
  int n = x.n_elem;
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
  //***#pragma omp parallel for
    for(int j=0; j<nrows; j++){
      for(int h=0; h<ncols; h++){
        for(int i=0; i<n; i++){
          result(j,h) += x(i)(j,h)/(n+.0);
        }
      }
    }
  return result;
}
