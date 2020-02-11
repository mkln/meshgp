#include <RcppArmadillo.h>
#include <omp.h>


//[[Rcpp::export]]
arma::mat Kpp(const arma::mat& x1, const arma::mat& x2, const arma::vec& theta, bool same=false);


//[[Rcpp::export]]
arma::mat Kppc(const arma::mat& coords,
               const arma::uvec& ind1, const arma::uvec& ind2, 
               const arma::vec& theta, bool same=false);


//[[Rcpp::export]]
arma::mat Kpp_mp(const arma::mat& x1, const arma::mat& x2, const arma::vec& theta, bool same=false);

//[[Rcpp::export]]
arma::mat xKpp(const arma::mat& x1, const arma::mat& x2, const arma::field<arma::vec>& params);

//[[Rcpp::export]]
arma::mat KppG(const arma::mat& coords,
               const arma::uvec& ind1, const arma::uvec& ind2, 
               const arma::vec& theta, bool same=false);

//[[Rcpp::export]]
arma::mat Kpp_choice(const arma::mat& coords,
                     const arma::uvec& ind1, const arma::uvec& ind2,
                     const arma::vec& theta, bool same=false);