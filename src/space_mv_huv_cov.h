
#include "RcppArmadillo.h"

using namespace std;

// Apanasovich & Genton 2010 
// cross-covariances with different autocovariances 
// eq. (7) p. 20

// phi(x) : exp(-c x^\gamma)
// psi(x) : (a x^\alpha + 1)^\beta



inline double fphi(const double& h, const double& phi){
  return exp(-phi * h);
}

inline double sqrt_fpsi(const double& x, const double& a, const double& beta){
  // square root of psi function
  // used in conjunction with setting alpha=0.5 and gamma=0.5 in covariances
  return exp(0.5*beta * log1p(a*x));// pow(a * x + 1, 0.5*beta);
}

inline arma::mat vec_to_symmat(const arma::vec& x){
  int k = x.n_elem; // = p(p-1)/2
  int p = ( 1 + sqrt(1 + 8*k) )/2;
  
  arma::mat res = arma::zeros(p, p);
  int start_i=1;
  int ix=0;
  for(int j=0; j<p; j++){
    for(int i=start_i; i<p; i++){
      res(i, j) = x(ix);
      ix ++;
    }
    start_i ++;
  } 
  return arma::symmatl(res);
}

double C_base(const double& h, const double& u, const double& v, const arma::vec& params, const int& q, const int& dim);

void mvCovAG20107_inplace(arma::mat& res,
                       const arma::mat& coords, 
                       const arma::uvec& qv_block,
                       const arma::uvec& ind1, const arma::uvec& ind2, 
                       const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                       const arma::mat& Dmat, bool same=false);

//[[Rcpp::export]]
arma::mat mvCovAG20107(const arma::mat& coords, const arma::uvec& qv_block, 
                    const arma::uvec& ind1, const arma::uvec& ind2, 
                    const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                    const arma::mat& Dmat, bool same=false);