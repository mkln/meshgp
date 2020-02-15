#include "nonseparable_huv_cov.h"
//#include "RcppArmadillo.h"

using namespace std;

// phi(x) : exp(-c x^\gamma)
// psi(x) : (a x^\alpha + 1)^\beta

double fphi(double x, double c){
  return exp(-c * x);
}

double sqrt_fpsi(double x, double a, const double beta){
  // square root of psi function
  // used in conjunction with setting alpha=0.5 and gamma=0.5 in covariances
  return exp(0.5*beta * log1p(a*x));// pow(a * x + 1, 0.5*beta);
}

double xCovHUV_base(double h, double u, double v, const arma::vec& params){
  // Gneiting 2002 uses phi and psi as above ( beta here is kappa in covariance_functions.cpp )
  // hsq = ||h||^2, u = |u|^2 are squared norms. v = d_ij is "distance" between var. i and var. j
  // d=2
  // we also set 
  // alpha=0.5
  // gamma=0.5 

  double sigmasq   = params(0);
  double a_psi2    = params(1);
  double beta_psi2 = params(2);
  double a_psi1    = params(3);
  double beta_psi1 = params(4);
  double c_phi1    = params(5);
  
  double psi2_sqrt = sqrt_fpsi(v,           a_psi2, beta_psi2); // alpha_psi2=0.5
  double psi1_sqrt = sqrt_fpsi(u/psi2_sqrt, a_psi1, beta_psi1); // alpha_psi1=0.5
  double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
  
  return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1 / psi2_sqrt;
}

arma::mat vec_to_symmat(const arma::vec& x){
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


void xCovHUV_inplace(arma::mat& res,
                  const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same){
  
  int p = Dmat.n_cols;
  if(same){
    for(int i=0; i<ind1.n_elem; i++){
      for(int j=i; j<ind2.n_elem; j++){
        arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
        double h = arma::norm(delta.subvec(0, 1));
        double u = abs(delta(2));
        if(p > 1){
          for(int r=0; r<p; r++){
            for(int s=0; s<p; s++){
              double v = Dmat(r,s);
              int rix = i*p + r;
              int cix = j*p + s;
              //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
              res(rix, cix) = xCovHUV_base(h, u, v, cparams);
            }
          }
        } else {
          res(i, j) = xCovHUV_base(h, u, 0, cparams);
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    //int cc = 0;
    for(int i=0; i<ind1.n_elem; i++){
      for(int j=0; j<ind2.n_elem; j++){
        arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
        double h = arma::norm(delta.subvec(0, 1));
        double u = abs(delta(2));
        if(p > 1){
          for(int r=0; r<p; r++){
            for(int s=0; s<p; s++){
              double v = Dmat(r,s);
              int rix = i*p + r;
              int cix = j*p + s;
              //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
              res(rix, cix) = xCovHUV_base(h, u, v, cparams);
            }
          }
        } else {
          res(i, j) = xCovHUV_base(h, u, 0, cparams);
        }
      }
    }
    //return res;
  }
}


arma::mat xCovHUV(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same){
  
  int p = Dmat.n_cols;
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1 * p, n2 * p);
  xCovHUV_inplace(res, coords, ind1, ind2, cparams, Dmat, same);
  return res;
}
