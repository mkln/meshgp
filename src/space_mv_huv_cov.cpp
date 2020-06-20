#include "space_mv_huv_cov.h"
//#include "RcppArmadillo.h"

using namespace std;

// Apanasovich & Genton 2010 
// cross-covariances with different autocovariances 
// eq. (7) p. 20

// phi(x) : exp(-c x^\gamma)
// psi(x) : (a x^\alpha + 1)^\beta



double C_base(const double& h, const double& u, const double& v, const arma::vec& params, const int& q, const int& dim){
  if(dim < 3){
    // no time, only space
    if(q > 2){
      // multivariate  space. v plays role of time of Gneiting 2002
      double a_psi1    = params(0);
      double beta_psi1 = params(1);
      double c_phi1    = params(2);
      double psi1_sqrt = sqrt_fpsi(v, a_psi1, beta_psi1); // alpha_psi1=0.5
      return fphi(h/psi1_sqrt, c_phi1) / (psi1_sqrt * psi1_sqrt);
    } else {
      if(q == 2){
        // multivariate  space. v plays role of time of Gneiting 2002
        double c_phi1    = params(0);
        double psi1_sqrt = sqrt(v + 1);//sqrt_fpsi(v, a_psi1, beta_psi1); // alpha_psi1=0.5
        //return fphi(h/psi1_sqrt, c_phi1) / (psi1_sqrt * psi1_sqrt);
        return fphi(h/psi1_sqrt, c_phi1) / (v+1.0);
      } else {
        // 1 variable no time = exp covariance
        double phi       = params(0);
        return fphi(h, phi); 
      }
    }
  } else {
    throw 1;
  }
}

void mvCovAG20107_inplace(arma::mat& res,
                       const arma::mat& coords, 
                       const arma::uvec& qv_block,
                       const arma::uvec& ind1, const arma::uvec& ind2, 
                       const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                       const arma::mat& Dmat, bool same){
  int d = coords.n_cols;
  int p = Dmat.n_cols;
  int v_ix_i;
  int v_ix_j;
  double h;
  double u;
  double v;
  arma::rowvec delta = arma::zeros<arma::rowvec>(d);
  
  // ai1 params: cparams.subvec(0, p-1);
  // ai2 params: cparams.subvec(p, 2*p-1);
  // phi_i params: cparams.subvec(2*p, 3*p-1);
  // C_base params: cparams.subvec(3*p, 3*p + k - 1); // k = q>2? 3 : 1;
  
  if(same){
    for(int i=0; i<ind1.n_elem; i++){
      v_ix_i = qv_block(ind1(i));
      double ai1_sq = ai1(v_ix_i) * ai1(v_ix_i);
      double ai2_sq = ai2(v_ix_i) * ai2(v_ix_i);
      arma::rowvec cxi = coords.row(ind1(i));
      
      for(int j=i; j<ind2.n_elem; j++){
        delta = cxi - coords.row(ind2(j));
        h = arma::norm(delta.subvec(0, 1));
        u = d < 3? 0 : abs(delta(2));
        
        v_ix_j = qv_block(ind2(j));
        v = Dmat(v_ix_i, v_ix_j);
        
        if(v == 0){ // v_ix_i == v_ix_j
          res(i, j) = ai1_sq * C_base(h, u, 0, thetamv, p, d) + 
                      ai2_sq * fphi(h, phi_i(v_ix_i));
        } else {
          res(i, j) =  ai1(v_ix_i) * ai1(v_ix_j) * C_base(h, u, v, thetamv, p, d);
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(int i=0; i<ind1.n_elem; i++){
      v_ix_i = qv_block(ind1(i));
      double ai1_sq = ai1(v_ix_i) * ai1(v_ix_i);
      double ai2_sq = ai2(v_ix_i) * ai2(v_ix_i);
      arma::rowvec cxi = coords.row(ind1(i));
      
      for(int j=0; j<ind2.n_elem; j++){
        delta = cxi - coords.row(ind2(j));
        h = arma::norm(delta.subvec(0, 1));
        u = d < 3? 0 : abs(delta(2));
        
        v_ix_j = qv_block(ind2(j));
        v = Dmat(v_ix_i, v_ix_j);
        
        if(v == 0){ // v_ix_i == v_ix_j
          res(i, j) = ai1_sq * C_base(h, u, 0, thetamv, p, d) + 
                      ai2_sq * fphi(h, phi_i(v_ix_i));
        } else {
          res(i, j) =  ai1(v_ix_i) * ai1(v_ix_j) * C_base(h, u, v, thetamv, p, d);
        }
      }
    }
    //return res;
  }
}

arma::mat mvCovAG20107(const arma::mat& coords, const arma::uvec& qv_block, 
                    const arma::uvec& ind1, const arma::uvec& ind2, 
                    const arma::vec& ai1, const arma::vec& ai2, const arma::vec& phi_i, const arma::vec& thetamv, 
                    const arma::mat& Dmat, bool same){
  
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                       ai1, ai2, phi_i, thetamv, 
                       Dmat, same);
  return res;
}
