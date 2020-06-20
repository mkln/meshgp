#include "covariance_functions.h"
//#include "RcppArmadillo.h"

using namespace std;


// exponential covariance
arma::mat cexpcov(const arma::mat& x, const arma::mat& y, double sigmasq, double phi, bool same){
  // 0 based indexing
  if(same){
    arma::mat pmag = arma::sum(x % x, 1);
    int np = x.n_rows;
    arma::mat K = sigmasq * exp(-phi * sqrt(abs(arma::repmat(pmag.t(), np, 1) + arma::repmat(pmag, 1, np) - 2 * x * x.t())));
    return K;
  } else {
    arma::mat pmag = arma::sum(x % x, 1);
    arma::mat qmag = arma::sum(y % y, 1);
    int np = x.n_rows;
    int nq = y.n_rows;
    arma::mat K = sigmasq * exp(-phi * sqrt(abs(arma::repmat(qmag.t(), np, 1) + arma::repmat(pmag, 1, nq) - 2 * x * y.t())));
    return K;
  }
}

double xCovHUV_base(double h, double u, double v, const arma::vec& params, int q, int dim){
  //Rcpp::Rcout << "h: " << h << " u: " << u << " v: " << v << " | q: " << q << " dim: " << dim << endl;
  // Gneiting 2002 uses phi and psi as above ( beta here is kappa in covariance_functions.cpp )
  // hsq = ||h||^2, u = |u|^2 are squared norms. v = d_ij is "distance" between var. i and var. j
  // d=2
  // we also set 
  // alpha=0.5
  // gamma=0.5 
  if(dim == 3){
    if(q > 2){
      // full multivariate
      // sigmasq + 5 params
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
    } else {
      if(q == 2){
        // multivariate but fixing a_psi2=1, beta_psi2=1 ?
        // sigmasq + 3 params + v
        double sigmasq   = params(0);
        double a_psi2    = 1.0;
        double beta_psi2 = 1.0;
        double a_psi1    = params(1);
        double beta_psi1 = params(2);
        double c_phi1    = params(3);
        double psi2_sqrt = sqrt(v + 1.0);//sqrt_fpsi(v,           a_psi2, beta_psi2); // alpha_psi2=0.5
        
        double psi1_sqrt = sqrt_fpsi(u/psi2_sqrt, a_psi1, beta_psi1); // alpha_psi1=0.5
        double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
        
        return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1 / psi2_sqrt;
      } else {
        // univariate spacetime -- using Gneiting 2002
        // multivariate but fixing a_psi2=1, beta_psi2=1 ?
        // sigmasq + 3 params
        double sigmasq   = params(0);
        double a_psi1    = params(1);
        double beta_psi1 = params(2);
        double c_phi1    = params(3);
        double psi1_sqrt = sqrt_fpsi(u, a_psi1, beta_psi1); // alpha_psi1=0.5
        double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
        
        return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1;
      }
    }
  } else {
    // no time, only space
    if(q > 2){// number of variables - which we consider q=k
      /*
       // space-variable nonseparable psi1=const psi2=const in Apanasovich&Genton2010 (4)
       double sigmasq   = params(0);
       double a_psi4    = params(1);
       double beta_psi4 = params(2);
       double a_psi3    = params(3);
       double beta_psi3 = params(4);
       double c_phi1    = params(5);
       double c_phi3    = 1.0;
       
       double psi4_sqrt = sqrt_fpsi(v, a_psi4, beta_psi4);
       double psi3_sqrt = sqrt_fpsi(h, a_psi3, beta_psi3);
       double phi1      = fphi(h, c_phi1);
       double phi3      = fphi(v/psi3_sqrt, c_phi3);
       
       double psi3_sqrt_powq = psi3_sqrt;
       for(int i=0; i<q-1; i++){
       psi3_sqrt_powq *= psi3_sqrt_powq;
       }
       return sigmasq / (psi3_sqrt_powq * psi4_sqrt) * phi1 * phi3;*/
      
      // multivariate  space. v plays role of time of Gneiting 2002
      // sigmasq + 3 params
      double sigmasq   = params(0);
      double a_psi1    = params(1);
      double beta_psi1 = params(2);
      double c_phi1    = params(3);
      double psi1_sqrt = sqrt_fpsi(v, a_psi1, beta_psi1); // alpha_psi1=0.5
      double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
      
      return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1;
    } else {
      if(q == 2){
        // multivariate  space. v plays role of time of Gneiting 2002
        // sigmasq + 1 params + v
        double sigmasq   = params(0);
        //double a_psi1    = params(1);
        //double beta_psi1 = params(2);
        double c_phi1    = params(1);
        double psi1_sqrt = sqrt(v + 1);//sqrt_fpsi(v, a_psi1, beta_psi1); // alpha_psi1=0.5
        double phi1      = fphi(h/psi1_sqrt, c_phi1); // gamma_phi1=0.5
        
        return sigmasq / (psi1_sqrt * psi1_sqrt) * phi1;
      } else {
        // 1 variable no time = exp covariance
        double sigmasq   = params(0);
        double phi       = params(1);
        return sigmasq * fphi(h, phi); 
      }
      
    }
  }
  
}

void xCovHUV_inplace(arma::mat& res,
                  const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same){
  int d = coords.n_cols;
  int p = Dmat.n_cols;
  if((d == 2) & (p < 2)){
    res = cexpcov(coords.rows(ind1), coords.rows(ind2), cparams(0), cparams(1), same);
  } else {
    if(same){
      for(int i=0; i<ind1.n_elem; i++){
        for(int j=i; j<ind2.n_elem; j++){
          arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
          double h = arma::norm(delta.subvec(0, 1));
          double u = d > 2? abs(delta(2)) : 0;
          if(p > 1){
            for(int r=0; r<p; r++){
              for(int s=0; s<p; s++){
                double v = Dmat(r,s);
                int rix = i*p + r;
                int cix = j*p + s;
                //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
                res(rix, cix) = xCovHUV_base(h, u, v, cparams, p, d);
              }
            }
          } else {
            res(i, j) = xCovHUV_base(h, u, 0, cparams, p, d);
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
          double u = d > 2? abs(delta(2)) : 0;
          if(p > 1){
            for(int r=0; r<p; r++){
              for(int s=0; s<p; s++){
                double v = Dmat(r,s);
                int rix = i*p + r;
                int cix = j*p + s;
                //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
                res(rix, cix) = xCovHUV_base(h, u, v, cparams, p, d);
              }
            }
          } else {
            res(i, j) = xCovHUV_base(h, u, 0, cparams, p, d);
          }
        }
      }
      //return res;
  }
  }
}

arma::mat xCovHUV(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& cparams, const arma::mat& Dmat, bool same){
  
  int q = Dmat.n_cols;
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1 * q, n2 * q);
  xCovHUV_inplace(res, coords, ind1, ind2, cparams, Dmat, same);
  return res;
}



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
  if((d == 2) & (p < 2)){
    res = cexpcov(coords.rows(ind1), coords.rows(ind2), thetamv(0), thetamv(1), same);
  } else {
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
