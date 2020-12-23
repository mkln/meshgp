#include "covariance_functions.h"
//#include "RcppArmadillo.h"

using namespace std;


CovarianceParams::CovarianceParams(){
  q = 0;
  npars = 0;
}

CovarianceParams::CovarianceParams(int dd, int q_in, int covmodel=-1){
  q = q_in;
  
  covariance_model = covmodel;
  if(covariance_model == -1){
    // determine automatically
    if(dd == 2){
      covariance_model = 0;
      n_cbase = q > 2? 3: 1;
      npars = 3*q + n_cbase;
    } else {
      if(q_in > 1){
        Rcpp::Rcout << "Multivariate on many inputs not implemented yet." << endl;
        throw 1;
      }
      covariance_model = 1;
    }
  } else {
    if(covariance_model == 2){
      if(dd == 2){
        if(q > 2){
          npars = 3;//1+3;
        } else {
          npars = 1;//1+1;
        }
      } else {
        if(q > 2){
          npars = 5;//1+5;
        } else {
          npars = 3;//1+3; // sigmasq + alpha + beta + phi
        }
      }
    }
  }
  
}

void CovarianceParams::transform(const arma::vec& theta){
  if(covariance_model == 0){
    if(q > 1){
      // multivariate spatial
      // from vector to all covariance components
      int k = theta.n_elem - npars; // number of cross-distances = p(p-1)/2
      arma::vec cparams = theta.subvec(0, npars - 1);
      ai1 = cparams.subvec(0, q-1);
      ai2 = cparams.subvec(q, 2*q-1);
      phi_i = cparams.subvec(2*q, 3*q-1);
      thetamv = cparams.subvec(3*q, 3*q+n_cbase-1);
      
      if(k>0){
        Dmat = vec_to_symmat(theta.subvec(npars, npars + k - 1));
      } else {
        Dmat = arma::zeros(1,1);
      }
    } else {
      // univariate spatial 
      ai1 = arma::ones(1) * theta(0); //sigmasq
      thetamv = arma::ones(1) * theta(1); //phi
      Dmat = arma::zeros(1,1);
    }

  }
  if(covariance_model == 1){
    // univariate with several inputs
    sigmasq = theta(0);
    kweights = theta.subvec(1, theta.n_elem-1);
  }
  if(covariance_model == 2){
    //arma::vec Kparam = data.theta; 
    int k = q * (q-1) / 2; // number of cross-distances = p(p-1)/2
    int cdims = theta.n_elem - k;
    thetamv = theta.subvec(0, cdims-1);
    
    if(k>0){
      Dmat = vec_to_symmat(theta.subvec(cdims, theta.n_elem-1));
    } else {
      Dmat = arma::zeros(1,1);
    }
  }
  if(covariance_model == 3){
    thetamv = theta;
  }
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

// exponential covariance
arma::mat cexpcov(const arma::mat& x, const arma::mat& y, const double& sigmasq, const double& phi, bool same){
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


double xCovHUV_base(const double& h, const double& u, const double& v, 
                    const arma::vec& params, const int& q, const int& dim){
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
        //double a_psi2    = 1.0;
        //double beta_psi2 = 1.0;
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


// matern covariance with nu = p + 1/2, and p=0,1,2
void matern_halfint_inplace(arma::mat& res,
    const arma::mat& x, const arma::mat& y, const double& phi, const double& sigmasq, bool same, int twonu){
  // 0 based indexing
  //arma::mat res = arma::zeros(x.n_rows, y.n_rows);
  double nugginside = 0;//1e-7;
  if(same){
    for(int i=0; i<x.n_rows; i++){
      arma::rowvec cri = x.row(i);
      for(int j=i; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          if(twonu == 1){
            res(i, j) = sigmasq * exp(-hphi);
          } else {
            if(twonu == 3){
              res(i, j) = sigmasq * exp(-hphi) * (1 + hphi);
            } else {
              if(twonu == 5){
                res(i, j) = sigmasq * (1 + hphi + hphi*hphi / 3.0) * exp(-hphi);
              }
            }
          }
        } else {
          res(i, j) = sigmasq + nugginside;
        }
      }
    }
    res = arma::symmatu(res);
  } else {
    for(int i=0; i<x.n_rows; i++){
      arma::rowvec cri = x.row(i);
      for(int j=0; j<y.n_rows; j++){
        arma::rowvec delta = cri - y.row(j);
        double hphi = arma::norm(delta) * phi;
        if(hphi > 0.0){
          if(twonu == 1){
            res(i, j) = sigmasq * exp(-hphi);
          } else {
            if(twonu == 3){
              res(i, j) = sigmasq * exp(-hphi) * (1 + hphi);
            } else {
              if(twonu == 5){
                res(i, j) = sigmasq * (1 + hphi + hphi*hphi / 3.0) * exp(-hphi);
              }
            }
          }
        } else {
          res(i, j) = sigmasq + nugginside;
        }
      }
    }
  }
  //return res;
}

//[[Rcpp::export]]
arma::mat xCovHUVc(const arma::mat& coords1, const arma::mat& coords2,
                   const arma::vec& params, bool same, int twonu){
  arma::mat res = arma::zeros(coords1.n_rows, coords2.n_rows);
  // sigmasq, phi
  matern_halfint_inplace(res, coords1, coords2, params(1), params(0), same, twonu);
  return res;
}


void xCovHUV_inplace(arma::mat& res,
                  const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& params, const arma::mat& Dmat, bool same, int twonu){

  int d = coords.n_cols;
  int p = Dmat.n_cols;
  if((d == 2) & (p < 2)){
    matern_halfint_inplace(res, coords.rows(ind1), coords.rows(ind2), params(1), params(0), same, twonu);
      //cexpcov(coords.rows(ind1), coords.rows(ind2), params(0), params(1), same);
  } else {
    if(same){
      for(int i=0; i<ind1.n_elem; i++){
        arma::rowvec cri = coords.row(ind1(i));
        for(int j=i; j<ind2.n_elem; j++){
          arma::rowvec delta = cri - coords.row(ind2(j));
          double h = arma::norm(delta.subvec(0, 1));
          double u = d > 2? abs(delta(2)) : 0;
          if(p > 1){
            for(int r=0; r<p; r++){
              for(int s=0; s<p; s++){
                double v = Dmat(r,s);
                int rix = i*p + r;
                int cix = j*p + s;
                //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
                res(rix, cix) = xCovHUV_base(h, u, v, params, p, d);
              }
            }
          } else {
            res(i, j) = xCovHUV_base(h, u, 0, params, p, d);
          }
        }
      }
      res = arma::symmatu(res);
    } else {
      //int cc = 0;
      for(int i=0; i<ind1.n_elem; i++){
        arma::rowvec cri = coords.row(ind1(i));
        for(int j=0; j<ind2.n_elem; j++){
          arma::rowvec delta = cri - coords.row(ind2(j));
          double h = arma::norm(delta.subvec(0, 1));
          double u = d > 2? abs(delta(2)) : 0;
          if(p > 1){
            for(int r=0; r<p; r++){
              for(int s=0; s<p; s++){
                double v = Dmat(r,s);
                int rix = i*p + r;
                int cix = j*p + s;
                //printf("i: %d, j: %d, r: %d, s: %d", i, j, r, s);
                res(rix, cix) = xCovHUV_base(h, u, v, params, p, d);
              }
            }
          } else {
            res(i, j) = xCovHUV_base(h, u, 0, params, p, d);
          }
        }
      }
      //return res;
  }
  }
}

arma::mat xCovHUV(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                  const arma::vec& params, const arma::mat& Dmat, bool same, int twonu){
  
  int q = Dmat.n_cols;
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1 * q, n2 * q);
  xCovHUV_inplace(res, coords, ind1, ind2, params, Dmat, same, twonu);
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
                          const CovarianceParams& covpars, bool same){
  int d = coords.n_cols;
  int p = covpars.Dmat.n_cols;
  if((d == 2) & (p < 2)){
    res = cexpcov(coords.rows(ind1), coords.rows(ind2), covpars.ai1(0), covpars.thetamv(0), same);
  } else {
    int v_ix_i;
    int v_ix_j;
    double h;
    double u;
    double v;
    arma::rowvec delta = arma::zeros<arma::rowvec>(d);
    
    // covpars.ai1 params: cparams.subvec(0, p-1);
    // covpars.ai2 params: cparams.subvec(p, 2*p-1);
    // covpars.phi_i params: cparams.subvec(2*p, 3*p-1);
    // C_base params: cparams.subvec(3*p, 3*p + k - 1); // k = q>2? 3 : 1;
    
    if(same){
      for(int i=0; i<ind1.n_elem; i++){
        v_ix_i = qv_block(ind1(i));
        double ai1_sq = covpars.ai1(v_ix_i) * covpars.ai1(v_ix_i);
        double ai2_sq = covpars.ai2(v_ix_i) * covpars.ai2(v_ix_i);
        arma::rowvec cxi = coords.row(ind1(i));
        
        for(int j=i; j<ind2.n_elem; j++){
          delta = cxi - coords.row(ind2(j));
          h = arma::norm(delta.subvec(0, 1));
          u = d < 3? 0 : abs(delta(2));
          
          v_ix_j = qv_block(ind2(j));
          v = covpars.Dmat(v_ix_i, v_ix_j);
          
          if(v == 0){ // v_ix_i == v_ix_j
            res(i, j) = ai1_sq * C_base(h, u, 0, covpars.thetamv, p, d) + 
              ai2_sq * fphi(h, covpars.phi_i(v_ix_i));
          } else {
            res(i, j) =  covpars.ai1(v_ix_i) * covpars.ai1(v_ix_j) * C_base(h, u, v, covpars.thetamv, p, d);
          }
        }
      }
      res = arma::symmatu(res);
    } else {
      for(int i=0; i<ind1.n_elem; i++){
        v_ix_i = qv_block(ind1(i));
        double ai1_sq = covpars.ai1(v_ix_i) * covpars.ai1(v_ix_i);
        double ai2_sq = covpars.ai2(v_ix_i) * covpars.ai2(v_ix_i);
        arma::rowvec cxi = coords.row(ind1(i));
        
        for(int j=0; j<ind2.n_elem; j++){
          delta = cxi - coords.row(ind2(j));
          h = arma::norm(delta.subvec(0, 1));
          u = d < 3? 0 : abs(delta(2));
          
          v_ix_j = qv_block(ind2(j));
          v = covpars.Dmat(v_ix_i, v_ix_j);
          
          if(v == 0){ // v_ix_i == v_ix_j
            res(i, j) = ai1_sq * C_base(h, u, 0, covpars.thetamv, p, d) + 
              ai2_sq * fphi(h, covpars.phi_i(v_ix_i));
          } else {
            res(i, j) =  covpars.ai1(v_ix_i) * covpars.ai1(v_ix_j) * C_base(h, u, v, covpars.thetamv, p, d);
          }
        }
      }
      //return res;
    }
  }
}


arma::mat mvCovAG20107(const arma::mat& coords, const arma::uvec& qv_block, 
                       const arma::uvec& ind1, const arma::uvec& ind2, 
                       const CovarianceParams& covpars, bool same){
  
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                       covpars, same);
  return res;
}



// for predictions
arma::mat mvCovAG20107x(const arma::mat& coords1,
                        const arma::uvec& qv_block1,
                        const arma::mat& coords2,
                        const arma::uvec& qv_block2,
                        const arma::vec& ai1, const arma::vec& ai2,
                        const arma::vec& phi_i, const arma::vec& thetamv,
                        const arma::mat& Dmat, bool same){
  
  arma::mat res = arma::zeros(coords1.n_rows, coords2.n_rows);
  
  int d = coords1.n_cols;
  int p = Dmat.n_cols;
  if((d == 2) & (p < 2)){
    res = cexpcov(coords1, coords2, ai1(0), thetamv(0), false);
  } else {
    int v_ix_i;
    int v_ix_j;
    double h;
    double u;
    double v;
    arma::rowvec delta = arma::zeros<arma::rowvec>(d);
    // covpars.ai1 params: cparams.subvec(0, p-1);
    // covpars.ai2 params: cparams.subvec(p, 2*p-1);
    // covpars.phi_i params: cparams.subvec(2*p, 3*p-1);
    // C_base params: cparams.subvec(3*p, 3*p + k - 1); // k = q>2? 3 : 1;
    for(int i=0; i<coords1.n_rows; i++){
      v_ix_i = qv_block1(i);
      double ai1_sq = ai1(v_ix_i) * ai1(v_ix_i);
      double ai2_sq = ai2(v_ix_i) * ai2(v_ix_i);
      arma::rowvec cxi = coords1.row(i);
      
      for(int j=0; j<coords2.n_rows; j++){
        delta = cxi - coords2.row(j);
        h = arma::norm(delta.subvec(0, 1));
        u = d < 3? 0 : abs(delta(2));
        
        v_ix_j = qv_block2(j);
        v = Dmat(v_ix_i, v_ix_j);
        
        if(v == 0){ // v_ix_i == v_ix_j
          res(i, j) = ai1_sq * C_base(h, u, 0, thetamv, p, d) + 
            ai2_sq * fphi(h, phi_i(v_ix_i));
        } else {
          res(i, j) =  ai1(v_ix_i) * ai1(v_ix_j) * C_base(h, u, v, thetamv, p, d);
        }
      }
    }
    return res;
  }
}


void NonspatialUnivariate_inplace(arma::mat& res,
                                  const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                                  const CovarianceParams& covpars, bool same){
  int d = coords.n_cols;
  if(same){
    for(int i=0; i<ind1.n_elem; i++){
      arma::rowvec cri = coords.row(ind1(i));
      for(int j=i; j<ind2.n_elem; j++){
        arma::rowvec deltasq = cri - coords.row(ind2(j));
        double weighted = (arma::accu(covpars.kweights.t() % deltasq % deltasq));
        res(i, j) = covpars.sigmasq * exp(-weighted) + (weighted == 0? 1e-3 : 0);
      }
    }
    res = arma::symmatu(res);
  } else {
    //int cc = 0;
    for(int i=0; i<ind1.n_elem; i++){
      arma::rowvec cri = coords.row(ind1(i));
      for(int j=0; j<ind2.n_elem; j++){
        arma::rowvec deltasq = cri - coords.row(ind2(j));
        double weighted = (arma::accu(covpars.kweights.t() % deltasq % deltasq));
        res(i, j) = covpars.sigmasq * exp(-weighted) + (weighted == 0? 1e-3 : 0);
      }
    }
  }
  
}

arma::mat NonspatialUnivariate(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                               const CovarianceParams& covpars, bool same){
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  NonspatialUnivariate_inplace(res, coords, ind1, ind2, covpars, same);
  return res;
}


void ExpCorrelPlusNugget_inplace(arma::mat& res,
                                 const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                                 const CovarianceParams& covpars, bool same){
  
  res = cexpcov(coords.rows(ind1), coords.rows(ind2), 1.0, covpars.phi, same);
  if(same){
    res.diag() += covpars.tausq;
  }
}

arma::mat ExpCorrelPlusNugget(const arma::mat& coords, const arma::uvec& ind1, const arma::uvec& ind2, 
                              const CovarianceParams& covpars, bool same){
  int n1 = ind1.n_elem;
  int n2 = ind2.n_elem;
  arma::mat res = arma::zeros(n1, n2);
  ExpCorrelPlusNugget_inplace(res, coords, ind1, ind2, covpars, same);  
}

void Covariancef_inplace(arma::mat& res,
                         const arma::mat& coords, const arma::uvec& qv_block, 
                         const arma::uvec& ind1, const arma::uvec& ind2, 
                         const CovarianceParams& covpars, bool same){
  
  
  if(covpars.covariance_model == 0){
    mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                         covpars, same);
  }
  if(covpars.covariance_model == 1){
    NonspatialUnivariate_inplace(res, coords, ind1, ind2, covpars, same);
  }
}



arma::mat Covariancef(const arma::mat& coords, const arma::uvec& qv_block, 
                      const arma::uvec& ind1, const arma::uvec& ind2, 
                      const CovarianceParams& covpars, bool same){
  arma::mat res;
  if(covpars.covariance_model < 0){
    Rcpp::Rcout << "Covariance model not implemented " << endl;
    throw 1;
  }
  if(covpars.covariance_model == 0){
    int n1 = ind1.n_elem;
    int n2 = ind2.n_elem;
    res = arma::zeros(n1, n2);
    mvCovAG20107_inplace(res, coords, qv_block, ind1, ind2, 
                         covpars, same);
  }
  if(covpars.covariance_model == 1){
    int n1 = ind1.n_elem;
    int n2 = ind2.n_elem;
    res = arma::zeros(n1, n2);
    NonspatialUnivariate_inplace(res, coords, ind1, ind2, covpars, same);
  }
  return res;
}

