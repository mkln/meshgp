#include "covariance_functions.h"

using namespace std;

arma::mat Kpp(const arma::mat& x1, const arma::mat& x2, const arma::vec& theta, bool same){
  // exponential covariance
  bool time = x1.n_cols == 3;
  double sigmasq = theta(0);
  double phi = theta(1);
  double phi2 = 0;
  if(time){
    phi2 = theta(2);
  }
  if(same){ // x1==x2
    arma::mat outmat = sigmasq*arma::eye(x1.n_rows, x1.n_rows);
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=i+1; j<outmat.n_cols; j++){
        arma::rowvec space_delta = x1.submat(i, 0, i, 1) - x2.submat(j, 0, j, 1); 
        double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
        
        if(time){
          double time_delta = abs(x1(i, 2) - x2(j, 2)); 
          outmat(i, j) = sigmasq*exp(-phi * rsq) * exp(-phi2 * (time_delta));
        } else {
          outmat(i, j) = sigmasq*exp(-phi * rsq);
        }
      }
    }
    return arma::symmatu(outmat);
    
  } else {
    arma::mat outmat = arma::zeros(x1.n_rows, x2.n_rows);
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=0; j<outmat.n_cols; j++){
        arma::rowvec space_delta = x1.submat(i, 0, i, 1) - x2.submat(j, 0, j, 1); 
        double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
        outmat(i, j) = sigmasq*exp(-phi * rsq);
        
        if(time){
          double time_delta = abs(x1(i, 2) - x2(j, 2)); 
          outmat(i, j) *= exp(-phi2 * (time_delta));
        }
      }
    }
    return outmat;
  }
}


arma::mat Kpp_choice(const arma::mat& coords,
                     const arma::uvec& ind1, const arma::uvec& ind2,
                     const arma::vec& theta, bool same){
  if(coords.n_cols == 2){
    return Kppc(coords, ind1, ind2, theta, same);
  } else {
    return Kppc(coords, ind1, ind2, theta, same);
  }
}

arma::mat Kppc(const arma::mat& coords,
               const arma::uvec& ind1, const arma::uvec& ind2, 
               const arma::vec& theta, bool same){
  
  // exponential covariance
  bool time = coords.n_cols == 3;
  double sigmasq = theta(0);
  double phi1 = theta(1);
  double phi2 = 0;
  if(time){
    phi2 = theta(2);
  }
  if(same){ // x1==x2
    arma::mat outmat = sigmasq*arma::eye(ind1.n_elem, ind2.n_elem);
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=i+1; j<outmat.n_cols; j++){
        if(!time){
          arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
          double rsq = pow(arma::accu(pow(delta, 2.0)), .5);
          outmat(i, j) = sigmasq*exp(-phi1 * rsq);
        } else {
          arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
          
          arma::rowvec space_delta = delta.subvec(0, 1);
          double spaceD = arma::accu(pow(space_delta, 2.0));
          
          double time_delta = delta(2);
          double timeD = pow(time_delta, 2.0);
          
          outmat(i, j) = sigmasq*exp(-pow(phi1 * spaceD + phi2 * timeD, .5));
        }
      }
    }
    return arma::symmatu(outmat);
    
  } else {
    arma::mat outmat = arma::zeros(ind1.n_elem, ind2.n_elem);
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=0; j<outmat.n_cols; j++){
        if(!time){
          arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
          double rsq = pow(arma::accu(pow(delta, 2.0)), .5);
          outmat(i, j) = sigmasq*exp(-phi1 * rsq);
        } else {
          arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
          arma::rowvec space_delta = delta.subvec(0, 1);
          double time_delta = delta(2);
          
          double spaceD = arma::accu(pow(space_delta, 2.0));
          double timeD = pow(time_delta, 2.0);
          
          outmat(i, j) = sigmasq*exp(-pow(phi1 * spaceD + phi2 * timeD, .5));
        }
      }
    }
    return outmat;
  }
}

arma::mat KppG(const arma::mat& coords,
               const arma::uvec& ind1, const arma::uvec& ind2, 
               const arma::vec& theta, bool same){
  // Gneiting: Stationary Covariance Functions for Space-Time Data 
  // (16) in exponential form like NNGP AOAS
  double sigmasq = theta(0);
  double aa = theta(1);
  double cc = theta(2);
  double kk = theta(3);
  
  if(same){ // x1==x2
    arma::mat outmat = sigmasq * arma::eye(ind1.n_elem, ind2.n_elem);
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=i+1; j<outmat.n_cols; j++){
        arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
        
        arma::rowvec space_delta = delta.subvec(0, 1);
        double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
        double time_delta = abs(delta(2)); 
        
        double u2a1 = aa * pow(time_delta, 2) + 1;
        
        outmat(i, j) = sigmasq / pow(u2a1, kk) * 
          exp(- cc * rsq / pow(u2a1, kk/2.0) );
      }
    }
    return arma::symmatu(outmat);
    
  } else {
    arma::mat outmat = arma::zeros(ind1.n_elem, ind2.n_elem);
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=0; j<outmat.n_cols; j++){
        
        arma::rowvec delta = coords.row(ind1(i)) - coords.row(ind2(j));
        arma::rowvec space_delta = delta.subvec(0, 1);
        
        double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
        double time_delta = abs(delta(2)); 
        
        double u2a1 = aa * pow(time_delta, 2) + 1;
        
        outmat(i, j) = sigmasq / pow(u2a1, kk) * 
          exp(- cc * rsq / pow(u2a1, kk/2.0) );
      }
    }
    return outmat;
  }
}


/*
 //[[Rcpp::export]]
 arma::mat Kpp_t(const arma::mat& x1, const arma::mat& x2, const arma::vec& theta){
 // exponential covariance
 bool time = x1.n_cols == 3;
 double sigmasq = theta(0);
 arma::vec phi = theta.subvec(1,2);
 
 arma::mat outmat = arma::zeros(x1.n_rows, x2.n_rows);
 for(unsigned int i=0; i<outmat.n_rows; i++){
 for(unsigned int j=0; j<outmat.n_cols; j++){
 arma::rowvec delta = x1.submat(i, 0, i, 2) - x2.submat(j, 0, j, 2); 
 double rsq = pow(arma::accu(pow(phi % delta, 2.0)), .5);
 outmat(i, j) = sigmasq*exp(-delta * phi);
 }
 }
 return outmat;
 
 }*/


arma::mat Kpp_mp(const arma::mat& x1, const arma::mat& x2, const arma::vec& theta, bool same){
  // exponential covariance
  bool time = x1.n_cols == 3;
  double sigmasq = theta(0);
  double phi = theta(1);
  double phi2 = 0;
  if(time){
    phi2 = theta(2);
  }
  if(same){ // x1==x2
    arma::mat outmat = sigmasq*arma::eye(x1.n_rows, x1.n_rows);
#pragma omp parallel for
    for(int ix=0; ix < outmat.n_elem; ix++){
      arma::uvec ind = arma::ind2sub(arma::size(outmat), ix);
      int i = ind(0);
      int j = ind(1);
      if(i <= j){
        arma::rowvec space_delta = x1.submat(i, 0, i, 1) - x2.submat(j, 0, j, 1); 
        double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
        if(time){
          double time_delta = x1(i, 2) - x2(j, 2); 
          outmat(i, j) = sigmasq*exp(-phi * rsq) * exp(-phi2 * abs(time_delta));
        } else {
          outmat(i, j) = sigmasq*exp(-phi * rsq);
        }
      }
    }
    return arma::symmatu(outmat);
    
  } else {
    arma::mat outmat = arma::zeros(x1.n_rows, x2.n_rows);
#pragma omp parallel for
    for(unsigned int i=0; i<outmat.n_rows; i++){
      for(unsigned int j=0; j<outmat.n_cols; j++){
        arma::rowvec space_delta = x1.submat(i, 0, i, 1) - x2.submat(j, 0, j, 1); 
        double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
        outmat(i, j) = sigmasq*exp(-phi * rsq);
        
        if(time){
          double time_delta = x1(i, 2) - x2(j, 2); 
          outmat(i, j) *= exp(-phi2 * abs(time_delta));
        }
      }
    }
    return outmat;
  }
}

arma::mat xKpp(const arma::mat& x1, const arma::mat& x2, const arma::field<arma::vec>& params){
  
  int dim = params(0).n_cols;
  
  arma::vec sigmas = params(0);
  arma::vec phis = params(1);
  
  arma::mat result = arma::zeros(x1.n_rows * dim, x2.n_rows * dim);
  
  for(int i=0; i<x1.n_rows; i++){
    for(int j=0; j<x2.n_rows; j++){
      arma::rowvec space_delta = x1.submat(i, 0, i, x1.n_cols-1) - x2.submat(j, 0, j, x2.n_cols-1); 
      double rsq = pow(arma::accu(pow(space_delta, 2.0)), .5);
      
      for(int h=0; h<dim; h++){
        result( dim * i + h, dim * j + h ) = sigmas(h) * exp( - phis(h) * rsq );
      }
    }
  }
  
  return result;
  
}
