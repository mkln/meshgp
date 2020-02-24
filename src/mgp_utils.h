#ifndef MGP_UTILS 
#define MGP_UTILS

#include "RcppArmadillo.h"


using namespace std;

arma::vec armarowsum(const arma::mat& x);

arma::vec armacolsum(const arma::mat& x);

arma::sp_mat Zify(const arma::mat& x);


bool compute_block(bool predicting, int block_ct, bool rfc);

// everything that changes during MCMC
struct MeshData {
  
  arma::vec theta; 
  
  arma::vec wcore; 
  arma::field<arma::mat> w_cond_mean_K;
  arma::field<arma::mat> w_cond_prec;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::vec loglik_w_comps;
  double loglik_w;
  
  arma::uvec track_chol_fails;
  bool cholfail;
  
};

void print_data(const MeshData& data);


arma::vec kdiagchol(const arma::mat& Koo, const arma::mat& Kos, const arma::mat& Kssi);

#endif

