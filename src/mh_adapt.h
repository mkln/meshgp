#include <RcppArmadillo.h>


const double EPS = 0.1; // 0.01
const double tau_accept = 0.234; // target
const double g_exp = .01;
const int g0 = 500; // iterations before starting adaptation
const double rho_max = 2;
const double rho_min = 5;


inline bool do_I_accept(double logaccept){ //, string name_accept, string name_count, List mcmc_pars){
  double acceptj = 1.0;
  if(!arma::is_finite(logaccept)){
    acceptj = 0.0;
  } else {
    if(logaccept < 0){
      acceptj = exp(logaccept);
    }
  }
  double u = arma::randu();
  if(u < acceptj){
    return true;
  } else {
    return false;
  }
}

inline void adapt(const arma::vec& param, 
           arma::vec& sumparam, 
           arma::mat& prodparam,
           arma::mat& paramsd, 
           arma::vec& sd_param, // mcmc sd
           int mc, 
           double accept_ratio){
  // adaptive update for theta = [sigmasq, phi, tau]
  
  int siz = param.n_rows;
  sumparam += param;
  prodparam += param * param.t();
  
  
  
  sd_param(mc+1) = sd_param(mc) + pow(mc+1.0, -g_exp) * (accept_ratio - tau_accept);
  //clog << sd_param.col(j).row(mc) << endl;
  if(sd_param(mc+1) > exp(rho_max)){
    sd_param(mc+1) = exp(rho_max);
  } else {
    if(sd_param(mc+1) < exp(-rho_min)){
      sd_param(mc+1) = exp(-rho_min);
    }
  }

  
  if(mc > g0){
    paramsd = sd_param(mc+1) / (mc+1.0) * 
      ( prodparam - (sumparam*sumparam.t())/(mc+0.0) ) +
      sd_param(mc+1) * EPS;
    
  }
}

inline double logistic(double x, double l=1){
  return l/(1.0+exp(-x));
}

inline double logit(double x, double l=1){
  return log(x/(l-x));
}

inline arma::vec par_transf_fwd(arma::vec par){
  if(par.n_elem > 1){
    // gneiting nonsep 
    par(0) = log(par(0));
    par(1) = log(par(1));
    par(2) = logit(par(2));
    return par;
  } else {
    return log(par);
  }
}


inline arma::vec par_transf_back(arma::vec par){
  if(par.n_elem > 1){
    // gneiting nonsep 
    par(0) = exp(par(0));
    par(1) = exp(par(1));
    par(2) = logistic(par(2));
    return par;
  } else {
    return exp(par);
  }
}

inline arma::vec par_huvtransf_fwd(arma::vec par, int npars, double l=2){
  if(npars == 1){
    par(0) = log(par(0));
    return par;
  } else {
    // apanasovich&genton huv nonsep 
    par(0) = log(par(0));
    par(1) = logit(par(1));
    par(2) = log(par(2));
    par(3) = logit(par(3));
    par(4) = log(par(4));
    
    for(int j=5; j<par.n_elem; j++){
      par(j) = logit(par(j), l);
    }
    return par;
  }
}


inline arma::vec par_huvtransf_back(arma::vec par, int npars, double l=2){
  if(npars == 1){
    par(0) = exp(par(0));
    return par;
  } else {
    // apanasovich&genton huv nonsep 
    par(0) = exp(par(0));
    par(1) = logistic(par(1));
    par(2) = exp(par(2));
    par(3) = logistic(par(3));
    par(4) = exp(par(4));
    
    for(int j=5; j<par.n_elem; j++){
      par(j) = logistic(par(j), l);
    }
    return par;
  }
}


inline bool unif_bounds(arma::vec& par, const arma::mat& bounds){
  bool out_of_bounds = false;
  for(int i=0; i<par.n_elem; i++){
    arma::rowvec ibounds = bounds.row(i);
    if( par(i) < ibounds(0) ){
      out_of_bounds = true;
      par(i) = ibounds(0) + 1e-5;
    }
    if( par(i) > ibounds(1) ){
      out_of_bounds = true;
      par(i) = ibounds(1) - 1e-5;
    }
  }
  return out_of_bounds;
}

inline double lognormal_proposal_logscale(const double& xnew, const double& xold){
  // returns  + log x' - log x
  // to be + to log prior ratio log pi(x') - log pi(x)
  return log(xnew) - log(xold);
}

inline double normal_proposal_logitscale(const double& xnew, const double& xold, int l=1){
  return log(xnew * (l-xnew)) - log(xold * (l-xold));
}

inline double calc_jacobian(int k, const arma::vec& new_param, 
                            const arma::vec& param, int npars){
  
  if(npars == 1){
    return lognormal_proposal_logscale(new_param(0), param(0));
  } else {
    double norm_prop_logitbound_varsim = 0;
    for(int vj=0; vj<k; vj++){
      norm_prop_logitbound_varsim += normal_proposal_logitscale(new_param(5+vj), param(5+vj), 2);
    }
    
    double par1   = lognormal_proposal_logscale(new_param(0), param(0)); // 
    double nnsep1 = normal_proposal_logitscale(new_param(1), param(1)); // 
    double par2   = lognormal_proposal_logscale(new_param(2), param(2)); // 
    double nnsep2 = normal_proposal_logitscale(new_param(3), param(3)); // 
    double par3   = lognormal_proposal_logscale(new_param(4), param(4)); // 
    
    return par1 + par2 + par3 + nnsep1 + nnsep2 + 
      norm_prop_logitbound_varsim;
  }
}


inline double lognormal_logdens(const double& x, const double& m, const double& ssq){
  return -.5*(2*PI*ssq) - .5/ssq * pow(log(x) - m, 2) - log(x);
}

inline double gamma_logdens(const double& x, const double& a, const double& b){
  return -lgamma(a) + a*log(b) + (a-1)*log(x) - b*x;
}


