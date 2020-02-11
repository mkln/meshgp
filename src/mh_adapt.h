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


inline double lognormal_proposal_logscale(const double& xnew, const double& xold){
  // returns  + log x' - log x
  // to be + to log prior ratio log pi(x') - log pi(x)
  return log(xnew) - log(xold);
}


inline double lognormal_logdens(const double& x, const double& m, const double& ssq){
  return -.5*(2*PI*ssq) - .5/ssq * pow(log(x) - m, 2) - log(x);
}

inline double gamma_logdens(const double& x, const double& a, const double& b){
  return -lgamma(a) + a*log(b) + (a-1)*log(x) - b*x;
}
