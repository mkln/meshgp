#include "qmeshgp_devel.h"
#include "interrupt_handler.h"
#include "mgp_utils.h"


//[[Rcpp::export]]
Rcpp::List qmeshgp_dev_mcmc(
    const arma::mat& y, 
    const arma::mat& X, 
    const arma::mat& Z,
    
    const arma::mat& coords, 
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    const arma::uvec& predictable_blocks,
    
    const arma::field<arma::uvec>& indexing_grid,
    const arma::field<arma::uvec>& indexing_obs,
    
    const arma::mat& set_unif_bounds_in,
    const arma::mat& beta_Vi,
    const arma::vec& sigmasq_ab,
    const arma::vec& tausq_ab,
    
    const arma::mat& start_w,
    const arma::vec& theta,
    const arma::vec& beta,
    const double& tausq,
    const double& sigmasq,
    
    const arma::mat& mcmcsd,
    
    int mcmc_keep = 100,
    int mcmc_burn = 100,
    int mcmc_thin = 1,
    
    int num_threads = 1,
    
    bool adapting=false,
    bool cache=false,
    bool cache_gibbs=false,
    bool rfc=false,
    bool verbose=false,
    bool debug=false,
    bool printall=false,
    bool saving=true,
    
    bool sample_beta=true,
    bool sample_tausq=true,
    bool sample_sigmasq=true,
    bool sample_theta=true,
    bool sample_w=true){
  
  Rcpp::Rcout << "Preparing for MCMC." << endl;
#ifdef _OPENMP
  omp_set_num_threads(num_threads);
#endif
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  
  bool verbose_mcmc = printall;
  
  double tempr=1;
  
  int n = coords.n_rows;
  int d = coords.n_cols;
  int q = Z.n_cols;
  
  int k;
  int npars;
  double dlim=0;
  Rcpp::Rcout << "d=" << d << " q=" << q << ".\n";
  Rcpp::Rcout << "Lower and upper bounds for priors:\n";
  arma::mat set_unif_bounds = set_unif_bounds_in;
  
  if(d == 2){
    if(q > 2){
      npars = 1 + 3;
    } else {
      npars = 1 + 1;
    }
  } else {
    if(q > 2){
      npars = 1+5;
    } else {
      npars = 1+3; // sigmasq + alpha + beta + phi
    }
  }
  
  k = q * (q-1)/2;
  
  Rcpp::Rcout << "Number of pars: " << npars << " plus " << k << " for multivariate\n"; 
  npars += k; // for xCovHUV + Dmat for variables (excludes sigmasq)
  
  Rcpp::Rcout << set_unif_bounds << endl;
  
  MeshGPdev mesh(y, X, Z, coords, blocking,
                  
                  parents, children, layer_names, layer_gibbs_group,
                  predictable_blocks,
                  indexing_grid, indexing_obs,
                  
                  start_w, beta, theta, 1.0/tausq, sigmasq,
                  beta_Vi, sigmasq_ab, tausq_ab,
                  
                  cache, cache_gibbs, rfc,
                  verbose, debug);
  
  arma::mat b_mcmc = arma::zeros(X.n_cols, mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(1, mcmc_keep);
  arma::mat theta_mcmc = arma::zeros(npars, mcmc_keep);
  arma::vec llsave = arma::zeros(mcmc_keep);
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mesh.w.n_rows, q);
    yhat_mcmc(i) = arma::zeros(mesh.y.n_rows, 1);
  }
  
  mesh.get_loglik_comps_w( mesh.param_data );
  mesh.get_loglik_comps_w( mesh.alter_data );
  /*if(sample_w){
    mesh.gibbs_sample_w();
    mesh.predict(true);
    mesh.get_loglik_w(mesh.param_data);
  }*/
  
  arma::vec param = mesh.param_data.theta;
  double current_loglik = tempr*mesh.param_data.loglik_w;
  //if(verbose & debug){
    Rcpp::Rcout << "starting from ll_wy: " << mesh.param_data.loglik_w << " " << mesh.alter_data.loglik_w << endl; 
    Rcpp::Rcout << "starting from ll_y: " << mesh.param_data.ll_y_all << " " << mesh.alter_data.ll_y_all << endl; 
  //}
  
  double logaccept;
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  
  MHAdapter adaptivemc(param.n_elem, mcmc, mcmcsd, .5);
  
  int msaved = 0;
  bool interrupted = false;
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations." << endl;
  
  
  double ll_upd_msg;
  
  
  Rcpp::List recovered;
  
  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0; int num_chol_fails=0;
  try { 
    for(m=0; m<mcmc; m++){
      
      mesh.predicting = false;
      mx = m-mcmc_burn;
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          mesh.predicting = true;
        } 
      }
      
      if(printall){
        tick_mcmc = std::chrono::steady_clock::now();
      }
      ll_upd_msg = current_loglik;
      
      start = std::chrono::steady_clock::now();
      if(sample_w){
        mesh.gibbs_sample_w();
        mesh.get_loglik_w(mesh.param_data);
        current_loglik = tempr*mesh.param_data.loglik_w;
        if(mesh.predicting){
          mesh.predict();
        }
        
      }
      
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_w & verbose){
        Rcpp::Rcout << "[w] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
        //mesh.get_loglik_w(mesh.param_data);
        //Rcpp::Rcout << " >>>> CHECK : " << mesh.param_data.loglik_w << endl;
      }
      
      
      ll_upd_msg = current_loglik;
      
      
      ll_upd_msg = current_loglik;
      start = std::chrono::steady_clock::now();
      if(sample_theta){
        adaptivemc.count_proposal();
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          adaptivemc.paramsd * arma::randn(param.n_elem), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        
        //mesh.alter_data.sigmasq = new_param(0);
        mesh.theta_update(mesh.alter_data, new_param); 
        
        mesh.get_loglik_comps_w( mesh.alter_data );
        
        bool accepted = !out_unif_bounds;
        double new_loglik = 0;
        double prior_logratio = 0;
        double jacobian = 0;
        
        if(!mesh.alter_data.cholfail){
          new_loglik = tempr*mesh.alter_data.loglik_w;
          current_loglik = tempr*mesh.param_data.loglik_w;
          
          
          
          if(std::isnan(current_loglik)){
            Rcpp::Rcout << "At nan loglik: error. \n";
            throw 1;
          }
          
          prior_logratio = calc_prior_logratio(new_param, param);
          jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
          
          logaccept = new_loglik - current_loglik + //prior_logratio + 
            //prior_logratio +
            jacobian;
          
          //Rcpp::Rcout << "new: " << new_loglik << " vs old: " << current_loglik << " logaccept: " << logaccept << endl;
          
          if(std::isnan(logaccept)){
            Rcpp::Rcout << new_param.t() << endl;
            Rcpp::Rcout << param.t() << endl;
            Rcpp::Rcout << new_loglik << " " << current_loglik << " " << jacobian << endl;
            throw 1;
          }
          
          accepted = do_I_accept(logaccept);
        } else {
          accepted = false;
          num_chol_fails ++;
          printf("[warning] chol failure #%d at mh proposal -- auto rejected\n", num_chol_fails);
          Rcpp::Rcout << new_param.t() << "\n";
        }
      
        if(accepted){
          
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          adaptivemc.count_accepted();
          
          current_loglik = new_loglik;
          mesh.accept_make_change();
          param = new_param;
          
          std::chrono::steady_clock::time_point end_copy = std::chrono::steady_clock::now();
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted from " <<  ll_upd_msg << " to " << current_loglik << ", "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() << "us.\n"; 
          } 
          
        } else {
          
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << ")" << endl;
          }
        }
        
        adaptivemc.update_ratios();
        
        if(adapting){
          adaptivemc.adapt(par_huvtransf_fwd(param, set_unif_bounds), m); // **
        }
        
        if((m>0) & (mcmc > 100)){
          if(!(m % (mcmc / 10))){
            
            
            interrupted = checkInterrupt();
            if(interrupted){
              throw 1;
            }
            end_mcmc = std::chrono::steady_clock::now();
            if(true){
              int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
              int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
              adaptivemc.print_summary(time_tick, time_mcmc, m, mcmc);
              
              tick_mcmc = std::chrono::steady_clock::now();
            }
          } 
        }
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_theta & verbose){
        Rcpp::Rcout << "[theta] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. ";
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_beta){
        mesh.gibbs_sample_beta();
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_beta & verbose){
        Rcpp::Rcout << "[beta] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
        if(verbose || debug){
          Rcpp::Rcout << endl;
        }
      }
      
      start = std::chrono::steady_clock::now();
      if(sample_tausq){
        mesh.gibbs_sample_tausq();
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_tausq & verbose){
        Rcpp::Rcout << "[tausq] " 
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " 
                    << endl; 
      }
      
      if(sample_tausq || sample_beta){
        mesh.update_ll_after_beta_tausq();
      }
      
      if(printall){
        //Rcpp::checkUserInterrupt();
        interrupted = checkInterrupt();
        if(interrupted){
          throw 1;
        }
        
        int itertime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tick_mcmc ).count();
        
        adaptivemc.print(itertime, m);
        
        for(int pp=0; pp<npars; pp++){
          printf("theta%1d=%.4f ", pp, mesh.param_data.theta(pp));
        }
        printf("\n");
        
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      //save
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          tausq_mcmc.col(msaved) = 1.0 / mesh.tausq_inv;
          
          b_mcmc.col(msaved) = mesh.Bcoeff;
          
          theta_mcmc.col(msaved) = mesh.param_data.theta;
          llsave(msaved) = current_loglik;
          
          w_mcmc(msaved) = mesh.w;
          Rcpp::RNGScope scope;
          yhat_mcmc(msaved) = mesh.X * mesh.Bcoeff + mesh.Zw + pow(1.0/mesh.tausq_inv, .5) * arma::randn(n);
          msaved++;
        }
      }
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC done [" 
                << mcmc_time
                <<  "ms]" << endl;
    
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("recover") = recovered
    );
    
  } catch (...) {
    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted." << endl;
    
    if(msaved==0){
      msaved=0;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("m") = msaved,
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc
    );
  }
  
}











