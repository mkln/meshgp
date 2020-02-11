#include "qmeshgp.h"
#include "interrupt_handler.h"

Rcpp::List gen_recovery_data(const MeshGP& mesh){
  
  Rcpp::List data = 
    Rcpp::List::create(
      Rcpp::Named("y_available") = mesh.y_available,
      Rcpp::Named("X_available") = mesh.X_available,
      Rcpp::Named("blocking") = mesh.blocking,
      Rcpp::Named("na_ix_all") = mesh.na_ix_all,
      Rcpp::Named("na_1_blocks") = mesh.na_1_blocks,
      Rcpp::Named("na_ix_blocks") = mesh.na_ix_blocks,
      Rcpp::Named("n_loc_ne_blocks") = mesh.n_loc_ne_blocks,
      Rcpp::Named("block_ct_obs") = mesh.block_ct_obs,
      Rcpp::Named("u_by_block_groups") = mesh.u_by_block_groups
    );
  
  Rcpp::List model = 
    Rcpp::List::create(
      Rcpp::Named("parents") = mesh.parents,
      Rcpp::Named("children") = mesh.children,
      Rcpp::Named("block_names") = mesh.block_names,
      Rcpp::Named("block_groups") = mesh.block_groups,
      Rcpp::Named("indexing") = mesh.indexing,
      Rcpp::Named("parents_indexing") = mesh.parents_indexing,
      Rcpp::Named("children_indexing") = mesh.children_indexing
    );
  
  Rcpp::List caching = 
    Rcpp::List::create(
      Rcpp::Named("coords_caching") = mesh.coords_caching,
      Rcpp::Named("coords_caching_ix") = mesh.coords_caching_ix,
      Rcpp::Named("parents_caching") = mesh.parents_caching,
      Rcpp::Named("parents_caching_ix") = mesh.parents_caching_ix,
      Rcpp::Named("kr_caching") = mesh.kr_caching,
      Rcpp::Named("kr_caching_ix") = mesh.kr_caching_ix,
      Rcpp::Named("gibbs_caching") = mesh.gibbs_caching,
      Rcpp::Named("gibbs_caching_ix") = mesh.gibbs_caching_ix
    );
  
  Rcpp::List params =
    Rcpp::List::create(
      Rcpp::Named("w") = mesh.w,
      Rcpp::Named("Bcoeff") = mesh.Bcoeff,
      Rcpp::Named("theta") = mesh.param_data.theta,
      Rcpp::Named("tausq_inv") = mesh.tausq_inv,
      Rcpp::Named("sigmasq") = mesh.sigmasq
    );
  
  Rcpp::List settings =
    Rcpp::List::create(
      Rcpp::Named("cached") = mesh.cached,
      Rcpp::Named("cached_gibbs") = mesh.cached_gibbs,
      Rcpp::Named("rfc_dep") = mesh.rfc_dep,
      Rcpp::Named("verbose") = mesh.verbose,
      Rcpp::Named("debug") = mesh.debug
    );
  
  return Rcpp::List::create(
      Rcpp::Named("model_data") = data,
      Rcpp::Named("model") = model,
      Rcpp::Named("caching") = caching,
      Rcpp::Named("params") = params,
      Rcpp::Named("settings") = settings
    );
  
}

//[[Rcpp::export]]
arma::mat list_mean(const arma::field<arma::mat>& x){
  int n = x.n_elem;
  int nrows = x(0).n_rows;
  int ncols = x(0).n_cols;
  
  arma::mat result = arma::zeros(nrows, ncols);
  
#pragma omp parallel for
  for(int j=0; j<nrows; j++){
    for(int h=0; h<ncols; h++){
      for(int i=0; i<n; i++){
        result(j,h) += x(i)(j,h)/(n+.0);
      }
    }
  }
  return result;
}

//[[Rcpp::export]]
Rcpp::List qmeshgp_mcmc(
                   const arma::mat& y, 
                   const arma::mat& X, 
                   const arma::mat& coords, 
                   const arma::uvec& blocking,
                   
                   const arma::field<arma::uvec>& parents,
                   const arma::field<arma::uvec>& children,
                   
                   const arma::vec& layer_names,
                   const arma::vec& layer_gibbs_group,
                   
                   const arma::field<arma::uvec>& indexing,
                   
                   const arma::vec& phi1_prior,
                   const arma::vec& phi2_prior,
                   
                   const arma::mat& start_w,
                   const arma::vec& theta,
                   const arma::vec& beta,
                   const double& tausq,
                   const double& sigmasq,
                   
                   const arma::mat& mcmcsd,
                   
                   const Rcpp::List& recover,
                   
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
  
  omp_set_num_threads(num_threads);
  std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_all = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_all = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point start_mcmc = std::chrono::steady_clock::now();
  std::chrono::steady_clock::time_point end_mcmc = std::chrono::steady_clock::now();
  
  std::chrono::steady_clock::time_point tick_mcmc = std::chrono::steady_clock::now();
  
  bool verbose_mcmc = printall;
  
  int npars = coords.n_cols <= 2? 1: 2; // phi or a/c/kappa [sigmasq, tausq have gibbs step]
  MeshGP mesh = MeshGP();
  
  bool recover_mesh = recover.length() > 0;
  if(recover_mesh){
    mesh = MeshGP(y, X, coords, blocking, recover);
  } else {
    mesh = MeshGP(y, X, coords, blocking,
                
                parents, children, layer_names, layer_gibbs_group,
                indexing,
                
                start_w, beta, theta, 1.0/tausq, sigmasq,
                
                cache, cache_gibbs, rfc,
                verbose, debug);
  }
  
  
  arma::mat b_mcmc = arma::zeros(X.n_cols, mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(1, mcmc_keep);
  arma::mat sigmasq_mcmc = arma::zeros(1, mcmc_keep);
  arma::mat theta_mcmc = arma::zeros(npars, mcmc_keep);
  arma::vec llsave = arma::zeros(mcmc_keep);
  
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
#pragma omp parallel for
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mesh.w.n_rows, 1);
    yhat_mcmc(i) = arma::zeros(mesh.y.n_rows, 1);
  }

  
  mesh.get_loglik_comps_w( mesh.param_data );
  mesh.get_loglik_comps_w( mesh.alter_data );
  
  arma::vec param = mesh.param_data.theta;
  double current_loglik = mesh.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  }
  
  double logaccept, prior_logratio;
  
  double propos_count = 0;
  double accept_count = 0;
  double accept_ratio = 0;
  double propos_count_local = 0;
  double accept_count_local = 0;
  double accept_ratio_local = 0;
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  int msaved = 0;
  bool interrupted = false;
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations." << endl;
  
  arma::vec sumparam = arma::zeros(npars);
  arma::mat prodparam = arma::zeros(npars, npars);
  arma::mat paramsd = mcmcsd; // proposal sd
  arma::vec sd_param = arma::zeros(mcmc +1); // mcmc sd
  
  double ll_upd_msg;
  
  Rcpp::List recovered;

  start_all = std::chrono::steady_clock::now();
  int m=0; int mx=0;
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
        current_loglik = mesh.param_data.loglik_w;
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
      start = std::chrono::steady_clock::now();
      if(sample_sigmasq){
        mesh.gibbs_sample_sigmasq();
        current_loglik = mesh.param_data.loglik_w;
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_sigmasq & verbose){
        Rcpp::Rcout << "[sigmasq] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. ";
        //Rcpp::Rcout << " >>>> CHECK from: " << mesh.param_data.loglik_w << endl;
        //mesh.get_loglik_comps_w( mesh.param_data );
        //Rcpp::Rcout << " >>>> CHECK with: " << mesh.param_data.loglik_w << endl;
        //current_loglik = mesh.param_data.loglik_w;
      }
  
      ll_upd_msg = current_loglik;
      start = std::chrono::steady_clock::now();
      if(sample_theta){
        propos_count++;
        propos_count_local++;
        
        // theta
        arma::vec new_param = param;
        
        new_param = exp(log(param) + paramsd * arma::randn(npars));
        
        bool out_unif_bounds = false;
        if(coords.n_cols == 2){
          // phi range
          if(new_param(0) < phi1_prior(0)){
            new_param(0) = phi1_prior(0) - .001;
            out_unif_bounds = true;
          } 
          if(new_param(0) > phi1_prior(1)){
            new_param(0) = phi1_prior(1) + .001;
            out_unif_bounds = true;
          }
        } else {
          // phi1 range
          if(new_param(0) < phi1_prior(0)){
            new_param(0) = phi1_prior(0) - .001;
            out_unif_bounds = true;
          } 
          if(new_param(0) > phi1_prior(1)){
            new_param(0) = phi1_prior(1) + .001;
            out_unif_bounds = true;
          }
          // phi2 range
          if(new_param(1) < phi2_prior(0)){
            new_param(1) = phi2_prior(0) - .001;
            out_unif_bounds = true;
          } 
          if(new_param(1) > phi2_prior(1)){
            new_param(1) = phi2_prior(1) + .001;
            out_unif_bounds = true;
          }
        }
        
        mesh.theta_update(mesh.alter_data, new_param); 
        mesh.get_loglik_comps_w( mesh.alter_data );
        
        double new_loglik = mesh.alter_data.loglik_w;
        current_loglik = mesh.param_data.loglik_w;
        
        //print_data(mesh.alter_data);
        //print_data(mesh.param_data);
        //Rcpp::Rcout << new_loglik << " <- " << current_loglik << endl;
        
        if(isnan(current_loglik)){
          Rcpp::Rcout << "At nan loglik: error. \n";
          throw 1;
        }
  
        prior_logratio = 0;//gamma_logdens(1.0/new_param(0), 2.0, 1.0/2.0) - gamma_logdens(1.0/param(0), 2.0, 1.0/2.0);  // sigmasq
        //lognormal_logdens(new_param(0), 0, 1) - lognormal_logdens(param(0), 0, 1);// + //sigmasq
        //lognormal_logdens(new_param(1), 0, 1) - lognormal_logdens(param(1), 0, 1) + //phi
        //gamma_logdens(new_param(npars-1), a, b) - gamma_logdens(param(npars-1), a, b);  // tausq_inv
        
        if(coords.n_cols == 2){
          logaccept = new_loglik - current_loglik + prior_logratio +
            lognormal_proposal_logscale(new_param(0), param(0));
        } else {
          logaccept = new_loglik - current_loglik + prior_logratio +
            lognormal_proposal_logscale(new_param(0), param(0)) + 
            lognormal_proposal_logscale(new_param(1), param(1));
        }
        
        bool accepted = do_I_accept(logaccept);
        if(out_unif_bounds){
          accepted = false;
        }
        
        if(accepted){
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          accept_count++;
          accept_count_local++;
          
          current_loglik = new_loglik;
          mesh.accept_make_change();
          param = new_param;
          
          std::chrono::steady_clock::time_point end_copy = std::chrono::steady_clock::now();
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted from " <<  ll_upd_msg << " to " << current_loglik << ", "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() << "us. " 
                        << endl << new_param.t() << endl;
            //mesh.get_loglik_w(mesh.param_data);
            //Rcpp::Rcout << " >>>> CHECK : " << mesh.param_data.loglik_w << endl;
          } 
          
        } else {
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << ")" << endl;
          }
        }
        
        accept_ratio = accept_count/propos_count;
        accept_ratio_local = accept_count_local/propos_count_local;
        
        if(adapting){
          adapt(log(param), sumparam, prodparam, paramsd, sd_param, m, accept_ratio); // **
        }
        
        if((m>0) & (mcmc > 100) & !printall){
          if(!(m % (mcmc / 10))){
            //Rcpp::Rcout << paramsd << endl;
            accept_count_local = 0;
            propos_count_local = 0;
            
            interrupted = checkInterrupt();
            if(interrupted){
              throw 1;
            }
            end_mcmc = std::chrono::steady_clock::now();
            if(true){
              
              int time_tick = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - tick_mcmc).count();
              int time_mcmc = std::chrono::duration_cast<std::chrono::milliseconds>(end_mcmc - start_mcmc).count();
              printf("%.1f%% %dms (total: %dms) ~ MCMC acceptance %.2f%% (total: %.2f%%) \n",
                     floor(100.0*(m+0.0)/mcmc),
                     time_tick,
                     time_mcmc,
                     accept_ratio_local*100, accept_ratio*100);

              //Rcpp::Rcout << "proposal sd: " << arma::trans(paramsd.diag())
              //            << "  theta now: " << param.t() << endl;
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
      
      if(printall){
        //Rcpp::checkUserInterrupt();
        interrupted = checkInterrupt();
        if(interrupted){
          throw 1;
        }
        //Rcpp::Rcout << m+1 <<  "-th iteration " <<
        int itertime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tick_mcmc ).count();

        printf("%5d-th iteration [ %dms ] ~ tsq=%.4f ssq=%.4f ", 
               m+1, itertime, 1.0/mesh.tausq_inv, mesh.sigmasq);
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
          sigmasq_mcmc.col(msaved) = mesh.sigmasq;
          b_mcmc.col(msaved) = mesh.Bcoeff;
          theta_mcmc.col(msaved) = mesh.param_data.theta;
          //yhat_mcmc.col(msaved) = mesh.X_concat * mesh.Bcoeff + mesh.w_concat;
          llsave(msaved) = current_loglik;
          
          w_mcmc(msaved) = mesh.w;
          yhat_mcmc(msaved) = mesh.X * mesh.Bcoeff + mesh.w;
          msaved++;
        }
      }
    }
    
    end_all = std::chrono::steady_clock::now();
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC done [" 
                << mcmc_time
                <<  "ms]" << endl;
  
    if(saving){
      recovered = gen_recovery_data(mesh);
    }
    
    
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc,
      Rcpp::Named("tausq_mcmc") = tausq_mcmc,
      Rcpp::Named("sigmasq_mcmc") = sigmasq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("paramsd") = paramsd,
      Rcpp::Named("ll") = llsave,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("recover") = recovered
      );
  
  } catch (...) {
    end_all = std::chrono::steady_clock::now();
  
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted. Returning partial saved results if any." << endl;
    
    if(msaved==0){
      msaved=1;
    }
    if(saving){
      recovered = gen_recovery_data(mesh);
    }
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc.cols(0, msaved-1),
      Rcpp::Named("tausq_mcmc") = tausq_mcmc.cols(0, msaved-1),
      Rcpp::Named("sigmasq_mcmc") = sigmasq_mcmc.cols(0, msaved-1),
      Rcpp::Named("theta_mcmc") = theta_mcmc.cols(0, msaved-1),
      Rcpp::Named("paramsd") = paramsd,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0,
      Rcpp::Named("recover") = recovered
      );
  }

}

//[[Rcpp::export]]
Rcpp::List qmeshgp_dry(
    const arma::mat& y,
    const arma::mat& X,
    const arma::mat& coords,
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    
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
    
    bool sample_beta=true,
    bool sample_tausq=true,
    bool sample_sigmasq=true,
    bool sample_theta=true,
    bool sample_w=true){
  
  MeshGP mesh(y, X, coords, blocking, 
      parents, children, layer_names, layer_gibbs_group, 
      indexing, 
      start_w, beta, theta, 1.0/tausq, sigmasq, 
      cache, cache_gibbs, rfc, 
      verbose, debug);
  
  return gen_recovery_data(mesh);
  
}


















