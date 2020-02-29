#include "qmeshgp_svc.h"
#include "interrupt_handler.h"
#include "mgp_utils.h"

Rcpp::List gen_recovery_data(const MeshGPsvc& mesh){
  Rcpp::List data = 
    Rcpp::List::create(
      Rcpp::Named("na_ix_all") = mesh.na_ix_all,
      Rcpp::Named("na_1_blocks") = mesh.na_1_blocks,
      Rcpp::Named("na_ix_blocks") = mesh.na_ix_blocks,
      Rcpp::Named("n_loc_ne_blocks") = mesh.n_loc_ne_blocks,
      Rcpp::Named("block_ct_obs") = mesh.block_ct_obs,
      Rcpp::Named("u_by_block_groups") = mesh.u_by_block_groups,
      Rcpp::Named("indexing") = mesh.indexing,
      Rcpp::Named("parents_indexing") = mesh.parents_indexing,
      Rcpp::Named("children_indexing") = mesh.children_indexing
    );
  
  Rcpp::List model = 
    Rcpp::List::create(
      Rcpp::Named("parents") = mesh.parents,
      Rcpp::Named("children") = mesh.children,
      Rcpp::Named("block_names") = mesh.block_names,
      Rcpp::Named("block_groups") = mesh.block_groups
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
Rcpp::List qmeshgp_svc_mcmc(
    const arma::mat& y, 
    const arma::mat& X, 
    const arma::mat& Z,
    
    const arma::mat& coords, 
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::mat& set_unif_bounds_in,
    
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
  
  Rcpp::Rcout << "Preparing for MCMC." << endl;
  omp_set_num_threads(num_threads);
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
    if(q == 1){
      npars = 1; // phi
    } else {
      npars = 5;
    }
  } else {
    if(q > 2){
      npars = 5;
    } else {
      npars = 3;
    }
  }
  
  k = q * (q-1)/2;
  
  Rcpp::Rcout << "Number of pars: " << npars << " plus " << k << " for multivariate\n"; 
  npars += k; // for xCovHUV + Dmat for variables (excludes sigmasq)

  if(set_unif_bounds.n_rows < npars){
    arma::mat vbounds = arma::zeros(k, 2);
    if(npars > 5){
      // multivariate
      dlim = sqrt(q+.0);
      vbounds.col(0) += 1e-5;
      vbounds.col(1) += dlim - 1e-5;
    } else {
      vbounds.col(0) += 1e-5;
      vbounds.col(1) += set_unif_bounds(0, 1);//.fill(arma::datum::inf);
    }
    set_unif_bounds = arma::join_vert(set_unif_bounds, vbounds);
  }
  Rcpp::Rcout << set_unif_bounds << endl;
  
  MeshGPsvc mesh = MeshGPsvc();
  bool recover_mesh = recover.length() > 0;
  if(recover_mesh){
    mesh = MeshGPsvc(y, X, Z, coords, blocking, recover);
  } else {
    mesh = MeshGPsvc(y, X, Z, coords, blocking,
                  
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
  
  // field avoids limit in size of objects -- ideally this should be a cube
  arma::field<arma::mat> w_mcmc(mcmc_keep);
  arma::field<arma::mat> yhat_mcmc(mcmc_keep);
  
#pragma omp parallel for
  for(int i=0; i<mcmc_keep; i++){
    w_mcmc(i) = arma::zeros(mesh.w.n_rows, q);
    yhat_mcmc(i) = arma::zeros(mesh.y.n_rows, 1);
  }
  
  mesh.get_loglik_comps_w( mesh.param_data );
  mesh.get_loglik_comps_w( mesh.alter_data );
  
  arma::vec param = mesh.param_data.theta;
  double current_loglik = tempr*mesh.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  }
  
  double logaccept;
  
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
        current_loglik = tempr*mesh.param_data.loglik_w;
        //Rcpp::Rcout << "loglik: " << current_loglik << "\n";
        //mesh.get_loglik_comps_w( mesh.param_data );
        //current_loglik = mesh.param_data.loglik_w;
        //Rcpp::Rcout << "recalc: " << mesh.param_data.loglik_w << "\n";
      }
      end = std::chrono::steady_clock::now();
      if(verbose_mcmc & sample_sigmasq & verbose){
        Rcpp::Rcout << "[sigmasq] "
                    << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. \n";
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
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          paramsd * arma::randn(npars), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        
        mesh.theta_update(mesh.alter_data, new_param); 
        mesh.get_loglik_comps_w( mesh.alter_data );
        
        bool accepted = !out_unif_bounds;
        double new_loglik = 0;
        double prior_logratio = 0;
        double jacobian = 0;
        
        if(!mesh.alter_data.cholfail){
          new_loglik = tempr*mesh.alter_data.loglik_w;
          current_loglik = tempr*mesh.param_data.loglik_w;
          
          if(isnan(current_loglik)){
            Rcpp::Rcout << "At nan loglik: error. \n";
            throw 1;
          }
          
          //prior_logratio = calc_prior_logratio(k, new_param, param, npars, dlim);
          jacobian       = calc_jacobian(new_param, param, set_unif_bounds);
          logaccept = new_loglik - current_loglik + //prior_logratio + 
            jacobian;
          
          if(isnan(logaccept)){
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
          
          accept_count++;
          accept_count_local++;
          
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
        
        accept_ratio = accept_count/propos_count;
        accept_ratio_local = accept_count_local/propos_count_local;
        
        if(adapting){
          adapt(par_huvtransf_fwd(param, set_unif_bounds), sumparam, prodparam, paramsd, sd_param, m, accept_ratio); // **
        }
        
        if((m>0) & (mcmc > 100)){
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
        
        int itertime = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now()-tick_mcmc ).count();
        
        printf("%5d-th iteration [ %dms ] ~ tsq=%.4f ssq=%.4f | MCMC acceptance %.2f%% (total: %.2f%%)\n", 
               m+1, itertime, 1.0/mesh.tausq_inv, mesh.sigmasq, accept_ratio_local*100, accept_ratio*100);
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
          llsave(msaved) = current_loglik;
          
          w_mcmc(msaved) = mesh.w;
          yhat_mcmc(msaved) = mesh.X * mesh.Bcoeff + mesh.Zw;
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


Rcpp::List qmeshgp_svc_dry(
    const arma::mat& y,
    const arma::mat& X,
    const arma::mat& Z,
    
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
  
  MeshGPsvc mesh(y, X, Z, coords, blocking, 
              parents, children, layer_names, layer_gibbs_group, 
              indexing, 
              start_w, beta, theta, 1.0/tausq, sigmasq, 
              cache, cache_gibbs, rfc, 
              verbose, debug);
  
  return gen_recovery_data(mesh);
  
}


















