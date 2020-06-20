#include "qmeshgp_mv.h"
#include "interrupt_handler.h"
#include "mgp_utils.h"

//[[Rcpp::export]]
Rcpp::List qmeshgp_mv_mcmc(
    const arma::vec& y, 
    const arma::mat& X, 
    
    const arma::mat& coords, 
    const arma::uvec& mv_id,
    
    const arma::uvec& blocking,
    
    const arma::field<arma::uvec>& parents,
    const arma::field<arma::uvec>& children,
    
    const arma::vec& layer_names,
    const arma::vec& layer_gibbs_group,
    
    const arma::field<arma::uvec>& indexing,
    
    const arma::mat& set_unif_bounds_in,
    const arma::mat& beta_Vi,
    
    const arma::vec& tausq_ab,
    
    const arma::vec& start_w,
    const arma::vec& theta,
    const arma::vec& beta,
    const double& tausq,
    
    const arma::mat& mcmcsd,
    
    int mcmc_keep = 100,
    int mcmc_burn = 100,
    int mcmc_thin = 1,
    
    int num_threads = 1,
    
    bool adapting=false,
    bool cache=false,
    bool cache_gibbs=false,
    
    bool verbose=false,
    bool debug=false,
    bool printall=false,
    
    bool sample_beta=true,
    bool sample_tausq=true,
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
  
  arma::uvec mv_id_uniques = arma::unique(mv_id);
  int q  = mv_id_uniques.n_elem;//Z.n_cols;

  int k;
  int npars;
  double dlim=0;
  Rcpp::Rcout << "d=" << d << " q=" << q << ".\n";
  //Rcpp::Rcout << "Lower and upper bounds for priors:\n";
  
  if(d == 2){
    int n_cbase = q > 2? 3: 1;
    npars = 3*q + n_cbase;
  }
  
  k = q * (q-1)/2;
  
  Rcpp::Rcout << "Number of pars: " << npars << " plus " << k << " for multivariate\n"; 
  npars += k; // for xCovHUV + Dmat for variables (excludes sigmasq)
  
  /*
  if(set_unif_bounds.n_rows < npars){
    arma::mat vbounds = arma::zeros(k, 2);
    if(npars > 1+5){
      // multivariate
      dlim = sqrt(q+.0);
      vbounds.col(0) += 1e-5;
      vbounds.col(1) += dlim - 1e-5;
    } else {
      vbounds.col(0) += 1e-5;
      vbounds.col(1) += set_unif_bounds(0, 1);//.fill(arma::datum::inf);
    }
    set_unif_bounds = arma::join_vert(set_unif_bounds, vbounds);
  }*/
  // metropolis search limits
  arma::mat tsqi_unif_bounds = arma::zeros(q, 2);
  tsqi_unif_bounds.col(0).fill(1e-5);
  tsqi_unif_bounds.col(1).fill(1000-1e-5);
  //arma::mat set_unif_bounds = set_unif_bounds_in;
  arma::mat set_unif_bounds = arma::join_vert(set_unif_bounds_in, tsqi_unif_bounds);
  //Rcpp::Rcout << set_unif_bounds << endl;
  arma::mat metropolis_sd = arma::zeros(set_unif_bounds.n_rows, set_unif_bounds.n_rows);
  metropolis_sd.submat(0, 0, npars-1, npars-1) = mcmcsd;
  metropolis_sd.submat(npars, npars, npars+q-1, npars+q-1) = .1 * arma::eye(q, q);
  

  MeshGPmv mesh(y, X, coords, mv_id, blocking,
                
                parents, children, layer_names, layer_gibbs_group,
                indexing,
                
                start_w, beta, theta, 1.0/tausq, 
                beta_Vi, tausq_ab,
                
                cache, cache_gibbs, 
                verbose, debug);

  arma::cube b_mcmc = arma::zeros(X.n_cols, q, mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_keep);
  
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
  mesh.gibbs_sample_w(true);
  mesh.get_loglik_w(mesh.param_data);
  
  //arma::vec param = mesh.param_data.theta;
  arma::vec param = arma::join_vert(mesh.param_data.theta, mesh.tausq_inv );
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
  
  arma::vec sumparam = arma::zeros(param.n_elem); // include sigmasq
  arma::mat prodparam = arma::zeros(param.n_elem, param.n_elem);
  arma::mat paramsd = metropolis_sd; // proposal sd
  arma::vec sd_param = arma::zeros(mcmc +1); // mcmc sd
  
  double ll_upd_msg;
  bool needs_update = true;
  //Rcpp::List recovered;
  
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
        mesh.gibbs_sample_w(needs_update);
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
      if(sample_theta){
        propos_count++;
        propos_count_local++;
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          paramsd * arma::randn(param.n_elem), set_unif_bounds);
        
        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        arma::vec theta_proposal = new_param.subvec(0, npars-1);
        
        mesh.theta_update(mesh.alter_data, theta_proposal);
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
          jacobian  = calc_jacobian(new_param, param, set_unif_bounds);
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
          needs_update = true;
          
          if(sample_tausq & true){
            start = std::chrono::steady_clock::now();
            mesh.gibbs_sample_tausq();
            end = std::chrono::steady_clock::now();
            if(verbose_mcmc & sample_tausq & verbose){
              Rcpp::Rcout << "[tausq] " 
                          << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " 
                          << endl; 
            }
          }
          
          
          std::chrono::steady_clock::time_point end_copy = std::chrono::steady_clock::now();
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] accepted from " <<  ll_upd_msg << " to " << current_loglik << ", "
                        << std::chrono::duration_cast<std::chrono::microseconds>(end_copy - start_copy).count() << "us.\n"; 
          } 
          
        } else {
          if(verbose_mcmc & sample_theta & debug & verbose){
            Rcpp::Rcout << "[theta] rejected (log accept. " << logaccept << ")" << endl;
          }
          needs_update = false;
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
      if(sample_tausq & false){
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
        
        printf("%5d-th iteration [ %dms ] ~ MCMC acceptance %.2f%% (total: %.2f%%)\n", 
               m+1, itertime, 1.0/mesh.tausq_inv, accept_ratio_local*100, accept_ratio*100);
        for(int pp=0; pp<npars; pp++){
          printf("theta%1d=%.4f ", pp, mesh.param_data.theta(pp));
        }
        printf("\n");
        for(int pp=0; pp<q; pp++){
          printf("tsq%1d=%.4f ", pp, 1.0/mesh.tausq_inv(pp));
        }
        printf("~\n");
        
        tick_mcmc = std::chrono::steady_clock::now();
      }
      
      //save
      if(mx >= 0){
        if(mx % mcmc_thin == 0){
          tausq_mcmc.col(msaved) = 1.0 / mesh.tausq_inv;
          
          b_mcmc.slice(msaved) = mesh.Bcoeff;
          theta_mcmc.col(msaved) = mesh.param_data.theta;
          llsave(msaved) = current_loglik;
          
          w_mcmc(msaved) = mesh.w;
          Rcpp::RNGScope scope;
          yhat_mcmc(msaved) = mesh.XB + mesh.w + pow(1.0/mesh.tausq_inv_long, .5) % arma::randn(n);
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
      Rcpp::Named("pix") = mesh.parents_indexing,
      Rcpp::Named("paramsd") = paramsd,
      Rcpp::Named("ll") = llsave,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0
    );
    
  } catch (...) {
    end_all = std::chrono::steady_clock::now();
    
    double mcmc_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_all - start_all).count();
    Rcpp::Rcout << "MCMC has been interrupted. Returning partial saved results if any." << endl;
    
    if(msaved==0){
      msaved=1;
    }
    
    return Rcpp::List::create(
      Rcpp::Named("w_mcmc") = w_mcmc,
      Rcpp::Named("yhat_mcmc") = yhat_mcmc,
      Rcpp::Named("beta_mcmc") = b_mcmc.slices(0, msaved-1),
      Rcpp::Named("tausq_mcmc") = tausq_mcmc.cols(0, msaved-1),
      Rcpp::Named("theta_mcmc") = theta_mcmc.cols(0, msaved-1),
      Rcpp::Named("paramsd") = paramsd,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0
    );
  }
  
}
