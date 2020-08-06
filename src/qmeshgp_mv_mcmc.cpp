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
    const double& sigmasq,
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
  
  double tempr = 1;
  
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
    if(q == 1){
      npars = 2; //##
    } else {
      int n_cbase = q > 2? 3: 1;
      npars = 3*q + n_cbase; //##
    }
  } else {
    Rcpp::Rcout << "d>2 not implemented for multivariate outcomes, yet " << endl;
    throw 1;
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
  //arma::mat tsqi_unif_bounds = arma::zeros(q, 2);
  //tsqi_unif_bounds.col(0).fill(1e-5);
  //tsqi_unif_bounds.col(1).fill(1000-1e-5);
  //arma::mat set_unif_bounds = set_unif_bounds_in;
  arma::mat set_unif_bounds = set_unif_bounds_in;//.rows(arma::zeros<arma::uvec>(1));//arma::join_vert(set_unif_bounds_in, tsqi_unif_bounds);
  //Rcpp::Rcout << set_unif_bounds << endl;
  arma::mat metropolis_sd = arma::zeros(set_unif_bounds.n_rows, set_unif_bounds.n_rows);
  metropolis_sd.submat(0, 0, npars-1, npars-1) = mcmcsd.submat(0, 0, npars-1, npars-1);
  //metropolis_sd.submat(npars, npars, npars+q-1, npars+q-1) = .1 * arma::eye(q, q);
  
  
  MeshGPmv mesh(y, X, coords, mv_id, blocking,
                
                parents, children, layer_names, layer_gibbs_group,
                indexing,
                
                start_w, beta, sigmasq, theta, 1.0/tausq, 
                beta_Vi, tausq_ab,
                
                cache, cache_gibbs, 
                verbose, debug);

  arma::cube b_mcmc = arma::zeros(X.n_cols, q, mcmc_keep);
  arma::mat tausq_mcmc = arma::zeros(q, mcmc_keep);
  
  arma::vec sigmasq_mcmc = arma::zeros(mcmc_keep);
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
  /*if(sample_w){
    mesh.gibbs_sample_w(true);
    mesh.predict(true);
    mesh.get_loglik_w(mesh.param_data);
  }*/
  
  arma::vec param = mesh.param_data.theta;
  double current_loglik = tempr*mesh.param_data.loglik_w;
  if(verbose & debug){
    Rcpp::Rcout << "starting from ll: " << current_loglik << endl; 
  }
  
  double logaccept;
  
  /*
  double propos_count = 0;
  double accept_count = 0;
  double accept_ratio = 0;
  double propos_count_local = 0;
  double accept_count_local = 0;
  double accept_ratio_local = 0;
  */
  
  // adaptive params
  int mcmc = mcmc_thin*mcmc_keep + mcmc_burn;
  
  MHAdapter adaptivemc(param.n_elem, mcmc, metropolis_sd);
  
  int msaved = 0;
  bool interrupted = false;
  
  Rcpp::Rcout << "Running MCMC for " << mcmc << " iterations." << endl;
  
  //arma::vec sumparam = arma::zeros(param.n_elem); // include sigmasq
  //arma::mat prodparam = arma::zeros(param.n_elem, param.n_elem);
  //arma::mat paramsd = metropolis_sd; // proposal sd
  //arma::vec sd_param = arma::zeros(mcmc +1); // mcmc sd
  
  double ll_upd_msg;
  bool needs_update = true;
  
  //Rcpp::List recovered;
  
  arma::vec predict_theta = mesh.param_data.theta;
  
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
      
      if(sample_w){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_w(needs_update || true);
        mesh.get_loglik_w(mesh.param_data);
        current_loglik = tempr*mesh.param_data.loglik_w;
        
        if(mesh.predicting){
          bool predict_update = arma::approx_equal(predict_theta, mesh.param_data.theta, "absdiff", 1e-8);
          mesh.predict(true);
          predict_theta = mesh.param_data.theta;
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
      }
      
      
      if(sample_sigmasq || false){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_sigmasq();
        current_loglik = tempr*mesh.param_data.loglik_w;
        
        //Rcpp::Rcout << "loglik: " << current_loglik << "\n";
        mesh.get_loglik_comps_w( mesh.param_data );
        //current_loglik = mesh.param_data.loglik_w;
        //Rcpp::Rcout << "recalc: " << mesh.param_data.loglik_w << "\n";
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & sample_sigmasq & verbose){
          Rcpp::Rcout << "[sigmasq] "
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. \n";
          //Rcpp::Rcout << " >>>> CHECK from: " << mesh.param_data.loglik_w << endl;
          //mesh.get_loglik_comps_w( mesh.param_data );
          //Rcpp::Rcout << " >>>> CHECK with: " << mesh.param_data.loglik_w << endl;
          //current_loglik = mesh.param_data.loglik_w;
        }
      }

      ll_upd_msg = current_loglik;
      start = std::chrono::steady_clock::now();
      if(sample_theta){
        //propos_count++;
        //propos_count_local++;
        adaptivemc.count_proposal();
        
        // theta
        Rcpp::RNGScope scope;
        arma::vec new_param = param;
        
        new_param = par_huvtransf_back(par_huvtransf_fwd(param, set_unif_bounds) + 
          adaptivemc.paramsd * arma::randn(param.n_elem), set_unif_bounds);

        bool out_unif_bounds = unif_bounds(new_param, set_unif_bounds);
        arma::vec theta_proposal = new_param;
        
        mesh.theta_update(mesh.alter_data, theta_proposal);
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
            prior_logratio +
            jacobian;
          /*
          if(m == mcmc-1){
            Rcpp::Rcout << "param: " << param.t() << endl;
            Rcpp::Rcout << "new param: " << new_param.t() << endl;
            Rcpp::Rcout << "logaccept: " << logaccept << endl;
            Rcpp::Rcout << new_loglik << " vs " << current_loglik << endl;
            Rcpp::Rcout << "jacobian: " << jacobian << endl;
            Rcpp::Rcout << "prior: " << sigmasq_mhr << endl;
          }*/
          if(std::isnan(logaccept)){
            Rcpp::Rcout << new_param.t() << endl;
            Rcpp::Rcout << param.t() << endl;
            Rcpp::Rcout << new_loglik << " " << current_loglik << " " << jacobian << endl;
            throw 1;
          }
          
          accepted = do_I_accept(logaccept);
          
          //Rcpp::Rcout << "accepted? " << (accepted? "YES" : "NO") << endl;
          //Rcpp::Rcout << "---- "<< endl;
          
        } else {
          accepted = false;
          num_chol_fails ++;
          printf("[warning] chol failure #%d at mh proposal -- auto rejected\n", num_chol_fails);
          Rcpp::Rcout << new_param.t() << "\n";
        }
      
        if(accepted){
          std::chrono::steady_clock::time_point start_copy = std::chrono::steady_clock::now();
          
          adaptivemc.count_accepted();
          //accept_count++;
          //accept_count_local++;
          
          current_loglik = new_loglik;
          mesh.accept_make_change();
          param = new_param;
          needs_update = true;
          
          if(sample_tausq & false){
            start = std::chrono::steady_clock::now();
            mesh.gibbs_sample_tausq();
            end = std::chrono::steady_clock::now();
            if(verbose_mcmc & sample_tausq & verbose){
              Rcpp::Rcout << "[tausq] " 
                          << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " 
                          << endl; 
            }
          }
          if(sample_sigmasq & false){
            start = std::chrono::steady_clock::now();
            mesh.gibbs_sample_sigmasq();
            current_loglik = tempr*mesh.param_data.loglik_w;
            
            //Rcpp::Rcout << "loglik: " << current_loglik << "\n";
            mesh.get_loglik_comps_w( mesh.param_data );
            //current_loglik = mesh.param_data.loglik_w;
            //Rcpp::Rcout << "recalc: " << mesh.param_data.loglik_w << "\n";
            end = std::chrono::steady_clock::now();
            if(verbose_mcmc & sample_sigmasq & verbose){
              Rcpp::Rcout << "[sigmasq] "
                          << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. \n";
              //Rcpp::Rcout << " >>>> CHECK from: " << mesh.param_data.loglik_w << endl;
              //mesh.get_loglik_comps_w( mesh.param_data );
              //Rcpp::Rcout << " >>>> CHECK with: " << mesh.param_data.loglik_w << endl;
              //current_loglik = mesh.param_data.loglik_w;
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
        
        //accept_ratio = accept_count/propos_count;
        //accept_ratio_local = accept_count_local/propos_count_local;
        adaptivemc.update_ratios();
        
        if(adapting){
          adaptivemc.adapt(par_huvtransf_fwd(param, set_unif_bounds), m); // **
        }
        
        if((m>0) & (mcmc > 100)){
          if(!(m % (mcmc / 10))){
            //Rcpp::Rcout << paramsd << endl;
            //accept_count_local = 0;
            //propos_count_local = 0;
            
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
      
      if(sample_beta){
        start = std::chrono::steady_clock::now();
        mesh.gibbs_sample_beta();
        end = std::chrono::steady_clock::now();
        if(verbose_mcmc & sample_beta & verbose){
          Rcpp::Rcout << "[beta] " 
                      << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. "; 
          if(verbose || debug){
            Rcpp::Rcout << endl;
          }
        }
      }
      
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
          sigmasq_mcmc(msaved) = mesh.sigmasq;
          theta_mcmc.col(msaved) = mesh.param_data.theta;//arma::join_vert(mesh.sigmasq * arma::ones(1), mesh.param_data.theta);
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
      Rcpp::Named("sigmasq_mcmc") = sigmasq_mcmc,
      Rcpp::Named("theta_mcmc") = theta_mcmc,
      Rcpp::Named("pix") = mesh.parents_indexing,
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("ll") = llsave,
      Rcpp::Named("parents_indexing") = mesh.parents_indexing,
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
      Rcpp::Named("paramsd") = adaptivemc.paramsd,
      Rcpp::Named("mcmc_time") = mcmc_time/1000.0
    );
  }
  
}



void theta_transform(arma::vec& ai1, arma::vec& ai2, 
                     arma::vec& phi_i, arma::vec& thetamv,
                     arma::mat& Dmat,
                     const arma::vec& theta, int npars, int dd, int q){
  // from vector to all covariance components
  int k = theta.n_elem - npars; // number of cross-distances = p(p-1)/2
  
  arma::vec cparams = theta.subvec(0, npars - 1);//##arma::join_vert(sigmasq * arma::ones(1), data.theta.subvec(0, npars - 1));
  if((dd == 2) & (q == 1)){
    thetamv = cparams;//arma::join_vert(sigmasq * arma::ones(1), cparams);
  } else {
    int n_cbase = q > 2? 3: 1;
    ai1 = cparams.subvec(0, q-1);
    ai2 = cparams.subvec(q, 2*q-1);
    phi_i = cparams.subvec(2*q, 3*q-1);
    thetamv = cparams.subvec(3*q, 3*q+n_cbase-1);
    
    if(k>0){
      Dmat = vec_to_symmat(theta.subvec(npars, npars + k - 1));
    } else {
      Dmat = arma::zeros(1,1);
    }
  } 
}



arma::mat Kpred(const arma::vec& w_par,
                const arma::mat& coordsc,
                const arma::uvec& qv_blockc,
                const arma::mat& coordsx,
                const arma::uvec& qv_blockx,
                const arma::vec& theta,
                int npars, int dd, int q){
  
  arma::vec ai1, ai2, phi_i, thetamv;
  arma::mat Dmat;
  theta_transform(ai1, ai2, phi_i, thetamv, Dmat, theta, npars, dd, q);
  
  arma::mat Kcc = mvCovAG20107_cx(coordsc, qv_blockc,
                                  coordsc, qv_blockc,
                                  ai1, ai2, phi_i, thetamv, Dmat, true);
  arma::mat Kxxi = arma::inv_sympd(mvCovAG20107_cx(coordsx, qv_blockx,
                                                   coordsx, qv_blockx,
                                                   ai1, ai2, phi_i, thetamv, Dmat, true));
  arma::mat Kcx = mvCovAG20107_cx(coordsc, qv_blockc,
                                  coordsx, qv_blockx,
                                  ai1, ai2, phi_i, thetamv, Dmat, false);
  
  
  arma::vec result = arma::zeros(coordsc.n_rows);
  arma::vec rvec = arma::randn(coordsc.n_rows);
  arma::vec w_pred_mean = Kcx * Kxxi * w_par;
  arma::mat Rpred = Kcc - Kcx * Kxxi * Kcx.t();
  arma::vec Rpred_sqrtdiag = sqrt(abs(Rpred.diag()));
  
  result = w_pred_mean + Rpred_sqrtdiag % rvec;
  //for(int i=0; i<coordsc.n_rows; i++){
  //double sqrtR = sqrt(Rpred(i, i));
  //result(i) = w_pred_mean(i) + Rpred_sqrtdiag(i) * rvec(i);
  //}
  
  return result;
}

//[[Rcpp::export]]
Rcpp::List mvmesh_predict_base(const arma::mat& newcoords,
                               const arma::uvec& new_mv_id,
                               const arma::mat& newx,
                               const arma::cube& beta_mcmc,
                               const arma::mat& theta_mcmc,
                               const arma::field<arma::mat>& w_mcmc,
                               const arma::mat& tausq_mcmc,
                               const arma::field<arma::uvec>& indexing,
                               const arma::field<arma::uvec>& parents_indexing,
                               const arma::field<arma::uvec>& parents,
                               const arma::mat& coords,
                               const arma::uvec& block_ref,
                               const arma::uvec& mv_id,
                               int npars, int dd, int pp,
                               int n_threads = 10
){
  omp_set_num_threads(n_threads);
  
  int mcmc = w_mcmc.n_elem;
  int nout = newcoords.n_rows;
  arma::mat w_pred = arma::zeros(nout, mcmc);
  arma::mat y_pred = arma::zeros(nout, mcmc);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int b = beta_mcmc.n_rows;
  for(int i=0; i<nout; i++){
    Rcpp::Rcout << i << endl;
    
    arma::mat coordsc = newcoords.rows(i*oneuv);
    int mid = new_mv_id(i)-1;
    arma::uvec block_mvid = oneuv * mid;
    arma::mat newx_i = newx.rows(i * oneuv);
//#pragma omp parallel for
    for(int m=0; m<mcmc; m++){
      
      int uref = block_ref(i) - 1;
      
      arma::uvec block_par_index = parents_indexing(uref);
      
      if(parents(uref).n_elem <= dd){
        block_par_index = arma::join_vert(indexing(uref), block_par_index);
      } 
      
      arma::mat block_par_coords = coords.rows(block_par_index);
      arma::uvec block_par_mvid = mv_id.elem(block_par_index)-1;
      
      arma::vec theta = theta_mcmc.col(m);
      arma::vec w_par = w_mcmc(m).rows(block_par_index);
      
      arma::mat w_pred_im = Kpred(w_par, 
                     coordsc, block_mvid,
                     block_par_coords, block_par_mvid,
                     theta, npars, dd, pp);
      
      w_pred(i, m) = w_pred_im(0,0);
      
      arma::vec beta = beta_mcmc.subcube(0, mid, m, b-1, mid, m);
      double xbeta = arma::conv_to<double>::from(newx_i * beta);
      
      arma::vec rn = arma::randn(1);
      y_pred(i, m) = w_pred(i, m) + xbeta + sqrt(tausq_mcmc(mid, m)) * rn(0);
    }
  }
  
  return Rcpp::List::create(
    Rcpp::Named("w_pred") = w_pred,
    Rcpp::Named("y_pred") = y_pred
  );
}

//[[Rcpp::export]]
Rcpp::List mvmesh_predict_by_block_base(const arma::field<arma::mat>& newcoords,
                               const arma::field<arma::uvec>& new_mv_id,
                               const arma::field<arma::mat>& newx,
                               const arma::uvec& names,
                               
                               const arma::field<arma::mat>& w_mcmc,
                               const arma::mat& theta_mcmc,
                               const arma::cube& beta_mcmc,
                               const arma::mat& tausq_mcmc,
                               
                               const arma::field<arma::uvec>& indexing,
                               const arma::field<arma::uvec>& parents_indexing,
                               const arma::field<arma::uvec>& parents,
                               const arma::mat& coords,
                               
                               const arma::uvec& mv_id,
                               int npars, int dd, int pp,
                               int n_threads = 10
){
  omp_set_num_threads(n_threads);
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  
  int mcmc = w_mcmc.n_elem;
  int n_blocks = newcoords.n_elem;
  int b = beta_mcmc.n_rows;
  
  arma::field<arma::mat> w_pred(n_blocks);
  arma::field<arma::mat> y_pred(n_blocks);
  
  Rcpp::Rcout << "Predictions " << endl;

  
  #pragma omp parallel for
  for(int j=0; j<n_blocks; j++){

    int nout = newcoords(j).n_rows;
    w_pred(j) = arma::zeros(nout, mcmc);
    y_pred(j) = arma::zeros(nout, mcmc);
    
    int uref = names(j) - 1;
    arma::uvec block_par_index = parents_indexing(uref);
    
    if(parents(uref).n_elem <= dd){
      block_par_index = arma::join_vert(indexing(uref), block_par_index);
    }
    arma::mat block_par_coords = coords.rows(block_par_index);
    arma::uvec block_par_mvid = mv_id.elem(block_par_index)-1;
    
    arma::mat Kxxi, Kcc, Kcx;
    arma::vec theta_last = arma::zeros(theta_mcmc.n_rows);
    
    arma::mat coordsc = newcoords(j);
    arma::uvec block_mvid = new_mv_id(j) -1;
    
    for(int m=0; m<mcmc; m++){
      arma::vec theta = theta_mcmc.col(m);
      arma::vec w_par = w_mcmc(m).rows(block_par_index);
    
      arma::vec ai1, ai2, phi_i, thetamv;
      arma::mat Dmat;
      theta_transform(ai1, ai2, phi_i, thetamv, Dmat, theta, npars, dd, pp);
      
      if(!arma::approx_equal(theta, theta_last, "abs_diff", 1e-6)){
        Kxxi = arma::inv_sympd(mvCovAG20107_cx(block_par_coords, block_par_mvid,
                                               block_par_coords, block_par_mvid,
                                               ai1, ai2, phi_i, thetamv, Dmat, true));
        
        Kcc = mvCovAG20107_cx(coordsc, block_mvid,
                              coordsc, block_mvid,
                              ai1, ai2, phi_i, thetamv, Dmat, true);
        
        
        Kcx = mvCovAG20107_cx(coordsc, block_mvid,
                              block_par_coords, block_par_mvid,
                              ai1, ai2, phi_i, thetamv, Dmat, false);
        
        theta_last = theta;
      }
      
      
      arma::vec result = arma::zeros(coordsc.n_rows);
      arma::vec rvec = arma::randn(coordsc.n_rows);
      arma::vec w_pred_mean = Kcx * Kxxi * w_par;
      arma::mat Rpred = Kcc - Kcx * Kxxi * Kcx.t();
      arma::vec Rpred_sqrtdiag = sqrt(abs(Rpred.diag()));
      
      result = w_pred_mean + Rpred_sqrtdiag % rvec;
      
      w_pred(j).col(m) = result;
      
      for(int i=0; i<nout; i++){
        int mid = new_mv_id(j)(i)-1;
        arma::mat newx_i = newx(j).rows(i * oneuv);
        arma::vec beta = beta_mcmc.subcube(0, mid, m, b-1, mid, m);
        double xbeta = arma::conv_to<double>::from(newx_i * beta);
        
        arma::vec rn = arma::randn(1);
        y_pred(j)(i, m) = w_pred(j)(i, m) + xbeta + sqrt(tausq_mcmc(mid, m)) * rn(0);
      }
      
    }
    
  }
  return Rcpp::List::create(
    Rcpp::Named("w_pred") = w_pred,
    Rcpp::Named("y_pred") = y_pred
  );
}