#define ARMA_DONT_PRINT_ERRORS


#include "includes.h"
#include <RcppEigen.h>
#include <Eigen/CholmodSupport>

// with indexing
// without block extensions (obs with NA are left in)

using namespace std;

//#include <RcppArmadilloExtensions/sample.h>

//arma::uvec ashuffle(const arma::uvec& x){
  //  Rcpp::RNGScope scope;
//  return Rcpp::RcppArmadillo::sample(x, x.n_elem, false); 
//}

const double hl2pi = -.5 * log(2 * M_PI);

class MeshGPsvc {
public:
  // meta
  int n;
  int p;
  int q;
  int dd;
  int n_blocks;
  int npars;
  
  arma::uvec used;
  
  // data
  arma::mat y;
  arma::mat X;
  arma::mat Z;
  
  arma::mat y_available;
  arma::mat X_available;
  arma::mat Z_available;
  
  arma::mat coords;
  
  int twonu;
  
  // block membership
  arma::uvec blocking;
  
  arma::uvec reference_blocks; // named
  int n_ref_blocks;
  
  
  arma::field<arma::sp_mat> Zblock;
  arma::vec Zw;
  
  // indexing info
  arma::field<arma::uvec> indexing; 
  arma::field<arma::uvec> parents_indexing; 
  arma::field<arma::uvec> children_indexing;
  
  // NA data
  arma::field<arma::vec> na_1_blocks; // indicator vector by block
  arma::field<arma::uvec> na_ix_blocks;
  arma::uvec na_ix_all;
  int n_loc_ne_blocks;
  
  // regression
  arma::mat XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  // sigmasq tausq priors
  arma::vec sigmasq_ab;
  arma::vec tausq_ab;
  
  // dependence
  arma::field<arma::sp_mat> Ib;
  arma::field<arma::uvec>   parents; // i = parent block names for i-labeled block (not ith block)
  arma::field<arma::uvec>   children; // i = children block names for i-labeled block (not ith block)
  arma::vec                 block_names; //  i = block name (+1) of block i. all dependence based on this name
  arma::uvec                ref_block_names;
  
  arma::vec                 block_groups; // same group = sample in parallel given all others
  arma::vec                 block_ct_obs; // 0 if no available obs in this block, >0=count how many available
  int                       n_gibbs_groups;
  arma::field<arma::vec>    u_by_block_groups;
  arma::uvec                u_predicts;
  int                       predict_group_exists;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  arma::field<arma::vec>    dim_by_parent;
  
  // params
  arma::mat w;
  arma::vec Bcoeff; // sampled
  double    tausq_inv;
  double    sigmasq;
  
  arma::mat rand_norm_mat;
  
  // params with mh step
  MeshData param_data; 
  MeshData alter_data;
  
  // setup
  bool predicting;
  bool rfc_dep;
  bool cached;
  bool cached_gibbs;
  bool verbose;
  bool debug;
  
  // debug var
  Rcpp::List debug_stuff; 
  
  void message(string s);
  
  // init / indexing
  void init_indexing();
  void init_meshdata(double, const arma::vec&);
  void init_finalize();
  void fill_zeros_Kcache();
  void na_study();
  void make_gibbs_groups();
  
  
  
  // init / caching obj
  void init_cache();
  
  // caching
  arma::vec coords_caching; 
  arma::vec coords_caching_ix;
  arma::vec parents_caching;
  arma::vec parents_caching_ix;
  arma::vec kr_caching;
  arma::vec kr_caching_ix;
  // caching of w sampling covariance matrices
  arma::vec gibbs_caching;
  arma::vec gibbs_caching_ix;

  // caching some matrices // ***
  arma::field<arma::mat> K_coords_cache;
  //arma::field<arma::mat> K_parents_cache_noinv;
  arma::field<arma::mat> Kcp_cache;
  
  // MCMC
  void get_loglik_w(MeshData& data);
  bool get_loglik_comps_w(MeshData& data);
  bool get_cond_comps_loglik_w(MeshData& data);
  bool get_cond_comps_loglik_w_nocache(MeshData& data);
  void get_loglik_comps_y_collapsed(MeshData& data);
  
  void gibbs_sample_w();
  void gibbs_sample_w_omp();
  void gibbs_sample_w_omp_nocache();
  void predict();
  
  void gibbs_sample_beta();
  void gibbs_sample_sigmasq();
  void gibbs_sample_tausq();
  
  // changing the values, no sampling
  void tausq_update(double);
  arma::vec cparams;
  arma::mat Dmat;
  void theta_transform(const MeshData&);
  void theta_update(MeshData&, const arma::vec&);
  void beta_update(const arma::vec&);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  MeshGPsvc();
  
  // build everything
  MeshGPsvc(
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    const arma::mat& Z_in,
    const arma::mat& coords_in, 
    const arma::uvec& blocking_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& layers_names,
    const arma::vec& block_group_in,
    
    const arma::field<arma::uvec>& indexing_in,
    
    const arma::mat& w_in,
    const arma::vec& beta_in,
    const arma::vec& theta_in,
    double tausq_inv_in,
    double sigmasq_in,
    
    int twonu_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& sigmasq_ab_in,
    const arma::vec& tausq_ab_in,
    
    bool use_cache,
    bool use_cache_gibbs,
    bool use_rfc,
    bool v,
    bool debugging);
  
  // init from previously initialized object
  MeshGPsvc(
    const arma::mat& y_in, 
    const arma::mat& X_in,
    const arma::mat& Z_in,
    const arma::mat& coords_in, 
    const arma::uvec& blocking_in,
    const Rcpp::List& recover);
  
};

void MeshGPsvc::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

MeshGPsvc::MeshGPsvc(){
  
}

MeshGPsvc::MeshGPsvc(
  const arma::mat& y_in, 
  const arma::mat& X_in, 
  const arma::mat& Z_in,
  
  const arma::mat& coords_in, 
  const arma::uvec& blocking_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  
  const arma::field<arma::uvec>& indexing_in,
  
  const arma::mat& w_in,
  const arma::vec& beta_in,
  const arma::vec& theta_in,
  double tausq_inv_in,
  double sigmasq_in,
  
  int twonu_in,
  
  const arma::mat& beta_Vi_in,
  const arma::vec& sigmasq_ab_in,
  const arma::vec& tausq_ab_in,
  
  bool use_cache=true,
  bool use_cache_gibbs=false,
  bool use_rfc=false,
  bool v=false,
  bool debugging=false){
  
  printf("~ MeshGPsvc initialization.\n");
  
  start_overall = std::chrono::steady_clock::now();
  
  cached = use_cache;
  if(cached){
    cached_gibbs = use_cache_gibbs;
    printf("~ caching ON: convenient if locations on (semi)regular grid.\n");
  } else {
    cached_gibbs = false;
    printf("~ caching OFF: consider enabling if locations are (semi)regularly spaced.\n");
  }
  
  verbose = v;
  debug = debugging;
  
  message("MeshGPsvc::MeshGPsvc assign values.");
  
  y                   = y_in;
  X                   = X_in;
  Z                   = Z_in;
  
  coords              = coords_in;
  blocking            = blocking_in;
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_blocks            = block_names.n_elem;
  
  Rcpp::Rcout << n_gibbs_groups << " groups for gibbs " << endl;
  //Rcpp::Rcout << block_groups_labels << endl;
  
  na_ix_all   = arma::find_finite(y.col(0));
  y_available = y.rows(na_ix_all);
  X_available = X.rows(na_ix_all); 
  Z_available = Z.rows(na_ix_all);
  
  
  
  indexing    = indexing_in;
  
  n  = na_ix_all.n_elem;
  p  = X.n_cols;
  q  = Z.n_cols;
  dd = coords.n_cols;
  
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
  
  printf("%d observed locations, %d to predict, %d total\n",
         n, y.n_elem-n, y.n_elem);
  
  // init
  dim_by_parent       = arma::field<arma::vec> (n_blocks);
  Ib                  = arma::field<arma::sp_mat> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  
  tausq_inv        = tausq_inv_in;
  
  Bcoeff           = beta_in;
  w                = w_in;
  Zw = armarowsum(Z % w);
  
  twonu = twonu_in;
  
  predicting = true;
  rfc_dep    = use_rfc;
  
  
  // now elaborate
  message("MeshGPsvc::MeshGPsvc : init_indexing()");
  init_indexing();
  
  message("MeshGPsvc::MeshGPsvc : na_study()");
  na_study();
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("MeshGPsvc::MeshGPsvc : init_meshdata()");
  init_meshdata(sigmasq_in, theta_in);
  sigmasq = sigmasq_in;
  
  message("MeshGPsvc::MeshGPsvc : init_finalize()");
  init_finalize();
  
  // prior for beta
  XtX   = X_available.t() * X_available;
  Vi    = beta_Vi_in;//.00001 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  // priors for tausq and sigmasq
  sigmasq_ab = sigmasq_ab_in;
  tausq_ab = tausq_ab_in;
  
  message("MeshGPsvc::MeshGPsvc : make_gibbs_groups()");
  make_gibbs_groups();
  
  //caching;
  if(cached){
    message("MeshGPsvc::MeshGPsvc : init_cache()");
    init_cache();
    fill_zeros_Kcache();
  }
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "MeshGPsvc::MeshGPsvc initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}

MeshGPsvc::MeshGPsvc(const arma::mat& y_in, 
               const arma::mat& X_in, 
               const arma::mat& Z_in,
               const arma::mat& coords_in, 
               const arma::uvec& blocking_in,
               const Rcpp::List& recover
){
  
  printf("~ MeshGPsvc initialization from provided initialization data.\n");
  
  start_overall = std::chrono::steady_clock::now();
  
  y                   = y_in;
  X                   = X_in;
  Z                   = Z_in;
  coords              = coords_in;
  blocking            = blocking_in;
  
  Rcpp::List model_data     = recover["model_data"];
  Rcpp::List model          = recover["model"];
  Rcpp::List model_caching  = recover["caching"];
  Rcpp::List model_params   = recover["params"];
  Rcpp::List model_settings = recover["settings"];
  
  printf("[1 ");
  na_ix_all   = arma::find_finite(y.col(0));
  y_available = y.rows(na_ix_all);
  X_available = X.rows(na_ix_all); 
  Z_available = Z.rows(na_ix_all);
  
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  printf("2 ");
  parents           = Rcpp::as<arma::field<arma::uvec> >(Rcpp::wrap(model["parents"])); 
  children          = Rcpp::as<arma::field<arma::uvec> >(Rcpp::wrap(model["children"]));
  block_names       = Rcpp::as<arma::vec>(Rcpp::wrap(model["block_names"]));
  block_groups      = Rcpp::as<arma::vec>(Rcpp::wrap(model["block_groups"]));
  
  printf("3 ");
  indexing          = Rcpp::as<arma::field<arma::uvec> >(Rcpp::wrap(model_data["indexing"]));
  parents_indexing  = Rcpp::as<arma::field<arma::uvec> >(Rcpp::wrap(model_data["parents_indexing"])); 
  children_indexing = Rcpp::as<arma::field<arma::uvec> >(Rcpp::wrap(model_data["children_indexing"]));
  
  printf("4 ");
  na_1_blocks       = Rcpp::as<arma::field<arma::vec> >(Rcpp::wrap(model_data["na_1_blocks"]));
  na_ix_blocks      = Rcpp::as<arma::field<arma::uvec> >(Rcpp::wrap(model_data["na_ix_blocks"]));
  n_loc_ne_blocks   = Rcpp::as<int>(Rcpp::wrap(model_data["n_loc_ne_blocks"]));
  block_ct_obs      = Rcpp::as<arma::vec>(Rcpp::wrap(model_data["block_ct_obs"]));
  u_by_block_groups = Rcpp::as<arma::field<arma::vec> >(Rcpp::wrap(model_data["u_by_block_groups"]));
  
  printf("5 ");
  coords_caching     = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["coords_caching"]));
  coords_caching_ix  = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["coords_caching_ix"]));
  parents_caching    = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["parents_caching"]));
  parents_caching_ix = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["parents_caching_ix"]));
  kr_caching         = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["kr_caching"]));
  kr_caching_ix      = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["kr_caching_ix"]));
  gibbs_caching      = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["gibbs_caching"]));
  gibbs_caching_ix   = Rcpp::as<arma::vec>(Rcpp::wrap(model_caching["gibbs_caching_ix"]));
  
  printf("6 ");
  w                = Rcpp::as<arma::mat>(Rcpp::wrap(model_params["w"]));
  Bcoeff           = Rcpp::as<arma::vec>(Rcpp::wrap(model_params["Bcoeff"]));
  param_data.theta = Rcpp::as<arma::vec>(Rcpp::wrap(model_params["theta"]));
  tausq_inv        = Rcpp::as<double>(Rcpp::wrap(model_params["tausq_inv"]));
  param_data.sigmasq = Rcpp::as<double>(Rcpp::wrap(model_params["sigmasq"]));
  Zw               = armarowsum(Z % w);
  
  Vi    = Rcpp::as<arma::mat>(Rcpp::wrap(model_params["Vi"]));
  sigmasq_ab = Rcpp::as<arma::vec>(Rcpp::wrap(model_params["sigmasq_ab"]));
  tausq_ab   = Rcpp::as<arma::vec>(Rcpp::wrap(model_params["tausq_ab"]));
  
  printf("7 ");
  cached       = Rcpp::as<bool>(Rcpp::wrap(model_settings["cached"]));
  cached_gibbs = Rcpp::as<bool>(Rcpp::wrap(model_settings["cached_gibbs"]));
  rfc_dep      = Rcpp::as<bool>(Rcpp::wrap(model_settings["rfc_dep"]));
  verbose      = Rcpp::as<bool>(Rcpp::wrap(model_settings["verbose"])); 
  debug        = Rcpp::as<bool>(Rcpp::wrap(model_settings["debug"]));
  predicting   = true;
  
  n                   = na_ix_all.n_elem;
  p                   = X.n_cols;
  q                   = Z.n_cols;
  dd                  = coords.n_cols;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_blocks            = block_names.n_elem;
  XtX                 = X_available.t() * X_available;
  
  bprim               = arma::zeros(p);
  Vim                 = Vi * bprim;
  
  
  printf("8 ");
  if(dd == 2){
    if(q > 2){
      npars = 3;
    } else {
      npars = 1;
    }
  } else {
    if(q > 2){
      npars = 5;
    } else {
      npars = 3; // sigmasq + alpha + beta + phi
    }
  }
  
  fill_zeros_Kcache();
  
  // Zblock
  Zblock = arma::field<arma::sp_mat> (n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    Zblock(u) = Zify( Z.rows(indexing(u)) );
  }
  
  printf("9 ");
  // init_finalize
  dim_by_parent       = arma::field<arma::vec> (n_blocks);
  Ib                  = arma::field<arma::sp_mat> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  init_finalize();
  
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_cholprec = arma::field<arma::mat> (n_blocks);
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  
  alter_data = param_data; 
  
  printf("10]\n");
  
  printf("%d observed locations, %d to predict, %d total\n",
         n, y.n_elem-n, y.n_elem);
  
  message("Finished initializing from recovered data.");
}

void MeshGPsvc::make_gibbs_groups(){
  // checks -- errors not allowed. use check_groups.cpp to fix errors.
  for(int g=0; g<n_gibbs_groups; g++){
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_groups(u) == block_groups_labels(g)){
        if(indexing(u).n_elem > 0){ //**
          
          for(int pp=0; pp<parents(u).n_elem; pp++){
            if(block_groups(parents(u)(pp)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " <--- " << parents(u)(pp) 
                          << ": same group (" << block_groups(u) 
                          << ")." << endl;
              throw 1;
            }
          }
          for(int cc=0; cc<children(u).n_elem; cc++){
            if(block_groups(children(u)(cc)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                          << ": same group (" << block_groups(u) 
                          << ")." << endl;
              throw 1;
            }
          }
        }
      }
    }
  }
  
  int gx=0;
  
  if(n_blocks > n_ref_blocks){
    u_predicts = arma::zeros<arma::uvec>(n_blocks - n_ref_blocks);
    predict_group_exists = 1;
  } else {
    predict_group_exists = 0;
  }
  
  Rcpp::Rcout << "predict group? " << predict_group_exists << endl;
  Rcpp::Rcout << "n. groups " << n_gibbs_groups << endl;
  
  arma::field<arma::vec> u_by_block_groups_temp(n_gibbs_groups);
  u_by_block_groups = arma::field<arma::vec>(n_gibbs_groups-predict_group_exists);
  /// create list of groups for gibbs
  
  for(int g=0; g<n_gibbs_groups; g++){
    u_by_block_groups_temp(g) = arma::zeros(0);
    
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      
      if(block_groups(u) == block_groups_labels(g)){
        if(block_ct_obs(u) > 0){ //**
          
          arma::vec uhere = arma::zeros(1) + u;
          u_by_block_groups_temp(g) = arma::join_vert(u_by_block_groups_temp(g), uhere);
        } 
      }
      
    }
    if(u_by_block_groups_temp(g).n_elem > 0){
      u_by_block_groups(gx) = u_by_block_groups_temp(g);
      gx ++;
    }
  }
  
  Rcpp::Rcout << "next loop" << endl;

  
  if(predict_group_exists == 1){
    int p=0; 
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i) - 1;
      if(block_ct_obs(u) == 0){
        u_predicts(p) = u;
        p ++;
      }
    }
    
  }
  //u_by_block_groups = u_by_block_groups_temp;
}

void MeshGPsvc::na_study(){
  // prepare stuff for NA management
  printf("~ NA management for predictions\n");
  
  na_1_blocks = arma::field<arma::vec> (n_blocks);//(y_blocks.n_elem);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);//(y_blocks.n_elem);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros(n_blocks);//(y_blocks.n_elem);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks;i++){//y_blocks.n_elem; i++){
    arma::vec yvec = y.rows(indexing(i));//y_blocks(i);
    na_1_blocks(i) = arma::zeros(yvec.n_elem);
    na_1_blocks(i).elem(arma::find_finite(yvec)).fill(1);
    na_ix_blocks(i) = arma::find(na_1_blocks(i) == 1); 
    
  }
  
  n_ref_blocks = 0;
  for(int i=0; i<n_blocks;i++){//y_blocks.n_elem; i++){
    block_ct_obs(i) = arma::accu(na_1_blocks(i));
    if(block_ct_obs(i) > 0){
      n_loc_ne_blocks += indexing(i).n_elem;//coords_blocks(i).n_rows;
      n_ref_blocks += 1;
    }
  }
  
  int j=0;
  reference_blocks = arma::zeros<arma::uvec>(n_ref_blocks);
  ref_block_names = arma::zeros<arma::uvec>(n_ref_blocks);
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i) - 1;
    if(block_ct_obs(u) > 0){
      reference_blocks(j) = i;
      ref_block_names(j) = u;
      j ++;
    } 
  }
  
}

void MeshGPsvc::fill_zeros_Kcache(){
  // ***
  K_coords_cache = arma::field<arma::mat> (coords_caching.n_elem);
  Kcp_cache = arma::field<arma::mat> (kr_caching.n_elem);
  //K_parents_cache_noinv = arma::field<arma::mat> (parents_caching.n_elem);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    K_coords_cache(i) = arma::zeros(q*indexing(u).n_elem, q*indexing(u).n_elem);
  }
  
  
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    Kcp_cache(i) = arma::zeros(q*indexing(u).n_elem, q*parents_indexing(u).n_elem);
  }
}

void MeshGPsvc::init_cache(){
  // coords_caching stores the layer names of those layers that are representative
  // coords_caching_ix stores info on which layers are the same in terms of rel. distance
  
  printf("~ Starting to search block duplicates for caching. ");
  
  Rcpp::Rcout << "Coords";
  //coords_caching_ix = caching_pairwise_compare_uc(coords_blocks, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching_ix = caching_pairwise_compare_uci(coords, indexing, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching = arma::unique(coords_caching_ix);
  
  Rcpp::Rcout << ". Parent coords";
  //parents_caching_ix = caching_pairwise_compare_uc(parents_coords, block_names, block_ct_obs);
  parents_caching_ix = caching_pairwise_compare_uci(coords, parents_indexing, block_names, block_ct_obs);
  parents_caching = arma::unique(parents_caching_ix);
  
  Rcpp::Rcout << ". KR coords";
  arma::field<arma::mat> kr_pairing(n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents_indexing(u).n_elem > 0){//parents_coords(u).n_rows > 0){
      arma::mat cmat = coords.rows(indexing(u));
      arma::mat pmat = coords.rows(parents_indexing(u));
      kr_pairing(u) = arma::join_vert(cmat, pmat);
    } else {
      kr_pairing(u) = arma::zeros(arma::size(parents_indexing(u)));//arma::zeros(arma::size(parents_coords(u))); // no parents
    }
  }
  kr_caching_ix = caching_pairwise_compare_uc(kr_pairing, block_names, block_ct_obs);
  //kr_caching_ix = caching_pairwise_compare_u(kr_pairing, block_names, block_ct_obs);
  kr_caching = arma::unique(kr_caching_ix);
  
  Rcpp::Rcout << "." << endl;
  
  if(cached_gibbs){
    arma::field<arma::mat> gibbs_pairing(n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i = 0; i<n_blocks; i++){
      int u = block_names(i)-1;
      arma::mat cmat = coords.rows(indexing(u));
      gibbs_pairing(u) = arma::join_vert(cmat.rows(na_ix_blocks(u)), cmat); 
      
      if(parents_indexing(u).n_elem){//parents_coords(u).n_rows > 0){
        arma::mat pmat = coords.rows(parents_indexing(u));
        gibbs_pairing(u) = arma::join_vert(gibbs_pairing(u), pmat);//parents_coords(u)); // Sigi_p
      } 
      if(children(u).n_elem>0){
        arma::mat chmat = coords.rows(children_indexing(u));
        gibbs_pairing(u) = arma::join_vert(gibbs_pairing(u), chmat);//children_coords(u));
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          arma::mat pcmat = coords.rows(parents_indexing(child));
          gibbs_pairing(u) = arma::join_vert(gibbs_pairing(u), pcmat);//parents_coords(child));
        }
      }
    }
    //gibbs_caching_ix = caching_pairwise_compare_uc(gibbs_pairing, block_names, block_ct_obs);
    gibbs_caching_ix = caching_pairwise_compare_u(gibbs_pairing, block_names, block_ct_obs);
    gibbs_caching = arma::unique(gibbs_caching_ix);
  } else {
    gibbs_caching = arma::zeros(0);
  }
  
  double c_cache_perc = 100- 100*(coords_caching.n_elem+0.0) / n_blocks;
  double p_cache_perc = 100- 100*(parents_caching.n_elem+0.0) / n_blocks;
  double k_cache_perc = 100- 100*(kr_caching.n_elem+0.0) / n_blocks;
  double g_cache_perc = 0;
  
  if(cached_gibbs){
    g_cache_perc = 100- 100*(gibbs_caching.n_elem+0.0) / n_blocks;
  }
  
  printf("~ Caching stats c: %d [%.2f%%] / p: %d [%.2f%%] / k: %d [%.2f%%] / g: %d [%.2f%%] \n",
         coords_caching.n_elem, c_cache_perc, 
         parents_caching.n_elem, p_cache_perc, 
         kr_caching.n_elem, k_cache_perc, 
         gibbs_caching.n_elem, g_cache_perc );
  
}

void MeshGPsvc::init_meshdata(double sigmasq_in, const arma::vec& theta_in){
  // block params
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_cholprec = arma::field<arma::mat> (n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    param_data.w_cond_mean_K(i) = arma::zeros(q*indexing(i).n_elem, q*parents_indexing(i).n_elem);
    param_data.w_cond_cholprec(i) = arma::zeros(q*indexing(i).n_elem, q*indexing(i).n_elem);
  }
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  param_data.theta          = theta_in; //arma::join_vert(arma::ones(1) * sigmasq_in, theta_in);
  param_data.cholfail       = false;
  param_data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  param_data.sigmasq          = sigmasq_in;
  alter_data                = param_data; 
}

void MeshGPsvc::init_indexing(){
  
  Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
  printf("~ Indexing\n");
  message("[init_indexing] indexing, parent_indexing, children_indexing");
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));//arma::find( blocking == parents(u)(pi)+1 ); // parents are 0-indexed 
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
    }
    if(children(u).n_elem > 0){
      arma::field<arma::uvec> cixs(children(u).n_elem);
      for(int ci=0; ci<children(u).n_elem; ci++){
        cixs(ci) = indexing(children(u)(ci));//arma::find( blocking == children(u)(ci)+1 ); // children are 0-indexed 
      }
      children_indexing(u) = field_v_concat_uv(cixs);
    }
    Zblock(u) = Zify( Z.rows(indexing(u)) );
  }
  
}

void MeshGPsvc::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem > 0){
      //Rcpp::Rcout << "Ib " << parents(u).n_elem << endl;
      //Ib(u) = arma::eye<arma::sp_mat>(coords_blocks(u).n_rows, coords_blocks(u).n_rows);
      Ib(u) = arma::eye<arma::sp_mat>(indexing(u).n_elem, indexing(u).n_elem);
      for(int j=0; j<Ib(u).n_cols; j++){
        if(na_1_blocks(u)(j) == 0){
          Ib(u)(j,j) = 0;//1e-6;
        }
      }
      //Rcpp::Rcout << "dim_by_parent " << parents(u).n_elem << endl;
      // number of coords of the jth parent of the child
      dim_by_parent(u) = arma::zeros(parents(u).n_elem + 1);
      for(int j=0; j<parents(u).n_elem; j++){
        dim_by_parent(u)(j+1) = indexing(parents(u)(j)).n_elem;//coords_blocks(parents(u)(j)).n_rows;
      }
      dim_by_parent(u) = arma::cumsum(dim_by_parent(u));
    }
  }
  
  message("[init_finalize] u_is_which_col_f");
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    //Rcpp::Rcout << "block: " << u << "\n";
    
    //if(coords_blocks(u).n_elem > 0){ //**
    if(indexing(u).n_elem > 0){
      // children-parent relationship variables
      u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (q*children(u).n_elem);
      for(int c=0; c<children(u).n_elem; c++){
        int child = children(u)(c);
        //Rcpp::Rcout << "child: " << child << "\n";
        // which parent of child is u which we are sampling
        arma::uvec u_is_which = arma::find(parents(child) == u, 1, "first"); 
        
        // which columns correspond to it
        int firstcol = dim_by_parent(child)(u_is_which(0));
        int lastcol = dim_by_parent(child)(u_is_which(0)+1);
        //Rcpp::Rcout << "from: " << firstcol << " to: " << lastcol << endl; 
        
        int dimen = parents_indexing(child).n_elem;
        /*
        arma::vec colix = arma::zeros(q*dimen);//parents_coords(child).n_rows);//(w_cond_mean_K(child).n_cols);
        //arma::uvec indx_scheme = arma::regspace<arma::uvec>(0, q, q*(dimen-1));
        
        for(int s=0; s<q; s++){
          //arma::uvec c_indices = s + indx_scheme.subvec(firstcol, lastcol-1);

          int shift = s * dimen;
          colix.subvec(shift + firstcol, shift + lastcol-1).fill(1);
          //colix.elem(c_indices).fill(1);
        }
        //Rcpp::Rcout << indx_scheme << "\n";
        //Rcpp::Rcout << colix << "\n";
        
        //Rcpp::Rcout << "visual representation of which ones we are looking at " << endl
        //            << colix.t() << endl;
        
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = arma::find(colix == 1); // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = arma::find(colix != 1); // u parent of c is NOT in these columns for c
        */
        // / / /
        arma::uvec temp = arma::regspace<arma::uvec>(0, q*dimen-1);
        arma::umat result = arma::trans(arma::umat(temp.memptr(), q, temp.n_elem/q));
        arma::uvec rowsel = arma::zeros<arma::uvec>(result.n_rows);
        rowsel.subvec(firstcol, lastcol-1).fill(1);
        arma::umat result_local = result.rows(arma::find(rowsel==1));
        arma::uvec result_local_flat = arma::vectorise(arma::trans(result_local));
        arma::umat result_other = result.rows(arma::find(rowsel==0));
        arma::uvec result_other_flat = arma::vectorise(arma::trans(result_other));
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = result_local_flat; // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = result_other_flat; // u parent of c is NOT in these columns for c
      }
    }
  }
}

void MeshGPsvc::get_loglik_w(MeshData& data){
  start = std::chrono::steady_clock::now();
  if(verbose){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if((block_ct_obs(u) > 0) & calc_this_block){
    //if( (block_ct_obs(u) > 0) & compute_block(predicting, block_ct_obs(u), rfc_dep) ){
      //double expcore = -.5 * arma::conv_to<double>::from( (w_blocks(u) - data.w_cond_mean(u)).t() * data.w_cond_prec(u) * (w_blocks(u) - data.w_cond_mean(u)) );
      arma::mat w_x = arma::vectorise( arma::trans( w.rows(indexing(u)) ) );
      
      if(parents(u).n_elem > 0){
        arma::vec w_pars = arma::vectorise( arma::trans( w.rows(parents_indexing(u)) ));
        w_x -= data.w_cond_mean_K(u) * w_pars;
      }
      
      //w_x = w_x % na_1_blocks(u);
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      
      data.loglik_w_comps(u) = //bloc   k_ct_obs(u)//
        (q*indexing(u).n_elem + .0) 
        * hl2pi -.5 * data.wcore(u);
    //} else {
    //  data.wcore(u) = 0;
    //  data.loglik_w_comps(u) = 0;
    //}
  }
  
  data.logdetCi = arma::accu(data.logdetCi_comps);
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps);
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  //print_data(data);
}

Eigen::VectorXd armavec_to_vectorxd(arma::vec arma_A) {
  
  Eigen::VectorXd eigen_B = Eigen::Map<Eigen::VectorXd>(arma_A.memptr(),
                                                        arma_A.n_elem);
  return eigen_B;
}

void expand_grid_with_values_(arma::umat& locs,
                              arma::vec& vals,
                              int rowstart, int rowend,
                              const arma::uvec& x1,
                              const arma::uvec& x2,
                              const arma::mat& values){
  
  for(int i=rowstart; i<rowend; i++){
    arma::uvec ix;
    try {
      ix = arma::ind2sub(arma::size(values), i-rowstart);
    } catch (...) {
      Rcpp::Rcout << arma::size(values) << " " << i-rowstart << " " << i <<" " << rowstart << " " << rowend << endl;
      throw 1;
    }
    locs(0, i) = x1(ix(0));
    locs(1, i) = x2(ix(1));
    vals(i) = values(ix(0), ix(1));
  }
}

void MeshGPsvc::get_loglik_comps_y_collapsed(MeshData& data){
  // this is slow.
  // caching to build the precision matrix is possible but
  // cholmod takes about half the time
  // and it's out of our control
  
  start_overall = std::chrono::steady_clock::now();
  message("[get_loglik_comps_y_collapsed] start.");
  
  tausq_inv = data.theta(0);
  cparams = data.theta.subvec(1, data.theta.n_elem-1);
  Dmat = arma::zeros(0);
  
  start = std::chrono::steady_clock::now();
  arma::uvec Adims = arma::zeros<arma::uvec>(n_blocks+1);
  arma::uvec Ddims = arma::zeros<arma::uvec>(n_blocks+1);
  
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      Adims(i+1) = indexing(u).n_elem * parents_indexing(u).n_elem;
    }
    Ddims(i+1) = indexing(u).n_elem * indexing(u).n_elem;
  }
  
  int Asize = arma::accu(Adims);
  Adims = arma::cumsum(Adims);
  
  arma::umat Hlocs = arma::zeros<arma::umat>(2, Asize);
  arma::vec Hvals = arma::zeros(Asize);
  
  int Dsize = arma::accu(Ddims);
  Ddims = arma::cumsum(Ddims);
  
  arma::umat Dlocs2 = arma::zeros<arma::umat>(2, Dsize);
  arma::vec Dvals2 = arma::zeros(Dsize);
  
  //arma::field<arma::mat> K_coords_cache(coords_caching.n_elem);
  //arma::field<arma::mat> K_parents_cache(parents_caching.n_elem);
  arma::field<arma::mat> K_cholcp_cache(kr_caching.n_elem);
  arma::field<arma::mat> w_cond_mean_cache(kr_caching.n_elem); // +++++++++
  //arma::field<arma::mat> Kcp_cache(kr_caching.n_elem);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    if(block_ct_obs(u) > 0){
      xCovHUV_inplace(K_coords_cache(i), coords, indexing(u), indexing(u), cparams, Dmat, true, twonu);
    }
  }
  
  data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    if(block_ct_obs(u) > 0){//compute_block(predicting, block_ct_obs(u), false)){s
      int u_cached_ix = coords_caching_ix(u);
      arma::uvec cx = arma::find( coords_caching == u_cached_ix );
      arma::mat Kcc = K_coords_cache(cx(0));
      
      // +++++++++++++++++
      arma::mat Kxx = arma::zeros(q*parents_indexing(u).n_elem, q*parents_indexing(u).n_elem);
      xCovHUV_inplace(Kxx, coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true, twonu);
      arma::mat Kxxi_c = arma::inv(arma::trimatl(arma::chol(arma::symmatu( Kxx ), "lower")));

      xCovHUV_inplace(Kcp_cache(i), coords, indexing(u), parents_indexing(u), cparams, Dmat, false, twonu);
      
      // +++++++++++++++++
      arma::mat Kcx_Kxxic = Kcp_cache(i) * Kxxi_c.t();
      w_cond_mean_cache(i) = Kcx_Kxxic * Kxxi_c;
      // +++++++++++++++++
      
      try {
        arma::mat Kinside = Kcc - Kcx_Kxxic*Kcx_Kxxic.t();
        K_cholcp_cache(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
          Kinside
        ) , "lower")));
        if(!Kinside.is_symmetric()){
          data.track_chol_fails(u) = 2;
          Rcpp::Rcout << "Error - Kinside not symmetric for some reason " << endl;
        }
      } catch (...) {
        data.track_chol_fails(u) = 3;
        K_cholcp_cache(i) = arma::eye(arma::size(Kcc));
        Rcpp::Rcout << "Error in inv chol symmatu (Kcc - Kcx Kxx Kxc) at " << u << endl;
      }
      //Rcpp::Rcout << "krig: " << arma::size(K_cholcp_cache(i)) << "\n";
    }
  }
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    int u_cached_ix = coords_caching_ix(u);
    arma::uvec cx = arma::find( coords_caching == u_cached_ix );
    arma::mat Kcc = K_coords_cache(cx(0));
    
    arma::mat cond_mean_K, cond_mean, cond_cholprec;//, w_parents;
    
    if( parents(u).n_elem > 0 ){
      int kr_cached_ix = kr_caching_ix(u);
      arma::uvec cpx = arma::find(kr_caching == kr_cached_ix);
      
      //arma::mat Kcx = Kcp_cache(cpx(0));//Kpp(coords_blocks(u), parents_coords(u), theta);
      cond_mean_K = w_cond_mean_cache(cpx(0));// Kcx * Kxxi; // +++++++++++
      cond_cholprec = K_cholcp_cache(cpx(0));
      
      data.w_cond_mean_K(u) = cond_mean_K;
      data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
      /*
      expand_grid_with_values_(Hlocs, Hvals, Adims(i), Adims(i+1),
                               indexing(u), parents_indexing(u), cond_mean_K);
      
      expand_grid_with_values_(Dlocs2, Dvals2, Ddims(i), Ddims(i+1),
                               indexing(u), indexing(u), data.w_cond_prec(u));
      */
    } else {
      //Rcpp::Rcout << "no parents " << endl;
      cond_mean_K = arma::zeros(arma::size(parents(u)));
      cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
      data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
      
      /*
      expand_grid_with_values_(Dlocs2, Dvals2, Ddims(i), Ddims(i+1),
                               indexing(u), indexing(u), data.w_cond_prec(u));
      */
    }
    
    //if(block_ct_obs(u) > 0){
    arma::vec ccholprecdiag = cond_cholprec.diag();//(na_ix_blocks(u), na_ix_blocks(u));
    data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));//(na_ix_blocks(u))));
  }

  double ldetCi = arma::accu(data.logdetCi_comps);
  double ldettsqi = n * log(tausq_inv);
  end = std::chrono::steady_clock::now();
  //timings(0) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  // building the precision matrix directly:
  
  start = std::chrono::steady_clock::now();
  arma::field<arma::umat> blockrow_tripls(n_blocks);
  arma::field<arma::mat> blockrow_vals(n_blocks);
  
#pragma omp parallel for
  for(int i=0; i<n_blocks; i++){
    int ui = block_names(i) - 1;
    blockrow_tripls(i) = arma::zeros<arma::umat>(0, 2);
    blockrow_vals(i) = arma::zeros(0, 1);
    
    for(int j=i; j<n_blocks; j++){
      // compute the upper triangular part of the precision matrix
      // col block name
      int uj = block_names(j) - 1;
      
      arma::mat Ci_block;
      
      if(ui == uj){
        // block diagonal part
        Ci_block = data.w_cond_prec(ui);
        
        for(int c=0; c<children(ui).n_elem; c++){
          int child = children(ui)(c);
          arma::mat AK_u = data.w_cond_mean_K(child).cols(u_is_which_col_f(ui)(c)(0));
          Ci_block += AK_u.t() * data.w_cond_prec(child) * AK_u;
        }
        
        // locations to fill: indexing(ui) x indexing(uj) 
        arma::umat tripl_locs(indexing(ui).n_elem * indexing(uj).n_elem, 2);
        arma::mat tripl_val = arma::zeros(Ci_block.n_elem);
        
        for(int ix=0; ix<indexing(ui).n_elem; ix++){
          for(int jx=0; jx<indexing(uj).n_elem; jx++){
            int vecix = arma::sub2ind(arma::size(indexing(ui).n_elem, indexing(uj).n_elem), ix, jx);
            tripl_locs(vecix, 0) = indexing(ui)(ix);
            tripl_locs(vecix, 1) = indexing(uj)(jx);
            tripl_val(vecix, 0) = Ci_block(ix, jx);
          }
        }
        
        //Rcpp::Rcout << ui << endl;
        //Rcpp::Rcout << arma::size(blockrow_tripls(i)) << " " << arma::size(tripl_locs) << endl;
        blockrow_tripls(i) = arma::join_vert(blockrow_tripls(i), tripl_locs);
        //Rcpp::Rcout << arma::size(blockrow_vals(i)) << " " << arma::size(tripl_val) << endl;
        blockrow_vals(i) = arma::join_vert(blockrow_vals(i), tripl_val);
        //Rcpp::Rcout << "- " << endl;
        
      } else {
        bool nonempty=false;
        
        // is there anything between these two? 
        // they are either:
        // 1: ui is parent of uj 
        // 2: ui is child of uj <-- we're going by row so this should not be important?
        // 3: ui and uj have common child
        arma::uvec oneuv = arma::ones<arma::uvec>(1);
        
        arma::uvec ui_is_ujs_parent = arma::find(children(ui) == uj);
        //Rcpp::Rcout << parents(uj) << endl;
        if(ui_is_ujs_parent.n_elem > 0){
          nonempty = true;
          
          // ui is a parent of uj
          int c = ui_is_ujs_parent(0); // ui is uj's c-th parent
          //Rcpp::Rcout << " loc 5 b " << c << " " << arma::size(param_data.w_cond_mean_K(uj)) << endl;
          //Rcpp::Rcout << u_is_which_col_f(ui) << endl;
          arma::mat AK_u = data.w_cond_mean_K(uj).cols(u_is_which_col_f(ui)(c)(0));
          //Rcpp::Rcout << " loc 5 c " << endl;
          Ci_block = - AK_u.t() * data.w_cond_prec(uj);
        } else {
          // common children? in this case we can only have one common child
          arma::uvec commons = arma::intersect(children(uj), children(ui));
          if(commons.n_elem > 0){
            nonempty = true;
            
            int child = commons(0);
            arma::uvec find_ci = arma::find(children(ui) == child);
            int ci = find_ci(0);
            arma::uvec find_cj = arma::find(children(uj) == child);
            int cj = find_cj(0);
            arma::mat AK_ui = data.w_cond_mean_K(child).cols(u_is_which_col_f(ui)(ci)(0));
            arma::mat AK_uj = data.w_cond_mean_K(child).cols(u_is_which_col_f(uj)(cj)(0));
            Ci_block = AK_ui.t() * data.w_cond_prec(child) * AK_uj;
          }
        }
        
        if(nonempty){
          // locations to fill: indexing(ui) x indexing(uj) and the transposed lower block-triangular part
          arma::umat tripl_locs1(indexing(ui).n_elem * indexing(uj).n_elem, 2);
          arma::mat tripl_val1 = arma::vectorise(Ci_block);
          
          for(int jx=0; jx<indexing(uj).n_elem; jx++){
            for(int ix=0; ix<indexing(ui).n_elem; ix++){
              int vecix = arma::sub2ind(arma::size(indexing(ui).n_elem, indexing(uj).n_elem), ix, jx);
              tripl_locs1(vecix, 0) = indexing(ui)(ix);
              tripl_locs1(vecix, 1) = indexing(uj)(jx);
            }
          }
          
          blockrow_tripls(i) = arma::join_vert(blockrow_tripls(i), tripl_locs1);
          blockrow_vals(i) = arma::join_vert(blockrow_vals(i), tripl_val1);
          
          arma::umat tripl_locs2(indexing(ui).n_elem * indexing(uj).n_elem, 2);
          arma::mat tripl_val2 = arma::vectorise(arma::trans(Ci_block));
          
          for(int jx=0; jx<indexing(uj).n_elem; jx++){
            for(int ix=0; ix<indexing(ui).n_elem; ix++){
              int vecix = arma::sub2ind(arma::size(indexing(uj).n_elem, indexing(ui).n_elem), jx, ix);
              tripl_locs2(vecix, 0) = indexing(uj)(jx);
              tripl_locs2(vecix, 1) = indexing(ui)(ix);
            }
          }
          
          blockrow_tripls(i) = arma::join_vert(blockrow_tripls(i), tripl_locs2);
          blockrow_vals(i) = arma::join_vert(blockrow_vals(i), tripl_val2);
        }
      }
      
    }
  }
  
  //Rcpp::Rcout << "concatenating " << endl;
  arma::umat Cilocs = field_v_concatm(blockrow_tripls);
  arma::vec Civals = field_v_concatm(blockrow_vals);
  
  //Rcpp::Rcout << "triplets " << endl;
  typedef Eigen::Triplet<double> T;
  std::vector<T> tripletList_Ci;
  tripletList_Ci.reserve(Cilocs.n_rows);
  for(int i=0; i<Cilocs.n_rows; i++){
    tripletList_Ci.push_back(T(Cilocs(i, 0), Cilocs(i, 1), Civals(i)));
  }
  
  //Rcpp::Rcout << "gen sparse mat " << endl;
  Eigen::SparseMatrix<double> Ci(n,n);
  Ci.setFromTriplets(tripletList_Ci.begin(), tripletList_Ci.end());
  end = std::chrono::steady_clock::now();
  //timings(0) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  Eigen::SparseMatrix<double> I_eig(n, n);
  I_eig.setIdentity();
  
  /*
   // EIGEN build via matrix multiply - slower
   
  std::vector<T> tripletList_H;
  std::vector<T> tripletList_Dic2;
  
  start = std::chrono::steady_clock::now();
   //Rcpp::Rcout << "Step 4 " << endl;
  
  tripletList_H.reserve(Hlocs.n_cols);
  for(int i=0; i<Hlocs.n_cols; i++){
    tripletList_H.push_back(T(Hlocs(0, i), Hlocs(1, i), Hvals(i)));
  }
  Eigen::SparseMatrix<double> He(n,n);
  He.setFromTriplets(tripletList_H.begin(), tripletList_H.end());
  
  //Rcpp::Rcout << "Step 5 " << endl;
  tripletList_Dic2.reserve(Dlocs2.n_cols);
  for(int i=0; i<Dlocs2.n_cols; i++){
    tripletList_Dic2.push_back(T(Dlocs2(0, i), Dlocs2(1, i), Dvals2(i)));
  }
  Eigen::SparseMatrix<double> Dice2(n,n);
  Dice2.setFromTriplets(tripletList_Dic2.begin(), tripletList_Dic2.end());
  
  end = std::chrono::steady_clock::now();
  timings(1) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  start = std::chrono::steady_clock::now();
  //Rcpp::Rcout << "Step 6 " << endl;
  Eigen::SparseMatrix<double> L = (I_eig-He).triangularView<Eigen::Lower>().transpose();
  end = std::chrono::steady_clock::now();
  timings(2) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> Ci = (L * Dice2 * L.transpose());
  end = std::chrono::steady_clock::now();
  timings(3) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  */
      
  start = std::chrono::steady_clock::now();
  Eigen::SparseMatrix<double> Ci_tausqi = Ci + tausq_inv * I_eig;
  end = std::chrono::steady_clock::now();
  //timings(4) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  start = std::chrono::steady_clock::now();
  Eigen::CholmodDecomposition<Eigen::SparseMatrix<double> > solver;
  solver.compute(Ci_tausqi);
  double ldetCitsqi = solver.logDeterminant();
  
  Eigen::VectorXd ye = armavec_to_vectorxd(y);
  double logdet_overall = -ldetCitsqi + ldetCi + ldettsqi;
  
  double ycore1 = -.5 * tausq_inv * ye.transpose() * ye;
  double ycore2 = .5 * tausq_inv * tausq_inv * ye.transpose() * solver.solve(ye);
  data.loglik_w = logdet_overall + ycore1 + ycore2;
  
  end = std::chrono::steady_clock::now();
  //timings(5) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  
  if(verbose & debug){
    end_overall = std::chrono::steady_clock::now();
    double timer_all = std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count();
    //Rcpp::Rcout << timings.t() << endl;
  }
}

bool MeshGPsvc::get_loglik_comps_w(MeshData& data){
  if(cached){
    return get_cond_comps_loglik_w(data);
  } else {
    return get_cond_comps_loglik_w_nocache(data);
  }
}

/*
void MeshGPsvc::get_cond_comps_loglik_w(MeshData& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w] start.");
  
  //arma::field<arma::mat> K_coords_cache(coords_caching.n_elem);
  arma::field<arma::mat> K_parents_cache(parents_caching.n_elem);
  arma::field<arma::mat> K_cholcp_cache(kr_caching.n_elem);
  //arma::field<arma::mat> Kcp_cache(kr_caching.n_elem);
  
  arma::vec Kparam = arma::join_vert(arma::ones(1)*sigmasq, data.theta); 
  int k = data.theta.n_elem - npars; // number of cross-distances = p(p-1)/2
  
  arma::vec cparams = Kparam.subvec(0, npars);
  arma::mat Dmat;
  if(k>0){
    Dmat = vec_to_symmat(Kparam.subvec(npars+1, npars+k));
  } else {
    Dmat = arma::zeros(1,1);
  }
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); // layer name of ith representative
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if(calc_this_block){
    if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
      //uhm ++;
      //K_coords_cache(i) = Kpp(coords_blocks(u), coords_blocks(u), Kparam, true);
      xCovHUV_inplace(K_coords_cache(i), coords, indexing(u), indexing(u), cparams, Dmat, true);
    }
  }
  
#pragma omp parallel for // **
  for(int i=0; i<parents_caching.n_elem; i++){
    int u = parents_caching(i); // layer name of ith representative
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if(calc_this_block){
    if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
      //uhm ++;
      xCovHUV_inplace(K_parents_cache_noinv(i), coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
      K_parents_cache(i) = arma::inv_sympd( K_parents_cache_noinv(i) );
      //Rcpp::Rcout << "parents: " << arma::size(K_parents_cache(i)) << "\n";
    }
  }
  data.track_chol_fails = arma::zeros<arma::uvec>(kr_caching.n_elem);
  
  #pragma omp parallel for
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    
    if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
      int u_cached_ix = coords_caching_ix(u);
      arma::uvec cx = arma::find( coords_caching == u_cached_ix );
      arma::mat Kcc = K_coords_cache(cx(0));
      
      int p_cached_ix = parents_caching_ix(u);
      arma::uvec px = arma::find( parents_caching == p_cached_ix );
      arma::mat Kxxi = K_parents_cache(px(0)); //arma::inv_sympd(  Kpp(parents_coords(u), parents_coords(u), theta, true) );

      xCovHUV_inplace(Kcp_cache(i), coords, indexing(u), parents_indexing(u), cparams, Dmat);
      
      try {
        K_cholcp_cache(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
          Kcc - Kcp_cache(i) * Kxxi * Kcp_cache(i).t() // -----------
        ) , "lower")));
      } catch (...) {
        data.track_chol_fails(i) = 1;
        K_cholcp_cache(i) = arma::eye(arma::size(Kcc));
        
        Rcpp::Rcout << "Error in inv chol symmatu (Kcc - Kcx Kxx Kxc) at " << u << endl;
        
      }
    }
  }
  
  if(arma::all(data.track_chol_fails == 0)){
    data.cholfail = false;
    
#pragma omp parallel for // **
    for(int i=0; i<n_blocks; i++){
      int u = block_names(i)-1;
      
      if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
        int u_cached_ix = coords_caching_ix(u);
        arma::uvec cx = arma::find( coords_caching == u_cached_ix );
        arma::mat Kcc = K_coords_cache(cx(0));
        
        arma::mat cond_mean_K, cond_mean, cond_cholprec;//, w_parents;
        
        if( parents(u).n_elem > 0 ){
          int p_cached_ix = parents_caching_ix(u);
          arma::uvec px = arma::find( parents_caching == p_cached_ix );
          arma::mat Kxxi = K_parents_cache(px(0)); //arma::inv_sympd(  Kpp(parents_coords(u), parents_coords(u), theta, true) );
          
          int kr_cached_ix = kr_caching_ix(u);
          arma::uvec cpx = arma::find(kr_caching == kr_cached_ix);
          
          arma::mat Kcx = Kcp_cache(cpx(0));//Kpp(coords_blocks(u), parents_coords(u), theta);
          cond_mean_K = Kcx * Kxxi;
          cond_cholprec = K_cholcp_cache(cpx(0));//arma::inv(arma::trimatl(arma::chol( arma::symmatu(Kcc - cond_mean_K * Kcx.t()) , "lower")));
        } else {
          //Rcpp::Rcout << "no parents " << endl;
          cond_mean_K = arma::zeros(arma::size(parents(u)));
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
        }
        //Rcpp::Rcout << "cond_mean_K " << arma::size(cond_mean_K) << endl;
        data.w_cond_mean_K(u) = cond_mean_K;
        data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
        
        if(block_ct_obs(u) > 0){
          arma::vec ccholprecdiag = cond_cholprec.diag();//(na_ix_blocks(u), na_ix_blocks(u));
          data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));//(na_ix_blocks(u))));
          
          arma::vec w_x = arma::vectorise(arma::trans( w.rows(indexing(u)) ));
          if(parents(u).n_elem > 0){
            arma::vec w_pars = arma::vectorise(arma::trans( w.rows(parents_indexing(u)) ));
            w_x -= data.w_cond_mean_K(u) * w_pars;
          }
          //w_x = w_x % na_1_blocks(u);
          data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
          //Rcpp::Rcout << "u: " << u << " wcore: " << data.wcore(u) << endl; 
          data.loglik_w_comps(u) = //block_ct_obs(u)//
            (q*indexing(u).n_elem+.0) 
            * hl2pi -.5 * data.wcore(u);
          
        } else {
          data.logdetCi_comps(u) = 0;
          data.wcore(u) = 0;
          data.loglik_w_comps(u) = 0;
          //Rcpp::Rcout << "i: " << i << ", u: " << u << " ldCi " << logdetCi_comps(u) << endl;
        }
      } else {
        data.logdetCi_comps(u) = 0;
        data.wcore(u) = 0;
        data.loglik_w_comps(u) = 0;
      }
    }
    data.logdetCi = arma::accu(data.logdetCi_comps);
    data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps);
    
  } else {
    data.cholfail = true;
    throw 1;
  }
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_chached_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

*/

void MeshGPsvc::theta_transform(const MeshData& data){
  //arma::vec Kparam = data.theta; 
  int k = data.theta.n_elem - npars; // number of cross-distances = p(p-1)/2
  
  //Rcpp::Rcout << data.theta << endl
    //         << "npars " << npars << endl;
  
  cparams = arma::join_vert(arma::ones(1) * sigmasq, data.theta.subvec(0, npars - 1));
  
  if(k>0){
    Dmat = vec_to_symmat(data.theta.subvec(npars, npars + k - 1));
  } else {
    Dmat = arma::zeros(1,1);
  }
  //Rcpp::Rcout << "theta_transform" << endl
   //           << data.theta << endl
  //            << cparams << endl
   //           << Dmat << endl;
}

bool MeshGPsvc::get_cond_comps_loglik_w(MeshData& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w] start.");
  
  //arma::field<arma::mat> K_coords_cache(coords_caching.n_elem);
  //arma::field<arma::mat> K_parents_cache(parents_caching.n_elem);
  arma::field<arma::mat> K_cholcp_cache(kr_caching.n_elem);
  arma::field<arma::mat> w_cond_mean_cache(kr_caching.n_elem); // +++++++++
  //arma::field<arma::mat> Kcp_cache(kr_caching.n_elem);
  
  theta_transform(data);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); // layer name of ith representative
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if(calc_this_block){
    //if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
    if(block_ct_obs(u) > 0){
      //uhm ++;
      //K_coords_cache(i) = Kpp(coords_blocks(u), coords_blocks(u), Kparam, true);
      xCovHUV_inplace(K_coords_cache(i), coords, indexing(u), indexing(u), cparams, Dmat, true, twonu);
    }
  }
  
  data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  int chol_error = -1;
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    try {
      if(block_ct_obs(u) > 0){
        int u_cached_ix = coords_caching_ix(u);
        arma::uvec cx = arma::find( coords_caching == u_cached_ix );
        arma::mat Kcc = K_coords_cache(cx(0));
        
        // +++++++++++++++++
        arma::mat Kxx = arma::zeros(q*parents_indexing(u).n_elem, q*parents_indexing(u).n_elem);
        xCovHUV_inplace(Kxx, coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true, twonu);
        arma::mat Kxxi_c = arma::inv(arma::trimatl(arma::chol(arma::symmatu( Kxx ), "lower")));
        // +++++++++++++++++
        
        xCovHUV_inplace(Kcp_cache(i), coords, indexing(u), parents_indexing(u), cparams, Dmat, false, twonu);
        
        // +++++++++++++++++
        arma::mat Kcx_Kxxic = Kcp_cache(i) * Kxxi_c.t();
        w_cond_mean_cache(i) = Kcx_Kxxic * Kxxi_c;
        // +++++++++++++++++
        
        arma::mat Kinside = Kcc - Kcx_Kxxic*Kcx_Kxxic.t();
        K_cholcp_cache(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
          Kinside
        ) , "lower")));
        //Rcpp::Rcout << "krig: " << arma::size(K_cholcp_cache(i)) << "\n";
      }
    } catch (...) {
      chol_error = 1;
    }
  }
  
  end = std::chrono::steady_clock::now();
  //timing(0) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(chol_error > 0){
    return false;
  }
  
  if(verbose & debug){
    
    Rcpp::Rcout << "[cache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  start = std::chrono::steady_clock::now();
  
  //if(arma::all(data.track_chol_fails == 0)){
  //  data.cholfail = false;
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i = 0; i<n_ref_blocks; i++){
      int r = reference_blocks(i);
      int u = block_names(r)-1;

      try {
        int u_cached_ix = coords_caching_ix(u);
        arma::uvec cx = arma::find( coords_caching == u_cached_ix );
        arma::mat Kcc = K_coords_cache(cx(0));
        
        arma::mat cond_mean_K, cond_mean, cond_cholprec;//, w_parents;
        
        if( parents(u).n_elem > 0 ){
          // ---------------
          //int p_cached_ix = parents_caching_ix(u);
          //arma::uvec px = arma::find( parents_caching == p_cached_ix );
          //arma::mat Kxxi = K_parents_cache(px(0)); //arma::inv_sympd(  Kpp(parents_coords(u), parents_coords(u), theta, true) );
          
          int kr_cached_ix = kr_caching_ix(u);
          arma::uvec cpx = arma::find(kr_caching == kr_cached_ix);
          
          //arma::mat Kcx = Kcp_cache(cpx(0));//Kpp(coords_blocks(u), parents_coords(u), theta);
          cond_mean_K = w_cond_mean_cache(cpx(0));// Kcx * Kxxi; // +++++++++++
          cond_cholprec = K_cholcp_cache(cpx(0));//arma::inv(arma::trimatl(arma::chol( arma::symmatu(Kcc - cond_mean_K * Kcx.t()) , "lower")));
          //Rcpp::Rcout << arma::size(cond_cholprec) << endl;
        } else {
          //Rcpp::Rcout << "no parents " << endl;
          cond_mean_K = arma::zeros(arma::size(parents(u)));
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
        }
        //Rcpp::Rcout << "cond_mean_K " << arma::size(cond_mean_K) << endl;
        data.w_cond_mean_K(u) = cond_mean_K;
        data.w_cond_cholprec(u) = cond_cholprec;
        data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
        
        //if(block_ct_obs(u) > 0){
        arma::vec ccholprecdiag = cond_cholprec.diag();//(na_ix_blocks(u), na_ix_blocks(u));
        data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));//(na_ix_blocks(u))));
        
        arma::vec w_x = arma::vectorise(arma::trans( w.rows(indexing(u)) ));
        if(parents(u).n_elem > 0){
          arma::vec w_pars = arma::vectorise(arma::trans( w.rows(parents_indexing(u)) ));
          w_x -= data.w_cond_mean_K(u) * w_pars;
        }
        //w_x = w_x % na_1_blocks(u);
        data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
        //Rcpp::Rcout << "u: " << u << " wcore: " << data.wcore(u) << endl; 
        data.loglik_w_comps(u) = //block_ct_obs(u)//
          (q*indexing(u).n_elem+.0) 
          * hl2pi -.5 * data.wcore(u);
      } catch(...) {
        chol_error = 2;
      }
    }
    data.logdetCi = arma::accu(data.logdetCi_comps);
    data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps);
  
  end = std::chrono::steady_clock::now();
  //timing(1) = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  if(chol_error > 0){
    return false;
  }

  if(verbose & debug){
    
    Rcpp::Rcout << "[get_cond_comps_chached_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  return true;
}


bool MeshGPsvc::get_cond_comps_loglik_w_nocache(MeshData& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w_nocache] start. ");
  
  theta_transform(data);

  int chol_error=-1;
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //if(compute_block(predicting, block_ct_obs(u), false) ){
    if(block_ct_obs(u) > 0){
      // skip calculating stuff for blocks in the predict-only area 
      // when it's not prediction time
      arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true, twonu);
      arma::mat cond_mean_K, cond_mean, cond_cholprec;
      
      if( parents(u).n_elem > 0 ){
        arma::mat Kxxi = arma::inv_sympd(  xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true, twonu) );
        arma::mat Kcx = xCovHUV(coords, indexing(u), parents_indexing(u), cparams, Dmat, false, twonu);
        cond_mean_K = Kcx * Kxxi;
        try {
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( arma::symmatu(Kcc - cond_mean_K * Kcx.t()) , "lower")));
        } catch (...) {
          chol_error = 1;
          cond_cholprec = arma::zeros(arma::size(Kcc));
        }
        
      } else {
        cond_mean_K = arma::zeros(0, 0);
        try {
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
        } catch (...) {
          chol_error = 2;
          cond_cholprec = arma::zeros(arma::size(Kcc));
        }
      }
      
      data.w_cond_mean_K(u) = cond_mean_K;
      data.w_cond_cholprec(u) = cond_cholprec;
      data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
    
      //message("[get_cond_comps_loglik_w_nocache] get lik");
      
      arma::vec w_x = arma::vectorise(arma::trans( w.rows(indexing(u)) ));
      if(parents(u).n_elem > 0){
        arma::vec w_pars = arma::vectorise(arma::trans( w.rows(parents_indexing(u))));
        w_x -= data.w_cond_mean_K(u) * w_pars;
      }
      
      if(block_ct_obs(u) > 0){
        arma::vec ccholprecdiag = cond_cholprec.diag();
        data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));
        
        //w_x = w_x % na_1_blocks(u);
        data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
        data.loglik_w_comps(u) = (q*indexing(u).n_elem+.0) 
          * hl2pi -.5 * data.wcore(u);
      }
    } else {
      Rcpp::Rcout << "should not read this 2 " << endl;
    }
  }
  if(chol_error > 0){
    Rcpp::Rcout << "Error with Cholesky. Possibly change limits for MCMC search" << endl;
    return false;
  }
  
  data.logdetCi = arma::accu(data.logdetCi_comps.subvec(0, n_blocks-1));
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps.subvec(0, n_blocks-1));
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_loglik_w_nocache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  return true;
}

void MeshGPsvc::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv * XtX + Vi), "lower"); 
  arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
  
  arma::mat Xprecy = Vim + tausq_inv * X_available.t() * ( y_available - Zw.rows(na_ix_all));// + ywmeandiff );
  Rcpp::RNGScope scope;
  Bcoeff = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy + arma::randn(p));
  //Bcoeff = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy);// + arma::randn(p));
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void MeshGPsvc::gibbs_sample_sigmasq(){
  start = std::chrono::steady_clock::now();
  
  double oldsigmasq = sigmasq;
  double aparam = sigmasq_ab(0) + n_loc_ne_blocks*q/2.0; //
  double bparam = 1.0/( sigmasq_ab(1) + .5 * oldsigmasq * arma::accu( param_data.wcore ));
  
  Rcpp::RNGScope scope;
  double sigmasqinv = R::rgamma(aparam, bparam);
  sigmasq = 1.0/sigmasqinv;
  
  double old_new_ratio = oldsigmasq / sigmasq;
  
  if(std::isnan(old_new_ratio)){
    Rcpp::Rcout << oldsigmasq << " -> " << sigmasq << " ? " << bparam << endl;
    Rcpp::Rcout << "Error with sigmasq" << endl;
    throw 1;
  }
  // change all K

  //for(int i=0; i<n_blocks; i++){
    //int u = block_names(i)-1;
    //if(compute_block(false, block_ct_obs(u), rfc_dep)){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i = 0; i<n_ref_blocks; i++){
      int r = reference_blocks(i);
      int u = block_names(r)-1;
      param_data.logdetCi_comps(u) += //block_ct_obs(u)//
        (q*indexing(u).n_elem + 0.0) 
      * 0.5*( log(oldsigmasq) - log(sigmasq) );
      
      param_data.w_cond_prec(u) *= old_new_ratio;// * param_data.w_cond_prec(u);
      param_data.w_cond_cholprec(u) *= sqrt(old_new_ratio);// * param_data.w_cond_cholprec(u);
      param_data.wcore(u) *= old_new_ratio;// * param_data.wcore(u);//arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      param_data.loglik_w_comps(u) = //block_ct_obs(u)//
        (q*indexing(u).n_elem + .0) 
        * hl2pi -.5 * param_data.wcore(u);
    //} else {
    //  param_data.wcore(u) = 0;
    //  param_data.loglik_w_comps(u) = 0;
    //}
  }
  
  param_data.logdetCi = arma::accu(param_data.logdetCi_comps);
  param_data.loglik_w = param_data.logdetCi + arma::accu(param_data.loglik_w_comps);
  
  //Rcpp::Rcout << "sigmasq wcore: " << arma::accu( param_data.wcore ) << endl; //##
  //Rcpp::Rcout << "sigmasq logdetCi: " << param_data.logdetCi << endl; //##
  //Rcpp::Rcout << "sigmasq loglik_w: " << param_data.loglik_w << endl; //##
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sigmasq] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " //<< " ... "
                << aparam << " : " << bparam << " " << arma::accu( param_data.wcore ) << " --> " << sigmasq
                << endl;
  }
}

void MeshGPsvc::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  arma::vec XB_availab = X_available * Bcoeff;
  arma::vec Zw_availab = Zw.rows(na_ix_all);
  
  arma::mat yrr = y_available - XB_availab - Zw_availab;
  double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
  double aparam = tausq_ab(0) + n/2.0;
  double bparam = 1.0/( tausq_ab(1) + .5 * bcore );
  
  Rcpp::RNGScope scope;
  tausq_inv = R::rgamma(aparam, bparam);
  //tausq_inv = aparam*bparam; 
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_tausq] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " //<< " ... "
                << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv
                << endl;
  }
}

void MeshGPsvc::gibbs_sample_w(){
  if(cached_gibbs){
    gibbs_sample_w_omp();
  } else {
    gibbs_sample_w_omp_nocache();
  }
  //print_data(param_data);
}

void MeshGPsvc::gibbs_sample_w_omp(){
  message("[gibbs_sample_w_omp]");
  
  // keep seed
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp] sampling big stdn matrix size " << coords.n_rows << "," << q << endl;
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, q);
  
  arma::field<arma::mat> Sigi_chol_cached(gibbs_caching.n_elem);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<gibbs_caching.n_elem; i++){
    int u = gibbs_caching(i);
    //Rcpp::Rcout << "gibbs_sample_w_omp caching step - block " << u << "\n";
    
    if(block_ct_obs(u) > 0){
      arma::mat Sigi_tot = param_data.w_cond_prec(u); // Sigi_p
      
      for(int c=0; c<children(u).n_elem; c++){
        int child = children(u)(c);
        arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));//u_is_which_col);
        arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
        Sigi_tot += AK_uP * AK_u;
      }
      Sigi_tot += tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u);
      Sigi_chol_cached(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
    }
  }
  start_overall = std::chrono::steady_clock::now();
  
  for(int g=0; g<n_gibbs_groups-predict_group_exists; g++){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      if(block_ct_obs(u) > 0){
      arma::mat Sigi_chol, Sigi_tot;
      //if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
        arma::mat Smu_c, Smu_tot, Smu_y; //Sigi_p, 
        
        int u_cached_ix = gibbs_caching_ix(u);
        arma::uvec gx = arma::find( gibbs_caching == u_cached_ix );
        Sigi_chol = Sigi_chol_cached(gx(0));
        
        Smu_tot = arma::zeros(q*indexing(u).n_elem, 1);
        if(parents(u).n_elem>0){
          arma::vec w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u) )));
          Smu_tot += param_data.w_cond_prec(u) * param_data.w_cond_mean_K(u) * w_par;//param_data.w_cond_mean(u);
        }
        //Rcpp::Rcout << "Smu 1: " << arma::accu(abs(Smu_tot)) << endl;
        
        //Rcpp::Rcout << "gibbs_sample_w_omp loop step 2 - block " << u << "\n";
        
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //Rcpp::Rcout << "1- c: " << child << endl;
          arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));//u_is_which_col);
          //Rcpp::Rcout << "2- c: " << child << endl;
          arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
          arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));//u_isnot_which_col);
          //Rcpp::Rcout << "3- c: " << child << endl;
          arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
          
          arma::vec w_child = arma::vectorise(arma::trans(w.rows( indexing(child) )));
          //Rcpp::Rcout << "4- c: " << child << endl;
          //Rcpp::Rcout << arma::size(w_parents_of_child) << "\n" << u_is_which_col_f(u)(c)(1) << endl;
          arma::vec w_par_child = arma::vectorise(arma::trans(w_parents_of_child ));
          Smu_tot += AK_uP * ( w_child - AK_others * w_par_child.rows(u_is_which_col_f(u)(c)(1)) );
        }
        
        //Rcpp::Rcout << "gibbs_sample_w_omp loop step 3 - block " << u << "\n";
        //Rcpp::Rcout << "Smu 2: " << arma::accu(abs(Smu_tot)) << endl;
        Smu_tot += Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
            ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff ));
        
        // sample
        
        arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        //arma::vec rnvec = arma::randn(q*indexing(u).n_elem);
        arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec);
        //arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot);// + rnvec);
        //Rcpp::Rcout << "gibbs_sample_w_omp loop step 4 - block " << u << "\n";
        
        w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
        
      } 
    }
  }
  
  Zw = armarowsum(Z % w);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_omp] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}

void MeshGPsvc::gibbs_sample_w_omp_nocache(){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] " << endl;
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, q);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << endl << " ? " << endl;
  
  int counter = 0;
  int crow = 0;
  int ccol = 0;
  
  start_overall = std::chrono::steady_clock::now();
  for(int g=0; g<n_gibbs_groups-predict_group_exists; g++){
    //int g = gibbs_groups_reorder(go);
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
          
          arma::mat Smu_tot = arma::zeros(q*indexing(u).n_elem, 1);
          arma::mat Sigi_tot = param_data.w_cond_prec(u); // Sigi_p
          
          arma::vec w_par;
          if(parents(u).n_elem>0){
            w_par = arma::vectorise(arma::trans( w.rows( parents_indexing(u) ) ));
            Smu_tot += Sigi_tot * param_data.w_cond_mean_K(u) * w_par;//param_data.w_cond_mean(u);
          }
          
          //arma::uvec ug = arma::zeros<arma::uvec>(1) + g;
          // indexes being used as parents in this group
          
          for(int c=0; c<children(u).n_elem; c++){
            int child = children(u)(c);
            //clog << "g: " << g << " ~ u: " << u << " ~ child " << c << " - " << child << "\n";
            //Rcpp::Rcout << u_is_which_col_f(u)(c)(0).t() << "\n";
            //Rcpp::Rcout << u_is_which_col_f(u)(c)(1).t() << "\n";
            
            arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
            arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
            arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
            
            arma::vec w_child = arma::vectorise(arma::trans( w.rows( indexing(child) ) ));
            
            //Rcpp::Rcout << "child part 3 \n";
            arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
            arma::vec w_par_child = arma::vectorise(arma::trans(w_parents_of_child));
            arma::vec w_par_child_select = w_par_child.rows(u_is_which_col_f(u)(c)(1));
            
            Sigi_tot += AK_uP * AK_u;
            Smu_tot += AK_uP * ( w_child - AK_others * w_par_child_select );
          }
          
          Sigi_tot += tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u);
          Smu_tot += Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % 
            ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff ));
          
          counter ++;
          crow += Sigi_tot.n_rows;
          ccol += Sigi_tot.n_cols;
          
          //start = std::chrono::steady_clock::now();
          arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
          //end = std::chrono::steady_clock::now();

          //if (u == 1){
          //  Rcpp::Rcout << Sigi_chol << endl;
          //}
          // sample
          arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
          //arma::vec rnvec = arma::randn(q*indexing(u).n_elem);
          arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec); //************************
          
          w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
          
          
          // update w-loglike comps
          if(parents(u).n_elem > 0){
            w_temp -= param_data.w_cond_mean_K(u) * w_par;
          }
          
          //w_x = w_x % na_1_blocks(u);
          param_data.wcore(u) = arma::conv_to<double>::from(w_temp.t() * param_data.w_cond_prec(u) * w_temp);
          
          param_data.loglik_w_comps(u) = //bloc   k_ct_obs(u)//
            (q*indexing(u).n_elem + .0) 
            * hl2pi -.5 * param_data.wcore(u);
          
          
    }
  }
  
  Zw = armarowsum(Z % w);
  param_data.logdetCi = arma::accu(param_data.logdetCi_comps);
  param_data.loglik_w = param_data.logdetCi + arma::accu(param_data.loglik_w_comps);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
  
  
  
  
}

void MeshGPsvc::predict(){
  if(predict_group_exists == 1){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_predicts.n_elem; i++){
      int u = u_predicts(i);
      
      // only predictions at this block. 
      // sample from conditional MVN 
      arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true, twonu);
      arma::mat Kxxi = arma::inv_sympd( xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true, twonu) );
      arma::mat Kcx = xCovHUV(coords, indexing(u), parents_indexing(u), cparams, Dmat, false, twonu);
        
        //Rcpp::Rcout << "? 1 " << endl;
      param_data.w_cond_mean_K(u) = Kcx * Kxxi;
      param_data.w_cond_cholprec(u) = arma::flipud(arma::fliplr(arma::chol( arma::symmatu(Kcc - param_data.w_cond_mean_K(u) * Kcx.t()) , "lower")));
    
    
      arma::vec w_par = arma::vectorise(arma::trans(w.rows(parents_indexing(u))));
      
      arma::vec phimean = param_data.w_cond_mean_K(u) * w_par;
      
      arma::mat cholK = param_data.w_cond_cholprec(u);
      
      arma::vec normvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
      arma::vec w_temp = phimean + cholK * normvec;
      //arma::vec w_temp = phimean;//************************
      w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
      
    }
  }
}

void MeshGPsvc::theta_update(MeshData& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void MeshGPsvc::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void MeshGPsvc::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void MeshGPsvc::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}