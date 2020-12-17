#define ARMA_DONT_PRINT_ERRORS
#include "includes.h"

using namespace std;

const double hl2pi = -.5 * log(2.0 * M_PI);

class MultivariateMeshGP {
public:
  // meta
  int n;
  int p;
  int q;
  int dd;
  int n_blocks;
  
  int covariance_model;
  CovarianceParams covpars;
  
  // data
  arma::vec y;
  arma::mat X;
  //arma::mat Z;
  
  arma::vec y_available;
  arma::mat X_available;
  //arma::mat Z_available;
  
  arma::mat coords;
  arma::uvec qvblock_c;
  
  arma::uvec reference_blocks; // named
  int n_ref_blocks;
  
  //arma::field<arma::sp_mat> Zblock;
  //arma::vec Zw;
  
  // indexing info
  arma::field<arma::uvec> indexing; 
  arma::field<arma::uvec> indexing_obs;
  arma::field<arma::uvec> parents_indexing; 
  arma::field<arma::uvec> children_indexing;
  
  // NA data
  arma::field<arma::vec> na_1_blocks; // indicator vector by block
  arma::field<arma::uvec> na_ix_blocks;
  arma::uvec na_ix_all;
  
  // variable data
  arma::field<arma::uvec> ix_by_q;
  arma::field<arma::uvec> ix_by_q_a; // storing indices using only available data
  
  int n_loc_ne_blocks;
  
  // regression
  arma::field<arma::mat> XtX;
  arma::mat Vi; 
  arma::mat Vim;
  arma::vec bprim;
  
  // tausq priors
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
  int                       predict_group_exists;
  //arma::uvec predictable_blocks; //*** not actually being used at the time
  arma::uvec                u_predicts;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  arma::field<arma::vec>    dim_by_parent;
  
  arma::uvec oneuv;
  
  // params
  arma::mat w;
  arma::mat Bcoeff; // sampled
  arma::mat rand_norm_mat;

  arma::vec XB;
  arma::vec tausq_inv; // tausq for the l=q variables
  arma::vec tausq_inv_long; // storing tausq_inv at all locations
  
  // params with mh step
  MeshDataMV param_data; 
  MeshDataMV alter_data;
  
  // setup
  bool predicting;
  bool cached;
  bool forced_grid;
  
  bool verbose;
  bool debug;
  
  // debug var
  Rcpp::List debug_stuff; 
  
  void message(string s);
  
  // init / indexing
  void init_indexing();
  void init_meshdata(const arma::vec&);
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
  
  // caching some matrices // ***
  arma::field<arma::mat> K_coords_cache;
  arma::field<arma::mat> Kcp_cache;
  arma::field<arma::mat> Kxxi_cache; //***
  
  // MCMC
  bool get_loglik_comps_w(MeshDataMV& data);
  bool get_cond_comps_loglik_w(MeshDataMV& data);
  bool get_cond_comps_loglik_w_nocache(MeshDataMV& data);
  
  void gibbs_sample_w();
  void gibbs_sample_w_omp_inputgrid();
  void gibbs_sample_w_omp_forcedgrid();
  void gibbs_sample_beta();
  void gibbs_sample_tausq();
  void logpost_refresh_after_gibbs(); //***
  
  void predict(bool);
  
  double logpost;
  
  // changing the values, no sampling;
  void theta_update(MeshDataMV&, const arma::vec&);
  void beta_update(const arma::vec&);
  void tausq_update(double);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  MultivariateMeshGP();
  
  // build everything
  MultivariateMeshGP(
    const arma::vec& y_in, 
    const arma::mat& X_in, 
    
    const arma::mat& coords_in, 
    const arma::uvec& mv_id_in,
    
    const arma::uvec& unused1_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& layers_names,
    const arma::vec& block_group_in,
    
    const arma::field<arma::uvec>& indexing_in,
    const arma::field<arma::uvec>& indexing_obs_in,
    
    const arma::vec& w_in,
    const arma::vec& beta_in,
    const double& unused2_in,
    const arma::vec& theta_in,
    double tausq_inv_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& tausq_ab_in,
    
    bool use_cache,
    bool use_forced_grid,
    
    bool verbose_in,
    bool debugging);
  
};

void MultivariateMeshGP::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

MultivariateMeshGP::MultivariateMeshGP(){
  
}

MultivariateMeshGP::MultivariateMeshGP(
  const arma::vec& y_in, 
  const arma::mat& X_in, 
  
  const arma::mat& coords_in, 
  const arma::uvec& mv_id_in,
  
  const arma::uvec& unused1_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  
  const arma::field<arma::uvec>& indexing_in,
  const arma::field<arma::uvec>& indexing_obs_in,
  
  const arma::vec& w_in,
  const arma::vec& beta_in,
  const double& unused2_in,
  const arma::vec& theta_in,
  double tausq_inv_in,
  
  const arma::mat& beta_Vi_in,
  const arma::vec& tausq_ab_in,
  
  bool use_cache=true,
  bool use_forced_grid=false,
  
  bool verbose_in=false,
  bool debugging=false){
  
  
  verbose = verbose_in;
  debug = debugging;
  
  message("MultivariateMeshGP::MultivariateMeshGP initialization.\n");
  
  forced_grid = use_forced_grid;
  start_overall = std::chrono::steady_clock::now();
  
  if(forced_grid){
    cached = true;
    message("MGP on a latent grid");
  } else {
    cached = use_cache;
    if(cached){
      message("MGP on provided grid, caching activated (data are gridded)");
    } else {
      message("MGP on provided grid, caching deactivated (data are not gridded)");
    }
  }
  
  
  message("MultivariateMeshGP::MultivariateMeshGP assign values.");
  
  // data
  y                   = y_in;
  X                   = X_in;
  
  na_ix_all   = arma::find_finite(y.col(0));
  y_available = y.rows(na_ix_all);
  X_available = X.rows(na_ix_all); 
  
  n  = na_ix_all.n_elem;
  p  = X.n_cols;
  
  // spatial coordinates and dimension
  coords              = coords_in;
  dd = coords.n_cols;
  
  // outcome variables
  qvblock_c           = mv_id_in-1;
  arma::uvec mv_id_uniques = arma::unique(mv_id_in);
  q  = mv_id_uniques.n_elem;
  
  // NAs at blocks of outcome variables 
  ix_by_q = arma::field<arma::uvec>(q);
  ix_by_q_a = arma::field<arma::uvec>(q);
  arma::uvec qvblock_c_available = qvblock_c.rows(na_ix_all);
  for(int j=0; j<q; j++){
    ix_by_q(j) = arma::find(qvblock_c == j);
    ix_by_q_a(j) = arma::find(qvblock_c_available == j);
  }
  
  // DAG
  parents             = parents_in;
  children            = children_in;
  block_names         = block_names_in;
  block_groups        = block_groups_in;
  block_groups_labels = arma::unique(block_groups);
  n_gibbs_groups      = block_groups_labels.n_elem;
  n_blocks            = block_names.n_elem;

  //Z_available = Z.rows(na_ix_all);
  //Zw = arma::zeros(coords.n_rows);
  
  // domain partitioning
  indexing    = indexing_in;
  indexing_obs = indexing_obs_in;

  oneuv = arma::ones<arma::uvec>(1);
  
  // initial values
  w = arma::zeros(w_in.n_rows, 1);
  w.col(0) = w_in;
  
  tausq_inv        = arma::ones(q) * tausq_inv_in;
  tausq_inv_long   = arma::ones(y.n_elem) * tausq_inv_in;
  XB = arma::zeros(coords.n_rows);
  Bcoeff           = arma::zeros(p, q);
  for(int j=0; j<q; j++){
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * beta_in;
    Bcoeff.col(j) = beta_in;
  }
  
  // prior params
  XtX = arma::field<arma::mat>(q);
  for(int j=0; j<q; j++){
    XtX(j)   = X_available.rows(ix_by_q_a(j)).t() * 
      X_available.rows(ix_by_q_a(j));
  }
  Vi    = beta_Vi_in;//.00001 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  tausq_ab = tausq_ab_in;
  
  // initialize covariance model
  covpars = CovarianceParams(dd, q, -1);
  
  // init
  dim_by_parent       = arma::field<arma::vec> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  
  predicting = true;
  
  // now elaborate
  message("MultivariateMeshGP::MultivariateMeshGP : init_indexing()");
  init_indexing();
  
  message("MultivariateMeshGP::MultivariateMeshGP : na_study()");
  na_study();
  // now we know where NAs are, we can erase them
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("MultivariateMeshGP::MultivariateMeshGP : init_finalize()");
  init_finalize();
  
  message("MultivariateMeshGP::MultivariateMeshGP : make_gibbs_groups()");
  // quick check for groups
  make_gibbs_groups();
  
  //caching;
  if(cached){
    message("MultivariateMeshGP::MultivariateMeshGP : init_cache()");
    init_cache();
    fill_zeros_Kcache();
  }
  
  init_meshdata(theta_in);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "MultivariateMeshGP::MultivariateMeshGP initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}


void MultivariateMeshGP::make_gibbs_groups(){
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
                          << ")." << "\n";
              throw 1;
            }
          }
          for(int cc=0; cc<children(u).n_elem; cc++){
            if(block_groups(children(u)(cc)) == block_groups_labels(g)){
              Rcpp::Rcout << u << " ---> " << children(u)(cc) 
                          << ": same group (" << block_groups(u) 
                          << ")." << "\n";
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
}

void MultivariateMeshGP::na_study(){
  // prepare stuff for NA management
  
  na_1_blocks = arma::field<arma::vec> (n_blocks);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros(n_blocks);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks;i++){
    arma::vec yvec = y.rows(indexing_obs(i));
    na_1_blocks(i) = arma::zeros(yvec.n_elem);
    na_1_blocks(i).elem(arma::find_finite(yvec)).fill(1);
    na_ix_blocks(i) = arma::find(na_1_blocks(i) == 1); 
    
  }

  n_ref_blocks = 0;
  for(int i=0; i<n_blocks; i++){
    block_ct_obs(i) = arma::accu(na_1_blocks(i));
    if(block_ct_obs(i) > 0){
      n_loc_ne_blocks += indexing(i).n_elem;
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

void MultivariateMeshGP::fill_zeros_Kcache(){
  message("[fill_zeros_Kcache]");
  // ***
  K_coords_cache = arma::field<arma::mat> (coords_caching.n_elem);
  Kcp_cache = arma::field<arma::mat> (kr_caching.n_elem);
  Kxxi_cache = arma::field<arma::mat> (coords_caching.n_elem);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    K_coords_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    Kxxi_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
  }
  
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    Kcp_cache(i) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
  }
  message("[fill_zeros_Kcache] done.");
}

void MultivariateMeshGP::init_cache(){
  // coords_caching stores the layer names of those layers that are representative
  // coords_caching_ix stores info on which layers are the same in terms of rel. distance
  
  message("[init_cache]");
  //coords_caching_ix = caching_pairwise_compare_uc(coords_blocks, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching_ix = caching_pairwise_compare_uci(coords, indexing, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching = arma::unique(coords_caching_ix);
  
  //parents_caching_ix = caching_pairwise_compare_uc(parents_coords, block_names, block_ct_obs);
  parents_caching_ix = caching_pairwise_compare_uci(coords, parents_indexing, block_names, block_ct_obs);
  parents_caching = arma::unique(parents_caching_ix);
  
  arma::field<arma::mat> kr_pairing(n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents_indexing(u).n_elem > 0){//parents_coords(u).n_rows > 0){
      arma::mat cmat = coords.rows(indexing(u));
      arma::mat pmat = coords.rows(parents_indexing(u));
      arma::mat kr_mat_c = arma::join_vert(cmat, pmat);
      
      kr_pairing(u) = kr_mat_c;//arma::join_horiz(kr_mat_c, kr_mat_mvid);
    } else {
      kr_pairing(u) = arma::zeros(arma::size(parents_indexing(u)));//arma::zeros(arma::size(parents_coords(u))); // no parents
    }
  }
  
  kr_caching_ix = caching_pairwise_compare_uc(kr_pairing, block_names, block_ct_obs);
  //kr_caching_ix = caching_pairwise_compare_u(kr_pairing, block_names, block_ct_obs);

  kr_caching = arma::unique(kr_caching_ix);
  
  if(verbose){
    Rcpp::Rcout << "Caching stats c: " << coords_caching.n_elem 
                << " k: " << kr_caching.n_elem << "\n";
  }
  message("[init_cache]");
}

void MultivariateMeshGP::init_meshdata(const arma::vec& theta_in){
  message("[init_meshdata]");
  // block params
  
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_cholprec = arma::field<arma::mat> (n_blocks);
  
  param_data.zH_obs = arma::field<arma::mat> (n_blocks); //***
  param_data.Ddiag = arma::field<arma::vec> (n_blocks);//***
  param_data.KcxKxxi_obs = arma::field<arma::mat> (n_blocks); //***
  
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i) - 1;
    param_data.w_cond_cholprec(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    param_data.w_cond_mean_K(u) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
    param_data.KcxKxxi_obs(i) = arma::zeros(q*indexing_obs(i).n_elem, q*indexing(i).n_elem);
    param_data.zH_obs(i) = arma::zeros(q*indexing_obs(i).n_elem, q*indexing(i).n_elem); //***
    param_data.Ddiag(i) = arma::zeros(q*indexing_obs(i).n_elem); //***
  }
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  
  // ***
  param_data.wcore         = arma::zeros(n_blocks, 1);
  param_data.loglik_w_comps = arma::zeros(n_blocks, 1);
  param_data.loglik_w       = 0; 
  param_data.ll_y_all       = 0; 
  
  param_data.theta          = theta_in;//##
    
  alter_data                = param_data; 
  message("[init_meshdata] done.");
}

void MultivariateMeshGP::init_indexing(){
  //Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
  message("[init_indexing] indexing, parent_indexing, children_indexing");
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(parents(u).n_elem > 0){
      arma::field<arma::uvec> pixs(parents(u).n_elem);
      for(int pi=0; pi<parents(u).n_elem; pi++){
        pixs(pi) = indexing(parents(u)(pi));
      }
      parents_indexing(u) = field_v_concat_uv(pixs);
    }
  }
}

void MultivariateMeshGP::init_finalize(){
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    
    if(indexing_obs(u).n_elem > 0){ //***
      // number of coords of the jth parent of the child
      dim_by_parent(u) = arma::zeros(parents(u).n_elem + 1);
      for(int j=0; j<parents(u).n_elem; j++){
        dim_by_parent(u)(j+1) = indexing(parents(u)(j)).n_elem;//coords_blocks(parents(u)(j)).n_rows;
      }
      dim_by_parent(u) = arma::cumsum(dim_by_parent(u));
    }
  }
  message("[init_finalize] u_is_which_col_f");
  
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    if(indexing(u).n_elem > 0){
      // children-parent relationship variables
      u_is_which_col_f(u) = arma::field<arma::field<arma::uvec> > (children(u).n_elem);
      for(int c=0; c<children(u).n_elem; c++){
        int child = children(u)(c);
        //Rcpp::Rcout << "child: " << child << "\n";
        // which parent of child is u which we are sampling
        arma::uvec u_is_which = arma::find(parents(child) == u, 1, "first"); 
        
        // which columns correspond to it
        int firstcol = dim_by_parent(child)(u_is_which(0));
        int lastcol = dim_by_parent(child)(u_is_which(0)+1);

        int dimen = parents_indexing(child).n_elem;

        // / / /
        arma::uvec result = arma::regspace<arma::uvec>(0, dimen-1);
        arma::uvec rowsel = arma::zeros<arma::uvec>(result.n_rows);
        rowsel.subvec(firstcol, lastcol-1).fill(1);
        arma::uvec result_local = result.rows(arma::find(rowsel==1));
        arma::uvec result_other = result.rows(arma::find(rowsel==0));
        u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        u_is_which_col_f(u)(c)(0) = result_local; // u parent of c is in these columns for c
        u_is_which_col_f(u)(c)(1) = result_other; // u parent of c is NOT in these columns for c
        
      }
    }
  }
}

bool MultivariateMeshGP::get_loglik_comps_w(MeshDataMV& data){
  if(cached){
    return get_cond_comps_loglik_w(data);
  } else {
    return get_cond_comps_loglik_w_nocache(data);
  }
}

bool MultivariateMeshGP::get_cond_comps_loglik_w(MeshDataMV& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w] start.");
  
  arma::field<arma::mat> K_cholcp_cache(kr_caching.n_elem);
  arma::field<arma::mat> w_cond_mean_cache(kr_caching.n_elem);
  
  //arma::vec timings = arma::zeros(10);
  
  arma::vec cparams = data.theta;
  covpars.transform(cparams);
  
  int errtype = -1;
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); // layer name of ith representative
    if(compute_block(predicting, block_ct_obs(u), false)){
      Covariancef_inplace(K_coords_cache(i), coords, qvblock_c, 
                          indexing(u), indexing(u), covpars, true);
      if(forced_grid){
        try {
          Kxxi_cache(i) = arma::inv_sympd( K_coords_cache(i) ); //***
        } catch(...) {
          errtype = 1;
        }
      }
    }
  }
  
  if(errtype > 0){
    return false;
  }
  //end = std::chrono::steady_clock::now();
  //timings(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    try {
      if(block_ct_obs(u) > 0){
        int u_cached_ix = coords_caching_ix(u);
        
        //start = std::chrono::steady_clock::now();
        arma::uvec cx = arma::find( coords_caching == u_cached_ix );
        arma::mat Kcc = K_coords_cache(cx(0));
        
        arma::mat Kxx = Covariancef(coords, qvblock_c, parents_indexing(u), parents_indexing(u), covpars, true);
        arma::mat Kxxi_c = arma::inv(arma::trimatl(arma::chol(Kxx, "lower")));
        
        Covariancef_inplace(Kcp_cache(i), coords, qvblock_c, indexing(u), parents_indexing(u), covpars, false);
        
        arma::mat Kcx_Kxxic = Kcp_cache(i) * Kxxi_c.t();
        w_cond_mean_cache(i) = Kcx_Kxxic * Kxxi_c;
        
        arma::mat Kinside = Kcc - Kcx_Kxxic * Kcx_Kxxic.t();
        
        K_cholcp_cache(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
            Kinside) , "lower")));
        
        //end = std::chrono::steady_clock::now();
        //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      }
    } catch (...) {
      errtype = 2;
    }
  }

  if(errtype > 0){
    if(verbose){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "ai1: " << covpars.ai1.t() << "\n"
                  << "ai2: " << covpars.ai2.t() << "\n"
                  << "phi_i: " << covpars.phi_i.t() << "\n"
                  << "thetamv: " << covpars.thetamv.t() << "\n"
                  << "and Dmat: " << covpars.Dmat << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  
  arma::mat ll_y = arma::zeros(w.n_rows, 1);
  
  //start = std::chrono::steady_clock::now();
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i = 0; i<n_ref_blocks; i++){
      int r = reference_blocks(i);
      int u = block_names(r)-1;
      int u_cached_ix = coords_caching_ix(u);
      arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
      arma::mat Kcc = K_coords_cache(cx(0));
      
      arma::mat w_x = w.rows(indexing(u));
      arma::mat cond_mean_K, cond_mean, cond_cholprec;
      
      if( parents(u).n_elem > 0 ){
        int kr_cached_ix = kr_caching_ix(u);
        arma::uvec cpx = arma::find(kr_caching == kr_cached_ix, 1, "first" );
        cond_mean_K = w_cond_mean_cache(cpx(0));
        cond_cholprec = K_cholcp_cache(cpx(0));
        w_x = w_x - cond_mean_K * w.rows(parents_indexing(u));
      } else {
        cond_mean_K = arma::zeros(arma::size(parents(u)));
        cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
      }
      data.w_cond_mean_K(u) = cond_mean_K;
      data.w_cond_cholprec(u) = cond_cholprec;
      data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
      
      arma::vec ccholprecdiag = cond_cholprec.diag();//(na_ix_blocks(u), na_ix_blocks(u));
      data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));//(na_ix_blocks(u))));
    
      data.wcore.row(u) = w_x.t() * data.w_cond_prec(u) * w_x;
      data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * data.wcore.row(u);
    
      if(forced_grid){
        arma::mat Kxxi = Kxxi_cache(cx(0));
        arma::mat Kcx = Covariancef(coords, qvblock_c, indexing_obs(u), indexing(u), covpars, false);
        data.KcxKxxi_obs(u) = Kcx * Kxxi;
        //arma::mat ZK = Zblock(u) * Kcx;
        data.zH_obs(u) = data.KcxKxxi_obs(u); //*** should include Z
        //arma::mat zHHz_obs = data.zH_obs(u) * Kcx.t() * Zblock(u).t();
        
        // if the grid is forced it likely is not overlapping with the data
        for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
          if(na_1_blocks(u)(ix) == 1){
            arma::mat Kcc = Covariancef(coords, qvblock_c, 
                                        indexing_obs(u).row(ix), indexing_obs(u).row(ix), 
                                        covpars, true);
            arma::mat Kexpl = data.KcxKxxi_obs(u).row(ix) * arma::trans(Kcx.row(ix));
            double Rj = Kcc(0,0)-Kexpl(0,0);
            data.Ddiag(u)(ix) = std::max(Rj, 0.0);
            
            if(Rj > 1e-5){
              double ysigmasq = data.Ddiag(u)(ix) + 
                1.0/tausq_inv_long(indexing_obs(u)(ix));
              
              // here we dont use the non-reference conditional means that was calculated before
              // because we have proposed values for theta
       
              double KXw = arma::conv_to<double>::from(data.zH_obs(u).row(ix) * w.rows(indexing(u)));
              double ytilde = y(indexing_obs(u)(ix)) - XB(indexing_obs(u)(ix)) - KXw;
              ll_y.row(indexing_obs(u)(ix)) = hl2pi -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
            }
          }
        }
      }
    }
    //data.logdetCi = arma::accu(data.logdetCi_comps);
    //double w_contribution = arma::accu(data.loglik_w_comps);
    //data.ll_y_all = arma::accu(ll_y);
    
    data.loglik_w = arma::accu(data.logdetCi_comps) + 
      arma::accu(data.loglik_w_comps) +
      arma::accu(ll_y);
    
    //end = std::chrono::steady_clock::now();
    //timings(8) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_cached_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  return true;
}

bool MultivariateMeshGP::get_cond_comps_loglik_w_nocache(MeshDataMV& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w_nocache] start. ");
  
  arma::vec cparams = data.theta;
  covpars.transform(cparams);
  
  int errtype = -1;
  
  arma::vec pm_w = arma::zeros(n_blocks);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //if(compute_block(predicting, block_ct_obs(u), false) ){
    try {
      if(block_ct_obs(u) > 0){
        // skip calculating stuff for blocks in the predict-only area 
        // when it's not prediction time
        arma::mat cond_mean_K, cond_mean, cond_cholprec;
        
        arma::mat Kcc = Covariancef(coords, qvblock_c, indexing(u), indexing(u), covpars, true);

        if( parents(u).n_elem > 0 ){
          arma::mat Kxxi =
            arma::inv_sympd(Covariancef(coords, qvblock_c, parents_indexing(u), parents_indexing(u), covpars, true));
          arma::mat Kcx = Covariancef(coords, qvblock_c, indexing(u), parents_indexing(u), covpars, false);
          cond_mean_K = Kcx * Kxxi;
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( arma::symmatu(Kcc - cond_mean_K * Kcx.t()) , "lower")));
        } else {
          cond_mean_K = arma::zeros(0, 0);
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
        }
        
        if(errtype < 0){
          data.w_cond_mean_K(u) = cond_mean_K;
          data.w_cond_cholprec(u) = cond_cholprec;
          data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
          
          arma::mat w_x = w.rows(indexing(u));
          if(parents(u).n_elem > 0){
            w_x -= data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
          }
          
          arma::vec ccholprecdiag = cond_cholprec.diag();
          data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));
          
          data.wcore.row(u) = //arma::conv_to<double>::from(
            w_x.t() * data.w_cond_prec(u) * w_x;//);
          data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * data.wcore.row(u);
          
        }
      }
    } catch (...) {
      errtype = 1;
    }  
  }
  if(errtype > 0){
    if(verbose & debug){
      Rcpp::Rcout << "Cholesky failed at some point. Here's the value of theta that caused this" << "\n";
      Rcpp::Rcout << "ai1: " << covpars.ai1.t() << "\n"
                  << "ai2: " << covpars.ai2.t() << "\n"
                  << "phi_i: " << covpars.phi_i.t() << "\n"
                  << "thetamv: " << covpars.thetamv.t() << "\n"
                  << "and Dmat: " << covpars.Dmat << "\n";
      Rcpp::Rcout << " -- auto rejected and proceeding." << "\n";
    }
    return false;
  }
  
  //data.logdetCi = arma::accu(data.logdetCi_comps);
  data.loglik_w = arma::accu(data.logdetCi_comps) + arma::accu(data.loglik_w_comps);
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_loglik_w_nocache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  return true;
}

void MultivariateMeshGP::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat bmat = arma::randn(p, q);
  
  for(int j=0; j<q; j++){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * XtX(j) + Vi), "lower");
    arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
    arma::vec w_available = w.submat(na_ix_all, 0*oneuv);
    arma::mat Xprecy_j = Vim + tausq_inv(j) * X_available.rows(ix_by_q_a(j)).t() * 
      (y_available.rows(ix_by_q_a(j)) - w_available.rows(ix_by_q_a(j)));
    
    Bcoeff.col(j) = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j + bmat.col(j));
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * Bcoeff.col(j);
  }
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void MultivariateMeshGP::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  arma::vec Zw_availab = w.submat(na_ix_all, 0*oneuv);
  arma::vec XB_availab = XB.rows(na_ix_all);
  logpost = 0;
  for(int j=0; j<q; j++){
    
    arma::mat yrr = y_available.rows(ix_by_q_a(j)) - XB_availab.rows(ix_by_q_a(j)) - Zw_availab.rows(ix_by_q_a(j));
    double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
    
    double aparam = 2.001 + ix_by_q_a(j).n_elem/2.0;
    double bparam = 1.0/( 1.0 + .5 * bcore );
    
    Rcpp::RNGScope scope;
    tausq_inv(j) = R::rgamma(aparam, bparam);
    // fill all (not just available) corresponding to same variable.
    tausq_inv_long.rows(ix_by_q(j)).fill(tausq_inv(j));
    
    logpost += 0.5 * (ix_by_q_a(j).n_elem + .0) * log(tausq_inv(j)) - 0.5*tausq_inv(j)*bcore;
    
    if(verbose & debug){
      Rcpp::Rcout << "[gibbs_sample_tausq] " << j << ", "
                  << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv(j)
                  << "\n";
    }
  }
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_tausq] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. \n";
  }
}

void MultivariateMeshGP::logpost_refresh_after_gibbs(){
  message("[logpost_refresh_after_gibbs]");
  start = std::chrono::steady_clock::now();
  
  arma::vec ll_y = arma::zeros(w.n_rows);
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    arma::mat w_x = w.rows(indexing(u));
    if(parents(u).n_elem > 0){
      w_x -= param_data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
    }
    
    param_data.wcore.row(u) = //arma::conv_to<double>::from(
      w_x.t() * param_data.w_cond_prec(u) * w_x;
    param_data.loglik_w_comps.row(u) = (indexing(u).n_elem+.0) * hl2pi -.5 * param_data.wcore.row(u);
    
    if(forced_grid){
      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(abs(param_data.Ddiag(u)(ix) > 1e-5)){//if(na_1_blocks(u)(ix) == 1){
          double ysigmasq = param_data.Ddiag(u)(ix) + 1.0/tausq_inv_long(indexing_obs(u)(ix));
        
          double ytilde = y(indexing_obs(u)(ix)) - XB(indexing_obs(u)(ix)) - 
            arma::conv_to<double>::from(w.row(indexing_obs(u)(ix))); //*** should include Z
          ll_y(indexing_obs(u)(ix)) = hl2pi -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
        
        }
      }
    }
  }
  //param_data.logdetCi = arma::accu(param_data.logdetCi_comps);
  //double w_contribution = arma::accu(param_data.loglik_w_comps);
  //param_data.ll_y_all = arma::accu(ll_y);
  
  param_data.loglik_w = arma::accu(param_data.logdetCi_comps) + 
    arma::accu(param_data.loglik_w_comps) + 
    arma::accu(ll_y);
  
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[logpost_refresh_after_gibbs] " 
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
                << "us.\n";
  }
}

void MultivariateMeshGP::gibbs_sample_w(){
  if(!forced_grid){
    // sampling at same locations as data
    gibbs_sample_w_omp_inputgrid();
  } else {
    // sampling at a fixed grid regardless of data
    gibbs_sample_w_omp_forcedgrid();
  }
}

void MultivariateMeshGP::gibbs_sample_w_omp_inputgrid(){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] " << "\n";
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, 1);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << "\n" << " ? " << "\n";
  start_overall = std::chrono::steady_clock::now();
  //arma::uvec gibbs_groups_reorder = ashuffle(arma::regspace<arma::uvec>(0, n_gibbs_groups-1));
  //needs_update = true; 
  
  int errtype=-1;
  
  for(int g=0; g<n_gibbs_groups-predict_group_exists; g++){
    //int g = gibbs_groups_reorder(go);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(compute_block(predicting, block_ct_obs(u), false)){
        if((block_ct_obs(u) > 0)){
          arma::mat Smu_tot = arma::zeros(indexing(u).n_elem, 1);
          
          arma::mat Sigi_tot = param_data.w_cond_prec(u);
          arma::vec Sigi_diag_add = tausq_inv_long.rows(indexing(u)) % na_1_blocks(u);
          Sigi_tot.diag() += Sigi_diag_add;
          
          if(parents(u).n_elem>0){
            Smu_tot += param_data.w_cond_prec(u) * param_data.w_cond_mean_K(u) * 
              w.rows( parents_indexing(u) );
          }
          
          for(int c=0; c<children(u).n_elem; c++){
            int child = children(u)(c);
            //clog << "g: " << g << " ~ u: " << u << " ~ child " << c << " - " << child << "\n";
            arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
            arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
            arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
            
            arma::mat w_child = w.rows(indexing(child));
            arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
            arma::mat w_par_child_select = w_parents_of_child.rows(u_is_which_col_f(u)(c)(1));
            
            Sigi_tot += AK_uP * AK_u;
            Smu_tot += AK_uP * ( w_child - AK_others * w_par_child_select );
          }
          
          arma::mat Smu_y_add = ((tausq_inv_long.rows(indexing(u)) % na_1_blocks(u)) % 
            ( y.rows(indexing(u)) - XB.rows(indexing(u)) ));
          
          Smu_tot += //Zblock(u).t() * 
            Smu_y_add;
          
          //start = std::chrono::steady_clock::now();
          arma::mat Sigi_chol;
          try{
            Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
            // sample
            arma::mat rnvec = rand_norm_mat.rows(indexing(u));
            w.rows(indexing(u)) = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec); 
            
          } catch (...) {
            errtype = 11;
          }
        }
      //} 
    }
    if(errtype > 0){
      Rcpp::Rcout << "Error in Cholesky when sampling w " << endl;
      throw 1;
    }
  }
  
  //Zw = armarowsum(Z % w);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}

void MultivariateMeshGP::gibbs_sample_w_omp_forcedgrid(){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] " << "\n";
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows);
  start_overall = std::chrono::steady_clock::now();
  
  for(int g=0; g<n_gibbs_groups-predict_group_exists; g++){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      if((block_ct_obs(u) > 0)){
        arma::mat Smu_tot = arma::zeros(indexing(u).n_elem, 1);
        arma::mat Sigi_tot = param_data.w_cond_prec(u);
        
        if(parents(u).n_elem>0){
          Smu_tot += param_data.w_cond_prec(u) * param_data.w_cond_mean_K(u) * w.rows( parents_indexing(u) );//param_data.w_cond_mean(u);
        }
        
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          //clog << "g: " << g << " ~ u: " << u << " ~ child " << c << " - " << child << "\n";
          arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
          arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
          arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
          
          arma::mat w_child = w.rows(indexing(child));
          arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
          arma::mat w_par_child_select = w_parents_of_child.rows(u_is_which_col_f(u)(c)(1));
          
          Sigi_tot += AK_uP * AK_u;
          Smu_tot += AK_uP * ( w_child - AK_others * w_par_child_select );
        }
        
        
        arma::vec u_tausq_inv = arma::zeros(indexing_obs(u).n_elem);
        for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
          if(na_1_blocks(u)(ix) == 1){
            u_tausq_inv(ix) = 1.0 / (1.0/tausq_inv_long(indexing_obs(u)(ix)) + param_data.Ddiag(u)(ix));
          }
        }
        arma::mat Zt_Ditau = param_data.zH_obs(u).t() * arma::diagmat(u_tausq_inv);
        
        arma::mat Sigi_y = Zt_Ditau * param_data.zH_obs(u);
        arma::mat Smu_y = Zt_Ditau * //findme
          ( y.rows(indexing_obs(u)) - XB.rows(indexing_obs(u)));
        
        Sigi_tot += Sigi_y;
        Smu_tot += Smu_y;
        
        arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
        
        // sample
        //arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
        arma::mat rnvec = rand_norm_mat.rows(indexing(u));
        w.rows(indexing(u)) = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec); 
        
        // instead of sampling the non-reference,
        // we store the conditional mean so we can reuse it!
        // (in fact we never need samples of non-reference locs)
        w.rows(indexing_obs(u)) = param_data.KcxKxxi_obs(u) * w.rows(indexing(u));
      }
    }
  }
  //Zw = armarowsum(Z % w);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

void MultivariateMeshGP::predict(bool needs_update=true){
  start_overall = std::chrono::steady_clock::now();
  //arma::vec timings = arma::zeros(5);
  arma::vec cparams = param_data.theta;
  covpars.transform(cparams);
  
  if(predict_group_exists == 1){
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_predicts.n_elem; i++){
      int u = u_predicts(i);
      // only predictions at this block. 
      arma::mat Kxxi = arma::inv_sympd( 
        Covariancef(coords, qvblock_c, 
                    parents_indexing(u), parents_indexing(u), covpars, true));
      
      arma::mat Kcx = 
        Covariancef(coords, qvblock_c, 
                    indexing(u), parents_indexing(u), covpars, false);
      
      arma::vec Var1diagx = arma::zeros(indexing(u).n_elem);
      for(int j=0; j<indexing(u).n_elem; j++){
        arma::mat result = Covariancef(coords, qvblock_c, indexing(u).row(j), indexing(u).row(j), covpars, true);
        Var1diagx(j) = result(0,0);
      }
      arma::mat KxxiKcx = Kxxi * Kcx.t();
      arma::vec Var2diagx = 
        arma::trans(arma::sum(Kcx.t() % KxxiKcx, 0));
      arma::mat prVarx = abs(Var1diagx - Var2diagx);
      
      w.rows(indexing(u)) = KxxiKcx.t() * w.rows(parents_indexing(u)) + 
        pow(prVarx, .5) % rand_norm_mat.rows(indexing(u));
      
      //***
      if(forced_grid){
        // need to make predictions at the non-reference locs
        arma::mat Kyx = Covariancef(coords, qvblock_c, indexing_obs(u), parents_indexing(u), covpars, false);
        arma::vec Var1diag = arma::zeros(indexing_obs(u).n_elem);
        for(int j=0; j<indexing_obs(u).n_elem; j++){
          arma::mat result = Covariancef(coords, qvblock_c, indexing_obs(u).row(j), indexing_obs(u).row(j), covpars, true);
          Var1diag(j) = result(0,0);
        }
        
        arma::mat KxxiKyx = Kxxi * Kyx.t();
        arma::vec Var2diag = 
          arma::trans(arma::sum(Kyx.t() % KxxiKyx, 0));
        arma::mat prVar = Var1diag - Var2diag;
        
        w.rows(indexing_obs(u)) = KxxiKyx.t() * w.rows(parents_indexing(u)) + 
          pow(prVar, .5) % rand_norm_mat.rows(indexing_obs(u));
      }
    }
  }
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[predict] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

void MultivariateMeshGP::theta_update(MeshDataMV& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void MultivariateMeshGP::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void MultivariateMeshGP::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void MultivariateMeshGP::accept_make_change(){
  std::swap(param_data, alter_data);
}