#include <RcppArmadillo.h>
#include <omp.h>
#include <stdexcept>

#include "R.h"
#include "find_nan.h"
#include "mh_adapt.h"
#include "field_v_concatm.h"
#include "caching_pairwise_compare.h"
#include "covariance_functions.h"
#include "debug.h"
#include "mgp_utils.h"
// with indexing
// without block extensions (obs with NA are left in)

using namespace std;

//#include <RcppArmadilloExtensions/sample.h>

//arma::uvec ashuffle(const arma::uvec& x){
  //  Rcpp::RNGScope scope;
//  return Rcpp::RcppArmadillo::sample(x, x.n_elem, false); 
//}

const double hl2pi = -.5 * log(2.0 * M_PI);

class MeshGPmv {
public:
  // meta
  int n;
  int p;
  int q;
  int dd;
  int n_blocks;
  int npars;
  
  // data
  arma::vec y;
  arma::mat X;
  //arma::mat Z;
  
  arma::vec y_available;
  arma::mat X_available;
  //arma::mat Z_available;
  
  arma::mat coords;
  arma::uvec mv_id;
  arma::uvec qvblock_c;
  
  // block membership
  arma::uvec blocking;
  
  arma::uvec reference_blocks; // named
  int n_ref_blocks;
  
  //arma::field<arma::sp_mat> Zblock;
  //arma::vec Zw;
  
  // indexing info
  arma::field<arma::uvec> indexing; 
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
  int                       predict_group_exists;
  arma::uvec                u_predicts;
  arma::vec                 block_groups_labels;
  // for each block's children, which columns of parents of c is u? and which instead are of other parents
  arma::field<arma::field<arma::field<arma::uvec> > > u_is_which_col_f; 
  arma::field<arma::vec>    dim_by_parent;
  
  // params
  arma::vec w;
  arma::mat Bcoeff; // sampled
  double sigmasq;
  arma::mat rand_norm_mat;
  
  arma::vec XB;
  arma::vec tausq_inv; // tausq for the l=q variables
  arma::vec tausq_inv_long; // storing tausq_inv at all locations
  
  // params with mh step
  MeshDataMV param_data; 
  MeshDataMV alter_data;
  
  // setup
  bool predicting;
  //bool rfc_dep;
  bool cached;
  bool cached_gibbs;
  
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
  // caching of w sampling covariance matrices
  arma::vec gibbs_caching;
  arma::vec gibbs_caching_ix;
  
  // caching some matrices // ***
  arma::field<arma::mat> K_coords_cache;
  arma::field<arma::mat> K_parents_cache_noinv;
  arma::field<arma::mat> Kcp_cache;
  
  // MCMC
  void get_loglik_w(MeshDataMV& data);
  void get_loglik_comps_w(MeshDataMV& data);
  void get_cond_comps_loglik_w(MeshDataMV& data);
  void get_cond_comps_loglik_w_nocache(MeshDataMV& data);
  
  void gibbs_sample_w(bool);
  void gibbs_sample_w_omp(bool);
  void gibbs_sample_w_omp_nocache(bool);
  
  void predict(bool);
  
  void gibbs_sample_beta();
  void gibbs_sample_sigmasq();
  void gibbs_sample_tausq();
  
  // changing the values, no sampling
  void tausq_update(double);
  int n_cbase;
  arma::vec ai1;
  arma::vec ai2;
  arma::vec phi_i;
  arma::vec thetamv;
  arma::mat Dmat;
  void theta_transform(const MeshDataMV&);
  void theta_update(MeshDataMV&, const arma::vec&);
  void beta_update(const arma::vec&);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  MeshGPmv();
  
  // build everything
  MeshGPmv(
    const arma::vec& y_in, 
    const arma::mat& X_in, 
    
    const arma::mat& coords_in, 
    const arma::uvec& mv_id_in,
    
    const arma::uvec& blocking_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& layers_names,
    const arma::vec& block_group_in,
    
    const arma::field<arma::uvec>& indexing_in,
    
    const arma::vec& w_in,
    const arma::vec& beta_in,
    const double& sigmasq_in,
    const arma::vec& theta_in,
    double tausq_inv_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& tausq_ab_in,
    
    bool use_cache,
    bool use_cache_gibbs,
    
    bool v,
    bool debugging);
  
};

void MeshGPmv::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

MeshGPmv::MeshGPmv(){
  
}

MeshGPmv::MeshGPmv(
  const arma::vec& y_in, 
  const arma::mat& X_in, 
  
  const arma::mat& coords_in, 
  const arma::uvec& mv_id_in,
  
  const arma::uvec& blocking_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  
  const arma::field<arma::uvec>& indexing_in,
  
  const arma::vec& w_in,
  const arma::vec& beta_in,
  const double& sigmasq_in,
  const arma::vec& theta_in,
  double tausq_inv_in,
  
  const arma::mat& beta_Vi_in,
  const arma::vec& tausq_ab_in,
  
  bool use_cache=true,
  bool use_cache_gibbs=false,
  
  bool v=false,
  bool debugging=false){
  
  printf("~ MeshGPmv initialization.\n");
  
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
  
  message("MeshGPmv::MeshGPmv assign values.");
  
  y                   = y_in;
  X                   = X_in;
  //Z                   = Z_in;
  
  coords              = coords_in;
  mv_id               = mv_id_in;
  qvblock_c           = mv_id-1;
  
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
  //Z_available = Z.rows(na_ix_all);
  
  //Zw = arma::zeros(coords.n_rows);
  
  indexing    = indexing_in;
  
  n  = na_ix_all.n_elem;
  p  = X.n_cols;
  arma::uvec mv_id_uniques = arma::unique(mv_id);
  q  = mv_id_uniques.n_elem;//Z.n_cols;
  dd = coords.n_cols;
  
  
  ix_by_q = arma::field<arma::uvec>(q);
  ix_by_q_a = arma::field<arma::uvec>(q);
  arma::uvec qvblock_c_available = qvblock_c.rows(na_ix_all);
  for(int j=0; j<q; j++){
    ix_by_q(j) = arma::find(qvblock_c == j);
    ix_by_q_a(j) = arma::find(qvblock_c_available == j);
  }
  
  if(dd == 2){
    if(q == 1){
      npars = 1+1; //##
    } else {
      int n_cbase = q > 2? 3: 1;
      npars = 3*q + n_cbase; // ## 
    }
  } else {
    Rcpp::Rcout << "d>2 not implemented for multivariate outcomes, yet " << endl;
    throw 1;
  }
  
  
  
  printf("%d observed locations, %d to predict, %d total\n",
         n, y.n_elem-n, y.n_elem);
  
  // init
  dim_by_parent       = arma::field<arma::vec> (n_blocks);
  Ib                  = arma::field<arma::sp_mat> (n_blocks);
  u_is_which_col_f    = arma::field<arma::field<arma::field<arma::uvec> > > (n_blocks);
  
  sigmasq          = sigmasq_in;
  tausq_inv        = arma::ones(q) * tausq_inv_in;
  tausq_inv_long   = arma::ones(y.n_elem) * tausq_inv_in;
  XB = arma::zeros(coords.n_rows);
  Bcoeff           = arma::zeros(p, q);
  for(int j=0; j<q; j++){
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * beta_in;
    Bcoeff.col(j) = beta_in;
  }
  
  w                = w_in;
  
  predicting = true;
  //rfc_dep    = use_rfc;
  
  
  // now elaborate
  message("MeshGPmv::MeshGPmv : init_indexing()");
  init_indexing();
  
  
  message("MeshGPmv::MeshGPmv : na_study()");
  na_study();
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("MeshGPmv::MeshGPmv : init_finalize()");
  init_finalize();
  
  // prior for beta
  // prior for beta
  XtX = arma::field<arma::mat>(q);
  for(int j=0; j<q; j++){
    XtX(j)   = X_available.rows(ix_by_q_a(j)).t() * 
      X_available.rows(ix_by_q_a(j));
  }
  
  Vi    = beta_Vi_in;//.00001 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  // priors for tausq and sigmasq
  //sigmasq_ab = sigmasq_ab_in;
  tausq_ab = tausq_ab_in;
  
  message("MeshGPmv::MeshGPmv : make_gibbs_groups()");
  make_gibbs_groups();
  
  //caching;
  if(cached){
    message("MeshGPmv::MeshGPmv : init_cache()");
    init_cache();
    fill_zeros_Kcache();
  }
  
  init_meshdata(theta_in);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "MeshGPmv::MeshGPmv initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}


void MeshGPmv::make_gibbs_groups(){
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
  
  //u_by_block_groups = u_by_block_groups_temp;
}

void MeshGPmv::na_study(){
  // prepare stuff for NA management
  
  na_1_blocks = arma::field<arma::vec> (n_blocks);//(y_blocks.n_elem);
  na_ix_blocks = arma::field<arma::uvec> (n_blocks);//(y_blocks.n_elem);
  n_loc_ne_blocks = 0;
  block_ct_obs = arma::zeros(n_blocks);//(y_blocks.n_elem);
  
#pragma omp parallel for
  for(int i=0; i<n_blocks;i++){//y_blocks.n_elem; i++){
    arma::vec yvec = y.rows(indexing(i));//y_blocks(i);
    na_1_blocks(i) = arma::zeros(yvec.n_elem);
    na_1_blocks(i).elem(arma::find_finite(yvec)).fill(1);
    na_ix_blocks(i) = arma::find(na_1_blocks(i) == 1); 
    
  }

  n_ref_blocks = 0;
  for(int i=0; i<n_blocks; i++){//y_blocks.n_elem; i++){
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

void MeshGPmv::fill_zeros_Kcache(){
  // ***
  K_coords_cache = arma::field<arma::mat> (coords_caching.n_elem);
  Kcp_cache = arma::field<arma::mat> (kr_caching.n_elem);
  K_parents_cache_noinv = arma::field<arma::mat> (parents_caching.n_elem);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    K_coords_cache(i) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
  }
  
  for(int i=0; i<parents_caching.n_elem; i++){
    int u = parents_caching(i); 
    K_parents_cache_noinv(i) = arma::zeros(parents_indexing(u).n_elem, parents_indexing(u).n_elem);
  }
  
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    Kcp_cache(i) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
  }
}

void MeshGPmv::init_cache(){
  // coords_caching stores the layer names of those layers that are representative
  // coords_caching_ix stores info on which layers are the same in terms of rel. distance
  
  printf("~ Starting to search block duplicates for caching\n");
  //coords_caching_ix = caching_pairwise_compare_uc(coords_blocks, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching_ix = caching_pairwise_compare_uci(coords, indexing, block_names, block_ct_obs); // uses block_names(i)-1 !
  coords_caching = arma::unique(coords_caching_ix);
  
  //parents_caching_ix = caching_pairwise_compare_uc(parents_coords, block_names, block_ct_obs);
  parents_caching_ix = caching_pairwise_compare_uci(coords, parents_indexing, block_names, block_ct_obs);
  parents_caching = arma::unique(parents_caching_ix);
  
  arma::field<arma::mat> kr_pairing(n_blocks);
#pragma omp parallel for
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
  
  //kr_caching_ix = caching_pairwise_compare_uc(kr_pairing, block_names, block_ct_obs);
  kr_caching_ix = caching_pairwise_compare_u(kr_pairing, block_names);

  kr_caching = arma::unique(kr_caching_ix);
  
  if(cached_gibbs){
    arma::field<arma::mat> gibbs_pairing(n_blocks);
#pragma omp parallel for
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
    gibbs_caching_ix = caching_pairwise_compare_u(gibbs_pairing, block_names);
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

void MeshGPmv::init_meshdata(const arma::vec& theta_in){
  
  
  // block params
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_cholprec = arma::field<arma::mat> (n_blocks);
  for(int i=0; i<n_blocks; i++){
    int u=block_names(i) - 1;
    param_data.w_cond_cholprec(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
    param_data.w_cond_mean_K(u) = arma::zeros(indexing(u).n_elem, parents_indexing(u).n_elem);
  }
  param_data.Sigi_chol = arma::field<arma::mat>(n_blocks);
  
  if(cached_gibbs){
    param_data.Sigi_chol_cached = arma::field<arma::mat>(gibbs_caching.n_elem);
  }
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  param_data.theta          = theta_in;//##
    
  param_data.cholfail       = false;
  param_data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  //param_data.sigmasq          = sigmasq_in;
  alter_data                = param_data; 
}

void MeshGPmv::init_indexing(){
  
  //Zblock = arma::field<arma::sp_mat> (n_blocks);
  parents_indexing = arma::field<arma::uvec> (n_blocks);
  children_indexing = arma::field<arma::uvec> (n_blocks);
  
  printf("~ Indexing\n");
  message("[init_indexing] indexing, parent_indexing, children_indexing");
  
#pragma omp parallel for
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
    //Zblock(u) = Zify( Z.rows(indexing(u)) );
  }
  
}

void MeshGPmv::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
#pragma omp parallel for //**
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
  
  for(int i=0; i<n_blocks; i++){
    int u = block_names(i)-1;
    //Rcpp::Rcout << "block: " << u << "\n";
    
    //if(coords_blocks(u).n_elem > 0){ //**
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
        //Rcpp::Rcout << "from: " << firstcol << " to: " << lastcol << endl; 
        
        int dimen = parents_indexing(child).n_elem;
        /*
        arma::vec colix = arma::zeros(dimen);//parents_coords(child).n_rows);//(w_cond_mean_K(child).n_cols);
        //arma::uvec indx_scheme = arma::regspace<arma::uvec>(0, q, q*(dimen-1));
        
        for(int s=0; s<q; s++){
          //arma::uvec c_indices = s + indx_scheme.subvec(firstcol, lastcol-1);

          int shift = s * dimen;
          colix.subvec(shift + firstcol, shift + lastcol-1).fill(1);
          //colix.elem(c_indices).fill(1);
        }*/
        //Rcpp::Rcout << indx_scheme << "\n";
        //Rcpp::Rcout << colix << "\n";
        
        //Rcpp::Rcout << "visual representation of which ones we are looking at " << endl
        //            << colix.t() << endl;
        
        //u_is_which_col_f(u)(c) = arma::field<arma::uvec> (2);
        //u_is_which_col_f(u)(c)(0) = arma::find(colix == 1); // u parent of c is in these columns for c
        //u_is_which_col_f(u)(c)(1) = arma::find(colix != 1); // u parent of c is NOT in these columns for c
        
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

void MeshGPmv::get_loglik_w(MeshDataMV& data){
  start = std::chrono::steady_clock::now();
  if(verbose){
    Rcpp::Rcout << "[get_loglik_w] entering \n";
  }
#pragma omp parallel for //**
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if((block_ct_obs(u) > 0) & calc_this_block){
    //if( (block_ct_obs(u) > 0) & compute_block(predicting, block_ct_obs(u), false) ){
      //double expcore = -.5 * arma::conv_to<double>::from( (w_blocks(u) - data.w_cond_mean(u)).t() * data.w_cond_prec(u) * (w_blocks(u) - data.w_cond_mean(u)) );
      arma::mat w_x = w.rows(indexing(u));
      
      if(parents(u).n_elem > 0){
        //arma::vec w_pars = arma::vectorise( arma::trans(  ));
        w_x -= data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
      }
      
      //w_x = w_x % na_1_blocks(u);
      data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
      
      data.loglik_w_comps(u) = //block_ct_obs(u)//
        (indexing(u).n_elem + .0) 
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

void MeshGPmv::get_loglik_comps_w(MeshDataMV& data){
  if(cached){
    get_cond_comps_loglik_w(data);
  } else {
    get_cond_comps_loglik_w_nocache(data);
  }
  //print_data(data);
}

void MeshGPmv::theta_transform(const MeshDataMV& data){
  // from vector to all covariance components
  int k = data.theta.n_elem - npars; // number of cross-distances = p(p-1)/2
  
  arma::vec cparams = data.theta.subvec(0, npars - 1);//##arma::join_vert(sigmasq * arma::ones(1), data.theta.subvec(0, npars - 1));
  if((dd == 2) & (q == 1)){
    thetamv = cparams;//arma::join_vert(sigmasq * arma::ones(1), cparams);
  } else {
    int n_cbase = q > 2? 3: 1;
    ai1 = cparams.subvec(0, q-1);
    ai2 = cparams.subvec(q, 2*q-1);
    phi_i = cparams.subvec(2*q, 3*q-1);
    thetamv = cparams.subvec(3*q, 3*q+n_cbase-1);
    
    if(k>0){
      Dmat = vec_to_symmat(data.theta.subvec(npars, npars + k - 1));
    } else {
      Dmat = arma::zeros(1,1);
    }
  } 
}


void MeshGPmv::get_cond_comps_loglik_w(MeshDataMV& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w] start.");
  
  //arma::field<arma::mat> K_coords_cache(coords_caching.n_elem);
  //arma::field<arma::mat> K_parents_cache(parents_caching.n_elem);
  arma::field<arma::mat> K_cholcp_cache(kr_caching.n_elem);
  arma::field<arma::mat> w_cond_mean_cache(kr_caching.n_elem); // +++++++++
  //arma::field<arma::mat> Kcp_cache(kr_caching.n_elem);
  
  //arma::vec timings = arma::zeros(10);
  
  theta_transform(data);
  
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); // layer name of ith representative
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if(calc_this_block){
    if(compute_block(predicting, block_ct_obs(u), false)){
      //uhm ++;
      //K_coords_cache(i) = Kpp(coords_blocks(u), coords_blocks(u), Kparam, true);
      //xCovHUV_inplace(K_coords_cache(i), coords, indexing(u), indexing(u), cparams, Dmat, true);
      mvCovAG20107_inplace(K_coords_cache(i), coords, qvblock_c, indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
    }
  }
  
  //end = std::chrono::steady_clock::now();
  //timings(0) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();

  //data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  
#pragma omp parallel for
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    //Rcpp::Rcout << block_ct_obs(u) << " " << compute_block(predicting, block_ct_obs(u), false) << endl;
    if(block_ct_obs(u) > 0){//compute_block(predicting, block_ct_obs(u), false)){ //***********
      int u_cached_ix = coords_caching_ix(u);
      
      //start = std::chrono::steady_clock::now();
      arma::uvec cx = arma::find( coords_caching == u_cached_ix );
      arma::mat Kcc = K_coords_cache(cx(0));
      mvCovAG20107_inplace(Kcc, coords, qvblock_c, indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
      //end = std::chrono::steady_clock::now();
      //timings(1) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //start = std::chrono::steady_clock::now();
      // +++++++++++++++++
      arma::mat Kxx = arma::zeros(parents_indexing(u).n_elem, parents_indexing(u).n_elem);
      //xCovHUV_inplace(Kxx, coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
      //end = std::chrono::steady_clock::now();
      //timings(2) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //start = std::chrono::steady_clock::now();
      mvCovAG20107_inplace(Kxx, coords, qvblock_c, parents_indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
      //end = std::chrono::steady_clock::now();
      //timings(3) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //Rcpp::Rcout << arma::size(Kxx) << endl;
      
      //start = std::chrono::steady_clock::now();
      arma::mat Kxxi_c = arma::inv(arma::trimatl(arma::chol(Kxx, "lower")));
      
      //end = std::chrono::steady_clock::now();
      //timings(4) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      /*double nanxi = arma::accu(Kxxi_c);
      if(std::isnan(nanxi)){
        //Rcpp::Rcout << data.sigmasq << endl;
        Rcpp::Rcout << "Error in inv tri chol sym(Kxx) at " << u << endl;
        data.track_chol_fails(u) = 1;
      }*/
      // +++++++++++++++++
      //start = std::chrono::steady_clock::now();
      //xCovHUV_inplace(Kcp_cache(i), coords, indexing(u), parents_indexing(u), cparams, Dmat);
      mvCovAG20107_inplace(Kcp_cache(i), coords, qvblock_c, indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, false);
      //end = std::chrono::steady_clock::now();
      //timings(5) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //start = std::chrono::steady_clock::now();
      // +++++++++++++++++
      arma::mat Kcx_Kxxic = Kcp_cache(i) * Kxxi_c.t();
      w_cond_mean_cache(i) = Kcx_Kxxic * Kxxi_c;
      // +++++++++++++++++
      //end = std::chrono::steady_clock::now();
      //timings(6) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
      //start = std::chrono::steady_clock::now();
      //try {
      arma::mat Kinside = Kcc - Kcx_Kxxic*Kcx_Kxxic.t();
      try {
        K_cholcp_cache(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu(
          Kinside
        ) , "lower")));
      } catch (...) {
        Rcpp::Rcout << Kcc << endl;
        Rcpp::Rcout << Kcx_Kxxic << endl;
        Rcpp::Rcout << Kinside << endl;
        Rcpp::Rcout << coords.rows(indexing(u)) << endl;
        Rcpp::Rcout << coords.rows(parents_indexing(u)) << endl;
        Rcpp::Rcout << qvblock_c.rows(indexing(u)) << endl;
        Rcpp::Rcout << qvblock_c.rows(parents_indexing(u)) << endl;
        Rcpp::Rcout << ai1 << " " << ai2 << " " << phi_i << " " << thetamv << " " << Dmat << endl;
        Rcpp::Rcout << "Kinside error. " << endl; 
        throw 1;
      }
      
      //end = std::chrono::steady_clock::now();
      //timings(7) += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
      
       /* if(!Kinside.is_symmetric()){
          data.track_chol_fails(u) = 2;
          Rcpp::Rcout << "Error - Kinside not symmetric for some reason " << endl;
        }
      } catch (...) {
        data.track_chol_fails(u) = 3;
        K_cholcp_cache(i) = arma::eye(arma::size(Kcc));
        Rcpp::Rcout << "Error in inv chol symmatu (Kcc - Kcx Kxx Kxc) at " << u << endl;
        Rcpp::Rcout << Kcc << endl;
        throw 1;
      }*/
      //Rcpp::Rcout << "krig: " << arma::size(K_cholcp_cache(i)) << "\n";
    }
  }

  
  //if(arma::all(data.track_chol_fails == 0)){
  //  data.cholfail = false;
  //start = std::chrono::steady_clock::now();
  #pragma omp parallel for // **
    for(int i = 0; i<n_ref_blocks; i++){
      int r = reference_blocks(i);
      int u = block_names(r)-1;
      //for(int i=0; i<n_blocks; i++){
        //int u=block_names(i)-1;
      //if(compute_block(predicting, block_ct_obs(u), false)){
        int u_cached_ix = coords_caching_ix(u);
        arma::uvec cx = arma::find( coords_caching == u_cached_ix, 1, "first" );
        arma::mat Kcc = K_coords_cache(cx(0));
        
        arma::vec w_x = w.rows(indexing(u));
        arma::mat cond_mean_K, cond_mean, cond_cholprec;//, w_parents;
        
        if( parents(u).n_elem > 0 ){
          // ---------------
          //int p_cached_ix = parents_caching_ix(u);
          //arma::uvec px = arma::find( parents_caching == p_cached_ix );
          //arma::mat Kxxi = K_parents_cache(px(0)); //arma::inv_sympd(  Kpp(parents_coords(u), parents_coords(u), theta, true) );
          
          int kr_cached_ix = kr_caching_ix(u);
          arma::uvec cpx = arma::find(kr_caching == kr_cached_ix, 1, "first" );
          
          //arma::mat Kcx = Kcp_cache(cpx(0));//Kpp(coords_blocks(u), parents_coords(u), theta);
          cond_mean_K = w_cond_mean_cache(cpx(0));// Kcx * Kxxi; // +++++++++++
          cond_cholprec = K_cholcp_cache(cpx(0));//arma::inv(arma::trimatl(arma::chol( arma::symmatu(Kcc - cond_mean_K * Kcx.t()) , "lower")));
          w_x = w_x - cond_mean_K * w.rows(parents_indexing(u));
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
          
          //w_x = w_x % na_1_blocks(u);
          data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
          //Rcpp::Rcout << "u: " << u << " wcore: " << data.wcore(u) << endl; 
          data.loglik_w_comps(u) = //block_ct_obs(u)//
              (indexing(u).n_elem+.0) 
            * hl2pi -.5 * data.wcore(u);
          /*
        } else {
          Rcpp::Rcout << "should not read this " << endl;
          data.logdetCi_comps(u) = 0;
          data.wcore(u) = 0;
          data.loglik_w_comps(u) = 0;
          //Rcpp::Rcout << "i: " << i << ", u: " << u << " ldCi " << logdetCi_comps(u) << endl;
        }*/
      /*} else {
        data.logdetCi_comps(u) = 0;
        data.wcore(u) = 0;
        data.loglik_w_comps(u) = 0;
      }*/
    }
    data.logdetCi = arma::accu(data.logdetCi_comps);
    data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps);
    //end = std::chrono::steady_clock::now();
    //timings(8) = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    
    //Rcpp::Rcout << timings.t() << endl;
    
   /* 
  } else {
    data.cholfail = true;
    Rcpp::Rcout << "Failure: " << endl;
    arma::uvec fail1 = arma::find(data.track_chol_fails == 1);
    Rcpp::Rcout << fail1.t() << endl;
    Rcpp::Rcout << "^ ^ some u for which there has been a failure of type 1 (inverting chol C_[j][j]) ^ ^" << endl;
    
    arma::uvec fail2 = arma::find(data.track_chol_fails == 2);
    Rcpp::Rcout << fail2.t() << endl;
    Rcpp::Rcout << "^ ^ some u for which there has been a failure of type 2 (Rj not symmetric) ^ ^" << endl;
    
    arma::uvec fail3 = arma::find(data.track_chol_fails == 3);
    Rcpp::Rcout << fail3.t() << endl;
    Rcpp::Rcout << "^ ^ some u for which there has been a failure of type 3 (inverting chol Rj) ^ ^" << endl;
    
    //Rcpp::Rcout << cparams.t() << endl << Dmat << endl;
    throw 1;
  }*/
  
  if(verbose){
    
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_cached_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}


void MeshGPmv::get_cond_comps_loglik_w_nocache(MeshDataMV& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w_nocache] start. ");
  
  theta_transform(data);

#pragma omp parallel for // **
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    //if(compute_block(predicting, block_ct_obs(u), false) ){
      if(block_ct_obs(u) > 0){
        // skip calculating stuff for blocks in the predict-only area 
        // when it's not prediction time
        arma::mat Kcc = //xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
         mvCovAG20107(coords, qvblock_c, indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
        arma::mat cond_mean_K, cond_mean, cond_cholprec;
        
        if( parents(u).n_elem > 0 ){
          arma::mat Kxxi = arma::inv_sympd(  //xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true) );
            mvCovAG20107(coords, qvblock_c, parents_indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true) );
          arma::mat Kcx = mvCovAG20107(coords, qvblock_c, indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, false);//xCovHUV(coords, indexing(u), parents_indexing(u), cparams, Dmat);
          cond_mean_K = Kcx * Kxxi;
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( arma::symmatu(Kcc - cond_mean_K * Kcx.t()) , "lower")));
        } else {
          cond_mean_K = arma::zeros(0, 0);
          cond_cholprec = arma::inv(arma::trimatl(arma::chol( Kcc , "lower")));
        }
        
        data.w_cond_mean_K(u) = cond_mean_K;
        data.w_cond_cholprec(u) = cond_cholprec;
        data.w_cond_prec(u) = cond_cholprec.t() * cond_cholprec;
      
        //message("[get_cond_comps_loglik_w_nocache] get lik");
        
        arma::vec w_x = w.rows(indexing(u));
        if(parents(u).n_elem > 0){
          w_x -= data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
        }
        
        arma::vec ccholprecdiag = cond_cholprec.diag();
        data.logdetCi_comps(u) = arma::accu(log(ccholprecdiag));
        
        //w_x = w_x % na_1_blocks(u);
        data.wcore(u) = arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
        data.loglik_w_comps(u) = (indexing(u).n_elem+.0) 
          * hl2pi -.5 * data.wcore(u);
      } else {
        Rcpp::Rcout << "should not read this 2 " << endl;
      }
    //} else {
    //  data.wcore(u) = 0;
    //  data.loglik_w_comps(u) = 0;
    //  data.logdetCi_comps(u) = 0;
    //}
  }
  
  data.logdetCi = arma::accu(data.logdetCi_comps);
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps);
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_loglik_w_nocache] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void MeshGPmv::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  Rcpp::RNGScope scope;
  arma::mat bmat = arma::randn(p, q);
  
  for(int j=0; j<q; j++){
    arma::mat Si_chol = arma::chol(arma::symmatu(tausq_inv(j) * XtX(j) + Vi), "lower");
    arma::mat Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
    arma::vec w_available = w.rows(na_ix_all);
    arma::mat Xprecy_j = Vim + tausq_inv(j) * X_available.rows(ix_by_q_a(j)).t() * 
      (y_available.rows(ix_by_q_a(j)) - w_available.rows(ix_by_q_a(j)));
    
    
    Bcoeff.col(j) = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy_j + bmat.col(j));
    //Rcpp::Rcout << "j: " << j << endl
    //     << Bmu << endl;
    
    XB.rows(ix_by_q(j)) = X.rows(ix_by_q(j)) * Bcoeff.col(j);
  }
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void MeshGPmv::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  for(int j=0; j<q; j++){
    arma::vec Zw_availab = w.rows(na_ix_all);
    arma::vec XB_availab = XB.rows(na_ix_all);
    arma::mat yrr = y_available.rows(ix_by_q_a(j)) - XB_availab.rows(ix_by_q_a(j)) - Zw_availab.rows(ix_by_q_a(j));
    
    //Rcpp::Rcout << arma::join_horiz( arma::join_horiz(y_available.subvec(0, 10), XB_availab.subvec(0, 10)), Zw_availab.subvec(0, 10)) << endl;
    
    double bcore = arma::conv_to<double>::from( yrr.t() * yrr );
    
    double aparam = 2.001 + ix_by_q_a(j).n_elem/2.0;
    double bparam = 1.0/( 1.0 + .5 * bcore );
    
    Rcpp::RNGScope scope;
    tausq_inv(j) = R::rgamma(aparam, bparam);
    // fill all (not just available) corresponding to same variable.
    tausq_inv_long.rows(ix_by_q(j)).fill(tausq_inv(j));
    
    if(verbose){
      end = std::chrono::steady_clock::now();
      Rcpp::Rcout << "[gibbs_sample_tausq] " << j << ", "
                  << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() << "us. " //<< " ... "
                  << aparam << " : " << bparam << " " << bcore << " --> " << 1.0/tausq_inv(j)
                  << endl;
    }
  }
}


void MeshGPmv::gibbs_sample_sigmasq(){
  start = std::chrono::steady_clock::now();
  
  double oldsigmasq = sigmasq;
  double aparam = 2.01 + n_loc_ne_blocks/2.0; //
  double bparam = 1.0/( 1.0 + .5 * oldsigmasq * arma::accu( param_data.wcore ));
  
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
  // int u = block_names(i)-1;
  // if(block_ct_obs(u)>0){
  #pragma omp parallel for
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    param_data.logdetCi_comps(u) += //block_ct_obs(u)//
      (indexing(u).n_elem + 0.0) 
      * 0.5*log(old_new_ratio);
    
    param_data.w_cond_prec(u) = old_new_ratio * param_data.w_cond_prec(u);
    param_data.w_cond_cholprec(u) = sqrt(old_new_ratio) * param_data.w_cond_cholprec(u);
    param_data.wcore(u) = old_new_ratio * param_data.wcore(u);//arma::conv_to<double>::from(w_x.t() * data.w_cond_prec(u) * w_x);
    param_data.loglik_w_comps(u) = //block_ct_obs(u)//
      (indexing(u).n_elem + .0) 
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


void MeshGPmv::gibbs_sample_w(bool needs_update=true){
  if(cached_gibbs){
    gibbs_sample_w_omp(needs_update);
  } else {
    gibbs_sample_w_omp_nocache(needs_update);
  }
}

void MeshGPmv::gibbs_sample_w_omp(bool needs_update=true){
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows);
  
  if(needs_update){
  #pragma omp parallel for
    for(int i=0; i<gibbs_caching.n_elem; i++){
      int u = gibbs_caching(i);
      //Rcpp::Rcout << "gibbs_sample_w_omp caching step - block " << u << "\n";
      if(block_ct_obs(u) > 0){
        arma::mat Sigi_tot = param_data.w_cond_prec(u);
        Sigi_tot.diag() += tausq_inv_long.rows(indexing(u)) % na_1_blocks(u);
        
        for(int c=0; c<children(u).n_elem; c++){
          int child = children(u)(c);
          arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));//u_is_which_col);
          arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
          Sigi_tot += AK_uP * AK_u;
        }
        
        param_data.Sigi_chol_cached(i) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
      }
    }
  }
  
  
  for(int g=0; g<n_gibbs_groups-predict_group_exists; g++){
    //int g = gibbs_groups_reorder(go);
#pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      
      //if(compute_block(predicting, block_ct_obs(u), false)){
      //if((block_ct_obs(u) > 0)){
      arma::mat Smu_tot = arma::zeros(indexing(u).n_elem, 1);
      if(parents(u).n_elem>0){
        Smu_tot += param_data.w_cond_prec(u) * param_data.w_cond_mean_K(u) * w.rows( parents_indexing(u) );//param_data.w_cond_mean(u);
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
        
        arma::vec w_child = w.rows(indexing(child));
        arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
        arma::vec w_par_child_select = w_parents_of_child.rows(u_is_which_col_f(u)(c)(1));
        
        Smu_tot += AK_uP * ( w_child - AK_others * w_par_child_select );
      }
      
      Smu_tot += //Zblock(u).t() * 
        ((tausq_inv_long.rows(indexing(u)) % na_1_blocks(u)) % 
        ( y.rows(indexing(u)) - XB.rows(indexing(u)) ));
      
      int u_cached_ix = gibbs_caching_ix(u);
      arma::uvec gx = arma::find( gibbs_caching == u_cached_ix );
      
      //end = std::chrono::steady_clock::now();
      
      // sample
      //arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
      arma::vec rnvec = rand_norm_mat.rows(indexing(u));
      w.rows(indexing(u)) = param_data.Sigi_chol_cached(gx(0)).t() * 
        (param_data.Sigi_chol_cached(gx(0)) * Smu_tot + rnvec); 
      
      //}
      
      //} 
    }
  }
  
  
}

void MeshGPmv::gibbs_sample_w_omp_nocache(bool needs_update=true){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] " << endl;
  }
  
  // keep seed
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] sampling big stdn matrix size " << n << "," << q << endl;
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << endl << " ? " << endl;
  start_overall = std::chrono::steady_clock::now();
  //arma::uvec gibbs_groups_reorder = ashuffle(arma::regspace<arma::uvec>(0, n_gibbs_groups-1));
  //needs_update = true; 
  
  for(int g=0; g<n_gibbs_groups-predict_group_exists; g++){
    //int g = gibbs_groups_reorder(go);
    #pragma omp parallel for
    for(int i=0; i<u_by_block_groups(g).n_elem; i++){
      int u = u_by_block_groups(g)(i);
      //if(compute_block(predicting, block_ct_obs(u), false)){
        if((block_ct_obs(u) > 0)){
          arma::mat Smu_tot, Sigi_tot, Sigi_diag_add;
          
          Smu_tot = arma::zeros(indexing(u).n_elem, 1);
          if(needs_update){
            Sigi_tot = param_data.w_cond_prec(u);
            Sigi_diag_add = tausq_inv_long.rows(indexing(u)) % na_1_blocks(u);
            Sigi_tot.diag() += Sigi_diag_add;
          }
          
          if(parents(u).n_elem>0){
            Smu_tot += param_data.w_cond_prec(u) * param_data.w_cond_mean_K(u) * w.rows( parents_indexing(u) );//param_data.w_cond_mean(u);
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
            
            arma::vec w_child = w.rows(indexing(child));
            arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
            arma::vec w_par_child_select = w_parents_of_child.rows(u_is_which_col_f(u)(c)(1));
            
            if(needs_update){
              Sigi_tot += AK_uP * AK_u;
            }
            
            Smu_tot += AK_uP * ( w_child - AK_others * w_par_child_select );
          }
          
          arma::vec Smu_y_add = ((tausq_inv_long.rows(indexing(u)) % na_1_blocks(u)) % 
            ( y.rows(indexing(u)) - XB.rows(indexing(u)) ));
//          Rcpp::Rcout << arma::join_horiz(Smu_y_add, Sigi_diag_add) << endl
//                      << "-- "<< endl;
          Smu_tot += //Zblock(u).t() * 
            Smu_y_add;
          
          //start = std::chrono::steady_clock::now();
          if(needs_update){
            param_data.Sigi_chol(u) = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
          } 
          //end = std::chrono::steady_clock::now();

          //if (u == 1){
          //  Rcpp::Rcout << param_data.Sigi_chol(u) << endl;
          //}
          
          // sample
          //arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
          arma::vec rnvec = rand_norm_mat.rows(indexing(u));
          w.rows(indexing(u)) = param_data.Sigi_chol(u).t() * (param_data.Sigi_chol(u) * Smu_tot + rnvec); 
          
        }
        
      //} 
    }
  }
  
  //debug_stuff = Rcpp::List::create(
  //  Rcpp::Named("rand_norm_tracker") = rand_norm_tracker,
  //  Rcpp::Named("rand_child_tracker") = rand_child_tracker,
  //  Rcpp::Named("rand_norm_mat") = rand_norm_mat
  //);
  
  //Zw = armarowsum(Z % w);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
  
}

void MeshGPmv::predict(bool needs_update=true){
  if(predict_group_exists == 1){
  #pragma omp parallel for
    for(int i=0; i<u_predicts.n_elem; i++){
      int u = u_predicts(i);
      
    //Rcpp::Rcout << u << endl; 
    //Rcpp::Rcout << "right? "<< block_ct_obs(u) << endl;
    // only predictions at this block. 
    // sample from conditional MVN 
      //**************
      if(true){
        // skip calculating stuff for blocks in the predict-only area 
        // when it's not prediction time
        if(needs_update){
          arma::mat Kcc = mvCovAG20107(coords, qvblock_c, indexing(u), indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true);
          
          arma::mat Kxxi = arma::inv_sympd(  //xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true) );
            mvCovAG20107(coords, qvblock_c, parents_indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true) );
          arma::mat Kcx = mvCovAG20107(coords, qvblock_c, indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, false);//xCovHUV(coords, indexing(u), parents_indexing(u), cparams, Dmat);
          
          //Rcpp::Rcout << "? 1 " << endl;
          param_data.w_cond_mean_K(u) = Kcx * Kxxi;
          param_data.w_cond_cholprec(u) = arma::flipud(arma::fliplr(arma::chol( arma::symmatu(Kcc - param_data.w_cond_mean_K(u) * Kcx.t()) , "lower")));
        }
        //***********
        
        //Rcpp::Rcout << "? 2 " << endl;
        arma::vec phimean = param_data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
        
        //Rcpp::Rcout << "? 3 " << endl;
        arma::mat cholK = param_data.w_cond_cholprec(u);
        
        //Rcpp::Rcout << "? 4 " << endl;
        //arma::vec normvec = arma::randn(indexing(u).n_elem);
        arma::vec rnvec = rand_norm_mat.rows(indexing(u));
        w.rows(indexing(u)) = phimean + cholK * rnvec;
      } else {
        arma::mat Kxxi, Kcx;
        
        if(needs_update){
          Kxxi = arma::inv_sympd(  //xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true) );
            mvCovAG20107(coords, qvblock_c, parents_indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, true) );
          Kcx = mvCovAG20107(coords, qvblock_c, indexing(u), parents_indexing(u), ai1, ai2, phi_i, thetamv, Dmat, false);//xCovHUV(coords, indexing(u), parents_indexing(u), cparams, Dmat);
          param_data.w_cond_mean_K(u) = Kcx * Kxxi;
        }
        
        //param_data.w_cond_cholprec(u) = arma::zeros(indexing(u).n_elem, indexing(u).n_elem);
        arma::vec phimean = param_data.w_cond_mean_K(u) * w.rows(parents_indexing(u));
        
        for(int cx=0; cx<indexing(u).n_elem; cx++){
          if(needs_update){
            arma::uvec cuv = indexing(u)(cx) * arma::ones<arma::uvec>(1);
            arma::mat Kcc = mvCovAG20107(coords, qvblock_c, cuv, cuv, ai1, ai2, phi_i, thetamv, Dmat, true);
            //Rcpp::Rcout << arma::size(param_data.w_cond_mean_K(u)) << " " << arma::endl;
            arma::mat Rcoord = Kcc - param_data.w_cond_mean_K(u).row(cx) * arma::trans(Kcx.row(cx));
            param_data.w_cond_cholprec(u)(cx, cx) = sqrt(Rcoord(0,0));
          }
          
          arma::vec normvec = arma::randn(1);
          
          w.row(indexing(u)(cx)) = phimean(cx) + param_data.w_cond_cholprec(u)(cx, cx) * normvec;
        }
        
      }
    }
  
  }
}

void MeshGPmv::theta_update(MeshDataMV& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void MeshGPmv::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void MeshGPmv::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void MeshGPmv::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}