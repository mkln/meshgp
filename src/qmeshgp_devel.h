#include "includes.h"
 
// with indexing
// without block extensions (obs with NA are left in)

using namespace std;

// everything that changes during MCMC
struct MeshDataDev {
  
  double sigmasq;
  arma::vec theta; 
  
  arma::vec wcore; 
  arma::field<arma::mat> w_cond_mean_K;
  arma::field<arma::mat> w_cond_cholprec;
  arma::field<arma::mat> w_cond_prec;
  
  arma::field<arma::mat> zH_obs; //***
  //arma::field<arma::mat> zHHz_obs; //***
  arma::field<arma::vec> Ddiag; //***
  //arma::field<arma::mat> ZK; //***
  arma::field<arma::mat> KcxKxxi_obs; // ***
  double ll_y_all;
  
  arma::vec logdetCi_comps;
  double logdetCi;
  
  arma::vec loglik_w_comps;
  double loglik_w;
  
  arma::uvec track_chol_fails;
  bool cholfail;
  
};

const double hl2pi = -.5 * log(2 * M_PI);

class MeshGPdev {
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
  arma::field<arma::uvec> indexing_obs; // which are obs
  
  // NA data
  arma::field<arma::vec> na_1_blocks; // indicator vector by block
  arma::field<arma::uvec> na_ix_blocks;
  arma::uvec na_ix_all;
  int n_loc_ne_blocks;
  
  // regression
  arma::mat XtX;
  arma::mat Sigma_chol_Bcoeff;
  
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
  arma::uvec predictable_blocks; //***
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
  arma::vec Di_obs;
  
  arma::mat rand_norm_mat;
  
  // params with mh step
  MeshDataDev param_data; 
  MeshDataDev alter_data;
  
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

  // caching some matrices 
  arma::field<arma::mat> K_coords_cache;
  arma::field<arma::mat> Kcp_cache;
  arma::field<arma::mat> Kxxi_cache; //***
  
  // MCMC
  void get_loglik_w(MeshDataDev& data);
  void get_loglik_comps_w(MeshDataDev& data);
  void get_cond_comps_loglik_w(MeshDataDev& data);
  
  void gibbs_sample_w();
  void gibbs_sample_w_omp();
  void gibbs_sample_w_omp_nocache();
  void predict();
  
  void gibbs_sample_beta();
  void gibbs_sample_tausq();
  void update_ll_after_beta_tausq(); //***
  
  // changing the values, no sampling
  void tausq_update(double);
  arma::vec cparams;
  arma::mat Dmat;
  void theta_transform(const MeshDataDev&);
  void theta_update(MeshDataDev&, const arma::vec&);
  void beta_update(const arma::vec&);
  
  // avoid expensive copies
  void accept_make_change();
  
  std::chrono::steady_clock::time_point start_overall;
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  std::chrono::steady_clock::time_point end_overall;
  
  // empty
  MeshGPdev();
  
  // build everything
  MeshGPdev(
    const arma::mat& y_in, 
    const arma::mat& X_in, 
    const arma::mat& Z_in,
    const arma::mat& coords_in, 
    const arma::uvec& blocking_in,
    
    const arma::field<arma::uvec>& parents_in,
    const arma::field<arma::uvec>& children_in,
    
    const arma::vec& layers_names,
    const arma::vec& block_group_in,
    const arma::uvec& predictable_blocks_in,
    
    const arma::field<arma::uvec>& indexing_in,
    const arma::field<arma::uvec>& indexing_obs_in,
    
    const arma::mat& w_in,
    const arma::vec& beta_in,
    const arma::vec& theta_in,
    double tausq_inv_in,
    double sigmasq_in,
    
    const arma::mat& beta_Vi_in,
    const arma::vec& sigmasq_ab_in,
    const arma::vec& tausq_ab_in,
    
    bool use_cache,
    bool use_cache_gibbs,
    bool use_rfc,
    bool v,
    bool debugging);
  
};

void MeshGPdev::message(string s){
  if(verbose & debug){
    Rcpp::Rcout << s << "\n";
  }
}

MeshGPdev::MeshGPdev(){
  
}

MeshGPdev::MeshGPdev(
  const arma::mat& y_in, 
  const arma::mat& X_in, 
  const arma::mat& Z_in,
  
  const arma::mat& coords_in, 
  const arma::uvec& blocking_in,
  
  const arma::field<arma::uvec>& parents_in,
  const arma::field<arma::uvec>& children_in,
  
  const arma::vec& block_names_in,
  const arma::vec& block_groups_in,
  const arma::uvec& predictable_blocks_in,
  
  const arma::field<arma::uvec>& indexing_in,
  const arma::field<arma::uvec>& indexing_obs_in,
  
  const arma::mat& w_in,
  const arma::vec& beta_in,
  const arma::vec& theta_in,
  double tausq_inv_in,
  double sigmasq_in,
  
  const arma::mat& beta_Vi_in,
  const arma::vec& sigmasq_ab_in,
  const arma::vec& tausq_ab_in,
  
  bool use_cache=true,
  bool use_cache_gibbs=false,
  bool use_rfc=false,
  bool v=false,
  bool debugging=false){
  
  printf("~ MeshGPdev initialization.\n");
  
  start_overall = std::chrono::steady_clock::now();
  
  cached = true;
  cached_gibbs = false;
  
  verbose = v;
  debug = debugging;
  
  message("MeshGPdev::MeshGPdev assign values.");
  
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
  predictable_blocks = predictable_blocks_in; //***
  Rcpp::Rcout << n_gibbs_groups << " groups for gibbs " << endl;
  //Rcpp::Rcout << block_groups_labels << endl;
  
  na_ix_all   = arma::find_finite(y.col(0));
  y_available = y.rows(na_ix_all);
  X_available = X.rows(na_ix_all); 
  Z_available = Z.rows(na_ix_all);
  
  indexing    = indexing_in;
  indexing_obs = indexing_obs_in;
  
  n  = na_ix_all.n_elem;
  p  = X.n_cols;
  q  = Z.n_cols;
  dd = coords.n_cols;
  
  if(dd == 2){
    if(q > 2){
      npars = 1+3;
    } else {
      npars = 1+1;
    }
  } else {
    if(q > 2){
      npars = 1+5;
    } else {
      npars = 1+3; // sigmasq + alpha + beta + phi
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
  
  predicting = true;
  rfc_dep    = use_rfc;
  
  
  // now elaborate
  message("MeshGPdev::MeshGPdev : init_indexing()");
  init_indexing();
  
  message("MeshGPdev::MeshGPdev : na_study()");
  na_study();
  y.elem(arma::find_nonfinite(y)).fill(0);
  
  message("MeshGPdev::MeshGPdev : init_meshdata()");
  init_meshdata(sigmasq_in, theta_in);
  sigmasq = sigmasq_in;
  
  message("MeshGPdev::MeshGPdev : init_finalize()");
  init_finalize();
  
  // prior for beta
  XtX   = X_available.t() * X_available;
  Vi    = beta_Vi_in;//.00001 * arma::eye(p,p);
  bprim = arma::zeros(p);
  Vim   = Vi * bprim;
  
  arma::mat Si_chol = arma::chol(arma::symmatu(//tausq_inv * 
    XtX + Vi), "lower"); 
  Sigma_chol_Bcoeff = arma::inv(arma::trimatl(Si_chol));
  
  
  // priors for tausq and sigmasq
  sigmasq_ab = sigmasq_ab_in;
  tausq_ab = tausq_ab_in;
  
  message("MeshGPdev::MeshGPdev : make_gibbs_groups()");
  make_gibbs_groups();
  
  //caching;
  //if(cached){
  
    message("MeshGPdev::MeshGPdev : init_cache()");
    init_cache();
    fill_zeros_Kcache();
  //}
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "MeshGPdev::MeshGPdev initializing took "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us.\n";
  }
  
}


void MeshGPdev::make_gibbs_groups(){
  message("[make_gibbs_groups]");
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
    u_predicts = 999999 + arma::zeros<arma::uvec>(n_blocks - n_ref_blocks);
    predict_group_exists = 1;
  } else {
    predict_group_exists = 0;
  }
  
  Rcpp::Rcout << "predict group? " << predict_group_exists << endl;
  
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
      if((block_ct_obs(u) == 0) & (predictable_blocks(u) == 1)){
        u_predicts(p) = u;
        //Rcpp::Rcout << u << endl;
        p ++;
      }
    }
  }
  
  u_predicts = u_predicts.elem(arma::find(u_predicts != 999999));
  
  message("[make_gibbs_groups] done.");

  //u_by_block_groups = u_by_block_groups_temp;
}

void MeshGPdev::na_study(){
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
    // ***
    arma::vec yvec = y.rows(indexing_obs(i));//y_blocks(i);
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

void MeshGPdev::fill_zeros_Kcache(){
  K_coords_cache = arma::field<arma::mat> (coords_caching.n_elem);
  Kxxi_cache = arma::field<arma::mat> (coords_caching.n_elem); //***
  Kcp_cache = arma::field<arma::mat> (kr_caching.n_elem);
  //K_parents_cache_noinv = arma::field<arma::mat> (parents_caching.n_elem);
  
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i);
    K_coords_cache(i) = arma::zeros(q*indexing(u).n_elem, q*indexing(u).n_elem);
    Kxxi_cache(i) = arma::zeros(q*indexing(u).n_elem, q*indexing(u).n_elem); //***
  }
  
  
  for(int i=0; i<kr_caching.n_elem; i++){
    int u = kr_caching(i);
    Kcp_cache(i) = arma::zeros(q*indexing(u).n_elem, q*parents_indexing(u).n_elem);
  }
}

void MeshGPdev::init_cache(){
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
  kr_caching_ix = caching_pairwise_compare_uc(kr_pairing, block_names, block_ct_obs); //***
  //kr_caching_ix = caching_pairwise_compare_u(kr_pairing, block_names);
  kr_caching = arma::unique(kr_caching_ix);
  
  Rcpp::Rcout << "." << endl;
  
  double c_cache_perc = 100- 100*(coords_caching.n_elem+0.0) / n_blocks;
  double p_cache_perc = 100- 100*(parents_caching.n_elem+0.0) / n_blocks;
  double k_cache_perc = 100- 100*(kr_caching.n_elem+0.0) / n_blocks;
  double g_cache_perc = 0;
  
  
  printf("~ Caching stats c: %d [%.2f%%] / p: %d [%.2f%%] / k: %d [%.2f%%] \n",
         coords_caching.n_elem, c_cache_perc, 
         parents_caching.n_elem, p_cache_perc, 
         kr_caching.n_elem, k_cache_perc);
  
}

void MeshGPdev::init_meshdata(double sigmasq_in, const arma::vec& theta_in){
  // block params
  param_data.wcore         = arma::zeros(n_blocks);
  param_data.w_cond_mean_K = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_prec   = arma::field<arma::mat> (n_blocks);
  param_data.w_cond_cholprec = arma::field<arma::mat> (n_blocks);
  
  param_data.zH_obs = arma::field<arma::mat> (n_blocks); //***
  //param_data.zHHz_obs = arma::field<arma::mat> (n_blocks); //***
  param_data.Ddiag = arma::field<arma::vec> (n_blocks);//***
  //param_data.ZK = arma::field<arma::mat> (n_blocks);//***
  param_data.KcxKxxi_obs = arma::field<arma::mat> (n_blocks); //***
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){
    param_data.w_cond_mean_K(i) = arma::zeros(q*indexing(i).n_elem, q*parents_indexing(i).n_elem);
    param_data.w_cond_cholprec(i) = arma::zeros(q*indexing(i).n_elem, q*indexing(i).n_elem);
    param_data.KcxKxxi_obs(i) = arma::zeros(q*indexing_obs(i).n_elem, q*indexing(i).n_elem);
    param_data.zH_obs(i) = arma::zeros(q*indexing_obs(i).n_elem, q*indexing(i).n_elem); //***
    //param_data.zHHz_obs(i) = arma::zeros(q*indexing_obs(i).n_elem, q*indexing_obs(i).n_elem); //***
    param_data.Ddiag(i) = arma::zeros(q*indexing_obs(i).n_elem); //***
    //param_data.ZK(i) = arma::mat(q*indexing_obs(i).n_elem, q*indexing(i).n_elem); //***
  }
  
  // loglik w for updating theta
  param_data.logdetCi_comps = arma::zeros(n_blocks);
  param_data.logdetCi       = 0;
  param_data.loglik_w_comps = arma::zeros(n_blocks);
  param_data.loglik_w       = 0;
  param_data.ll_y_all       = 0;
  param_data.theta          = theta_in; //arma::join_vert(arma::ones(1) * sigmasq_in, theta_in);
  param_data.cholfail       = false;
  param_data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  param_data.sigmasq          = sigmasq_in;
  alter_data                = param_data; 
}

void MeshGPdev::init_indexing(){
  
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
  }
  
}

void MeshGPdev::init_finalize(){
  
  message("[init_finalize] dim_by_parent, parents_coords, children_coords");
  
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<n_blocks; i++){ // all blocks
    int u = block_names(i)-1; // layer name
    //if(coords_blocks(u).n_elem > 0){
    if(indexing(u).n_elem > 0){
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
    
    //***
    if(indexing_obs(u).n_elem > 0){
      //Rcpp::Rcout << "Ib " << parents(u).n_elem << endl;
      //Ib(u) = arma::eye<arma::sp_mat>(coords_blocks(u).n_rows, coords_blocks(u).n_rows);
      Ib(u) = arma::eye<arma::sp_mat>(indexing_obs(u).n_elem, indexing_obs(u).n_elem);
      for(int j=0; j<Ib(u).n_cols; j++){
        if(na_1_blocks(u)(j) == 0){
          Ib(u)(j,j) = 0;//1e-6;
        }
      }
      Zblock(u) = Zify( Z.rows(indexing_obs(u)) ); //***
    }
  }
}

void MeshGPdev::get_loglik_w(MeshDataDev& data){
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
  data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps) + data.ll_y_all;
  
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
  
  //print_data(data);
}

void MeshGPdev::get_loglik_comps_w(MeshDataDev& data){
  get_cond_comps_loglik_w(data);
}


void MeshGPdev::theta_transform(const MeshDataDev& data){
  //arma::vec Kparam = data.theta; 
  int k = data.theta.n_elem - npars; // number of cross-distances = p(p-1)/2
  
  //Rcpp::Rcout << data.theta << endl
    //         << "npars " << npars << endl;
  
  cparams = data.theta;//arma::join_vert(arma::ones(1) * sigmasq, data.theta.subvec(0, npars - 1));
  
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

void MeshGPdev::get_cond_comps_loglik_w(MeshDataDev& data){
  start = std::chrono::steady_clock::now();
  message("[get_cond_comps_loglik_w] start.");
  
  //arma::field<arma::mat> K_coords_cache(coords_caching.n_elem);
  //arma::field<arma::mat> K_parents_cache(parents_caching.n_elem);
  arma::field<arma::mat> K_cholcp_cache(kr_caching.n_elem);
  arma::field<arma::mat> w_cond_mean_cache(kr_caching.n_elem); // +++++++++
  //arma::field<arma::mat> Kcp_cache(kr_caching.n_elem);
  
  theta_transform(data);
  
  message("[get_cond_comps_loglik_w] step 1 - coords.");
#ifdef _OPENMP
#pragma omp parallel for 
#endif
  for(int i=0; i<coords_caching.n_elem; i++){
    int u = coords_caching(i); // layer name of ith representative
    //bool calc_this_block = predicting == false? (block_ct_obs(u) > 0) : predicting;
    //if(calc_this_block){
    if(compute_block(predicting, block_ct_obs(u), rfc_dep)){
      //uhm ++;
      //K_coords_cache(i) = Kpp(coords_blocks(u), coords_blocks(u), Kparam, true);
      xCovHUV_inplace(K_coords_cache(i), coords, indexing(u), indexing(u), cparams, Dmat, true);
      Kxxi_cache(i) = arma::inv_sympd( K_coords_cache(i) ); //***
    }
  }

  data.track_chol_fails = arma::zeros<arma::uvec>(n_blocks);
  
  
  message("[get_cond_comps_loglik_w] step 2 - kr.");
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
      xCovHUV_inplace(Kxx, coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true);
      arma::mat Kxxi_c = arma::inv(arma::trimatl(arma::chol(arma::symmatu( Kxx ), "lower")));
      /*
      double nanxi = arma::accu(Kxxi_c);
      if(std::isnan(nanxi)){
        Rcpp::Rcout << data.sigmasq << endl;
        Rcpp::Rcout << "Error in inv tri chol sym(Kxx) at " << u << endl;
        data.track_chol_fails(u) = 1;
      }*/
      // +++++++++++++++++
      
      xCovHUV_inplace(Kcp_cache(i), coords, indexing(u), parents_indexing(u), cparams, Dmat);
      
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
  
  //if(arma::all(data.track_chol_fails == 0)){
  //  data.cholfail = false;
    
    message("[get_cond_comps_loglik_w] step 3 - blocks.");
  //Rcpp::Rcout << "here." << endl;
  arma::vec ll_y = arma::zeros(w.n_rows);
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i = 0; i<n_ref_blocks; i++){
      int r = reference_blocks(i);
      int u = block_names(r)-1;
      //if(compute_block(predicting, block_ct_obs(u), false)){
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
          /*
        } else {
          Rcpp::Rcout << "you should not read this " << endl;
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
      
      //int u_cached_ix = coords_caching_ix(u);
      //arma::uvec cx = arma::find( coords_caching == u_cached_ix );
      arma::mat Kxxi = Kxxi_cache(cx(0));
      arma::mat Kcx = xCovHUV(coords, indexing_obs(u), indexing(u), cparams, Dmat, false);
      //Kcc = xCovHUV(coords, indexing_obs(u), indexing_obs(u), cparams, Dmat, true);
      //Rcpp::Rcout << arma::size(Zblock(u)) << " " << arma::size(Kcx) << " " << arma::size(Kxxi) << endl;
      data.KcxKxxi_obs(u) = Kcx * Kxxi;
      
      arma::mat ZK = Zblock(u) * Kcx;
      data.zH_obs(u) = Zblock(u) * data.KcxKxxi_obs(u);
      //arma::mat zHHz_obs = data.zH_obs(u) * Kcx.t() * Zblock(u).t();
      data.Ddiag(u) = arma::trans(arma::sum(data.zH_obs(u).t() % (ZK.t()), 0)); //data.zHHz_obs(u).diag();
      //Rcpp::Rcout << arma::size(zHHz_obs) << " " << arma::size(data.Ddiag(u)) << endl;
      
      for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
        if(na_1_blocks(u)(ix) == 1){
          //Rcpp::Rcout << "data.Ddiag(u)(i) " << data.Ddiag(u)(i) << endl;
          //Rcpp::Rcout << "y(indexing_obs(u)(i)) " << y(indexing_obs(u)(i)) << endl;
          //Rcpp::Rcout << "X.row(indexing_obs(u)(i)) * Bcoeff " << X.row(indexing_obs(u)(i)) * Bcoeff << endl;
          double ysigmasq = data.Ddiag(u)(ix) + 1.0/tausq_inv;
          double ytilde = y(indexing_obs(u)(ix)) - 
            arma::conv_to<double>::from(X.row(indexing_obs(u)(ix)) * Bcoeff);// - Zw.row(indexing_obs(u)(i)));
          ll_y(indexing_obs(u)(ix)) = -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
        }
      }
    }
    
    data.logdetCi = arma::accu(data.logdetCi_comps);
    data.ll_y_all = arma::accu(ll_y);
    data.loglik_w = data.logdetCi + arma::accu(data.loglik_w_comps) + data.ll_y_all;
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
    
    Rcpp::Rcout << cparams.t() << endl << Dmat << endl;
    throw 1;
  }*/
    
    
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[get_cond_comps_chached_loglik_w] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}

void MeshGPdev::gibbs_sample_beta(){
  message("[gibbs_sample_beta]");
  start = std::chrono::steady_clock::now();
  
  arma::mat Xprecy = Vim + //tausq_inv * 
    X_available.t() * ( y_available - Zw.rows(na_ix_all));// + ywmeandiff );
  Rcpp::RNGScope scope;
  Bcoeff = Sigma_chol_Bcoeff.t() * (Sigma_chol_Bcoeff * Xprecy + pow(tausq_inv, -.5) * arma::randn(p));
  
  if(verbose){
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_beta] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
  }
}


void MeshGPdev::gibbs_sample_tausq(){
  start = std::chrono::steady_clock::now();
  
  arma::vec XB_availab = X_available * Bcoeff;
  arma::vec Zw_availab = Zw.rows(na_ix_all);
  
  arma::mat yrr = y_available - XB_availab - 
    Zw_availab;
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

void MeshGPdev::update_ll_after_beta_tausq(){
  
  arma::vec ll_y = arma::zeros(w.n_rows);
  for(int i = 0; i<n_ref_blocks; i++){
    int r = reference_blocks(i);
    int u = block_names(r)-1;
    
    for(int ix=0; ix<indexing_obs(u).n_elem; ix++){
      if(na_1_blocks(u)(ix) == 1){
        //Rcpp::Rcout << "data.Ddiag(u)(i) " << data.Ddiag(u)(i) << endl;
        //Rcpp::Rcout << "y(indexing_obs(u)(i)) " << y(indexing_obs(u)(i)) << endl;
        //Rcpp::Rcout << "X.row(indexing_obs(u)(i)) * Bcoeff " << X.row(indexing_obs(u)(i)) * Bcoeff << endl;
        double ysigmasq = param_data.Ddiag(u)(ix) + 1.0/tausq_inv;
        double ytilde = y(indexing_obs(u)(ix)) - 
          arma::conv_to<double>::from(X.row(indexing_obs(u)(ix)) * Bcoeff);// - Zw.row(indexing_obs(u)(i)));
        ll_y(indexing_obs(u)(ix)) = -.5 * log(ysigmasq) - 1.0/(2*ysigmasq)*pow(ytilde, 2);
      }
    }
  }
  param_data.ll_y_all = arma::accu(ll_y);
  param_data.loglik_w = param_data.logdetCi + arma::accu(param_data.loglik_w_comps) + param_data.ll_y_all;
  
}

void MeshGPdev::gibbs_sample_w(){
  gibbs_sample_w_omp_nocache();
}

void MeshGPdev::gibbs_sample_w_omp_nocache(){
  if(verbose & debug){
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] " << endl;
  }
  
  Rcpp::RNGScope scope;
  rand_norm_mat = arma::randn(coords.n_rows, q);
  //Rcpp::Rcout << rand_norm_mat.head_rows(10) << endl << " ? " << endl;
  
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
            //Rcpp::Rcout << "g: " << g << " ~ u: " << u << " ~ child " << c << " - " << child << "\n";
            
            arma::uvec child_is_ref = arma::find(ref_block_names == child, 1, "first");
            if(child_is_ref.n_elem>0){
              //Rcpp::Rcout << u_is_which_col_f(u)(c)(0).t() << "\n";
              //Rcpp::Rcout << u_is_which_col_f(u)(c)(1).t() << "\n";
              
              arma::mat AK_u = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(0));
              arma::mat AK_uP = AK_u.t() * param_data.w_cond_prec(child);
              arma::mat AK_others = param_data.w_cond_mean_K(child).cols(u_is_which_col_f(u)(c)(1));
              //Rcpp::Rcout << "here 1" << endl;
              arma::vec w_child = arma::vectorise(arma::trans( w.rows( indexing(child) ) ));
              
              arma::mat w_parents_of_child = w.rows( parents_indexing(child) );
              arma::vec w_par_child = arma::vectorise(arma::trans(w_parents_of_child));
              arma::vec w_par_child_select = w_par_child.rows(u_is_which_col_f(u)(c)(1));
              Sigi_tot += AK_uP * AK_u;
              Smu_tot += AK_uP * ( w_child - AK_others * w_par_child_select );
            }
          }
          
          //*** findme
          //Rcpp::Rcout << arma::size(param_data.ZK(u)) << " " << arma::size(Ib(u)) << " " << arma::size(param_data.Ddiag(u)) << endl;
          //arma::mat ZDinv = //param_data.ZK(u).t() * Ib(u) * 
          //  Ib(u) * arma::diagmat(1.0/(param_data.Ddiag(u) + 1.0/tausq_inv)) * Ib(u);
          //arma::mat Sigi_y = Zblock(u).t() * ZDinv * Zblock(u); //param_data.ZK(u);
          //arma::mat Smu_y = Zblock(u).t() * ZDinv *
          //  (y.rows(indexing_obs(u)) - X.rows(indexing_obs(u)) * Bcoeff);
          arma::mat Sigi_y = tausq_inv * param_data.zH_obs(u).t() * Ib(u) * param_data.zH_obs(u);
          arma::mat Smu_y = param_data.zH_obs(u).t() * ((tausq_inv * na_1_blocks(u)) % //findme
            ( y.rows(indexing_obs(u)) - X.rows(indexing_obs(u)) * Bcoeff ));
          Sigi_tot += Sigi_y;
          Smu_tot += Smu_y;
          //Sigi_tot += tausq_inv * Zblock(u).t() * Ib(u) * Zblock(u);
          //Smu_tot += Zblock(u).t() * ((tausq_inv * na_1_blocks(u)) % //findme
           // ( y.rows(indexing(u)) - X.rows(indexing(u)) * Bcoeff ));
          
          //start = std::chrono::steady_clock::now();
          arma::mat Sigi_chol = arma::inv(arma::trimatl(arma::chol( arma::symmatu( Sigi_tot ), "lower")));
          //end = std::chrono::steady_clock::now();
          arma::vec rnvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
          //arma::vec rnvec = arma::randn(q*indexing(u).n_elem);
          arma::vec w_temp = Sigi_chol.t() * (Sigi_chol * Smu_tot + rnvec); 
          w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
          
          //***
          w.rows(indexing_obs(u)) = param_data.KcxKxxi_obs(u) * w_temp;
          
          Zw.rows(indexing(u)) = armarowsum(Z.rows(indexing(u)) % w.rows(indexing(u)));
          Zw.rows(indexing_obs(u)) = param_data.zH_obs(u) * w_temp; //***
          
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

  //Zw = armarowsum(Z % w); //***
  param_data.logdetCi = arma::accu(param_data.logdetCi_comps);
  param_data.loglik_w = param_data.logdetCi + arma::accu(param_data.loglik_w_comps);
  
  if(verbose){
    end_overall = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[gibbs_sample_w_omp_nocache] gibbs loops "
                << std::chrono::duration_cast<std::chrono::microseconds>(end_overall - start_overall).count()
                << "us. " << "\n";
  }
}

void MeshGPdev::predict(){
  if(predict_group_exists == 1){
    if(verbose & debug){
      Rcpp::Rcout << "[predict] " << endl;
    }
    
#ifdef _OPENMP
#pragma omp parallel for 
#endif
    for(int i=0; i<u_predicts.n_elem; i++){
      int u = u_predicts(i);
      // only predictions at this block. 
      // sample from conditional MVN 
      arma::mat Kcc = xCovHUV(coords, indexing(u), indexing(u), cparams, Dmat, true);
      arma::mat Kxxi = arma::inv_sympd( xCovHUV(coords, parents_indexing(u), parents_indexing(u), cparams, Dmat, true) );
      arma::mat Kcx = xCovHUV(coords, indexing(u), parents_indexing(u), cparams, Dmat, false);
        
      //Rcpp::Rcout << "? 1 " << endl;
      param_data.w_cond_mean_K(u) = Kcx * Kxxi;
      param_data.w_cond_cholprec(u) = arma::flipud(arma::fliplr(arma::chol( arma::symmatu(Kcc - param_data.w_cond_mean_K(u) * Kcx.t()) , "lower")));
    
    
      arma::vec w_par = arma::vectorise(arma::trans(w.rows(parents_indexing(u))));
      
      arma::vec phimean = param_data.w_cond_mean_K(u) * w_par;
      
      arma::mat cholK = param_data.w_cond_cholprec(u);
      
      arma::vec normvec = arma::vectorise(rand_norm_mat.rows(indexing(u)));
      arma::vec w_temp = phimean + cholK * normvec;
      
      w.rows(indexing(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
      
      //***
      Zw.rows(indexing(u)) = armarowsum(Z.rows(indexing(u)) % w.rows(indexing(u)));
      
      //***
      arma::mat Kyx = xCovHUV(coords, indexing_obs(u), parents_indexing(u), cparams, Dmat, false);
      //arma::mat Kyy = xCovHUV(coords, indexing_obs(u), indexing_obs(u), cparams, Dmat, true);
      
      arma::vec Var1diag = arma::zeros(indexing_obs(u).n_elem);
      arma::uvec oneuv = arma::ones<arma::uvec>(1);
      for(int j=0; j<indexing_obs(u).n_elem; j++){
        arma::mat result = xCovHUV(coords, indexing_obs(u).row(j), indexing_obs(u).row(j), cparams, Dmat, true);
        Var1diag(j) = result(0,0);
      }
      //arma::mat KyxKxxiKyx = Kyx*Kxxi*Kyx.t();
      arma::mat KxxiKyx = Kxxi * Kyx.t();
      //arma::vec Var1diag = Kyy.diag();
      arma::vec Var2diag = //KyxKxxiKyx.diag() ;//
        arma::trans(arma::sum(Kyx.t() % KxxiKyx, 0));
      arma::vec prVar = Var1diag - Var2diag;
      
      normvec = arma::vectorise(rand_norm_mat.rows(indexing_obs(u)));
      w_temp = Kyx * Kxxi * w_par + pow(prVar, .5) % normvec;
        
      w.rows(indexing_obs(u)) = arma::trans(arma::mat(w_temp.memptr(), q, w_temp.n_elem/q));
      Zw.rows(indexing_obs(u)) = Zblock(u) * w_temp;//***
        
      
      
    }
  }
}

void MeshGPdev::theta_update(MeshDataDev& data, const arma::vec& new_param){
  message("[theta_update] Updating theta");
  data.theta = new_param;
}

void MeshGPdev::tausq_update(double new_tausq){
  tausq_inv = 1.0/new_tausq;
}

void MeshGPdev::beta_update(const arma::vec& new_beta){ 
  Bcoeff = new_beta;
}

void MeshGPdev::accept_make_change(){
  // theta has changed
  std::swap(param_data, alter_data);
}