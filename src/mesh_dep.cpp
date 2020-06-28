#include <RcppArmadillo.h>
#include <omp.h>
#include "R.h"
#include <stdexcept>
#include <string>

#include "find_nan.h"

using namespace std;

arma::vec noseqdup(arma::vec x, bool& has_changed, int maxc, int na=-1, int pred=4){
  
  arma::uvec locs = arma::find( (x != na) % (x != pred) );
  arma::vec xlocs = x.elem(locs);
  for(int i=1; i<xlocs.n_elem; i++){
    if(xlocs(i)==xlocs(i-1)){
      xlocs(i) += 1+maxc;
      has_changed = true;
    }
  }
  x(locs) = xlocs;
  return x;
}

//[[Rcpp::export]]
arma::field<arma::uvec> blanket(const arma::field<arma::uvec>& parents, 
                                const arma::field<arma::uvec>& children,
                                const arma::uvec& names,
                                const arma::uvec& block_ct_obs){
  arma::uvec oneuv = arma::ones<arma::uvec>(1);
  int n_blocks = names.n_elem;
  arma::field<arma::uvec> mb(n_blocks);
  
  for(int i=0; i<n_blocks; i++){
    int u = names(i) - 1;
    if(block_ct_obs(u) > 0){
      // block cannot be the same color as other nodes in blanket
      arma::uvec u_blanket = arma::zeros<arma::uvec>(0);
      for(int p=0; p<parents(u).n_elem; p++){
        int px = parents(u)(p);
        u_blanket = arma::join_vert(u_blanket, px*oneuv);
      }
      for(int c=0; c<children(u).n_elem; c++){
        int cx = children(u)(c);
        u_blanket = arma::join_vert(u_blanket, cx*oneuv);
        
        for(int pc=0; pc<parents(cx).n_elem; pc++){
          int pcx = parents(cx)(pc);
          if(pcx != u){
            u_blanket = arma::join_vert(u_blanket, pcx*oneuv);
          }
        }
      }
      mb(u) = u_blanket;
    }
  }
  return mb;
}

arma::ivec std_setdiff(arma::ivec& x, arma::ivec& y) {
  std::vector<int> a = arma::conv_to< std::vector<int> >::from(arma::sort(x));
  std::vector<int> b = arma::conv_to< std::vector<int> >::from(arma::sort(y));
  std::vector<int> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                      std::inserter(out, out.end()));
  return arma::conv_to<arma::ivec>::from(out);
}

//[[Rcpp::export]]
arma::ivec coloring(const arma::field<arma::uvec>& blanket, const arma::uvec& block_names, const arma::uvec& block_ct_obs){
  int n_blocks = blanket.n_elem;
  arma::ivec oneiv = arma::ones<arma::ivec>(1);
  arma::ivec color_picker = arma::zeros<arma::ivec>(1);
  arma::ivec colors = arma::zeros<arma::ivec>(n_blocks) - 1;
  
  int u = block_names(0) - 1;
  colors(u) = 0;
  
  for(int i=1; i<n_blocks; i++){
    u = block_names(i) - 1;
    if(block_ct_obs(u) > 0){
      
      arma::ivec neighbor_colors = colors(blanket(u));
      
      arma::ivec neighbor_colors_used = neighbor_colors(arma::find(neighbor_colors > -1));
      
      arma::ivec colors_available = std_setdiff(color_picker, neighbor_colors_used);
      
      int choice_color = -1;
      if(colors_available.n_elem > 0){
        choice_color = arma::min(colors_available);
      } else {
        choice_color = arma::max(color_picker) + 1;
        color_picker = arma::join_vert(color_picker, oneiv * choice_color);
      }
      
      if(false){
        Rcpp::Rcout << "-- -- -- -- "<< endl;
        Rcpp::Rcout << "u: " << u << endl;
        Rcpp::Rcout << "neighbors: " << blanket(u).t() << endl;
        Rcpp::Rcout << "colors of neighbors of u: " << neighbor_colors << endl;
        Rcpp::Rcout << "colors of neighbors of u (used): " << neighbor_colors_used << endl;
        Rcpp::Rcout << "colors available: " << colors_available << endl;
        Rcpp::Rcout << "color assigned to " << u << ": " << choice_color << endl;
      }
      
      colors(u) = choice_color;
    }
    
  }
  return colors;
}


//[[Rcpp::export]]
arma::mat mesh_gibbs_groups(const arma::mat& layers_descr, 
                              const arma::uvec& Mv, bool rfc){
  
  Rcpp::Rcout << "~ Grouping nodes for parallel Gibbs sampling ";
  std::chrono::steady_clock::time_point start, start_all;
  std::chrono::steady_clock::time_point end, end_all;
  
  start_all = std::chrono::steady_clock::now();
  
  int dimen = 2;
  if(layers_descr.n_cols > 5){
    dimen = 3;
  }
  
  int num_blocks = arma::prod(Mv);
  
  if(dimen == 2){
    arma::mat Col2 = arma::zeros(Mv(0), Mv(1))-1;
    #pragma omp parallel for 
    for(int i=1; i<Mv(0)+1; i++){
      arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_alli.n_rows > 0){
        for(int j=1; j<Mv(1)+1; j++){
          arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
          if(filter_allj.n_rows > 0){
            Col2(i-1, j-1) = filter_allj(0, 4);
          }
        }
      }
    }
    
    int maxg = Col2.max();
    bool has_changed=true;
    int it = 0;
    
    int Imax = Col2.n_rows-1;
    int Jmax = Col2.n_cols-1;
    
    while((has_changed) & it<20){
      for(int i=1; i<Mv(0)+1; i++){
        for(int j=1; j<Mv(1)+1; j++){
          has_changed = false;
          // no repeated color along each axis

          // 1xJ
          arma::mat col_ax2 = Col2.submat(i-1,    0, 
                                          i-1, Jmax); 
          Col2.row(i-1) = arma::trans(noseqdup(arma::vectorise(col_ax2), has_changed, maxg, -1, 4));
          
          // Ix1
          arma::mat col_ax1 = Col2.submat(   0, j-1,
                                          Imax, j-1); 
          Col2.col(j-1) = noseqdup(arma::vectorise(col_ax1), has_changed, maxg, -1, 4);
          
        }
      }
      it++;
      Rcpp::Rcout << it << " " ;
    }
    Rcpp::Rcout << " done. " << endl;
    arma::mat fincol = arma::zeros(num_blocks, 3);
    it=0;
    for(int i=0; i<Mv(0); i++){
      for(int j=0; j<Mv(1); j++){
        fincol.row(it) = arma::rowvec({i+1.0, j+1.0, Col2(i,j)});
        it++;
      
      }
    } 
    return fincol;
  } else {
    arma::cube Col3 = arma::zeros(Mv(0), Mv(1), Mv(2))-1;
    #pragma omp parallel for 
    for(int i=1; i<Mv(0)+1; i++){
      arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_alli.n_rows > 0){
        for(int j=1; j<Mv(1)+1; j++){
          arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
          if(filter_allj.n_rows > 0){
            for(int h=1; h<Mv(2)+1; h++){
              arma::mat filter_allh = filter_allj.rows(arma::find(filter_allj.col(2) == h));
              if(filter_allh.n_rows > 0){ 
                Col3(i-1, j-1, h-1) = filter_allh(0, 5);
              }
            }
          }
        }
      }
    }
    
    int maxg = Col3.max();
    bool has_changed=true;
    int it = 0;
    
    int Imax = Col3.n_rows-1;
    int Jmax = Col3.n_cols-1;
    int Hmax = Col3.n_slices-1;
    
    while((has_changed) & it<20){
      for(int i=1; i<Mv(0)+1; i++){
        for(int j=1; j<Mv(1)+1; j++){
          for(int h=1; h<Mv(2)+1; h++){
            has_changed = false;
            // no repeated color along each axis
            
            // 1x1xH
            //Rcpp::Rcout << "1" << endl;
            arma::cube col_ax1 = Col3.subcube(i-1, j-1, 0, 
                                              i-1, j-1, Hmax); 
            
            //Rcpp::Rcout << "2" << endl;
            col_ax1.subcube(0, 0, 0, 0, 0, Hmax) = noseqdup(arma::vectorise(col_ax1), has_changed, maxg, -1, 8);
            
            //Rcpp::Rcout << "3" << endl;
            Col3.subcube(i-1, j-1, 0, 
                         i-1, j-1, Hmax) = col_ax1;
            
            //Rcpp::Rcout << "4" << endl;
            // 1xJx1
            arma::cube col_ax2 = Col3.subcube(i-1,    0, h-1,
                                              i-1, Jmax, h-1); 
            //Rcpp::Rcout << "5 " << arma::size(col_ax2) << " " << Jmax << endl;
            col_ax2.subcube(0, 0, 0, 0, Jmax, 0) = noseqdup(arma::vectorise(col_ax2), has_changed, maxg, -1, 8);
            
            //Rcpp::Rcout << "6" << endl;
            Col3.subcube(i-1, 0,    h-1, 
                         i-1, Jmax, h-1) = col_ax2;
            //Rcpp::Rcout << "7" << endl;
            // Ix1x1
            arma::cube col_ax3 = Col3.subcube(   0, j-1, h-1, 
                                              Imax, j-1, h-1); 
            //Rcpp::Rcout << "8" << endl;
            col_ax3.subcube(0, 0, 0, Imax, 0, 0) = noseqdup(arma::vectorise(col_ax3), has_changed, maxg, -1, 8);
            //Rcpp::Rcout << "9" << endl;
            Col3.subcube(0,    j-1, h-1, 
                         Imax, j-1, h-1) = col_ax3;
            
          }
        }
      }
      it++;
      Rcpp::Rcout << it << " " ;
    }
    Rcpp::Rcout << " done. " << endl;
    arma::mat fincol = arma::zeros(num_blocks, 4);
    it=0;
    for(int i=0; i<Mv(0); i++){
      for(int j=0; j<Mv(1); j++){
        for(int h=0; h<Mv(2); h++){
          fincol.row(it) = arma::rowvec({i+1.0, j+1.0, h+1.0, Col3(i,j,h)});
          it++;
        }
      }
    } 
    return fincol;
  }
  
  
}


arma::vec turbocolthreshold(const arma::vec& col1, const arma::vec& thresholds){
  //col %>% sapply(function(x) as.character(1+sum(x >= thresholds)))
  
  arma::vec result = arma::zeros(col1.n_elem);
  for(int i=0; i<col1.n_elem; i++){
    int overthreshold = 1;
    for(int j=0; j<thresholds.n_elem; j++){
      if(col1(i) >= thresholds(j)){
        overthreshold += 1;
      }
    }
    result(i) = overthreshold;
  }
  return result;
}

arma::vec kthresholds(arma::vec& x,
                      int k){
  arma::vec res(k-1);
  
  for(unsigned int i=1; i<k; i++){
    unsigned int Q1 = i * x.n_elem / k;
    std::nth_element(x.begin(), x.begin() + Q1, x.end());
    res(i-1) = x(Q1);
  }
  
  return res;
}


//[[Rcpp::export]]
arma::vec kthresholdscp(arma::vec x,
                      int k){
  arma::vec res(k-1);
  
  for(unsigned int i=1; i<k; i++){
    unsigned int Q1 = i * x.n_elem / k;
    std::nth_element(x.begin(), x.begin() + Q1, x.end());
    res(i-1) = x(Q1);
  }
  
  return res;
}

//[[Rcpp::export]]
arma::mat part_axis_parallel(const arma::mat& coords, const arma::vec& Mv, int n_threads, bool verbose=false){
  
  if(verbose){
    Rcpp::Rcout << "~ Axis-parallel partitioning... ";
  }
  arma::mat resultmat = arma::zeros(arma::size(coords));
  
//#pragma omp parallel for num_threads(n_threads)
  for(int j=0; j<coords.n_cols; j++){
    //std::vector<double> cjv = arma::conv_to<std::vector<double> >::from(coords.col(j));
    arma::vec cja = coords.col(j);
    arma::vec thresholds = kthresholds(cja, Mv(j));
    resultmat.col(j) = turbocolthreshold(coords.col(j), thresholds);
  }
  if(verbose){
    Rcpp::Rcout << "done." << endl;
  }
  
  
  return resultmat;
}


//[[Rcpp::export]]
arma::mat part_axis_parallel_fixed(const arma::mat& coords, const arma::field<arma::vec>& thresholds, int n_threads){
  //Rcpp::Rcout << "~ Axis-parallel partitioning [fixed thresholds]... ";
  arma::mat resultmat = arma::zeros(arma::size(coords));
  
  #pragma omp parallel for num_threads(n_threads)
  for(int j=0; j<coords.n_cols; j++){
    //std::vector<double> cjv = arma::conv_to<std::vector<double> >::from(coords.col(j));
    arma::vec cja = coords.col(j);
    arma::vec thresholds_col = thresholds(j);
    resultmat.col(j) = turbocolthreshold(coords.col(j), thresholds_col);
  }
  //Rcpp::Rcout << "done." << endl;
  return resultmat;
}

//[[Rcpp::export]]
Rcpp::List mesh_graph_cpp(const arma::mat& layers_descr, 
                          const arma::uvec& Mv, bool rfc,
                          bool verbose=true){
  // coords_layering is a matrix
  // Var1 Var2 [Var3] L1 L2 [L3] layer na_which
  // layers_descr = coords_layering %>% select(-contains("Var")) 
  //                                %>% group_by(L1, L2, L3, layer) 
  //                                %>% summarize(na_which = sum(na_which))
  //                                %>% unique()
  
  std::chrono::steady_clock::time_point start, start_all;
  std::chrono::steady_clock::time_point end, end_all;
  
  start_all = std::chrono::steady_clock::now();
  
  int dimen = 2;
  if(layers_descr.n_cols > 4){
    dimen = 3;
  }
  
  if(verbose){
    Rcpp::Rcout << "~ Building cubic mesh, d = " << dimen << endl;
    /*if(!rfc){
      Rcpp::Rcout << "~ S covers T: prediction iterations may take longer." << endl; 
    } else {
      Rcpp::Rcout << "~ S on all D: change rfc to F if very large gaps to fill." << endl;
    }*/
  }
  
  int num_blocks = arma::prod(Mv);
  
  arma::vec lnames = layers_descr.col(dimen);
  arma::field<arma::vec> parents(num_blocks);
  arma::field<arma::vec> children(num_blocks);
  arma::uvec uzero = arma::zeros<arma::uvec>(1);
  
  //Rcpp::Rcout << "1" << endl;
  for(int i=0; i<num_blocks; i++){
    parents(i) = arma::zeros(dimen) - 1;
    children(i) = arma::zeros(dimen) - 1;
  }
  
  arma::mat blocks_ref = layers_descr;
  
  if(!rfc){
    blocks_ref = blocks_ref.rows(arma::find(layers_descr.col(dimen+1) > 0));//layers_preds;
  }
  
  arma::cube Q, Qall;
  arma::mat Qm, Qmall;
  
  
  //Rcpp::Rcout << "2" << endl;
  if(dimen == 2){
    
    start = std::chrono::steady_clock::now();
    Qm = arma::zeros(Mv(0), Mv(1))-1;
  #pragma omp parallel for
    for(int i=1; i<Mv(0)+1; i++){
      arma::mat filter_i    = blocks_ref.rows(arma::find(blocks_ref.col(0) == i));
      if(filter_i.n_rows > 0){
        for(int j=1; j<Mv(1)+1; j++){
          arma::mat filter_j    = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){ 
            Qm(i-1, j-1) = filter_j(0, 2);
          } 
        }
      }
    }
    
    if(!rfc){
      Qmall = arma::zeros(Mv(0), Mv(1))-1;
      #pragma omp parallel for
      for(int i=1; i<Mv(0)+1; i++){
        arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
        if(filter_alli.n_rows > 0){
          for(int j=1; j<Mv(1)+1; j++){
            arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
            if(filter_allj.n_rows > 0){
              Qmall(i-1, j-1) = filter_allj(0, 2);
            }
          }
        }
      }
      end = std::chrono::steady_clock::now();
    }
    
    //Rcpp::Rcout << "[part 0] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
    //            << endl;
    
    int Imax = Qm.n_rows-1;
    int Jmax = Qm.n_cols-1;
    
    start = std::chrono::steady_clock::now();
    #pragma omp parallel for num_threads(7)
    for(int i=1; i<Mv(0)+1; i++){
      for(int j=1; j<Mv(1)+1; j++){
        //for(int h=1; h<M+1; h++){
        int layern = Qm(i-1, j-1);
        if(layern != -1){
          //start_1 = std::chrono::steady_clock::now();
          
          if(j < Mv(1)){
            arma::vec q1 = arma::vectorise(Qm.submat(i-1, j,  
                                                     i-1, Jmax));
            
            arma::uvec locator_sub_ijh_1 = arma::find(q1 != -1);
            if(locator_sub_ijh_1.n_elem > 0){
              int prop_layern = q1(locator_sub_ijh_1(0));
              children(layern-1)(0) = prop_layern-1;
              parents(prop_layern-1)(0) = layern-1;
            }
          }
          
          //end_1 = std::chrono::steady_clock::now();
          //timed_1 += std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count();
          
          //start_2 = std::chrono::steady_clock::now();
          
          if(i < Mv(0)){
            arma::vec q2 = Qm.submat(i,    j-1, 
                                     Imax, j-1);
            
            arma::uvec locator_sub_ijh_2 = arma::find(q2 != -1);
            if(locator_sub_ijh_2.n_elem > 0){
              int prop_layern = q2(locator_sub_ijh_2(0));
              children(layern-1)(1) = prop_layern-1;
              parents(prop_layern-1)(1) = layern-1;
            }
          }
          
          //end_2 = std::chrono::steady_clock::now();
          //timed_2 += std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count();
          
        }
        
        //}
      }
    }
    
    end = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "[part 1] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
    //            << endl;
    
    if(!rfc){
      start = std::chrono::steady_clock::now();
      arma::uvec empties = arma::find(Qm == -1);
#pragma omp parallel for
      for(int i=0; i<empties.n_elem; i++){
        arma::uvec ijh = arma::ind2sub(arma::size(Qm), empties(i));
        
        int layern = Qmall(ijh(0), ijh(1));
        if(layern > -1){
          //Rcpp::Rcout << ijh.t() << endl;
          arma::vec axis1 = arma::vectorise(Qm.row(ijh(0))); // fix row ijh(0), look at cols
          arma::vec axis2 = arma::vectorise(Qm.col(ijh(1)));
          
          arma::uvec nne1 = arma::find(axis1 > -1);
          arma::uvec nne2 = arma::find(axis2 > -1);
          
          // free cols
          arma::uvec loc1before = arma::find(nne1 < ijh(1));
          arma::uvec loc1after = arma::find(nne1 > ijh(1));
          
          // free rows
          arma::uvec loc2before = arma::find(nne2 < ijh(0));
          arma::uvec loc2after = arma::find(nne2 > ijh(0));
          
          parents(layern-1) = arma::zeros(4) -1;
          if(loc1before.n_elem > 0){
            //Rcpp::Rcout << "1" << endl;
            arma::uvec nne1_before = nne1(loc1before);
            parents(layern-1)(0) = arma::conv_to<int>::from(
              axis1(nne1_before.tail(1)) - 1);
          }
          if(loc1after.n_elem > 0){
            //Rcpp::Rcout << "2" << endl;
            arma::uvec nne1_after  = nne1(loc1after);
            parents(layern-1)(1) = arma::conv_to<int>::from(
              axis1(nne1_after.head(1)) - 1);
          }
          if(loc2before.n_elem > 0){
            //Rcpp::Rcout << "3" << endl;
            arma::uvec nne2_before = nne2(loc2before);
            parents(layern-1)(2) = arma::conv_to<int>::from(
              axis2(nne2_before.tail(1)) - 1);
          }
          if(loc2after.n_elem > 0){
            //Rcpp::Rcout << "4" << endl;
            arma::uvec nne2_after  = nne2(loc2after);
            parents(layern-1)(3) = arma::conv_to<int>::from(
              axis2(nne2_after.head(1)) - 1);
          }
        } 
        
      }
      end = std::chrono::steady_clock::now();
    }
    
    //Rcpp::Rcout << "[part 2] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //            << "us.\n";
    
  } else {
    start = std::chrono::steady_clock::now();
    Q = arma::zeros(Mv(0), Mv(1), Mv(2))-1;
    #pragma omp parallel for 
    for(int i=1; i<Mv(0)+1; i++){
      arma::mat filter_i = blocks_ref.rows(arma::find(blocks_ref.col(0) == i));
      if(filter_i.n_rows > 0){
        for(int j=1; j<Mv(1)+1; j++){
          arma::mat filter_j = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){
            for(int h=1; h<Mv(2)+1; h++){
              arma::mat filter_h = filter_j.rows(arma::find(filter_j.col(2) == h));
              if(filter_h.n_rows > 0){ 
                Q(i-1, j-1, h-1) = filter_h(0, 3);
              } 
            }
          }
        }
      }
      
    }
    
    if(!rfc){
      Qall = arma::zeros(Mv(0), Mv(1), Mv(2))-1;
      #pragma omp parallel for 
      for(int i=1; i<Mv(0)+1; i++){
        arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
        if(filter_alli.n_rows > 0){
          for(int j=1; j<Mv(1)+1; j++){
            arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
            if(filter_allj.n_rows > 0){
              for(int h=1; h<Mv(2)+1; h++){
                arma::mat filter_allh = filter_allj.rows(arma::find(filter_allj.col(2) == h));
                if(filter_allh.n_rows > 0){ 
                  Qall(i-1, j-1, h-1) = filter_allh(0, 3);
                }
              }
            }
          }
        }
        
      }
    }
    
    
    end = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "[part 0] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
    //            << endl;
    
    int Imax = Q.n_rows-1;
    int Jmax = Q.n_cols-1;
    int Hmax = Q.n_slices-1;
    start = std::chrono::steady_clock::now();
    
    #pragma omp parallel for
    for(int i=1; i<Mv(0)+1; i++){
      for(int j=1; j<Mv(1)+1; j++){
        for(int h=1; h<Mv(2)+1; h++){
          int layern = Q(i-1, j-1, h-1);
          if(layern != -1){
            //start_1 = std::chrono::steady_clock::now();
            
            if(j < Mv(1)){
              arma::vec q1 = arma::vectorise(Q.subcube(i-1, j,    h-1, 
                                                       i-1, Jmax, h-1));
              
              arma::uvec locator_sub_ijh_1 = arma::find(q1 != -1);
              if(locator_sub_ijh_1.n_elem > 0){
                int prop_layern = q1(locator_sub_ijh_1(0));
                children(layern-1)(0) = prop_layern-1;
                parents(prop_layern-1)(0) = layern-1;
              }
            }
            
            //end_1 = std::chrono::steady_clock::now();
            //timed_1 += std::chrono::duration_cast<std::chrono::microseconds>(end_1 - start_1).count();
            
            //start_2 = std::chrono::steady_clock::now();
            
            if(i < Mv(0)){
              arma::vec q2 = Q.subcube(i,    j-1, h-1, 
                                       Imax, j-1, h-1);
              
              arma::uvec locator_sub_ijh_2 = arma::find(q2 != -1);
              if(locator_sub_ijh_2.n_elem > 0){
                int prop_layern = q2(locator_sub_ijh_2(0));
                children(layern-1)(1) = prop_layern-1;
                parents(prop_layern-1)(1) = layern-1;
              }
            }
            
            //end_2 = std::chrono::steady_clock::now();
            //timed_2 += std::chrono::duration_cast<std::chrono::microseconds>(end_2 - start_2).count();
            
            //start_3 = std::chrono::steady_clock::now();
            
            if(h < Mv(2)){
              arma::vec q3 = arma::vectorise(Q.subcube(i-1, j-1, h, 
                                                       i-1, j-1, Hmax));
              
              arma::uvec locator_sub_ijh_3 = arma::find(q3 != -1);
              if(locator_sub_ijh_3.n_elem > 0){
                int prop_layern = q3(locator_sub_ijh_3(0));
                children(layern-1)(2) = prop_layern-1;
                parents(prop_layern-1)(2) = layern-1;
              }
            }
            
            //end_3 = std::chrono::steady_clock::now();
            //timed_3 += std::chrono::duration_cast<std::chrono::microseconds>(end_3 - start_3).count();
            
          }
          
        }
      }
    }
    
    end = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "[part 1] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
    //            << endl;
    
    if(!rfc){
      start = std::chrono::steady_clock::now();
      arma::uvec empties = arma::find(Q == -1);
      for(int i=0; i<empties.n_elem; i++){
        arma::uvec ijh = arma::ind2sub(arma::size(Q), empties(i));
        
        int layern = Qall(ijh(0), ijh(1), ijh(2));
        if(layern > -1){
          //Rcpp::Rcout << ijh.t() << endl;
          arma::vec axis1 = arma::vectorise(Q.subcube(ijh(0), ijh(1), 0, 
                                                      ijh(0), ijh(1), Hmax)); 
          arma::vec axis2 = arma::vectorise(Q.subcube(ijh(0), 0,    ijh(2), 
                                                      ijh(0), Jmax, ijh(2))); 
          arma::vec axis3 = arma::vectorise(Q.subcube(0,    ijh(1), ijh(2), 
                                                      Imax, ijh(1), ijh(2))); 
          
          arma::uvec nne1 = arma::find(axis1 > -1);
          arma::uvec nne2 = arma::find(axis2 > -1);
          arma::uvec nne3 = arma::find(axis3 > -1);
          
          // free slices
          arma::uvec loc1before = arma::find(nne1 < ijh(2));
          arma::uvec loc1after = arma::find(nne1 > ijh(2));
          
          // free cols
          arma::uvec loc2before = arma::find(nne2 < ijh(1));
          arma::uvec loc2after = arma::find(nne2 > ijh(1));
          
          // free rows
          arma::uvec loc3before = arma::find(nne3 < ijh(0));
          arma::uvec loc3after = arma::find(nne3 > ijh(0));
          
          parents(layern-1) = arma::zeros(6) -1;
          
          if(loc1before.n_elem > 0){
            //Rcpp::Rcout << "1" << endl;
            arma::uvec nne1_before = nne1(loc1before);
            parents(layern-1)(0) = arma::conv_to<int>::from(
              axis1(nne1_before.tail(1)) - 1);
          }
          if(loc1after.n_elem > 0){
            //Rcpp::Rcout << "2" << endl;
            arma::uvec nne1_after  = nne1(loc1after);
            parents(layern-1)(1) = arma::conv_to<int>::from(
              axis1(nne1_after.head(1)) - 1);
          }
          if(loc2before.n_elem > 0){
            //Rcpp::Rcout << "3" << endl;
            arma::uvec nne2_before = nne2(loc2before);
            parents(layern-1)(2) = arma::conv_to<int>::from(
              axis2(nne2_before.tail(1)) - 1);
          }
          if(loc2after.n_elem > 0){
            //Rcpp::Rcout << "4" << endl;
            arma::uvec nne2_after  = nne2(loc2after);
            parents(layern-1)(3) = arma::conv_to<int>::from(
              axis2(nne2_after.head(1)) - 1);
          }
          if(loc3before.n_elem > 0){
            //Rcpp::Rcout << "5" << endl;
            arma::uvec nne3_before = nne3(loc3before); 
            parents(layern-1)(4) = arma::conv_to<int>::from(
              axis3(nne3_before.tail(1)) - 1);
          }
          if(loc3after.n_elem > 0){
            //Rcpp::Rcout << "6" << endl;
            arma::uvec nne3_after  = nne3(loc3after);
            parents(layern-1)(5) = arma::conv_to<int>::from(
              axis3(nne3_after.head(1)) - 1);
          }
        }
        
        
      }
      end = std::chrono::steady_clock::now();
    }
    //Rcpp::Rcout << "[part 2] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //            << "us.\n";
    
  }
  
  
  #pragma omp parallel for
  for(int i=0; i<parents.n_elem; i++){
    parents(i) = parents(i).elem(arma::find(parents(i) != -1));
    children(i) = children(i).elem(arma::find(children(i) != -1));
  }
  
  end_all = std::chrono::steady_clock::now();
  //Rcpp::Rcout << "[overall] "
  //            << std::chrono::duration_cast<std::chrono::microseconds>(end_all - start_all).count()
  //            << "us.\n";
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = lnames
  );
}




