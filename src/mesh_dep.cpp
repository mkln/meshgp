#include <RcppArmadillo.h>
#include <omp.h>
#include "R.h"
#include <stdexcept>
#include <string>

#include "find_nan.h"

using namespace std;


//[[Rcpp::export]]
arma::vec noseqdup(arma::vec x, bool& has_changed, int maxc, int na=-1, int pred=2){
  
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
    //***#pragma omp parallel for 
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
          Col2.row(i-1) = arma::trans(noseqdup(arma::vectorise(col_ax2), has_changed, maxg, -1, 2));
          
          // Ix1
          arma::mat col_ax1 = Col2.submat(   0, j-1,
                                          Imax, j-1); 
          Col2.col(j-1) = noseqdup(arma::vectorise(col_ax1), has_changed, maxg, -1, 2);
          
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
    //***#pragma omp parallel for 
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
    Rcpp::Rcout << maxg << endl;
    
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
            col_ax1.subcube(0, 0, 0, 0, 0, Hmax) = noseqdup(arma::vectorise(col_ax1), has_changed, maxg, -1, 2);
            
            //Rcpp::Rcout << "3" << endl;
            Col3.subcube(i-1, j-1, 0, 
                         i-1, j-1, Hmax) = col_ax1;
            
            //Rcpp::Rcout << "4" << endl;
            // 1xJx1
            arma::cube col_ax2 = Col3.subcube(i-1,    0, h-1,
                                              i-1, Jmax, h-1); 
            //Rcpp::Rcout << "5 " << arma::size(col_ax2) << " " << Jmax << endl;
            col_ax2.subcube(0, 0, 0, 0, Jmax, 0) = noseqdup(arma::vectorise(col_ax2), has_changed, maxg, -1, 2);
            
            //Rcpp::Rcout << "6" << endl;
            Col3.subcube(i-1, 0,    h-1, 
                         i-1, Jmax, h-1) = col_ax2;
            //Rcpp::Rcout << "7" << endl;
            // Ix1x1
            arma::cube col_ax3 = Col3.subcube(   0, j-1, h-1, 
                                              Imax, j-1, h-1); 
            //Rcpp::Rcout << "8" << endl;
            col_ax3.subcube(0, 0, 0, Imax, 0, 0) = noseqdup(arma::vectorise(col_ax3), has_changed, maxg, -1, 2);
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



//[[Rcpp::export]]
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

//[[Rcpp::export]]
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
arma::mat part_axis_parallel(const arma::mat& coords, const arma::vec& Mv, int n_threads){
  Rcpp::Rcout << "~ Axis-parallel partitioning... ";
  arma::mat resultmat = arma::zeros(arma::size(coords));
  
//#pragma omp parallel for num_threads(n_threads)
  for(int j=0; j<coords.n_cols; j++){
    //std::vector<double> cjv = arma::conv_to<std::vector<double> >::from(coords.col(j));
    arma::vec cja = coords.col(j);
    arma::vec thresholds = kthresholds(cja, Mv(j));
    resultmat.col(j) = turbocolthreshold(coords.col(j), thresholds);
  }
  Rcpp::Rcout << "done." << endl;
  
  return resultmat;
}

//[[Rcpp::export]]
Rcpp::List mesh_graph_cpp(const arma::mat& layers_descr, 
                          const arma::uvec& Mv, bool rfc){
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
  if(layers_descr.n_cols > 5){
    dimen = 3;
  }
  Rcpp::Rcout << "~ Building cubic mesh, d = " << dimen << endl;
  if(!rfc){
    Rcpp::Rcout << "~ S covers T: prediction iterations may take longer." << endl; 
  } else {
    Rcpp::Rcout << "~ S on all D: change rfc to F if very large gaps to fill." << endl;
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
  //***#pragma omp parallel for
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
      //***#pragma omp parallel for
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
    //***#pragma omp parallel for num_threads(7)
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
//***#pragma omp parallel for
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
    //***#pragma omp parallel for 
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
      //***#pragma omp parallel for 
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
    
    //***#pragma omp parallel for
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
  
  
  //***#pragma omp parallel for
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




// backup of old functions

Rcpp::List mesh_dep_cpp(const arma::mat& layers_descr, 
                        int M){
  // coords_layering is a matrix
  // Var1 Var2 [Var3] L1 L2 [L3] layer na_which
  // layers_descr = coords_layering %>% select(-contains("Var")) 
  //                                %>% group_by(L1, L2, L3, layer) 
  //                                %>% summarize(na_which = sum(na_which))
  //                                %>% unique()
  
  
  int dimen = 2;
  if(layers_descr.n_cols > 4){
    dimen = 3;
  }
  
  arma::field<arma::vec> parents(layers_descr.n_rows);
  arma::field<arma::vec> children(layers_descr.n_rows);
  
  if(dimen == 2){
  
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M; j++){
        arma::mat filtered_ld = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j)));
        int layern = filtered_ld(0, 2);
        int na_which = filtered_ld(0, 3);
        if(na_which > 0){
          int s = 1;
          bool found = false;
          while(!found){
            arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j+s)));
            int prop_layern = prop_child(0, 2);
            int prop_na_which = prop_child(0, 3);
            if(prop_na_which > 0){
              arma::vec newlayer = arma::zeros(1) + prop_layern;
              children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
              newlayer = newlayer - prop_layern + layern;
              parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
              found = true;
            } else {
              if(j + s == M){
                found = true;
              }
              s ++;
            }
          }
        }
      }
    }
    
    for(int j=1; j<M+1; j++){
      for(int i=1; i<M; i++){
        arma::mat filtered_ld = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j)));
        int layern = filtered_ld(0, 2);
        int na_which = filtered_ld(0, 3);
        if(na_which > 0){
          int s = 1;
          bool found = false;
          while(!found){
            arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i+s) % (layers_descr.col(1) == j)));
            int prop_layern = prop_child(0, 2);
            int prop_na_which = prop_child(0, 3);
            if(prop_na_which > 0){
              arma::vec newlayer = arma::zeros(1) + prop_layern;
              children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
              newlayer = newlayer - prop_layern + layern;
              parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
              found = true;
            } else {
              if(i + s == M){
                found = true;
              }
              s ++;
            }
          }
        }
      }
    }
  
  } else {
    
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        for(int h=1; h<M; h++){
          arma::mat filtered_ld = layers_descr.rows(
            arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h))
          );
          
          int layern = filtered_ld(0, 3);
          int na_which = filtered_ld(0, 4);
          if(na_which > 0){
            int s = 1;
            bool found = false;
            while(!found){
              arma::mat prop_child = layers_descr.rows(
                arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h+s))
              );
              int prop_layern = prop_child(0, 3);
              int prop_na_which = prop_child(0, 4);
              if(prop_na_which > 0){
                arma::vec newlayer = arma::zeros(1) + prop_layern;
                children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
                newlayer = newlayer - prop_layern + layern;
                parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
                found = true;
              } else {
                if(h + s == M){
                  found = true;
                }
                s ++;
              }
            }
          }
        }
      }
    }
    
    for(int i=1; i<M+1; i++){
      for(int h=1; h<M+1; h++){
        for(int j=1; j<M; j++){
          arma::mat filtered_ld = layers_descr.rows(
            arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h))
          );
          
          int layern = filtered_ld(0, 3);
          int na_which = filtered_ld(0, 4);
          if(na_which > 0){
            int s = 1;
            bool found = false;
            while(!found){
              arma::mat prop_child = layers_descr.rows(
                arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j+s) % (layers_descr.col(2) == h))
              );
              int prop_layern = prop_child(0, 3);
              int prop_na_which = prop_child(0, 4);
              if(prop_na_which > 0){
                arma::vec newlayer = arma::zeros(1) + prop_layern;
                children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
                newlayer = newlayer - prop_layern + layern;
                parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
                found = true;
              } else {
                if(j + s == M){
                  found = true;
                }
                s ++;
              }
            }
          }
        }
      }
    }
    
    for(int j=1; j<M+1; j++){
      for(int h=1; h<M+1; h++){
        for(int i=1; i<M; i++){
          arma::mat filtered_ld = layers_descr.rows(
            arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h))
          );
          
          int layern = filtered_ld(0, 3);
          int na_which = filtered_ld(0, 4);
          if(na_which > 0){
            int s = 1;
            bool found = false;
            while(!found){
              arma::mat prop_child = layers_descr.rows(
                arma::find((layers_descr.col(0) == i+s) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h))
              );
              int prop_layern = prop_child(0, 3);
              int prop_na_which = prop_child(0, 4);
              if(prop_na_which > 0){
                arma::vec newlayer = arma::zeros(1) + prop_layern;
                children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
                newlayer = newlayer - prop_layern + layern;
                parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
                found = true;
              } else {
                if(i + s == M){
                  found = true;
                }
                s ++;
              }
            }
          }
        }
      }
    }
    
  }
  
  // dependence for empty layers
  arma::mat layers_empty = layers_descr.rows(arma::find(layers_descr.col(dimen+1) == 0));
  
  //arma::field<arma::vec> parents_pred(layers_descr.n_rows);
  //arma::field<arma::vec> children_pred(layers_descr.n_rows);
  
  if(dimen == 2){
    for(int el=0; el<layers_empty.n_rows; el++){
      
      int i = layers_empty(el, 0);
      int j = layers_empty(el, 1);
      int lhere = layers_empty(el, 2) -1;
      if(i > 1){
        for(int ix=i-1; ix>0; ix--){
          arma::mat prop_parent = layers_descr.rows(arma::find((layers_descr.col(0) == ix) % (layers_descr.col(1) == j)));
          int layern = prop_parent(0, 2);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(2) == layern));
          int nbl_here = not_na_count(0, 3);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            parents(lhere) = arma::join_vert(parents(lhere), newl);
            break;
          }
        }
      }
      if(j > 1){
        for(int jx=j-1; jx>0; jx--){
          arma::mat prop_parent = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == jx)));
          int layern = prop_parent(0, 2);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(2) == layern));
          int nbl_here = not_na_count(0, 3);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            parents(lhere) = arma::join_vert(parents(lhere), newl);
            break;
          }
        }
      }
      if(i < M){
        for(int ix=i+1; ix<M+1; ix++){
          arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == ix) % (layers_descr.col(1) == j)));
          int layern = prop_child(0, 2);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(2) == layern));
          int nbl_here = not_na_count(0, 3);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            children(lhere) = arma::join_vert(children(lhere), newl);
            break;
          }
        }
      }
      if(j < M){
        for(int jx=j+1; jx<M+1; jx++){
          arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == jx)));
          int layern = prop_child(0, 2);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(2) == layern));
          int nbl_here = not_na_count(0, 3);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            children(lhere) = arma::join_vert(children(lhere), newl);
            break;
          }
        }
      }
        
    }
  } else {
    for(int el=0; el<layers_empty.n_rows; el++){
      
      int i = layers_empty(el, 0);
      int j = layers_empty(el, 1);
      int h = layers_empty(el, 2);
      int lhere = layers_empty(el, 3) -1;
      if(i > 1){
        for(int ix=i-1; ix>0; ix--){
          arma::mat prop_parent = layers_descr.rows(arma::find((layers_descr.col(0) == ix) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h)));
          int layern = prop_parent(0, 3);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(3) == layern));
          int nbl_here = not_na_count(0, 4);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            parents(lhere) = arma::join_vert(parents(lhere), newl);
            break;
          }
        }
      }
      if(j > 1){
        for(int jx=j-1; jx>0; jx--){
          arma::mat prop_parent = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == jx) % (layers_descr.col(2) == h)));
          int layern = prop_parent(0, 3);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(3) == layern));
          int nbl_here = not_na_count(0, 4);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            parents(lhere) = arma::join_vert(parents(lhere), newl);
            break;
          }
        }
      }
      if(h > 1){
        for(int hx=h-1; hx>0; hx--){
          arma::mat prop_parent = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == hx)));
          int layern = prop_parent(0, 3);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(3) == layern));
          int nbl_here = not_na_count(0, 4);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            parents(lhere) = arma::join_vert(parents(lhere), newl);
            break;
          }
        }
      }
      if(i < M){
        for(int ix=i+1; ix<M+1; ix++){
          arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == ix) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h)));
          int layern = prop_child(0, 3);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(3) == layern));
          int nbl_here = not_na_count(0, 4);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            children(lhere) = arma::join_vert(children(lhere), newl);
            break;
          }
        }
      }
      if(j < M){
        for(int jx=j+1; jx<M+1; jx++){
          arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == jx) % (layers_descr.col(2) == h)));
          int layern = prop_child(0, 3);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(3) == layern));
          int nbl_here = not_na_count(0, 4);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            children(lhere) = arma::join_vert(children(lhere), newl);
            break;
          }
        }
      }
      if(h < M){
        for(int hx=h+1; hx<M+1; hx++){
          arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == hx)));
          int layern = prop_child(0, 3);
          arma::mat not_na_count = layers_descr.rows(arma::find(layers_descr.col(3) == layern));
          int nbl_here = not_na_count(0, 4);
          if(nbl_here > 0){
            arma::vec newl = arma::zeros(1) + layern-1;
            children(lhere) = arma::join_vert(children(lhere), newl);
            break;
          }
        }
      }
    }
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = layers_descr.col(dimen)
  );
}



Rcpp::List mesh_dep_rfc_cpp(const arma::mat& layers_descr, 
                              int M){
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
  Rcpp::Rcout << "~ building hypercube graph, d=" << dimen << " M=" << M << ", S on all D" << endl;
  
  int num_blocks = pow(M, dimen);
  
  arma::vec lnames = layers_descr.col(dimen);
  arma::field<arma::vec> parents(num_blocks);
  arma::field<arma::vec> children(num_blocks);
  arma::uvec uzero = arma::zeros<arma::uvec>(1);
  
  //Rcpp::Rcout << "1" << endl;
  for(int i=0; i<num_blocks; i++){
    parents(i) = arma::zeros(dimen) - 1;
    children(i) = arma::zeros(dimen) - 1;
  }
  
  arma::cube Q, Qall;
  arma::mat Qm, Qmall;
  
  //Rcpp::Rcout << "2" << endl;
  if(dimen == 2){
    
    start = std::chrono::steady_clock::now();
    Qm = arma::zeros(M, M)-1;
    //***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      arma::mat filter_i    = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_i.n_rows > 0){
        for(int j=1; j<M+1; j++){
          arma::mat filter_j    = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){ 
            Qm(i-1, j-1) = filter_j(0, 2);
          } 
        }
      }
    }
    end = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "[part 0] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
    //            << endl;
    
    int Imax = Qm.n_rows-1;
    int Jmax = Qm.n_cols-1;
    
    start = std::chrono::steady_clock::now();
    //***#pragma omp parallel for num_threads(7)
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        //for(int h=1; h<M+1; h++){
        int layern = Qm(i-1, j-1);
        if(layern != -1){
          //start_1 = std::chrono::steady_clock::now();
          
          if(j < M){
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
          
          if(i < M){
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

  } else {
    start = std::chrono::steady_clock::now();
    Q = arma::zeros(M, M, M)-1;
    //***#pragma omp parallel for 
    for(int i=1; i<M+1; i++){
      arma::mat filter_i = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_i.n_rows > 0){
        for(int j=1; j<M+1; j++){
          arma::mat filter_j = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){
            for(int h=1; h<M+1; h++){
              arma::mat filter_h = filter_j.rows(arma::find(filter_j.col(2) == h));
              if(filter_h.n_rows > 0){ 
                Q(i-1, j-1, h-1) = filter_h(0, 3);
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
    
    //***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        for(int h=1; h<M+1; h++){
          int layern = Q(i-1, j-1, h-1);
          if(layern != -1){
            //start_1 = std::chrono::steady_clock::now();
            
            if(j < M){
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
            
            if(i < M){
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
            
            if(h < M){
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

    //Rcpp::Rcout << "[part 2] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //            << "us.\n";
    
  }
  
  
  //***#pragma omp parallel for
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



Rcpp::List mesh_dep_norfc_cpp(const arma::mat& layers_descr, 
                           int M){
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
  Rcpp::Rcout << "~ building hypercube graph, d=" << dimen << " M=" << M << ", S covers T" << endl; 
  
  int num_blocks = pow(M, dimen);
  
  arma::vec lnames = layers_descr.col(dimen);
  arma::field<arma::vec> parents(num_blocks);
  arma::field<arma::vec> children(num_blocks);
  arma::uvec uzero = arma::zeros<arma::uvec>(1);
  
  //Rcpp::Rcout << "1" << endl;
  for(int i=0; i<num_blocks; i++){
    parents(i) = arma::zeros(dimen) - 1;
    children(i) = arma::zeros(dimen) - 1;
  }
  
  arma::mat layers_notempty = layers_descr.rows(arma::find(layers_descr.col(dimen+1) > 0));//layers_preds;

  arma::cube Q, Qall;
  arma::mat Qm, Qmall;
  
  //Rcpp::Rcout << "2" << endl;
  if(dimen == 2){
    
    start = std::chrono::steady_clock::now();
    Qm = arma::zeros(M, M)-1;
//***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      arma::mat filter_i    = layers_notempty.rows(arma::find(layers_notempty.col(0) == i));
      if(filter_i.n_rows > 0){
        for(int j=1; j<M+1; j++){
          arma::mat filter_j    = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){ 
            Qm(i-1, j-1) = filter_j(0, 2);
          } 
        }
      }
    }
    Qmall = arma::zeros(M, M)-1;
    //***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_alli.n_rows > 0){
        for(int j=1; j<M+1; j++){
          arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
          if(filter_allj.n_rows > 0){
            Qmall(i-1, j-1) = filter_allj(0, 2);
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    //Rcpp::Rcout << "[part 0] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() 
    //            << endl;
    
    int Imax = Qm.n_rows-1;
    int Jmax = Qm.n_cols-1;
    
    start = std::chrono::steady_clock::now();
    //***#pragma omp parallel for num_threads(7)
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        //for(int h=1; h<M+1; h++){
          int layern = Qm(i-1, j-1);
          if(layern != -1){
            //start_1 = std::chrono::steady_clock::now();
            
            if(j < M){
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
            
            if(i < M){
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
    start = std::chrono::steady_clock::now();
    arma::uvec empties = arma::find(Qm == -1);
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
    //Rcpp::Rcout << "[part 2] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //            << "us.\n";
    
  } else {
    start = std::chrono::steady_clock::now();
    Q = arma::zeros(M, M, M)-1;
//***#pragma omp parallel for 
    for(int i=1; i<M+1; i++){
      arma::mat filter_i = layers_notempty.rows(arma::find(layers_notempty.col(0) == i));
      if(filter_i.n_rows > 0){
        for(int j=1; j<M+1; j++){
          arma::mat filter_j = filter_i.rows(arma::find(filter_i.col(1) == j));
          if(filter_j.n_rows > 0){
            for(int h=1; h<M+1; h++){
              arma::mat filter_h = filter_j.rows(arma::find(filter_j.col(2) == h));
              if(filter_h.n_rows > 0){ 
                Q(i-1, j-1, h-1) = filter_h(0, 3);
              } 
            }
          }
        }
      }
      
    }
    
    Qall = arma::zeros(M, M, M)-1;
//***#pragma omp parallel for 
    for(int i=1; i<M+1; i++){
      arma::mat filter_alli = layers_descr.rows(arma::find(layers_descr.col(0) == i));
      if(filter_alli.n_rows > 0){
        for(int j=1; j<M+1; j++){
          arma::mat filter_allj = filter_alli.rows(arma::find(filter_alli.col(1) == j));
          if(filter_allj.n_rows > 0){
            for(int h=1; h<M+1; h++){
              arma::mat filter_allh = filter_allj.rows(arma::find(filter_allj.col(2) == h));
              if(filter_allh.n_rows > 0){ 
                Qall(i-1, j-1, h-1) = filter_allh(0, 3);
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
    
//***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        for(int h=1; h<M+1; h++){
          int layern = Q(i-1, j-1, h-1);
          if(layern != -1){
            //start_1 = std::chrono::steady_clock::now();
            
            if(j < M){
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
            
            if(i < M){
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
            
            if(h < M){
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
    
    //Rcpp::Rcout << "[part 2] "
    //            << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
    //            << "us.\n";
    
  }
    
    
//***#pragma omp parallel for
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



Rcpp::List mesh_depall_cpp_old(const arma::mat& layers_descr, 
                           int M){
  // coords_layering is a matrix
  // Var1 Var2 [Var3] L1 L2 [L3] layer na_which
  // layers_descr = coords_layering %>% select(-contains("Var")) 
  //                                %>% group_by(L1, L2, L3, layer) 
  //                                %>% summarize(na_which = sum(na_which))
  //                                %>% unique()
  
  
  int dimen = 2;
  if(layers_descr.n_cols > 4){
    dimen = 3;
  }
  
  arma::field<arma::vec> parents(layers_descr.n_rows);
  arma::field<arma::vec> children(layers_descr.n_rows);
  
  
  if(dimen == 2){
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        arma::mat filtered_ld = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j)));
        int layern = filtered_ld(0, 2);
        
        arma::mat prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j+1)));
        if(prop_child.n_rows > 0){
          int prop_layern = prop_child(0, 2);
          arma::vec newlayer = arma::zeros(1) + prop_layern;
          children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
          newlayer = newlayer - prop_layern + layern;
          parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
        }
        
        prop_child = layers_descr.rows(arma::find((layers_descr.col(0) == i+1) % (layers_descr.col(1) == j)));
        if(prop_child.n_rows > 0){
          int prop_layern = prop_child(0, 2);
          arma::vec newlayer = arma::zeros(1) + prop_layern;
          children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
          newlayer = newlayer - prop_layern + layern;
          parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
        }
      }
    }
    
  } else {
    for(int i=1; i<M+1; i++){
      for(int j=1; j<M+1; j++){
        for(int h=1; h<M+1; h++){
          arma::mat filtered_ld = layers_descr.rows(
            arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h))
          );
          int layern = filtered_ld(0, 3);
          
          arma::mat prop_child = layers_descr.rows(
            arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h+1))
          );
          if(prop_child.n_rows > 0){
            int prop_layern = prop_child(0, 3);
            arma::vec newlayer = arma::zeros(1) + prop_layern;
            children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
            newlayer = newlayer - prop_layern + layern;
            parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
          }
          
          prop_child = layers_descr.rows(
            arma::find((layers_descr.col(0) == i) % (layers_descr.col(1) == j+1) % (layers_descr.col(2) == h))
          );
          if(prop_child.n_rows > 0){
            int prop_layern = prop_child(0, 3);
            arma::vec newlayer = arma::zeros(1) + prop_layern;
            children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
            newlayer = newlayer - prop_layern + layern;
            parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
          }
          
          prop_child = layers_descr.rows(
            arma::find((layers_descr.col(0) == i+1) % (layers_descr.col(1) == j) % (layers_descr.col(2) == h))
          );
          if(prop_child.n_rows > 0){
            int prop_layern = prop_child(0, 3);
            arma::vec newlayer = arma::zeros(1) + prop_layern;
            children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
            newlayer = newlayer - prop_layern + layern;
            parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
          }
        }
        
      }
    }
  }
  
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = layers_descr.col(dimen)
  );
}




Rcpp::List mesh_dep_rfc_backup_cpp(arma::mat layers_descr, 
                                   int M){
  // coords_layering is a matrix
  // Var1 Var2 [Var3] L1 L2 [L3] layer na_which
  // layers_descr = coords_layering %>% select(-contains("Var")) 
  //                                %>% group_by(L1, L2, L3, layer) 
  //                                %>% summarize(na_which = sum(na_which))
  //                                %>% unique()
  
  // problem with this is that it assumes that i+1, j+1, h+1 exist
  int dimen = 2;
  if(layers_descr.n_cols > 4){
    dimen = 3;
  }
  
  Rcpp::Rcout << "~ building hypercube graph, d=" << dimen << " M=" << M << ", S on all D" << endl;
  
  int num_blocks = pow(M, dimen);
  
  arma::vec lnames = layers_descr.col(dimen);
  arma::field<arma::vec> parents(num_blocks);
  arma::field<arma::vec> children(num_blocks);
  
  //Rcpp::Rcout << "1" << endl;
  for(int i=0; i<num_blocks; i++){
    parents(i) = arma::zeros(dimen) - 1;
    children(i) = arma::zeros(dimen) - 1;
  }
  
  //Rcpp::Rcout << "2" << endl;
  if(dimen == 2){
    for(int i=1; i<M+1; i++){
      arma::uvec locator_i = arma::find(
        ( (layers_descr.col(0) == i) + (layers_descr.col(0) == i+1) ) );
      
      if(locator_i.n_elem > 0){
        arma::mat sub_i = layers_descr.rows(locator_i);
        
        for(int j=1; j<M+1; j++){
          //Rcpp::Rcout << "i: " << i << " j: " << j << endl;
          
          arma::uvec locator_ij = arma::find(
            ( (sub_i.col(1) == j) + (sub_i.col(1) == j+1) ) );
          
          if(locator_ij.n_elem > 0){
            arma::mat sub_ij = sub_i.rows(locator_ij);
            arma::uvec locator_sub_ij = arma::find((sub_ij.col(0) == i) % (sub_ij.col(1) == j));
            
            if(locator_sub_ij.n_elem > 0){
              arma::mat filtered_ld = sub_ij.rows(locator_sub_ij);
              int layern = filtered_ld(0, 2);
              //Rcpp::Rcout << layern << endl;
              
              //Rcpp::Rcout << "first axis" << endl;
              arma::uvec locator_sub_ij_1 = arma::find((sub_ij.col(0) == i) % (sub_ij.col(1) == j+1));
              if(locator_sub_ij_1.n_elem > 0){
                arma::mat prop_child = sub_ij.rows(locator_sub_ij_1);
                if(prop_child.n_rows > 0){
                  int prop_layern = prop_child(0, 2);
                  //Rcpp::Rcout << "> " << prop_layern << endl;
                  
                  children(layern-1)(0) = prop_layern-1;
                  parents(prop_layern-1)(0) = layern-1;
                }
              }
              
              //Rcpp::Rcout << "second axis" << endl;
              arma::uvec locator_sub_ij_2 = arma::find((sub_ij.col(0) == i+1) % (sub_ij.col(1) == j));
              if(locator_sub_ij_2.n_elem > 0){
                arma::mat prop_child = sub_ij.rows(locator_sub_ij_2);
                
                if(prop_child.n_rows > 0){
                  int prop_layern = prop_child(0, 2);
                  //Rcpp::Rcout << "> " << prop_layern << endl;
                  children(layern-1)(1) = prop_layern-1;
                  parents(prop_layern-1)(1) = layern-1;
                }
              }
            }
          }
        }
        
        layers_descr = layers_descr.rows( arma::find( layers_descr.col(0) != i ) );
      }
      
    }
    
  } else {
    
    //***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      
      arma::uvec locator_i = arma::find(
        ( (layers_descr.col(0) == i) + (layers_descr.col(0) == i+1) ) );
      
      if(locator_i.n_elem > 0){
        arma::mat sub_i = layers_descr.rows(locator_i);
        
        for(int j=1; j<M+1; j++){
          
          arma::uvec locator_ij = arma::find(
            ( (sub_i.col(1) == j) + (sub_i.col(1) == j+1) ) );
          
          if(locator_ij.n_elem > 0){
            arma::mat sub_ij = sub_i.rows(locator_ij);
            
            for(int h=1; h<M+1; h++){
              
              arma::uvec locator_ijh = arma::find(
                ( (sub_ij.col(2) == h) + (sub_ij.col(2) == h+1) ) );
              
              if(locator_ijh.n_elem > 0){
                arma::mat sub_ijh = sub_ij.rows(locator_ijh);
                
                arma::uvec locator_sub_ijh = arma::find(
                  (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
                
                if(locator_sub_ijh.n_elem > 0){
                  arma::mat filtered_ld = sub_ijh.rows(locator_sub_ijh);
                  
                  int layern = filtered_ld(0, 3);
                  
                  arma::uvec locator_sub_ijh_1 = arma::find((sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h+1));
                  if(locator_sub_ijh_1.n_elem > 0){
                    arma::mat prop_child = sub_ijh.rows(locator_sub_ijh_1);
                    
                    if(prop_child.n_rows > 0){
                      int prop_layern = prop_child(0, 3);
                      //arma::vec newlayer = arma::zeros(1) + prop_layern;
                      //children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
                      //newlayer = newlayer - prop_layern + layern;
                      //parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
                      children(layern-1)(0) = prop_layern-1;
                      parents(prop_layern-1)(0) = layern-1;
                    }
                  }
                  
                  arma::uvec locator_sub_ijh_2 = arma::find((sub_ijh.col(0) == i) % (sub_ijh.col(1) == j+1) % (sub_ijh.col(2) == h));
                  if(locator_sub_ijh_2.n_elem > 0){
                    arma::mat prop_child = sub_ijh.rows(locator_sub_ijh_2);
                    
                    if(prop_child.n_rows > 0){
                      int prop_layern = prop_child(0, 3);
                      //arma::vec newlayer = arma::zeros(1) + prop_layern;
                      //children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
                      //newlayer = newlayer - prop_layern + layern;
                      //parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
                      children(layern-1)(1) = prop_layern-1;
                      parents(prop_layern-1)(1) = layern-1;
                    }
                  }
                  
                  arma::uvec locator_sub_ijh_3 = arma::find((sub_ijh.col(0) == i+1) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
                  if(locator_sub_ijh_3.n_elem > 0){
                    arma::mat prop_child = sub_ijh.rows(locator_sub_ijh_3);
                    
                    if(prop_child.n_rows > 0){
                      int prop_layern = prop_child(0, 3);
                      //arma::vec newlayer = arma::zeros(1) + prop_layern;
                      //children(layern-1) = arma::join_vert(children(layern-1), newlayer-1);
                      //newlayer = newlayer - prop_layern + layern;
                      //parents(prop_layern-1) = arma::join_vert(parents(prop_layern-1), newlayer-1);
                      children(layern-1)(2) = prop_layern-1;
                      parents(prop_layern-1)(2) = layern-1;
                    }
                  }
                }
                
              }
              
            }
            
            sub_i = sub_i.rows(arma::find( sub_i.col(1) != j) );
          }
          
        }
        
        layers_descr = layers_descr.rows( arma::find( layers_descr.col(0) != i ) );
      }
      
      
    }
  }
  
//***#pragma omp parallel for
  for(int i=0; i<parents.n_elem; i++){
    parents(i) = parents(i).elem(arma::find(parents(i) != -1));
    children(i) = children(i).elem(arma::find(children(i) != -1));
  }
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = lnames
  );
}


Rcpp::List mesh_dep_norfc_backup_cpp(arma::mat layers_descr, 
                            int M){
  // coords_layering is a matrix
  // Var1 Var2 [Var3] L1 L2 [L3] layer na_which
  // layers_descr = coords_layering %>% select(-contains("Var")) 
  //                                %>% group_by(L1, L2, L3, layer) 
  //                                %>% summarize(na_which = sum(na_which))
  //                                %>% unique()
  
  
  int dimen = 2;
  if(layers_descr.n_cols > 4){
    dimen = 3;
  }
  Rcpp::Rcout << "> building graph, d=" << dimen << endl; 
  
  int num_blocks = pow(M, dimen);
  
  arma::vec lnames = layers_descr.col(dimen);
  arma::field<arma::vec> parents(num_blocks);
  arma::field<arma::vec> children(num_blocks);
  arma::uvec uzero = arma::zeros<arma::uvec>(1);
  
  //Rcpp::Rcout << "1" << endl;
  for(int i=0; i<num_blocks; i++){
    parents(i) = arma::zeros(dimen) - 1;
    children(i) = arma::zeros(dimen) - 1;
  }
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  
  arma::mat layers_preds = layers_descr.rows(arma::find(layers_descr.col(dimen+1) > 0));
  arma::mat layers_notempty = layers_preds;
  arma::mat layers_empty = layers_descr.rows(arma::find(layers_descr.col(dimen+1) == 0));
  
  //Rcpp::Rcout << "2" << endl;
  if(dimen == 2){
    
    for(int i=1; i<M+1; i++){
      arma::uvec locator_i = arma::find( layers_preds.col(0) >= i );
      
      if(locator_i.n_elem > 0){
        arma::mat sub_i = layers_preds.rows(locator_i);
        
        for(int j=1; j<M+1; j++){
          //Rcpp::Rcout << "i: " << i << " j: " << j << endl;
          
          arma::uvec locator_ij = arma::find( sub_i.col(1) >= j );
          
          if(locator_ij.n_elem > 0){
            arma::mat sub_ij = sub_i.rows(locator_ij);
            arma::uvec locator_sub_ij = arma::find((sub_ij.col(0) == i) % (sub_ij.col(1) == j));
            
            if(locator_sub_ij.n_elem > 0){
              arma::mat filtered_ld = sub_ij.rows(locator_sub_ij);
              
              if(filtered_ld(0, dimen+1) > 0){
                // nonempty
                int layern = filtered_ld(0, 2);
                //Rcpp::Rcout << "nonempty: " <<  layern << endl;
                
                //Rcpp::Rcout << "first axis" << endl;
                arma::uvec locator_sub_ij_1 = arma::find(
                  (sub_ij.col(0) == i) % (sub_ij.col(1) > j));
                
                if(locator_sub_ij_1.n_elem > 0){
                  arma::mat prop_child = sub_ij.row(locator_sub_ij_1(0));
                  if(prop_child.n_rows > 0){
                    int prop_layern = prop_child(0, 2);
                    //Rcpp::Rcout << "> " << prop_layern << endl;
                    
                    children(layern-1)(0) = prop_layern-1;
                    parents(prop_layern-1)(0) = layern-1;
                  }
                }
                
                //Rcpp::Rcout << "second axis" << endl;
                arma::uvec locator_sub_ij_2 = arma::find(
                  (sub_ij.col(0) > i) % (sub_ij.col(1) == j));
                if(locator_sub_ij_2.n_elem > 0){
                  arma::mat prop_child = sub_ij.row(locator_sub_ij_2(0));
                  
                  if(prop_child.n_rows > 0){
                    int prop_layern = prop_child(0, 2);
                    //Rcpp::Rcout << "> " << prop_layern << endl;
                    children(layern-1)(1) = prop_layern-1;
                    parents(prop_layern-1)(1) = layern-1;
                  }
                }
              } 
            } 
          }
        }
        //layers_descr = layers_descr.rows( arma::find( layers_descr.col(0) != i ) );
      }
      
    }
    
    
    for(int i=1; i<M+1; i++){
      arma::uvec locator_i = arma::find( layers_descr.col(0) >= i );
      
      if(locator_i.n_elem > 0){
        arma::mat sub_i = layers_descr.rows(locator_i);
        
        for(int j=1; j<M+1; j++){
          arma::uvec locator_ij = arma::find( sub_i.col(1) >= j );
          
          if(locator_ij.n_elem > 0){
            arma::mat sub_ij = sub_i.rows(locator_ij);
            arma::uvec locator_sub_ij = arma::find((sub_ij.col(0) == i) % (sub_ij.col(1) == j));
            
            if(locator_sub_ij.n_elem > 0){
              arma::mat filtered_ld = sub_ij.rows(locator_sub_ij);
              
              if(filtered_ld(0, dimen+1) == 0){
                // all NA
                int layern = filtered_ld(0, 2);
                
                //Rcpp::Rcout << "first axis" << endl;
                arma::uvec locator_pred_ij_1 = arma::find(
                  (layers_preds.col(0) == i) + (layers_preds.col(1) == j));
                
                arma::mat candidates = layers_preds.rows(locator_pred_ij_1);
                parents(layern-1) = arma::zeros(4) -1;
                
                arma::uvec x_prev = arma::find(candidates.col(0) < i);
                if(x_prev.n_elem > 0){
                  int x1 = arma::conv_to<int>::from(x_prev.tail(1));
                  parents(layern-1)(0) = candidates(x1, 2)-1;
                }
                
                arma::uvec x_aftr = arma::find(candidates.col(0) > i);
                if(x_aftr.n_elem > 0){
                  int x2 = arma::conv_to<int>::from(x_aftr.head(1));
                  parents(layern-1)(1) = candidates(x2, 2)-1;
                }
                
                arma::uvec y_prev = arma::find(candidates.col(1) < j);
                if(y_prev.n_elem > 0){
                  int y1 = arma::conv_to<int>::from(y_prev.tail(1));
                  parents(layern-1)(2) = candidates(y1, 2)-1;
                }
                
                arma::uvec y_aftr = arma::find(candidates.col(1) > j);
                if(y_aftr.n_elem > 0){
                  int y2 = arma::conv_to<int>::from(y_aftr.head(1));
                  parents(layern-1)(3) = candidates(y2, 2)-1;
                }
              }
            } 
          }
        }
      }
    }
    
    
  } else {
    
    /*
     for(int i=1; i<M+1; i++){
     arma::uvec locator_i = arma::find( layers_notempty.col(0) >= i );
     
     if(locator_i.n_elem > 0){
     arma::mat sub_i = layers_notempty.rows(locator_i);
     
     for(int j=1; j<M+1; j++){
     //Rcpp::Rcout << "i: " << i << " j: " << j << endl;
     
     arma::uvec locator_ij = arma::find( sub_i.col(1) >= j );
     
     if(locator_ij.n_elem > 0){
     arma::mat sub_ij = sub_i.rows(locator_ij);
     
     for(int h=1; h<M+1; h++){
     arma::uvec locator_ijh = arma::find(sub_ij.col(2) >= h);
     
     if(locator_ijh.n_elem > 0){
     
     arma::mat sub_ijh = sub_ij.rows(locator_ijh);
     
     arma::uvec locator_sub_ijh = arma::find(
     (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
     
     if(locator_sub_ijh.n_elem > 0){
     arma::mat filtered_ld = sub_ijh.rows(locator_sub_ijh);
     
     if(filtered_ld(0, dimen+1) > 0){
     // nonempty
     int layern = filtered_ld(0, 3);
     //Rcpp::Rcout << "nonempty: " <<  layern << endl;
     
     //Rcpp::Rcout << "first axis" << endl;
     arma::uvec locator_sub_ijh_1 = arma::find(
     (sub_ijh.col(0) == i) % (sub_ijh.col(1) > j) % (sub_ijh.col(2) == h));
     
     if(locator_sub_ijh_1.n_elem > 0){
     arma::mat prop_child = sub_ijh.row(locator_sub_ijh_1(0));
     if(prop_child.n_rows > 0){
     int prop_layern = prop_child(0, 3);
     //Rcpp::Rcout << "> " << prop_layern << endl;
     
     children(layern-1)(0) = prop_layern-1;
     parents(prop_layern-1)(0) = layern-1;
     }
     }
     
     //Rcpp::Rcout << "second axis" << endl;
     arma::uvec locator_sub_ijh_2 = arma::find(
     (sub_ijh.col(0) > i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
     if(locator_sub_ijh_2.n_elem > 0){
     arma::mat prop_child = sub_ijh.row(locator_sub_ijh_2(0));
     
     if(prop_child.n_rows > 0){
     int prop_layern = prop_child(0, 3);
     //Rcpp::Rcout << "> " << prop_layern << endl;
     children(layern-1)(1) = prop_layern-1;
     parents(prop_layern-1)(1) = layern-1;
     }
     }
     
     //Rcpp::Rcout << "third axis" << endl;
     arma::uvec locator_sub_ijh_3 = arma::find(
     (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) > h));
     if(locator_sub_ijh_3.n_elem > 0){
     arma::mat prop_child = sub_ijh.row(locator_sub_ijh_3(0));
     
     if(prop_child.n_rows > 0){
     int prop_layern = prop_child(0, 3);
     //Rcpp::Rcout << "> " << prop_layern << endl;
     children(layern-1)(2) = prop_layern-1;
     parents(prop_layern-1)(2) = layern-1;
     }
     }
     
     } 
     } 
     
     }
     
     }
     
     //sub_i = sub_i.rows( arma::find( sub_i.col(1) > j ) );
     }
     }
     //layers_notempty = layers_notempty.rows( arma::find( layers_notempty.col(0) > i ) );
     }
     
     }
     */
    
    start = std::chrono::steady_clock::now();
//***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      //layers_notempty = layers_notempty.rows( arma::find( layers_notempty.col(0) >= i ) );
      arma::uvec locator_i = arma::find( layers_notempty.col(0) >= i );
      
      if(locator_i.n_elem > 0){
        arma::mat sub_i = layers_notempty.rows(locator_i);
        
        for(int j=1; j<M+1; j++){
          //Rcpp::Rcout << "i: " << i << " j: " << j << endl;
          sub_i = sub_i.rows( arma::find( sub_i.col(1) >= j ) );
          //arma::uvec locator_ij = arma::find( sub_i.col(1) >= j );
          
          if(sub_i.n_rows > 0){
            arma::mat sub_ij = sub_i;//.rows(locator_ij);
            
            for(int h=1; h<M+1; h++){
              arma::uvec locator_ijh = arma::find(sub_ij.col(2) >= h);
              
              if(locator_ijh.n_elem > 0){
                
                arma::mat sub_ijh = sub_ij.rows(locator_ijh);
                
                arma::uvec locator_sub_ijh = arma::find(
                  (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
                
                if(locator_sub_ijh.n_elem > 0){
                  arma::mat filtered_ld = sub_ijh.rows(locator_sub_ijh);
                  
                  if(filtered_ld(0, dimen+1) > 0){
                    // nonempty
                    int layern = filtered_ld(0, 3);
                    //Rcpp::Rcout << "nonempty: " <<  layern << endl;
                    
                    //Rcpp::Rcout << "first axis" << endl;
                    arma::uvec locator_sub_ijh_1 = arma::find(
                      (sub_ijh.col(0) == i) % (sub_ijh.col(1) > j) % (sub_ijh.col(2) == h));
                    
                    if(locator_sub_ijh_1.n_elem > 0){
                      arma::mat prop_child = sub_ijh.row(locator_sub_ijh_1(0));
                      if(prop_child.n_rows > 0){
                        int prop_layern = prop_child(0, 3);
                        //Rcpp::Rcout << "> " << prop_layern << endl;
                        
                        children(layern-1)(0) = prop_layern-1;
                        parents(prop_layern-1)(0) = layern-1;
                      }
                    }
                    
                    //Rcpp::Rcout << "second axis" << endl;
                    arma::uvec locator_sub_ijh_2 = arma::find(
                      (sub_ijh.col(0) > i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
                    if(locator_sub_ijh_2.n_elem > 0){
                      arma::mat prop_child = sub_ijh.row(locator_sub_ijh_2(0));
                      
                      if(prop_child.n_rows > 0){
                        int prop_layern = prop_child(0, 3);
                        //Rcpp::Rcout << "> " << prop_layern << endl;
                        children(layern-1)(1) = prop_layern-1;
                        parents(prop_layern-1)(1) = layern-1;
                      }
                    }
                    
                    //Rcpp::Rcout << "third axis" << endl;
                    arma::uvec locator_sub_ijh_3 = arma::find(
                      (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) > h));
                    if(locator_sub_ijh_3.n_elem > 0){
                      arma::mat prop_child = sub_ijh.row(locator_sub_ijh_3(0));
                      
                      if(prop_child.n_rows > 0){
                        int prop_layern = prop_child(0, 3);
                        //Rcpp::Rcout << "> " << prop_layern << endl;
                        children(layern-1)(2) = prop_layern-1;
                        parents(prop_layern-1)(2) = layern-1;
                      }
                    }
                    
                  } 
                } 
                
              }
              
            }
            
          }
        }
        
      }
      
    }
    end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[part 1] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
    
    
    /*
     for(int i=1; i<M+1; i++){
     arma::uvec locator_i = arma::find( layers_descr.col(0) >= i );
     
     if(locator_i.n_elem > 0){
     arma::mat sub_i = layers_descr.rows(locator_i);
     
     for(int j=1; j<M+1; j++){
     arma::uvec locator_ij = arma::find( sub_i.col(1) >= j );
     
     if(locator_ij.n_elem > 0){
     arma::mat sub_ij = sub_i.rows(locator_ij);
     
     for(int h=1; h<M+1; h++){
     arma::uvec locator_ijh = arma::find(sub_ij.col(2) >= h);
     
     if(locator_ijh.n_elem > 0){
     
     arma::mat sub_ijh = sub_ij.rows(locator_ijh);
     arma::uvec locator_sub_ijh = arma::find(
     (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
     
     
     if(locator_sub_ijh.n_elem > 0){
     arma::mat filtered_ld = sub_ijh.rows(locator_sub_ijh);
     
     if(filtered_ld(0, dimen+1) == 0){
     // all NA
     int layern = filtered_ld(0, 3);
     
     //Rcpp::Rcout << "first axis" << endl;
     arma::uvec locator_pred_ijh_1 = arma::find(
     (layers_preds.col(0) == i) + (layers_preds.col(1) == j) + (layers_preds.col(2) == h));
     
     arma::mat candidates = layers_preds.rows(locator_pred_ijh_1);
     parents(layern-1) = arma::zeros(6) -1;
     
     arma::uvec x_prev = arma::find(
     (candidates.col(0) < i) % (candidates.col(1) == j) % (candidates.col(2) == h));
     if(x_prev.n_elem > 0){
     int x1 = arma::conv_to<int>::from(x_prev.tail(1));
     parents(layern-1)(0) = candidates(x1, 3)-1;
     }
     
     arma::uvec x_aftr = arma::find(
     (candidates.col(0) > i) % (candidates.col(1) == j) % (candidates.col(2) == h));
     if(x_aftr.n_elem > 0){
     int x2 = arma::conv_to<int>::from(x_aftr.head(1));
     parents(layern-1)(1) = candidates(x2, 3)-1;
     }
     
     arma::uvec y_prev = arma::find(
     (candidates.col(0) == i) % (candidates.col(1) < j) % (candidates.col(2) == h));
     if(y_prev.n_elem > 0){
     int y1 = arma::conv_to<int>::from(y_prev.tail(1));
     parents(layern-1)(2) = candidates(y1, 3)-1;
     }
     
     arma::uvec y_aftr = arma::find(
     (candidates.col(0) == i) % (candidates.col(1) > j) % (candidates.col(2) == h));
     if(y_aftr.n_elem > 0){
     int y2 = arma::conv_to<int>::from(y_aftr.head(1));
     parents(layern-1)(3) = candidates(y2, 3)-1;
     }
     
     arma::uvec z_prev = arma::find(
     (candidates.col(0) == i) % (candidates.col(1) == j) % (candidates.col(2) < h));
     if(z_prev.n_elem > 0){
     int z1 = arma::conv_to<int>::from(z_prev.tail(1));
     parents(layern-1)(4) = candidates(z1, 3)-1;
     }
     
     arma::uvec z_aftr = arma::find(
     (candidates.col(0) == i) % (candidates.col(1) == j) % (candidates.col(2) > h));
     if(z_aftr.n_elem > 0){
     int z2 = arma::conv_to<int>::from(z_aftr.head(1));
     parents(layern-1)(5) = candidates(z2, 3)-1;
     }
     }
     } 
     
     }
     
     }
     
     }
     }
     }
     }
     */
    
    start = std::chrono::steady_clock::now();
//***#pragma omp parallel for
    for(int i=1; i<M+1; i++){
      //layers_empty = layers_empty.rows( arma::find(layers_empty.col(0) != i-1) );
      arma::uvec locator_i = arma::find(layers_empty.col(0) == i);
      
      if(locator_i.n_elem > 0){
        arma::mat sub_i = layers_empty.rows(locator_i);
        arma::uvec predsubs_i = arma::find(layers_preds.col(0) == i);
        
        for(int j=1; j<M+1; j++){
          sub_i = sub_i.rows( arma::find(sub_i.col(1) != j-1) );
          arma::uvec locator_ij = arma::find( sub_i.col(1) == j );
          
          if(locator_ij.n_elem > 0){
            arma::mat sub_ij = sub_i.rows(locator_ij);
            arma::uvec predsubs_ij = arma::join_vert( predsubs_i, arma::find(layers_preds.col(1) == j) );
            
            for(int h=1; h<M+1; h++){
              //sub_ij = sub_ij.rows( arma::find(sub_ij.col(2) != h-1) );
              arma::uvec locator_ijh = arma::find(sub_ij.col(2) == h);
              
              if(locator_ijh.n_elem > 0){
                arma::mat sub_ijh = sub_ij.rows(locator_ijh);
                
                //arma::uvec locator_sub_ijh = arma::find(
                //  (sub_ijh.col(0) == i) % (sub_ijh.col(1) == j) % (sub_ijh.col(2) == h));
                
                //if(locator_sub_ijh.n_elem > 0){
                arma::mat filtered_ld = sub_ijh;//sub_ijh.rows(locator_sub_ijh);
                
                // all NA
                int layern = filtered_ld(0, 3);
                
                //Rcpp::Rcout << "first axis" << endl;
                
                //arma::uvec locator_pred_ijh_1 = arma::find(
                //  (layers_preds.col(0) == i) + (layers_preds.col(1) == j) + (layers_preds.col(2) == h));
                //arma::mat candidates = layers_preds.rows(locator_pred_ijh_1);
                
                arma::uvec predsubs_ijh = arma::join_vert( predsubs_ij, arma::find(layers_preds.col(2) == h) );
                
                arma::mat candidates = layers_preds.rows(predsubs_ijh);//locator_pred_ijh_1);
                
                arma::mat candidates_i = candidates.rows(arma::find(
                  (candidates.col(1) == j) % (candidates.col(2) == h)
                ));
                arma::mat candidates_j = candidates.rows(arma::find(
                  (candidates.col(0) == i) % (candidates.col(2) == h)
                ));
                arma::mat candidates_h = candidates.rows(arma::find(
                  (candidates.col(0) == i) % (candidates.col(1) == j)
                ));
                
                //arma::uvec x_prev = arma::find((candidates.col(0) < i) % (candidates.col(1) == j) % (candidates.col(2) == h));
                arma::uvec x_prev = arma::find(candidates_i.col(0) < i);// % (candidates.col(1) == j) % (candidates.col(2) == h));
                parents(layern-1) = arma::zeros(6) -1;
                if(x_prev.n_elem > 0){
                  int x1 = arma::conv_to<int>::from(x_prev.tail(1));
                  parents(layern-1)(0) = candidates_i(x1, 3)-1;
                }
                
                arma::uvec x_aftr = arma::find(candidates_i.col(0) > i);// % (candidates.col(1) == j) % (candidates.col(2) == h));
                if(x_aftr.n_elem > 0){
                  int x2 = arma::conv_to<int>::from(x_aftr.head(1));
                  parents(layern-1)(1) = candidates_i(x2, 3)-1;
                }
                
                //arma::uvec y_prev = arma::find((candidates.col(0) == i) % (candidates.col(1) < j) % (candidates.col(2) == h));
                arma::uvec y_prev = arma::find(candidates_j.col(1) < j);
                
                if(y_prev.n_elem > 0){
                  int y1 = arma::conv_to<int>::from(y_prev.tail(1));
                  parents(layern-1)(2) = candidates_j(y1, 3)-1;
                }
                
                //arma::uvec y_aftr = arma::find((candidates.col(0) == i) % (candidates.col(1) > j) % (candidates.col(2) == h));
                arma::uvec y_aftr = arma::find(candidates_j.col(1) > j);
                if(y_aftr.n_elem > 0){
                  int y2 = arma::conv_to<int>::from(y_aftr.head(1));
                  parents(layern-1)(3) = candidates_j(y2, 3)-1;
                }
                
                //arma::uvec z_prev = arma::find((candidates.col(0) == i) % (candidates.col(1) == j) % (candidates.col(2) < h));
                arma::uvec z_prev = arma::find(candidates_h.col(2) < h);
                if(z_prev.n_elem > 0){
                  int z1 = arma::conv_to<int>::from(z_prev.tail(1));
                  parents(layern-1)(4) = candidates_h(z1, 3)-1;
                }
                
                arma::uvec z_aftr = arma::find(candidates_h.col(2) > h);
                if(z_aftr.n_elem > 0){
                  int z2 = arma::conv_to<int>::from(z_aftr.head(1));
                  parents(layern-1)(5) = candidates_h(z2, 3)-1;
                }
                
              }
            }
          }
        }
      }
    }
    end = std::chrono::steady_clock::now();
    
    //end = std::chrono::steady_clock::now();
    Rcpp::Rcout << "[part 2] "
                << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
                << "us.\n";
    
  }
  
  
  
  //***#pragma omp parallel for
  for(int i=0; i<parents.n_elem; i++){
    parents(i) = parents(i).elem(arma::find(parents(i) != -1));
    children(i) = children(i).elem(arma::find(children(i) != -1));
  }
  
  return Rcpp::List::create(
    Rcpp::Named("parents") = parents,
    Rcpp::Named("children") = children,
    Rcpp::Named("names") = lnames
  );
}
