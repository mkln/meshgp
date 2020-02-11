#include "csplit.h"


// PARTITIONING

arma::uvec relabel(const arma::uvec& x, const arma::uvec& ux, arma::uvec& counts){
  //arma::uvec ux = arma::unique(x);
  int maxux = arma::max(ux);
  arma::uvec labs = -1+arma::zeros<arma::uvec>(maxux+1);
  int u=0;
  for(int i=0; i<ux.n_elem; i++){
    labs(ux(i)) = u;
    u++;
  }    
  arma::uvec re_x = x;
  for(int i=0; i<x.n_elem; i++){
    re_x(i) = labs(x(i));
    counts(labs(x(i))) += 1;
  }
  //Rcpp::Rcout << counts << endl;
  return re_x;   
}

arma::field<arma::uvec> csplit(const arma::uvec& x_orig){
  arma::uvec ux = arma::unique(x_orig);
  arma::uvec counts = arma::zeros<arma::uvec>(ux.n_elem);
  //Rcpp::Rcout << " ~ csplit :: relabel" << endl;
  arma::uvec x = relabel(x_orig, ux, counts);
  //Rcpp::Rcout << " ~ csplit :: chist" << endl;
  //arma::uvec counts = chist(x, ux.n_elem);
  
  //Rcpp::Rcout << " ~ csplit :: init" << endl;
  arma::field<arma::uvec> result(counts.n_elem);
  arma::uvec ix = arma::zeros<arma::uvec>(counts.n_elem);
  for(int i=0; i<result.n_elem; i++){
    result(i) = arma::zeros<arma::uvec>(counts(i));
  }
  
  //Rcpp::Rcout << " ~ csplit :: fill" << endl;
  for(int i=0; i<x.n_elem; i++){
    result(x(i))(ix(x(i))) = i;
    ix(x(i)) += 1;
  }
  return result;
}


arma::vec cpprange(int n){
  arma::vec rangetemp = arma::linspace(0, n, n+1);
  return 1.0/n * rangetemp.head(n).tail(n-1);
}

arma::umat thresholding(const arma::mat& coords, int ell){
  //arma::mat resultmat = arma::zeros(arma::size(coords));
  arma::vec thresholds = cpprange(ell);
  arma::umat cthresh = coords > thresholds(0);
  for(int i=1; i<thresholds.n_elem; i++){
    //Rcpp::Rcout << "~ thresholding " << i << endl;
    cthresh += coords > thresholds(i);
  }
  return cthresh;
}

arma::uvec ms_seq(const arma::mat& coords, int ell){
  //Rcpp::Rcout << "~ start partitioning... " << endl;
  //arma::mat resultmat = arma::zeros(arma::size(coords));
  arma::vec thresholds = cpprange(ell);//kthresholds(cja, Mv(j));
  
  //Rcpp::Rcout << "~ thresholding " << endl;
  arma::umat cthresh = thresholding(coords, ell);
  
  //Rcpp::Rcout << "~ interacting " << endl;
  int base = thresholds.n_elem+1;
  
  arma::uvec result = arma::zeros<arma::uvec>(cthresh.n_rows);
  for(int i=0; i<result.n_elem; i++){
    result(i) = 0;
    for(int j=0; j<cthresh.n_cols; j++){
      result(i) += cthresh(i,j) * pow(base, j);
    }
  }

  return result;
}

