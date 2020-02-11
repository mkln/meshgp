/*#include <RcppArmadillo.h>
#include <omp.h>

using namespace std;


//[[Rcpp::export]]
arma::field<arma::mat> turbosplit(const arma::mat& splitthis, const arma::vec& splitter){
  
  std::chrono::steady_clock::time_point start;
  std::chrono::steady_clock::time_point end;
  
  start = std::chrono::steady_clock::now();
  
  arma::vec uniques = arma::unique(splitter);
  arma::field<arma::mat> result(uniques.n_elem);
  
//#pragma omp parallel for
  for(int i=0; i<uniques.n_elem; i++){
    result(i) = splitthis.rows(arma::find(splitter == uniques(i)));
  }
  
  end = std::chrono::steady_clock::now();
  Rcpp::Rcout << "turbosplit "
              << std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
              << "us.\n";
  
  return result;
}
*/