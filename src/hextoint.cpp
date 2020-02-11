#include <RcppArmadillo.h>
#include <stdlib.h>
#include <math.h>

//[[Rcpp::export]]
int hex_to_dec(const std::string& hex_value){
  int decimal_value = 0;
  std::stringstream ss;
  ss << hex_value;
  ss >> std::hex >> decimal_value; 
  return decimal_value;
}

//[[Rcpp::export]]
std::vector<int> hex_to_dec_vec(const std::vector<std::string>& vec_hex){
  int len = vec_hex.size();
  std::vector<int> vec_dec(len);
  
  for( int i=0; i < len; i++ ) {
    vec_dec.at(i) =  hex_to_dec(vec_hex.at(i));
  }
  return vec_dec;
}