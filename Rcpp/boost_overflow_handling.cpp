#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>


using namespace Rcpp;
using namespace arma;


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]


namespace myspace
{
using namespace boost::math::policies;
// Define a policy that does not throw on overflow:
typedef policy<overflow_error<errno_on_error> > my_policy;
// Define the special functions in this scope to use the policy:   
BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(my_policy)
}

// Now we can use myspace::cyl_bessel_i etc.
// They will automatically use "my_policy":



// [[Rcpp::export]]
double rcpp_besselI(double x, double nu, bool T_or_F){
  Environment base("package:base");
  Function besselI = base["besselI"];
  double out = as<double>(besselI(x, nu, T_or_F));
  return out;
}


// [[Rcpp::export]]
double rcpp_boost_besselI(double x, double nu, bool T_or_F){
  if(T_or_F == false){
    return (boost::math::cyl_bessel_i(nu, x));
  }else{
    return (exp(-x) * (boost::math::cyl_bessel_i(nu, x)));
  }
}


// [[Rcpp::export]]
double rcpp_boost_besselI_overflow_fixed(double x, double nu, bool T_or_F){   
  // using my_policy so that it does *not* throw on overflow
  //despite the large value unlike default policy boost::math::cyl_bessel_i
  
  if(T_or_F == false){
    return (myspace::cyl_bessel_i(nu, x));
  }else{
    return (exp(-x) * (myspace::cyl_bessel_i(nu, x)));
  }
}


// Writing log-BesselI function:

// [[Rcpp::export]]
double lrcpp_besselI(double x, double nu, bool T_or_F){
  Environment base("package:base");
  Function besselI = base["besselI"];
  double out = as<double>(besselI(x, nu, T_or_F));
  return log(out);
}


// [[Rcpp::export]]
double lrcpp_boost_besselI(double x, double nu, bool T_or_F){
  if(T_or_F == false){
    return log(boost::math::cyl_bessel_i(nu, x));
  }else{
    return (log(boost::math::cyl_bessel_i(nu, x)) - x);
  }
}


// [[Rcpp::export]]
double lrcpp_boost_besselI_overflow_fixed(double x, double nu, bool T_or_F){
  if(T_or_F == false){
    return log(myspace::cyl_bessel_i(nu, x));
  }else{
    return (log(myspace::cyl_bessel_i(nu, x)) - x);
  }
}


//====================================================================================
// Performance testing in R below:
//====================================================================================

/*** R
library(rstiefel)
library(microbenchmark)

besselI(5,500,T)
rcpp_besselI(5,500,T)
rcpp_boost_besselI(5,500,T)
rcpp_boost_besselI_overflow_fixed(5,500,T)

log(besselI(5,500,T))
lrcpp_besselI(5,500,T)
lrcpp_boost_besselI(5,500,T)
lrcpp_boost_besselI_overflow_fixed(5,500,T)

microbenchmark(besselI(5,500,T), log(besselI(5,500,T)), 
               rcpp_besselI(5,500,T), lrcpp_besselI(5,500,T), 
               rcpp_boost_besselI(5,500,T), lrcpp_boost_besselI(5,500,T), 
               rcpp_boost_besselI_overflow_fixed(5,500,T), lrcpp_boost_besselI_overflow_fixed(5,500,T)
               )

*/
