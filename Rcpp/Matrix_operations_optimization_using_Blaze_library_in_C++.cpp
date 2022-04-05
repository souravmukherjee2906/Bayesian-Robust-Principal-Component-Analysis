//====================================================================================
// This file deals with optimizations of matrix operations using Blaze, an open-source
// and high-performance C++ math library for dense and sparse arithmetic.
//====================================================================================

#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>

#include <truncnorm.h>  // It is a part of RcppDist.h header file.
#include <RcppTN.h>     // It is a part of RcppTN.h header file for 
                        // Truncated Normal distribution.
#include <Rfast.h>      // For optimized matrix and vector operations.


using namespace Rcpp;
using namespace arma;


// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppDist)]]
// [[Rcpp::depends(RcppTN)]]
// [[Rcpp::depends(Rfast)]]


//------------------------------------------------------------------------------------
// Defining a policy that does not throw overflow error when using Boost C++ libraries.
//------------------------------------------------------------------------------------

namespace myspace
{
using namespace boost::math::policies;

// Define a policy that does not throw on overflow:
typedef policy<overflow_error<errno_on_error> > my_policy;

// Define the special functions in this scope to use the policy:   
BOOST_MATH_DECLARE_SPECIAL_FUNCTIONS(my_policy)
}


//------------------------------------------------------------------------------------
// Functions for matrix operations without using any library from C++.
//------------------------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix matrix_multiply(NumericMatrix A, NumericMatrix B){
  int n = A.nrow(), r = A.ncol(), p = B.ncol();
  NumericMatrix out(n,p);
  for(int i=0; i<n; ++i){
    for(int j=0; j<p; ++j){
      double temp = 0;
      for(int k=0; k<r; ++k){
        temp = temp + (A(i,k) * B(k,j));
      }
      out(i,j) = temp;
    }
  }
  return out;
}


// [[Rcpp::export]]
NumericMatrix matrix_scalar_multiply(double k, NumericMatrix A){
  int n = A.nrow(), p = A.ncol();
  NumericMatrix out(n,p);
  for(int i=0; i<n; ++i){
    for(int j=0; j<p; ++j){
      out(i,j) = k*A(i,j);
    }
  }
  return out;
}


// [[Rcpp::export]]
NumericMatrix matrix_add(NumericMatrix A, NumericMatrix B){
  int n = A.nrow(), p = A.ncol();
  NumericMatrix out(n,p);
  for(int i=0; i<n; ++i){
    for(int j=0; j<p; ++j){
      out(i,j) = A(i,j) + B(i,j);
    }
  }
  return out;
}


//------------------------------------------------------------------------------------
// Functions for matrix operations using the Armadillo library in C++.
//------------------------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix matrix_multiply1(NumericMatrix A, NumericMatrix B){
  arma::mat Am = Rcpp::as< arma::mat >(A);
  arma::mat Bm = Rcpp::as< arma::mat >(B);
  return Rcpp::wrap( Am * Bm );
}


// [[Rcpp::export]]
arma::mat matrix_multiply2(arma::mat A, arma::mat B){
  return A*B;
}


// [[Rcpp::export]]
NumericMatrix matrix_3_multiply1(NumericMatrix A, NumericMatrix B, NumericMatrix C){
  arma::mat Am = Rcpp::as< arma::mat >(A);
  arma::mat Bm = Rcpp::as< arma::mat >(B);
  arma::mat Cm = Rcpp::as< arma::mat >(C);
  return Rcpp::wrap( Am * Bm * Cm);
}


// [[Rcpp::export]]
arma::mat matrix_3_multiply2(arma::mat A, arma::mat B, arma::mat C){
return A*B*C;  
}


// [[Rcpp::export]]
NumericMatrix matrix_scalar_multiply1(double k, NumericMatrix A){
  arma::mat Am = Rcpp::as< arma::mat >(A);
  return Rcpp::wrap(k*Am);
}


// [[Rcpp::export]]
NumericMatrix matrix_add1(NumericMatrix A, NumericMatrix B){
  arma::mat Am = Rcpp::as< arma::mat >(A);
  arma::mat Bm = Rcpp::as< arma::mat >(B);
  return Rcpp::wrap(Am + Bm);
}


//------------------------------------------------------------------------------------
// Function for matrix multiplication using the 'Rfast' package in R.
//------------------------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix rcpp_mat_mult(NumericMatrix A, NumericMatrix B){ 
  Environment Rfast("package:Rfast");
  Function mat_mult = Rfast["mat.mult"];
  NumericMatrix out = mat_mult(A,B);
  return out;
}


//------------------------------------------------------------------------------------
// Lines 158-161 contains the fastest matrix multiplication algorithm among all of the 
// above.
// It directly uses the functions from the Rfast.h header file in the Blaze C++ library.
//------------------------------------------------------------------------------------

// [[Rcpp::export]]
NumericMatrix mat_mult(NumericMatrix A, NumericMatrix B){
return Rfast::matrix::matrix_multiplication(A,B);
}


// [[Rcpp::export]]
NumericMatrix trans(NumericMatrix A){
  return Rfast::matrix::transpose(A);
}



//====================================================================================
// Performance testing in R below:
//====================================================================================

/*** R

library(microbenchmark)
library(Rfast)
library(tictoc)


A <- matrix(rnorm(3000*150), nrow = 3000)
B <- matrix(rnorm(3000*3000), nrow = 3000)
C <- matrix(rnorm(3000*150), nrow = 3000)


tic()
D <- matrix_multiply(matrix_multiply(trans(A),B), C)
toc()

tic()
E <- mat_mult(mat_mult(trans(A),B), C)
toc()

tic()
M <- t(A)%*%B%*%C
toc()


microbenchmark(A%*%B, 
               matrix_multiply(A,B), 
               matrix_multiply1(A,B), 
               matrix_multiply2(A,B), 
               rcpp_mat_mult(A,B), 
               mat_mult(A,B)
               )

microbenchmark(t(A), 
               transpose(A), 
               trans(A)
               )
               
microbenchmark(matrix_add(A,B), 
               matrix_add1(A,B)
               )
               
microbenchmark(matrix_scalar_multiply(50,A), 
               matrix_scalar_multiply1(50,A)
               )
               
microbenchmark(A%*%B%*%C, 
               matrix_multiply(matrix_multiply(A,B),C), 
               matrix_3_multiply1(A,B,C), 
               matrix_3_multiply2(A,B,C), 
               mat_mult(mat_mult(A,B),C)
               )
*/
