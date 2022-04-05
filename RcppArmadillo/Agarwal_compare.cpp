#define ARMA_64BIT_WORD
#include <RcppArmadillo.h>
#include <algorithm>
#include <boost/math/special_functions/bessel.hpp>

#include <truncnorm.h>      //It is a part of RcppDist.h  header file

#include <RcppTN.h>           // It is a part of RcppTN.h header file for Truncated Normal distribution specially



using namespace Rcpp;
using namespace arma;

// [[Rcpp::depends(RcppArmadillo)]]
// [[Rcpp::depends(BH)]]
// [[Rcpp::depends(RcppDist)]]
// [[Rcpp::depends(RcppTN)]]


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
NumericMatrix diagmatrix (NumericVector v){
  int r = v.length();
  NumericMatrix out(r,r);
  for(int i=0; i<r; ++i){
    out(i,i) = v[i];
  }
  return out;
}


// [[Rcpp::export]]
NumericMatrix rcpp_only_diag_cumprod(NumericVector v){
  int r = v.length();
  NumericVector cump_rev = cumprod(rev(v));
  NumericVector rev_cump_rev = rev(cump_rev);
  NumericMatrix out(r); //equivalent to matrix(0, nrow = 2, ncol = 2)
  for(int i=0; i<r; ++i){
    out(i,i) = rev_cump_rev[i];
  }
  return out;
}


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
NumericMatrix rcpp_only_rustiefel(int n, int r){ 
  Environment rstiefel("package:rstiefel");
  Function rustiefel = rstiefel["rustiefel"];
  NumericMatrix out = rustiefel(n,r);
  return out;
}


// Writing SVD function:

// [[Rcpp::export]]
List rcpp_svd(NumericMatrix A){ 
  Environment base("package:base");
  Function svd = base["svd"];
  List out = svd(A);
  return out;
}


// [[Rcpp::export]]
List arma_svd(NumericMatrix A){
  mat A1(A.begin(), A.nrow(), A.ncol(), false);
  mat U1, V1;
  vec d1;
  svd(U1,d1,V1,A1);
  return List::create(_["d"] = wrap(d1), _["u"] = wrap(U1), _["v"] = wrap(V1));
}


// Writing NullC function:

// [[Rcpp::export]]
List arma_qr(NumericMatrix A){
  mat A1(A.begin(), A.nrow(), A.ncol(), false);
  mat Q,R;
  qr(Q,R,A1);
  return List::create(_["Q"] = wrap(Q), _["R"] = wrap(R));
}

// [[Rcpp::export]]
NumericMatrix arma_NullC(NumericMatrix M){
  int n = M.nrow(), p = M.ncol();
  mat M1(M.begin(), n, p, false);
  int rank_M = rank(M1);
  NumericMatrix M_Q = arma_qr(M)["Q"];
  // int set;
  if(rank_M == 0){
    //  set = n - 1;
    return (M_Q(_ , Range(0, n-1)));
  }else{
    //  set = rank(M1);
    if(rank_M < n){
      return (M_Q(_ , Range(rank_M, n - 1)));
    }else{
      NumericMatrix out(n, 0);
      return out;
    }
  }
}


// Writing BesselI function:

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



// Writing rmf.vector function:

// [[Rcpp::export]]
double rcpp_rW(double kap, int m){
  Environment rstiefel("package:rstiefel");
  Function rW = rstiefel["rW"];
  double out = as<double>(rW(kap, m));
  return out;
}

// [[Rcpp::export]]
NumericMatrix rmf_vector1(NumericVector kmu){
  double kap = sqrt(sum(pow(kmu, 2.0)));
  NumericVector mu = kmu/kap;
  int m = mu.length();
  NumericMatrix u(m, 1);
  if(kap==0){
    u(_, 0) = rnorm(m, 0, 1);
    u(_, 0) = u(_, 0)/sqrt(sum(pow(u(_, 0), 2.0)));
    // NumericMatrix u1(m, 1, u.begin());
  }else{
    if(m==1){
      double rb = rbinom(1, 1, 1/(1 + exp(2*kap*mu[0])))[0];
      u(0, 0) = pow((double)(-1), rb);
    }else{
      double W = rcpp_rW(kap, m);
      NumericVector V = rnorm(m-1, 0, 1);
      V = V/sqrt(sum(pow(V, 2.0)));
      V = pow(1 - pow(W, 2.0), 0.5) * V;
      //  NumericVector V = pow(1 - pow(W, 2.0), 0.5) * (rnorm(m-1, 0, 1)/sqrt(sum(pow(rnorm(m-1, 0, 1), 2.0))));
      V.push_back(W);
      //    NumericVector x = V;
      
      V.attr("dim") = Dimension(m, 1);
      mu.attr("dim") = Dimension(m, 1);
      
      /*    NumericMatrix cbind_nullc_mu(m, m-1);
       NumericMatrix mu_matrix = as<NumericMatrix>(mu);
       cbind_nullc_mu = rcpp_NullC(mu_matrix);
       cbind_nullc_mu.push_back(mu);*/
      
      NumericMatrix cbind_nullc_mu = cbind(arma_NullC(as<NumericMatrix>(mu)) , mu);
      
      //     u = matrix_multiply(cbind_nullc_mu, as<NumericMatrix>(x));
      u = matrix_multiply(cbind_nullc_mu, as<NumericMatrix>(V));
    }
  }
  return u;
}


// Writing rmf.matrix function:



// [[Rcpp::export]]
NumericMatrix rmf_matrix4 (NumericMatrix M){
  int m = M.nrow(), p = M.ncol();
  NumericMatrix X(m,p);
  if (p == 1){
    X = rmf_vector1(M(_, 0));
  }
  if(p > 1){
    List svdM = rcpp_svd(M);  
    NumericVector d = svdM["d"];
    NumericMatrix u = svdM["u"];
    NumericMatrix v = svdM["v"];
    
    NumericMatrix H = matrix_multiply(u, diagmatrix(d));
    int R = H.ncol();
    bool cmet = false;
    int rej = 0;
    NumericMatrix U(m,R);
    while(!cmet){
      U(_, 0) = rmf_vector1(H(_,0));
      double lr = 0;
      for(int j=1; j< R; ++j){
        NumericMatrix N = arma_NullC(U(_, Range(0,j-1)));
        
        /*   NumericVector Hj = H(_,j);
         Hj.attr("dim") = Dimension(m,1);
         
         NumericMatrix x_mat = matrix_multiply(transpose(N), as<NumericMatrix>(Hj)); 
         */
        int N_col = N.ncol();
        NumericVector x_vec(N_col);
        for(int k=0; k < N_col; ++k){
          double temp = 0;
          for(int i=0; i<m; ++i){
            temp = temp + (N(i,k)*H(i,j));
          }
          x_vec[k] = temp;
        }
        
        NumericMatrix x = rmf_vector1(x_vec);
        U(_, j) = matrix_multiply(N, x);
        if(d[j] > 0){                                           
          double xn = sqrt(sum(pow(x_vec, 2.0)));
          double xd = sqrt(sum(pow(H(_, j), 2.0)));
          //double lbr = log(rcpp_boost_besselI(xn, 0.5*(m-j-2), true)) - log(rcpp_boost_besselI(xd, 0.5*(m-j-2), true));   //??
          double lbr = lrcpp_boost_besselI_overflow_fixed(xn, 0.5*(m-j-2), true) - lrcpp_boost_besselI_overflow_fixed(xd, 0.5*(m-j-2), true);
          
          
          NumericVector lbr_vec(1);                                             //??
          lbr_vec[0] = lbr;
          if(all(is_na(lbr_vec)) == true){
            lbr = 0.5 * (log(xd) - log(xn));
          }
          
          lr = lr + lbr + (xn - xd) + (0.5*(m-j-2)*(log(xd) - log(xn)));         //??
        }
      }
      cmet = (log(runif(1,0,1)[0]) < lr);
      rej = rej + (1 - 1*cmet);   
    }
    X = matrix_multiply(U , transpose(v));
  }
  return X;
}


// [[Rcpp::export]]
NumericMatrix rcpp_only_rmf_matrix(NumericMatrix M){ 
  Environment rstiefel("package:rstiefel");
  Function rmf_matrix = rstiefel["rmf.matrix"];
  NumericMatrix out = rmf_matrix(M);
  return out;
}


// [[Rcpp::export]]
double rcpp_only_rtruncnorm(double a, double b, double mean, double sd){
  Environment truncnorm("package:truncnorm"); 
  Function rtruncnorm = truncnorm["rtruncnorm"];
  SEXP out1 = rtruncnorm(1, a, b, mean, sd);
  double out = Rcpp::as<double>(out1);
  return out;   
}


// [[Rcpp::export]]
double rcppdist_rtruncnorm(double a, double b, double mean, double sd){
  return r_truncnorm(mean, sd, a, b);
}


// [[Rcpp::export]]
double rcpptn_rtruncnorm(double a, double b, double mean, double sd){
  return (RcppTN::rtn1(mean, sd, a, b));
}




// [[Rcpp::export]]
double rcpp_only_rinvgamma(double shape, double rate){ 
  Environment invgamma("package:invgamma"); 
  Function rinvgamma = invgamma["rinvgamma"];
  SEXP out1 = rinvgamma(Named("n", 1), Named("shape", shape), Named("rate", rate));
  double out = as<double>(out1);
  return out;   
}


// [[Rcpp::export]]
double rcpp_only_invgamma(double shape, double rate){
  double scale = 1/rate;
  double out_gamma = Rcpp::rgamma(1, shape, scale)[0];
  double out = 1/(out_gamma);
  return out;
}






// [[Rcpp::export]]
NumericVector rcpp_only_d_update_mat(NumericVector d, NumericMatrix A, double Sigma2, int n, int p, int r) {
  //RNGScope scope; ??
  for(int i = 0; i < r; ++i){
    
    NumericMatrix D_cap(r);
    double num = 0;
    double denom = 0;
    
    for(int j=0; j< r; ++j){
      if(j <= i){
        D_cap(j,j) = 1;
        for(int l=j; l< r; ++l){
          if(l==i){
            D_cap(j,j) = D_cap(j,j);
          }else{
            D_cap(j,j) = D_cap(j,j) * d[l];
          }
        }
      }
      
      num = num + (D_cap(j,j)*A(j,j));
      denom = denom + pow(D_cap(j,j), 2.0);
    }
    
    denom = denom + Sigma2;
    
    
    if (i < r-1){
      d[i] = rcpp_only_rtruncnorm(1, R_PosInf, num/denom, sqrt(Sigma2/denom));
    }
    else{
      d[i] = rcpp_only_rtruncnorm(0, R_PosInf, num/denom, sqrt(Sigma2/denom));
    }
    
  }
  return d;
}



// [[Rcpp::export]]
NumericVector rcpptn_d_update_mat(NumericVector d, NumericMatrix A, double Sigma2, int n, int p, int r) {
  //RNGScope scope; ??
  for(int i = 0; i < r; ++i){
    
    NumericMatrix D_cap(r);
    double num = 0;
    double denom = 0;
    
    for(int j=0; j< r; ++j){
      if(j <= i){
        D_cap(j,j) = 1;
        for(int l=j; l< r; ++l){
          if(l==i){
            D_cap(j,j) = D_cap(j,j);
          }else{
            D_cap(j,j) = D_cap(j,j) * d[l];
          }
        }
      }
      
      num = num + (D_cap(j,j)*A(j,j));
      denom = denom + pow(D_cap(j,j), 2.0);
    }
    
    denom = denom + Sigma2;
    
    if (i < r-1){
      d[i] = rcpptn_rtruncnorm(1, R_PosInf, num/denom, sqrt(Sigma2/denom));
    }
    else{
      d[i] = rcpptn_rtruncnorm(0, R_PosInf, num/denom, sqrt(Sigma2/denom));
    }
    
  }
  return d;
}





// creating the function for 1:burn_in below:



// [[Rcpp::export]]
List Simul_burnin4 (NumericMatrix Y, NumericMatrix U, NumericMatrix U2, NumericMatrix D, NumericMatrix D2, NumericMatrix V, NumericMatrix V2, NumericMatrix S, NumericMatrix S2, NumericVector d, NumericVector d2, double Sigma2, double Sigma22, double tow2, double q1, double q2, int n, int p, int r, int burn_in, double a, double b){
  int c = 0;
  
  NumericMatrix L(n,p);
  NumericMatrix L2(n,p);
  NumericMatrix S_count(n,p);
  NumericMatrix S_count2(n,p);
  
  for(int count = 0; count < burn_in; ++count){
    
    //double inv_Sigma2 = 1/Sigma2;
    //double inv_Sigma22 = 1/Sigma22;
    
    // Update the value of U for Method 1 and 2
    NumericMatrix U11(n,r);
    NumericMatrix U22(n,r);
    for(int i=0; i <n; ++i){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        double temp2 = 0;
        for(int j=0; j<p; ++j){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*V(j,l)*D(l,l))/Sigma2);
          temp2 = temp2 + (((Y(i,j) - S2(i,j))*V2(j,l)*D2(l,l))/Sigma22);
        }
        U11(i,l) = temp1;
        U22(i,l) = temp2;
      }
    }
    
    U = rmf_matrix4(U11);
    //    Rcout << "The value of U : \n" << U << "\n";
    U2 = rmf_matrix4(U22);
    //    Rcout << "The value of U2 : \n" << U2 << "\n";
    
    // Update the value of V for Method 1 and 2
    NumericMatrix V11(p,r);
    NumericMatrix V22(p,r);
    for(int j=0; j <p; ++j){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        double temp2 = 0;
        for(int i=0; i<n; ++i){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*U(i,l)*D(l,l))/Sigma2);
          temp2 = temp2 + (((Y(i,j) - S2(i,j))*U2(i,l)*D2(l,l))/Sigma22);
        }
        V11(j,l) = temp1;
        V22(j,l) = temp2;
      }
    }
    
    V = rmf_matrix4(V11);
    //    Rcout << "The value of V : \n" << V << "\n";
    V2 = rmf_matrix4(V22);
    //    Rcout << "The value of V2 : \n" << V2 << "\n";
    
    
    // Update the value of d one at a time for Method 1 and 2
    d = rcpp_only_d_update_mat(clone(d), matrix_multiply(transpose(matrix_multiply(matrix_add(Y, matrix_scalar_multiply(-1, S)), V)), U), Sigma2, n, p, r);
    //    Rcout << "The value of d : \n" << d << "\n";
    d2 = rcpp_only_d_update_mat(clone(d2), matrix_multiply(transpose(matrix_multiply(matrix_add(Y, matrix_scalar_multiply(-1, S2)), V2)), U2), Sigma22, n, p, r);
    //    Rcout << "The value of d2 : \n" << d2 << "\n";
    
    // Once we got the updated set of elements in vector d, it's time to Update D for both Methods
    D = rcpp_only_diag_cumprod(d);
    //    Rcout << "The value of D : \n" << D << "\n";
    D2 = rcpp_only_diag_cumprod(d2);
    //    Rcout << "The value of D2 : \n" << D2 << "\n";
    
    //    Rcout << "The value of L : \n" << L << "\n";
    //    Rcout << "The value of L2 : \n" << L2 << "\n";
    
    
    
    // Update the value of L and then S for Method 1 and 2
    NumericMatrix S_rnorm(n, p);
    NumericMatrix S_runif(n, p);
    NumericMatrix S_rnorm2(n, p);
    for(int j=0; j < p; ++j){
      S_rnorm(_ , j) = Rcpp::rnorm(n, 0, sqrt((tow2 * Sigma2)/(tow2 + Sigma2)));
      S_runif(_ , j) = Rcpp::runif(n, 0, 1);
      S_rnorm2(_ , j) = Rcpp::rnorm(n, 0, sqrt((tow2 * Sigma22)/(tow2 + Sigma22)));
    }
    
    // NumericMatrix S_mu_rnorm(n,p);
    // NumericMatrix Q_star_matrix(n, p);
    NumericMatrix S_MCC(n,p);
    // NumericMatrix S_mu_rnorm2(n,p);
    // NumericMatrix Q_star_matrix2(n,p);
    NumericMatrix S_MCC2(n,p);
    
    double rate = 0;
    double rate2 = 0;
    
    for(int i=0; i<n; ++i){
      for(int j=0; j<p; ++j){
        
        // It is needed for updating L
        double temp1 = 0;
        double temp2 = 0;
        for(int l=0; l<r; ++l){
          temp1 = temp1 + (U(i,l)*D(l,l)*V(j,l));
          temp2 = temp2 + (U2(i,l)*D2(l,l)*V2(j,l));
        }
        L(i,j) = temp1;
        L2(i,j) = temp2;    
        
        // It is needed for updating S
        // S_mu_rnorm(i,j) = ((tow2/(tow2 + Sigma2))*(Y(i,j) - L(i,j))) + S_rnorm(i,j);
        // S_mu_rnorm2(i,j) = ((tow2/(tow2 + Sigma22))*(Y(i,j) - L2(i,j))) + S_rnorm2(i,j);
        
        // Q_star_matrix(i,j) = q1/(q1 + (((1-q1) * sqrt(Sigma2) * exp((tow2 * pow(Y(i,j) - L(i,j), 2.0))/(2 * Sigma2 * (tow2 + Sigma2))))/sqrt(tow2 + Sigma2)));
        // Q_star_matrix2(i,j) = q2/(q2 + (((1-q2) * sqrt(Sigma22) * exp((tow2 * pow(Y(i,j) - L2(i,j), 2.0))/(2 * Sigma22 * (tow2 + Sigma22))))/sqrt(tow2 + Sigma22)));
        
        if(S_runif(i,j) > (q1/(q1 + (((1-q1) * sqrt(Sigma2) * exp((tow2 * pow(Y(i,j) - L(i,j), 2.0))/(2 * Sigma2 * (tow2 + Sigma2))))/sqrt(tow2 + Sigma2))))){
          S_MCC(i,j) = 1;
          S_count(i,j) = 0;
        }else{
          S_MCC(i,j) = 0;
          S_count(i,j) = 1;
        }
        
        if(S_runif(i,j) > (q2/(q2 + (((1-q2) * sqrt(Sigma22) * exp((tow2 * pow(Y(i,j) - L2(i,j), 2.0))/(2 * Sigma22 * (tow2 + Sigma22))))/sqrt(tow2 + Sigma22))))){
          S_MCC2(i,j) = 1;
          S_count2(i,j) = 0;
        }else{
          S_MCC2(i,j) = 0;
          S_count2(i,j) = 1;
        }
        
        S(i,j) = (((tow2/(tow2 + Sigma2))*(Y(i,j) - L(i,j))) + S_rnorm(i,j)) * S_MCC(i,j);
        S2(i,j) = (((tow2/(tow2 + Sigma22))*(Y(i,j) - L2(i,j))) + S_rnorm2(i,j)) * S_MCC2(i,j);
        
        rate = rate + pow(Y(i,j) - L(i,j) - S(i,j), 2.0);
        rate2 = rate2 + pow(Y(i,j) - L2(i,j) - S2(i,j), 2.0);
      }
    }
    
    //    Rcout << "The value of S : \n" << S << "\n";
    //    Rcout << "The value of S2 : \n" << S2 << "\n";
    // Update the value of Sigma2 for Method 1 and 2
    //    double shape = (n * p * 0.5) + a;
    //    rate = b + (0.5 * rate);
    //    rate2 = b + (0.5 * rate2);
    Sigma2 = rcpp_only_rinvgamma((n * p * 0.5) + a, b + (0.5 * rate));
    Sigma22 = rcpp_only_rinvgamma((n * p * 0.5) + a, b + (0.5 * rate2));
    //    Rcout << "The value of Sigma2 : \n" << Sigma2 << "\n";
    //    Rcout << "The value of Sigma22 : \n" << Sigma22 << "\n";
    
    c = c + 1;
    
  }
  
  return List::create(_["L"] = L, _["L2"] = L2, _["U"] = U, _["U2"] = U2, _["V"] = V, _["V2"] = V2, _["S"] = S, _["S2"] = S2, _["S_count"] = S_count, _["S_count2"] = S_count2, _["d"] = d, _["d2"] = d2, _["D"] = D, _["D2"] = D2, _["Sigma2"] = Sigma2, _["Sigma22"] = Sigma22, _["c"] = c);
}




// [[Rcpp::export]]
List Simul_burnin4_tn (NumericMatrix Y, NumericMatrix U, NumericMatrix D, NumericMatrix V, NumericMatrix S, NumericVector d, double Sigma2, double tow2, double q1, int n, int p, int r, int burn_in, double a, double b){
  
  NumericMatrix L(n,p);
  NumericMatrix S_count(n,p);
  
  for(int count = 0; count < burn_in; ++count){
    
    // Update the value of U for Method 1
    NumericMatrix U11(n,r);
    for(int i=0; i <n; ++i){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        for(int j=0; j<p; ++j){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*V(j,l)*D(l,l))/Sigma2);
        }
        U11(i,l) = temp1;
      }
    }
    
    U = rmf_matrix4(U11);
    
    // Update the value of V for Method 1
    NumericMatrix V11(p,r);
    for(int j=0; j <p; ++j){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        for(int i=0; i<n; ++i){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*U(i,l)*D(l,l))/Sigma2);
        }
        V11(j,l) = temp1;
      }
    }
    
    V = rmf_matrix4(V11);
    
    // Update the value of d one at a time for Method 1
    d = rcpptn_d_update_mat(clone(d), matrix_multiply(transpose(matrix_multiply(matrix_add(Y, matrix_scalar_multiply(-1, S)), V)), U), Sigma2, n, p, r);
    
    // Once we got the updated set of elements in vector d, it's time to Update D
    D = rcpp_only_diag_cumprod(d);
    
    // Update the value of L and then S for Method 1
    NumericMatrix S_rnorm(n, p);
    NumericMatrix S_runif(n, p);
    for(int j=0; j < p; ++j){
      S_rnorm(_ , j) = Rcpp::rnorm(n, 0, sqrt((tow2 * Sigma2)/(tow2 + Sigma2)));
      S_runif(_ , j) = Rcpp::runif(n, 0, 1);
    }
    
    NumericMatrix S_MCC(n,p);
    
    double rate = 0;
    
    for(int i=0; i<n; ++i){
      for(int j=0; j<p; ++j){
        
        // It is needed for updating L
        double temp1 = 0;
        for(int l=0; l<r; ++l){
          temp1 = temp1 + (U(i,l)*D(l,l)*V(j,l));
        }
        L(i,j) = temp1;
        
        if(S_runif(i,j) > (q1/(q1 + (((1-q1) * sqrt(Sigma2) * exp((tow2 * pow(Y(i,j) - L(i,j), 2.0))/(2 * Sigma2 * (tow2 + Sigma2))))/sqrt(tow2 + Sigma2))))){
          S_MCC(i,j) = 1;
          S_count(i,j) = 0;
        }else{
          S_MCC(i,j) = 0;
          S_count(i,j) = 1;
        }
        
        S(i,j) = (((tow2/(tow2 + Sigma2))*(Y(i,j) - L(i,j))) + S_rnorm(i,j)) * S_MCC(i,j);
        
        rate = rate + pow(Y(i,j) - L(i,j) - S(i,j), 2.0);
      }
    }
    
    Sigma2 = rcpp_only_invgamma((n * p * 0.5) + a, b + (0.5 * rate));
  }
  
  return List::create(_["L"] = L, _["U"] = U, _["V"] = V, _["S"] = S, _["S_count"] = S_count, _["d"] = d, _["D"] = D, _["Sigma2"] = Sigma2);
}



//After Burn in:


// [[Rcpp::export]]
List Simul_after_burnin4(NumericMatrix Y, NumericMatrix U, NumericMatrix U2, NumericMatrix D, NumericMatrix D2, NumericMatrix V, NumericMatrix V2, NumericMatrix L, NumericMatrix L2, NumericMatrix S, NumericMatrix S2, NumericMatrix S_count, NumericMatrix S_count2, NumericVector d, NumericVector d2, NumericVector d_star, double Sigma2, double Sigma22, double tow2, double q1, double q2, int n, int p, int r, int K, int burn_in, double a, double b){
  int c1 = 0;
  int c2 = 0;
  
  NumericVector d_sum = clone(d);
  NumericVector d_sum2 = clone(d2);
  NumericVector d_bar(r);
  NumericVector d_bar2(r);
  NumericMatrix L_sum = clone(L);
  NumericMatrix L_sum2 = clone(L2);
  NumericMatrix S_sum = clone(S);
  NumericMatrix S_sum2 = clone(S2);
  double Sigma2_sum = Sigma2;
  double Sigma2_sum2 = Sigma22;
  
  List dbar_list((K - burn_in)+1);
  dbar_list[0] = clone(d);
  List dbar_list2((K - burn_in)+1);
  dbar_list2[0] = clone(d2);
  
  NumericMatrix L_hat(n,p);
  NumericMatrix L_hat2(n,p);
  double Sigma2_bar = 0;
  double Sigma2_bar2 = 0;
  double distance_d = 0;
  double distance_d2 = 0;
  double distance_Sigma2 = 0;
  double distance_Sigma22 = 0;
  
  for(int count = 0; count < (K - burn_in); ++count){
    
    // Update the value of U for Method 1 and 2
    NumericMatrix U11(n,r);
    NumericMatrix U22(n,r);
    for(int i=0; i <n; ++i){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        double temp2 = 0;
        for(int j=0; j<p; ++j){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*V(j,l)*D(l,l))/Sigma2);
          temp2 = temp2 + (((Y(i,j) - S2(i,j))*V2(j,l)*D2(l,l))/Sigma22);
        }
        U11(i,l) = temp1;
        U22(i,l) = temp2;
      }
    }
    
    U = rmf_matrix4(U11);
    //    Rcout << "The value of U : \n" << U << "\n";
    U2 = rmf_matrix4(U22);
    //    Rcout << "The value of U2 : \n" << U2 << "\n";
    
    // Update the value of V for Method 1 and 2
    NumericMatrix V11(p,r);
    NumericMatrix V22(p,r);
    for(int j=0; j <p; ++j){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        double temp2 = 0;
        for(int i=0; i<n; ++i){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*U(i,l)*D(l,l))/Sigma2);
          temp2 = temp2 + (((Y(i,j) - S2(i,j))*U2(i,l)*D2(l,l))/Sigma22);
        }
        V11(j,l) = temp1;
        V22(j,l) = temp2;
      }
    }
    
    V = rmf_matrix4(V11);
    //    Rcout << "The value of V : \n" << V << "\n";
    V2 = rmf_matrix4(V22);
    //    Rcout << "The value of V2 : \n" << V2 << "\n";
    
    // Update the value of d one at a time for Method 1 and 2
    d = rcpp_only_d_update_mat(clone(d), matrix_multiply(transpose(matrix_multiply(matrix_add(Y, matrix_scalar_multiply(-1, S)), V)), U), Sigma2, n, p, r);
    //    Rcout << "The value of d : \n" << d << "\n";
    d2 = rcpp_only_d_update_mat(clone(d2), matrix_multiply(transpose(matrix_multiply(matrix_add(Y, matrix_scalar_multiply(-1, S2)), V2)), U2), Sigma22, n, p, r);
    //    Rcout << "The value of d2 : \n" << d2 << "\n";
    
    // Once we got the updated set of elements in vector d, it's time to Update D for both Methods
    D = rcpp_only_diag_cumprod(d);
    //    Rcout << "The value of D : \n" << D << "\n";
    D2 = rcpp_only_diag_cumprod(d2);
    //    Rcout << "The value of D2 : \n" << D2 << "\n";
    
    // Update the value of L and then S for Method 1 and 2
    NumericMatrix S_rnorm(n, p);
    NumericMatrix S_runif(n, p);
    NumericMatrix S_rnorm2(n, p);
    for(int j=0; j < p; ++j){
      S_rnorm(_ , j) = Rcpp::rnorm(n, 0, sqrt((tow2 * Sigma2)/(tow2 + Sigma2)));
      S_runif(_ , j) = Rcpp::runif(n, 0, 1);
      S_rnorm2(_ , j) = Rcpp::rnorm(n, 0, sqrt((tow2 * Sigma22)/(tow2 + Sigma22)));
    }
    
    
    NumericMatrix S_MCC(n,p);
    NumericMatrix S_MCC2(n,p);
    
    double rate = 0;
    double rate2 = 0;
    
    for(int i=0; i<n; ++i){
      for(int j=0; j<p; ++j){
        
        // It is needed for updating L
        double temp1 = 0;
        double temp2 = 0;
        for(int l=0; l<r; ++l){
          temp1 = temp1 + (U(i,l)*D(l,l)*V(j,l));
          temp2 = temp2 + (U2(i,l)*D2(l,l)*V2(j,l));
        }
        L(i,j) = temp1;
        L2(i,j) = temp2;    
        
        // Update the value of L_sum and L_hat for Method 1 and 2
        L_sum(i,j) = L_sum(i,j) + L(i,j);
        L_sum2(i,j) = L_sum2(i,j) + L2(i,j);
        L_hat(i,j) = ((double)1/(count + 2)) * L_sum(i,j);
        L_hat2(i,j) = ((double)1/(count + 2)) * L_sum2(i,j);
        
        if(S_runif(i,j) > (q1/(q1 + (((1-q1) * sqrt(Sigma2) * exp((tow2 * pow(Y(i,j) - L(i,j), 2.0))/(2 * Sigma2 * (tow2 + Sigma2))))/sqrt(tow2 + Sigma2))))){
          S_MCC(i,j) = 1;
          S_count(i,j) = S_count(i,j);
        }else{
          S_MCC(i,j) = 0;
          S_count(i,j) = S_count(i,j) + 1;
        }
        
        if(S_runif(i,j) > (q2/(q2 + (((1-q2) * sqrt(Sigma22) * exp((tow2 * pow(Y(i,j) - L2(i,j), 2.0))/(2 * Sigma22 * (tow2 + Sigma22))))/sqrt(tow2 + Sigma22))))){
          S_MCC2(i,j) = 1;
          S_count2(i,j) = S_count2(i,j);
        }else{
          S_MCC2(i,j) = 0;
          S_count2(i,j) = S_count2(i,j) + 1;
        }
        
        S(i,j) = (((tow2/(tow2 + Sigma2))*(Y(i,j) - L(i,j))) + S_rnorm(i,j)) * S_MCC(i,j);
        S2(i,j) = (((tow2/(tow2 + Sigma22))*(Y(i,j) - L2(i,j))) + S_rnorm2(i,j)) * S_MCC2(i,j);
        
        // Update the value of S_sum for Method 1 and 2
        S_sum(i,j) = S_sum(i,j) + S(i,j);
        S_sum2(i,j) = S_sum2(i,j) + S2(i,j);
        
        rate = rate + pow(Y(i,j) - L(i,j) - S(i,j), 2.0);
        rate2 = rate2 + pow(Y(i,j) - L2(i,j) - S2(i,j), 2.0);
      }
    }
    
    //    Rcout << "The value of S : \n" << S << "\n";
    //    Rcout << "The value of S2 : \n" << S2 << "\n";
    // Update the value of Sigma2 for Method 1 and 2
    //    double shape = (n * p * 0.5) + a;
    //    rate = b + (0.5 * rate);
    //    rate2 = b + (0.5 * rate2);
    Sigma2 = rcpp_only_rinvgamma((n * p * 0.5) + a, b + (0.5 * rate));
    Sigma22 = rcpp_only_rinvgamma((n * p * 0.5) + a, b + (0.5 * rate2));
    //    Rcout << "The value of Sigma2 : \n" << Sigma2 << "\n";
    //    Rcout << "The value of Sigma22 : \n" << Sigma22 << "\n";
    
    
    // Update d_sum and d_bar for both Methods
    d_sum = d_sum + d;  // Rcpp only allows element-wise arithmetic operatiosn for vectors, not for matrices.
    //    Rcout << "The value of d_sum : \n" << d_sum << "\n";
    d_sum2 = d_sum2 + d2;
    //    Rcout << "The value of d_sum2 : \n" << d_sum2 << "\n";
    
    d_bar = d_sum/(count + 2);
    //    Rcout << "The value of d_bar : \n" << d_bar << "\n";
    d_bar2 = d_sum2/(count + 2);
    //    Rcout << "The value of d_bar2 : \n" << d_bar2 << "\n";
    
    
    // Update Sigma2_sum and Sigma2_bar for both methods
    Sigma2_sum = Sigma2_sum + Sigma2;
    //    Rcout << "The value of Sigma2_sum : \n" << Sigma2_sum << "\n";
    Sigma2_sum2 = Sigma2_sum2 + Sigma22;
    //    Rcout << "The value of Sigma2_sum2 : \n" << Sigma2_sum2 << "\n";
    
    Sigma2_bar = Sigma2_sum/(count + 2);
    //    Rcout << "The value of Sigma2_bar : \n" << Sigma2_bar << "\n";
    Sigma2_bar2 = Sigma2_sum2/(count + 2);
    //    Rcout << "The value of Sigma2_bar2 : \n" << Sigma2_bar2 << "\n";
    
    
    // Update dbar_list for both methods
    dbar_list[count + 1] = clone(d_bar);
    dbar_list2[count + 1] = clone(d_bar2);
    
    
    // Update the distances for both methods
    distance_d = max(abs(d_bar - d_star));
    //    Rcout << "The value of distance_d : \n" << distance_d << "\n";
    distance_d2 = max(abs(d_bar2 - d_star));
    //    Rcout << "The value of distance_d2 : \n" << distance_d2 << "\n";
    
    distance_Sigma2 = std::abs(Sigma2_bar - 0.01);  // For scalar version, abs() is for int type. Either use std::abs() for double type or fabs() for float type.
    /*  if(Sigma2_bar >= 0.01){
     distance_Sigma2 = Sigma2_bar - 0.01;
    }else{
     distance_Sigma2 = 0.01 - Sigma2_bar;
    }    */
    //    Rcout << "The value of distance_Sigma2 : \n" << distance_Sigma2 << "\n";
    distance_Sigma22 = std::abs(Sigma2_bar2 - 0.01);
    //    Rcout << "The value of distance_Sigma22 : \n" << distance_Sigma22 << "\n";
    
    c1 = c1 + 1;
    c2 = c2 + 1;
    
  }
  
  List Method1 = List::create(_["L"] = L, _["L_hat"] = L_hat, _["U"] = U, _["V"] = V, _["S"] = S, _["S_count"] = S_count, _["S_sum"] = S_sum, _["d"] = d, _["d_bar"] = d_bar, _["dbar_list"] = dbar_list, _["D"] = D, _["Sigma2"] = Sigma2, _["Sigma2_bar"] = Sigma2_bar, _["distance_d"] = distance_d, _["distance_Sigma2"] = distance_Sigma2, _["c1"] = c1);
  List Method2 = List::create(_["L2"] = L2, _["L_hat2"] = L_hat2, _["U2"] = U2, _["V2"] = V2, _["S2"] = S2, _["S_count2"] = S_count2, _["S_sum2"] = S_sum2, _["d2"] = d2, _["d_bar2"] = d_bar2, _["dbar_list2"] = dbar_list2, _["D2"] = D2, _["Sigma22"] = Sigma22, _["Sigma2_bar2"] = Sigma2_bar2, _["distance_d2"] = distance_d2, _["distance_Sigma22"] = distance_Sigma22, _["c2"] = c2);
  return List::create(_["M1"] = Method1, _["M2"] = Method2);
}




// [[Rcpp::export]]
List Simul_after_burnin4_tn(NumericMatrix Y, NumericMatrix U, NumericMatrix D, NumericMatrix V, NumericMatrix L, NumericMatrix S, NumericMatrix S_count, NumericVector d, double Sigma2, double tow2, double q1, int n, int p, int r, int K, int burn_in, double a, double b){
  //NumericVector d_sum = clone(d);
  //NumericVector d_bar(r);
  NumericMatrix L_sum = clone(L);
  NumericMatrix S_sum = clone(S);
  
  NumericMatrix L_hat(n,p);
  //double distance_d = 0;
  
  for(int count = 0; count < (K - burn_in); ++count){
    
    // Update the value of U for Method 1
    NumericMatrix U11(n,r);
    for(int i=0; i <n; ++i){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        for(int j=0; j<p; ++j){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*V(j,l)*D(l,l))/Sigma2);
        }
        U11(i,l) = temp1;
      }
    }
    
    U = rmf_matrix4(U11);
    
    // Update the value of V for Method 1
    NumericMatrix V11(p,r);
    for(int j=0; j <p; ++j){
      for(int l=0; l<r; ++l){
        double temp1 = 0;
        for(int i=0; i<n; ++i){
          temp1 = temp1 + (((Y(i,j) - S(i,j))*U(i,l)*D(l,l))/Sigma2);
        }
        V11(j,l) = temp1;
      }
    }
    
    V = rmf_matrix4(V11);
    
    // Update the value of d one at a time for Method 1
    d = rcpptn_d_update_mat(clone(d), matrix_multiply(transpose(matrix_multiply(matrix_add(Y, matrix_scalar_multiply(-1, S)), V)), U), Sigma2, n, p, r);
    
    // Once we got the updated set of elements in vector d, it's time to Update D
    D = rcpp_only_diag_cumprod(d);
    
    // Update the value of L and then S for Method 1
    NumericMatrix S_rnorm(n, p);
    NumericMatrix S_runif(n, p);
    for(int j=0; j < p; ++j){
      S_rnorm(_ , j) = Rcpp::rnorm(n, 0, sqrt((tow2 * Sigma2)/(tow2 + Sigma2)));
      S_runif(_ , j) = Rcpp::runif(n, 0, 1);
    }
    
    
    NumericMatrix S_MCC(n,p);
    
    double rate = 0;
    
    for(int i=0; i<n; ++i){
      for(int j=0; j<p; ++j){
        
        // It is needed for updating L
        double temp1 = 0;
        for(int l=0; l<r; ++l){
          temp1 = temp1 + (U(i,l)*D(l,l)*V(j,l));
        }
        L(i,j) = temp1;
        
        // Update the value of L_sum and L_hat for Method 1
        L_sum(i,j) = L_sum(i,j) + L(i,j);
        L_hat(i,j) = ((double)1/(count + 2)) * L_sum(i,j);
        
        if(S_runif(i,j) > (q1/(q1 + (((1-q1) * sqrt(Sigma2) * exp((tow2 * pow(Y(i,j) - L(i,j), 2.0))/(2 * Sigma2 * (tow2 + Sigma2))))/sqrt(tow2 + Sigma2))))){
          S_MCC(i,j) = 1;
          S_count(i,j) = S_count(i,j);
        }else{
          S_MCC(i,j) = 0;
          S_count(i,j) = S_count(i,j) + 1;
        }
        
        
        S(i,j) = (((tow2/(tow2 + Sigma2))*(Y(i,j) - L(i,j))) + S_rnorm(i,j)) * S_MCC(i,j);
        
        // Update the value of S_sum for Method 1
        S_sum(i,j) = S_sum(i,j) + S(i,j);
        
        rate = rate + pow(Y(i,j) - L(i,j) - S(i,j), 2.0);
      }
    }
    
    Sigma2 = rcpp_only_invgamma((n * p * 0.5) + a, b + (0.5 * rate));
    
    // Update d_sum and d_bar
    //d_sum = d_sum + d;  // Rcpp only allows element-wise arithmetic operatiosn for vectors, not for matrices.
    
    //d_bar = d_sum/(count + 2);
    
    
    // Update the distances
    //distance_d = max(abs(d_bar - d_star));
    
  }
  
  return List::create(_["L_hat"] = L_hat, _["S_count"] = S_count, _["S_sum"] = S_sum);   
}


/*** R
Sys.time()
# User Input
q1 <- (1 - 0.2171)   # For any n, Actual data Y_(n*n) (in particular, S_star_(n*n)) will be generated according to this q1
#q2 <- 0.90   # value of q for Method 2
#n_values_seq <- seq.int(from = 50, to = 100, by = 50)  # n = 100:100:300
#n_values_seq <- 100

## WE ARE CONSIDERING n*n SQUARE MATRICES ONLY.

## Other input values
a <- 2 # shape parameter for the inverse gamma distribution of sigma^2  ( should be > 0)        ## CHANGE
b <- 1  # rate parameter for the inverse gamma distribution of sigma^2  ( should be > 0)       ## CHANGE
K <- 13000  # number of iterations
tow2 <- 20   # the variance tow^2 of the normal distribution in the mixture prior of S. For the time-being, it's value is taken to be 10.
burn_in <- 5000     #16000

## Load all required packages
library(truncnorm)
library(rstiefel)
library(invgamma)   #as arma_invgamma is used here.
library(pracma)
library(psych)
library(mccr)
library(caret)
library(coda)                   
library(e1071)
library(ggplot2)
library(microbenchmark)

## Create the pdf file for traceplots for Method 1: q1 = 0.95

pdf('Agarwal_error_vs_rank.pdf', width = 11.694, height = 8.264)

rank_ratio <- seq(from = 0.05, to = 0.10, by = 0.05)
rank_vec <- rep(0, length(rank_ratio))
error_vec <- rep(0, length(rank_ratio))

n <- 100
p <- n

# Getting E_star
Sigma2_star <- 1/(n^2)
E_star <- matrix(rnorm(n*p, mean = 0, sd = sqrt(Sigma2_star)), nrow = n, ncol = p)
#E_star <- matrix(0, nrow = n, ncol = p)

# Getting S_star (according to q1 = 0.95)
S_star_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
S_star_runif <- matrix(runif(n*p), nrow = n, ncol = p)
S_actual_MCC <- (S_star_runif > q1)*1
S_star <- S_star_rnorm * S_actual_MCC



for(j in 1:length(rank_ratio)){
  
  r <- round(rank_ratio[j] * n)
  rank_vec[j] <- r
  
  # Getting U_star of order n*r
  # generate a random orthonormal matrix of order n*n. The randomness is meant w.r.t (additively invariant) Haar measure on O(n).
  U_star <- randortho(n, type = "orthonormal")[ ,1:r]                         #takes the first r many columns
  
  # Getting V_star of order p*r
  V_star <- randortho(p, type = "orthonormal")[ ,1:r]
  
  # Getting D_star of order r*r
  d_star <- c(runif(r-1, min = 1, max = 2), runif(1, min = 0.5, max = 1.5))         
  
  D_star <- diag(cumprod(d_star))
  
  L_star <- U_star %*% D_star %*% t(V_star)
  
  ## Define the true model
  Y <- (U_star %*% D_star %*% t(V_star)) + S_star + E_star
  
  #################### AFTER GENERATING TRUE DATA Y
  
  ## Simulation of E from the prior of sigma^2
  Sigma20 <- rinvgamma(1, shape = a, rate = b)     # same as rcpp_only_rinvgamma(a, b) 
  
  
  ## Simulation of S from the prior which is 0 w.p q [where, q = 1 - 1/p] and N(0,tow2) w.p (1-q)
  ## and also getting intial value of S_count
  S0_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
  S0_runif <- matrix(runif(n*p), nrow = n, ncol = p)
  S0_MCC <- (S0_runif > q1)*1
  
  S0 <- S0_rnorm * S0_MCC
  S0_count <- (S0_runif < q1)*1        #For any element of S0_count, it puts 1 when any element of S0 is 0.
  
  d0 <- c(rtruncnorm(r-1, a=1, b=Inf, mean=0, sd=1), rtruncnorm(1, a=0, b=Inf, mean=0, sd=1))
  
  D0 <- diag(cumprod(d0))
  
  ## Simulation of U of order n*r from uniform prior on the stiefel manifold R^(n*r)
  U0 <- rustiefel(n,r)
  
  ## Simulation of V of order p*r from uniform prior on the stiefel manifold R^(p*r)
  V0 <- rustiefel(p,r)
  
  L0 <- U0 %*% D0 %*% t(V0)
  
  
  ## BURN IN 
  Simul_burnin_list <- Simul_burnin4_tn(Y, U0, D0, V0, S0, d0, Sigma20, tow2, q1, n, p, r, burn_in, a, b)
  
  ## Output AFTER BURN IN
  U0 <- Simul_burnin_list[["U"]]
  V0 <- Simul_burnin_list[["V"]]
  d0 <- Simul_burnin_list[["d"]]
  D0 <- Simul_burnin_list[["D"]]
  L0 <- Simul_burnin_list[["L"]]
  S0 <- Simul_burnin_list[["S"]]
  S0_count <- Simul_burnin_list[["S_count"]]
  Sigma20 <- Simul_burnin_list[["Sigma2"]]
  
  
  
  ## Output AFTER ALL iterations
  
  Simul_afterburnin_list <- Simul_after_burnin4_tn(Y, U0, D0, V0, L0, S0, S0_count, d0, Sigma20, tow2, q1, n, p, r, K, burn_in, a, b)
  
  L_hat <- Simul_afterburnin_list[["L_hat"]]
  S_count <- Simul_afterburnin_list[["S_count"]]
  S_sum <- Simul_afterburnin_list[["S_sum"]]
  
  ## Getting S_hat, estimate of S_star for Method 1:
  J <- matrix(1, nrow = n, ncol = p)       # matrix whose all elements are 1
  avg_S <- S_sum / ((K - burn_in + 1)*J - S_count)
  avg_S[!is.finite(avg_S)] <- 0            # If for any elemnet of avg_S, denom is 0, then NA is replaced by 0 
  # since in that case, S_count[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC <- (S_count < ((K - burn_in + 1)/2))*1
  S_hat <- avg_S * S_predicted_MCC
  
  error_vec[j] <- (norm(L_star - L_hat, type = "F"))^2 + (norm(S_star - S_hat, type = "F"))^2
  
  
}  





daf <- data.frame(rank_vec, error_vec)
## About ggplot below: when inside a loop, you have to print your ggplot object. If outside a loop, without print function works fine.
print(ggplot(daf, aes(rank_vec)) + 
        geom_line(aes(y = error_vec), colour="Blue") +    # 1st layer of plot with q = 0.95
        theme_gray() +
        labs(title = paste("Error vs. rank"), y = paste("Frobenius norm error squared")) + 
        theme(plot.title = element_text(hjust = 0.5)) +   # By default, title is left aligned in ggplot. Use it to center the titile.
        coord_cartesian(ylim = c(0, 40)))



dev.off()

Sys.time()


*/