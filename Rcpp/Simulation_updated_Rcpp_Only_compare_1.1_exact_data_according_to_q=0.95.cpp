#include <Rcpp.h>
#include <algorithm>

using namespace Rcpp;


// [[Rcpp::export]]
NumericMatrix rcpp_only_diag_cumprod(NumericVector v){
  int r = v.size();
  NumericVector cump_rev = cumprod(rev(v));
  NumericVector rev_cump_rev = rev(cump_rev);
  NumericMatrix out(r); //equivalent to matrix(0, nrow = 2, ncol = 2)
  for(int i=0; i<r; ++i){
    for(int j=0; j<r; ++j){
      if(i==j){
        out(i,j) = rev_cump_rev[i];
      }else{
        out(i,j) = 0;
      }
    }
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
    for(int j=0; j< r; ++j){
      for(int k=0; k< r; ++k){
        if((j == k) && (j <= i)){
          D_cap(j,j) = 1;
          for(int l=j; l< r; ++l){
            if(l==i){
              D_cap(j,j) = D_cap(j,j);
            }else{
              D_cap(j,j) = D_cap(j,j) * d[l];
            }
          }
        }
        else
          D_cap(j,k) = 0;
      }
    }
    
    /*  
     D_cap = D_cap/d[i];
     
     double num = matrix_trace(matrix_multiply(D_cap,A));
     double denom = matrix_trace(matrix_dot_multiply(D_cap,D_cap)) + Sigma2;
     double mu = num/denom;
     double sigma = sqrt(Sigma2/denom);
     
     if (i < r-1){
     d[i] = rcpp_only_rtruncnorm(1, R_PosInf, mu, sigma);
     }
     else{
     d[i] = rcpp_only_rtruncnorm(0, R_PosInf, mu, sigma);
     }
     */    
    
    double num = 0;
    for(int j=0; j<r; ++j){
      num = num + (D_cap(j,j)*A(j,j));
    }
    
    double denom = 0;
    for(int j=0; j<r; ++j){
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


// creating the function for 1:burn_in below:

// [[Rcpp::export]]
List Simul_burnin(NumericMatrix Y, NumericMatrix U, NumericMatrix U2, NumericMatrix D, NumericMatrix D2, NumericMatrix V, NumericMatrix V2, NumericMatrix S, NumericMatrix S2, NumericVector d, NumericVector d2, double Sigma2, double Sigma22, double tow2, double q1, double q2, int n, int p, int r, int burn_in, double a, double b){
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
    
    U = rcpp_only_rmf_matrix(U11);
    //    Rcout << "The value of U : \n" << U << "\n";
    U2 = rcpp_only_rmf_matrix(U22);
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
    
    V = rcpp_only_rmf_matrix(V11);
    //    Rcout << "The value of V : \n" << V << "\n";
    V2 = rcpp_only_rmf_matrix(V22);
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


// Creating the function for AFTER BURN IN:

// [[Rcpp::export]]
List Simul_after_burnin(NumericMatrix Y, NumericMatrix U, NumericMatrix U2, NumericMatrix D, NumericMatrix D2, NumericMatrix V, NumericMatrix V2, NumericMatrix L, NumericMatrix L2, NumericMatrix S, NumericMatrix S2, NumericMatrix S_count, NumericMatrix S_count2, NumericVector d, NumericVector d2, NumericVector d_star, double Sigma2, double Sigma22, double tow2, double q1, double q2, int n, int p, int r, int K, int burn_in, double a, double b){
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
    
    U = rcpp_only_rmf_matrix(U11);
    //    Rcout << "The value of U : \n" << U << "\n";
    U2 = rcpp_only_rmf_matrix(U22);
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
    
    V = rcpp_only_rmf_matrix(V11);
    //    Rcout << "The value of V : \n" << V << "\n";
    V2 = rcpp_only_rmf_matrix(V22);
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

