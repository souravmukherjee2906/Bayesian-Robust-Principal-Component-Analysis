if(!require(Rcpp)) install.packages("Rcpp")
library(Rcpp)
sourceCpp("Simulation_updated_Rcpp_Only_compare_1.1_exact_data_according_to_q=0.95.cpp")


# User Input
q1 <- 0.95   # For any n, Actual data Y_(n*n) (in particular, S_star_(n*n)) will be generated according to q = 0.95
q2 <- 0.90   # value of q for Method 2, which is q = 0.90
#n_values_seq <- seq.int(from = 50, to = 100, by = 50)  # n = 100:100:300
n_values_seq <- 100

## WE ARE CONSIDERING n*n SQUARE MATRICES ONLY.

## Other input values
a <- 5000 # shape parameter for the inverse gamma distribution of sigma^2  ( should be > 0)        ## CHANGE
b <- 1  # rate parameter for the inverse gamma distribution of sigma^2  ( should be > 0)       ## CHANGE
K <- 40000  # number of iterations
tow2 <- 20   # the variance tow^2 of the normal distribution in the mixture prior of S. For the time-being, it's value is taken to be 10.
burn_in <- 10000

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

## Create the pdf file for traceplots for Method 1: q1 = 0.95 and Method 2: q2 = 0.90
# pdf('5.1.Rcpp_compare_5.1_after=30k_burnin=10k_5000.1.pdf', width = 11.694, height = 8.264)

iterations <- seq(0, (K - burn_in))  # Required for ggplot in Traceplot.

##### Start the simulation for the different methods: Method 1:q = 0.95, Method 2:q = 0.90, Method 3:Inexact ALM - rrpca & Method 4:ALM - Candes.

for(n in n_values_seq){
  
  #n <- 100  #*#
  #r <- 3
  p <- n
  r <- round(0.05*n)    # 1<= r <= n
  
  ##### Actual S_star in actual data is generated according to q = 0.95
  
  # Getting E_star where each entry is iid N(0, 0.01) i.e. actual value of Sigma^2 is 0.01
  #  E_star <- matrix(rnorm(n*p, mean = 0, sd = 0.1), nrow = n, ncol = p)     
  E_star <- matrix(0, nrow = n, ncol = p)
  
  # Getting S_star (according to q1 = 0.95)
  S_star_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
  S_star_runif <- matrix(runif(n*p), nrow = n, ncol = p)
  S_actual_MCC <- (S_star_runif > q1)*1
  S_star <- S_star_rnorm * S_actual_MCC
  
  # Getting U_star of order n*r
  # generate a random orthonormal matrix of order n*n. The randomness is meant w.r.t (additively invariant) Haar measure on O(n).
  U_star <- randortho(n, type = "orthonormal")[ ,1:r]                         #takes the first r many columns
  
  # Getting V_star of order p*r
  V_star <- randortho(p, type = "orthonormal")[ ,1:r]
  
  # Getting D_star of order r*r
  d_star <- c(runif(r-1, min = 1, max = 2), runif(1, min = 0.5, max = 1.5))          ## CHANGE
  #print(d_star)
  
  D_star <- rcpp_only_diag_cumprod(d_star)
  #print(D_star)
  
  L_star <- U_star %*% D_star %*% t(V_star)
  
  ## Define the true model
  Y <- (U_star %*% D_star %*% t(V_star)) + S_star + E_star
  #print(Y)
  
  #################### AFTER GENERATING TRUE DATA Y
  
  ##### Method 1 & 2: Our method where prior being generated with q = 0.95 and q = 0.90
  
  ## Simulation of E from the prior of sigma^2
  Sigma20 <- rcpp_only_rinvgamma(a, b)     # same as rinvgamma(1, shape = a, rate = b)
  #print(Sigma20)
  
  ## Simulation of S from the prior which is 0 w.p q [where, q = 1 - 1/p] and N(0,tow2) w.p (1-q)
  ## and also getting intial value of S_count
  S0_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
  S0_runif <- matrix(runif(n*p), nrow = n, ncol = p)
  S0_MCC <- (S0_runif > q1)*1
  S0_MCC2 <- (S0_runif > q2)*1
  S0 <- S0_rnorm * S0_MCC
  S02 <- S0_rnorm * S0_MCC2
  #print(S0)  
  S0_count <- (S0_runif < q1)*1        #For any element of S0_count, it puts 1 when any element of S0 is 0.
  S0_count2 <- (S0_runif < q2)*1
  #print(S0_count)
  
  ## Simulation of D of order r*r from prior of d
  d0 <- c(rtruncnorm(r-1, a=1, b=Inf, mean=0, sd=1), rtruncnorm(1, a=0, b=Inf, mean=0, sd=1))
  #print("Initial choice of d")
  #print(d0)
  D0 <- rcpp_only_diag_cumprod(d0)
  #print(D0)
  
  ## Simulation of U of order n*r from uniform prior on the stiefel manifold R^(n*r)
  U0 <- rcpp_only_rustiefel(n,r)
  #print(U0)
  
  ## Simulation of V of order p*r from uniform prior on the stiefel manifold R^(p*r)
  V0 <- rcpp_only_rustiefel(p,r)
  #print(V0)
  
  L0 <- U0 %*% D0 %*% t(V0)
  #print(L0)
  
  ### Simulation from the full conditional posterior distributions with K many iterations.
  ## Initially we take the input values of U, V, d, D, S, Sigma2 and L to be above 
  ## values which we got initially after simulation from their respective priors.
  ## BURN IN 
  
  Simul_burnin_list <- Simul_burnin(Y, U0, U0, D0, D0, V0, V0, S0, S02, d0, d0, Sigma20, Sigma20, tow2, q1, q2, n, p, r, burn_in, a, b)
  
  
  ## Output AFTER BURN IN
  U0 <- Simul_burnin_list[["U"]]
  #print(U0)
  U02 <- Simul_burnin_list[["U2"]]
  #print(U02)
  
  V0 <- Simul_burnin_list[["V"]]
  #print(V0)
  V02 <- Simul_burnin_list[["V2"]]
  #print(V02)
  
  d0 <- Simul_burnin_list[["d"]]
  #print("After Burn in d")
  #print(d0)
  d02 <- Simul_burnin_list[["d2"]]
  #print("After Burn in d2")
  #print(d02)
  
  D0 <- Simul_burnin_list[["D"]]
  #print(D0)
  D02 <- Simul_burnin_list[["D2"]]
  #print(D02)
  
  L0 <- Simul_burnin_list[["L"]]
  #print(L0)
  L02 <- Simul_burnin_list[["L2"]]
  #print(L02)
  
  S0 <- Simul_burnin_list[["S"]]
  #print(S0)
  S02 <- Simul_burnin_list[["S2"]]
  #print(S02)
  
  S0_count <- Simul_burnin_list[["S_count"]]
  #print(S0_count)
  S0_count2 <- Simul_burnin_list[["S_count2"]]
  #print(S0_count2)
  
  Sigma20 <- Simul_burnin_list[["Sigma2"]]
  #print(Sigma20)
  Sigma202 <- Simul_burnin_list[["Sigma22"]]
  #print(Sigma202)
  
  ## Output AFTER ALL iterations
  
  Simul_afterburnin_list <- Simul_after_burnin(Y, U0, U02, D0, D02, V0, V02, L0, L02, S0, S02, S0_count, S0_count2, d0, d02, d_star, Sigma20, Sigma202, tow2, q1, q2, n, p, r, K, burn_in, a, b)
  
  L_hat <- Simul_afterburnin_list[["M1"]][["L_hat"]]
  L_hat2 <- Simul_afterburnin_list[["M2"]][["L_hat2"]]
  
  S_count <- Simul_afterburnin_list[["M1"]][["S_count"]]
  S_count2 <- Simul_afterburnin_list[["M2"]][["S_count2"]]
  
  S_sum <- Simul_afterburnin_list[["M1"]][["S_sum"]]
  S_sum2 <- Simul_afterburnin_list[["M2"]][["S_sum2"]]
  
  d_bar <- Simul_afterburnin_list[["M1"]][["d_bar"]]    # d_bar at the last iteration for Method 1
  #print(d_bar)
  #Simul_afterburnin_list[["M1"]][["d"]]
  #Simul_burnin_list[["d"]]  
  d_bar2 <- Simul_afterburnin_list[["M2"]][["d_bar2"]]    # d_bar at the last iteration for Method 2
  #print(d_bar2)
  #Simul_afterburnin_list[["M2"]][["d2"]]
  #Simul_burnin_list[["d2"]]
  
  dbar_list <- Simul_afterburnin_list[["M1"]][["dbar_list"]]
  dbar_list2 <- Simul_afterburnin_list[["M2"]][["dbar_list2"]]
  
  Sigma2_bar <- Simul_afterburnin_list[["M1"]][["Sigma2_bar"]]
  Sigma2_bar2 <- Simul_afterburnin_list[["M2"]][["Sigma2_bar2"]]
  
  distance_d <- Simul_afterburnin_list[["M1"]][["distance_d"]]
  distance_d2 <- Simul_afterburnin_list[["M2"]][["distance_d2"]]
  
  distance_Sigma2 <- Simul_afterburnin_list[["M1"]][["distance_Sigma2"]]
  distance_Sigma22 <- Simul_afterburnin_list[["M2"]][["distance_Sigma22"]]
  
  ##### Outputs for Method 1, Method 2, Method 3(Inexact ALM - rrpca) and Method 4(ALM - Candes):
  
  ## Getting S_hat, estimate of S_star for Method 1:
  J <- matrix(1, nrow = n, ncol = p)       # matrix whose all elements are 1
  avg_S <- S_sum / ((K - burn_in + 1)*J - S_count)
  avg_S[!is.finite(avg_S)] <- 0            # If for any elemnet of avg_S, denom is 0, then NA is replaced by 0 
  # since in that case, S_count[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC <- (S_count < ((K - burn_in + 1)/2))*1
  S_hat <- avg_S * S_predicted_MCC
  
  ## Getting S_hat2, estimate of S_star for Method 2:
  avg_S2 <- S_sum2 / ((K - burn_in + 1)*J - S_count2)
  avg_S2[!is.finite(avg_S2)] <- 0            # If for any elemnet of avg_S2, denom is 0, then NA is replaced by 0 
  # since in that case, S_count2[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC2 <- (S_count2 < ((K - burn_in + 1)/2))*1
  S_hat2 <- avg_S2 * S_predicted_MCC2
  
  ## Getting L_hat_IALM and S_hat_IALM, estimates of L_star and S_star respectively for Method 3: Inexact ALM - rrpca.
  library(rsvd)
  L_hat_IALM <- rrpca(Y, maxiter = 500)$L        # default maxiter = 50
  S_hat_IALM <- rrpca(Y, maxiter = 500)$S
  S_predicted_MCC_IALM <- (S_hat_IALM != 0)*1
  
  ## Getting L_hat_rpca and S_hat_rpca, estimates of L_star and S_star respectively for Method 4: ALM - Candes.
  detach("package:rsvd", unload = T)
  library(rpca)
  L_hat_rpca <- rpca(Y, max.iter = 5000)$L       # default max.iter = 5000
  S_hat_rpca <- rpca(Y, max.iter = 5000)$S
  S_predicted_MCC_rpca <- (S_hat_rpca != 0)*1
  
  print("The value of n and p")
  print(n)
  print(p)
  print("Does rpca function (ALM algorithm described in candes paper) converge for by default max iteration = 5000? If TRUE then YES")
  print(rpca(Y)$convergence$converged)         # rpca(Y, max.iter = m)$convergence$converged ,if m is not 5000.
  print("Number of performed iterations")
  print(rpca(Y)$convergence$iterations)        # rpca(Y, max.iter = m)$convergence$iterations ,if m is not 5000.
  
  ## Finding sensitivity and specificity for all Methods:
  S_actual_factor <- factor(as.factor(S_actual_MCC), levels = c("0", "1"))     ## for the Actual data.
  S_predic_factor <- factor(as.factor(S_predicted_MCC), levels = c("0", "1"))  ## for Method 1: q = 0.95
  S_predic_factor2 <- factor(as.factor(S_predicted_MCC2), levels = c("0", "1"))  ## for Method 2: q = 0.90
  S_predic_factor_IALM <- factor(as.factor(S_predicted_MCC_IALM), levels = c("0", "1"))  ## for Method 3: Inexact ALM - rrpca
  S_predic_factor_rpca <- factor(as.factor(S_predicted_MCC_rpca), levels = c("0", "1"))  ## for Method 4: ALM - Candes
  
  print("Sensitivity and Specificity for Method 1: q = 0.95")
  print(sensitivity(S_predic_factor, S_actual_factor, positive = "1"))
  print(specificity(S_predic_factor, S_actual_factor, negative = "0"))
  print("Sensitivity and Specificity for Method 2: q = 0.90")
  print(sensitivity(S_predic_factor2, S_actual_factor, positive = "1"))
  print(specificity(S_predic_factor2, S_actual_factor, negative = "0"))
  print("Sensitivity and Specificity for Method 3: Inexact ALM - rrpca")
  print(sensitivity(S_predic_factor_IALM, S_actual_factor, positive = "1"))
  print(specificity(S_predic_factor_IALM, S_actual_factor, negative = "0"))
  print("Sensitivity and Specificity for Method 4: ALM - Candes")
  print(sensitivity(S_predic_factor_rpca, S_actual_factor, positive = "1"))
  print(specificity(S_predic_factor_rpca, S_actual_factor, negative = "0"))
  
  ## Mattews Correlation Coefficient for all Methods:
  print("Mattews Correlation Coefficient(MCC) for S for Method 1: q = 0.95")
  print(mccr(S_actual_MCC, S_predicted_MCC))        # 0 is taken as 0(negative) and non-zero values are taken as 1 (positive)
  print("Mattews Correlation Coefficient(MCC) for S for Method 2: q = 0.90")
  print(mccr(S_actual_MCC, S_predicted_MCC2))        # 0 is taken as 0(negative) and non-zero values are taken as 1 (positive)
  print("Mattews Correlation Coefficient(MCC) for S for Method 3: Inexact ALM - rrpca")
  print(mccr(S_actual_MCC, S_predicted_MCC_IALM))        # 0 is taken as 0(negative) and non-zero values are taken as 1 (positive)
  print("Mattews Correlation Coefficient(MCC) for S for Method 4: ALM - Candes")
  print(mccr(S_actual_MCC, S_predicted_MCC_rpca))        # 0 is taken as 0(negative) and non-zero values are taken as 1 (positive)
  
  ## Finding Relative Ratio for L and S for all Methods:
  print("Relative Ratio for L and S for Method 1: q = 0.95")
  print(norm(L_star - L_hat, type = "F")/ norm(L_star, type = "F"))  # Frobenius norm distance
  print(norm(S_star - S_hat, type = "F")/ norm(S_star, type = "F"))  # Frobenius norm distance
  print("Relative Ratio for L and S for Method 2: q = 0.90")
  print(norm(L_star - L_hat2, type = "F")/ norm(L_star, type = "F"))  # Frobenius norm distance
  print(norm(S_star - S_hat2, type = "F")/ norm(S_star, type = "F"))  # Frobenius norm distance
  print("Relative Ratio for L and S for Method 3: Inexact ALM - rrpca")
  print(norm(L_star - L_hat_IALM, type = "F")/ norm(L_star, type = "F"))  # Frobenius norm distance
  print(norm(S_star - S_hat_IALM, type = "F")/ norm(S_star, type = "F"))  # Frobenius norm distance
  print("Relative Ratio for L and S for Method 4: ALM - Candes")
  print(norm(L_star - L_hat_rpca, type = "F")/ norm(L_star, type = "F"))  # Frobenius norm distance
  print(norm(S_star - S_hat_rpca, type = "F")/ norm(S_star, type = "F"))  # Frobenius norm distance
  
  ## Maximum modulus of all the elements in (L_star - L_hat) for all Methods:
  print("Maximum modulus of all the elements in (L_star - L_hat) for Method 1: q = 0.95")
  print(max(abs(L_star - L_hat)))
  print("Maximum modulus of all the elements in (L_star - L_hat) for Method 2: q = 0.90")
  print(max(abs(L_star - L_hat2)))
  print("Maximum modulus of all the elements in (L_star - L_hat) for Method 3: Inexact ALM - rrpca")
  print(max(abs(L_star - L_hat_IALM)))
  print("Maximum modulus of all the elements in (L_star - L_hat) for Method 4: ALM - Candes")
  print(max(abs(L_star - L_hat_rpca)))
  
  ## Maximum modulus of all the elements in (S_star - S_hat) for all Methods:
  print("Maximum modulus of all the elements in (S_star - S_hat) for Method 1: q = 0.95")
  print(max(abs(S_star - S_hat)))
  print("Maximum modulus of all the elements in (S_star - S_hat) for Method 2: q = 0.90")
  print(max(abs(S_star - S_hat2)))
  print("Maximum modulus of all the elements in (S_star - S_hat) for Method 3: Inexact ALM - rrpca")
  print(max(abs(S_star - S_hat_IALM)))
  print("Maximum modulus of all the elements in (S_star - S_hat) for Method 4: ALM - Candes")
  print(max(abs(S_star - S_hat_rpca)))
  
  dbar_array <- simplify2array(dbar_list)                                                       
  dbar_array2 <- simplify2array(dbar_list2)
  
  
  for (i in 1:r) {
    daf <- data.frame(iterations, mcmc(dbar_array[i,]), mcmc(dbar_array2[i,]))
    ## About ggplot below: when inside a loop, you have to print your ggplot object. If outside a loop, without print function works fine.
    print(ggplot(daf, aes(iterations)) + 
            geom_line(aes(y = mcmc(dbar_array[i,])), colour="Black") +    # 1st layer of plot with q = 0.95
            geom_line(aes(y = mcmc(dbar_array2[i,])), colour="Blue") +    # 2nd layer of plot with q = 0.90
            geom_hline(yintercept = d_star[i], colour="Red") +              # 3rd layer with the actual value of d_star[i]
            theme_gray() +
            labs(title = paste("Trace Plot of d_bar[", i, "]", ", when n =", n, "\n Black: q = 0.95, Blue: q = 0.90, Red: d_star[", i, "]", "=", round(d_star[i], digits = 5)), y = paste("d_bar[", i, "]")) + 
            theme(plot.title = element_text(hjust = 0.5)) +   # By default, title is left aligned in ggplot. Use it to center the titile.
            coord_cartesian(ylim = c(0, 5)))
  }
  
  print("d_star")
  print(d_star)
  print("d_bar at the last iteration for Method 1")
  print(d_bar)
  print("d_bar at the last iteration for Method 2")
  print(d_bar2)
  print("L infinity distance between d_bar and d_star at the last iteration for Method 1")
  print(distance_d)
  print("L infinity distance between d_bar2 and d_star at the last iteration for Method 2")
  print(distance_d2)
  #print("Sigma2_bar at the last iteration")
  #print(Sigma2_bar)
  print("Absolute distance between Sigma2_bar and 0.01 at the last iteration for Method 1")
  print(distance_Sigma2)
  print("Absolute distance between Sigma2_bar2 and 0.01 at the last iteration for Method 2")
  print(distance_Sigma22)
  
}   # end of for loop for n_values
# dev.off()  
