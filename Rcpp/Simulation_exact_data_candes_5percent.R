## We are trying to decompose a noisy image matrix into low-rank component (foreground),
## sparse component and noise component (background).
## This file contains codes for the simulation of estimates of the low-rank and sparse
## components for several iterations.

## Actual data is generated according to E. Candes's paper (does not involve any q).
## See lines 98 - 125.


#------------------------------------------------------------------------------------
## Remove everything from the Global environment
#------------------------------------------------------------------------------------
rm(list = ls())


#------------------------------------------------------------------------------------
## Install the required packages into R if it's not already installed.
#------------------------------------------------------------------------------------
if(!require(Rcpp)) install.packages("Rcpp")
if(!require(truncnorm)) install.packages("truncnorm")
if(!require(rstiefel)) install.packages("rstiefel")
if(!require(invgamma)) install.packages("invgamma")
if(!require(pracma)) install.packages("pracma")
if(!require(psych)) install.packages("psych")
if(!require(mccr)) install.packages("mccr")
if(!require(caret)) install.packages("caret")
if(!require(coda)) install.packages("coda")
if(!require(e1071)) install.packages("e1071")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(Matrix)) install.packages("Matrix")


#------------------------------------------------------------------------------------
## Load the required packages into the R environment.
#------------------------------------------------------------------------------------
library(Rcpp)
library(truncnorm)
library(rstiefel)
library(invgamma)
library(pracma)
library(psych)
library(mccr)
library(caret)
library(coda)                   
library(e1071)
library(ggplot2)
library(Matrix)   # needed for the generating random sparse matrix true S_star, 
                  # as described in E. Candes's paper.


#------------------------------------------------------------------------------------
## Source all the necessary functions from the corresponding .cpp file.
#------------------------------------------------------------------------------------
sourceCpp("Functions.cpp")


#------------------------------------------------------------------------------------
## User provided values
#------------------------------------------------------------------------------------
# Value of q for Method 1.
q1 <- 0.95
# Value of q for Method 2.
q2 <- 0.90

# Values of n. 
n_values_seq <- seq.int(from = 100, to = 3000, by = 100)

## WE ARE CONSIDERING n*n SQUARE MATRICES ONLY.

# Other input values
a <- 5000 # Shape parameter for the inverse gamma distribution of sigma^2.
b <- 1  # Rate parameter for the inverse gamma distribution of sigma^2.
# We can simulate with another choice of (a,b), e.g., (a,b) = (4,5).
K <- 40000  # Total number of iterations.
burn_in <- 10000 # Number of burn-ins.
tow2 <- 20  # The variance tow^2 of the normal distribution in the mixture prior of S.
iterations <- seq(0, (K - burn_in))


#------------------------------------------------------------------------------------
## Create a pdf file containing the traceplots for Method 1: q1 = 0.95 and 
## Method 2: q2 = 0.90
#------------------------------------------------------------------------------------
pdf('Simulation_AfterBurnin=30k_Burnin=10k_a=5000_b=1_TrueMethod_candes_5percent.pdf', 
    width = 11.694, height = 8.264)


#------------------------------------------------------------------------------------
## Start the simulation for the different methods: 
## Method 1: q = 0.95, Method 2: q = 0.90, 
## Method 3: Inexact ALM - rrpca, Method 4: ALM - Candes.
#------------------------------------------------------------------------------------

for(n in n_values_seq){
  p <- n
  r <- round(0.05*n)    # 1 <= r <= n
  
  ## Actual data is generated according to Candes's paper (does not involve any q).
  ## There is no E_star in the data generating process in Candes's paper. 
  ## So, E_star = 0 in this case.
  
  ## Generating L_star as described in Candes's paper.
  L_star1 <- matrix(rnorm(n*r, mean = 0, sd = sqrt(1/n)), nrow = n, ncol = r)
  L_star2 <- matrix(rnorm(p*r, mean = 0, sd = sqrt(1/p)), nrow = p, ncol = r)  
  # here, p = n in Candes's paper.
  L_star <- L_star1 %*% t(L_star2)  # it is rank-r matrix, as described in Candes's paper.
  
  ## Generating S_star as described in Candes's paper with 5% non-zero elements.
  ## We can also opt for 10% non-zero elements in S_star, in which case we need to use
  ## nnz = round(0.10*(n^2)) in "P_omega_sparse" defined below.
  ## If 10% choice is used, change the name of the pdf file in line 84 accordingly.
  P_omega_sparse <- rsparsematrix(nrow = n, ncol = p, nnz = round(0.05*(n^2)), rand.x = NULL)
  P_omega_compressed <- P_omega_sparse*1     # Sparse Matrix in the "dgCMatrix" class
                                             # in a compressed form containing 1 and 0.
  P_omega <- as.matrix(P_omega_compressed)   # Regular spare matrix containing 1 and 0.
  E_unif <- matrix(runif(n*p), nrow = n, ncol = p) # here, p = n in Candes's paper.
  E_ber1 <- (E_unif >= 0.5)*1          # entries have 0 and 1
  E_ber2 <- (-1)*((E_unif < 0.5)*1)    # entries have 0 and -1
  E_ber <- E_ber1 + E_ber2             # entries are independent Bernoulli (-1, 1)
                                       # as described in Candes's paper.
  S_star <- P_omega * E_ber            # it is a sparse matrix with elements in (-1,0,1) 
                                       # and number of non-zero entries = round(0.05*n^2), 
                                       # as described in Candes's paper.
  S_actual_MCC <- P_omega              # it is needed to calculate Sensititvity, 
                                       # Specificity and MCC later.
  
  ## Define the true model, as described in candes paper.
  Y <- L_star + S_star
  
  ## Getting (True) D_star (and d_star) by using Singular value
  ## decomposition: L = UDV' implies D = U'LV.
  ## This (True) d_star has been created solely for the purpose of caculating 
  ## distance_d = max(abs(d_bar - d_star)) later during each iteration of (Our) 
  ## Method 1 and 2.
  
  ## Getting (True) D_star by using the Singular Value Decomposition, 
  ## when actual data is generated according to Candes's paper.
  D_star <- t(svd(L_star, nu = r, nv = r)$u) %*% L_star %*% (svd(L_star, nu = r, nv = r)$v)
  
  ## Getting (True) d_star from above D_star, 
  ## using the way we defined d_star: d[i] = D[i,i]/D[i+1, i+1] and d[r] = D[r,r].
  D_star_vec <- diag(D_star)
  d_star <- D_star_vec %*% diag(c(1/D_star_vec[-1], 1))
  
  
  ## AFTER GENERATING TRUE DATA Y
  
  ## Method 1 & 2: Our method where prior being generated with q = 0.95 and q = 0.90
  
  ## Simulation of E from the prior of sigma^2.
  Sigma20 <- rcpp_only_rinvgamma(a, b)  # same as rinvgamma(1, shape = a, rate = b) in R.
  
  ## Simulation of S from the prior which is 0 with probability q [where, q = 1 - 1/p], 
  ## and N(0,tow2) with probability (1-q).
  ## Also getting intial value of S_count.
  S0_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
  S0_runif <- matrix(runif(n*p), nrow = n, ncol = p)
  S0_MCC <- (S0_runif > q1)*1
  S0_MCC2 <- (S0_runif > q2)*1
  S0 <- S0_rnorm * S0_MCC
  S02 <- S0_rnorm * S0_MCC2 
  S0_count <- (S0_runif < q1)*1  # For any element of S0_count, it puts 1 when the 
                                 # corresponding element of S0 is 0.
  S0_count2 <- (S0_runif < q2)*1
  
  ## Simulation of D of order r*r from prior of d.
  d0 <- c(rtruncnorm(r-1, a=1, b=Inf, mean=0, sd=1), rtruncnorm(1, a=0, b=Inf, mean=0, sd=1))
  D0 <- rcpp_only_diag_cumprod(d0)
  
  ## Simulation of U of order n*r from uniform prior on the stiefel manifold R^(n*r).
  U0 <- rcpp_only_rustiefel(n,r)
  
  ## Simulation of V of order p*r from uniform prior on the stiefel manifold R^(p*r).
  V0 <- rcpp_only_rustiefel(p,r)
  
  ## Value of L.
  L0 <- U0 %*% D0 %*% t(V0)
  
  ## Simulation from the full conditional posterior distributions with K many iterations.
  ## Initially we take the input values of U, V, d, D, S, Sigma2 and L to be above 
  ## values which we got initially after simulation from their respective priors.
  
  ## BURN IN
  Simul_burnin_list <- Simul_burnin(Y, U0, U0, D0, D0, V0, V0, S0, S02, d0, d0, 
                                    Sigma20, Sigma20, tow2, q1, q2, n, p, r, 
                                    burn_in, a, b)
  
  ## Outputs after iterating for 1:BurnIn.
  U0 <- Simul_burnin_list[["U"]]
  U02 <- Simul_burnin_list[["U2"]]
  V0 <- Simul_burnin_list[["V"]]
  V02 <- Simul_burnin_list[["V2"]]
  d0 <- Simul_burnin_list[["d"]]
  d02 <- Simul_burnin_list[["d2"]]
  D0 <- Simul_burnin_list[["D"]]
  D02 <- Simul_burnin_list[["D2"]]
  L0 <- Simul_burnin_list[["L"]]
  L02 <- Simul_burnin_list[["L2"]]
  S0 <- Simul_burnin_list[["S"]]
  S02 <- Simul_burnin_list[["S2"]]
  S0_count <- Simul_burnin_list[["S_count"]]
  S0_count2 <- Simul_burnin_list[["S_count2"]]
  Sigma20 <- Simul_burnin_list[["Sigma2"]]
  Sigma202 <- Simul_burnin_list[["Sigma22"]]
  
  ## POST BURN IN
  Simul_afterburnin_list <- Simul_after_burnin(Y, U0, U02, D0, D02, V0, V02, L0, 
                                               L02, S0, S02, S0_count, S0_count2, 
                                               d0, d02, d_star, Sigma20, Sigma202, 
                                               tow2, q1, q2, n, p, r, K, burn_in, 
                                               a, b)
  
  L_hat <- Simul_afterburnin_list[["M1"]][["L_hat"]]
  L_hat2 <- Simul_afterburnin_list[["M2"]][["L_hat2"]]
  
  S_count <- Simul_afterburnin_list[["M1"]][["S_count"]]
  S_count2 <- Simul_afterburnin_list[["M2"]][["S_count2"]]
  
  S_sum <- Simul_afterburnin_list[["M1"]][["S_sum"]]
  S_sum2 <- Simul_afterburnin_list[["M2"]][["S_sum2"]]
  
  d_bar <- Simul_afterburnin_list[["M1"]][["d_bar"]]  # d_bar at the last iteration for Method 1.
  d_bar2 <- Simul_afterburnin_list[["M2"]][["d_bar2"]] # d_bar at the last iteration for Method 2.
  
  dbar_list <- Simul_afterburnin_list[["M1"]][["dbar_list"]]
  dbar_list2 <- Simul_afterburnin_list[["M2"]][["dbar_list2"]]
  
  Sigma2_bar <- Simul_afterburnin_list[["M1"]][["Sigma2_bar"]]
  Sigma2_bar2 <- Simul_afterburnin_list[["M2"]][["Sigma2_bar2"]]
  
  distance_d <- Simul_afterburnin_list[["M1"]][["distance_d"]]
  distance_d2 <- Simul_afterburnin_list[["M2"]][["distance_d2"]]
  
  distance_Sigma2 <- Simul_afterburnin_list[["M1"]][["distance_Sigma2"]]
  distance_Sigma22 <- Simul_afterburnin_list[["M2"]][["distance_Sigma22"]]
  
  
  ## Outputs after iterating for (BurnIn + 1):(Total iteration - BurnIn).
  ## Outputs for Method 1, Method 2, 
  ## Method 3(Inexact ALM - rrpca) and Method 4(ALM - Candes):
  
  ## Getting S_hat, estimate of S_star for Method 1:
  J <- matrix(1, nrow = n, ncol = p)  # matrix whose all elements are 1.
  avg_S <- S_sum / ((K - burn_in + 1)*J - S_count)
  avg_S[!is.finite(avg_S)] <- 0       # For any elemnet of avg_S, if denom is 0 
                                      # then NA is replaced by 0.
                                      # The reason is because in that case, 
                                      # S_count[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC <- (S_count < ((K - burn_in + 1)/2))*1
  S_hat <- avg_S * S_predicted_MCC
  
  ## Getting S_hat2, estimate of S_star for Method 2:
  avg_S2 <- S_sum2 / ((K - burn_in + 1)*J - S_count2)
  avg_S2[!is.finite(avg_S2)] <- 0     # For any elemnet of avg_S2, if denom is 0 
                                      # then NA is replaced by 0.
                                      # The reason is because in that case, 
                                      # S_count2[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC2 <- (S_count2 < ((K - burn_in + 1)/2))*1
  S_hat2 <- avg_S2 * S_predicted_MCC2
  
  ## Getting L_hat_IALM and S_hat_IALM, estimates of L_star and S_star respectively
  ## for Method 3: Inexact ALM - rrpca.
  library(rsvd)
  L_hat_IALM <- rrpca(Y, maxiter = 500)$L  # default maxiter = 50.
  S_hat_IALM <- rrpca(Y, maxiter = 500)$S
  S_predicted_MCC_IALM <- (S_hat_IALM != 0)*1
  
  ## Getting L_hat_rpca and S_hat_rpca, estimates of L_star and S_star respectively 
  ## for Method 4: ALM - Candes.
  detach("package:rsvd", unload = T)
  library(rpca)
  L_hat_rpca <- rpca(Y, max.iter = 5000)$L  # default max.iter = 5000.
  S_hat_rpca <- rpca(Y, max.iter = 5000)$S
  S_predicted_MCC_rpca <- (S_hat_rpca != 0)*1
  
  ## Print the outputs.
  print("The value of n and p")
  print(n)
  print(p)
  print("Does rpca function (ALM algorithm described in candes paper) converge for by default max iteration = 5000? If TRUE then YES")
  print(rpca(Y)$convergence$converged)     # rpca(Y, max.iter = m)$convergence$converged, if m is not 5000.
  print("Number of performed iterations")
  print(rpca(Y)$convergence$iterations)    # rpca(Y, max.iter = m)$convergence$iterations, if m is not 5000.
  
  ## Finding sensitivity and specificity for all Methods:
  S_actual_factor <- factor(as.factor(S_actual_MCC), levels = c("0", "1"))  ## for the Actual data.
  S_predic_factor <- factor(as.factor(S_predicted_MCC), levels = c("0", "1"))  ## for Method 1: q = 0.95
  S_predic_factor2 <- factor(as.factor(S_predicted_MCC2), levels = c("0", "1"))  ## for Method 2: q = 0.90
  S_predic_factor_IALM <- factor(as.factor(S_predicted_MCC_IALM), levels = c("0", "1"))  ## for Method 3: Inexact ALM - rrpca.
  S_predic_factor_rpca <- factor(as.factor(S_predicted_MCC_rpca), levels = c("0", "1"))  ## for Method 4: ALM - Candes.
  
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
  ## 0 is taken as 0(negative) and non-zero values are taken as 1 (positive).
  print("Mattews Correlation Coefficient(MCC) for S for Method 1: q = 0.95")
  print(mccr(S_actual_MCC, S_predicted_MCC))
  print("Mattews Correlation Coefficient(MCC) for S for Method 2: q = 0.90")
  print(mccr(S_actual_MCC, S_predicted_MCC2))
  print("Mattews Correlation Coefficient(MCC) for S for Method 3: Inexact ALM - rrpca")
  print(mccr(S_actual_MCC, S_predicted_MCC_IALM))
  print("Mattews Correlation Coefficient(MCC) for S for Method 4: ALM - Candes")
  print(mccr(S_actual_MCC, S_predicted_MCC_rpca))
  
  ## Finding Relative Ratio for L and S for all Methods:
  print("Relative Ratio for L and S for Method 1: q = 0.95")
  print(norm(L_star - L_hat, type = "F")/ norm(L_star, type = "F"))  # Frobenius norm distance
  print(norm(S_star - S_hat, type = "F")/ norm(S_star, type = "F"))
  print("Relative Ratio for L and S for Method 2: q = 0.90")
  print(norm(L_star - L_hat2, type = "F")/ norm(L_star, type = "F"))
  print(norm(S_star - S_hat2, type = "F")/ norm(S_star, type = "F"))
  print("Relative Ratio for L and S for Method 3: Inexact ALM - rrpca")
  print(norm(L_star - L_hat_IALM, type = "F")/ norm(L_star, type = "F"))
  print(norm(S_star - S_hat_IALM, type = "F")/ norm(S_star, type = "F"))
  print("Relative Ratio for L and S for Method 4: ALM - Candes")
  print(norm(L_star - L_hat_rpca, type = "F")/ norm(L_star, type = "F"))
  print(norm(S_star - S_hat_rpca, type = "F")/ norm(S_star, type = "F"))
  
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
    ## About ggplot below: when inside a loop, you have to print your ggplot object. 
    ## If outside a loop, without print function works fine.
    print(ggplot(daf, aes(iterations)) + 
            geom_line(aes(y = mcmc(dbar_array[i,])), colour="Black") +  # 1st layer of plot with q = 0.95
            geom_line(aes(y = mcmc(dbar_array2[i,])), colour="Blue") +  # 2nd layer of plot with q = 0.90
            geom_hline(yintercept = d_star[i], colour="Red") +  # 3rd layer with the actual value of d_star[i].
            theme_gray() +
            labs(title = paste("Trace Plot of d_bar[", i, "]", 
                               ", when n =", n, 
                               "\n Black: q = 0.95, Blue: q = 0.90, Red: d_star[", i, "]", 
                               "=", round(d_star[i], digits = 5)), 
                 y = paste("d_bar[", i, "]")) + 
            theme(plot.title = element_text(hjust = 0.5)) +
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
  print("Absolute distance between Sigma2_bar and 0.01 at the last iteration for Method 1")
  print(distance_Sigma2)
  print("Absolute distance between Sigma2_bar2 and 0.01 at the last iteration for Method 2")
  print(distance_Sigma22)
  
}  # end of the for loop: for(n in n_values_seq)

dev.off()
