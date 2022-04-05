## We are trying to decompose a noisy image matrix into low-rank component (foreground),
## sparse component and noise component (background).
## This file contains codes for generating the plot of Frobenius norm (L^2-norm) of the 
## overall error Vs. rank of the low-rank component.

## Actual data is generated according to 'Agarwal, Negahban and Wainwright (2012)' paper.
## See lines 95-105 and 112-128.


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


#------------------------------------------------------------------------------------
## Source all the necessary functions from the corresponding .cpp file.
#------------------------------------------------------------------------------------
sourceCpp("Functions_Agarwal_compare.cpp")


#------------------------------------------------------------------------------------
## User provided values
#------------------------------------------------------------------------------------

## Value of q.
q <- (1 - 0.2171)   # For any n, Actual data Y_(n*n) (in particular, S_star_(n*n)) 
                    # will be generated according to this q.

## Value of n may be taken in between the range from 100 to 3000. 
## n_values_seq <- seq.int(from = 100, to = 3000, by = 100)

## WE ARE CONSIDERING n*n SQUARE MATRICES ONLY.

## Other input values
a <- 2 # shape parameter for the inverse gamma distribution of sigma^2 (should be > 0)
b <- 1 # rate parameter for the inverse gamma distribution of sigma^2 (should be > 0)
K <- 13000  # Total number of iterations.
tow2 <- 20  # the variance tow^2 of the normal distribution in the mixture prior of S.
burn_in <- 5000   # Number of burn-ins.
rank_ratio <- seq(from = 0.05, to = 0.10, by = 0.05)


#------------------------------------------------------------------------------------
## Create a pdf file containing the plot of Frobenius norm (L^2-norm) of the 
## overall error Vs. rank of the low-rank component.
#------------------------------------------------------------------------------------
pdf('Agarwal_error_vs_rank.pdf', width = 11.694, height = 8.264)


#------------------------------------------------------------------------------------
## Start the simulation.
#------------------------------------------------------------------------------------
rank_vec <- rep(0, length(rank_ratio))
error_vec <- rep(0, length(rank_ratio))

## For our illustration purpose, we only consider n = 100. However, we can take any
## other positive integer value of n (lines 62-63), and get the corresponding plots. 
## Please change the value of n in line 92 below accordingly.
n <- 100
p <- n

## Actual data is generated according to 'Agarwal, Negahban and Wainwright (2012)' paper.

## Getting E_star
Sigma2_star <- 1/(n^2)
E_star <- matrix(rnorm(n*p, mean = 0, sd = sqrt(Sigma2_star)), nrow = n, ncol = p)

## Getting S_star (according to q = 1 - 0.2171)
S_star_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
S_star_runif <- matrix(runif(n*p), nrow = n, ncol = p)
S_actual_MCC <- (S_star_runif > q)*1
S_star <- S_star_rnorm * S_actual_MCC

for(j in 1:length(rank_ratio)) {
  
  r <- round(rank_ratio[j] * n)
  rank_vec[j] <- r
  
  ## Getting U_star of order n*r.
  ## Generate a random orthonormal matrix of order n*n. 
  ## The randomness is meant w.r.t (additively invariant) Haar measure on O(n).
  U_star <- randortho(n, type = "orthonormal")[ ,1:r] # takes the first r many columns.
  
  ## Similarly getting V_star of order p*r.
  V_star <- randortho(p, type = "orthonormal")[ ,1:r]
  
  ## Getting D_star of order r*r
  d_star <- c(runif(r-1, min = 1, max = 2), runif(1, min = 0.5, max = 1.5))
  D_star <- diag(cumprod(d_star))
  
  ## Getting L_star of order n*p
  L_star <- U_star %*% D_star %*% t(V_star)
  
  ## Define the true model
  Y <- (U_star %*% D_star %*% t(V_star)) + S_star + E_star
  
  ## AFTER GENERATING TRUE DATA Y
  
  ## Simulation of E from the prior of sigma^2
  Sigma20 <- rinvgamma(1, shape = a, rate = b)  # same as rcpp_only_rinvgamma(a, b). 
  
  ## Simulation of S from the prior which is 0 with probability q [where, q = 1 - 1/p], 
  ## and N(0,tow2) with probability (1-q).
  ## Also getting intial value of S_count.
  S0_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
  S0_runif <- matrix(runif(n*p), nrow = n, ncol = p)
  S0_MCC <- (S0_runif > q)*1
  S0 <- S0_rnorm * S0_MCC
  S0_count <- (S0_runif < q)*1  # For any element of S0_count, it puts 1 when the 
                                # corresponding element of S0 is 0.
  
  ## Simulation of D of order r*r from prior of d.
  d0 <- c(rtruncnorm(r-1, a=1, b=Inf, mean=0, sd=1), 
          rtruncnorm(1, a=0, b=Inf, mean=0, sd=1))
  D0 <- diag(cumprod(d0))
  
  ## Simulation of U of order n*r from uniform prior on the stiefel manifold R^(n*r)
  U0 <- rustiefel(n,r)
  ## Simulation of V of order p*r from uniform prior on the stiefel manifold R^(p*r)
  V0 <- rustiefel(p,r)
  ## Value of L.
  L0 <- U0 %*% D0 %*% t(V0)
  
  ## Simulation from the full conditional posterior distributions with K many iterations.
  ## Initially we take the input values of U, V, d, D, S, Sigma2 and L to be above 
  ## values which we got initially after simulation from their respective priors.
  
  ## BURN IN 
  Simul_burnin_list <- Simul_burnin_tn(Y, U0, D0, V0, S0, d0, Sigma20, 
                                        tow2, q1, n, p, r, burn_in, a, b)
  
  ## Outputs after iterating for 1:BurnIn.
  U0 <- Simul_burnin_list[["U"]]
  V0 <- Simul_burnin_list[["V"]]
  d0 <- Simul_burnin_list[["d"]]
  D0 <- Simul_burnin_list[["D"]]
  L0 <- Simul_burnin_list[["L"]]
  S0 <- Simul_burnin_list[["S"]]
  S0_count <- Simul_burnin_list[["S_count"]]
  Sigma20 <- Simul_burnin_list[["Sigma2"]]
  
  ## POST BURN IN
  Simul_afterburnin_list <- Simul_after_burnin_tn(Y, U0, D0, V0, L0, S0, 
                                                  S0_count, d0, Sigma20, 
                                                  tow2, q1, n, p, r, K, 
                                                  burn_in, a, b)
  
  L_hat <- Simul_afterburnin_list[["L_hat"]]
  S_count <- Simul_afterburnin_list[["S_count"]]
  S_sum <- Simul_afterburnin_list[["S_sum"]]
  
  
  ## Outputs after iterating for (BurnIn + 1):(Total iteration - BurnIn).
  ## Getting S_hat, estimate of S_star.
  J <- matrix(1, nrow = n, ncol = p)  # matrix whose all elements are 1
  avg_S <- S_sum / ((K - burn_in + 1)*J - S_count)
  avg_S[!is.finite(avg_S)] <- 0       # For any elemnet of avg_S, if denom is 0 
                                      # then NA is replaced by 0.
                                      # The reason is because in that case, 
                                      # S_count[i,j] = (K+1), which is >= (K+1)/2.
  S_predicted_MCC <- (S_count < ((K - burn_in + 1)/2))*1
  S_hat <- avg_S * S_predicted_MCC
  
  ## Update the jth element of the Frobenius norm error vector.
  error_vec[j] <- (norm(L_star - L_hat, type = "F"))^2 + (norm(S_star - S_hat, type = "F"))^2
}  


#------------------------------------------------------------------------------------
## Plot.
#------------------------------------------------------------------------------------
daf <- data.frame(rank_vec, error_vec)
## About ggplot below: when inside a loop, you have to print your ggplot object. 
## If outside a loop, without print function works fine.
print(ggplot(daf, aes(rank_vec)) + 
        geom_line(aes(y = error_vec), colour="Blue") +
        theme_gray() +
        labs(title = paste("Error vs. rank"), 
             y = paste("Frobenius norm error squared")) + 
        theme(plot.title = element_text(hjust = 0.5)) +  # By default, title is left aligned in 
                                                         # ggplot. Use it to center the title.
        coord_cartesian(ylim = c(0, 40)))

dev.off()
