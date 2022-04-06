library(rstan)
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

q1 <- 0.95  # For any n, Actual data Y_(n*n) (in particular, S_star_(n*n)) 
            # will be generated according to q = 0.95
q2 <- 0.90  # value of q for Method 2, which is q = 0.90

# n_values_seq <- seq.int(from = 100, to = 3000, by = 100)  # n = 100:100:3000
n <- 1000
p <- n
r <- round(0.05*n)

print(paste("n is :", n))
print(paste("p is :", p))
print(paste("r is :", r))

## Other input values
a <- 4  # shape parameter for the inverse gamma distribution of sigma^2 (should be > 0).
b <- 5  # rate parameter for the inverse gamma distribution of sigma^2 (should be > 0).
tow2 <- 20   # the variance tow^2 of the normal distribution in the mixture prior of S.
tow02 <- 10^(-4)


## Getting E_star where each entry is iid N(0, True_Sigma^2)
Sigma2_star <- 0.01
E_star <- matrix(rnorm(n*p, mean = 0, sd = sqrt(Sigma2_star)), nrow = n, ncol = p)

## Getting S_star (according to q1 = 0.95)
S_star_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
S_star_runif <- matrix(runif(n*p), nrow = n, ncol = p)
S_actual_MCC <- (S_star_runif > q1)*1
S_star <- S_star_rnorm * S_actual_MCC

## Getting U_star of order n*r.
## First generate a random orthonormal matrix of order n*n. Then take the first r many columns.
## The randomness is meant w.r.t (additively invariant) Haar measure on O(n).
U_star <- randortho(n, type = "orthonormal")[ ,1:r]

## Getting V_star of order p*r
V_star <- randortho(p, type = "orthonormal")[ ,1:r]

## Getting D_star of order r*r
d_star <- c(runif(r-1, min = 1, max = 2), runif(1, min = 0.5, max = 1.5))
D_star <- diag(cumprod(d_star[r:1])[r:1])

## Getting L_star of order n*p
L_star <- U_star %*% D_star %*% t(V_star)

## True Data
Y <- (U_star %*% D_star %*% t(V_star)) + S_star + E_star


## Simulation using Stan programming.

Simulation.from.stan <- function(n, p, r, q, tow02, tow2, a, b, Y){
  # Compile the model
  Simul_model <- stan_model('Rstan_Simulation_origform.stan')
  
  # Initialize the parameters
  init_fn <- function(i){
    X_10 <- matrix(rnorm(n*r), nrow = n, ncol = r)
    X_20 <- matrix(rnorm(p*r), nrow = p, ncol = r)
    d0 <- c(rtruncnorm(r-1, a=1, b=Inf, mean=0, sd=1), 
            rtruncnorm(1, a=0, b=Inf, mean=0, sd=1))    # prior of d
    D0 <- diag(cumprod(d0[r:1])[r:1])
    S_rnorm <- matrix(rnorm(n*p, mean = 0, sd = sqrt(tow2)), nrow = n, ncol = p)
    S_runif <- matrix(runif(n*p), nrow = n, ncol = p)
    S_MCC <- (S_runif > q1)*1
    S0 <- S_rnorm * S_MCC
    sigma20 <- rinvgamma(1, shape = a, rate = b)
    U0 <- svd(X_10)$u %*% t(svd(X_10)$v)
    V0 <- svd(X_20)$u %*% t(svd(X_20)$v)
    S_U0 <- t(X_10)%*%X_10
    S_V0 <- t(X_20)%*%X_20
    L0 <- U0 %*% D0 %*% t(V0)
    return(list(X_1 = X_10, X_2 = X_20, d = d0, S = S0, sigma2 = sigma20, U = U0,  
                V = V0, S_U = S_U0, S_V = S_V0, L = L0))
  }
  
  chains_vec <- 1:4
  init_l <- lapply(chains_vec, function(i) init_fn(i))
  
  # Pass data to stan and run the model
  Simul_fit <- sampling(Simul_model, 
                        data = list(n=n, p=p, r=r, q = q1, tow02=tow02, tow2=tow2, a=a, b=b, Y=Y), 
                        pars = c("d","L","S","sigma2"), chains = 4, iter = 4000, warmup = 2000, 
                        thin = 1, init = init_l, include = TRUE, cores = getOption("mc.cores", 4L)
                        )
                        
  # Diagnose
  print(Simul_fit)
  
  # Extract the parameters.
  Simul_params <- extract(Simul_fit)
  print(str(Simul_params))
  
  d_array <- Simul_params$d
  S_array <- Simul_params$S
  L_array <- Simul_params$L
  sigma2_vec <- Simul_params$sigma2
  
  d_hat <- apply(d_array, 2, FUN = mean)
  S_hat <- apply(S_array, c(2,3), FUN = mean)
  L_hat <- apply(L_array, c(2,3), FUN = mean)
  sigma2_hat <- mean(sigma2_vec)
}
