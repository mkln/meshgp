rm(list=ls())
library(spNNGP)
library(meshgp)
library(tidyverse)
library(magrittr)

set.seed(2020)

SS <- 150 # coord values for jth dimension 
n <- SS^2 # tot obs

xlocs <- seq(0.0, 1, length.out=SS+2) %>% head(-1) %>% tail(-1)
coords <- expand.grid(xlocs, xlocs) %>% 
  arrange(Var1, Var2)
cx <- coords %>% as.matrix()
ix <- 1:nrow(cx) -1

# data gen
X <- as.matrix(cbind(rnorm(n), rnorm(n)))
Z <- matrix(1, nrow=n)
B <- as.matrix(c(-.2,.6))
p <- length(B)

sigma.sq <- 1.5
tau.sq <- .05
phi <- 5

# sample from full GP
#system.time(R12 <- xCovHUV(cx, ix, ix, c(sigma.sq, phi), matrix(0), T))
#system.time(w <- meshgp::mvn(1, rep(0,n), R12) %>% t())

# sampling from QMGP directly, use for SS>100
Mv_true <- round(c(SS/10, SS/10))
system.time(w <- meshgp::rqmeshgp(cx, Mv_true, c(sigma.sq, phi), matrix(0)))
w <- w-mean(w) 

# plot spatial process
coords %>% cbind(w) %>% plotspatial()

# generate output
y_full <- X%*%B + w + sqrt(tau.sq) * rnorm(n)
X_full <- X

# make some na: 0=na
lna <- 1:n %>% sapply(function(x) ifelse(rbinom(1, 1, .9)==1, 1, NA))
y <- y_full * lna

simdata <- coords %>% 
  cbind(w, y) %>% 
  as.data.frame() 

mcmc_keep <- 1000
mcmc_burn <- 1000
mcmc_thin <- 2

# MESH
Mv <- c(30, 30)
set.seed(1)
mesh_time <- system.time({
  meshout <- meshgp(y, X, Z, coords, Mv=Mv,
                     mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin),
                     num_threads = 11)
})
print(mesh_time) 
#   user  system elapsed 
#221.240   1.645  21.388 

(beta_est <- meshout$beta_mcmc %>% apply(1, mean))
(sigmasq_est <- meshout$sigmasq_mcmc %>% mean())
(tausq_est <- meshout$tausq_mcmc %>% mean())
(theta_est <- meshout$theta_mcmc %>% apply(1, mean))

meshgp_df <- coords %>% 
  cbind(meshout$w_mcmc %>% meshgp::list_mean(),
        meshout$yhat_mcmc %>% meshgp::list_mean()) %>%
  as.data.frame() %>% 
  arrange(Var1, Var2) 
colnames(meshgp_df)[3:4] <- c("w_meshgp", "y_meshgp")

# results
results <- simdata %>% left_join(meshgp_df) %>%
  arrange(Var1, Var2)

results %>% plotspatial(nrow=2)


###############################################
# Compare with NNGP
###############################################
n.samples <- mcmc_keep*mcmc_thin + mcmc_burn
starting <- list("phi"=3/0.5, "sigma.sq"=50, "tau.sq"=1)
tuning <- list("phi"=0.1, "sigma.sq"=0.1, "tau.sq"=0.1)
priors.1 <- list("beta.Norm"=list(rep(0,p), diag(1e3,p)),
                 "phi.Unif"=c(.01, 3/0.1), "sigma.sq.IG"=c(2, 2),
                 "tau.sq.IG"=c(2.01, 1))
cov.model <- "exponential"
verbose <- TRUE

n.neighbors <- 10
nngp_time <- system.time({
    m.s <- spNNGP::spNNGP(y_full[!is.na(lna)] ~ X_full[!is.na(lna),] - 1, 
                  coords=coords[!is.na(lna),] %>% as.matrix(), starting=starting, method="sequential", n.neighbors=n.neighbors,
                  tuning=tuning, priors=priors.1, cov.model="exponential", return.neighbor.info=T,
                  n.samples=n.samples, n.omp.threads=7)
    nngp_pred <- spNNGP::spPredict(m.s, X_full[is.na(lna),], coords=coords[is.na(lna),] %>% as.matrix(), n.omp.threads=7)
  }
)
print(nngp_time)
#   user  system elapsed 
#734.666   0.716 110.624 

print(mesh_time / nngp_time)

w_nngp <- m.s$p.w.samples %>% apply(1, mean)
nngp_df_in <- coords[!is.na(lna),] %>% cbind(w_nngp)
nngp_df <- coords[is.na(lna),] %>% cbind(nngp_pred$p.w.0 %>% apply(1, mean)) 
colnames(nngp_df)[3] <- "w_nngp"
nngp_df %<>% rbind(nngp_df_in)

results %<>% left_join(nngp_df)


perf_compare <- results %>% select(Var1, Var2, w, w_meshgp, w_nngp, y)

# RMSE in recovering latent spatial process
perf_compare %>% filter(!complete.cases(y)) %$% mean((w-w_meshgp)^2)
perf_compare %>% filter(!complete.cases(y)) %$% mean((w-w_nngp)^2)

perf_compare %>% plotspatial(nrow=2)

