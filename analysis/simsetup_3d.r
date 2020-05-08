rm(list=ls())
library(tidyverse)
library(magrittr)
library(gapfill)

#devtools::install_github("mkln/meshgp")
library(meshgp)


set.seed(1) 
# use powers of two to make layering easy for now.
dd <- 3 # dimension
SS <- 40 # coord values for jth dimension 
TT <- 10
nr <- SS^2 * TT # tot obs

xlocs <- seq(0.0, 1, length.out=SS+2) %>% head(-1) %>% tail(-1)
tlocs <- seq(0.0, 1, length.out=TT+2) %>% head(-1) %>% tail(-1)
coords <- expand.grid(xlocs, xlocs, tlocs) %>% arrange(Var1, Var2, Var3)
cx <- coords %>% as.matrix()
uni_time <- unique(coords$Var3)

dnr <- coords$Var1 %>% unique() %>% length()
dnc <- coords$Var2 %>% unique() %>% length()
dns <- coords$Var3 %>% unique() %>% length()


# data gen
## parameter grid
sigmasq_vals <- 1
tausq_vals <- c(.001, .05, .1)
a_vals <- c(5, 50, 500) # time a
beta_vals <- c(0.05, 0.5, .95) # separability
c_vals <- c(1, 5, 25) # space c

pars_grid <- expand.grid(sigmasq_vals, tausq_vals, a_vals, beta_vals, c_vals)
colnames(pars_grid) <- c("sigmasq", "tausq", "a_vals", "beta_vals", "c_vals")
results <- pars_grid
results %<>% mutate(mesh_abs = NA,
                    mesh_rmse = NA,
                    mesh_covg = NA,
                    gf_abs = NA,
                    gf_rmse = NA,
                    gf_covg = NA)

for(s in 1:nrow(pars_grid)){
  
  cat("Simulation", s, '\n')
  
  X <- matrix(1, nrow=nr)
  colnames(X) <- "X_1"
  Z <- matrix(1, nrow=nr)
  colnames(Z) <- "Z_1"
  p <- 1
  
  
  sigma.sq <- pars_grid[s, "sigmasq"]
  tau.sq <- pars_grid[s, "tausq"]
  aa <- pars_grid[s, "a_vals"]
  bb <- pars_grid[s, "beta_vals"]
  cc <- pars_grid[s, "c_vals"]
  Dmat <- matrix(0, ncol=1)
  
  print(pars_grid[s,])
  
  cxi <- 1:nrow(cx)-1
  
  #w <- meshgp::rqmeshgp(cx, c(10,10,5), c(sigma.sq, aa, bb, cc), Dmat)
  
  
  R123 <- xCovHUV(cx, cxi, cxi, c(sigma.sq, aa, bb, cc), Dmat, T)#,
  w <- meshgp::mvn(1, rep(0,nr), R123) %>% t()

  
  y_full <- w + sqrt(tau.sq) * rnorm(nr) 
  
  # make some na: 0=na
  na_which <- 1:nr 
  full_times <- uni_time %>% sample(2)
  missing_times <- uni_time %>% setdiff(full_times) %>% sample(2)
  for(mt in missing_times){
    selector <- coords$Var3 %in% mt
    na_which[selector] <- NA
    save_some <- which(selector) %>% sample(10)
    na_which[save_some] <- 1
  }
  
  nonmissing_times <- uni_time %>% setdiff(full_times) %>% setdiff(missing_times)
  for(nmt in nonmissing_times){
    circle_center <- runif(2, 0, 1)
    
    missing_circle <- ((coords$Var1-circle_center[1])^2 + 
                         (coords$Var2-circle_center[2])^2 < .05) & 
      (coords$Var3 %in% nmt)
    na_which[missing_circle] <- NA
  }
  
  
  y <- y_full * na_which#[!is.na(na_which)]# 
  
  p <- ncol(X)
  
  
  #########
  # Q-MGP #
  #########
  
  mcmc_keep <- 1000
  mcmc_burn <- 5000
  mcmc_thin <- 2
  
  set.seed(1)
  mesh_time <- system.time({
    test <- meshgp(y, X, Z, coords, axis_partition=c(10, 10, 5), 
                   mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin),
                   num_threads = 11,
                   debug       = list(sample_beta=F, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T))
  })
  
  (beta_est <- test$beta_mcmc %>% apply(1, mean))
  (sigmasq_est <- test$sigmasq_mcmc %>% mean())
  (tausq_est <- test$tausq_mcmc %>% mean())
  (theta_est <- test$theta_mcmc %>% apply(1, mean))
  
  meshgp_df <- coords %>% 
    cbind("y_meshgp" = test$yhat_mcmc %>% list_mean(),
          "y_meshgp_lo" = test$yhat_mcmc %>% list_qtile(.05),
          "y_meshgp_hi" = test$yhat_mcmc %>% list_qtile(.95)) %>%
    as.data.frame() %>% 
    arrange(Var1, Var2, Var3) 
  
  ###########
  # Gapfill #
  ###########
  
  dfc <- coords %>% cbind(y)
  
  darray <- dfc %>% 
    split(dfc$Var3) %>% 
    lapply(function(x) x %>% select(-Var3) %>% spread(Var2, y) %>% select(-Var1) %>% as.matrix()) %>%
    abind::abind(along=3)
  
  gfarray <- array(0, dim=c(dnr, dnc, 1, dns))
  gfarray[,,1,] <- darray
  
  
  gapfill_time <- system.time({
    filled_all <- Gapfill(gfarray, verbose=T, 
                          dopar = F,
                          nPredict = 3, predictionInterval = TRUE) %$% fill 
    filled <- filled_all[,,1,,1]
    ciLo <- filled_all[,,1,,2]
    ciUp <- filled_all[,,1,,3]
  })
  
  filled_long <- filled %>% 
    plyr::alply(3, function(x) reshape2::melt(x)) %>%
    bind_rows()
  Lo_long <- ciLo %>% 
    plyr::alply(3, function(x) reshape2::melt(x)) %>%
    bind_rows()
  Up_long <- ciUp %>% 
    plyr::alply(3, function(x) reshape2::melt(x)) %>%
    bind_rows()
  gfresults <- coords %>% as.data.frame() %>% 
    arrange(Var3, Var2, Var1) %>% 
    cbind(filled_long$value, Lo_long$value, Up_long$value) %>% as.data.frame()
  colnames(gfresults) <- c("Var1", "Var2", "Var3", "y_gf", "y_gf_lo", "y_gf_hi")
  
  
  ###########
  # Results #
  ###########
  
  analysis_plot_df <- coords %>% cbind(y, y_full) %>% rename(y_miss=y, y=y_full) %>%
    left_join(meshgp_df) %>% left_join(gfresults) %>%
    mutate(mesh_inrange = (y_meshgp_lo<y) & (y<y_meshgp_hi),
           gf_inrange = (y_gf_lo<y) & (y<y_gf_hi))
  
  analysis_plot_df %>% filter(!complete.cases(y_miss)) %>% 
    select(y, y_meshgp) %>% as.matrix() %>% cor()
  
  results[s, "mesh_rmse"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% sqrt(mean((y-y_meshgp)^2))
  results[s, "gf_rmse"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% sqrt(mean((y-y_gf)^2))
  
  results[s, "mesh_abs"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(abs(y-y_meshgp))
  results[s, "gf_abs"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(abs(y-y_gf))
  
  results[s, "mesh_covg"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(mesh_inrange)
  results[s, "gf_covg"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(gf_inrange)
  
  print(results[1:s,])
}

#save(file="postprocess/mesh_vs_gapfill_gp_3d_nonsep_onlyclouds.RData", list=c("results"))

# sample plot
analysis_plot_df %>% filter(Var3==unique(Var3)[6]) %>% 
  select(-Var3, -contains("inrange"), -contains("_hi"), -contains("_lo")) %>% plotspatial()



