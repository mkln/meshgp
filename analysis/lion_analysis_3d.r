rm(list=ls())
library(tidyverse)
library(magrittr)
library(gapfill)
library(meshgp)


set.seed(2020)

load(file="analysis/lion_frames.RData")

lion_frames_sub <- lion_frames[c(-1, -2, -33, -34)] %>% lapply(function(x)
  x[seq(1,nrow(x),4), seq(1,ncol(x),4)])

lion_frames_sub2 <- lion_frames_sub %>% lapply(function(M) M[nrow(M):1, ncol(M):1])

lion_frames_full <- lion_frames_sub#[5:7]
lion_array_full <- lion_frames_full %>% abind::abind(along=3)
#### full data
lion_array_true <- seq_len(length(lion_frames_full)) %>% 
  lapply(function(i) lion_frames_full[[i]] %>% reshape2::melt() %>% as.data.frame() %>% mutate(Var3=i)) %>%
  bind_rows() 

lion_array_true %<>% mutate(Var1 = Var1/max(Var2),
                            Var3 = Var3/max(Var2),
                            Var2 = Var2/max(Var2))

lion_array_true %<>% rename(y = value) %>% arrange(Var1, Var2, Var3) %>% select(Var1, Var2, Var3, y)

# dataset with missing values
slicesize <- lion_frames_sub[[1]] %>% dim()

TT <- length(lion_frames_sub)
t_noisy <- 1:TT %>% sample(5, replace=F) %>% sort()
t_missing <- 1:TT %>% setdiff(t_noisy) %>% sample(5, replace=F) %>% sort()
t_cloud <- 1:TT %>% setdiff(t_noisy) %>% setdiff(t_missing) %>% sample(5, replace=F) %>% sort()

for(i in t_noisy){
  # random location of missing
  slicedNA <- matrix(runif(slicesize[1]*slicesize[2], 0, 1) > .5, ncol=slicesize[2])
  slicedNA[slicedNA == 0] <- NA
  lion_frames_sub[[i]] <- lion_frames_sub[[i]] * slicedNA
}

for(i in t_missing){
  # "almost all" missing since gapfill gives NA if all are missing
  slicedNA <- matrix(NA, nrow=slicesize[1], ncol=slicesize[2])
  avloc <- 1:(slicesize[1]*slicesize[2]) %>% sample(10, replace=F)
  slicedNA[avloc] <- 1
  lion_frames_sub[[i]] <- lion_frames_sub[[i]] * slicedNA
}

## cloud cover simulation
d1 <- dim(lion_frames_sub[[1]])[1]
d2 <- dim(lion_frames_sub[[1]])[2]
for(i in t_cloud){
  # cloud cover simulation
  
  melted <- lion_frames_sub[[i]] %>% reshape2::melt() 
  melted %<>% mutate(Var1 = Var1/d2, 
                     Var2 = Var2/d2)
  
  (circle_center <- c(runif(1, 0, .8), runif(1, 0, 1)))
  
  missing_circle <- (melted$Var1-circle_center[1])^2 + 
    (melted$Var2-circle_center[2])^2 < .1
  melted$value[missing_circle] <- NA
  recon <- melted %>% reshape2::dcast(Var1 ~ Var2) %>% `[`(,-1) %>% as.matrix()
  dimnames(recon) <- NULL
  lion_frames_sub[[i]] <- recon
}


lion_array_df <- seq_len(length(lion_frames_sub)) %>% 
  lapply(function(i) lion_frames_sub[[i]] %>% reshape2::melt() %>% as.data.frame() %>% mutate(Var3=i)) %>%
  bind_rows()

lion_array_df %<>% mutate(Var1 = Var1/max(Var2),
                          Var3 = Var3/max(Var2),
                          Var2 = Var2/max(Var2)) %>% 
  rename(y=value) %>%
  select(Var1, Var2, Var3, y) %>% 
  arrange(Var1, Var2, Var3)

# data gen 
coords <- lion_array_df %>% select(contains("Var")) %>% as.matrix()
dd <- 3 # dimension
nr <- nrow(lion_array_df)
X <- coords
Z <- matrix(1, nrow=nr)# %>% cbind(abs(rnorm(nr)))
p <- ncol(X)
y <- lion_array_df$y #+ rnorm(nr, 0, .1)

dims <- coords[,1:3] %>% apply(2, function(x) x %>% unique() %>% length())
Mv <- c(5,21,10) #layers per side
prod(dims/Mv) * ncol(Z) # size of blocks 

mcmc_keep <- 1000
mcmc_burn <- 10000
mcmc_thin <- 5
#load("lion_hires_recover.RData")

recover_data <- list()
mcmcsd <- .1
mesh_time <- system.time({
  meshout <- meshgp(y, X, Z,  
                    coords, Mv=Mv, 
                    mcmc = list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin),
                    num_threads = 11,
                    settings    = list(adapting=T, mcmcsd=mcmcsd, 
                                       cache=T, cache_gibbs=F, 
                                       reference_full_coverage=F, saving=T,
                                       verbose=F, debug=F, printall=T, seed=NULL),
                    dry_run     = F,
                    recover     = recover_data)
})

#save(list="meshout", file="postprocess/lion_qmgp_out.RData")

recover_data <- meshout$recover
mcmcsd <- meshout$paramsd

meshgp_df <- coords %>% #cbind(test$w_mcmc[,mcmc]) %>%#
  cbind(meshout$yhat_mcmc %>% list_mean()) %>%
  cbind(meshout$yhat_mcmc %>% list_qtile(0.05)) %>%
  cbind(meshout$yhat_mcmc %>% list_qtile(0.95)) %>%
  cbind(meshout$w_mcmc %>% list_mean()) %>%
  #cbind(meshout$w_mcmc %>% list_qtile(0.025)) %>%
  #cbind(meshout$w_mcmc %>% list_qtile(0.975)) %>%
  as.data.frame()


colnames(meshgp_df) <- c(
  "Var1", "Var2", "Var3",
  "yhat", "yhat_low", "yhat_hi",
  paste0("w_", 1:ncol(Z))#,
  #paste0("w_low_", 1:ncol(Z)),
  #paste0("w_hi_", 1:ncol(Z))
)


meshanalysis <- lion_array_true %>% left_join(lion_array_df %>% rename(y_miss=y)) %>% 
  left_join(meshgp_df)

meshanalysis %$% #filter(!complete.cases(.)) %$% 
  mean((yhat-y)^2)

sigmasq <- meshout$sigmasq_mcmc %>% mean
theta <- meshout$theta_mcmc %>% apply(1, mean)

covarplot(sigmasq, theta, hmax=sqrt(2), umax=1, v=0, hrange=1, urange=1, ngrid=100)


###############################################
# Gapfill
###############################################
library(gapfill)


lion_gf_array <- lion_frames_sub %>% abind::abind(along=3)
lion_nr <- dim(lion_gf_array)[1]
lion_nc <- dim(lion_gf_array)[2]
lion_ns <- dim(lion_gf_array)[3]
gfarray <- array(0, dim=c(lion_nr, lion_nc, 1, lion_ns))
gfarray[,,1,] <- lion_gf_array

gapfill_time <- system.time({
  #registerDoParallel(11)
  filled_all <- Gapfill(gfarray, dopar=F, verbose=T, 
                        nPredict = 3, predictionInterval = TRUE) %$% fill 
  filled <- filled_all[,,1,,1]
  ciLo <- filled_all[,,1,,2]
  ciUp <- filled_all[,,1,,3]
})
#save(file="gapfillsave.Rdata", list=c("filled_all"))

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


###############################################
# Model checking
###############################################

analysis_plot_df <- lion_array_true %>% left_join(lion_array_df %>% rename(y_miss=y)) %>% 
  left_join(meshgp_df) %>% left_join(gfresults)
#save(file="postprocess/lion_postdata.RData", list=c("analysis_plot_df"))


analysis_plot_df %<>% 
  mutate(mesh_inrange = (yhat_low<y) & (y<yhat_hi),
         gf_inrange = (y_gf_lo<y) & (y<y_gf_hi))

analysis_plot_df %>% filter(!complete.cases(y_miss)) %>% 
  select(y, yhat) %>% as.matrix() %>% cor()

results <- matrix(0, ncol=6)
colnames(results) <- c("mesh_rmse","gf_rmse","mesh_abs","gf_abs","mesh_covg","gf_covg")
results[1, "mesh_rmse"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% sqrt(mean((y-yhat)^2))
results[1, "gf_rmse"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% sqrt(mean((y-y_gf)^2))

results[1, "mesh_abs"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(abs(y-yhat))
results[1, "gf_abs"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(abs(y-y_gf))

results[1, "mesh_covg"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(mesh_inrange)
results[1, "gf_covg"] <- analysis_plot_df %>% filter(!complete.cases(y_miss)) %$% mean(gf_inrange)

#save(file="postprocess/lion_mesh_gapfill_results.RData", 
#     list=c("results", "meshout", "mesh_time", "gapfill_time", "filled_all", "analysis_plot_df"))
