rm(list=ls())

library(lubridate)
library(spNNGP)
library(tidyverse)
library(magrittr)

github_token <- "2be6ec16fcad254251bac09c231bfdf8009f4b43"
#devtools::install_github("mkln/meshgp", auth_token=github_token)
library(meshgp)

set.seed(ss <- 1)

simdata <- F
if(simdata){
  #load("data/AllSimulatedTemps.RData")
  load(url("https://github.com/finnlindgren/heatoncomparison/raw/master/Data/AllSimulatedTemps.RData"))
  dfname <- all.sim.data 
  label <- "simulated"
} else {
  #load("data/AllSatelliteTemps.RData")
  load(url("https://github.com/finnlindgren/heatoncomparison/raw/master/Data/AllSatelliteTemps.RData"))
  dfname <- all.sat.temps
  label <- "realdata"
}

mapcenter <- dfname %>% select(Lon, Lat) %>% apply(2, mean)
mapcenter_rpmt <- matrix(1, nrow=nrow(dfname), ncol=1) %x% t(mapcenter)
rangeLon <- diff(range(dfname$Lon))

simdf <- dfname %>% 
  rename(Var1 = Lon, Var2 = Lat,
         y=MaskTemp, y_full=TrueTemp) %>% arrange(Var1, Var2)

coords <- simdf %>% select(Var1, Var2) %>% as.matrix() %>% 
  add(-mapcenter_rpmt) %>% 
  multiply_by(1/rangeLon) %>% 
  add(.5)

nr <- nrow(coords)
X <- coords#matrix(1, nrow=nr) 
Z <- matrix(1, nrow=nr)
ybar <- mean(simdf$y, na.rm=T)
y <- simdf$y - ybar

mcmc_keep <- 500
mcmc_burn <- 1000
mcmc_thin <- 2

mesh_mcmc <- list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin)
mesh_settings <- list(adapting=T, mcmcsd=.1, cache=T, cache_gibbs=F, 
                      reference_full_coverage=F, verbose=F, debug=F, printall=T, seed=NULL)
mesh_starting <- list(beta=NULL, tausq=0.1, sigmasq=1, theta=NULL, w=NULL)


# MESH
coords %>% apply(2, function(x) x %>% unique %>% length)
Mv <- c(100,50)#c(25,15) # 
(nr / prod(Mv) * ncol(Z))

set.seed(1)
mesh_time <- system.time({
  meshout <- meshgp(y, X, Z, coords, Mv=Mv, 
                    mcmc = mesh_mcmc,
                    num_threads = 11,
                    settings    = mesh_settings,
                    starting    = mesh_starting,
                    debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T),
                    dry_run     = F,
                    recover     = list())
})

beta_mcmc <- meshout$beta_mcmc
sigmasq_mcmc <- meshout$sigmasq_mcmc
tausq_mcmc <- meshout$tausq_mcmc
theta_mcmc <- meshout$theta_mcmc
w_mcmc <- meshout$w_mcmc
yhat_mcmc <- meshout$yhat_mcmc

(beta_est <- beta_mcmc %>% apply(1, mean))
(sigmasq_est <- sigmasq_mcmc %>% mean())
(tausq_est <- tausq_mcmc %>% mean())
(theta_est <- theta_mcmc %>% apply(1, mean))

meshgp_df <- meshout$coords %>% 
  cbind(w_mcmc    %>% list_mean(),
        yhat_mcmc %>% list_mean()) %>%
  as.data.frame() %>% 
  arrange(Var1, Var2) 

colnames(meshgp_df) <- c("Var1", "Var2", paste0("w_meshgp_", 1:ncol(Z)), "y_meshgp")

meshgp_df %<>% 
  mutate(y_meshgp = y_meshgp + ybar) %>%
  mutate(y_meshgp_low = y_meshgp %>% quantile(.025),
         y_meshgp_hi  = y_meshgp %>% quantile(.975))

model_compare <- coords %>% cbind(simdf$y_full, y) %>% as.data.frame() %>%
  rename(y_full = V3) %>% left_join(meshgp_df) %>%
  arrange(Var1, Var2) 

outsample <- model_compare %>% filter(!complete.cases(y)) 
outsample %$% sqrt(mean((y_meshgp-y_full)^2, na.rm=T))
outsample %$% mean(abs(y_meshgp-y_full), na.rm=T)
outsample %$% mean((y_meshgp_low<y_full) & (y_full<y_meshgp_hi), na.rm=T)

#> outsample %$% sqrt(mean((y_meshgp-y_full)^2, na.rm=T))
#[1] 1.58725
#> outsample %$% mean(abs(y_meshgp-y_full), na.rm=T)
#[1] 1.133814
#> outsample %$% mean((y_meshgp_low<y_full) & (y_full<y_meshgp_hi), na.rm=T)
#[1] 0.9291764

#save(file="heaton_{label}_perf_{Mv[1]}x{Mv[2]}.RData" %>% glue::glue(), 
#     list=c("outsample", "M", "model_compare", "mesh_mcmc", "mesh_settings", "mesh_starting",
#            "meshout", "mesh_time"))

##################
# POSTPROCESSING #
##################
plotdf <- model_compare %>% mutate(y = y + ybar) %>%
  #select(-y_meshgp_hi, -y_meshgp_low, -contains("w_")) %>% 
  #rename(MaskTemp = y,
  #       TrueTemp = y_full,
  #       Predicted = y_meshgp) %>%
  select(Var1, Var2, contains("w_")) %>%
  gather(z, zvalue, -Var1, -Var2) 

(plotted <- ggplot(plotdf, aes(Var1, Var2, fill=zvalue)) + geom_raster() +
    theme_minimal() + facet_grid(~z) +
    theme(legend.position="none") + labs(x=NULL, y=NULL) +
    scale_fill_viridis_c() + ggtitle("{tools::toTitleCase(label)} data from Heaton et al. (2019)" %>% glue::glue()))

#ggsave(filename="heaton_{label}_results_{Mv[1]}x{Mv[2]}.png", plot=plotted, width=10, height=3, units="in")



