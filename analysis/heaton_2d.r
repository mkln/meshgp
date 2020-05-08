rm(list=ls())
library(tidyverse)
library(magrittr)

#devtools::install_github("mkln/meshgp")
library(meshgp)

set.seed(ss <- 1)

simdata <- F
Mv <- c(50, 30)

#for(simdata in c(F, T)){
#  for(Mv in list(c(25, 15), c(50, 30), c(100, 60))){
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
    X <- cbind(coords)
    Z <- matrix(1, nrow=nr)
    ybar <- mean(simdf$y, na.rm=T)
    y <- simdf$y - ybar
    
    
    mcmc_keep <- 1000
    mcmc_burn <- 1000
    mcmc_thin <- 2
    
    mesh_mcmc <- list(keep=mcmc_keep, burn=mcmc_burn, thin=mcmc_thin)
    mesh_settings  <-  list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                            reference_full_coverage=F, 
                            verbose=F, debug=F, 
                            printall=T, saving=F)
    mesh_debug <- list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
    
    set.seed(1)
    mesh_time <- system.time({
      meshout <- meshgp(y, X, Z, coords, axis_partition=Mv, 
                        mcmc = mesh_mcmc,
                        settings = mesh_settings,
                        num_threads = 10)
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
            yhat_mcmc %>% list_mean() %>% add(ybar),
            yhat_mcmc %>% list_qtile(0.025) %>% add(ybar),
            yhat_mcmc %>% list_qtile(.975) %>% add(ybar)) %>%
      as.data.frame() %>% 
      arrange(Var1, Var2) 
    
    colnames(meshgp_df) <- c("Var1", "Var2", paste0("w_meshgp_", 1:ncol(Z)), 
                             "y_meshgp", "y_meshgp_low", "y_meshgp_hi")
    
    
    model_compare <- coords %>% cbind(simdf$y_full, y) %>% as.data.frame() %>%
      rename(y_full = V3) %>% left_join(meshgp_df) %>%
      arrange(Var1, Var2) 
    
    outsample <- model_compare %>% filter(!complete.cases(y)) 
    outsample %$% sqrt(mean((y_meshgp-y_full)^2, na.rm=T))
    outsample %$% mean(abs(y_meshgp-y_full), na.rm=T)
    outsample %$% mean((y_meshgp_low<y_full) & (y_full<y_meshgp_hi), na.rm=T)
    
    #save(file="postprocess/heaton/heaton_{label}_perf_{Mv[1]}x{Mv[2]}.RData" %>% glue::glue(), 
    #     list=c("outsample", "Mv", "model_compare", "mesh_mcmc", "mesh_settings", "mesh_debug",
    #            "meshout", "mesh_time", "meshgp_df"))
    
#  }
#}

###############
# Postprocess #
###############

plotdf <- model_compare %>% mutate(y = y + ybar) %>%
  select(Var1, Var2, contains("w_")) %>%
  gather(z, zvalue, -Var1, -Var2) 


gradient_colors <- c("#603511", "#B1853E",
                     "#FDFECE", "#52c234", 
                     "#476C1E", "#061700", "black")


(plotted <- ggplot(plotdf, aes(Var1, Var2, fill=zvalue)) + geom_raster() +
    theme_minimal() + facet_grid(~z) +
    theme(legend.position="none") + labs(x=NULL, y=NULL) +
    #scale_fill_viridis_c(option="E") + 
    scale_fill_gradientn(colours=gradient_colors,
                         na.value="white") +
    
    ggtitle("{tools::toTitleCase(label)} data from Heaton et al. (2019)" %>% glue::glue()))

#ggsave(filename="figures/heaton_{label}_results_{Mv[1]}x{Mv[2]}.png" %>% glue::glue(), 
#       plot=plotted, width=10, height=3, units="in")



