meshgp_dev <- function(y, X, Z, coords, axis_partition, grid_size=NULL,
                   mcmc        = list(keep=1000, burn=0, thin=1),
                   num_threads = 7,
                   settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                                      reference_full_coverage=F, verbose=F, debug=F, 
                                      printall=F, saving=T),
                   prior       = list(set_unif_bounds=NULL,
                                      beta=NULL,
                                      sigmasq=NULL,
                                      tausq=NULL, toplim=1e5, btmlim=1e-5),
                   starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                   debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
                   ){

  if(F){
    mcmc        = list(keep=100, burn=100, thin=1)
    num_threads = 5
    settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                       reference_full_coverage=F, verbose=F, debug=F, 
                       printall=F, saving=T)
    prior       = list(set_unif_bounds=NULL,
                       beta=NULL,
                       sigmasq=NULL,
                       tausq=NULL, toplim=1e5, btmlim=1e-5)
    starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL)
    debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
    
    save.image("devel.RData")
    
  }
  #rm(list=ls())
  #load("devel.RData")
  #mcmc_verbose <- T
  #mcmc_debug <- T
  
  # init
  cat(" Bayesian MeshGP in-development\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n\n")
  
  Mv <- axis_partition
  
  if(1){
    mcmc_keep <- mcmc$keep
    mcmc_burn <- mcmc$burn
    mcmc_thin <- mcmc$thin
    
    mcmc_adaptive    <- settings$adapting
    mcmc_cache       <- settings$cache
    mcmc_cache_gibbs <- settings$cache_gibbs
    mcmc_verbose     <- settings$verbose
    mcmc_debug       <- settings$debug
    mcmc_printall    <- settings$printall
    
    if(is.null(settings$saving)){
      saving         <- F
    } else {
      saving         <- settings$saving
    }
    
    #rfc_dependence <- settings$reference_full_coverage
    
    sample_beta    <- debug$sample_beta
    sample_tausq   <- debug$sample_tausq
    sample_sigmasq <- debug$sample_sigmasq
    sample_theta   <- debug$sample_theta
    sample_w       <- debug$sample_w
    
    dd             <- ncol(coords)
    p              <- ncol(X)
    q              <- ncol(Z)
    k              <- q * (q-1)/2
    nr             <- nrow(X)
    
    if(length(Mv) == 1){
      Mv <- rep(Mv, dd)
    }
    avg_block <- nr/prod(Mv)*q
    cat(sprintf("~ %s intervals along the axes (total M=%d)", paste(Mv, collapse=" "), prod(Mv)),"\n")
    cat(sprintf("~ %s average block size", avg_block),"\n")
    
    if(avg_block > 100){
      #goon <- readline(prompt="Average block size is large. Iterations will be slow. Continue? Y/n: ")
      print("Average block size is large. Iterations will be slow. ")
      #goon <- tolower(goon)
      #if(!(goon %in% c("y", ""))){
      #  return(0)
      #}
    }
    
    if(is.null(starting$beta)){
      start_beta   <- rep(0, p)
    } else {
      start_beta   <- starting$beta
    }
    
    if(is.null(starting$sigmasq)){
      start_sigmasq <- 10
    } else {
      start_sigmasq  <- starting$sigmasq
    }
    
    space_uni      <- (dd==2) & (q==1)
    space_biv      <- (dd==2) & (q==2) 
    space_mul      <- (dd==2) & (q >2)  
    stime_uni      <- (dd==3) & (q==1)
    stime_biv      <- (dd==3) & (q==2) 
    stime_mul      <- (dd==3) & (q >2)  
    
    if(is.null(starting$theta)){
      if(dd == 3){
        # spacetime
        if(q > 2){
          # multivariate
          start_theta <- c(10, 0.5, 10, 0.5, 10)
          start_theta <- c(start_theta, rep(1, k))
        } else {
          if(q == 2){
            # bivariate
            start_theta <- c(10, 0.5, 10, 1)
          } else {
            # univariate
            start_theta <- c(10, 0.5, 10)
          }
        }
      } else {
        # space
        if(q > 2){
          # multivariate
          start_theta <- c(10, 0.5, 10)
          start_theta <- c(start_theta, rep(1, k))
        } else {
          if(q == 2){
            # bivariate
            start_theta <- c(10, 1)
          } else {
            # univariate
            start_theta <- c(10)
          }
        }
      }
    } else {
      start_theta  <- starting$theta
    }
    
    start_theta <- c(start_sigmasq, start_theta)
    
    if(is.null(prior$btmlim)){
      btmlim <- 1e-5
    } else {
      btmlim <- prior$btmlim
    }
    
    if(is.null(prior$toplim)){
      toplim <- 1e5
    } else {
      toplim <- prior$toplim
    }
    
    if(is.null(prior$set_unif_bounds)){
      if(dd == 3){
        # spacetime
        if(q > 2){
          # multivariate
          set_unif_bounds <- matrix(rbind(#c(btmlim, toplim), # sigmasq
                                          c(btmlim, toplim), # alpha_1
                                          c(btmlim, 1-btmlim), # beta_1
                                          c(btmlim, toplim), # alpha_2
                                          c(btmlim, 1-btmlim), # beta_2
                                          c(btmlim, toplim)), # phi
                                    ncol=2)
        } else {
          # bivariate or univariate
          set_unif_bounds <- matrix(rbind(#c(btmlim, toplim),
                                          c(btmlim, toplim),
                                          c(btmlim, 1-btmlim),
                                          c(btmlim, toplim)), ncol=2)
        }
      } else {
        # space
        if(q > 2){
          # multivariate
          set_unif_bounds <- matrix(rbind(#c(btmlim, toplim),
                                          c(btmlim, toplim),
                                          c(btmlim, 1-btmlim),
                                          c(btmlim, toplim)), ncol=2)
          
        } else {
          # bivariate or univariate
          set_unif_bounds <- matrix(rbind(#c(btmlim, toplim),
                                          c(btmlim, toplim)), ncol=2)
        }
      }
      
      if(q > 1){
        kk <- q * (q-1) / 2
        vbounds <- matrix(0, nrow=kk, ncol=2)
        
        if(q > 2){
          dlim <- sqrt(q+.0)
        } else {
          dlim <- toplim
        }
        vbounds[,1] <- btmlim;
        vbounds[,2] <- dlim - btmlim
        set_unif_bounds <- rbind(set_unif_bounds, vbounds)
      }
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    # for sigmasq
    set_unif_bounds <- rbind(c(btmlim, toplim),
                             set_unif_bounds)
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$sigmasq)){
      sigmasq_ab <- c(2.01, 1)
    } else {
      sigmasq_ab <- prior$sigmasq
    }
    
    if(is.null(prior$tausq)){
      tausq_ab <- c(2.01, 1)
    } else {
      tausq_ab <- prior$tausq
    }
    
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(length(start_theta)) * settings$mcmcsd
    } else {
      mcmc_mh_sd <- settings$mcmcsd
    }
    
    if(is.null(starting$tausq)){
      start_tausq  <- .1
    } else {
      start_tausq    <- starting$tausq
    }
    
    
    
    if(is.null(starting$w)){
      start_w <- rep(0, q*nr) %>% matrix(ncol=q)
    } else {
      start_w <- starting$w #%>% matrix(ncol=q)
    }
  }

  # data management
  if(is.null(colnames(X))){
    orig_X_colnames <- colnames(X) <- paste0('X_', 1:ncol(X))
  } else {
    orig_X_colnames <- colnames(X)
    colnames(X)     <- paste0('X_', 1:ncol(X))
  }
  
  if(is.null(colnames(Z))){
    orig_Z_colnames <- colnames(Z) <- paste0('Z_', 1:ncol(Z))
  } else {
    orig_Z_colnames <- colnames(Z)
    colnames(Z)     <- paste0('Z_', 1:ncol(Z))
  }
  
  colnames(coords)  <- paste0('Var', 1:dd)
  
  na_which <- ifelse(!is.na(y), 1, NA)
  simdata <- 1:nrow(coords) %>% cbind(coords) %>% 
    cbind(y) %>% cbind(na_which) %>% 
    cbind(X) %>% cbind(Z) %>% as.data.frame()
  colnames(simdata)[1] <- "ix"
  
  #################
  # create grid   # 
  data_already_on_grid <- F
  if(!data_already_on_grid){
    gs <- round(nrow(simdata)^(1/ncol(coords)))
    gsize <- if(is.null(grid_size)){ rep(gs, ncol(coords)) } else { grid_size }
    if(dd == 2){
      gridcoords <- expand.grid(seq(min(coords[,1]), max(coords[,1]), length.out=gsize[1]),
                                seq(min(coords[,2]), max(coords[,2]), length.out=gsize[2]))
      #plot(coords, pch=19, cex=.2)
      #points(gridcoords, pch=19, cex=.2, col="red")
    } else {
      gridcoords <- expand.grid(seq(min(coords[,1]), max(coords[,1]), length.out=gsize[1]),
                                seq(min(coords[,2]), max(coords[,2]), length.out=gsize[2]),
                                seq(min(coords[,3]), max(coords[,3]), length.out=gsize[3]))
    }
    simdata <- simdata %>% full_join(gridcoords %>% mutate(thegrid = 1))
  }
  
  #################
  
  if(dd == 2){
    simdata %<>% arrange(Var1, Var2)
    coords <- simdata %>% dplyr::select(Var1, Var2)
    colnames(simdata)[4:5] <- c("y", "na_which")
  } else {
    simdata %<>% arrange(Var1, Var2, Var3)
    coords <- simdata %>% dplyr::select(Var1, Var2, Var3)
    colnames(simdata)[5:6] <- c("y", "na_which")
  }
  
  if(!is.matrix(coords)){
    coords %<>% as.matrix()
  }
  
  #if(length(recover) == 0){
    # Domain partitioning and gibbs groups
  if(!data_already_on_grid){
    gridded_coords <- simdata %>% filter(thegrid==1) %>% dplyr::select(contains("Var")) %>% as.matrix()
    fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(gridded_coords[,i], Mv[i])) 
  } else {
    fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], Mv[i])) 
  }
    
    
    # guaranteed to produce blocks using Mv
    system.time(fake_coords_blocking <- coords %>% 
                  as.matrix() %>% 
                  gen_fake_coords(fixed_thresholds, 1) )
    
    cat("Tessellating")
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% as.matrix() %>%
                  tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% cbind(data.frame(na_which=simdata$na_which)) )
    cat(".\n")
    
    # check if some blocks come up totally empty
    
    blocks_prop <- coords_blocking[,paste0("L", 1:dd)] %>% unique()
    blocks_fake <- fake_coords_blocking[,paste0("L", 1:dd)] %>% unique()
    if(nrow(blocks_fake) != nrow(blocks_prop)){
      cat("Adding fake coords to avoid empty blocks ~ don't like? Use lower Mv\n")
      # with current Mv, some blocks are completely empty
      # this messes with indexing. so we add completely unobserved coords
      suppressMessages(adding_blocks <- blocks_fake %>% dplyr::setdiff(blocks_prop) %>%
                         left_join(fake_coords_blocking))
      coords_blocking <- bind_rows(coords_blocking, adding_blocks)
      if(dd == 2){
        coords_blocking %<>% arrange(Var1, Var2)
      } else {
        coords_blocking %<>% arrange(Var1, Var2, Var3)
      }
      nr_full <- nrow(coords_blocking)
    } else {
      nr_full <- nrow(coords_blocking)
    }
    start_w <- rep(0, q*nr_full) %>% matrix(ncol=q)
    
    #if(!rfc_dependence){
      #emptyblocks <- coords_blocking %>% 
      #  group_by(block) %>% 
      #  summarize(avail = mean(!is.na(na_which))) %>%
      #  dplyr::filter(avail==0)
      #newcol <- coords_blocking$color %>% max() %>% add(1)
      #coords_blocking %<>% mutate(color = ifelse(block %in% emptyblocks$block, newcol, color))
    #}
    #c_unique <- coords_blocking %>% dplyr::select(block, color) %>% unique()
    #ggroup <- c_unique %$% block
    #gcolor <- c_unique %$% color
    
    blocking <- coords_blocking$block %>% factor() %>% as.integer()
    
    # DAG
    cat("Building graph")
    suppressMessages(parents_children <- mesh_graph_build(coords_blocking, Mv, F))
    parents                      <- parents_children[["parents"]] 
    children                     <- parents_children[["children"]] 
    block_names                  <- parents_children[["names"]] 
    block_groups                 <- parents_children[["groups"]]#[block_names]
    indexing                     <- (1:nr_full-1) %>% split(blocking)
    cat(".\n")
    
    suppressMessages(simdata <- coords_blocking %>% dplyr::select(-na_which) %>% left_join(simdata))
    #return(list(simdata=simdata,
    #            parents=parents,
    #            children=children))
    
    predictable_blocks <- simdata %>% mutate(thegrid = ifelse(is.na(thegrid), 0, 1)) %>%
      group_by(block) %>% summarise(thegrid=mean(thegrid, na.rm=T), perc_availab=mean(!is.na(y))) %>%
      mutate(predict_here = ifelse(thegrid==1, perc_availab>0, 1)) %>% arrange(block) %$% predict_here
    
    
    indexing_grid_ids <- simdata$thegrid %>% split(blocking)
    indexing_grid <- list()
    indexing_obs <- list()
    for(i in 1:length(indexing)){
      indexing_grid[[i]] <- indexing[[i]][which(indexing_grid_ids[[i]] == 1)]
      indexing_obs[[i]] <- indexing[[i]][which(is.na(indexing_grid_ids[[i]]))]
    }
    # finally prepare data
    sort_ix     <- simdata$ix
    
    y           <- simdata$y %>% matrix(ncol=1)
    X           <- simdata %>% dplyr::select(contains("X_")) %>% as.matrix()
    colnames(X) <- orig_X_colnames
    X[is.na(X)] <- 0 # NAs if added coords due to empty blocks
    
    Z           <- simdata %>% dplyr::select(contains("Z_")) %>% as.matrix()
    Z[is.na(Z)] <- 0
    colnames(Z) <- orig_Z_colnames
    
    na_which    <- simdata$na_which
    offsets     <- simdata$offsets
    
    coords <- simdata %>% dplyr::select(contains("Var")) %>% as.matrix()
    
    #block_groups <- check_gibbs_groups(block_groups, parents, children, block_names, blocking, 20)
    
    comp_time <- system.time({
      results <- qmeshgp_dev_mcmc(y, X, Z, coords, blocking,
                              
                              parents, children, 
                              block_names, block_groups,
                              predictable_blocks,
                              
                              indexing_grid, # use grid to sample w
                              indexing_obs,
                              
                              set_unif_bounds,
                              beta_Vi, 
                              sigmasq_ab,
                              tausq_ab,
                              
                              start_w, 
                              start_theta,
                              start_beta,
                              start_tausq,
                              start_sigmasq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                              
                              num_threads,
                              
                              mcmc_adaptive, # adapting
                              mcmc_cache, # use cache
                              mcmc_cache_gibbs,
                              F, # use all coords as reference even at empty blocks
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_printall, # print all iter
                              saving,
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta, 
                              sample_tausq, 
                              sample_sigmasq, 
                              sample_theta, 
                              sample_w) 
    })
    
    list2env(results, environment())
    return(list(coords    = coords,
                data = simdata,
                sort_ix      = sort_ix,
                
                indexing_grid = indexing_grid,
                
                beta_mcmc    = beta_mcmc,
                tausq_mcmc   = tausq_mcmc,
                
                theta_mcmc   = theta_mcmc,
                
                w_mcmc    = w_mcmc,
                yhat_mcmc = yhat_mcmc,
      
                runtime_all   = comp_time,
                runtime_mcmc  = mcmc_time
                ))
}
