mvmeshgp <- function(y, X, coords, mv_id, axis_partition, 
                   mcmc        = list(keep=1000, burn=0, thin=1),
                   num_threads = 10,
                   settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F,
                                      verbose=F, debug=F, 
                                      printall=F),
                   prior       = list(set_unif_bounds=NULL,
                                      beta=NULL,
                                      tausq=NULL),
                   starting    = list(beta=NULL, tausq=NULL, theta=NULL, w=NULL),
                   debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
                   ){

  
  if(F){
    axis_partition <- c(5, 5)
    coords <- coords_q
    
    num_threads <- 10
    mcmc        = list(keep=1000, burn=0, thin=1)
    settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                       verbose=F, debug=F, 
                       printall=F, saving=T)
    prior       = list(set_unif_bounds=NULL,
                       beta=NULL,
                       sigmasq=NULL,
                       tausq=NULL)
    starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL)
    debug       = list(sample_beta=T, sample_tausq=T, sample_theta=T, sample_w=T)
    
  }
  
  # init
  cat(" Bayesian MeshGP model with cubic tessellation & cubic mesh (Q-MGP)\n (Multivariate response)\n
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
    
    
    saving         <- F
    
    sample_beta    <- debug$sample_beta
    sample_tausq   <- debug$sample_tausq
    sample_sigmasq <- debug$sample_sigmasq
    sample_theta   <- debug$sample_theta
    sample_w       <- debug$sample_w
    
    dd             <- ncol(coords)
    p              <- ncol(X)
    q              <- length(unique(mv_id))
    k              <- q * (q-1)/2
    nr             <- nrow(X)
    
    if(length(Mv) == 1){
      Mv <- rep(Mv, dd)
    }
    avg_block <- nr/prod(Mv)*q
    
    if(is.null(starting$beta)){
      start_beta   <- rep(0, p)
    } else {
      start_beta   <- starting$beta
    }
    
    space_uni      <- (dd==2) & (q==1)
    space_biv      <- (dd==2) & (q==2) 
    space_mul      <- (dd==2) & (q >2)  
    stime_uni      <- (dd==3) & (q==1)
    stime_biv      <- (dd==3) & (q==2) 
    stime_mul      <- (dd==3) & (q >2)  
    
    
    toplim <- 1e5
    btmlim <- 1e-5
    
    
    if(dd == 2){
      if(q > 1){
        n_cbase <- ifelse(q > 2, 3, 1)
        npars <- 3*q + n_cbase - 1 + 1 ##//
        
        start_theta <- rep(2, npars) %>% c(rep(1, k))
        
        set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
        set_unif_bounds[,1] <- btmlim
        set_unif_bounds[,2] <- toplim
        
        if(n_cbase == 3){
          start_theta[npars-1] <- .5
          set_unif_bounds[npars-1,] <- c(btmlim, 1-btmlim)
        }
        
        vbounds <- matrix(0, nrow=k, ncol=2)
        if(q > 2){
          dlim <- sqrt(q+.0)
        } else {
          dlim <- 1e5
        }
        vbounds[,1] <- 1e-5;
        vbounds[,2] <- dlim - 1e-5
        set_unif_bounds <- rbind(set_unif_bounds, vbounds)
      
      } else {
        npars <- 1 + 1
        start_theta <- rep(10, npars)
        
        set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
        set_unif_bounds[,1] <- btmlim
        set_unif_bounds[,2] <- toplim
      }
      
    }
    
    if(!is.null(starting$theta)){
      start_theta <- starting$theta
    }
    
    
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
    
    if(is.null(starting$sigmasq)){
      start_sigmasq <- 10
    } else {
      start_sigmasq  <- starting$sigmasq
    }
    
    if(is.null(starting$w)){
      start_w <- rep(0, nrow(coords))
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
  
  colnames(coords)  <- paste0('Var', 1:dd)
  
  na_which <- ifelse(!is.na(y), 1, NA)
  simdata <- 1:nrow(coords) %>% cbind(coords, mv_id) %>% 
    cbind(y) %>% cbind(na_which) %>% 
    cbind(X) %>% as.data.frame()
  colnames(simdata)[1] <- "ix"
  
  simdata %<>% arrange(!!!syms(paste0("Var", 1:dd)))
  colnames(simdata)[dd + (2:4)] <- c("mv_id", "y", "na_which")
  
  coords <- simdata %>% select(contains("Var"))
  simdata %<>% mutate(type="obs")
  sort_ix     <- simdata$ix
  
  coords_u <- coords %>% unique()
  if(!is.matrix(coords)){
    coords %<>% as.matrix()
    coords_u %<>% as.matrix()
  }

  # Domain partitioning and gibbs groups
  fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], Mv[i])) 
  
  # guaranteed to produce blocks using Mv
  system.time(fake_coords_blocking <- coords %>% 
                as.matrix() %>% 
                gen_fake_coords(fixed_thresholds, 1) )
  
  # Domain partitioning and gibbs groups
  system.time(coords_blocking <- coords %>% 
                as.matrix() %>%
                tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% mutate(na_which = simdata$na_which, sort_ix=sort_ix) )
  
  coords_blocking %<>% dplyr::rename(ix=sort_ix)
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
    nr_full <- nr
  }

  emptyblocks <- coords_blocking %>% 
    group_by(block) %>% 
    summarize(avail = mean(!is.na(na_which))) %>%
    dplyr::filter(avail==0)
  newcol <- coords_blocking$color %>% max() %>% add(1)
  coords_blocking %<>% mutate(color = ifelse(block %in% emptyblocks$block, newcol, color))

  c_unique <- coords_blocking %>% dplyr::select(block, color) %>% unique()
  ggroup <- c_unique %$% block
  gcolor <- c_unique %$% color
  
  blocking <- coords_blocking$block %>% factor() %>% as.integer()
  
  #save.image(file="debug.RData")
  #load("debug.RData")
  
  # DAG
  suppressMessages(parents_children <- mesh_graph_build(coords_blocking %>% dplyr::select(-ix), Mv, F))
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  indexing                     <- (1:nr_full-1) %>% split(blocking)
  
  simdata <- coords_blocking %>% dplyr::select(-na_which) %>% left_join(simdata)
  
  #start_w <- rep(0, nr_full)
  
  # finally prepare data
  sort_ix     <- simdata$ix
  
  y           <- simdata$y %>% matrix(ncol=1)
  X           <- simdata %>% dplyr::select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  X[is.na(X)] <- 0 # NAs if added coords due to empty blocks
  
  na_which    <- simdata$na_which

  coords <- simdata %>% dplyr::select(contains("Var")) %>% as.matrix()
  mv_id  <- simdata$mv_id
  
  comp_time <- system.time({
      results <- qmeshgp_mv_mcmc(y, X, coords, mv_id, blocking,
                              
                              parents, children, 
                              block_names, block_groups,
                              indexing,
                              
                              set_unif_bounds,
                              beta_Vi, 
                              
                              tausq_ab,
                              
                              start_w, 
                              start_sigmasq,
                              start_theta,
                              start_beta,
                              start_tausq,
                              
                              mcmc_mh_sd,
                              
                              mcmc_keep, mcmc_burn, mcmc_thin,
                              
                              num_threads,
                              
                              mcmc_adaptive, # adapting
                              mcmc_cache, # use cache
                              mcmc_cache_gibbs,
                              
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_printall, # print all iter
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta, sample_tausq, 
                              sample_sigmasq,
                              sample_theta, sample_w) 
    })
  
    list2env(results, environment())
    return(list(coords    = coords,
                indexing = indexing,
                block_ct_obs = bco,
                beta_mcmc    = beta_mcmc,
                tausq_mcmc   = tausq_mcmc,
                sigmasq_mcmc = sigmasq_mcmc,
                theta_mcmc   = theta_mcmc,
                
                w_mcmc    = w_mcmc,
                yhat_mcmc = yhat_mcmc,
      
                runtime_all   = comp_time,
                runtime_mcmc  = mcmc_time,
                sort_ix      = sort_ix,
                paramsd   = paramsd,
                debug = debug
                ))
    
}
