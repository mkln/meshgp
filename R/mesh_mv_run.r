mvmeshgp <- function(y, X, coords, mv_id, axis_partition, 
                   thresholds_user = NULL,
                   mcmc        = list(keep=1000, burn=0, thin=1),
                   num_threads = 10,
                   settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F,
                                      verbose=F, debug=F, 
                                      printall=F),
                   prior       = list(set_unif_bounds=NULL,
                                      beta=NULL,
                                      sigmasq=NULL,
                                      tausq=NULL,
                                      toplim = 1e5,
                                      btmlim = 1e-2),
                   starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                   debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
                   ){

  if(F){
    #coords <- coords_q
    #X <- X[!is.na(y),-1,drop=F]
    coords <- coords_q#[!is.na(y),]
    #mv_id <- mv_id[!is.na(y)]
    #y <- y[!is.na(y)]
    
    num_threads <- 10
    mcmc        = list(keep=100, burn=0, thin=1)
    settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                       verbose=F, debug=F, 
                       printall=F, saving=T)
    prior       = list(set_unif_bounds=NULL,
                       beta=NULL,
                       sigmasq=NULL,
                       tausq=NULL)
    starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL)
    debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
    axis_partition <- rep(round(sqrt(nrow(coords)/5)), ncol(coords))
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
  
  # data management pt 1
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
    sample_sigmasq <- F#debug$sample_sigmasq
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
    
    if(dd == 2){
      if(q > 1){
        n_cbase <- ifelse(q > 2, 3, 1)
        npars <- 3*q + n_cbase ##//
        
        start_theta <- rep(2, npars) %>% c(rep(1, k))
        
        set_unif_bounds <- matrix(0, nrow=npars, ncol=2)
        set_unif_bounds[,1] <- btmlim
        set_unif_bounds[,2] <- toplim
        
        if(q>1){
          set_unif_bounds[2:q, 1] <- -toplim
        }
        
        if(n_cbase == 3){
          start_theta[npars-1] <- .5
          set_unif_bounds[npars-1,] <- c(btmlim, 1-btmlim)
        }
        
        vbounds <- matrix(0, nrow=k, ncol=2)
        if(q > 2){
          dlim <- sqrt(q+.0)
        } else {
          dlim <- toplim
        }
        vbounds[,1] <- btmlim;
        vbounds[,2] <- dlim - btmlim
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

  # data management pt 2
  if(1){
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
    
    simdata %<>% arrange(!!!syms(paste0("Var", 1:dd)), mv_id)
    colnames(simdata)[dd + (2:4)] <- c("mv_id", "y", "na_which")
    
    coords <- simdata %>% select(contains("Var"))
    simdata %<>% mutate(type="obs")
    sort_ix     <- simdata$ix
    
    if(!is.matrix(coords)){
      coords %<>% as.matrix()
    }
    
    # Domain partitioning and gibbs groups
    if(is.null(thresholds_user)){
      fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], Mv[i])) 
    } else {
      fixed_thresholds <- thresholds_user
    }
    
    # guaranteed to produce blocks using Mv
    system.time(fake_coords_blocking <- coords %>% 
                  as.matrix() %>% 
                  gen_fake_coords(fixed_thresholds, 1) )
    
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% 
                  as.matrix() %>%
                  tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% mutate(na_which = simdata$na_which, sort_ix=sort_ix) )
    
    coords_blocking %<>% dplyr::rename(ix=sort_ix)
    
    if(F){
      coords_blocking_mv <- coords_blocking %>% cbind(data.frame(mv_id=simdata$mv_id)) %>% mutate(block_mv_id = stringr::str_c(block, "-", mv_id))
      
      data_block_mv <- coords_blocking_mv %>% group_by(block_mv_id) %>% summarise(perc_avail = sum(na_which,na.rm=T)/n())
      predict_blocks <- data_block_mv %>% filter(perc_avail == 0) %$% block_mv_id
      
      pred_ix <- coords_blocking_mv %>% filter(block_mv_id %in% predict_blocks) %>% pull(ix)
      simdata_pred <- simdata %>% filter(ix %in% pred_ix)
      
      coords_blocking_mv %<>% filter(!(block_mv_id %in% predict_blocks))
      coords_blocking <- coords_blocking_mv %>% dplyr::select(-mv_id, -block_mv_id)
    }
    
    
    
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
      
      coords_blocking %<>% arrange(!!!syms(paste0("Var", 1:dd)))
      
      nr_full <- nrow(coords_blocking)
    } else {
      nr_full <- nr
    }
    
  }
  
  # DAG
  suppressMessages(parents_children <- mesh_graph_build(coords_blocking %>% dplyr::select(-ix), Mv, F))
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]] 
  block_groups                 <- parents_children[["groups"]]#[order(block_names)]
  
  simdata_in <- coords_blocking %>% dplyr::select(-na_which) %>% left_join(simdata)
  #simdata[is.na(simdata$ix), "ix"] <- seq(nr_start+1, nr_full)
  
  simdata_in %<>% arrange(!!!syms(paste0("Var", 1:dd)), mv_id)
  
  blocking <- simdata_in$block %>% factor() %>% as.integer()
  indexing                     <- (1:nrow(simdata_in)-1) %>% split(blocking)
  
  start_w <- rep(0, nrow(simdata_in))
  
  # finally prepare data
  sort_ix     <- simdata_in$ix
  
  y           <- simdata_in$y %>% matrix(ncol=1)
  X           <- simdata_in %>% dplyr::select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  X[is.na(X)] <- 0 # NAs if added coords due to empty blocks
  
  na_which    <- simdata_in$na_which

  coords <- simdata_in %>% dplyr::select(contains("Var")) %>% as.matrix()
  mv_id  <- simdata_in$mv_id
  mv_id[is.na(mv_id)] <- 1 # NAs if added coords due to empty blocks
  
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
  meshout <- list(coords = coords,
                  sort_ix = sort_ix,
                  mv_id = mv_id,
                  
                  beta_mcmc    = beta_mcmc,
                  tausq_mcmc   = tausq_mcmc,
                  theta_mcmc   = theta_mcmc,
                  
                  w_mcmc    = w_mcmc,
                  yhat_mcmc = yhat_mcmc,
                  
                  runtime_all   = comp_time,
                  runtime_mcmc  = mcmc_time,
                  
                  meshdata = list(parents_children = parents_children,
                                  indexing = indexing,
                                  parents_indexing = parents_indexing,
                                  blocking = blocking,
                                  data = simdata_in)
  )
  
  return(meshout)
    
}

mvmesh_predict_by_block <- function(meshout, newx, newcoords, new_mv_id, n_threads=10){
  dd <- ncol(newcoords)
  pp <- length(unique(meshout$mv_id))
  k <- pp * (pp-1) / 2
  npars <- nrow(meshout$theta_mcmc) - k
  sort_ix <- 1:nrow(newcoords)
  
  # for each predicting coordinate, find which block it belongs to
  # (instead of using original partitioning (convoluted), 
  # use NN since the main algorithm is adding coordinates in empty areas so NN will pick those up)
  in_coords <- meshout$coords#$meshdata$data %>% dplyr::select(contains("Var"))
  nn_of_preds <- in_coords %>% FNN::get.knnx(newcoords, k=1, algorithm="kd_tree") %$% nn.index
  
  block_ref <- meshout$meshdata$data$block[nn_of_preds]
  
  ## by block (same block = same parents)
  newcx_by_block     <- newcoords %>% as.data.frame() %>% split(block_ref) %>% lapply(as.matrix)
  new_mv_id_by_block <- new_mv_id %>% split(block_ref) %>% lapply(as.numeric)
  newx_by_block      <- newx %>% as.data.frame() %>% split(block_ref) %>% lapply(as.matrix)
  names_by_block     <- names(newcx_by_block) %>% as.numeric()
  
  sort_ix_by_block   <- sort_ix %>% split(block_ref)
  
  result <- mvmesh_predict_by_block_base(newcx_by_block, new_mv_id_by_block, newx_by_block, 
                                    names_by_block,
                                    meshout$w_mcmc,
                                    meshout$theta_mcmc, 
                                meshout$beta_mcmc,
                                meshout$tausq_mcmc,
                                meshout$meshdata$indexing,
                                meshout$meshdata$parents_indexing,
                                meshout$meshdata$parents_children$parents,
                                meshout$coords,
                                meshout$mv_id,
                                npars, dd, pp, n_threads)
  
  sort_ix_result <- do.call(c, sort_ix_by_block)
  coords_reconstruct <- do.call(rbind, newcx_by_block)
  mv_id_reconstruct <- do.call(c, new_mv_id_by_block)
  coords_df <- cbind(coords_reconstruct, mv_id_reconstruct, block_ref) %>% as.data.frame() %>%
    rename(mv_id = mv_id_reconstruct)
  
  #coords_df <- coords_df[order(sort_ix_result),]
  w_preds <- do.call(rbind, result$w_pred)#[order(sort_ix_result),]
  y_preds <- do.call(rbind, result$y_pred)#[order(sort_ix_result),]
  

  return(list("coords_pred" = coords_df,
              "w_pred" = w_preds,
              "y_pred" = y_preds))
}


mvmesh_predict <- function(meshout, newx, newcoords, new_mv_id, n_threads=10){
  #meshdata <- meshout$meshdata
  in_coords <- meshout$coords#meshdata$data %>% dplyr::select(contains("Var"))
  dd <- ncol(in_coords)
  pp <- length(unique(meshout$mv_id))
  k <- pp * (pp-1) / 2
  npars <- nrow(meshout$theta_mcmc) - 1
  mcmc <- meshout$w_mcmc %>% length()
  
  nn_of_preds <- in_coords %>% FNN::get.knnx(newcoords, k=1, algorithm="kd_tree") %$% nn.index
  
  #coords_ref <- meshout$meshdata$blocking[nn_of_preds] #%>% arrange(!!!syms(paste0("L", 1:dd)), block) 
  block_ref <- meshout$meshdata$blocking[nn_of_preds]
  
  newcx <- newcoords %>% as.matrix()
  
  result <- meshgp:::mvmesh_predict_base(newcx, new_mv_id, newx, 
                                meshout$beta_mcmc,
                                meshout$theta_mcmc, meshout$w_mcmc,
                                meshout$tausq_mcmc,
                                meshout$meshdata$indexing,
                                meshout$meshdata$parents_indexing,
                                meshout$meshdata$parents_children$parents,
                                meshout$coords,
                                block_ref, meshout$meshdata$data$mv_id,
                                npars, dd, pp, n_threads)
  return(result)
}
