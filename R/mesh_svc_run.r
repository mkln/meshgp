meshgp <- function(y, X, Z, coords, axis_partition, 
                   mcmc        = list(keep=1000, burn=0, thin=1),
                   num_threads = 7,
                   settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                                      reference_full_coverage=F, verbose=F, debug=F, 
                                      printall=F, saving=T, matern_twonu=1),
                   prior       = list(set_unif_bounds=NULL,
                                      beta=NULL,
                                      sigmasq=NULL,
                                      tausq=NULL, toplim=1e5, btmlim=1e-2),
                   starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                   debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T),
                   recover     = list()
                   ){

  # init
  cat(" Bayesian MeshGP model with cubic tessellation & cubic mesh (Q-MGP)\n
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
    
    if(!is.null(settings$matern_twonu)){
      matern_twonu <- settings$matern_twonu
    } else {
      matern_twonu <- 1
    }
    
    
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
    
    if(is.null(prior$beta)){
      beta_Vi <- diag(ncol(X)) * 1/100
    } else {
      beta_Vi <- prior$beta
    }
    
    if(is.null(prior$sigmasq)){
      sigmasq_ab <- c(3, 3)
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
      start_sigmasq <- 1
    } else {
      start_sigmasq  <- starting$sigmasq
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
  
  if(length(recover) == 0){
    # Domain partitioning and gibbs groups
    fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], Mv[i])) 
    
    # guaranteed to produce blocks using Mv
    system.time(fake_coords_blocking <- coords %>% 
                  as.matrix() %>% 
                  gen_fake_coords(fixed_thresholds, 1) )
    
    cat("Tessellating")
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% as.matrix() %>%
                  tessellation_axis_parallel_fix(fixed_thresholds, 1) %>% cbind(data.frame(na_which=simdata$na_which) ))
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
      nr_full <- nr
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
  } else {
    cat("Restoring tessellation and graph from recovered data.\n")
    # taking these from recovered data
    parents      <- list()
    children     <- list()
    block_names  <- numeric()
    block_groups <- numeric()
    blocking     <- numeric()
    indexing     <- list()
  }

    comp_time <- system.time({
      results <- qmeshgp_svc_mcmc(y, X, Z, coords, blocking,
                              
                              parents, children, 
                              block_names, block_groups,
                              indexing,
                              
                              set_unif_bounds,
                              beta_Vi, 
                              sigmasq_ab,
                              tausq_ab,
                              
                              start_w, 
                              start_theta,
                              start_beta,
                              start_tausq,
                              start_sigmasq,
                              
                              matern_twonu,
                              
                              mcmc_mh_sd,
                              
                              recover,
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
                              sample_beta, sample_tausq, sample_sigmasq, sample_theta, sample_w) 
    })
    
    return(list(coords    = coords,
                blocking = blocking,
                sort_ix      = sort_ix) %>% c(results)
                )
}


mesh_predict_by_block <- function(meshout, newcoords, twonu=1, n_threads=10){
  dd <- ncol(newcoords)
  
  sort_ix <- 1:nrow(newcoords)
  
  # for each predicting coordinate, find which block it belongs to
  # (instead of using original partitioning (convoluted), 
  # use NN since the main algorithm is adding coordinates in empty areas so NN will pick those up)
  in_coords <- meshout$coords#$meshdata$data %>% dplyr::select(contains("Var"))
  nn_of_preds <- in_coords %>% FNN::get.knnx(newcoords, k=1, algorithm="kd_tree") %$% nn.index
  
  block_ref <- meshout$blocking[nn_of_preds]
  
  ## by block (same block = same parents)
  newcx_by_block     <- newcoords %>% as.data.frame() %>% split(block_ref) %>% lapply(as.matrix)
  
  names_by_block     <- names(newcx_by_block) %>% as.numeric()
  
  sort_ix_by_block   <- sort_ix %>% split(block_ref)
  
  
  y_predicted <- meshgp::mesh_predict_by_block_base(newcx_by_block, 
                                         names_by_block,
                                         meshout$w_mcmc,
                                         meshout$theta_mcmc, 
                                         meshout$sigmasq_mcmc,
                                         meshout$tausq_mcmc,
                                         meshout$recover$model_data$indexing,
                                         meshout$recover$model_data$parents_indexing,
                                         meshout$recover$model$parents,
                                         meshout$coords, twonu, n_threads)
  
  sort_ix_result <- do.call(c, sort_ix_by_block)
  coords_reconstruct <- do.call(rbind, newcx_by_block)
  
  coords_df <- cbind(coords_reconstruct, block_ref) %>% as.data.frame()
  
  y_preds <- do.call(rbind, y_predicted)
  
  return(list("coords_pred" = coords_df,
              "y_pred" = y_preds))
}
