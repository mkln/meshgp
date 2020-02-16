meshgp <- function(y, X, Z, coords, Mv, 
                   mcmc        = list(keep=1000, burn=0, thin=1),
                   num_threads = 7,
                   settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                                      reference_full_coverage=F, verbose=F, debug=F, 
                                      printall=F, saving=T, seed=NULL),
                   prior       = list(set_unif_bounds=NULL),
                   starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL),
                   debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T),
                   dry_run     = F,
                   recover     = list()
                   ){
  if(F){
    mcmc        = list(keep=10, burn=0, thin=1)
    num_threads = 11
    settings    = list(adapting=T, mcmcsd=.3, cache=T, cache_gibbs=F, 
                       reference_full_coverage=F, verbose=T, debug=T, printall=F, seed=NULL)
    prior       = list(set_unif_bounds=matrix(rbind(c(1e-5, Inf), c(1e-5, 1-1e-5), c(1e-5, Inf), c(1e-5, 1-1e-5), c(1e-5, Inf)), ncol=2))
    starting    = list(beta=NULL, tausq=NULL, sigmasq=NULL, theta=NULL, w=NULL)
    debug       = list(sample_beta=T, sample_tausq=T, sample_sigmasq=T, sample_theta=T, sample_w=T)
    dry_run     = F
    recover     = list()
    X <- X_full
    Z <- Z_full
    y <- y_full
  }
  
  # init
  cat(" Bayesian MeshGP model with cubic tessellation & cubic mesh (Q-MGP)\n
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o
    ^     ^     ^
    |     |     | 
    o --> o --> o\n\n")
  
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
    
    rfc_dependence <- settings$reference_full_coverage
    
    seeded         <- ifelse(is.null(settings$seed), round(runif(1, 0, 1000)), settings$seed)
    
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
    space_uni      <- (q==1) & (dd==2)
    
    if(length(Mv) == 1){
      Mv <- rep(Mv, dd)
    }
    avg_block <- nr/prod(Mv)*q
    cat(sprintf("~ %s intervals along the axes (total M=%d)", paste(Mv, collapse=" "), prod(Mv)),"\n")
    cat(sprintf("~ %s average block size", avg_block),"\n")
    
    if(avg_block > 100){
      goon <- readline(prompt="Average block size is large. Iterations will be slow. Continue? Y/n: ")
      goon <- tolower(goon)
      if(!(goon %in% c("y", ""))){
        return(0)
      }
    }
    
    if(is.null(starting$beta)){
      start_beta   <- rep(0, p)
    } else {
      start_beta   <- starting$beta
    }
    
    if(is.null(starting$theta)){
      if(space_uni){
        start_theta <- 10
      } else {
        start_theta <- rep(1, 5 + k) # excluding sigmasq
        start_theta[c(1,3,5)] <- 10
        start_theta[c(2,4)] <- .5
      }
    } else {
      start_theta  <- starting$theta
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
      start_w <- rep(0, q*nr) %>% matrix(ncol=q)
    } else {
      start_w <- starting$w #%>% matrix(ncol=q)
    }
    
    if(is.null(prior$set_unif_bounds)){
      set_unif_bounds <- matrix(rbind(c(1e-5, Inf), 
                                      c(1e-5, 1-1e-5), 
                                      c(1e-5, Inf), 
                                      c(1e-5, 1-1e-5), 
                                      c(1e-5, Inf)), ncol=2)
    } else {
      set_unif_bounds <- prior$set_unif_bounds
    }
    
    if(length(settings$mcmcsd) == 1){
      mcmc_mh_sd <- diag(length(start_theta)) * settings$mcmcsd
    } else {
      mcmc_mh_sd <- settings$mcmcsd
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
    coords <- simdata %>% select(Var1, Var2)
    colnames(simdata)[4:5] <- c("y", "na_which")
  } else {
    simdata %<>% arrange(Var1, Var2, Var3)
    coords <- simdata %>% select(Var1, Var2, Var3)
    colnames(simdata)[5:6] <- c("y", "na_which")
  }
  
  sort_ix     <- simdata$ix
  
  y           <- simdata$y %>% matrix(ncol=1)
  X           <- simdata %>% select(contains("X_")) %>% as.matrix()
  colnames(X) <- orig_X_colnames
  Z           <- simdata %>% select(contains("Z_")) %>% as.matrix()
  colnames(Z) <- orig_Z_colnames
  na_which    <- simdata$na_which
  
  if(!is.matrix(coords)){
    coords %<>% as.matrix()
  }
  
  if((length(recover) == 0) || dry_run){
    # Domain partitioning and gibbs groups
    system.time(coords_blocking <- coords %>% tessellation_axis_parallel(Mv, num_threads) %>% cbind(na_which) )
    
    if(!rfc_dependence){
      emptyblocks <- coords_blocking %>% 
        group_by(block) %>% 
        summarize(avail = mean(!is.na(na_which))) %>%
        filter(avail==0)
      newcol <- coords_blocking$color %>% max() %>% add(1)
      coords_blocking %<>% mutate(color = ifelse(block %in% emptyblocks$block, newcol, color))
    }
    c_unique <- coords_blocking %>% select(block, color) %>% unique()
    ggroup <- c_unique %$% block
    gcolor <- c_unique %$% color
    
    blocking <- coords_blocking$block %>% factor() %>% as.integer()
    
    # DAG
    system.time(parents_children <- mesh_graph_build(coords_blocking, Mv, na_which, rfc_dependence))
    parents                      <- parents_children[["parents"]] 
    children                     <- parents_children[["children"]] 
    block_names                  <- parents_children[["names"]] 
    block_groups                 <- parents_children[["groups"]][order(block_names)]
    indexing                     <- (1:nr-1) %>% split(blocking)
    
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
  
  if(!dry_run){
    set.seed(seeded)
    comp_time <- system.time({
      results <- qmeshgp_svc_mcmc(y, X, Z, coords, blocking,
                              
                              parents, children, 
                              block_names, block_groups,
                              indexing,
                              
                              set_unif_bounds,
                              
                              start_w, 
                              start_theta,
                              start_beta,
                              start_tausq,
                              start_sigmasq,
                              
                              mcmc_mh_sd,
                              
                              recover,
                              mcmc_keep, mcmc_burn, mcmc_thin,
                              
                              num_threads,
                              
                              mcmc_adaptive, # adapting
                              mcmc_cache, # use cache
                              mcmc_cache_gibbs,
                              rfc_dependence, # use all coords as reference even at empty blocks
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_printall, # print all iter
                              saving,
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta, sample_tausq, sample_sigmasq, sample_theta, sample_w) 
    })
    
    list2env(results, environment())
    return(list(coords    = coords,
                
                beta_mcmc    = beta_mcmc,
                tausq_mcmc   = tausq_mcmc,
                sigmasq_mcmc = sigmasq_mcmc,
                theta_mcmc   = theta_mcmc,
                
                w_mcmc    = w_mcmc,
                yhat_mcmc = yhat_mcmc,
      
                seed      = seeded, 
                runtime_all   = comp_time,
                runtime_mcmc  = mcmc_time,
                sort_ix      = sort_ix,
                paramsd   = paramsd,
                recover   = recover
                ))
    
  } else {
    cat("Dry run -- No MCMC, just preprocessing.\n")
    
    # 
    modplot              <- coords_blocking %>% select(block, color) %>% unique() %>% cbind(block_groups[order(order(ggroup))]) 
    colnames(modplot)[3] <- "groupmod"
    
    suppressMessages(coords_blocking_mod  <- coords_blocking %>% left_join(modplot))
    gplotcols            <- c("#CB2314","#273046","#354823","#1E1E1E","#FAD510",
                              "#DD8D29", "magenta" ,"#46ACC8", "#E58601", "#B40F20")
    block_colorplot_text <- ggplot(coords_blocking_mod, aes(Var1, Var2, label=block, color=factor(groupmod))) + 
      geom_text(size=3) + 
      scale_color_manual(values=gplotcols)
    
    block_colorplot_tile <- ggplot(coords_blocking_mod, aes(Var1, Var2)) + 
      geom_tile(aes(fill=factor(groupmod))) +
      scale_fill_manual(values=gplotcols)
    
    infodf               <- coords_blocking %>% 
      group_by(block) %>% 
      summarise(blocksize=n()) 
    
    n_by_block           <- ggplot(infodf, aes(factor(blocksize))) + 
      geom_bar() + 
      theme_minimal() + 
      labs(x=NULL, y=NULL) + 
      ggtitle("Dist. of observations by block")
    
    infodf               <- y %>% split(blocking) %>% lapply(function(x) sum(!is.na(x))) %>% unlist()
    infodf               <- data.frame(blocksize=infodf)
    avail_by_block       <- ggplot(infodf, aes(factor(blocksize))) + 
      geom_bar() + 
      theme_minimal() + 
      labs(x=NULL, y=NULL) + 
      ggtitle("Dist. of not-NA observations by block")
    
    suppressMessages(meshinfo             <- simdata %>% left_join(coords_blocking))
    data_scatter         <- ggplot(coords_blocking, aes(Var1, Var2)) + 
      geom_point(aes(color=factor(na_which)), size=1) +
      theme(legend.position="bottom")
    
    datacolor_scatter    <- ggplot(meshinfo,# %>% filter(!is.na(na_which)), 
                                   aes(Var1, Var2)) + 
      geom_tile(aes(fill=factor(color))) + 
      geom_point(alpha=.2, aes(color=factor(color)))
    
    if(dd==3){
      gtext <- coords_blocking %>% group_by(L1, L2, L3, block) %>%
        summarize(Var1 = head(Var1, 1), Var2 = head(Var2, 1), Var3 = head(Var3, 1))  
      gtextplot <- ggplot(gtext, aes(Var1, Var2, label=block)) + geom_text() + facet_grid(~factor(Var3))
      
    } else {
      gtext <- coords_blocking %>% group_by(L1, L2, block) %>%
        summarize(Var1 = head(Var1, 1), Var2 = head(Var2, 1))  
      gtextplot <- ggplot(gtext, aes(Var1, Var2, label=block)) + geom_text()
    }
    
    
    infoplots <- list(
      availdata_scatter = data_scatter,
      colordata_scatter = datacolor_scatter,
      blocks_tile       = block_colorplot_tile,
      blocks_text       = block_colorplot_text,
      gtext             = gtextplot,
      blocks_obs_hist   = n_by_block,
      blocks_avail_hist = avail_by_block
    )
    
    comp_time <- system.time({
      results <- qmeshgp_svc_dry(y, X, Z, coords, blocking,
                              
                              parents, children, block_names, block_groups,
                              indexing,
                             
                              start_w, 
                              start_theta,
                              start_beta,
                              start_tausq,
                              start_sigmasq,
                              diag(length(start_theta)) * mcmc_mh_sd,
                             
                              mcmc_keep, mcmc_burn, mcmc_thin,
                              
                              num_threads,
                              
                              mcmc_adaptive, # adapting
                              mcmc_cache, # use cache
                              mcmc_cache_gibbs,
                              rfc_dependence, # use all coords as reference even at empty blocks
                              mcmc_verbose, mcmc_debug, # verbose, debug
                              mcmc_printall, # print all iter
                              
                              # sampling of:
                              # beta tausq sigmasq theta w
                              sample_beta, sample_tausq, sample_sigmasq, sample_theta, sample_w) 
    })
    
    return(list(infoplots = infoplots,
                groups    = block_groups,
                seed      = seeded, 
                coords    = coords_blocking_mod,
                recover   = results))
    }
}
