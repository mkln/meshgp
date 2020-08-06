

rqmeshgp_mv <- function(coords, mv_id, axis_partition, ai1, ai2, phi_i, thetamv, Dmat, cache=F, n_threads=4){
  
  set.seed(300)
  if(F){
    library(Matrix)
    library(meshgp)
    library(magrittr)
    library(tidyverse)
    
    xx <- seq(0.01, .99, length.out=40)
    cbase <- expand.grid(xx,xx)
    coords_mv <- bind_rows(cbase %>% mutate(mv_id=1), 
                           cbase %>% mutate(mv_id=2))#,
                           #cbase %>% mutate(mv_id=3))
    #coords_mv %<>% arrange(Var1, Var2, mv_id)
    coords <- coords_mv %>% dplyr::select(Var1, Var2)
    mv_id <- coords_mv$mv_id
    q <- 3
    ai1 <- ai2 <- phi_i <- rep(1, q)
    thetamv <- if(q>2) { rep(1, q) } else { 1 }
    thetamv <- if(q==1) { c(.1, .1) } else { thetamv }
    Dmat <- matrix(0, q, q)
    Dmat[lower.tri(Dmat)] <- .1
    Dmat[upper.tri(Dmat)] <- Dmat[lower.tri(Dmat)]
    
    cache <- F
    n_threads <- 4
    
    axis_partition <- c(10, 10)
  }
  
  Mv <- axis_partition
  nr <- nrow(coords)
  dd <- ncol(coords)
  na_which <- rep(1, nr)
  
  fixed_thresholds <- 1:dd %>% lapply(function(i) kthresholdscp(coords[,i], Mv[i])) 
  
  coords_blocking <- coords %>% as.matrix() %>%
    tessellation_civ(fixed_thresholds, 1) %>% 
    mutate(na_which = 1, mv_id = mv_id, ix = 1:n()) %>% 
    arrange(block, 
            !!!syms(paste0("Var", 1:dd)), mv_id)
  
  #coords_blocking %>% group_by(block) %>% summarise(
  #  L1 = L1[1], L2 = L2[1], .groups="drop") %>% 
  #  ggplot(aes(L1, L2, label=block, color=factor(block))) + 
  #  geom_text() +
  #  theme(legend.position="none")
  
  # DAG
  suppressMessages(parents_children <- mesh_graph_build(coords_blocking %>% dplyr::select(-ix, -mv_id), Mv, F))
  parents                      <- parents_children[["parents"]] 
  children                     <- parents_children[["children"]] 
  block_names                  <- parents_children[["names"]]

  blocking <- coords_blocking$block %>% factor() %>% as.integer()
  indexing <- (1:nr - 1) %>% split(blocking)
  
  # finally prepare data
  sort_ix     <- coords_blocking$ix
  
  coords <- coords_blocking %>% dplyr::select(contains("Var")) %>% as.matrix()
  mv_id  <- coords_blocking$mv_id
  
  
  qmgp_sampled <- qmgp_mv_sampler(coords, mv_id, blocking, parents, block_names, 
                                  indexing,
                   ai1, ai2, phi_i, thetamv, Dmat,
                   n_threads, # num threads
                   cache, # cache
                   F, # verb
                   F) # debug
  
  #qmgp_sampled$ImH %>% spamtree:::mimage()
  #qmgp_sampled <- qmgp_sampler(coords, blocking, parents, block_names, indexing, thetamv, matrix(0), 4)
  #Ci <- qmgp_Cinv(coords, blocking, parents, block_names, indexing, thetamv, matrix(0), 4)
  #rCi <- t(solve(t(chol(Ci[nr:1, nr:1]))))
  #qmgp_sampled <- as.numeric(rCi %*% rnorm(nrow(rCi)))
  
  sampled_data <- coords_blocking %>% 
    cbind(data.frame(w=qmgp_sampled)) %>%
    arrange(ix)
  
  #sampled_data %>% ggplot(aes(Var1, Var2, fill=w)) + geom_raster() + facet_grid(~mv_id)
  
  return( sampled_data$w )
  
}


qmeshgp_Cinv <- function(coords, Mv, theta, D, n_threads=4){
  
  nr <- nrow(coords)
  dd <- coords %>% ncol()
  na_which <- rep(1, nr)
  coords_blocking <- coords %>% tessellation_civ(Mv, 5, F) %>% cbind(na_which)
  
  coords_blocking %<>% mutate(sort_ix = 1:n())
  coords_blocking %<>% arrange(block, Var1, Var2)
  sort_ix <- coords_blocking %>% pull(sort_ix)
  coords_blocking %<>% dplyr::select(-sort_ix)
  
  coords_sorted <- coords_blocking %>% dplyr::select(contains("Var")) %>% as.matrix()
  blocking_sorted <- coords_blocking$block %>% #factor() %>% 
    as.integer()
  # DAG
  system.time(parents_children <- mesh_graph_build(coords_blocking, Mv, F))
  parents                      <- parents_children[["parents"]] 
  block_names                  <- parents_children[["names"]] 
  indexing                     <- (1:nr-1) %>% split(blocking_sorted)
  
  Cinv <- qmgp_Cinv(coords_sorted, blocking_sorted, parents, block_names, indexing,
                theta, D,n_threads, # num threads
                F, # cache
                               F, # verb
                               F) # debug
  sort_order <- order(sort_ix)
  return( Cinv[sort_order, sort_order] )
  
}