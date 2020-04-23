

rqmeshgp <- function(coords, Mv, theta, D, cache=F, n_threads=4){
  
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
  system.time(parents_children <- mesh_graph_build_sampling(coords_blocking, Mv, na_which, T, F))
  parents                      <- parents_children[["parents"]] 
  block_names                  <- parents_children[["names"]] 
  indexing                     <- (1:nr-1) %>% split(blocking_sorted)
  
  qmgp_sampled <- qmgp_sampler(coords_sorted, blocking_sorted, parents, block_names, indexing,
                   theta, D,
                   n_threads, # num threads
                   cache, # cache
                   F, # verb
                   F) # debug
  
  return( qmgp_sampled[order(sort_ix)] )
  
}


qmeshgp_cinv <- function(coords, Mv, theta, D, n_threads=4){
  
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
  system.time(parents_children <- mesh_graph_build_sampling(coords_blocking, Mv, na_which, T, F))
  parents                      <- parents_children[["parents"]] 
  block_names                  <- parents_children[["names"]] 
  indexing                     <- (1:nr-1) %>% split(blocking_sorted)
  
  Cinv <- qmgp_Cinv(coords_sorted, blocking_sorted, parents, block_names, indexing,
                theta, D,n_threads, # num threads
                F, # cache
                               F, # verb
                               F) # debug
  
  return( Cinv )
  
}