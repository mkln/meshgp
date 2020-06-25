mesh_dep <- function(coords_blocking, M){
  cbl <- coords_blocking %>% dplyr::select(-contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% group_by(L1, L2, L3, block) %>% summarize(na_which = sum(na_which))
  } else {
    cbl %<>% group_by(L1, L2, block) %>% summarize(na_which = sum(na_which))
  }
  blocks_descr <- unique(cbl) 
  return(mesh_dep_cpp(blocks_descr %>% as.matrix(), M))
}

mesh_graph_build <- function(coords_blocking, Mv, verbose=T){
  cbl <- coords_blocking %>% dplyr::select(-contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% group_by(L1, L2, L3, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n(), color=unique(color))
  } else {
    cbl %<>% group_by(L1, L2, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n(), color=unique(color))
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  
  rfc <- F
  graphed <- mesh_graph_cpp(blocks_descr, Mv, rfc, verbose)
  #groups <- mesh_gibbs_groups(blocks_descr, Mv, rfc)
  
  block_ct_obs <- coords_blocking %>% group_by(block) %>% 
    summarise(block_ct_obs = sum(na_which, na.rm=T)) %>% arrange(block) %$% 
    block_ct_obs# %>% `[`(order(block_names))
  
  graph_blanketed <- blanket(graphed$parents, graphed$children, graphed$names, block_ct_obs)
  groups <- coloring(graph_blanketed, graphed$names, block_ct_obs)
  #cinfo <- coords_blocking %>% group_by(block) %>% summarise(L1=unique(L1), L2=unique(L2)) %>%
  #  cbind(data.frame(coloring = test_coloring))
  #ggplot(cinfo, aes(L1, L2, fill=factor(coloring), label=block)) + geom_raster() + geom_text()
  blocks_descr %<>% as.data.frame() %>% arrange(block) %>% 
    cbind(groups) 
  groups <- blocks_descr$groups#[order(blocks_descr$block)]
  groups[groups == -1] <- max(groups)+1
  
  list2env(graphed, environment())
  return(list(parents = parents,
              children = children,
              names = names,
              groups = groups))
}

mesh_graph_build_sampling <- function(coords_blocking, Mv, na_which, rfc=F, verbose=F){
  cbl <- coords_blocking %>% dplyr::select(-contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% group_by(L1, L2, L3, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n(), color=unique(color))
  } else {
    cbl %<>% group_by(L1, L2, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n(), color=unique(color))
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  
  graphed <- mesh_graph_cpp(blocks_descr, Mv, rfc, verbose)
  
  list2env(graphed, environment())
  return(list(parents = parents,
              children = children,
              names = names))
}


