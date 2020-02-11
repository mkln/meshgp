mesh_dep <- function(coords_blocking, M){
  cbl <- coords_blocking %>% select(-contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% group_by(L1, L2, L3, block) %>% summarize(na_which = sum(na_which))
  } else {
    cbl %<>% group_by(L1, L2, block) %>% summarize(na_which = sum(na_which))
  }
  blocks_descr <- unique(cbl) 
  return(mesh_dep_cpp(blocks_descr %>% as.matrix(), M))
}

mesh_graph_build <- function(coords_blocking, Mv, na_which, rfc=F){
  cbl <- coords_blocking %>% select(-contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% group_by(L1, L2, L3, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n(), color=unique(color))
  } else {
    cbl %<>% group_by(L1, L2, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n(), color=unique(color))
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  
  graphed <- mesh_graph_cpp(blocks_descr, Mv, rfc)
  groups <- mesh_gibbs_groups(blocks_descr, Mv, rfc)
  
  if("L3" %in% colnames(coords_blocking)){
    colnames(groups) <- c("L1", "L2", "L3", "group")
  } else {
    colnames(groups) <- c("L1", "L2", "group")
  }
  groups <- blocks_descr %>% as.data.frame() %>% left_join(groups %>% as.data.frame()) %$% group
  
  list2env(graphed, environment())
  return(list(parents = parents,
              children = children,
              names = names,
              groups = groups))
}


mesh_graph_build_backup <- function(coords_blocking, M, rfc=F){
  cbl <- coords_blocking %>% select(-contains("Var"))
  if("L3" %in% colnames(coords_blocking)){
    cbl %<>% group_by(L1, L2, L3, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n())
  } else {
    cbl %<>% group_by(L1, L2, block) %>% summarize(na_which = sum(na_which, na.rm=T)/n())
  }
  blocks_descr <- unique(cbl) %>% as.matrix()
  if(rfc){
    result <- mesh_dep_rfc_backup_cpp(blocks_descr, M)
  } else {
    result <- mesh_dep_norfc_backup_cpp(blocks_descr, M)
  }
  return(result)
}
 
