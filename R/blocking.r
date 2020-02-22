# lattice blocking / mesh / 2d or 3d
blocking_col <- function(col, d, thresholds_in = NULL){
  if(is.null(thresholds_in)){
    thresh_points <- seq(0, 1, length.out=d+1) %>% head(-1) %>% tail(-1)
    thresholds <- col %>% quantile(thresh_points)
  } else {
    thresholds <- thresholds_in
  }
  
  #return( col %>% sapply(function(x) as.character(1+sum(x >= thresholds))) )
  return(col %>% turbocolthreshold(thresholds) %>% factor())
}

blocking_col2 <- function(col, d){
  thresholds <- col %>% quantile(seq(0, 1, length.out=d+1) %>% head(-1) %>% tail(-1))
  #return( col %>% sapply(function(x) as.character(1+sum(x >= thresholds))) )
  return(col %>% turbocolthreshold(thresholds) %>% factor())
}

blocking_xd <- function(coords, d, type="mesh", L=NULL){
  cat("~", type,"partitioning ~ \n")
  if(type=="mesh"){
    system.time(blocks_by_coord <- coords %>% apply(2, function(x) blocking_col(x, d)) )
    colnames(blocks_by_coord) <- paste0("L", 1:ncol(coords))
    
    block <- blocks_by_coord %>% 
      as.data.frame() %>% as.list() %>% interaction()
    blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
    result <- cbind(coords, blockdf) %>% as.data.frame() %>% arrange(Var1, Var2) %>% mutate(color = (L1+L2) %% 2)
    
    if(ncol(coords)==2){
      return(result)
    } else {
      result %<>% mutate(colorswitch = L3 %% 2 == 0) %>% mutate(color = ifelse(colorswitch, color, 1-color))
      return(result)
    }
    
  }
  if(type=="sequence"){
    blocks_by_coord <- coords[,1] %>% blocking_col(d)
    blockdf <- blocks_by_coord %>% 
      as.data.frame(stringsAsFactors=T) 
    colnames(blockdf) <- "block"
    return(blockdf)
  }
  

  if(type == "multiscale"){
    if(ncol(coords) == 2){
      return(multiscale_blocking(coords, d, L) %>% arrange(Var1, Var2))
    } else {
      return(multiscale_blocking(coords, d, L) %>% arrange(Var1, Var2, Var3))
    }
   
  }
  if(type=="nearest-neighbor"){
    blockdf <- coords %>% mutate(block = 1:n()) %>% select(-contains("Var"))
    return(blockdf)
  }
  
  if(type=="kmeans-nn-mesh"){
    kmeanout <- coords %>% as.matrix() %>% kmeans(centers=d)
    clustering <- data.frame(cluster=kmeanout$cluster)
    coords %<>% cbind(clustering)
    centerdata <- data.frame(center=kmeanout$centers, cluster=1:d)
    coords %<>% left_join(centerdata)
  }
  
  if(type %in% c("multiscale_old", "multiscale-limited_old")){
    mslevs <- d
    
    maxl <- log2(16)
    dimen <- coords %>% select(contains("Var")) %>% ncol()
    grow_coords <- coords
    grow_coords %<>% mutate(L1 = 1)
    
    for(l in 2:d){
      group_coords <- grow_coords %>% 
        group_by(!!rlang::sym(paste0("L", l-1))) %>% mutate(gi = group_indices()) %>%
        group_map(~ (function(df){ 
          clusters <- df %>% select(contains("Var")) %>% as.matrix() %>%
            dist() %>% kmeans(2)
          df_cluster <- data.frame(clusters$cluster) %>% rename(clusters = clusters.cluster)
          df %<>% bind_cols(df_cluster) %>% 
            mutate(!!rlang::sym(paste0("L", l)) := str_c(gi, clusters)) %>% 
            select(-clusters, -gi) 
        })(.))
      grow_coords %<>% left_join(group_coords %<>% bind_rows()) %>% 
        mutate(!!rlang::sym(paste0("L", l)) := 2^(l-1)-1 + as.numeric(factor(!!rlang::sym(paste0("L", l)))))
    }
    
    n_by_reg <- grow_coords %>% group_by(!!rlang::sym(paste0("L", mslevs))) %>% summarize(n_by_reg=n()) %$% n_by_reg
    
    coords_remaining <- grow_coords %>% mutate(ix = 1:n())
    coords_by_block <- list()
    for(l in 1:(mslevs-1)){
      coords_sub <- coords_remaining %>% mutate(grouping = !!rlang::sym(paste0("L", maxl))) %>%
        group_by(!!rlang::sym(paste0("L", maxl))) %>% #group_by(!!rlang::sym(paste0("L", l))) %>%
        group_map(~ (function(df) df[sample(1:nrow(df), min(nrow(df), 2^(l-1))),])(.)) %>%
        bind_rows() 
      sel_ix <- coords_sub$ix
      coords_sub %<>%
        rename(!!rlang::sym(paste0("L", maxl)) := grouping) %>%
        group_split(!!rlang::sym(paste0("L", l)))
      
      coords_remaining <- coords_remaining %>% filter(!(ix %in% sel_ix))
      
      n_by_reg <- coords_remaining %>% group_by(!!rlang::sym(paste0("L", mslevs))) %>% summarize(n_by_reg=n()) %$% n_by_reg
      coords_by_block[(2^(l-1)):(2^(l)-1)] <- coords_sub
      
      #cat(l, " ", min(n_by_reg), "\n")
    }
    # 
    l <- mslevs
    coords_sub <- coords_remaining %>% group_split(!!rlang::sym(paste0("L", l)))
    coords_by_block[(2^(l-1)):(2^(l)-1)] <- coords_sub 
    
    blockdf <- 1:length(coords_by_block) %>% lapply(function(j) 
      coords_by_block[[j]] %>% select(contains("Var")) %>% mutate(block = j)) %>%
      bind_rows() %>% arrange(!!!rlang::syms(paste0("Var", 1:dimen)))
    blockdf <- data.frame(block=blockdf$block)
    return(blockdf)
  }
  stop(glue::glue("No blocking performed. Right type '{type}'?"))
}



tessellation_axis_parallel <- function(coordsmat, Mv, n_threads){

  blocks_by_coord <- part_axis_parallel(coordsmat, Mv, n_threads) %>% apply(2, factor)
  colnames(blocks_by_coord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_coord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
  result <- cbind(coordsmat, blockdf) %>% 
                mutate(color = (L1+L2) %% 2)
  
  if(ncol(coords)==2){
    return(result)
  } else {
    result %<>% mutate(colorswitch = L3 %% 2 == 0) %>% mutate(color = ifelse(colorswitch, color, 1-color))
    return(result)
  }

}


tessellation_axis_parallel_fix <- function(coordsmat, thresholds, n_threads){
  
  blocks_by_coord <- part_axis_parallel_fixed(coordsmat, thresholds, n_threads) %>% apply(2, factor)
  colnames(blocks_by_coord) <- paste0("L", 1:ncol(coordsmat))
  
  block <- blocks_by_coord %>% 
    as.data.frame() %>% as.list() %>% interaction()
  blockdf <- data.frame(blocks_by_coord %>% apply(2, as.numeric), block=as.numeric(block))
  result <- cbind(coordsmat, blockdf) %>% 
    mutate(color = (L1+L2) %% 2)
  
  if(ncol(coords)==2){
    return(result)
  } else {
    result %<>% mutate(colorswitch = L3 %% 2 == 0) %>% mutate(color = ifelse(colorswitch, color, 1-color))
    return(result)
  }
  
}







