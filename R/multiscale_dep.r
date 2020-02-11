multiscale2_dep <- function(L){
  # L is the number of 2-way MS steps
  
  tot_n_reg <- 2^L -1
  
  parents <- list()
  children <- list()
  
  for(i in seq_len(tot_n_reg)){
    parents[[i]] <- -1
    children[[i]] <- -1
  }
  
  for(l in 2:L){
    parent_matrix <- matrix(0, nrow=2^(l-1), ncol=l-1)
    for(pp in 1:(l-1)){
      prev_lev_numbers <- (2^(pp - 1)):(2^pp - 1) 
      parent_matrix[,pp] <- prev_lev_numbers %x% rep(1, 2^(l - pp))
    }
    lev_here <- (2^(l - 1)):(2^l - 1) 
    for(i in 1:length(lev_here)){
      pi <- lev_here[i]
      parents[[pi]] <- parent_matrix[i,] -1
      for(ele in parent_matrix[i,]){
        children[[ele]] <- c(pi-1, children[[ele]])
      }
    }
  }
  
  parents %<>% lapply(function(xx) xx <- xx[xx != -1])
  children %<>% lapply(function(xx) xx <- xx[xx != -1])
  
  return(
    list(
      "parents" = parents,
      "children" = children,
      "names" = 1:tot_n_reg
    )
  )
  
}

multiscale4_dep <- function(L, limited=F){
  # L is the number of 2-way MS steps
  
  tot_n_reg <- 4^(0:(L-1)) %>% sum()
  
  parents <- list()
  children <- list()
  
  for(i in seq_len(tot_n_reg)){
    parents[[i]] <- -1
    children[[i]] <- -1
  }
  
  for(i in 2:5){
    parents[[i]] <- 0
    children[[1]] <- c(i-1, children[[1]])
  }
  
  for(l in 3:L){
    
    parent_matrix <- matrix(0, nrow=4^(l-1), ncol=l-1)
    if(!limited){
      parent_matrix[,1] <- 1
      for(pp in 2:(l-1)){
        lev_nums <- 4^(0:(pp-1)) %>% cumsum()
        lev_here <- (lev_nums[pp-1]+1) : lev_nums[pp]
        parent_matrix[,pp] <- lev_here %x% rep(1, 4^(l - pp))
      }
    } else {
      pp <- l-1
      lev_nums <- 4^(0:(pp-1)) %>% cumsum()
      lev_here <- (lev_nums[pp-1]+1) : lev_nums[pp]
      parent_matrix[,pp] <- lev_here %x% rep(1, 4^(l - pp))
    }
    
    lev_nums <- 4^(0:(l-1)) %>% cumsum()
    lev_here <- (lev_nums[l-1]+1) : lev_nums[l]
    for(i in 1:nrow(parent_matrix)){
      pi <- lev_here[i]
      valid_parents <- parent_matrix[i,]
      valid_parents <- valid_parents[valid_parents != 0]
      parents[[pi]] <- valid_parents -1
      for(ele in valid_parents){
        children[[ele]] <- c(pi-1, children[[ele]])
      }
    }
  }
  
  parents %<>% lapply(function(xx) xx <- xx[xx != -1])
  children %<>% lapply(function(xx) xx <- xx[xx != -1])
  
  return(
    list(
      "parents" = parents,
      "children" = children,
      "names" = 1:tot_n_reg
    )
  )
  
}

multiscale_base_dep <- function(L, base, limited=F){
  # L is the number of 2-way MS steps
  

   tot_n_reg <- (1 - base^L)/(1-base)#base^(0:(L-1)) %>% sum()
   #first find all parents, then build all children based on them
   parents <- list()
   
   for(i in seq_len(tot_n_reg)){
     parents[[i]] <- -1
   }
   
   for(i in 2:(base+1)){
     parents[[i]] <- 0
   }
   
   for(l in 3:L){
     parent_matrix <- matrix(0, nrow=base^(l-1), ncol=l-1)
     if(!limited){
       parent_matrix[,1] <- 1
       for(pp in 2:(l-1)){
         lev_nums <- base^(0:(pp-1)) %>% cumsum()
         lev_here <- (lev_nums[pp-1]+1) : lev_nums[pp]
         parent_matrix[,pp] <- lev_here %x% rep(1, base^(l - pp))
       }
     } else {
       pp <- l-1
       lev_nums <- base^(0:(pp-1)) %>% cumsum()
       lev_here <- (lev_nums[pp-1]+1) : lev_nums[pp]
       parent_matrix[,pp] <- lev_here %x% rep(1, base^(l - pp))
     }
     
     lev_nums <- base^(0:(l-1)) %>% cumsum()
     lev_here <- (lev_nums[l-1]+1) : lev_nums[l]
     
     parents <- ms_find_parents(parent_matrix, lev_here, parents)
   }
   parents %<>% lapply(function(xx) xx <- xx[xx != -1])
   
   children <- ms_find_children(L, base, length(parents))
   
     if(0){
       children <- list()
       
       for(l in 1:(L-1)){
         n_at_l <- base^(l-1)
         for(j in 1:n_at_l){
           cix <- (1-base^(l-1))/(1-base) + j
           locator <- rep(0, n_at_l)
           locator[j] <- 1
           children_x <- c()
           for(ln in (l+1):L){
             locator <- locator %x% rep(1, base)
             starting_from <- (1-base^(ln-1))/(1-base)
             children_x <- c(children_x, starting_from + which(locator == 1))
           }
           children[[cix]] <- children_x
         }
       }
       
       lastlev_n <- (1-base^L)/(1-base) - (1-base^(L-1))/(1-base)
       Lrange <- (length(children)+1):(length(children) + lastlev_n)
       for(cix in Lrange){
         children[[cix]] <- -1
       }
       
       children %<>% lapply(function(xx) xx <- xx[xx != -1] - 1)
     }
     
  return(
    list(
      "parents" = parents,
      "children" = children,
      "names" = 1:tot_n_reg
    )
  )
  
}

multiscale_step_by_4 <- function(current_df, n_target, last=F){
  cx <- current_df %>% 
    group_by(layer) %>%
    mutate(x1 = Var1 > quantile(Var1, .5), x2 = Var2 > quantile(Var2, .5)) %>% 
    ungroup() %>%
    mutate(newlayer = 2*x1 + x2 + 1) %>%
    mutate(newlayer = newlayer + 4*(layer-1)+1) %>%
    group_by(newlayer) %>% 
    mutate(ixl = 1:n()) 
  
  if(!last){
    new_step_df <- cx %>% 
      filter(ixl %in% round(seq(1, n(), n()/n_target))) %>%
      mutate(layer = newlayer) %>% ungroup() %>%
      select(-newlayer)
    
    remaining_df <- cx %>%
      filter(!(ix %in% unique(new_step_df$ix))) %>% 
      mutate(layer = newlayer) %>% ungroup() %>%
      select(-newlayer)
  } else {
    new_step_df <- cx %>% 
      mutate(layer = newlayer) %>% ungroup() %>%
      select(-newlayer)
    
    remaining_df <- "nothing"
  }
  
  
  results <- list()
  results[["remaining_df"]] <- remaining_df 
  results[["this_level_df"]] <- new_step_df
  return(results)
}

multiscale_step_by_8 <- function(current_df, n_target, last=F){
  cx <- current_df %>% 
    group_by(layer) %>%
    mutate(x1 = Var1 > quantile(Var1, .5), x2 = Var2 > quantile(Var2, .5), x3 = Var3 > quantile(Var3, .5)) %>% 
    ungroup() %>%
    mutate(newlayer = 4*x1 + 2*x2 + x3 + 1) %>%
    mutate(newlayer = newlayer + 8*(layer-1)+1) %>%
    group_by(newlayer) %>% 
    mutate(ixl = 1:n()) 
  
  if(!last){
    new_step_df <- cx %>% 
      filter(ixl %in% round(seq(1, n(), n()/n_target))) %>%
      mutate(layer = newlayer) %>% ungroup() %>%
      select(-newlayer)
    
    remaining_df <- cx %>%
      filter(!(ix %in% unique(new_step_df$ix))) %>% 
      mutate(layer = newlayer) %>% ungroup() %>%
      select(-newlayer)
  } else {
    new_step_df <- cx %>% 
      mutate(layer = newlayer) %>% ungroup() %>%
      select(-newlayer)
    
    remaining_df <- "nothing"
  }
  
  
  results <- list()
  results[["remaining_df"]] <- remaining_df 
  results[["this_level_df"]] <- new_step_df
  return(results)
}


multiscale_blocking <- function(coords, size_by_block, L){
  multiscale_df <- list()
  
  spacetime <- ncol(coords) == 3
  
  ms_step <- list()
  
  ms_step[["this_level_df"]] <- coords %>%
    mutate(layer = 1, ix = 1:n()) %>% 
    filter(ix %in% round(seq(1, n(), n()/size_by_block))) 
  
  ms_step[["remaining_df"]] <- coords %>%
    mutate(layer = 1, ix = 1:n()) %>% 
    filter(!(ix %in% round(seq(1, n(), n()/size_by_block))))
  
  multiscale_df[[1]] <- ms_step[["this_level_df"]]
  
  if(spacetime){
    for(l in 2:L){
      ms_step <- ms_step[["remaining_df"]] %>% multiscale_step_by_8(size_by_block, l==L)
      multiscale_df[[l]] <- ms_step[["this_level_df"]] %>% select(Var1, Var2, Var3, layer)
    }
  } else {
    for(l in 2:L){
      ms_step <- ms_step[["remaining_df"]] %>% multiscale_step_by_4(size_by_block, l==L)
      multiscale_df[[l]] <- ms_step[["this_level_df"]] %>% select(Var1, Var2, layer)
    }
  }
  
  result <- bind_rows(multiscale_df)
  return(result)
}

tot_n_blocks <- function(base, L){
  return(
    (1-base^(L+1)) / (1 - base)
  )
}

get_L <- function(x, base, n){
  return(
    ceiling(log(1 - n*(1-base)/x)/log(base) - 1)
  )
}


multiscale_dep <- function(layering, L, base, limited=F){

  parents_children <- multiscale_base_dep(L, base, limited=F)
  #parents_children2 <- multiscale4_dep(L, limited=F)
  
  unique_layers <- layering %>% unique()
  
  relabeling <- seq_len(length(unique_layers))
  names(relabeling) <- unique_layers %>% sort()
  
  relabel_fn <- function(x){
    x <- intersect(x+1, unique_layers)
    x <- relabeling[as.character(x)]-1
    return(x)
  }
  
  parents <- parents_children[["parents"]] %>% lapply(relabel_fn)
  parents[ relabeling ] <- parents[ names(relabeling) %>% as.numeric() ]
  parents <- parents[relabeling]
  
  children <- parents_children[["children"]]%>% lapply(relabel_fn)
  children[ relabeling ] <- children[ names(relabeling) %>% as.numeric() ]
  children <- children[relabeling]
  
  layer_names <- parents_children[["names"]] %>% intersect(unique_layers)
  layer_names <- relabeling[layer_names %>% as.character()]
  
  return(list(
    "parents" = parents,
    "children" = children,
    "names" = layer_names
  ))
}

