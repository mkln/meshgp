dependence_matrix <- function(coords_blocking, type="mesh", d=NULL){
  if(type == "mesh_matrix"){
    layers_descr <- unique(coords_blocking %>% select(-contains("Var")))
    dep <- matrix(0, nrow(layers_descr), nrow(layers_descr))
    dsize <- 1:(coords_blocking %>% select(contains("Var")) %>% ncol())
    df <- layers_descr[,-ncol(layers_descr)] %>% apply(2, as.numeric)
    for(i in 1:nrow(layers_descr)){
      for(j in 1:i){
        test <- sum(abs(df[i,dsize] - df[j,dsize])) <= 1
        #test <- min(abs(df[i,dsize] - df[j,dsize])) == 0
        dep[i,j] <- ifelse(test, 1, 0)
      }
    }
    rownames(dep) <- layers_descr$layer
    colnames(dep) <- layers_descr$layer
    diag(dep) <- 0
  }
  if(type == "mesh"){
    
  }
  if(type == "sequence"){
    layers_descr <- unique(coords_blocking %>% select(-contains("Var")))
    dep <- matrix(0, nrow(layers_descr), nrow(layers_descr))
    
    for(i in 1:nrow(layers_descr)){
      for(j in 1:i){
        dep[i,j] <- ifelse(j==i-1, 1, 0)
      }
    }
    rownames(dep) <- layers_descr$layer
    colnames(dep) <- layers_descr$layer
  }
  if(type == "multiscale"){
    depcols <- coords_blocking$layer %>% unique() %>% length() 
    LL <- depcols %>% add(1) %>% log2()
    dep <- matrix(0, nrow=depcols, ncol=depcols)
    
    for(l in (LL-1):2){
      norows <- 1:(2^(LL-l)-1)
      maxcol <- 2^(l)-1
      whichcols <- 1:maxcol
      dep[-norows, whichcols] <- dep[-norows, whichcols] + (diag(maxcol) %x% rep(1, 2^(LL-l)))
    }
    dep[-1,1] <- 1 
    colnames(dep) <- rownames(dep) <- 1:(2^LL - 1)
  }
  if(type == "multiscale-limited"){
    depcols <- coords_blocking$layer %>% unique() %>% length() 
    LL <- depcols %>% add(1) %>% log2()
    dep <- matrix(0, nrow=depcols, ncol=depcols)
    
    for(l in (LL-1):1){
      rangecols <- (2^(l-1)):(2^(l)-1)
      rangerows <- (2^(l)):(2^(l+1)-1)
      dep[rangerows, rangecols] <- diag(2^(l-1)) %x% rep(1, 2)
    }
    colnames(dep) <- rownames(dep) <- 1:(2^LL - 1)
  }
  if(type == "nearest-neighbor"){
    if(is.null(d)){
      stop("Nearest-neighbor but no d provided.")
    }
    depcols <- coords_blocking$layer %>% unique() %>% length() 
    dep <- matrix(0, nrow=depcols, ncol=depcols)
    for(l in 2:nrow(dep)){
      heremat <- coords_blocking[coords_blocking$layer == l, c("Var1", "Var2")] %>% as.matrix()
      possibles <- coords_blocking[coords_blocking$layer < l, c("Var1", "Var2")] %>% as.matrix()
      dists <- 1:nrow(possibles) %>% sapply(function(i) norm(heremat-possibles[i,]))
      lnames <- 1:(l-1)
      dep[l, lnames[order(dists)] %>% head(d)] <- 1
    }
    colnames(dep) <- rownames(dep) <- 1:ncol(dep)
  }
  
  return(dep)
}

plotgraph <- function(dep){
  if(ncol(dep) > 100){
    stop("Graph is too big")
  }
  #Plot the graph
  depnet <- dep
  diag(depnet) <- 0
  depnet <- - depnet
  depnet %>% t() %>% network() %>% plot(vertex.cex=2, arrowhead.cex=2, displaylabels=T)
}

plotmatrix <- function(mat){
  if(ncol(mat) > 1000){
    stop("Matrix is too big")
  }
  mat %<>% as.matrix()
  max_mat <- max(mat)
  min_mat <- min(mat)
  
  nbreaks <- 40
  brk <- lattice::do.breaks(c(min_mat, max_mat), nbreaks)
  
  rotate = function(mat) t(mat[nrow(mat):1,,drop=FALSE])
  
  if(min_mat*max_mat < 0){
    nneg <- sum(brk<0)
    npos <- sum(brk>=0)
    brk <- brk[brk!=0]
    brk_0 <- c(brk[brk<0], 0, brk[brk>0])
    negcols <- colorRampPalette(RColorBrewer::brewer.pal(9, "Reds"))(nneg+2)[(nneg+2):1] %>% tail(nneg)
    poscols <- colorRampPalette(RColorBrewer::brewer.pal(9, "Blues"))(npos)
    allcols <- c(negcols, "#FFFFFF", poscols)
    
    mat[mat==0] <- NA
    mat %>% rotate %>% lattice::levelplot(pretty=T, 
                                          col.regions=allcols, 
                                          at=brk, xlab=NULL, ylab=NULL, colorkey=F)
  } else {
    if(max_mat > 0){
      allcols <- colorRampPalette(RColorBrewer::brewer.pal(9, "GnBu"))(nbreaks)
      mat %>% rotate %>% lattice::levelplot(pretty=T, 
                                            col.regions=allcols, 
                                            at=brk, xlab=NULL, ylab=NULL, colorkey=F)
    } else {
      allcols <- colorRampPalette(RColorBrewer::brewer.pal(9, "OrRd"))(nbreaks)
      mat %>% rotate %>% lattice::levelplot(pretty=T, 
                                            col.regions=allcols, 
                                            at=brk, xlab=NULL, ylab=NULL, colorkey=F)
    }
    
  }
  
}

