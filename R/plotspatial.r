plotspatial <- function(dfxyz, na_val = 0, nrow=1, lims=NULL){
  # assumes first two columns are spatial coordinates,
  # other columns are values for faceting
  if(is.matrix(dfxyz)){
    dfxyz %<>% as.data.frame()
  }
  colnames(dfxyz)[1:2] <- c("coordx", "coordy")
  dfxyz_p <- dfxyz %>% select(-coordx, -coordy) %<>% mutate_all(function(x) ifelse(x==na_val, NA, x))
  dfxyz[,-c(1:2)] <- dfxyz_p
  dfl <- dfxyz %>% gather(z, zvalue, -coordx, -coordy)
  
  if(is.null(lims)){
    plotted <- ggplot(dfl, aes(coordx, coordy, fill=zvalue)) + geom_raster() +
      theme_minimal() + facet_wrap(~z, nrow=nrow) +
      theme(legend.position="none") + labs(x=NULL, y=NULL) +
      scale_fill_viridis_c()
  } else {
    plotted <- ggplot(dfl, aes(coordx, coordy, fill=zvalue)) + geom_raster() +
      theme_minimal() + facet_wrap(~z, nrow=nrow) +
      theme(legend.position="none") + labs(x=NULL, y=NULL) +
      scale_fill_viridis_c(limits=lims) 
  }
  
  print(plotted)
  return(plotted)
}
