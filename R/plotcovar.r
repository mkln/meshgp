
covarplot <- function(sigmasq, theta, hmax=sqrt(2), umax=NULL, v=NULL, hrange=1, urange=1, ngrid=100){
  if(is.null(umax)){
    d <- 2
  } else {
    d <- 3
  }
  if(is.null(v)){
    q <- 1
    v <- 0
  } else {
    q <- 2
  }
  
  h <- seq(0, hmax, length.out=ngrid)
  u <- seq(0, umax, length.out=ngrid)
  
  lagmat <- expand.grid(h, u, v) %>% as.matrix()
  
  theta_full <- c(sigmasq, theta)
  
  post_cov <- lagmat %>% 
    cbind(lagmat %>% apply(1, function(ll) meshgp::xCovHUV_base(ll[1], ll[2], ll[3], theta_full, q, d)) %>% as.data.frame())
  colnames(post_cov) <- c("hspace", "utime", "v", "Covariance")
  
  post_cov %<>% mutate(hspace_o = hspace*hrange,
                       utime_o = utime*urange)
  
  cvplot <- ggplot(post_cov, aes(x=utime_o, y=hspace_o, fill=Covariance, z=Covariance)) +
    geom_raster() + 
    metR::geom_text_contour(nudge_y=.02*hrange) +
    scale_y_continuous(limits=c(-0.02, hrange+.02)) +
    scale_x_continuous(limits=c(-0.02, urange+.02)) +
    geom_contour(color="black", lty="dotted") +
    scale_fill_gradient2(high="darkred", midpoint=0.05, low="darkblue") +
    theme_minimal() +
    labs(y="Spatial distance", x="Time lag") +
    theme(legend.position="none")
  
  return(cvplot)
}

