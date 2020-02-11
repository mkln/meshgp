plotgraph2 <- function (x, attrname = NULL, label = network.vertex.names(x), 
          coord = NULL, jitter = TRUE, thresh = 0, usearrows = TRUE, 
          mode = "fruchtermanreingold", displayisolates = TRUE, interactive = FALSE, 
          xlab = NULL, ylab = NULL, xlim = NULL, ylim = NULL, pad = 0.2, 
          label.pad = 0.5, displaylabels = !missing(label), boxed.labels = FALSE, 
          label.pos = 0, label.bg = "white", vertex.sides = 50, vertex.rot = 0, 
          vertex.lwd = 1, arrowhead.cex = 1, label.cex = 1, loop.cex = 1, 
          vertex.cex = 1, edge.col = 1, label.col = 1, vertex.col = 2, 
          label.border = 1, vertex.border = 1, edge.lty = 1, label.lty = NULL, 
          vertex.lty = 1, edge.lwd = 0, edge.label = NULL, edge.label.cex = 1, 
          edge.label.col = 1, label.lwd = par("lwd"), edge.len = 0.5, 
          edge.curve = 0.1, edge.steps = 50, loop.steps = 20, object.scale = 0.01, 
          uselen = FALSE, usecurve = FALSE, suppress.axes = TRUE, vertices.last = TRUE, 
          new = TRUE, layout.par = NULL, ...) 
{
  if (!is.network(x)) 
    stop("plot.network requires a network object.")
  if (network.size(x) == 0) 
    stop("plot.network called on a network of order zero - nothing to plot.")
  bellstate <- options()$locatorBell
  expstate <- options()$expression
  on.exit(options(locatorBell = bellstate, expression = expstate))
  options(locatorBell = FALSE, expression = Inf)
  "%iin%" <- function(x, int) (x >= int[1]) & (x <= int[2])
  if (is.hyper(x)) {
    xh <- network.initialize(network.size(x) + sum(!sapply(x$mel, 
                                                           is.null)), directed = is.directed(x))
    for (i in list.vertex.attributes(x)) {
      set.vertex.attribute(xh, attrname = i, value = get.vertex.attribute(x, 
                                                                          attrname = i, null.na = FALSE, unlist = FALSE), 
                           v = 1:network.size(x))
    }
    for (i in list.network.attributes(x)) {
      if (!(i %in% c("bipartite", "directed", "hyper", 
                     "loops", "mnext", "multiple", "n"))) 
        set.network.attribute(xh, attrname = i, value = get.network.attribute(x, 
                                                                              attrname = i, unlist = FALSE))
    }
    cnt <- 1
    for (i in 1:length(x$mel)) {
      if (!is.null(x$mel[[i]])) {
        for (j in x$mel[[i]]$outl) {
          if (!is.adjacent(xh, j, network.size(x) + cnt)) 
            add.edge(xh, j, network.size(x) + cnt, names.eval = names(x$mel[[i]]$atl), 
                     vals.eval = x$mel[[i]]$atl)
        }
        for (j in x$mel[[i]]$inl) {
          if (!is.adjacent(xh, network.size(x) + cnt, 
                           j)) {
            add.edge(xh, network.size(x) + cnt, j, names.eval = names(x$mel[[i]]$atl), 
                     vals.eval = x$mel[[i]]$atl)
          }
        }
        cnt <- cnt + 1
      }
    }
    cnt <- cnt - 1
    if (length(label) == network.size(x)) 
      label <- c(label, paste("e", 1:cnt, sep = ""))
    xh %v% "vertex.names" <- c(x %v% "vertex.names", paste("e", 
                                                           1:cnt, sep = ""))
    x <- xh
    n <- network.size(x)
    d <- as.matrix.network(x, matrix.type = "edgelist", attrname = attrname)
    if (!is.directed(x)) 
      usearrows <- FALSE
  }
  else if (is.bipartite(x)) {
    n <- network.size(x)
    d <- as.matrix.network(x, matrix.type = "edgelist", attrname = attrname)
    usearrows <- FALSE
  }
  else {
    n <- network.size(x)
    d <- as.matrix.network(x, matrix.type = "edgelist", attrname = attrname)
    if (!is.directed(x)) 
      usearrows <- FALSE
  }
  if (NCOL(d) == 2) {
    if (NROW(d) == 0) 
      d <- matrix(nrow = 0, ncol = 3)
    else d <- cbind(d, rep(1, NROW(d)))
  }
  diag <- has.loops(x)
  d[is.na(d)] <- 0
  edgetouse <- d[, 3] > thresh
  d <- d[edgetouse, , drop = FALSE]
  d.raw <- d
  if (!is.null(coord)) {
    cx <- coord[, 1]
    cy <- coord[, 2]
  }
  else {
    layout.fun <- try(match.fun(paste("network.layout.", 
                                      mode, sep = "")), silent = TRUE)
    if (class(layout.fun) == "try-error") 
      stop("Error in plot.network.default: no layout function for mode ", 
           mode)
    temp <- layout.fun(x, layout.par)
    cx <- temp[, 1]
    cy <- temp[, 2]
  }
  if (jitter) {
    cx <- jitter(cx)
    cy <- jitter(cy)
  }
  use <- displayisolates | (((sapply(x$iel, length) + sapply(x$oel, 
                                                             length)) > 0))
  if (is.null(xlab)) 
    xlab = ""
  if (is.null(ylab)) 
    ylab = ""
  if (is.null(xlim)) 
    xlim <- c(min(cx[use]) - pad, max(cx[use]) + pad)
  if (is.null(ylim)) 
    ylim <- c(min(cy[use]) - pad, max(cy[use]) + pad)
  xrng <- diff(xlim)
  yrng <- diff(ylim)
  xctr <- (xlim[2] + xlim[1])/2
  yctr <- (ylim[2] + ylim[1])/2
  if (xrng < yrng) 
    xlim <- c(xctr - yrng/2, xctr + yrng/2)
  else ylim <- c(yctr - xrng/2, yctr + xrng/2)
  baserad <- min(diff(xlim), diff(ylim)) * object.scale
  if (new) {
    plot(0, 0, xlim = xlim, ylim = ylim, type = "n", xlab = xlab, 
         ylab = ylab, asp = 1, axes = !suppress.axes, ...)
  }
  displaylabels <- displaylabels
  label <- plotArgs.network(x, "label", label)
  vertex.cex <- plotArgs.network(x, "vertex.cex", vertex.cex)
  vertex.radius <- rep(baserad * vertex.cex, length = n)
  vertex.sides <- plotArgs.network(x, "vertex.sides", vertex.sides)
  vertex.border <- plotArgs.network(x, "vertex.border", vertex.border)
  vertex.col <- plotArgs.network(x, "vertex.col", vertex.col)
  vertex.lty <- plotArgs.network(x, "vertex.lty", vertex.lty)
  vertex.rot <- plotArgs.network(x, "vertex.rot", vertex.rot)
  vertex.lwd <- plotArgs.network(x, "vertex.lwd", vertex.lwd)
  loop.cex <- plotArgs.network(x, "loop.cex", loop.cex)
  label.col <- plotArgs.network(x, "label.col", label.col)
  label.border <- plotArgs.network(x, "label.border", label.border)
  label.bg <- plotArgs.network(x, "label.bg", label.bg)
  if (!vertices.last) 
    network.vertex(cx[use], cy[use], radius = vertex.radius[use], 
                   sides = vertex.sides[use], col = vertex.col[use], 
                   border = vertex.border[use], lty = vertex.lty[use], 
                   rot = vertex.rot[use], lwd = vertex.lwd[use])
  nDrawEdges <- NROW(d)
  px0 <- numeric(nDrawEdges)
  py0 <- numeric(nDrawEdges)
  px1 <- numeric(nDrawEdges)
  py1 <- numeric(nDrawEdges)
  e.lwd <- numeric(nDrawEdges)
  e.curv <- numeric(nDrawEdges)
  e.type <- numeric(nDrawEdges)
  e.col <- character(nDrawEdges)
  e.hoff <- numeric(nDrawEdges)
  e.toff <- numeric(nDrawEdges)
  e.diag <- logical(nDrawEdges)
  e.rad <- numeric(nDrawEdges)
  if (NROW(d) > 0) {
    edge.col <- plotArgs.network(x, "edge.col", edge.col, 
                                 d = d)
    edge.lty <- plotArgs.network(x, "edge.lty", edge.lty, 
                                 d = d)
    edge.lwd <- plotArgs.network(x, "edge.lwd", edge.lwd, 
                                 d = d)
    if (!is.null(edge.curve)) {
      if (length(dim(edge.curve)) == 2) {
        edge.curve <- edge.curve[d[, 1:2]]
        e.curv.as.mult <- FALSE
      }
      else {
        if (length(edge.curve) == 1) 
          e.curv.as.mult <- TRUE
        else e.curv.as.mult <- FALSE
        edge.curve <- rep(edge.curve, length = NROW(d))
      }
    }
    else if (is.character(edge.curve) && (length(edge.curve) == 
                                          1)) {
      temp <- edge.curve
      edge.curve <- (x %e% edge.curve)[edgetouse]
      if (all(is.na(edge.curve))) 
        stop("Attribute '", temp, "' had illegal missing values for edge.curve or was not present in plot.network.default.")
      e.curv.as.mult <- FALSE
    }
    else {
      edge.curve <- rep(0, length = NROW(d))
      e.curv.as.mult <- FALSE
    }
    if (!is.null(edge.label)) {
      edge.label <- plotArgs.network(x, "edge.label", edge.label, 
                                     d = d)
      edge.label.col <- plotArgs.network(x, "edge.label.col", 
                                         edge.label.col, d = d)
      edge.label.cex <- plotArgs.network(x, "edge.label.cex", 
                                         edge.label.cex, d = d)
    }
    dist <- ((cx[d[, 1]] - cx[d[, 2]])^2 + (cy[d[, 1]] - 
                                              cy[d[, 2]])^2)^0.5
    tl <- d.raw * dist
    tl.max <- max(tl)
    for (i in 1:NROW(d)) {
      if (use[d[i, 1]] && use[d[i, 2]]) {
        px0[i] <- as.double(cx[d[i, 1]])
        py0[i] <- as.double(cy[d[i, 1]])
        px1[i] <- as.double(cx[d[i, 2]])
        py1[i] <- as.double(cy[d[i, 2]])
        e.toff[i] <- vertex.radius[d[i, 1]]
        e.hoff[i] <- vertex.radius[d[i, 2]]
        e.col[i] <- edge.col[i]
        e.type[i] <- edge.lty[i]
        e.lwd[i] <- edge.lwd[i]
        e.diag[i] <- d[i, 1] == d[i, 2]
        e.rad[i] <- vertex.radius[d[i, 1]] * loop.cex[d[i, 
                                                        1]]
        if (uselen) {
          if (tl[i] > 0) {
            e.len <- dist[i] * tl.max/tl[i]
            e.curv[i] <- edge.len * sqrt((e.len/2)^2 - 
                                           (dist[i]/2)^2)
          }
          else {
            e.curv[i] <- 0
          }
        }
        else {
          if (e.curv.as.mult) 
            e.curv[i] <- edge.curve[i] * d.raw[i]
          else e.curv[i] <- edge.curve[i]
        }
      }
    }
  }
  if (diag && (length(px0) > 0) && sum(e.diag > 0)) {
    network.loop(as.vector(px0)[e.diag], as.vector(py0)[e.diag], 
                 length = 1.5 * baserad * arrowhead.cex, angle = 25, 
                 width = e.lwd[e.diag] * baserad/10, col = e.col[e.diag], 
                 border = e.col[e.diag], lty = e.type[e.diag], offset = e.hoff[e.diag], 
                 edge.steps = loop.steps, radius = e.rad[e.diag], 
                 arrowhead = usearrows, xctr = mean(cx[use]), yctr = mean(cy[use]))
    if (!is.null(edge.label)) {
      network.edgelabel(px0, py0, 0, 0, edge.label[e.diag], 
                        directed = is.directed(x), cex = edge.label.cex[e.diag], 
                        col = edge.label.col[e.diag], loops = TRUE)
    }
  }
  if (length(px0) > 0) {
    px0 <- px0[!e.diag]
    py0 <- py0[!e.diag]
    px1 <- px1[!e.diag]
    py1 <- py1[!e.diag]
    e.curv <- e.curv[!e.diag]
    e.lwd <- e.lwd[!e.diag]
    e.type <- e.type[!e.diag]
    e.col <- e.col[!e.diag]
    e.hoff <- e.hoff[!e.diag]
    e.toff <- e.toff[!e.diag]
    e.rad <- e.rad[!e.diag]
  }
  if (!usecurve & !uselen) {
    if (length(px0) > 0) {
      network.arrow(as.vector(px0), as.vector(py0), as.vector(px1), 
                    as.vector(py1), length = 2 * baserad * arrowhead.cex, 
                    angle = 20, col = e.col, border = e.col, lty = e.type, 
                    width = e.lwd * baserad/10, offset.head = e.hoff, 
                    offset.tail = e.toff, arrowhead = usearrows)
      if (!is.null(edge.label)) {
        network.edgelabel(px0, py0, px1, py1, edge.label[!e.diag], 
                          directed = is.directed(x), cex = edge.label.cex[!e.diag], 
                          col = edge.label.col[!e.diag])
      }
    }
  }
  else {
    if (length(px0) > 0) {
      network.arrow(as.vector(px0), as.vector(py0), as.vector(px1), 
                    as.vector(py1), length = 2 * baserad * arrowhead.cex, 
                    angle = 20, col = e.col, border = e.col, lty = e.type, 
                    width = e.lwd * baserad/10, offset.head = e.hoff, 
                    offset.tail = e.toff, arrowhead = usearrows, 
                    curve = e.curv, edge.steps = edge.steps)
      if (!is.null(edge.label)) {
        network.edgelabel(px0, py0, px1, py1, edge.label[!e.diag], 
                          directed = is.directed(x), cex = edge.label.cex[!e.diag], 
                          col = edge.label.col[!e.diag], curve = e.curv)
      }
    }
  }
  if (vertices.last) 
    network.vertex(cx[use], cy[use], radius = vertex.radius[use], 
                   sides = vertex.sides[use], col = vertex.col[use], 
                   border = vertex.border[use], lty = vertex.lty[use], 
                   rot = vertex.rot[use], lwd = vertex.lwd[use])
  if (displaylabels & (!all(label == "")) & (!all(use == FALSE))) {
    if (label.pos == 0) {
      xhat <- yhat <- rhat <- rep(0, n)
      xoff <- cx[use] - mean(cx[use])
      yoff <- cy[use] - mean(cy[use])
      roff <- sqrt(xoff^2 + yoff^2)
      for (i in (1:n)[use]) {
        ij <- unique(c(d[d[, 2] == i & d[, 1] != i, 1], 
                       d[d[, 1] == i & d[, 2] != i, 2]))
        ij.n <- length(ij)
        if (ij.n > 0) {
          for (j in ij) {
            dx <- cx[i] - cx[j]
            dy <- cy[i] - cy[j]
            dr <- sqrt(dx^2 + dy^2)
            xhat[i] <- xhat[i] + dx/dr
            yhat[i] <- yhat[i] + dy/dr
          }
          xhat[i] <- xhat[i]/ij.n
          yhat[i] <- yhat[i]/ij.n
          rhat[i] <- sqrt(xhat[i]^2 + yhat[i]^2)
          if (!is.nan(rhat[i]) && rhat[i] != 0) {
            xhat[i] <- xhat[i]/rhat[i]
            yhat[i] <- yhat[i]/rhat[i]
          }
          else {
            xhat[i] <- xoff[i]/roff[i]
            yhat[i] <- yoff[i]/roff[i]
          }
        }
        else {
          xhat[i] <- xoff[i]/roff[i]
          yhat[i] <- yoff[i]/roff[i]
        }
        if (is.nan(xhat[i]) || xhat[i] == 0) 
          xhat[i] <- 0.01
        if (is.nan(yhat[i]) || yhat[i] == 0) 
          yhat[i] <- 0.01
      }
      xhat <- xhat[use]
      yhat <- yhat[use]
    }
    else if (label.pos < 5) {
      xhat <- switch(label.pos, 0, -1, 0, 1)
      yhat <- switch(label.pos, -1, 0, 1, 0)
    }
    else if (label.pos == 6) {
      xoff <- cx[use] - mean(cx[use])
      yoff <- cy[use] - mean(cy[use])
      roff <- sqrt(xoff^2 + yoff^2)
      xhat <- xoff/roff
      yhat <- yoff/roff
    }
    else {
      xhat <- 0
      yhat <- 0
    }
    os <- par()$cxy * mean(label.cex, na.rm = TRUE)
    lw <- strwidth(label[use], cex = label.cex)/2
    lh <- strheight(label[use], cex = label.cex)/2
    if (boxed.labels) {
      rect(cx[use] + xhat * vertex.radius[use] - (lh * 
                                                    label.pad + lw) * ((xhat < 0) * 2 + (xhat == 
                                                                                           0) * 1), cy[use] + yhat * vertex.radius[use] - 
             (lh * label.pad + lh) * ((yhat < 0) * 2 + (yhat == 
                                                          0) * 1), cx[use] + xhat * vertex.radius[use] + 
             (lh * label.pad + lw) * ((xhat > 0) * 2 + (xhat == 
                                                          0) * 1), cy[use] + yhat * vertex.radius[use] + 
             (lh * label.pad + lh) * ((yhat > 0) * 2 + (yhat == 
                                                          0) * 1), col = label.bg, border = label.border, 
           lty = label.lty, lwd = label.lwd)
    }
    text(cx[use] + xhat * vertex.radius[use] + (lh * label.pad + 
                                                  lw) * ((xhat > 0) - (xhat < 0)), cy[use] + yhat * 
           vertex.radius[use] + (lh * label.pad + lh) * ((yhat > 
                                                            0) - (yhat < 0)), label[use], cex = label.cex, col = label.col, 
         offset = 0)
  }
  if (interactive && ((length(cx) > 0) && (!all(use == FALSE)))) {
    os <- c(0.2, 0.4) * par()$cxy
    textloc <- c(min(cx[use]) - pad, max(cy[use]) + pad)
    tm <- "Select a vertex to move, or click \"Finished\" to end."
    tmh <- strheight(tm)
    tmw <- strwidth(tm)
    text(textloc[1], textloc[2], tm, adj = c(0, 0.5))
    fm <- "Finished"
    finx <- c(textloc[1], textloc[1] + strwidth(fm))
    finy <- c(textloc[2] - 3 * tmh - strheight(fm)/2, textloc[2] - 
                3 * tmh + strheight(fm)/2)
    finbx <- finx + c(-os[1], os[1])
    finby <- finy + c(-os[2], os[2])
    rect(finbx[1], finby[1], finbx[2], finby[2], col = "white")
    text(finx[1], mean(finy), fm, adj = c(0, 0.5))
    clickpos <- unlist(locator(1))
    if ((clickpos[1] %iin% finbx) && (clickpos[2] %iin% finby)) {
      cl <- match.call()
      cl$interactive <- FALSE
      cl$coord <- cbind(cx, cy)
      cl$x <- x
      return(eval.parent(cl))
    }
    else {
      clickdis <- sqrt((clickpos[1] - cx[use])^2 + (clickpos[2] - 
                                                      cy[use])^2)
      selvert <- match(min(clickdis), clickdis)
      if (all(label == "")) 
        label <- 1:n
      rect(textloc[1], textloc[2] - tmh/2, textloc[1] + 
             tmw, textloc[2] + tmh/2, border = "white", col = "white")
      tm <- "Where should I move this vertex?"
      tmh <- strheight(tm)
      tmw <- strwidth(tm)
      text(textloc[1], textloc[2], tm, adj = c(0, 0.5))
      fm <- paste("Vertex", label[use][selvert], "selected")
      finx <- c(textloc[1], textloc[1] + strwidth(fm))
      finy <- c(textloc[2] - 3 * tmh - strheight(fm)/2, 
                textloc[2] - 3 * tmh + strheight(fm)/2)
      finbx <- finx + c(-os[1], os[1])
      finby <- finy + c(-os[2], os[2])
      rect(finbx[1], finby[1], finbx[2], finby[2], col = "white")
      text(finx[1], mean(finy), fm, adj = c(0, 0.5))
      clickpos <- unlist(locator(1))
      cx[use][selvert] <- clickpos[1]
      cy[use][selvert] <- clickpos[2]
      cl <- match.call()
      cl$coord <- cbind(cx, cy)
      cl$x <- x
      return(eval.parent(cl))
    }
  }
  invisible(cbind(cx, cy))
}