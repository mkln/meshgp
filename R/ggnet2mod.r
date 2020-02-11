ggnet2m <- function (net, mode = "fruchtermanreingold", layout.par = NULL, 
          layout.exp = 0, alpha = 1, color = "grey75", shape = 19, 
          size = 9, max_size = 9, na.rm = NA, palette = NULL, alpha.palette = NULL, 
          alpha.legend = NA, color.palette = palette, color.legend = NA, 
          shape.palette = NULL, shape.legend = NA, size.palette = NULL, 
          size.legend = NA, size.zero = FALSE, size.cut = FALSE, size.min = NA, 
          size.max = NA, label = FALSE, label.alpha = 1, label.color = "black", 
          label.size = max_size/2, label.trim = FALSE, node.alpha = alpha, 
          node.color = color, node.label = label, node.shape = shape, 
          node.size = size, edge.alpha = 1, edge.color = "grey50", 
          edge.lty = "solid", edge.size = 0.25, edge.label = NULL, 
          edge.label.alpha = 1, edge.label.color = label.color, edge.label.fill = "white", 
          edge.label.size = max_size/2, arrow.size = 0, arrow.gap = 0, 
          arrow.type = "closed", legend.size = 9, legend.position = "right", 
          ...) 
{

  if (class(net) == "igraph" && "intergraph" %in% rownames(installed.packages())) {
    net = intergraph::asNetwork(net)
  }
  else if (class("net") == "igraph") {
    stop("install the 'intergraph' package to use igraph objects with ggnet2")
  }
  if (!network::is.network(net)) {
    net = try(network::network(net), silent = TRUE)
  }
  if (!network::is.network(net)) {
    stop("could not coerce net to a network object")
  }
  get_v = get("%v%", envir = as.environment("package:network"))
  get_e = get("%e%", envir = as.environment("package:network"))
  set_mode = function(x, mode = network::get.network.attribute(x, 
                                                               "bipartite")) {
    c(rep("actor", mode), rep("event", n_nodes - mode))
  }
  set_node = function(x, value, mode = TRUE) {
    if (is.null(x) || is.na(x) || is.infinite(x) || is.nan(x)) {
      stop(paste("incorrect", value, "value"))
    }
    else if (is.numeric(x) && any(x < 0)) {
      stop(paste("incorrect", value, "value"))
    }
    else if (length(x) == n_nodes) {
      x
    }
    else if (length(x) > 1) {
      stop(paste("incorrect", value, "length"))
    }
    else if (x %in% v_attr) {
      get_v(net, x)
    }
    else if (mode && x == "mode" & is_bip) {
      set_mode(net)
    }
    else {
      x
    }
  }
  set_edge = function(x, value) {
    if (is.null(x) || is.na(x) || is.infinite(x) || is.nan(x)) {
      stop(paste("incorrect", value, "value"))
    }
    else if (is.numeric(x) && any(x < 0)) {
      stop(paste("incorrect", value, "value"))
    }
    else if (length(x) == n_edges) {
      x
    }
    else if (length(x) > 1) {
      stop(paste("incorrect", value, "length"))
    }
    else if (x %in% e_attr) {
      get_e(net, x)
    }
    else {
      x
    }
  }
  set_attr = function(x) {
    if (length(x) == n_nodes) {
      x
    }
    else if (length(x) > 1) {
      stop(paste("incorrect coordinates length"))
    }
    else if (!x %in% v_attr) {
      stop(paste("vertex attribute", x, "was not found"))
    }
    else if (!is.numeric(get_v(net, x))) {
      stop(paste("vertex attribute", x, "is not numeric"))
    }
    else {
      get_v(net, x)
    }
  }
  set_name = function(x, y) {
    z = length(x) == 1 && x %in% v_attr
    z = ifelse(is.na(y), z, y)
    z = ifelse(isTRUE(z), x, z)
    ifelse(is.logical(z), "", z)
  }
  set_size = function(x) {
    y = x + (0 %in% x) * !size.zero
    y = scales::rescale_max(y)
    y = (scales::abs_area(max_size))(y)
    if (is.null(names(x))) 
      names(y) = x
    else names(y) = names(x)
    y
  }
  is_one = function(x) length(unique(x)) == 1
  is_col = function(x) all(is.numeric(x)) | all(network::is.color(x))
  n_nodes = network::network.size(net)
  n_edges = network::network.edgecount(net)
  v_attr = network::list.vertex.attributes(net)
  e_attr = network::list.edge.attributes(net)
  is_bip = network::is.bipartite(net)
  is_dir = ifelse(network::is.directed(net), "digraph", "graph")
  if (!is.numeric(arrow.size) || arrow.size < 0) {
    stop("incorrect arrow.size value")
  }
  else if (arrow.size > 0 & is_dir == "graph") {
    warning("network is undirected; arrow.size ignored")
    arrow.size = 0
  }
  if (!is.numeric(arrow.gap) || arrow.gap < 0 || arrow.gap > 
      1) {
    stop("incorrect arrow.gap value")
  }
  else if (arrow.gap > 0 & is_dir == "graph") {
    warning("network is undirected; arrow.gap ignored")
    arrow.gap = 0
  }
  if (network::is.hyper(net)) {
    stop("ggnet2 cannot plot hyper graphs")
  }
  if (network::is.multiplex(net)) {
    stop("ggnet2 cannot plot multiplex graphs")
  }
  if (network::has.loops(net)) {
    warning("ggnet2 does not know how to handle self-loops")
  }
  x = max_size
  if (!is.numeric(x) || is.infinite(x) || is.nan(x) || x < 
      0) {
    stop("incorrect max_size value")
  }
  data = data.frame(label = get_v(net, "vertex.names"), stringsAsFactors = FALSE)
  data$alpha = set_node(node.alpha, "node.alpha")
  data$color = set_node(node.color, "node.color")
  data$shape = set_node(node.shape, "node.shape")
  data$size = set_node(node.size, "node.size")
  if (length(na.rm) > 1) {
    stop("incorrect na.rm value")
  }
  else if (!is.na(na.rm)) {
    if (!na.rm %in% v_attr) {
      stop(paste("vertex attribute", na.rm, "was not found"))
    }
    x = which(is.na(get_v(net, na.rm)))
    message(paste("na.rm removed", length(x), "nodes out of", 
                  nrow(data)))
    if (length(x) > 0) {
      data = data[-x, ]
      network::delete.vertices(net, x)
      if (!nrow(data)) {
        warning("na.rm removed all nodes; nothing left to plot")
        return(invisible(NULL))
      }
    }
  }
  x = size
  if (length(x) == 1 && x %in% c("indegree", "outdegree", "degree", 
                                 "freeman")) {
    if ("package:igraph" %in% search()) {
      y = ifelse(is_dir == "digraph", "directed", "undirected")
      z = c(indegree = "in", outdegree = "out", degree = "all", 
            freeman = "all")[x]
      data$size = igraph::degree(igraph::graph.adjacency(as.matrix(net), 
                                                         mode = y), mode = z)
    }
    else {
      data$size = sna::degree(net, gmode = is_dir, cmode = ifelse(x == 
                                                                    "degree", "freeman", x))
    }
    size.legend = ifelse(is.na(size.legend), x, size.legend)
  }
  x = ifelse(is.na(size.min), 0, size.min)
  if (length(x) > 1 || !is.numeric(x) || is.infinite(x) || 
      is.nan(x) || x < 0) {
    stop("incorrect size.min value")
  }
  else if (x > 0 && !is.numeric(data$size)) {
    warning("node.size is not numeric; size.min ignored")
  }
  else if (x > 0) {
    x = which(data$size < x)
    message(paste("size.min removed", length(x), "nodes out of", 
                  nrow(data)))
    if (length(x) > 0) {
      data = data[-x, ]
      network::delete.vertices(net, x)
      if (!nrow(data)) {
        warning("size.min removed all nodes; nothing left to plot")
        return(invisible(NULL))
      }
    }
  }
  x = ifelse(is.na(size.max), 0, size.max)
  if (length(x) > 1 || !is.numeric(x) || is.infinite(x) || 
      is.nan(x) || x < 0) {
    stop("incorrect size.max value")
  }
  else if (x > 0 && !is.numeric(data$size)) {
    warning("node.size is not numeric; size.max ignored")
  }
  else if (x > 0) {
    x = which(data$size > x)
    message(paste("size.max removed", length(x), "nodes out of", 
                  nrow(data)))
    if (length(x) > 0) {
      data = data[-x, ]
      network::delete.vertices(net, x)
      if (!nrow(data)) {
        warning("size.max removed all nodes; nothing left to plot")
        return(invisible(NULL))
      }
    }
  }
  x = size.cut
  if (length(x) > 1 || is.null(x) || is.na(x) || is.infinite(x) || 
      is.nan(x)) {
    stop("incorrect size.cut value")
  }
  else if (isTRUE(x)) {
    x = 4
  }
  else if (is.logical(x) && !x) {
    x = 0
  }
  else if (!is.numeric(x)) {
    stop("incorrect size.cut value")
  }
  if (x >= 1 && !is.numeric(data$size)) {
    warning("node.size is not numeric; size.cut ignored")
  }
  else if (x >= 1) {
    x = unique(quantile(data$size, probs = seq(0, 1, by = 1/as.integer(x))))
    if (length(x) > 1) {
      data$size = cut(data$size, unique(x), include.lowest = TRUE)
    }
    else {
      warning("node.size is invariant; size.cut ignored")
    }
  }
  if (!is.null(alpha.palette)) {
    x = alpha.palette
  }
  else if (is.factor(data$alpha)) {
    x = levels(data$alpha)
  }
  else {
    x = unique(data$alpha)
  }
  if (!is.null(names(x))) {
    y = unique(na.omit(data$alpha[!data$alpha %in% names(x)]))
    if (length(y) > 0) {
      stop(paste("no alpha.palette value for", paste0(y, 
                                                      collapse = ", ")))
    }
  }
  else if (is.factor(data$alpha) || !is.numeric(x)) {
    data$alpha = factor(data$alpha)
    x = scales::rescale_max(1:length(levels(data$alpha)))
    names(x) = levels(data$alpha)
  }
  alpha.palette = x
  if (!is.null(color.palette)) {
    x = color.palette
  }
  else if (is.factor(data$color)) {
    x = levels(data$color)
  }
  else {
    x = unique(data$color)
  }
  if (length(x) == 1 && "RColorBrewer" %in% rownames(installed.packages()) && 
      x %in% rownames(RColorBrewer::brewer.pal.info)) {
    data$color = factor(data$color)
    n_groups = length(levels(data$color))
    n_colors = RColorBrewer::brewer.pal.info[x, "maxcolors"]
    if (n_groups > n_colors) {
      stop(paste0("too many node groups (", n_groups, ") for ", 
                  "ColorBrewer palette ", x, " (max: ", n_colors, 
                  ")"))
    }
    else if (n_groups < 3) {
      n_groups = 3
    }
    x = RColorBrewer::brewer.pal(n_groups, x)[1:length(levels(data$color))]
    names(x) = levels(data$color)
  }
  if (!is.null(names(x))) {
    y = unique(na.omit(data$color[!data$color %in% names(x)]))
    if (length(y) > 0) {
      stop(paste("no color.palette value for", paste0(y, 
                                                      collapse = ", ")))
    }
  }
  else if (is.factor(data$color) || !is_col(x)) {
    data$color = factor(data$color)
    x = gray.colors(length(x))
    names(x) = levels(data$color)
  }
  color.palette = x
  if (!is.null(shape.palette)) {
    x = shape.palette
  }
  else if (is.factor(data$shape)) {
    x = levels(data$shape)
  }
  else {
    x = unique(data$shape)
  }
  if (!is.null(names(x))) {
    y = unique(na.omit(data$shape[!data$shape %in% names(x)]))
    if (length(y) > 0) {
      stop(paste("no shape.palette value for", paste0(y, 
                                                      collapse = ", ")))
    }
  }
  else if (is.factor(data$shape) || !is.numeric(x)) {
    data$shape = factor(data$shape)
    x = (scales::shape_pal())(length(levels(data$shape)))
    names(x) = levels(data$shape)
  }
  shape.palette = x
  if (!is.null(size.palette)) {
    x = size.palette
  }
  else if (is.factor(data$size)) {
    x = levels(data$size)
  }
  else {
    x = unique(data$size)
  }
  if (!is.null(names(x))) {
    y = unique(na.omit(data$size[!data$size %in% names(x)]))
    if (length(y) > 0) {
      stop(paste("no size.palette value for", paste0(y, 
                                                     collapse = ", ")))
    }
  }
  else if (is.factor(data$size) || !is.numeric(x)) {
    data$size = factor(data$size)
    x = 1:length(levels(data$size))
    names(x) = levels(data$size)
  }
  size.palette = x
  l = node.label
  if (isTRUE(l)) {
    l = data$label
  }
  else if (length(l) > 1 & length(l) == n_nodes) {
    data$label = l
  }
  else if (length(l) == 1 && l %in% v_attr) {
    l = get_v(net, l)
  }
  else {
    l = ifelse(data$label %in% l, data$label, "")
  }
  if (is.character(mode) && length(mode) == 1) {
    mode = paste0("gplot.layout.", mode)
    if (!exists(mode)) {
      stop(paste("unsupported placement method:", mode))
    }
    xy = network::as.matrix.network.adjacency(net)
    xy = do.call(mode, list(xy, layout.par))
    xy = data.frame(x = xy[, 1], y = xy[, 2])
  }
  else if (is.character(mode) && length(mode) == 2) {
    xy = data.frame(x = set_attr(mode[1]), y = set_attr(mode[2]))
  }
  else if (is.numeric(mode) && is.matrix(mode)) {
    xy = data.frame(x = set_attr(mode[, 1]), y = set_attr(mode[, 
                                                               2]))
  }
  else {
    stop("incorrect mode value")
  }
  if (length(mode) == 1) {
    #xy$x = scale(xy$x, min(xy$x), diff(range(xy$x)))
    #xy$y = scale(xy$y, min(xy$y), diff(range(xy$y)))
  }
  data = cbind(data, xy)
  edges = network::as.matrix.network.edgelist(net)
  if (edge.color[1] == "color" && length(edge.color) == 2) {
    edge.color = ifelse(data$color[edges[, 1]] == data$color[edges[, 
                                                                   2]], as.character(data$color[edges[, 1]]), edge.color[2])
    if (!is.null(names(color.palette))) {
      x = which(edge.color %in% names(color.palette))
      edge.color[x] = color.palette[edge.color[x]]
    }
    edge.color[is.na(edge.color)] = edge.color[2]
  }
  edge.color = set_edge(edge.color, "edge.color")
  if (!is_col(edge.color)) {
    stop("incorrect edge.color value")
  }
  edges = data.frame(xy[edges[, 1], ], xy[edges[, 2], ])
  names(edges) = c("X1", "Y1", "X2", "Y2")
  if (!is.null(edge.label)) {
    edges$midX = (edges$X1 + edges$X2)/2
    edges$midY = (edges$Y1 + edges$Y2)/2
    edges$label = set_edge(edge.label, "edge.label")
    edge.label.alpha = set_edge(edge.label.alpha, "edge.label.alpha")
    if (!is.numeric(edge.label.alpha)) {
      stop("incorrect edge.label.alpha value")
    }
    edge.label.color = set_edge(edge.label.color, "edge.label.color")
    if (!is_col(edge.label.color)) {
      stop("incorrect edge.label.color value")
    }
    edge.label.size = set_edge(edge.label.size, "edge.label.size")
    if (!is.numeric(edge.label.size)) {
      stop("incorrect edge.label.size value")
    }
  }
  edge.lty = set_edge(edge.lty, "edge.lty")
  edge.size = set_edge(edge.size, "edge.size")
  if (!is.numeric(edge.size) || any(edge.size <= 0)) {
    stop("incorrect edge.size value")
  }
  p = ggplot(data, aes(x = x, y = y))
  if (nrow(edges) > 0) {
    if (arrow.gap > 0) {
      x.length = with(edges, X2 - X1)
      y.length = with(edges, Y2 - Y1)
      #arrow.gap = with(edges, arrow.gap/sqrt(x.length^2 + 
      #                                         y.length^2))
      edges = transform(edges, 
                        X1 = X1 + arrow.gap * x.length, 
                        Y1 = Y1 + arrow.gap * y.length, 
                        X2 = X1 + (1 - arrow.gap) * x.length, 
                        Y2 = Y1 + (1 - arrow.gap) * y.length)
    }
    p = p + geom_segment(data = edges, aes(x = X1, y = Y1, 
                                           xend = X2, yend = Y2), size = edge.size, color = edge.color, 
                         alpha = edge.alpha, lty = edge.lty, arrow = arrow(type = arrow.type, 
                                                                           length = unit(arrow.size, "pt")))
  }
  if (nrow(edges) > 0 && !is.null(edge.label)) {
    p = p + geom_point(data = edges, aes(x = midX, y = midY), 
                       alpha = edge.alpha, color = edge.label.fill, size = edge.label.size * 
                         1.5) + geom_text(data = edges, aes(x = midX, 
                                                            y = midY, label = label), alpha = edge.label.alpha, 
                                          color = edge.label.color, size = edge.label.size)
  }
  x = list()
  if (is.numeric(data$alpha) && is_one(data$alpha)) {
    x = c(x, alpha = unique(data$alpha))
  }
  if (!is.factor(data$color) && is_one(data$color)) {
    x = c(x, colour = unique(data$color))
  }
  if (is.numeric(data$shape) && is_one(data$shape)) {
    x = c(x, shape = unique(data$shape))
  }
  if (is.numeric(data$size) && is_one(data$size)) {
    x = c(x, size = unique(data$size))
  }
  else {
    x = c(x, size = max_size)
  }
  p = p + geom_point(aes(alpha = factor(alpha), color = factor(color), 
                         shape = factor(shape), size = factor(size)))
  if (is.numeric(data$alpha)) {
    v_alpha = unique(data$alpha)
    names(v_alpha) = unique(data$alpha)
    p = p + scale_alpha_manual("", values = v_alpha) + guides(alpha = FALSE)
  }
  else {
    p = p + scale_alpha_manual(set_name(node.alpha, alpha.legend), 
                               values = alpha.palette, breaks = names(alpha.palette), 
                               guide = guide_legend(override.aes = x))
  }
  if (!is.null(names(color.palette))) {
    p = p + scale_color_manual(set_name(node.color, color.legend), 
                               values = color.palette, breaks = names(color.palette), 
                               guide = guide_legend(override.aes = x))
  }
  else {
    v_color = unique(data$color)
    names(v_color) = unique(data$color)
    p = p + scale_color_manual("", values = v_color) + guides(color = FALSE)
  }
  if (is.numeric(data$shape)) {
    v_shape = unique(data$shape)
    names(v_shape) = unique(data$shape)
    p = p + scale_shape_manual("", values = v_shape) + guides(shape = FALSE)
  }
  else {
    p = p + scale_shape_manual(set_name(node.shape, shape.legend), 
                               values = shape.palette, breaks = names(shape.palette), 
                               guide = guide_legend(override.aes = x))
  }
  x = x[names(x) != "size"]
  if (is.numeric(data$size)) {
    v_size = set_size(unique(data$size))
    if (length(v_size) == 1) {
      v_size = as.numeric(names(v_size))
      p = p + scale_size_manual("", values = v_size) + 
        guides(size = FALSE)
    }
    else {
      p = p + scale_size_manual(set_name(node.size, size.legend), 
                                values = v_size, guide = guide_legend(override.aes = x))
    }
  }
  else {
    p = p + scale_size_manual(set_name(node.size, size.legend), 
                              values = set_size(size.palette), guide = guide_legend(override.aes = x))
  }
  if (!is_one(l) || unique(l) != "") {
    label.alpha = set_node(label.alpha, "label.alpha", mode = FALSE)
    if (!is.numeric(label.alpha)) {
      stop("incorrect label.alpha value")
    }
    label.color = set_node(label.color, "label.color", mode = FALSE)
    if (!is_col(label.color)) {
      stop("incorrect label.color value")
    }
    label.size = set_node(label.size, "label.size", mode = FALSE)
    if (!is.numeric(label.size)) {
      stop("incorrect label.size value")
    }
    x = label.trim
    if (length(x) > 1 || (!is.logical(x) & !is.numeric(x) & 
                          !is.function(x))) {
      stop("incorrect label.trim value")
    }
    else if (is.numeric(x) && x > 0) {
      l = substr(l, 1, x)
    }
    else if (is.function(x)) {
      l = x(l)
    }
    p = p + geom_text(label = l, alpha = label.alpha, color = label.color, 
                      size = label.size, ...)
  }
  x = range(data$x)
  if (!is.numeric(layout.exp) || layout.exp < 0) {
    stop("incorrect layout.exp value")
  }
  else if (layout.exp > 0) {
    x = scales::expand_range(x, layout.exp/2)
  }
  p = p + scale_x_continuous(breaks = NULL, limits = x) + scale_y_continuous(breaks = NULL) + 
    theme(panel.background = element_blank(), panel.grid = element_blank(), 
          axis.title = element_blank(), legend.key = element_blank(), 
          legend.position = legend.position, legend.text = element_text(size = legend.size), 
          legend.title = element_text(size = legend.size))
  return(p)
}