#' Divide a batch of "d" number of columns each row to "n_views" subsets
#'
#' @param d
#' @param n_views
#' @param overlap
#' @param shuffle
#'
#' @returns
#' @export
#'
#' @examples
make_subtab_views__alternative <- function(d, n_views, overlap = 0.5, shuffle = TRUE) {
  if (!is.numeric(d) || d < 1) {
    stop("make_subtab_views: d must be a positive integer")
  }

  if (!is.numeric(n_views) || n_views < 1 || n_views > d) {
    stop("make_subtab_views: n_views must be an integer between 1 and d")
  }

  if (!is.numeric(overlap) || overlap < 0 || overlap >= 1) {
    stop("overlap must be in [0, 1)")
  }

  perm <- if (shuffle) sample.int(d) else seq_len(d)

  # View size and step between consecutive views, in "positions".
  # size + (K - 1) * step = d, and size - step = overlap * size
  size <- if (n_views == 1) d else d / (1 + (n_views - 1) * (1 - overlap))
  step <- if (n_views == 1) 0 else (d - size) / (n_views - 1)

  # Position i (1..d) is represented by its center i - 0.5. The tiny shift makes
  # a center that falls exactly on a boundary belong to the later view
  centers <- seq_len(d) - 0.5 + 1e-9

  views <- matrix(FALSE, nrow = n_views, ncol = d)
  for (k in seq_len(n_views)) {
    from <- (k - 1) * step
    inside <- centers >= from & centers < from + size  # logical [d], by position
    views[k, perm[inside]] <- TRUE                     # position -> original column
  }

  return (views)
}






make_subtab_views_official <- function(d, n_views, overlap) {
  # TODO: hay que llamar a esto en qué momento? Antes de introducir ruido

  stopifnot(n_views >= 1, n_views <= d, overlap >= 0, overlap <= 1)


  s <- d %/% n_views  # Ancho base
  print(s)

  # Number of overlapping features between subsets
  n_overlap <- floor(overlap * s)
  print(n_overlap)

  # TRUE will be ancho base + overlapping features. FALSE remaining columns

  if (d %% n_views != 0) {
    warning(d %% n_views, " trailing column(s) are never used as input (as in the reference implementation).")
  }


  views <- matrix(FALSE, n_views, d)
  for (i in seq_len(n_views) - 1L) {                # i = 0..K-1
    from <- if (i == 0) 0 else i * s - n_overlap  # Same as in the official repo
    to   <- if (i == 0) s + n_overlap else (i + 1) * s
    views[i + 1, (from + 1):min(to, d)] <- TRUE     # slice [from:to) -> columns from+1..to
  }

  return(views)

  # TODO: Si solo hay una vista, el paper oficial la duplica para luego poder aplicar la pérdida contrastiva creo
}


apply_view_mask <- function(x, views, num_cont, cat_dims = NULL, add_indicator = TRUE) {

  num_cat <- length(cat_dims)
  num_features <- x$size(2)
  batch <- x$size(1)

  if (num_cont + num_cat != num_features) {
    stop("num_cont + length(cat_dims) (", num_cont + num_cat,
         ") must equal the number of columns of x (", num_features, ")")
  }

  # Transform to tensor and 0-1 (from FALSE-TRUE)
  if (!inherits(views, "torch_tensor")) {
    views <- torch::torch_tensor(views, dtype = torch::torch_bool(), device = x$device)
  }

  if (views$dim() == 1) views <- views$unsqueeze(1)

  if (views$size(2) != num_featuresd) {
    stop("views has ", views$size(2), " columns but x has ", num_features)
  }

  n_views <- views$size(1)


  x3   <- x$unsqueeze(1)       # [1, B, num_features]
  keep <- views$unsqueeze(2)   # [K, 1, num_features]

  parts <- list()
  observed <- NULL

  # Mask numerical features
  if (num_cont > 0) {
    keep_cont <- keep[, , 1:num_cont]$to(dtype = x$dtype)          # [K, 1, nc]
    parts[[length(parts) + 1]] <- x3[, , 1:num_cont] * keep_cont   # [K, B, nc]  Apply mask to numerical columns (* 0 or * 1)

    # Create binary tensor to indicate whether a feature has been masked or not
    if (add_indicator) {
      observed <- keep_cont$expand(c(n_views, batch, num_cont))$reshape(c(n_views * batch, num_cont))
    }
  }

  # Mask categorical features
  if (num_cat > 0) {
    cols <- (num_cont + 1):num_features
    tokens <- torch::torch_tensor(cat_dims + 2, dtype = x$dtype, device = x$device)  # [nc_cat]
    parts[[length(parts) + 1]] <- torch::torch_where(keep[, , cols], x3[, , cols], tokens)  # if keep, x3 (original), else mask token
  }

  masked <- torch::torch_cat(parts, dim = 3)$reshape(c(n_views * batch, num_features))

  return (list(x = masked,  # Masked data
               observed = observed,  # Indicator for numerical features
               n_views = n_views  # Number of views generated
               )
          )
}




