#

#' Split the columns of a dataset into SubTab views (same procedure as the official implementation)
#'
#' A view is a bubset of columns. Neighbouring views could share columns (overlap).
#'
#' Every view has exactly the same number of columns. This is required by the encoder, which is shared by all views. The last \code{d \%\% n_views} columns are not used by any view. A warning list them.
#'
#' @param d \code{Integer}. Total number of columns after one-hot encoding
#' @param n_views \code{Integer}. Number of views to create. It must be between 1 and \code{d}
#' @param overlap \code{Numeric} in [0,1]. Fraction of the base width by which each view is widened.
#' @param shuffle \code{Boolean}. If \code{TRUE}, the columns are randomly permuted. The official implementation does not do it.
#' @param column_names \code{Character}. Optional. Names of the columns, only used to make the warning about discarded columns easier to read.
#'
#' @returns A \code{logical} matrix of size [n_views, d]. \code{views[k, j] = TRUE} means that column \code{j} belongs to view \code{k}.
#' @noRd
make_subtab_views <- function(d, n_views, overlap = 0.5, shuffle = FALSE, column_names = NULL) {

  # Validate input
  if (!is.numeric(d) || length(d) != 1 || d < 1 || d != round(d)) {
    stop("make_subtab_views: 'd' must be a positive integer")
  }

  if (!is.numeric(n_views) || length(n_views) != 1 || n_views < 1 || n_views > d || n_views != round(n_views)) {
    stop("make_subtab_views: 'n_views' must be an integer between 1 and 'd'")
  }

  if (!is.numeric(overlap) || length(overlap) != 1 || overlap < 0 || overlap > 1) {
    stop("make_subtab_views: 'overlap' must be in [0, 1]")
  }

  if (!is.null(column_names) && length(column_names) != d) {
    stop("make_subtab_views: 'column_names' must have 'd' elements")
  }


  base_width <- d %/% n_views  # Integer division (python: //)
  print(base_width)
  n_overlap <- floor(overlap * base_width)  # Extra columns of each view that are shared with other views
  print(n_overlap)

  views_by_position <- matrix(FALSE, nrow = n_views, ncol = d)

  for (i in seq_len(n_views) - 1L) {  # i = 0..K-1
    # Slice [from, to) in 0-based indexes
    from <- if (i == 0) 0 else i * base_width - n_overlap

    to <- if (i == 0) {
      base_width + n_overlap   # First view: its block + overlap on the right
    } else {
      (i + 1) * base_width     # The rest: until the end of their block
    }

    views_by_position[i + 1, (from + 1):min(to, d)] <- TRUE
  }

  # Shuffle if required
  perm <- if (shuffle) sample.int(d) else seq_len(d)

  views <- matrix(FALSE, nrow = n_views, ncol = d)
  views[, perm] <- views_by_position  # Column perm[p] of 'views' receives column p of 'views_by_position'


  # Columns that are not used
  discarded <- which(colSums(views) == 0)

  if (length(discarded) > 0) {
    discarded_text <- if (is.null(column_names)) discarded else column_names[discarded]

    warning("make_subtab_views: ", d, " columns cannot be split evenly into ", n_views, " views of the same width. ",
            "As in the official implementation, the last ", length(discarded), " column(s) are not used by any view: ",
            paste(discarded_text, collapse = ", "), call. = FALSE)
  }

  return(views)
}







#' Divide a batch of \code{d} number of columns each row to \code{n_views} subsets
#'
#' @param d \code{Integer}. Total number of columns of the preprocessed data
#' @param n_views \code{Integer}. Number of views to create. It must be between 1 and \code{d}
#' @param overlap \code{Numeric} in [0,1]. Overlap between neighbouring views. Its
#' @param shuffle \code{Boolean}. If \code{TRUE}, the columns are randomly permuted once before being split, so the views do not depend on the column order (numerical columns come first and categorical come last).
#'
#' @returns A \code{logical} matrix of size [n_views, d]. \code{views[k,j] = TRUE} means that column \code{j} belongs to view \code{k}.
#' @noRd
make_subtab_views_adapted <- function(d, n_views, overlap, shuffle = TRUE) {
  # TODO: hay que llamar a esto en qué momento? Antes de introducir ruido

  if (!is.numeric(d) || d < 1) {
    stop("make_subtab_views: d must be a positive integer")
  }

  if (!is.numeric(n_views) || n_views < 1 || n_views > d) {
    stop("make_subtab_views: n_views must be an integer between 1 and d")
  }

  if (!is.numeric(overlap) || overlap < 0 || overlap >= 1) {
    stop("overlap must be in [0, 1)")
  }




  s <- d %/% n_views  # Ancho base. División integral. Devuelve un entero (python //)
  print(s)

  # Number of overlapping features between subsets
  n_overlap <- floor(overlap * s)
  print(n_overlap)

  # TRUE will be ancho base + overlapping features. FALSE remaining columns

  # if (d %% n_views != 0) {
  #   warning(d %% n_views, " column(s) (at the end) are never used as input (as in the official implementation).")
  # }

  # If d is not divisible, the remainder goes to the last view, which will be wider than the others
  if (d %% n_views != 0) {
    warning("make_subtab_views: ", d, " columns cannot be split evenly into ", n_views,
            " views. The last view is wider because it takes the remaining ", d %% n_views, " column(s).")
  }


  first_views <- matrix(FALSE, n_views, d)
  for (i in seq_len(n_views) - 1L) {                # i = 0..K-1
    from <- if (i == 0) 0 else i * s - n_overlap  # Same as in the official repo
    to   <- if (i == 0) {
      s + n_overlap  # First
    } else if (i == n_views - 1L) {  # Added this so that no column is avoided.
      d  # Last
    } else {
      (i + 1) * s  # Rest
    }


    if (i == 0) s + n_overlap else (i + 1) * s
    first_views[i + 1, (from + 1):min(to, d)] <- TRUE     # slice [from:to) -> columns from+1..to
  }


  # Shuffle if required. TRUE in the official repo (subtab > src > model.py)
  perm <- if (shuffle) sample.int(d) else seq_len(d)  # Vector of length d with random values between [1, d] if shuffle = TRUE. Ordered vector if shuffle = FALSE

  views <- matrix(FALSE, nrow = n_views, ncol = d)
  views[, perm] <- first_views  # Column perm[p] of views receives column p of first_views


  return(views)

  # TODO: Si solo hay una vista, el paper oficial la duplica para luego poder aplicar la pérdida contrastiva creo
}


apply_view_mask <- function(x, views, num_cont, cat_dims = NULL, add_indicator = TRUE) {

  x <- torch::torch_tensor(as.matrix(x), dtype = torch::torch_float32())  # TODO borrar esto en un futuro será tensor de base creo

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

  if (views$size(2) != num_features) {
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

    # Create binary tensor to indicate whether a feature has been masked or not.
    # When masked, numerical columns are set to 0, which is a common value. add_indicator help the model to determine whether a
    # numerical column has been masked or not.
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




