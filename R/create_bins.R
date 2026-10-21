#' Fit the bins on the training set.
#'
#' Computes, for every column, how raw values are mapped to bin indices. Bin indices are nominal classes, following BinXent. No order is implied.
#'
#' @section Numerical columns (paper, Section 4):
#' The bin boundaries are the quantiles t/T, t = 1..T-1, of the training
#' column. If the column has at most T distinct values, each distinct value is
#' its own bin.
#'
#' @section Categorical columns (our choice; the paper does not cover this):
#' K = number of levels declared in `cat_dims`.
#'  - K <= T: each level is its own bin (bin index = level code).
#'  - K  > T: the T-1 most frequent levels in the training set get one bin each and all the other levels, including levels unseen in training, share the bin "Others" (the last bin).
#'
#' @section Deviations from the paper:
#' 1. Empty bins produced by ties are removed, so bins are all non-empty in training. Values falling in a removed bin go to the nearest non-empty one.
#' 2. The kind of column (numerical or categorical) comes from 'num_cont' and 'cat_dims', not from a threshold on the number of distinct values.
#' 3. Categorical columns with more than T levels are grouped by frequency. The T most frequent colums have their own bins. Other columns falls under a common 'others' bin.
#'
#'
#' @param dataset_train Training partition of the dataset.
#' @param metadata_for_cat It contains information about categorical columns. Default is \code{NULL}. User do not have to provide it.
#' @param T Maximum number of bins per column. of target bins. Default: \code{10}.
#' @param handle_categorical Only \code{'top_frequent'} is implemented for now.
#'
#' @returns An object of class "metadata_for_binning" that contains the information needed to apply bins on a specific dataset.
get_bins = function(dataset_train, metadata_for_cat = NULL, T = 10, handle_categorical = "top_frequent") {

  # TODO: Check que pasa cuando no hay variables categóricas. Da error o no????
  # TODO: Check que metadata_for_cat no es null.
  num_cont <- length(metadata_for_cat$num_cols)
  cat_dims <- metadata_for_cat$cat_dims

  # Check exceptions
  if (!identical(handle_categorical, "top_frequent")) {
    stop("Only handle_categorical = 'top_frequent' is implemented")
  }

  if (!is.numeric(T) || T < 2) {
    stop("T must be a number >= 2")
  }

  dataset_train <- as.matrix(dataset_train)

  total_number_of_features <- num_cont + length(cat_dims)

  if (ncol(dataset_train) != total_number_of_features) {
    stop("dataset_train contains a different number of colums than the expected")
  }

  if (anyNA(dataset_train)) {
    stop("dataset_train contains NAs. Impute them before binning")
  }

  features <- vector("list", total_number_of_features)

  # Process and bin numerical features
  for (i in seq_len(num_cont)) {
    features[[i]] <- bin_numeric(dataset_train[, i], T)
  }

  # Process and bin categorical features
  for (i in seq_along(cat_dims)) {
    j <- num_cont + i

    # Check correct encoding
    codes <- check_codes(dataset_train[, j], cat_dims[i], j)

    features[[j]] <- bin_categorical(codes, cat_dims[i], T)
  }

  #print(features)

  return(structure(
    list(
      T = T,
      num_cont = num_cont,
      cat_dims = cat_dims,
      n_bins = vapply(features, function(f) as.integer(f$n_bins), integer(1)),
      features = features,
      handle_categorical = handle_categorical
    ),
    class = "metadata_for_binning"
  ))


}




bin_categorical <- function(x, dimension, T) {

  # Case A: unique values are less than T -> one bin per value
  if (dimension <= T) {
    return (list(
      type = "categorical",
      K = dimension,
      map = seq_len(dimension),
      n_bins = dimension,
      other_share = NULL
    ))
  }

  # Case B: unique values are bigger than T -> T-1 most frequent levels at its own category + bin "Others"
  freq <- tabulate(x, nbins = dimension)  # Count the number of occurrences in train
  ord <- order(-freq, seq_len(dimension))  # Order from higher to less frequency
  n_kept <- min(T-1L, sum(freq > 0))  # Number of levels with its own bin. T-1 or less if train has fewer levels than T
  # Assign bins to the most frequent
  map    <- rep.int(n_kept + 1L, dimension)
  map[ord[seq_len(n_kept)]] <- seq_len(n_kept)

  return(list(
    type = "categorical",
    K = dimension,
    map = map,
    n_bins = n_kept + 1L,
    other_share = mean(map[x] == n_kept + 1L)
  ))
}


# x: a column of the dataset
# T: max number of possible bins
bin_numeric <- function(x, T) {

  # Unique values of the column and sorted
  u <- sort(unique(x))

  # Case A: unique values are less than T -> one bin per value
  if (length(u) <= T) {
    cuts <- u[-1]  # Cut points are represented by all the elements (except the first one. [3,3,7,9] cuts would be [7,9] first bin (-inf, 7), second bin (7,9), third bin (9, inf))
    map <- seq_along(u)
    n_bins <- length(u)
    n_unique <- length(u)

    return(list(type = "numeric", cuts = cuts, map = map, n_bins = n_bins, n_unique = n_unique))
  }

  # Case B: unique values are bigger than T -> compute cuantiles
  cuts <- unique(stats::quantile(x, probs = seq_len(T - 1) / T, type = 7, names = FALSE))
  cuts <- unique(cuts)

  # Check unused bins
  raw <- findInterval(x = x, vec = cuts) + 1L  # Bin that contain and do not contain elements (repeated)
  used <- sort(unique(raw))  # Bins that contain elements

  # Map every raw bin to a consecutive index
  map <- vapply(seq_len(length(cuts) + 1L),
                function(r) which.min(abs(used - r)), integer(1))

  return (list(
    type = "numeric",
    cuts = cuts,
    map = map,
    n_bins = length(used),
    n_uniques = length(u)
  ))
}


# Check encoding of a column
check_codes <- function(x, dims, j) {
  codes <- as.integer(round(x))
  if (anyNA(codes) || any(codes < 1L | codes > dims)) {
    stop("Column ", j, ": categorical codes must be in 1..", dims,
         " (found: ", paste(sort(unique(codes)), collapse = ", "), ").")
  }
  codes
}







#' Apply fitted bins to a dataset (train, validation, etc)
#'
#'
#'
#' @param x Data with the same columns as the training data. NULL returns NULL, useful when there is no validation set.
#' @param bins Output of latter::get_bins().
#'
#' @returns An integer matrix with the bin index of every value.
apply_bins <- function(x, bins) {
  if (is.null(x)) return(NULL)  # For instance, if there is no validation set

  x <- as.matrix(x)

  d <- length(bins$features)
  if (ncol(x) != d) stop("x has ", ncol(x), " columns, expected ", d, ".")
  if (anyNA(x)) stop("x contains NAs; impute them before binning.")

  output <- matrix(NA_integer_, nrow = nrow(x), ncol = d, dimnames = dimnames(x))

  for (j in seq_len(d)) {
    feature <- bins$features[[j]]  # get_bins -> returned structure

    if (feature$type == "numeric") {
      raw <- if (length(feature$cuts) == 0) rep(1L, nrow(x)) else findInterval(x[, j], feature$cuts) + 1L
      output[, j] <- feature$map[raw]

    } else if (feature$type == "categorical") {
      code <- check_codes(x[,j], feature$K, j)
      output[, j] <- feature$map[code]

    } else {
      stop("The created bins has an erroneous type")
    }
  }

  storage.mode(output) <- "integer"  # Store it as integer, rather than double

  return (output)  # [num filas, num_columnas] - Para cada columna del batch, su índice o bin correcto
}







