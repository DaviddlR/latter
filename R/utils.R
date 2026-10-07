

#' Function to read and check a bundle containing a pretrained model
#'
#' @param bundle_path Path to file storing a bundle object, trained with "fit_extractor"
#' @returns Pretrained model's weights and trained recipe for preprocessing
#' @noRd
load_bundle = function(bundle_path) {


  if (is.character(bundle_path)){
    # Stored locally. Load pretrained model and recipe
    model_bundle <- torch::torch_load(paste(bundle_path, ".pt", sep=""))
  } else if (is.list(bundle_path)) {
    # Stored in RAM. No need to load.
    model_bundle <- bundle_path
  } else {
    stop("The 'bundle_path' parameter must be a .pt file created with 'scarf_fit' or a list object type SCARF bundle")
  }

  # Validate input

  # SCARF bundle
  if (is.list(model_bundle) && identical(model_bundle$bundle_type, "scarf_bundle")) {

    # Load encoder hyperparameters
    hparams <- model_bundle$encoder_hparams

    print(hparams)

    # Create a new encoder
    fitted_encoder <- scarf_encoder(
      num_cont = hparams$num_cont,
      cat_dims = hparams$cat_dims,
      hidden_dim = hparams$hidden_dim,
      num_hidden = hparams$num_hidden,
      dropout = hparams$dropout
    )

    # Load trained weights
    fitted_encoder$load_state_dict(model_bundle$encoder_state_dict)

    # Load recipe
    trained_recipe <- unserialize(model_bundle$recipe)

    # Load info for categorical data
    info_for_categorical <- model_bundle$metadata_for_cat

    return(list(
      encoder = fitted_encoder,
      recipe = trained_recipe,
      metadata_for_cat = info_for_categorical

    ))

  # VIME bundle
  } else if (is.list(model_bundle) && identical(model_bundle$bundle_type, "vime_bundle")) {

    print("Extracting a VIME bundle")

    # Load encoder hyperparameters
    hparams <- model_bundle$encoder_hparams

    print(hparams)

    # Create a new encoder
    fitted_encoder <- vime_encoder(
      num_cont = hparams$num_cont,
      cat_dims = hparams$cat_dims
    )

    # Load trained weights
    fitted_encoder$load_state_dict(model_bundle$encoder_state_dict)

    # Load recipe
    trained_recipe <- unserialize(model_bundle$recipe)

    # Load info for categorical data
    info_for_categorical <- model_bundle$metadata_for_cat

    return(list(
      encoder = fitted_encoder,
      recipe = trained_recipe,
      metadata_for_cat = info_for_categorical
    ))


  } else {
    stop("The input is not a valid bundle. Please, train a model using fit_extractor() and set the pretrained_model_path to the path in which the trained model is stored")
  }
}



#' Function to read and check a bundle containing a trained classifier
#'
#' @param bundle_path Path to file storing a bundle object, trained with "train_classifier_on_extracted_features"
#' @returns The classifier model and levels for preprocessing
#' @noRd
load_classifier_bundle = function(bundle_path) {

  # Load pretrained model and recipe
  model_bundle <- torch::torch_load(bundle_path)

  # Check if classifier comes from torch
  if(is.list(model_bundle) && identical(model_bundle$bundle_type, "classifier_torch_bundle")) {

    # Load encoder hyperparameters
    classifier_weights <- model_bundle$classifier_state_dict
    classifier_hparams <- model_bundle$classifier_hparams

    classifier_net <- classifier_network(
      input_dim = classifier_hparams$in_dim,
      n_classes = classifier_hparams$n_classes,
      dropout = classifier_hparams$dropout
    )

    # Load weights
    classifier_net$load_state_dict(classifier_weights)


    # Load levels
    trained_levels <- unserialize(model_bundle$levels)

    return(list(
      classifier = classifier_net,
      levels = trained_levels,
      type = "torch"
    ))

  # Check if classifier comes from parsnip
  } else if (is.list(model_bundle) && identical(model_bundle$bundle_type, "classifier_parsnip_bundle")) {

    trained_levels <- unserialize(model_bundle$levels)
    classifier_model <- unserialize(model_bundle$classifier_model)

    return(list(
      classifier = classifier_model,
      levels = trained_levels,
      type = "parsnip"
    ))

  } else {
    stop("The input is not a scarf_bundle. Please, train the model using scarf_fit() and set the pretrained_model_path to the path in which the trained model is stored")
  }

}






create_validation_set = function(x, validation_proportion) {
  n_samples <- nrow(x)
  validation_size <- floor(validation_proportion * n_samples)
  val_indices <- sample(seq_len(n_samples), size=validation_size)

  x_val <- x[val_indices, , drop=FALSE]
  x_train <- x[-val_indices, , drop=FALSE]

  return(list(
    x_tr = x_train,
    x_val = x_val
  ))
}



# Get information about categorical variables for transforming them to integers.
# If metadata is null, the method collect it from the training set. Else, it uses
# the metadata to transform new data (validation and test)
encode_categorical_data = function(data, metadata = NULL) {

  # If metadata is null, create it from data (should be training dataset)
  if (is.null(metadata)) {
    # Identify categorical and numerical columns
    cat_cols <- names(data)[sapply(data, function(col) is.factor(col) || is.character(col))]
    num_cols <- setdiff(names(data), cat_cols)

    cat_levels <- list()  # For each categorical column, a vector with its different possible values
    cat_dims <- integer()  # For each categorical column, an integer indicating the number of possible values

    # Get levels
    for (i in seq_along(cat_cols)) {
      col <- cat_cols[i]

      lvls <- if (is.factor(data[[col]])) levels(data[[col]]) else unique(training_data[[col]])
      cat_levels[[col]] <- lvls
      cat_dims[[col]] <- length(lvls)
    }

    # Store metadata
    metadata <- list(
      cat_cols = cat_cols,
      num_cols = num_cols,
      cat_levels = cat_levels,
      cat_dims = cat_dims
    )
  }


  # Transform data using the metadata from the training set
  data_processed <- data

  for (col in metadata$cat_cols) {
    known_levels <- metadata$cat_levels[[col]]

    possible_integer <- as.integer(factor(data_processed[[col]], levels = known_levels))

    # Manage unknown categories
    possible_integer[is.na(possible_integer)] <- length(known_levels) + 1
    data_processed[[col]] <- possible_integer

    #print(data[[col]])
    #print(data_processed[[col]])
  }

  # Reorder the data so that numerical columns appear first
  data_processed <- dplyr::relocate(data_processed, dplyr::all_of(metadata$num_cols), dplyr::all_of(metadata$cat_cols))

  return(list(
    "data_processed" = data_processed,
    "metadata_for_cat" = metadata
  ))

  # print(cat_levels)
  # print(cat_dims)

  # Transform categorical values to integer --
  # data_processed <- data
  #
  # for (col in cat_cols) {
  #   data_processed[[col]] <- as.integer(factor(data_processed[[col]], levels = cat_levels[[col]]))
  #
  #   # print(data[[col]])
  #   # print(data_processed[[col]])
  # }
  #
  # # Order the data so that numerical columns appear first
  # data_processed <- dplyr::relocate(data_processed, dplyr::all_of(num_cols), dplyr::all_of(cat_cols))
  #
  # return (data_processed)




}


# TODO: dependiendo de cómo gestionar las categóricas, llamará a una u otra función. Ahora mismo lo hace como "none", siguiendo el paper.
#' Get bins to encode numerical features
#'
#' @param dataset_train Training partition of the dataset
#' @param T Number of target bins. Default: 10
#'
#' @returns A named list where each element contains the vector of bin cuts for a numerical feature
get_bins = function(dataset_train, metadata_for_cat = NULL, T = 10, handle_categorical = "None") {

  # TODO: Check que pasa cuando no hay variables categóricas. Da error o no????
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

    features[[j]] <- bin_categorical(dataset_train[, i], cat_dims[i], T)
  }

  print(features)

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
    return (list(type = "categorical", K = dimension, map = seq_len(dimension), n_bins = dimension, other_share = NULL))
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
  cuts <- stats::quantile(x, probs = seq(0, 1, length.out = T + 1), type = 7, names = FALSE)
  cuts <- unique(cuts)

  # Check unused bins
  raw <- findInterval(x = x, vec = cuts) + 1L  # Bin that contain and do not contain elements (repeated)
  used <- sort(unique(raw))  # Bins that contain elements

  # Map every raw bin to a consecutive index
  map <- vapply(seq_len(length(cuts) + 1L),
                function(r) which.min(abs(used - r)), integer(1))

  return (list(type = "numeric", cuts = cuts, map = map, n_bins = length(used), n_uniques = length(u)))
}


# Check encoding of a column
check_codes <- function(x, dims, j) {
  codes <- as.integer(round(x))



  if (any(codes < 0 | codes > dims - 1)) {
    print(codes)
    print(unique(codes))
    stop("Column ", j, ": categorical codes must be in 0..", dims - 1,
         " (found values outside that range).")
  }

  return (codes)

}


# TODO: dependiendo de cómo gestionar las categóricas, llamará a una u otra función. Ahora mismo lo hace como "none", siguiendo el paper.
#' Get bins to encode numerical features
#'
#' @param dataset_train Training partition of the dataset
#' @param T Number of target bins. Default: 10
#'
#' @returns A named list where each element contains the vector of bin cuts for a numerical feature
.remove_get_bins = function(dataset_train, T = 10, handle_categorical = "None") {

  if (!identical(handle_categorical, "top_frequent")) {
    stop("Only handle_categorical = 'top_frequent' is implemented")
  }

  if (!is.numeric(T) || T < 2) {
    stop("T must be a number >= 2")
  }

  dataset_train <- as.data.frame(dataset_train)

  # At this point, every feature is numerical. Categorical via encode
  cols <- names(dataset_train)


  bins <- lapply(cols, function(col) {
    x <- na.omit(dataset_train[[col]])  # Remove possible NaN
    n_unique <- length(unique(x))

    if (n_unique < T) {
      cuts <- sort(unique(x))
    } else {
      cuts <- quantile(x, probs = seq(0, 1, length.out = T + 1), type = 7, names = FALSE)
      cuts <- unique(cuts)
    }

    return (cuts)
  })

  names(bins) <- cols

  print(bins)

  return (bins)

}


apply_bins = function(dataset, cuts) {
  binned_dataset <- as.data.frame(dataset)

  for (col in names(cuts)) {
    x <- binned_dataset[[col]]
    cut_vec <- cuts[[col]]

    # Just one bin
    if (length(cut_vec) <= 1) {
      binned_dataset[[col]] <- 1
    # Multiple bins
    } else {
      binned_dataset[[col]] <- findInterval(
        x,
        vec = cut_vec,
        rightmost.closed = TRUE,
        all.inside = TRUE
      )
    }
  }

  return(binned_dataset)
}












