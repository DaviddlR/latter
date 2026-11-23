

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

    return (list(
      encoder = fitted_encoder,
      recipe = trained_recipe,
      metadata_for_cat = info_for_categorical
    ))



  # BINNING bundle
  } else if (is.list(model_bundle) && identical(model_bundle$bundle_type, "binning_bundle")) {

    print("Extracting a Binning bundle")

    # Load encoder hyperparameters
    hparams <- model_bundle$encoder_hparams

    print(hparams)

    # Create a new encoder
    fitted_encoder <- binning_encoder(
      num_cont = hparams$num_cont,
      cat_dims = hparams$cat_dims,
      hidden_dim = hparams$hidden_dim,
      num_hidden = hparams$num_hidden
    )

    # Load trained weights
    fitted_encoder$load_state_dict(model_bundle$encoder_state_dict)

    # Load recipe
    trained_recipe <- unserialize(model_bundle$recipe)

    # Load info for categorical data
    info_for_categorical <- model_bundle$metadata_for_cat

    return (list(
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



# Corrupt a batch by replacing, for each element, N = number_of_rows * corruption_rate columns with values given by other in-batch instances
corrupt_batch = function(x, corruption_rate) {
  batch_size <- x$size(1)
  num_features <- x$size(2)

  mask <- torch::torch_rand_like(x) < corruption_rate

  random_indices <- torch::torch_randint(
    low = 1,
    high = batch_size + 1,  # 1 and +1 because R indices start at 1
    size = c(batch_size, num_features),
    device = x$device,
    dtype = torch::torch_long()
  )

  x_random <- torch::torch_gather(x, dim=1, index = random_indices)

  x_corrupted <- torch::torch_where(mask, x_random, x)

  return(list(
    x_corrupted = x_corrupted,
    mask = mask
  ))
}





#' Detect the binary columns of the training data
#'
#' A column is binary if it has at most two different values. It does not matter which values they are, so a one-hot column
#' is detected even if it was standardized afterwards (it still has two values). Constant columns count as binary.
#'
#' @param x A matrix (or data frame) of numerical columns.
#'
#' @returns A \code{logical} vector with one element per column. \code{TRUE} = binary.
#' @noRd
subtab_detect_binary_columns <- function(x) {

  x <- as.matrix(x)

  vapply(seq_len(ncol(x)), function(j) {
    values <- x[, j]
    length(unique(values[!is.na(values)])) <= 2
  }, logical(1))
}


#' Which columns of each view can receive noise
#'
#' @param view_indices Output of \code{subtab_view_indices()}, [n_views, view_width].
#' @param binary_columns Output of \code{subtab_detect_binary_columns()}, one element per column of the data.
#' @param noise_on_binary \code{Boolean}. If \code{TRUE}, every column can be corrupted (official behaviour) and the result is \code{NULL}.
#'   If \code{FALSE}, binary columns are excluded.
#' @param device Device of the tensor.
#'
#' @returns \code{NULL} (everything can be corrupted) or a boolean \code{torch_tensor} [n_views, view_width]. \code{TRUE} = can be corrupted.
#' @noRd
subtab_noise_eligibility <- function(view_indices, binary_columns, noise_on_binary = FALSE, device = "cpu") {

  if (noise_on_binary) return(NULL)

  eligible <- matrix(!binary_columns[view_indices], nrow = nrow(view_indices))

  torch::torch_tensor(eligible, dtype = torch::torch_bool(), device = device)
}




#' Add noise to the views
#'
#' @param x_views A \code{torch_tensor} [n_views * B, view_width] with the cropped views (output of \code{subtab_crop_views()}).
#' @param n_views Number of views.
#' @param noise_type \code{Character}. Available options are \code{["swap_noise", "gaussian_noise", "zero_out"]}.
#' @param masking_ratio \code{Numeric} in [0, 1]. Probability of corrupting each element.
#' @param noise_level \code{Numeric}. Standard deviation of the Gaussian noise. Only for \code{"gaussian_noise"}.
#' @param eligible \code{NULL} or a boolean \code{torch_tensor} [n_views, view_width] (output of \code{subtab_noise_eligibility()}).
#'   Elements that are not eligible are never corrupted.
#'
#' @returns A \code{torch_tensor} [n_views * B, view_width]. The input is not modified.
#' @noRd
subtab_add_noise <- function(x_views, n_views, noise_type = c("swap_noise", "gaussian_noise", "zero_out"),
                             masking_ratio = 0.3, noise_level = 0.1, eligible = NULL) {

  noise_type <- match.arg(noise_type)

  # Validation
  if (!is.numeric(masking_ratio) || length(masking_ratio) != 1 || masking_ratio < 0 || masking_ratio > 1) {
    stop("subtab_add_noise: 'masking_ratio' must be in [0, 1]")
  }
  if (!is.numeric(noise_level) || length(noise_level) != 1 || noise_level < 0) {
    stop("subtab_add_noise: 'noise_level' must be >= 0")
  }

  batch_size <- x_views$size(1) %/% n_views
  view_width <- x_views$size(2)

  # [n_views * B, w] -> [n_views, B, w]. It works because the rows are ordered view by view
  x <- x_views$reshape(c(n_views, batch_size, view_width))

  # Value that would replace each element
  replacement <- switch(
    noise_type,

    swap_noise = {
      permutation <- torch::torch_argsort(torch::torch_rand(n_views, batch_size, view_width, device = x$device), dim = 2)
      torch::torch_gather(x, dim = 2, index = permutation)  # out[k, b, j] = x[k, permutation[k, b, j], j]
    },

    gaussian_noise = x + torch::torch_randn_like(x) * noise_level,

    zero_out = torch::torch_zeros_like(x)
  )

  # Official: mask = binomial(1, p_m, x_bar.shape). One independent Bernoulli for every element
  corrupt <- torch::torch_rand_like(x) < masking_ratio

  # Columns that must not be corrupted. eligible [n_views, w] -> [n_views, 1, w] to broadcast over the rows
  if (!is.null(eligible)) corrupt <- corrupt & eligible$unsqueeze(2)

  noisy <- torch::torch_where(corrupt, replacement, x)

  noisy$reshape(c(n_views * batch_size, view_width))
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

      lvls <- if (is.factor(data[[col]])) levels(data[[col]]) else unique(data[[col]])
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




















