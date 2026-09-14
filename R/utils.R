

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





#' Get bins to encode numerical features
#'
#' @param dataset_train Training partition of the dataset
#' @param T Number of target bins. Default: 10
#'
#' @returns A named list where each element contains the vector of bin cuts for a numerical feature
get_bins = function(dataset_train, T = 10) {

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


apply_bins = function(dataset, bins) {

}












