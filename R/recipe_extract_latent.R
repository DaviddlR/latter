
# @details
# The algorithms that can be used for feature extraction require each predictor
#  to be numerical. For that reason, if your dataset
# contains categorical data, before using this step, you must:
# \itemize{
#   \item Impute missing values (i.e. \code{step_impute_median()})
#   \item Convert categorical data to numerical data (i.e.
#   \code{step_dummy(..., one_hot = TRUE)})
#   \item Normalize numerical data (i.e. \code{step_nomalize()})
# }
#    DEPRECATED. NOW USERS DO NOT NEED TO REMOVE CATEGORICAL COLUMNS





#' Extract latent features using Self-Supervised Learning methods
#'
#'
#'
#' @param recipe A recipe object. The step will be added to the sequence of
#' operations for this recipe
#' @param ... One or more selector expressions to choose which variables will be
#'  used to train the SSL model and extract features
#' @param role A character string specifying the role of the newly generated
#' variables. Defaults to \code{"predictor"}.
#' @param trained A logical to indicate if the quantities for preprocessing have
#'   been estimated.
#' @param pretraining_type A character specifying the SSL method to use.
#' Default is \code{'SCARF'}. Available options are \code{["SCARF", "VIME"]}
#' @param create_validation A \code{logical}. If \code{TRUE}, splits the
#' training data to create a validation set. Default is \code{FALSE}.
#' @param validation_proportion \code{Numeric}. Proportion of data (0 to 1)
#' allocated for validation if \code{create_validation = TRUE}. Default is
#' \code{0.1}.
#' @param batch_size An \code{integer} defining the number of samples per batch
#' during training. Default is \code{256}.
#' @param epochs An \code{integer} defining the number of training epochs.
#' Default is \code{150}.
#' @param batch_size_inference An integer specifying the batch size during the
#' inference or transformation phase (\code{bake}). Default is \code{32}.
#' @param pretrained_model The pretrained model object once the step has been
#' executed by \code{prep()}.
#' @param columns Columns to be processed. Should not be modified. Default is \code{NULL}.
#' @param skip A logical. Should the step be skipped when the recipe is baked.
#' by \code{bake()}? Defaults to \code{FALSE} so that the transformation is
#' applied to both training and test sets.
#' @param id A character string that is unique to this step to identify it.
#'
#' @returns An updated recipe object with the class \code{step_extract_latent}
#' added to the sequence of operations.
#' @export
#'
#' @examples
#' a <- 1
step_extract_latent <- function(
  recipe,
  ...,
  role = "predictor",
  trained = FALSE,
  pretraining_type = "SCARF",  # De aquí hacia abajo, los parámetros que necesito yo
  # exclude_columns = NULL,
  create_validation = FALSE,
  validation_proportion = 0.1,
  batch_size = 256,
  epochs = 150,
  # want_labels = FALSE,
  # label_column = NULL,
  batch_size_inference = 32,
  pretrained_model = NULL,  # Pretrained model to extract features after prep
  columns = NULL,  # Columns to be processed
  skip = FALSE,  # Skip and ID last arguments
  id = recipes::rand_id("extract_latent")

  ) {
    # Add step
    recipes::add_step(
      recipe,
      step_extract_latent_new(
        subclass = "extract_latent",
        terms = rlang::enquos(...),
        role = role,
        trained = trained,
        pretraining_type = pretraining_type,
        create_validation = create_validation,
        validation_proportion = validation_proportion,
        batch_size = batch_size,
        epochs = epochs,
        batch_size_inference = batch_size_inference,
        pretrained_model = pretrained_model,
        columns = NULL,
        skip = skip,
        id = id
      )
    )
  }





# Constructor
step_extract_latent_new <- function(subclass,
                                    terms,
                                    role,
                                    trained,
                                    pretraining_type,
                                    create_validation,
                                    validation_proportion,
                                    batch_size,
                                    epochs,
                                    batch_size_inference,
                                    pretrained_model,
                                    columns,
                                    skip,
                                    id) {
  recipes::step(
    subclass = subclass,
    terms = terms,
    role = role,
    trained = trained,
    pretraining_type = pretraining_type,
    create_validation = create_validation,
    validation_proportion = validation_proportion,
    batch_size = batch_size,
    epochs = epochs,
    batch_size_inference = batch_size_inference,
    pretrained_model = pretrained_model,
    columns = columns,
    skip = skip,
    id = id
  )
}









#' @importFrom recipes prep
#' @export
prep.step_extract_latent <- function(x, training, info = NULL, ...) {

  col_names <- recipes::recipes_eval_select(x$terms, training, info)  # Select columns that the user listed
  training_data <- as.data.frame(training[, col_names])

  # Prep logic. Adjust to new data (training)

  # We have SCARF_fit, so we can reuse this method
  pretrained_model <- fit_extractor(
    dataframe_train = training_data,
    pretraining_type = x$pretraining_type,
    exclude_columns = NULL,  # Force NULL. The user filter the columns with recipes
    create_validation = x$create_validation,
    validation_proportion = x$validation_proportion,
    batch_size = x$batch_size,
    n_epochs = x$epochs,
    save_path = NULL,  # Force NULL so that the model is stored in RAM, ready for bake and avoiding the need to store it locally.
    preprocess = FALSE  # FALSE so that it does not apply other preprocessing recipes. Assume that the user does it.
  )


  # Use the constructor function to return the updated object.
  step_extract_latent_new(
    subclass = "extract_latent",
    terms = x$terms,
    role = x$role,
    trained = TRUE,  # As prep is completed, we set trained to TRUE
    pretraining_type = x$pretraining_type,
    create_validation = x$create_validation,
    validation_proportion = x$validation_proportion,
    batch_size = x$batch_size,
    epochs = x$epochs,
    batch_size_inference = x$batch_size_inference,
    pretrained_model = pretrained_model,  # Store the pretrained model
    columns = col_names,
    skip = x$skip,
    id = x$id
  )

}





#' @importFrom recipes bake
#' @export
bake.step_extract_latent <- function(object, new_data, ...) {

  col_names <- object$columns  # Select columns that the user listed
  data_to_extract <- as.data.frame(new_data[, col_names], drop = FALSE)

  # Bake logic. Apply the pretrained model to new data
  extracted_data <- extract_features(
    dataframe = data_to_extract,
    pretrained_model = object$pretrained_model,
    exclude_columns = NULL,
    want_labels = FALSE,
    label_column = NULL,
    batch_size = object$batch_size_inference,
    preprocess = FALSE
  )


  remaining_data <- new_data[, !(names(new_data) %in% col_names), drop = FALSE]  # Not processed data

  features <- extracted_data$features
  colnames(features) <- paste0("extracted_dim_", 1:ncol(features))
  features_tibble <- tibble::as_tibble(features)

  return(tibble::as_tibble(cbind(remaining_data, features_tibble)))



}






# Three additional methods: print, tidy y require_pkgs


# Custom print method for step_extract_latent
#' @importFrom recipes print_step
#' @export
print.step_extract_latent <- function(x, width = max(20, options()$width - 30), ...) {

  title <- paste0("Latent feature extraction with ", x$pretraining_type)

  recipes::print_step(
    tr_obj = x$columns,  # Names after prep:
    untr_obj = x$terms,  # Names before prep
    trained = x$trained,  # Has it been prepped?
    title = title,  # What does this step do?
    width = width  # An estimate of how many characters to print on a line:
  )

  invisible(x)

}


# Tidy. Which columns have been or will be affected
#' @importFrom generics tidy
#' @export
tidy.step_extract_latent <- function(x, ...) {
  if (x$trained) {
    res <- tibble::tibble(
      terms = x$columns,
    )
  } else {
    res <- tibble::tibble(
      terms = recipes::sel2char(x$terms)
    )
  }

  res$id <- x$id
  res
}



# TODO
# Required packages
#' @importFrom recipes required_pkgs
#' @export
required_pkgs.step_extract_latent <- function(x, ...) {
  c("latter", "torch")
}




