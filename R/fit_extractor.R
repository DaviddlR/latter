#' Trains a SCARF encoder using a contrastive loss objective. It prepares the data using a recipe, applies random feature corruption and fits the model.
#'
#' @param dataframe_train A \code{data.frame} used to train de model.
#' @param pretraining_type A \code{character} indicating the pretraining objective. Default is \code{"SCARF"}, so that it follows SCARF pretraining. Available options are \code{["SCARF"]}.
#' @param exclude_columns A \code{character} of columns that the model should ignore during pretraining (i.e target or ID columns). Default is \code{NULL}.
#' @param create_validation \code{Boolean}. If \code{TRUE}, splits the training data to create a validation set. Default is \code{FALSE}.
#' @param validation_proportion \code{Numeric}. Proportion of data (0 to 1) allocated for validation if \code{create_validation = TRUE}. Default is \code{0.1}.
#' @param batch_size \code{Integer}. Number of samples per batch during training. Default is \code{256}.
#' @param n_epochs \code{Integer}. Number of training epochs. Default is \code{150}.
#' @param save_path \code{String}. Path where the pretrained bundle (.pt) will be saved. Extension ('.pt') should not be included. Default is \code{"SCARF"}, which saves a 'SCARF.pt' file in the current directory.
#' @param preprocess \code{Boolean}. Set if the data need preprocessing steps using 'recipes', such as 'step_normalize' or 'step_dummy'. Default is \code{TRUE}, meaning that this process is automatically done.
#'
#'
#' @returns Invisible \code{NULL}. The function saves a serialized list containing the encoder state dict, hyperparameters, and the preprocessing recipe to \code{save_path}.
#' @export
#'
#' @examples
#' \donttest{
#' if (torch::torch_is_installed()) {
#'
#'   # Create dummy dataset
#'   df_train <- data.frame(
#'     user_id = 1:120,
#'     age = rnorm(120, mean = 35, sd = 10),
#'     income = runif(120, 15000, 75000),
#'     risk_profile = factor(sample(c("Low", "Medium", "High"),
#'       120,
#'       replace = TRUE)),
#'     label = sample(0:1, 120, replace = TRUE)
#'   )
#'
#'   tmp_path <- tempfile(fileext = ".pt")
#'
#'   # Fit SCARF one epoch
#'   fit_extractor(
#'     dataframe_train = df_train,
#'     pretraining_type = "SCARF",
#'     exclude_columns = c("user_id", "label"),
#'     n_epochs = 1,
#'     save_path = tmp_path,
#'     preprocess = TRUE
#'   )
#'
#'
#'   # Remove temp file
#'   if (file.exists(tmp_path)) file.remove(tmp_path)
#' }
#' }
#'
fit_extractor <- function (
    dataframe_train,
    pretraining_type = "SCARF",
    exclude_columns = NULL,
    create_validation = FALSE,
    validation_proportion = 0.1,
    batch_size = 256,
    n_epochs = 1,
    save_path = "pretrained",
    preprocess = TRUE
) {


  # Go to pretraining_objectives.R
  if (identical(pretraining_type, "SCARF")) {
    # SCARF pretraining
    scarf_bundle <- scarf_fit(
      dataframe_train = dataframe_train,
      exclude_columns = exclude_columns,
      create_validation = create_validation,
      validation_proportion = validation_proportion,
      batch_size = batch_size,
      n_epochs = n_epochs,
      preprocess = preprocess
    )
  } else if (identical(pretraining_type, "VIME")) {

    print("VIME TRAINING :D")
    # VIME pretraining
    vime_bundle <- vime_fit(
      dataframe_train = dataframe_train,
      exclude_columns = exclude_columns,
      create_validation = create_validation,
      validation_proportion = validation_proportion,
      batch_size = batch_size,
      n_epochs = n_epochs,
      preprocess = preprocess
    )


  } else {
    stop("The selected 'pretraining_type' is not supported. Please select one of the available options: 'SCARF'")
  }

  # Store the pretrained model locally or in RAM memory

  # If save_path is not null, save model locally (it will be NULL when using it as a recipe, when stored in RAM)
  if (!is.null(save_path)){
    torch::torch_save(scarf_bundle, path = paste0(save_path, ".pt"))
    message("Pretrained model saved in ", save_path, ".pt")
  }

  # Return invisible for the recipe prep and bake
  return(invisible(scarf_bundle))


}

