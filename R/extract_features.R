#' Extract latent features from a given dataset using a pretrained SCARF model.
#'
#' @param dataframe A \code{data.frame} from which to extract features.
#' @param pretrained_model \code{String} or \code{list}. Path to the pretrained model (.pt file) if String or a bundle if \code{list}.
#' @param exclude_columns A \code{string} of columns that the model should ignore during inference (i.e target or ID columns). Default is \code{NULL}.
#' @param want_labels \code{Boolean}. If \code{TRUE}, the function extracts and returns the target labels alongside features. Default is \code{FALSE}.
#' @param label_column \code{String}. Name of the column containing the labels. Required if \code{want_labels = TRUE}. Default is \code{NULL}.
#' @param batch_size batch_size \code{integer}. Number of samples per batch during feature extraction. Default is \code{32}.
#' @param preprocess \code{Boolean} Set if the data need preprocessing steps using 'recipes', such as 'step_normalize' or 'step_dummy'. Default is \code{TRUE}, meaning that this process is automatically done.
#'
#' @returns A \code{list} containing two elements:
#' \itemize{
#'   \item \code{features}: A numeric \code{matrix} of the extracted latent representations.
#'   \item \code{features_labels}: A vector with the labels of each sample (or \code{NULL} if \code{want_labels = FALSE}).
#' }
#'
#' @export
#'
#' @examples
#'
#' \donttest{
#' # Create dummy dataset
#' df_train <- data.frame(
#'   user_id = 1:120,
#'   age = rnorm(120, mean = 35, sd = 10),
#'   income = runif(120, 15000, 75000),
#'   risk_profile = factor(sample(c("Low", "Medium", "High"), 120, replace = TRUE)),
#'   cancellation = sample(0:1, 120, replace = TRUE)
#' )
#'
#' tmp_path <- tempfile(fileext = ".pt")
#'
#' # Fit SCARF one epoch
#' fit_extractor(
#'   dataframe_train = df_train,
#'   pretraining_type = "SCARF",
#'   exclude_columns = c("user_id", "cancellation"),
#'   n_epochs = 1,
#'   save_path = tmp_path
#' )
#'
#'
#'
#' # Extract features
#' extracted <- extract_features(
#'   dataframe = df_train,
#'   pretrained_model = tmp_path,
#'   exclude_columns = c("user_id", "cancellation"),
#'   want_labels = TRUE,
#'   label_column = "cancellation"
#' )
#'
#' print(dim(extracted$features))
#'
#' # Remove temp file
#' if (file.exists(tmp_path)) file.remove(tmp_path)
#'
#' }
extract_features = function(
  dataframe,
  pretrained_model,
  exclude_columns = NULL,
  want_labels = FALSE,
  label_column = NULL,
  batch_size = 32,
  preprocess = TRUE
  ) {


  # Load bundle (inside, it validates if it comes from a valid pretraining)
  bundle <- load_bundle(bundle_path = pretrained_model)

  # Extract encoder and recipe (if exists)
  fitted_encoder <- bundle$encoder
  trained_recipe <- bundle$recipe
  metadata_for_categorical <- bundle$metadata_for_cat


  # Prepare data
  if (want_labels & is.null(label_column)) {
    stop("extract_features: if 'want_labels' is TRUE, then you have to specify the label column with the parameter 'label_column'")
  }


  dataframe_cleaned_xy <- prepare_data_for_feature_extraction(dataframe, trained_recipe, categorical_metadata = metadata_for_categorical, exclude_columns = exclude_columns, want_labels = want_labels, label_column = label_column, preprocess = preprocess)
  dataframe_cleaned <- dataframe_cleaned_xy$x
  dataframe_labels <- dataframe_cleaned_xy$y

  # Subtab crops the data into views, which are defined by the position of the columns. So the columns must be exactly the ones
  # used in pretraining, in the same order
  is_subtab <- identical(bundle$bundle_type, "subtab_bundle")

  if (is_subtab && !identical(colnames(dataframe_cleaned), bundle$column_names)) {
    missing_columns <- setdiff(bundle$column_names, colnames(dataframe_cleaned))
    extra_columns <- setdiff(colnames(dataframe_cleaned), bundle$column_names)

    stop("extract_features: the columns of the data are not the ones used to train this SubTab model. ",
         if (length(missing_columns) > 0) paste0("Missing: ", paste(missing_columns, collapse = ", "), ". ") else "",
         if (length(extra_columns) > 0) paste0("Not expected: ", paste(extra_columns, collapse = ", "), ". ") else "",
         "If the categorical variables are one-hot encoded, check that the dummies are the same as in training.")
  }

  dataset_ready <- create_tensor_dataset(dataframe_cleaned)

  dataloader_ready <- torch::dataloader(dataset_ready,
                                        batch_size = batch_size,
                                        shuffle = FALSE)

  # Prepare model
  device <- if(torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
  message("Inference in ", if (torch::cuda_is_available()) "GPU (CUDA)" else "CPU")

  fitted_encoder$to(device = device)
  fitted_encoder$eval()

  # Extract latent features using the trained model
  features_list <- list()

  torch::with_no_grad({
    coro::loop(
      for(batch in dataloader_ready) {

        # Take batch
        x_batch <- batch[[1]]$to(device = device)

        # Forward pass

        # For subtab, first crop the views, encode each one and aggregate them.
        # Others, the encoder takes the whole view

        if (is_subtab) {
          batch_latent <- subtab_extract_latent(
            encoder = fitted_encoder,
            x_batch = x_batch,
            view_indices = bundle$view_indices,
            aggregation = bundle$aggregation
          )
        } else {
          batch_latent <- fitted_encoder(x_batch)
        }



        # Store representations
        features_list[[length(features_list) + 1]] <- batch_latent$cpu()

      }
    )
  })

  # Concatenate representations
  all_features <- torch::torch_cat(features_list, dim=1)

  # Convert to matrix
  all_features <- as.matrix(all_features)

  # print("Extracted features: ")
  # print(dim(all_features))
  # print(length(dataframe_labels))

  return(list(
    features = all_features,
    features_labels = dataframe_labels
  ))

}





#' Latent representation of a batch with Subtab
#'
#' The batch is cropped into the views stored at pretraining, every view goes through the (shared) encoder and the representations of the views of each sample are combined.
#'
#' @param encoder A trained \code{subtab_encoder} (in eval mode).
#' @param x_batch A \code{torch_tensor} with the preprocessed batch. The columns must be the ones used in pretraining, in the same order.
#' @param view_indices Indexes of the columns of each view, [n_views, view_width] (stored in the bundle).
#' @param aggregation \code{Character}. How the views are combined. Only \code{"mean"} is available.
#'
#' @returns A \code{torch_tensor} [B, latent size].
#' @noRd
subtab_extract_latent <- function(encoder, x_batch, view_indices, aggregation = "mean") {

  if (!identical(aggregation, "mean")) {
    stop("subtab_extract_latent: only aggregation = 'mean' is available, but the bundle has '", aggregation, "'")
  }

  n_views <- nrow(view_indices)
  batch_size <- x_batch$size(1)

  views <- subtab_crop_views(x_batch, view_indices)  # [n_views * B, view_width]
  z <- encoder(views)                                # [n_views * B, latent size]

  # [n_views * B, h] -> [n_views, B, h] -> mean over the views -> [B, h]
  torch::torch_mean(z$reshape(c(n_views, batch_size, -1)), dim = 1)
}








