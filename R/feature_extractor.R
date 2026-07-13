#' Extract latent features from a given dataset using a pretrained SCARF model.
#'
#' @param dataframe A \code{data.frame} from which to extract features.
#' @param pretrained_model \code{String} or \code{list}. Path to the pretrained SCARF model (.pt file) if String or SCARF bundle if \code{list}.
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
#' scarf_fit(
#'   dataframe_train = df_train,
#'   exclude_columns = c("user_id", "cancellation"),
#'   n_epochs = 1,
#'   save_path = tmp_path
#' )
#'
#' # Extract features
#' extracted <- scarf_feature_extractor(
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
scarf_feature_extractor = function(
    dataframe,
    pretrained_model,
    exclude_columns = NULL,
    want_labels = FALSE,
    label_column = NULL,
    batch_size = 32,
    preprocess = TRUE
  ) {

  # Extract pretrained model and recipe
  bundle <- load_scarf_bundle(pretrained_model)


  fitted_encoder <- bundle$encoder
  trained_recipe <- bundle$recipe


  # Prepare data
  if (want_labels & is.null(label_column)) {
    stop("scarf_feature_extractor: if 'want_labels' is TRUE, then you have to specify the label column with the parameter 'label_column'")
  }


  dataframe_cleaned_xy <- prepare_scarf_data_for_feature_extraction(dataframe, trained_recipe, exclude_columns, want_labels = want_labels, label_column = label_column, preprocess = preprocess)
  dataframe_cleaned <- dataframe_cleaned_xy$x
  dataframe_labels <- dataframe_cleaned_xy$y


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
        batch_latent <- fitted_encoder(x_batch)

        # Store representations
        features_list[[length(features_list) + 1]] <- batch_latent$cpu()

      }
    )
  })

  # Concatenate representations
  all_features <- torch::torch_cat(features_list, dim=1)

  # Convert to matrix
  all_features <- as.matrix(all_features)

  print("Extracted features: ")
  print(dim(all_features))
  print(length(dataframe_labels))

  return(list(
    features = all_features,
    features_labels = dataframe_labels
    ))


}







