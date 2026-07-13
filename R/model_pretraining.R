



#' Trains a SCARF encoder using a contrastive loss objective. It prepares the data using a recipe, applies random feature corruption and fits the model.
#'
#' @param dataframe_train A \code{data.frame} used to train de model.
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
#'   scarf_fit(
#'     dataframe_train = df_train,
#'     exclude_columns = c("user_id", "label"),
#'     n_epochs = 1,
#'     save_path = tmp_path,
#'     preprocess = TRUE
#'   )
#'
#'   # Remove temp file
#'   if (file.exists(tmp_path)) file.remove(tmp_path)
#' }
#' }
#'
scarf_fit = function(
  dataframe_train,
  exclude_columns = NULL,
  create_validation = FALSE,
  validation_proportion = 0.1,
  batch_size = 256,
  n_epochs = 1,
  save_path = "SCARF",
  preprocess = TRUE
) {


  # Load and preprocess data
  preprocessed_datasets <- prepare_scarf_data(dataframe_train, exclude_columns = exclude_columns, create_validation = create_validation, validation_proportion = validation_proportion, preprocess = preprocess)

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set  # May be null
  recipe <- preprocessed_datasets$recipe  # May be null

  # Create training dataset and dataloader
  train_ds <- create_tensor_dataset(x_train)

  train_dl <- torch::dataloader(train_ds,
                         batch_size = batch_size,
                         shuffle = TRUE)

  # Create training dataset and dataloader (if required)
  val_dl <- NULL

  if(create_validation) {
    val_ds <- create_tensor_dataset(x_val)

    val_dl <- torch::dataloader(val_ds,
                         batch_size = batch_size,
                         shuffle=FALSE)
  }

  fitted <- SCARF_wrapper |>
    luz::setup(
      loss = nt_xent_loss(temperature = 0.5),
      optimizer = torch::optim_adam
    ) |>
    luz::set_hparams(
      in_dim = dim(x_train)[2],
      hidden_dim = 256,
      num_hidden = 4,
      head_hidden_dim = 256,
      head_num_hidden = 2,
      dropout = 0.0
    ) |>
    luz::set_opt_hparams(
      lr = 0.0001
    ) |>
    luz::fit(
      train_dl,
      epochs = n_epochs,
      valid_data = val_dl,
      callbacks = list(custom_scarf_step_callback(corruption_rate = 0.6))
    )


  # Save trained model AND the recipe required to apply the same preprocessing to the test set

  encoder_weights <- fitted$model$main_encoder$state_dict()

  hparams <- list(
    in_dim = dim(x_train)[2],
    hidden_dim = 256,
    num_hidden = 4,
    dropout = 0.0
  )

  model_bundle <- list(
    encoder_state_dict = encoder_weights,
    encoder_hparams = hparams,
    recipe = serialize(recipe, NULL),
    bundle_type = "scarf_bundle"
  )


  # If save_path is not null, save model locally (it will be NULL when using it as a recipe, when stored in RAM)
  if (!is.null(save_path)){
    torch::torch_save(model_bundle, path = paste0(save_path, ".pt"))
    message("Pretrained model saved in ", save_path, ".pt")
  }

  # Return invisible for the recipe prep and bake
  return(invisible(model_bundle))


}




custom_scarf_step_callback <- luz::luz_callback(

  name = "SCARF_custom_steps",

  initialize = function(corruption_rate = 0.6) {
    self$corruption_rate = corruption_rate
  },



  # Train. It receives a batch from the dataloader / tensor_dataset
  on_train_batch_begin = function() {

    #print(ctx$batch[[1]]$device)

    batch <- ctx$batch
    #target <- batch$y  # Label. Not used during pre-training

    x <- batch$x  # Data

    batch_size <- x$size(1)
    num_features <- x$size(2)

    mask <- torch::torch_rand_like(x) < self$corruption_rate

    random_indices <- torch::torch_randint(
      low = 1,
      high = batch_size + 1,  # 1 and +1 because R indices start at 1
      size = c(batch_size, num_features),
      device = x$device,
      dtype = torch::torch_long()
    )

    x_random <- torch::torch_gather(x, dim=1, index = random_indices)
    x_corrupted <- torch::torch_where(mask, x_random, x)

    #ctx$batch[[2]] <- batch$y
    ctx$batch[[1]] <- c(x, x_corrupted)



  },


  # Validation
  on_valid_batch_begin = function() {
    batch = ctx$batch
    #target = batch$y  # Label. Not used during pre-training

    x = batch$x  # Data

    batch_size = x$size(1)
    num_features = x$size(2)

    mask = torch::torch_rand_like(x) < self$corruption_rate

    random_indices <- torch::torch_randint(
      low = 1,
      high = batch_size + 1,  # 1 and +1 because R indices start at 1
      size = c(batch_size, num_features),
      device = x$device,
      dtype = torch::torch_long()
    )

    x_random <- torch::torch_gather(x, dim=1, index = random_indices)
    x_corrupted <- torch::torch_where(mask, x_random, x)

    #ctx$target <- batch$y
    ctx$input <- c(x, x_corrupted)
  },


  # Test / predict

)


# preprocessed_datasets = read_parquet_data("inst/extdata/UNSW_NB15_training-set.parquet", "inst/extdata/UNSW_NB15_testing-set.parquet")
#
# x_train <- preprocessed_datasets$train_set
# x_val <- preprocessed_datasets$val_set
# x_test <- preprocessed_datasets$test_set
#
# ejemplo_x <- x_train[1, , drop = FALSE]
#
#
# y_train <- preprocessed_datasets$train_label
# y_val <- preprocessed_datasets$val_label
# y_test <- preprocessed_datasets$test_label
#
# print(dim(x_train))
# print(dim(x_val))
# print(dim(x_test))
# print(dim(y_train))
# print(dim(y_val))
# print(dim(y_test))
#
# # Create tensor datasets
# train_ds <- create_tensor_dataset(x_train, y_train)
# val_ds <- create_tensor_dataset(x_val, y_val)
# test_ds <- create_tensor_dataset(x_test, y_test)
#
# print(train_ds[1]$x)
#
# # Create dataloader
# train_dl <- dataloader(train_ds,
#                        batch_size = 32,
#                        shuffle = TRUE)
#
# val_dl <- dataloader(val_ds,
#                      batch_size = 32,
#                      shuffle=FALSE)
#
# test_dl <- dataloader(test_ds,
#                       batch_size = 32,
#                       shuffle = FALSE)
#
#
#
# scarf_pretraining("inst/extdata/UNSW_NB15_training-set.parquet", "inst/extdata/UNSW_NB15_testing-set.parquet")











