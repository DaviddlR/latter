

#' Title
#'
#' @param path_train_data tdo
#' @param path_test_data tdo
#'
#' @returns tdo
#'
#' @export
#'
#' @examples
#' a <- 1
scarf_pretraining = function(path_train_data, path_test_data) {

  # Load data
  preprocessed_datasets = read_parquet_data(path_train_data, path_test_data)

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set
  x_test <- preprocessed_datasets$test_set

  y_train <- preprocessed_datasets$train_label
  y_val <- preprocessed_datasets$val_label
  y_test <- preprocessed_datasets$test_label

  # print(dim(x_train))
  # print(dim(x_val))
  # print(dim(x_test))
  # print(dim(y_train))
  # print(dim(y_val))
  # print(dim(y_test))

  # Create tensor datasets
  train_ds <- create_tensor_dataset(x_train, y_train)
  val_ds <- create_tensor_dataset(x_val, y_val)
  test_ds <- create_tensor_dataset(x_test, y_test)


  batch_size <- 256

  # Create dataloader
  train_dl <- dataloader(train_ds,
                         batch_size = batch_size,
                         shuffle = TRUE)

  val_dl <- dataloader(val_ds,
                       batch_size = batch_size,
                       shuffle=FALSE)

  test_dl <- dataloader(test_ds,
                        batch_size = batch_size,
                        shuffle = FALSE)


  # Create model, pretraining head and luz flow

  fitted <- SCARF_wrapper |>
    setup(
      loss = nt_xent_loss(temperature = 0.5),
      optimizer = torch::optim_adam
    ) |>
    set_hparams(
      in_dim = dim(x_train)[2],
      hidden_dim = 256,
      num_hidden = 4,
      head_hidden_dim = 256,
      head_num_hidden = 2,
      dropout = 0.0,
    ) |>
    set_opt_hparams(
      lr = 0.0001,
    ) |>
    fit(
      train_dl,
      epochs = 1,
      valid_data = val_dl,
      callbacks = list(custom_scarf_step_callback(corruption_rate = 0.6))
    )


  luz::luz_save(fitted, "scarf_trained.rds")







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
    target <- batch$y  # Label. Not used during pre-training

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

    ctx$batch[[2]] <- batch$y
    ctx$batch[[1]] <- c(x, x_corrupted)



  },


  # Validation
  on_valid_batch_begin = function() {
    batch = ctx$batch
    target = batch$y  # Label. Not used during pre-training

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

    ctx$target <- batch$y
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











