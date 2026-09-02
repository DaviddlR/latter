
############ VIME PRETRAINING ############
vime_fit = function(
    dataframe_train,
    exclude_columns = NULL,
    create_validation = FALSE,
    validation_proportion = 0.1,
    batch_size = 256,
    n_epochs = 1,
    preprocess = TRUE
) {

  # Load and preprocess data

  # Create training dataset and dataloader

  # Create validation dataset and dataloader (if required)


  # Create luz wrapper


  # Save trained model AND the recipe required to apply the same preprocessing to the test set

}



custom_vime_step_callback <- luz::luz_callback(

  name = "VIME_custom_steps",

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


    # Binary mask to float (0.0 , 1.0)
    mask_target <- mask$to(dtype = torch::torch_float())


    ctx$input <- x_corrupted  # Input for the forward method and "input" variable in the loss function
    ctx$target <- c(mask_target, x)  # Input for the "target" variable in the loss function. Not needed during forward




  },


  # Validation
  on_valid_batch_begin = function() {
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


    # Binary mask to float (0.0 , 1.0)
    mask_target <- mask$to(dtype = torch::torch_float())


    ctx$input <- x_corrupted  # Input for the forward method and "input" variable in the loss function
    ctx$target <- c(mask_target, x)  # The "target" variable in the loss function. Not needed during forward

  },


  # Test / predict

)






############ SCARF PRETRAINING ############

# Train a neural network with the SCARF pretraining objective
scarf_fit = function(
  dataframe_train,
  exclude_columns = NULL,
  create_validation = FALSE,
  validation_proportion = 0.1,
  batch_size = 256,
  n_epochs = 1,
  preprocess = TRUE
) {


  # Load and preprocess data
  preprocessed_datasets <- prepare_scarf_data(dataframe_train, exclude_columns = exclude_columns, create_validation = create_validation, validation_proportion = validation_proportion, preprocess = preprocess)

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set  # May be null
  recipe <- preprocessed_datasets$recipe  # May be null
  metadata_for_cat <- preprocessed_datasets$metadata_for_cat

  # Create training dataset and dataloader
  train_ds <- create_tensor_dataset(x_train)

  train_dl <- torch::dataloader(train_ds,
                         batch_size = batch_size,
                         shuffle = TRUE)

  # Create validation dataset and dataloader (if required)
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
      num_cont = length(metadata_for_cat$num_cols),
      cat_dims = metadata_for_cat$cat_dims,
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
    num_cont = length(metadata_for_cat$num_cols),
    cat_dims = metadata_for_cat$cat_dims,
    hidden_dim = 256,
    num_hidden = 4,
    dropout = 0.0
  )

  model_bundle <- list(
    encoder_state_dict = encoder_weights,
    encoder_hparams = hparams,
    metadata_for_cat = metadata_for_cat,
    recipe = serialize(recipe, NULL),
    bundle_type = "scarf_bundle"
  )

  return(invisible(model_bundle))


  # If save_path is not null, save model locally (it will be NULL when using it as a recipe, when stored in RAM)
  # if (!is.null(save_path)){
  #   torch::torch_save(model_bundle, path = paste0(save_path, ".pt"))
  #   message("Pretrained model saved in ", save_path, ".pt")
  # }
  #
  # # Return invisible for the recipe prep and bake
  # return(invisible(model_bundle))


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



















