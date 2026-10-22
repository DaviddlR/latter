


############ BINNING PRETRAINING ############
binning_fit = function(
    dataframe_train,
    handle_categorical = "top_frequent",   # None means do it exactly how it is done in the original paper
    corruption_during_binning = FALSE,
    exclude_columns = NULL,
    create_validation = FALSE,
    validation_proportion = 0.1,
    batch_size = 256,
    n_epochs = 1,
    preprocess = TRUE
) {

  # Load and preprocess data
  preprocessed_datasets <- prepare_data(
    dataframe_train,
    exclude_columns = exclude_columns,
    create_validation = create_validation,
    validation_proportion = validation_proportion,
    preprocess = preprocess
  )

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set  # May be null
  recipe <- preprocessed_datasets$recipe  # May be null
  metadata_for_cat <- preprocessed_datasets$metadata_for_cat

  #print(x_train)

  # TODO: T como hiperparámetro modificable
  bins <- get_bins(x_train, metadata_for_cat = metadata_for_cat, T = 10, handle_categorical = handle_categorical)
  x_train_binned <- apply_bins(x_train, bins)

  print("----- BINNED -----")
  #print(x_train_binned)

  x_val_binned <- apply_bins(x_val, bins)  # TODO: test que verifique que esto funciona si val es NULL

  # Create training dataset and dataloader
  train_ds <- create_tensor_dataset_for_binning(x_train, x_train_binned)

  train_dl <- torch::dataloader(train_ds,
                                batch_size = batch_size,
                                shuffle = TRUE)

  # Create validation dataset and dataloader (if required)
  val_dl <- NULL

  if(create_validation) {
    val_ds <- create_tensor_dataset_for_binning(x_val, x_val_binned)

    val_dl <- torch::dataloader(val_ds,
                                batch_size = batch_size,
                                shuffle=FALSE)
  }


  num_cont <- length(metadata_for_cat$num_cols)
  cat_dims <- metadata_for_cat$cat_dims

  # Create wrapper

  # Encoder hparams (we will need it to load the bundle)
  # TODO: hacer que esto sean parámetros que puede meter el usuario (?)
  encoder_hidden_dim <- 256
  encoder_num_hidden <- 2

  fitted <- binning_wrapper |>
    luz::setup(
      loss = binning_loss(),
      optimizer = torch::optim_adam
    ) |>
    luz::set_hparams(
      num_cont = num_cont,
      cat_dims = cat_dims,
      encoder_hidden_dim = encoder_hidden_dim,
      encoder_num_hidden = encoder_num_hidden,
      n_bins = bins$n_bins,
      T = bins$T,
      decoder_num_hidden = 2,
      decoder_head = "linear",
      decoder_mask_invalid = TRUE
    ) |>
    luz::set_opt_hparams(
      lr = 0.0001
    ) |>
    luz::fit(
      train_dl,
      epochs = n_epochs,
      valid_data = val_dl,
      callbacks = list(custom_binning_step_callback(want_corruption = corruption_during_binning, corruption_rate = 0.2)),
      # TODO: corruption rate como parametro modificable
    )


  # Save trained model AND the recipe required to apply the same preprocessing to the test set
  encoder_weights <- fitted$model$binning_encoder$state_dict()

  hparams <- list(
    num_cont = num_cont,
    cat_dims = cat_dims,
    hidden_dim = encoder_hidden_dim,
    num_hidden = encoder_num_hidden
  )

  model_bundle <- list(
    encoder_state_dict = encoder_weights,
    encoder_hparams = hparams,
    metadata_for_cat = metadata_for_cat,
    recipe = serialize(recipe, NULL),
    bundle_type = "binning_bundle"
  )

  return(invisible(model_bundle))

}


# Just if the user want to add corruption during the training phase
custom_binning_step_callback <- luz::luz_callback(

  name = "VIME_custom_steps",

  initialize = function(want_corruption, corruption_rate = 0.6) {
    self$want_corruption <- want_corruption
    self$corruption_rate <- corruption_rate
  },

  # Train. It receives a batch from the dataloader / tensor_dataset
  on_train_batch_begin = function() {


    if (self$want_corruption){
      batch <- ctx$batch
      #target <- batch$y  # Label. Not used during pre-training

      x <- batch$x  # Data

      corrupted_batch <- corrupt_batch(x, self$corruption_rate)  # utils.R

      x_corrupted <- corrupted_batch$x_corrupted

      if(is.null(x_corrupted)){
        print("ERRRRRROOOOOOORRRRRR en corrupción")
      }

      ctx$input <- x_corrupted  # Input for the forward method
      # We do not modify ctx$target. It comes from training dataset binned.
    }

  },


  # Validation
  on_valid_batch_begin = function() {
    #print(ctx$batch[[1]]$device)

    if (self$want_corruption){
      batch <- ctx$batch
      #target <- batch$y  # Label. Not used during pre-training

      x <- batch$x  # Data

      corrupted_batch <- corrupt_batch(x, self$corruption_rate)

      x_corrupted <- corrupted_batch$x_corrupted

      ctx$input <- x_corrupted  # Input for the forward method
      # We do not modify ctx$target. It comes from training dataset binned.
    }

  },


)






















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
  preprocessed_datasets <- prepare_data(
    dataframe_train,
    exclude_columns = exclude_columns,
    create_validation = create_validation,
    validation_proportion = validation_proportion,
    preprocess = preprocess
  )

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


  num_cont <- length(metadata_for_cat$num_cols)
  cat_dims <- metadata_for_cat$cat_dims


  # Create wrapper
  fitted <- vime_wrapper |>
    luz::setup(
      loss = vime_loss(alpha = 1.0, num_cont = num_cont, cat_dims = cat_dims),
      optimizer = torch::optim_adam  # Check
    ) |>
    luz::set_hparams(
      num_cont = num_cont,
      cat_dims = cat_dims
    ) |>
    luz::set_opt_hparams(
      lr = 0.0001
    ) |>
    luz::fit(
      train_dl,
      epochs = n_epochs,
      valid_data = val_dl,
      callbacks = list(custom_vime_step_callback(corruption_rate = 0.6))
    )


  # Save trained model AND the recipe required to apply the same preprocessing to the test set
  encoder_weights <- fitted$model$vime_encoder$state_dict()

  hparams <- list(
    num_cont = num_cont,
    cat_dims = cat_dims
  )

  model_bundle <- list(
    encoder_state_dict = encoder_weights,
    encoder_hparams = hparams,
    metadata_for_cat = metadata_for_cat,
    recipe = serialize(recipe, NULL),
    bundle_type = "vime_bundle"
  )

  return(invisible(model_bundle))

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

    corrupted_batch <- corrupt_batch(x, self$corruption_rate)  # utils.R

    x_corrupted <- corrupted_batch$x_corrupted
    mask <- corrupted_batch$mask

    # Binary mask to float (0.0 , 1.0)
    mask_target <- mask$to(dtype = torch::torch_float())

    ctx$input <- x_corrupted  # Input for the forward method
    ctx$target <- list(  # Input for the "target" variable in the loss function. Not needed during forward
      mask = mask_target,
      x = x
    )




  },


  # Validation
  on_valid_batch_begin = function() {
    #print(ctx$batch[[1]]$device)

    batch <- ctx$batch
    #target <- batch$y  # Label. Not used during pre-training

    x <- batch$x  # Data

    corrupted_batch <- corrupt_batch(x, self$corruption_rate)

    x_corrupted <- corrupted_batch$x_corrupted
    mask <- corrupted_batch$mask

    # Binary mask to float (0.0 , 1.0)
    mask_target <- mask$to(dtype = torch::torch_float())


    ctx$input <- x_corrupted  # Input for the forward method
    ctx$target <- list(  # Input for the "target" variable in the loss function. Not needed during forward
      mask = mask_target,
      x = x
    )

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
  preprocessed_datasets <- prepare_data(dataframe_train, exclude_columns = exclude_columns, create_validation = create_validation, validation_proportion = validation_proportion, preprocess = preprocess)

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

    corrupted_batch <- corrupt_batch(x, self$corruption_rate)
    x_corrupted <- corrupted_batch$x_corrupted



    ctx$input <- list(x, x_corrupted)
    #ctx$batch[[2]] <- batch$y
    #ctx$batch[[1]] <- c(x, x_corrupted)



  },


  # Validation
  on_valid_batch_begin = function() {
    batch = ctx$batch
    #target = batch$y  # Label. Not used during pre-training

    x = batch$x  # Data

    corrupted_batch <- corrupt_batch(x, self$corruption_rate)
    x_corrupted <- corrupted_batch$x_corrupted

    #ctx$target <- batch$y
    ctx$input <- list(x, x_corrupted)
  },


  # Test / predict

)



















