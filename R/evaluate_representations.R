



#' Train a classifier on top of the latent representations created by a pretrained model.
#'
#' @param df_train A \code{data.frame} containing the training samples and labels.
#' @param pretrained_model_path \code{String}. Path to the pretrained model for feature extraction (.pt file).
#' @param label_column \code{String}. Name of the column containing the labels. Required if \code{want_labels = TRUE}. Default is \code{NULL}.
#' @param num_classes \code{Integer}. Total number of unique classes in the target column.
#' @param exclude_columns A \code{string} of columns that the models should ignore (i.e target or ID columns). Default is \code{NULL}.
#' @param parsnip_classification_model A \code{parsnip} classification model that has been already created and loaded. Default is \code{NULL}
#' @param classification_model_type \code{String}. Type of architecture to train. Currently, \code{"MLP", "Random Forest", "XGB", "KNN", "SVM" and "C50"} are supported. Default is \code{NULL}.
#' @param n_epochs \code{Integer}. Number of training epochs if classification_model_type is 'MLP'. Default is \code{50}.
#' @param dropout \code{Numeric}. Dropout probability for the classification layers. Default is \code{0.2}.
#' @param doitsmall \code{Boolean}. If \code{TRUE}, sub-samples the training set to 5\% for quick experimentation. Default is \code{FALSE}.
#' @param save_path \code{String}. Path where the trained classifier bundle (.pt) will be saved. Extension ('.pt') should not be included. Default is \code{"classifier"}, which saves a 'classifier.pt' file in the current directory.
#'
#' @returns Invisible \code{NULL}. Saves a serialized classifier bundle containing the network weights, hyperparameters, and the target factor levels to disk.
#' @export
#'
#' @examples
#' \donttest{
#'
#' if (torch::torch_is_installed()) {
#'   df_train <- data.frame(
#'     id = 1:100,
#'     v1 = rnorm(100),
#'     v2 = factor(sample(c("A", "B"), 100, replace = TRUE)),
#'     target = sample(c("Class1", "Class2"), 100, replace = TRUE)
#'   )
#'
#'   tmp_pretrained <- tempfile(fileext = ".pt")
#'   tmp_class <- tempfile()
#'
#'   fit_extractor(
#'     dataframe_train = df_train,
#'     pretraining_type = "SCARF",
#'     exclude_columns = c("id", "target"),
#'     n_epochs = 1,
#'     save_path = tmp_pretrained
#'   )
#'
#'   # Train classifier
#'   train_classifier_on_extracted_features(
#'     df_train = df_train,
#'     pretrained_model = tmp_pretrained,
#'     label_column = "target",
#'     num_classes = 2,
#'     exclude_columns = c("id", "target"),
#'     classification_model_type = "MLP",
#'     save_path = tmp_class
#'   )
#'
#'   if (file.exists(tmp_pretrained)) file.remove(tmp_pretrained)
#'   if (file.exists(paste0(tmp_class, ".pt"))) file.remove(paste0(tmp_class, ".pt"))
#' }
#' }
train_classifier_on_extracted_features = function(df_train, pretrained_model_path, label_column, num_classes, exclude_columns = NULL, parsnip_classification_model = NULL, classification_model_type = NULL, n_epochs = 50, dropout = 0.2, doitsmall = FALSE, save_path = "classifier") {

  # TODO: borrar pretraining_type. No sirve para nada

  if(doitsmall) {

    print("Doing it small...")
    label_proportion <- 0.05

    df_train <- df_train |>
      dplyr::group_by(.data[[label_column]]) |>
      dplyr::sample_frac(label_proportion) |>
      dplyr::ungroup()
  }

  # Extract latent features of train set
  extracted_features <- extract_features(
    df_train,
    pretrained_model_path,
    exclude_columns = exclude_columns,
    want_labels = TRUE,
    label_column = label_column)

  features <- extracted_features$features
  y_train <- extracted_features$features_labels


  # Label encoder
  y_train_factor <- as.factor(y_train)
  train_levels <- levels(y_train_factor)  # Have to save train_levels
  print(train_levels)

  # y_val_encoded <- as.integer(factor(y_val, levels = train_levels))



  # TRAIN MLP
  # TODO: permitir que especifique su propio modelo MLP ??
  if (identical(classification_model_type, "MLP")) {

    # Convert factor labels to integer
    y_train_encoded <- as.integer(y_train_factor)

    # Set X and Y as tensors
    features_tensor <- torch::torch_tensor(features, dtype = torch::torch_float())
    y_train_tensor <- torch::torch_tensor(y_train_encoded, dtype = torch::torch_long())

    # Create dataset and dataloader and validation if required
    train_ds <- torch::tensor_dataset(features_tensor, y_train_tensor)

    train_dl <- torch::dataloader(
      train_ds,
      batch_size = 256,
      shuffle = TRUE,
    )

    val_dl <- NULL  # TODO

    # Create and train classification head
    fitted_classification_head <- classifier_network |>
      luz::setup(
        loss = torch::nn_cross_entropy_loss(),
        optimizer = torch::optim_adam,
      ) |>
      luz::set_hparams(
        input_dim = dim(features)[[2]],
        n_classes = num_classes,
        dropout = dropout,
      ) |>
      luz::set_opt_hparams(
        lr = 0.0001,
      ) |>
      luz::fit(
        train_dl,
        epochs = n_epochs,
        valid_data = val_dl,
      )

    print(fitted_classification_head)


    # Save classification model and train levels.


    classifier_weights <- fitted_classification_head$model$state_dict()

    hparams <- list(
      in_dim = dim(features)[[2]],
      n_classes = num_classes,
      dropout = dropout
    )

    model_bundle <- list(
      classifier_state_dict = classifier_weights,
      classifier_hparams = hparams,
      levels = serialize(train_levels, NULL),
      bundle_type = "classifier_torch_bundle"
    )

  # TRAIN MODEL USING PARSNIP
  # Train predefined model
  } else if (!(is.null(parsnip_classification_model))) {

    if (!requireNamespace("parsnip", quietly = TRUE)){
      stop("The 'parsnip' package is not installed. To train the specified classifier you need that package")
    }

    # Fit model
    fitted_model <- parsnip_classification_model |>
      parsnip::fit_xy(
        x = as.data.frame(features),
        y = y_train_factor
      )


    # Create model bundle to save
    model_bundle <- list(
      classifier_model = serialize(fitted_model, NULL),
      levels = serialize(train_levels, NULL),
      bundle_type = "classifier_parsnip_bundle"
    )

  # Train selected model type
  } else if (!(is.null(classification_model_type))) {

    if (!requireNamespace("parsnip", quietly = TRUE)){
      stop("The 'parsnip' package is not installed. To train the specified classifier you need that package")
    }


    # Create model or use the one provided by the user
    # TODO: check hyperparameters + validación de librerías?
    model_definition <- switch (classification_model_type,
      "Random Forest" = parsnip::rand_forest(mode = "classification", trees = 100) |> parsnip::set_engine("randomForest"),
      "XGB" = parsnip::boost_tree(mode = "classification") |> parsnip::set_engine("xgboost"),
      "SVM" = parsnip::svm_rbf(mode = "classification") |> parsnip::set_engine("kernlab"),
      "KNN" = parsnip::nearest_neighbor(mode = "classification", neighbors = 5) |> parsnip::set_engine("kknn"),
      "C50" = parsnip::decision_tree(mode = "classification") |> parsnip::set_engine("C5.0"),
      stop("Classification model not supported. Please, use one of the available options ['MLP', Random Forest', 'XGB', 'SVM', 'KNN', 'C50'] or create the model yourself using parsnip and send the object as hyperparameter of this function with 'parsnip_classification_model_object' = your_model'. ")
    )


    # Fit model
    fitted_model <- model_definition |>
      parsnip::fit_xy(
        x = as.data.frame(features),
        y = y_train_factor
      )


    # Create model bundle to save
    model_bundle <- list(
      classifier_model = serialize(fitted_model, NULL),
      levels = serialize(train_levels, NULL),
      bundle_type = "classifier_parsnip_bundle"
    )



  } else {
    stop("You have to define the classification model type or use one pre-defined parsnip model.")
  }


  # Save bundle

  torch::torch_save(model_bundle, path = paste0(save_path, ".pt"))



}




#' Obtain predictions of a classifier trained on top of latent representations
#'
#' @param df_test A \code{data.frame} representing the test/evaluation set.
#' @param pretrained_model_path \code{String}. Path to the pretrained model for feature extraction (.pt file).
#' @param label_column \code{String}. Name of the column containing the true labels (used for mapping or reporting).
#' @param classification_model_path \code{String}. Path prefix to the trained classifier bundle (excluding the ".pt" extension).
#' @param exclude_columns A \code{string} of columns that the models should ignore (i.e target or ID columns). Default is \code{NULL}.
#' @param return_classification_report \code{Boolean}. If \code{TRUE}, prints a confusion matrix and the global accuracy to the console. Default is \code{FALSE}.
#'
#' @returns A \code{list} containing:
#' \itemize{
#'   \item \code{predictions}: A String vector with the predicted class names for each sample.
#'   \item \code{probabilities}: An array containing the softmax probability scores for each class.
#' }
#'
#' @export
#'
#' @examples
#' \dontrun{
#'
#' if (torch::torch_is_installed()) {
#'   df_train <- data.frame(
#'     id = 1:100,
#'     v1 = rnorm(100),
#'     v2 = factor(sample(c("A", "B"), 100, replace = TRUE)),
#'     target = sample(c("Class1", "Class2"), 100, replace = TRUE)
#'   )
#'
#'   df_test <- data.frame(
#'     id = 1:20,
#'     v1 = rnorm(100),
#'     v2 = factor(sample(c("A", "B"), 100, replace = TRUE)),
#'     target = sample(c("Class1", "Class2"), 100, replace = TRUE)
#'   )
#'
#'   tmp_scarf <- tempfile(fileext = ".pt")
#'   tmp_class <- tempfile()
#'
#'   # SCARF pretraining and train classifier
#'   fit_extractor(
#'     df_train,
#'     "SCARF",
#'     exclude_columns = c("id", "target"),
#'     n_epochs = 1,
#'     save_path = tmp_scarf
#'   )
#'
#'   train_classifier_on_extracted_features(
#'     df_train,
#'     tmp_scarf,
#'     "target",
#'     num_classes = 2,
#'     exclude_columns = c("id", "target"),
#'     save_path = tmp_class
#'   )
#'
#'   results <- downstream_prediction(
#'     df_test = df_test,
#'     pretrained_model_path = tmp_scarf,
#'     label_column = "target",
#'     classification_model_path = tmp_class,
#'     exclude_columns = c("id", "target"),
#'     return_classification_report = TRUE
#'   )
#'
#'   if (file.exists(tmp_scarf)) file.remove(tmp_scarf)
#'   if (file.exists(paste0(tmp_class, ".pt"))) file.remove(paste0(tmp_class, ".pt"))
#' }
#'
#' }
#'
downstream_prediction = function(df_test, pretrained_model_path, label_column, classification_model_path, exclude_columns = NULL, return_classification_report = FALSE) {

  # Load model
  fitted_classifier_bundle <- load_classifier_bundle(paste0(classification_model_path, ".pt"))

  fitted_classifier <- fitted_classifier_bundle$classifier
  classifier_type <- fitted_classifier_bundle$type

  print(classifier_type)

  # Extract latent features of test set
  extracted_features_test <- extract_features(
    df_test,
    pretrained_model = pretrained_model_path,
    exclude_columns = exclude_columns,
    want_labels = TRUE,
    label_column = label_column,
    batch_size = 32
  )

  features_test <- extracted_features_test$features
  labels_test <- extracted_features_test$features_labels


  # Process labels
  y_test <- factor(labels_test, levels = fitted_classifier_bundle$levels)
  # Label encoder
  y_test_encoded <- as.integer(y_test)

  # If classifier is MLP
  if (identical(classifier_type, "torch")) {

    # Set X and Y as tensors
    features_test_tensor <- torch::torch_tensor(features_test, dtype = torch::torch_float())
    y_test_tensor <- torch::torch_tensor(y_test_encoded, dtype = torch::torch_long())

    print(y_test_tensor[2])

    # Create dataset and dataloader
    test_ds <- torch::tensor_dataset(features_test_tensor, y_test_tensor)

    test_dl <- torch::dataloader(
      test_ds,
      batch_size = 256,
      shuffle = FALSE
    )

    # Predict on the test set

    # Prepare model
    device <- if(torch::cuda_is_available()) torch::torch_device("cuda") else torch::torch_device("cpu")
    message("Ejecutando inferencia en: ", if (torch::cuda_is_available()) "GPU (CUDA)" else "CPU")

    fitted_classifier$to(device = device)
    fitted_classifier$eval()

    # Loop on the dataloader
    predictions <- list()

    torch::with_no_grad({
      coro::loop(
        for(batch in test_dl) {

          # Take batch
          x_batch <- batch[[1]]$to(device = device)

          # Forward pass
          batch_prediction <- fitted_classifier(x_batch)

          # Store predictions
          predictions[[length(predictions) + 1]] <- batch_prediction$cpu()
        }
      )
    })

    # Concatenate batches
    predictions <- torch::torch_cat(predictions, dim=1)

    print("Raw predictions: ")
    print(dim(predictions))
    print(predictions[2])



    # Get probabilities
    sm <- torch::nn_softmax(dim = 2)
    probabilities <- sm(predictions)
    probabilities <- as.array(probabilities)
    colnames(probabilities) <- fitted_classifier_bundle$levels

    print("Probabilities: ")
    print(probabilities)
    print(dim(probabilities))
    print(probabilities[2, ])


    # Get predicted class index
    pred_indices <- as.integer(torch::torch_argmax(probabilities, dim=2))
    print("pred_indices")
    print(pred_indices)
    print(pred_indices[2])

    # Get predicted class name (using train_levels)
    pred_label <- fitted_classifier_bundle$levels[pred_indices]
    print("pred_label")
    print(pred_label)
    print(pred_label[2])



    # If classifier is parsnip
  } else if (identical(classifier_type, "parsnip")) {

    features_df <- as.data.frame(features_test)
    prob_df <- parsnip::predict.model_fit(fitted_classifier, new_data = features_df, type = "prob")  # Get probabilities
    class_df <- parsnip::predict.model_fit(fitted_classifier, new_data = features_df, type = "class")  # Get class predictions

    print("test")
    print(class_df)  # de aqui tiene que salir pred_label
    print(fitted_classifier_bundle$levels)

    # Generate probabilities
    expected_cols <- paste0(".pred_", fitted_classifier_bundle$levels)
    probabilities <- as.matrix(prob_df[, expected_cols, drop = FALSE])
    colnames(probabilities) <- fitted_classifier_bundle$levels
    print("Probabilities")  # Good
    print(probabilities)
    print(dim(probabilities))
    print(probabilities[2, ])

    # Get class label
    pred_label <- as.character(class_df$.pred_class)
    print("pred_label")
    print(pred_label)
    print(pred_label[2])

    # Get predicted class index
    pred_indices <- match(pred_label, fitted_classifier_bundle$levels)
    print("pred_indices")
    print(pred_indices)
    print(pred_indices[2])




  } else {
    stop("The classifier model is not recognized. Please, train a model using 'train_classifier_on_representations'.")
  }






  # Evaluate if required
  if(return_classification_report){
    print("CLASSIFICATION REPORT")

    # Get num classes and adjust confusion matrix in case some classes were not predicted
    num_classes <- length(fitted_classifier_bundle$levels)
    all_levels <- 1:num_classes

    pred_factor <- factor(pred_indices, levels = all_levels)
    real_factor <- factor(y_test_encoded, levels = all_levels)

    confusion <- table(Predicted = pred_factor, Real = real_factor)
    print(confusion)
    accuracy <- sum(diag(confusion)) / sum(confusion)

    cat("Accuracy Global:", round(accuracy * 100, 2), "%\n\n")

  }

  # Return predictions and probabilities
  return(list(
    predictions = pred_label,
    probabilities = probabilities
  ))




}














