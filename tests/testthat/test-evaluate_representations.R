

# --------------------------------------------------------------------------- #
# Test train_classifier_on_extracted_features


classification_setup <- function() {
  df <- data.frame(
    id = 1:40,
    v1 = rnorm(40),
    v2 = runif(40),
    cat1 = factor(sample(c("a", "b"), 20, replace = TRUE)),
    target = factor(sample(c("a", "b"), 40, replace = TRUE))
  )

  tmp_pretrained <- tempfile(fileext = "")

  fit_extractor(
    dataframe_train = df,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    n_epochs = 1,
    save_path = tmp_pretrained,
    batch_size = 8
  )

  list(df = df, pretrained_path = tmp_pretrained)
}



prediction_pipeline_setup <- function() {

  # Train df
  df_tr <- data.frame(
    id = 1:40,
    v1 = rnorm(40),
    v2 = runif(40),
    target = factor(sample(c("a", "b"), 40, replace = TRUE))
  )

  # Test df
  df_te <- data.frame(
    id = 101:115,
    v1 = rnorm(15),
    v2 = runif(15),
    target = factor(sample(c("a", "b"), 15, replace = TRUE))
  )

  # Temp files
  tmp_scarf <- tempfile(fileext = "")
  tmp_mlp <- tempfile(fileext = "")
  tmp_parsnip <- tempfile(fileext = "")

  # Pretrain extractor
  fit_extractor(
    dataframe_train = df_tr,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    n_epochs = 1,
    save_path = tmp_scarf,
    batch_size = 8
  )

  # Train MLP classifier
  train_classifier_on_extracted_features(
    df_train = df_tr,
    pretrained_model_path = tmp_scarf,
    pretraining_type = "SCARF",
    label_column = "target",
    num_classes = 2,
    exclude_columns = c("id", "target"),
    classification_model_type = "MLP",
    n_epochs = 1,
    save_path = tmp_mlp
  )

  # Train parsnip classifier
  if (requireNamespace("parsnip", quietly = TRUE) && requireNamespace("randomForest", quietly = TRUE)) {
    train_classifier_on_extracted_features(
      df_train = df_tr,
      pretrained_model_path = tmp_scarf,
      pretraining_type = "SCARF",
      label_column = "target",
      num_classes = 2,
      exclude_columns = c("id", "target"),
      classification_model_type = "Random Forest",
      save_path = tmp_parsnip
    )
  }

  list(
    df_test = df_te,
    scarf_path = tmp_scarf,
    mlp_path = tmp_mlp,
    parsnip_path = tmp_parsnip
  )
}





test_that("train_classifier_on_extracted_features works with MLP classification head", {
  skip_if_not_installed("torch")

  setup <- classification_setup()
  on.exit(if (file.exists(paste0(setup$pretrained_path, ".pt"))) file.remove(paste0(setup$pretrained_path, ".pt")))

  tmp_class <- tempfile(fileext = "")
  on.exit(if (file.exists(paste0(tmp_class, ".pt"))) file.remove(paste0(tmp_class, ".pt")), add = TRUE)

  # Train MLP
  train_classifier_on_extracted_features(
    df_train = setup$df,
    pretrained_model_path = setup$pretrained_path,
    pretraining_type = "SCARF",
    label_column = "target",
    num_classes = 2,
    exclude_columns = c("id", "target"),
    classification_model_type = "MLP",
    n_epochs = 1,
    save_path = tmp_class
  )

  # Save_path = something, so it saves file locally
  expect_true(file.exists(paste0(tmp_class, ".pt")))

  # Check bundle
  loaded_bundle <- torch::torch_load(paste0(tmp_class, ".pt"))
  expect_type(loaded_bundle, "list")
  expect_equal(loaded_bundle$bundle_type, "classifier_torch_bundle")
  expect_named(loaded_bundle, c("classifier_state_dict", "classifier_hparams", "levels", "bundle_type"))

  # Check levels
  unserialized_levels <- unserialize(loaded_bundle$levels)
  expect_equal(sort(unserialized_levels), c("a", "b"))

})









test_that("train_classifier_on_extracted_features works with parsnip models", {
  skip_if_not_installed("torch")
  skip_if_not_installed("parsnip")
  skip_if_not_installed("randomForest")  # We check with random forest

  setup <- classification_setup()
  on.exit(if (file.exists(paste0(setup$pretrained_path, ".pt"))) file.remove(paste0(setup$pretrained_path, ".pt")))

  tmp_class <- tempfile(fileext = "")
  on.exit(if (file.exists(paste0(tmp_class, ".pt"))) file.remove(paste0(tmp_class, ".pt")), add = TRUE)

  train_classifier_on_extracted_features(
    df_train = setup$df,
    pretrained_model_path = setup$pretrained_path,
    pretraining_type = "SCARF",
    label_column = "target",
    num_classes = 2,
    exclude_columns = c("id", "target"),
    classification_model_type = "Random Forest",
    save_path = tmp_class
  )


  # Save_path = something, so it saves file locally
  expect_true(file.exists(paste0(tmp_class, ".pt")))

  # Check bundle
  loaded_bundle <- torch::torch_load(paste0(tmp_class, ".pt"))
  expect_type(loaded_bundle, "list")
  expect_equal(loaded_bundle$bundle_type, "classifier_parsnip_bundle")
  expect_named(loaded_bundle, c("classifier_model", "levels", "bundle_type"))

})






test_that("train_classifier works with custom parsnip classification model object", {
  skip_if_not_installed("torch")
  skip_if_not_installed("parsnip")

  setup <- classification_setup()
  on.exit(if (file.exists(paste0(setup$pretrained_path, ".pt"))) file.remove(paste0(setup$pretrained_path, ".pt")))

  tmp_class <- tempfile(fileext = "")
  on.exit(if (file.exists(paste0(tmp_class, ".pt"))) file.remove(paste0(tmp_class, ".pt")), add = TRUE)


  custom_model <- parsnip::logistic_reg() |> parsnip::set_engine("glm")

  train_classifier_on_extracted_features(
    df_train = setup$df,
    pretrained_model_path = setup$pretrained_path,
    pretraining_type = "SCARF",
    label_column = "target",
    num_classes = 2,
    exclude_columns = c("id", "target"),
    parsnip_classification_model = custom_model,
    save_path = tmp_class
  )

  # Save_path = something, so it saves file locally
  expect_true(file.exists(paste0(tmp_class, ".pt")))

  # Check bundle
  loaded_bundle <- torch::torch_load(paste0(tmp_class, ".pt"))
  expect_type(loaded_bundle, "list")
  expect_equal(loaded_bundle$bundle_type, "classifier_parsnip_bundle")
  expect_named(loaded_bundle, c("classifier_model", "levels", "bundle_type"))

})




test_that("train_classifier_on_extracted_features handles errors correctly", {
  skip_if_not_installed("torch")

  setup <- classification_setup()
  on.exit(if (file.exists(paste0(setup$pretrained_path, ".pt"))) file.remove(paste0(setup$pretrained_path, ".pt")))

  tmp_class <- tempfile(fileext = "")
  on.exit(if (file.exists(paste0(tmp_class, ".pt"))) file.remove(paste0(tmp_class, ".pt")), add = TRUE)

  # Error if the user do not define the classification model type and he does not use any pre-defined parsnip model
  expect_error(
    train_classifier_on_extracted_features(
      df_train = setup$df,
      pretrained_model_path = setup$pretrained_path,
      pretraining_type = "SCARF",
      label_column = "target",
      num_classes = 2
    ),
    regexp = "You have to define the classification model type or use one pre-defined parsnip model."
  )


  # Error if
  expect_error(
    train_classifier_on_extracted_features(
      df_train = setup$df,
      pretrained_model_path = setup$pretrained_path,
      pretraining_type = "SCARF",
      label_column = "target",
      num_classes = 2,
      classification_model_type = "NON_EXISTENT_MODEL"
    ),
    regexp = "Classification model not supported."
  )

})



# --------------------------------------------------------------------------- #
# Test downstream_prediction

test_that("downstream_prediction works with MLP classifier", {
  skip_if_not_installed("torch")

  setup <- prediction_pipeline_setup()


  on.exit({
    if (file.exists(paste0(setup$scarf_path, ".pt"))) file.remove(paste0(setup$scarf_path, ".pt"))
    if (file.exists(paste0(setup$mlp_path, ".pt"))) file.remove(paste0(setup$mlp_path, ".pt"))
    if (file.exists(paste0(setup$parsnip_path, ".pt"))) file.remove(paste0(setup$parsnip_path, ".pt"))
  })


  results <- downstream_prediction(
    df_test = setup$df_test,
    pretrained_model_path = setup$scarf_path,
    pretraining_type = "SCARF",
    label_column = "target",
    classification_model_path = setup$mlp_path,
    exclude_columns = c("id", "target"),
    return_classification_report = FALSE
  )

  # Check results
  expect_type(results, "list")
  expect_named(results, c("predictions", "probabilities"))

  # Check predictions
  expect_type(results$predictions, "character")
  expect_equal(length(results$predictions), nrow(setup$df_test))
  expect_true(all(results$predictions %in% c("a", "b")))

  # Check probabilities
  expect_true(is.array(results$probabilities))
  expect_equal(nrow(results$probabilities), nrow(setup$df_test))

  # Softmax sum equal 1
  row_sums <- apply(results$probabilities, 1, sum)
  expect_equal(row_sums, rep(1, nrow(setup$df_test)), tolerance = 1e-5)

})




test_that("downstream_prediction works with parsnip classifier", {
  skip_if_not_installed("torch")
  skip_if_not_installed("parsnip")
  skip_if_not_installed("randomForest")

  setup <- prediction_pipeline_setup()


  on.exit({
    if (file.exists(paste0(setup$scarf_path, ".pt"))) file.remove(paste0(setup$scarf_path, ".pt"))
    if (file.exists(paste0(setup$mlp_path, ".pt"))) file.remove(paste0(setup$mlp_path, ".pt"))
    if (file.exists(paste0(setup$parsnip_path, ".pt"))) file.remove(paste0(setup$parsnip_path, ".pt"))
  })


  results <- downstream_prediction(
    df_test = setup$df_test,
    pretrained_model_path = setup$scarf_path,
    pretraining_type = "SCARF",
    label_column = "target",
    classification_model_path = setup$parsnip_path,
    exclude_columns = c("id", "target"),
    return_classification_report = FALSE
  )

  # Check results
  expect_type(results, "list")
  expect_named(results, c("predictions", "probabilities"))

  # Check predictions
  expect_type(results$predictions, "character")
  expect_equal(length(results$predictions), nrow(setup$df_test))
  expect_true(all(results$predictions %in% c("a", "b")))

  # Check probabilities
  expect_true(is.array(results$probabilities))
  expect_equal(nrow(results$probabilities), nrow(setup$df_test))

  # Softmax sum equal 1
  row_sums <- apply(results$probabilities, 1, sum)
  expect_equal(row_sums, rep(1, nrow(setup$df_test)), tolerance = 1e-5)
})



test_that("downstream_prediction returns classification report if asked", {
  skip_if_not_installed("torch")

  setup <- prediction_pipeline_setup()


  on.exit({
    if (file.exists(paste0(setup$scarf_path, ".pt"))) file.remove(paste0(setup$scarf_path, ".pt"))
    if (file.exists(paste0(setup$mlp_path, ".pt"))) file.remove(paste0(setup$mlp_path, ".pt"))
    if (file.exists(paste0(setup$parsnip_path, ".pt"))) file.remove(paste0(setup$parsnip_path, ".pt"))
  })

  expect_output(
    downstream_prediction(
      df_test = setup$df_test,
      pretrained_model_path = setup$scarf_path,
      pretraining_type = "SCARF",
      label_column = "target",
      classification_model_path = setup$mlp_path,
      exclude_columns = c("id", "target"),
      return_classification_report = TRUE
    ),
    regexp = "CLASSIFICATION REPORT"
  )
})




















