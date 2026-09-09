# FINISHED



create_dummy_data <- function() {
  data.frame(
    id = 1:20,
    num1 = rnorm(20),
    num2 = runif(20),
    cat1 = factor(sample(c("a", "b"), 20, replace = TRUE)),
    target = sample(0:1, 20, replace = TRUE)
  )
}


# Want_label
test_that("extract_features works with model saved on disk and returns labels", {
  skip_if_not_installed("torch")

  df <- create_dummy_data()
  tmp_file <- tempfile(fileext = "")
  on.exit(if (file.exists(paste0(tmp_file, ".pt"))) file.remove(paste0(tmp_file, ".pt")))

  # Train model
  fit_extractor(
    dataframe_train = df,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    n_epochs = 1,
    save_path = tmp_file,
    batch_size = 8
  )

  # Extract features
  features <- extract_features(
    dataframe = df,
    pretrained_model = tmp_file,
    exclude_columns = c("id", "target"),
    want_labels = TRUE,
    label_column = "target",
    batch_size = 8
  )

  expect_true(is.matrix(features$features))
  expect_type(features, "list")
  expect_named(features, c("features", "features_labels"))

  expect_equal(nrow(features$features), nrow(df))
  expect_equal(length(features$features_labels), nrow(df))
  expect_equal(features$features_labels, df$target)

})





test_that("extract_features works with model bundle loaded in RAM", {
  skip_if_not_installed("torch")

  df <- create_dummy_data()

  # Train model
  pretrained_bundle <- fit_extractor(
    dataframe_train = df,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    n_epochs = 1,
    save_path = NULL,
    batch_size = 8
  )

  # Extract features with an already-loaded model
  features <- extract_features(
    dataframe = df,
    pretrained_model = pretrained_bundle,
    exclude_columns = c("id", "target"),
    want_labels = FALSE,
    batch_size = 8
  )

  expect_true(is.matrix(features$features))
  expect_type(features, "list")
  expect_named(features, c("features", "features_labels"))

  expect_equal(nrow(features$features), nrow(df))
  expect_null(features$features_labels)

})





test_that("extract_features check missing label_column when want_labels = TRUE", {

  skip_if_not_installed("torch")

  df <- create_dummy_data()

  # Train model
  pretrained_bundle <- fit_extractor(
    dataframe_train = df,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    n_epochs = 1,
    save_path = NULL,
    batch_size = 8
  )

  expect_error(
    extract_features(
      dataframe = df,
      pretrained_model = pretrained_bundle,  # NULL bundle just to test
      want_labels = TRUE,
      label_column = NULL
    ),
    regexp = "extract_features: if 'want_labels' is TRUE, then you have to specify the label column with the parameter 'label_column'"
  )
})



