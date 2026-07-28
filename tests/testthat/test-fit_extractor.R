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

# Simple test and output
test_that("fit_extractor with SCARF works and outputs correct structure", {
  skip_if_not_installed("torch")

  df <- create_dummy_data()
  tmp_file <- tempfile(fileext = "")

  on.exit(if (file.exists(paste0(tmp_file, ".pt"))) file.remove(paste0(tmp_file, ".pt")))

  res <- fit_extractor(
    dataframe_train = df,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    create_validation = TRUE,
    validation_proportion = 0.2,
    batch_size = 4,
    n_epochs = 1,
    save_path = tmp_file,
    preprocess = TRUE
  )

  # Check invisible return (for recipes)
  expect_type(res, "list")
  expect_named(res, c("encoder_state_dict", "encoder_hparams", "metadata_for_cat", "recipe", "bundle_type"))
  expect_equal(res$bundle_type, "scarf_bundle")

  # Save_path = something, so it saves file locally
  expect_true(file.exists(paste0(tmp_file, ".pt")))

  # Check saved file
  loaded_bundle <- torch::torch_load(paste0(tmp_file, ".pt"))
  expect_type(loaded_bundle, "list")
  expect_equal(loaded_bundle$bundle_type, "scarf_bundle")

})



# NULL save_path
test_that("fit_extractor handles NULL save_path without writing to disk", {
  skip_if_not_installed("torch")

  df <- create_dummy_data()

  res <- fit_extractor(
    dataframe_train = df,
    pretraining_type = "SCARF",
    exclude_columns = c("id", "target"),
    batch_size = 4,
    n_epochs = 1,
    save_path = NULL
  )

  # Check only returned object
  expect_type(res, "list")
  expect_equal(res$bundle_type, "scarf_bundle")
})



# Invalid method
test_that("fit_extractor throws errors on invalid inputs", {
  skip_if_not_installed("torch")

  df <- create_dummy_data()

  # Check error
  expect_error(
    fit_extractor(df, pretraining_type = "INVALID_METHOD"),
    regexp = "The selected 'pretraining_type' is not supported. Please select one of the available options: 'SCARF'"
  )

})







