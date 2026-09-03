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


# prepare_data
test_that("prepare_data correctly excludes columns and formats matrix", {

  df <- create_dummy_data()

  output <- prepare_data(
    dataframe_train = df,
    exclude_columns = c("id", "target"),
    create_validation = FALSE,
    preprocess = TRUE
  )

  # Expect correct format
  expect_type(output, "list")
  expect_true(is.matrix(output$train_set))

  # Expect NULL validation
  expect_null(output$val_set)

  # Expect exclude_columns are removed
  colnames_out <- colnames(output$train_set)
  expect_false("id" %in% colnames_out)
  expect_false("target" %in% colnames_out)

  # Expect a recipe
  expect_s3_class(output$recipe, "recipe")

})



# Check validation
test_that("prepare_data creates validation splits", {
  df <- create_dummy_data()

  output <- prepare_data(
    dataframe_train = df,
    exclude_columns = c("id", "target"),
    create_validation = TRUE,
    validation_proportion = 0.2,
    preprocess = TRUE
  )

  expect_true(is.matrix(output$val_set))

  # Check dimensions
  expect_equal(nrow(output$train_set), 16)
  expect_equal(nrow(output$val_set), 4)

  expect_equal(ncol(output$val_set), ncol(output$train_set))
})



test_that("prepare_data works without preprocess", {

  df <- create_dummy_data()

  output <- prepare_data(
    dataframe_train = df,
    exclude_columns = c("id", "target"),
    create_validation = TRUE,
    validation_proportion = 0.2,
    preprocess = FALSE
  )

  expect_true(is.matrix(output$train_set))
  expect_null(output$recipe)
  expect_equal(ncol(output$train_set), 3)

})




