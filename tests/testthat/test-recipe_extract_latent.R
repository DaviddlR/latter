create_dummy_data <- function() {
  data.frame(
    id = 1:20,
    num1 = rnorm(20),
    num2 = runif(20),
    num3 = rnorm(20, mean = 5),
    cat1 = factor(sample(c("a", "b", "c", "d"), 20, replace = TRUE)),
    target = factor(sample(c("a", "b"), 20, replace = TRUE))
  )
}


test_that("step_extract_latent integrates with recipes pipeline, prep and bake", {
  skip_if_not_installed("torch")
  skip_if_not_installed("recipes")

  df_train <- create_dummy_data()
  df_test <- create_dummy_data()

  # Create recipe
  recipe <- recipes::recipe(target ~ ., data = df_train) |>
    recipes::update_role(id, new_role = "id") |>
    step_extract_latent(
      recipes::all_predictors(),
      pretraining_type = "SCARF",
      epochs = 1,
      batch_size = 8,
      batch_size_inference = 8
    )

  expect_false(recipe$steps[[1]]$trained)  # Check not trained

  # Prep
  prepped_recipe <- recipes::prep(
    recipe,
    training = df_train
  )
  trained_step <- prepped_recipe$steps[[1]]


  expect_true(trained_step$trained)  # Check trained
  expect_equal(unname(trained_step$columns), c("num1", "num2", "num3", "cat1"))  # Check processed columns
  expect_equal(trained_step$pretrained_model$bundle_type, "scarf_bundle")  # Check pretrained model is stored

  # Bake
  baked_df <- recipes::bake(prepped_recipe, new_data = df_test)

  expect_s3_class(baked_df, "tbl_df")
  expect_equal(nrow(baked_df), nrow(df_test))

  expect_false(any(c("num1", "num2", "num3") %in% colnames(baked_df)))
  expect_true("id" %in% colnames(baked_df))
  expect_true("target" %in% colnames(baked_df))
  expect_true(any(grepl("^extracted_dim_", colnames(baked_df))))

})



test_that("tidy method works before and after prep", {
  skip_if_not_installed("torch")
  skip_if_not_installed("recipes")


  df <- create_dummy_data()

  rec <- recipes::recipe(target ~ num1 + num2 + cat1, data = df) |>
    step_extract_latent(num1, num2, cat1, epochs = 1, batch_size = 8)

  # untrained tidy
  tidy_unprepped <- generics::tidy(rec, number = 1)
  #print(tidy_unprepped)
  expect_s3_class(tidy_unprepped, "tbl_df")
  expect_true("terms" %in% colnames(tidy_unprepped))

  # trained tidy. Returns final column names.
  prepped_rec <- recipes::prep(rec, training = df)
  tidy_prepped <- generics::tidy(prepped_rec, number = 1)

  expect_s3_class(tidy_prepped, "tbl_df")
  expect_equal(unname(tidy_prepped$terms), c("num1", "num2", "cat1"))
})



test_that("print and required_pkgs methods work as expected", {
  skip_if_not_installed("torch")
  skip_if_not_installed("recipes")

  df <- create_dummy_data()

  rec <- recipes::recipe(target ~ ., data = df) |>
    step_extract_latent(
      recipes::all_predictors(),
      pretraining_type = "SCARF",
      epochs = 1,
      batch_size = 8,
      batch_size_inference = 8
    )

  expect_snapshot(print(rec))

  pkgs <- recipes::required_pkgs(rec)
  expect_type(pkgs, "character")
  expect_true("torch" %in% pkgs)
})















