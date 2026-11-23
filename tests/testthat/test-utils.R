as_float_tensor <- function(m) torch::torch_tensor(m, dtype = torch::torch_float32())


# ---------- Eligibility ----------

test_that("with noise_on_binary = TRUE every column is eligible (official behaviour): NULL", {
  skip_if_not(torch::torch_is_installed())

  idx <- subtab_view_indices(make_subtab_views(8, 2, 0.5))
  expect_null(subtab_noise_eligibility(idx, rep(FALSE, 8), noise_on_binary = TRUE))
})

test_that("with noise_on_binary = FALSE the binary columns of each view are excluded", {
  skip_if_not(torch::torch_is_installed())

  idx <- subtab_view_indices(make_subtab_views(8, 2, 0.5))  # Views 1:6 and 3:8
  binary_columns <- c(FALSE, TRUE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE)  # Columns 2 and 5 are binary

  eligible <- subtab_noise_eligibility(idx, binary_columns, noise_on_binary = FALSE)

  expected <- rbind(!binary_columns[1:6], !binary_columns[3:8])  # [n_views, view_width]
  expect_equal(eligible$shape, c(2, 6))
  expect_equal(torch::as_array(eligible), expected)
})


# ---------- Noise: general properties ----------

test_that("noise keeps the shape", {
  skip_if_not(torch::torch_is_installed())

  x_views <- torch::torch_randn(2 * 10, 4)

  for (noise_type in c("swap_noise", "gaussian_noise", "zero_out")) {
    out <- subtab_add_noise(x_views, n_views = 2, noise_type = noise_type, masking_ratio = 0.5)

    expect_equal(out$shape, c(20, 4))
  }
})

test_that("masking_ratio = 0 leaves every element untouched, for every noise type", {
  skip_if_not(torch::torch_is_installed())

  x_views <- torch::torch_randn(3 * 10, 4)

  for (noise_type in c("swap_noise", "gaussian_noise", "zero_out")) {
    out <- subtab_add_noise(x_views, n_views = 3, noise_type = noise_type, masking_ratio = 0)
    expect_equal(torch::as_array(out), torch::as_array(x_views))
  }
})


test_that("zero_out with masking_ratio = 1 gives all zeros", {
  skip_if_not(torch::torch_is_installed())

  out <- subtab_add_noise(torch::torch_randn(2 * 10, 4), n_views = 2, noise_type = "zero_out", masking_ratio = 1)
  expect_true(all(torch::as_array(out) == 0))
})

test_that("gaussian noise and noise_level = 0 changes nothing", {
  skip_if_not(torch::torch_is_installed())

  x_views <- torch::torch_randn(3 * 10, 5)
  out0 <- subtab_add_noise(x_views, n_views = 3, noise_type = "gaussian_noise", masking_ratio = 1, noise_level = 0)
  expect_equal(torch::as_array(out0), torch::as_array(x_views))
})



# ---------- Eligibility in the noise ----------

test_that("columns that are not eligible are never corrupted, for every noise type", {
  skip_if_not(torch::torch_is_installed())

  idx <- matrix(c(1, 2, 3, 4,
                  1, 2, 3, 4), nrow = 2, byrow = TRUE)
  eligible <- subtab_noise_eligibility(idx, c(FALSE, TRUE, FALSE, TRUE), noise_on_binary = FALSE)  # Columns 2 and 4 are protected

  x <- matrix(rnorm(2 * 40 * 4), nrow = 80, ncol = 4)
  x_tensor <- as_float_tensor(x)
  x <- torch::as_array(x_tensor)

  for (noise_type in c("swap_noise", "gaussian_noise", "zero_out")) {
    out <- torch::as_array(subtab_add_noise(x_tensor, n_views = 2, noise_type = noise_type, masking_ratio = 1, noise_level = 0.5, eligible = eligible))

    expect_equal(out[, c(2, 4)], x[, c(2, 4)])                     # Protected columns: untouched
    expect_false(isTRUE(all.equal(out[, c(1, 3)], x[, c(1, 3)])))  # The rest: corrupted
  }
})

test_that("without protection (official), swap noise creates impossible one-hot states; with protection it does not", {
  skip_if_not(torch::torch_is_installed())

  batch <- 300
  levels <- sample(1:3, batch, replace = TRUE)
  one_hot <- diag(3)[levels, ]  # 3 dummies of ONE variable: exactly one is active in every row
  x_tensor <- as_float_tensor(one_hot)

  invalid_rows <- function(out) mean(rowSums(torch::as_array(out)) != 1)

  # Official: every column is corrupted on its own
  official <- subtab_add_noise(x_tensor, n_views = 1, noise_type = "swap_noise", masking_ratio = 0.3, eligible = NULL)
  expect_gt(invalid_rows(official), 0.1)

  # Protected: the dummies are detected as binary and never corrupted
  idx <- matrix(1:3, nrow = 1)
  protected_eligibility <- subtab_noise_eligibility(idx, subtab_detect_binary_columns(one_hot), noise_on_binary = FALSE)
  protected <- subtab_add_noise(x_tensor, n_views = 1, noise_type = "swap_noise", masking_ratio = 0.3, eligible = protected_eligibility)
  expect_equal(invalid_rows(protected), 0)
})

test_that("with noise_on_binary = TRUE the binary columns are corrupted too", {
  skip_if_not(torch::torch_is_installed())

  dummy <- matrix(rep(c(0, 1), 100), ncol = 1)
  idx <- matrix(1, nrow = 1, ncol = 1)

  expect_null(subtab_noise_eligibility(idx, subtab_detect_binary_columns(dummy), noise_on_binary = TRUE))

  out <- torch::as_array(subtab_add_noise(as_float_tensor(dummy), n_views = 1, noise_type = "gaussian_noise",
                                          masking_ratio = 1, noise_level = 0.5, eligible = NULL))
  expect_true(any(out != 0 & out != 1))  # Values like 0.13 in a 0/1 column, as in the official implementation
})

# ---------- Validation ----------

test_that("invalid arguments are rejected", {
  skip_if_not(torch::torch_is_installed())

  x_views <- torch::torch_randn(2 * 10, 4)

  expect_error(subtab_add_noise(x_views, 2, masking_ratio = 1.5), "masking_ratio")
  expect_error(subtab_add_noise(x_views, 2, masking_ratio = -0.1), "masking_ratio")
  expect_error(subtab_add_noise(x_views, 2, noise_level = -1), "noise_level")
  expect_error(subtab_add_noise(x_views, 2, noise_type = "unknown"))
})

# ---------- Crop + noise ----------

test_that("crop and noise work together and the clean batch is untouched", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(20 * 8), nrow = 20, ncol = 8)
  x_tensor <- as_float_tensor(x)

  idx <- subtab_view_indices(make_subtab_views(8, 2, 0.5))
  eligible <- subtab_noise_eligibility(idx, subtab_detect_binary_columns(x), noise_on_binary = FALSE)

  views <- subtab_crop_views(x_tensor, idx)
  noisy <- subtab_add_noise(views, n_views = 2, noise_type = "swap_noise", masking_ratio = 0.3, eligible = eligible)

  expect_equal(noisy$shape, c(40, 6))
  expect_equal(torch::as_array(x_tensor), torch::as_array(as_float_tensor(x)))  # The target stays clean
})
