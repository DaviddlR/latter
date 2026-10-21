
meta <- function(n_num, cat_dims = integer(0)) {
  list(num_cols = seq_len(n_num), cat_dims = cat_dims)
}  # Simulate metadata for cat

fit_num <- function(x, T) get_bins(matrix(x, ncol = 1), meta(1), T = T)
fit_cat <- function(codes, K, T) get_bins(matrix(codes, ncol = 1), meta(0, K), T = T)
bin_of  <- function(x, bins) as.vector(apply_bins(matrix(x, ncol = 1), bins))

test_that("get_bins and apply_bins returns valid bins", {
  # Three columns: numeric with num values > T, numeric with num values < T, categorical with k (5) > T (4)
  x <- cbind(rnorm(200), sample(0:1, 200, TRUE), sample(1:5, 200, TRUE))

  meta <- list(num_cols = c("a", "b"), cat_dims = 5)  # Create metadata for cat

  bins <- get_bins(x, metadata_for_cat = meta, T = 4)
  out  <- apply_bins(x, bins)

  expect_true(is.integer(out))
  expect_true(all(out >= 1L))
  for (j in seq_len(ncol(out))) expect_lte(max(out[, j]), bins$n_bins[j])  # Cannot have more bins than n_bins[j]
  expect_null(apply_bins(NULL, bins))  # Simulate validation set as NULL (no validation)
})


test_that("bin numeric features: at most T distinct values one bin per value", {
  x <- c(3, 3, 3, 7, 9)
  bins <- fit_num(x, T = 10)

  expect_equal(bins$n_bins, 3L)
  expect_equal(bin_of(x, bins), c(1L, 1L, 1L, 2L, 3L))
  # a value not seen in training takes the bin of the closest lower value
  expect_equal(bin_of(8, bins), 2L)
})

test_that("bin numeric features: quantile bins follow the t/T rule", {
  x <- 1:8                                  # cuts: 2.75, 4.5, 6.25
  bins <- fit_num(x, T = 4)

  expect_equal(bins$n_bins, 4L)
  expect_equal(bin_of(x, bins), c(1L, 1L, 2L, 2L, 3L, 3L, 4L, 4L))
  # a value lying exactly on a cut goes to the upper bin: [b_{t-1}, b_t)
  expect_equal(bin_of(4.5, bins), 3L)
})

test_that("bin numeric features: exactly T bins (not T - 1) and T = 2 works", {
  bins <- fit_num(1:1000, T = 10)
  expect_equal(bins$n_bins, 10L)
  expect_equal(as.vector(table(bin_of(1:1000, bins))), rep(100L, 10))

  bins2 <- fit_num(1:10, T = 2)
  expect_equal(bins2$n_bins, 2L)
  expect_equal(bin_of(1:10, bins2), rep(1:2, each = 5))
})

test_that("bin numeric features: ties never create empty bins and common values shares bin", {
  x <- c(rep(0, 70), 1:30)                  # 70 % of the values are 0
  bins <- fit_num(x, T = 10)
  out <- bin_of(x, bins)

  expect_lte(bins$n_bins, 10L)
  expect_equal(bins$n_bins, 4L)
  expect_true(all(out[1:70] == 1L))                       # all zeros share a bin
})

test_that("bin numeric features: a constant column gets a single bin", {
  bins <- fit_num(rep(5, 20), T = 10)

  expect_equal(bins$n_bins, 1L)
  expect_equal(bin_of(c(5, 5, 99), bins), c(1L, 1L, 1L))
})

test_that("bin numeric features: apply_bins reuses the training cuts", {
  bins <- fit_num(1:8, T = 4)

  # values outside the training range fall in the first / last bin
  expect_equal(bin_of(c(-100, 100), bins), c(1L, 4L))
  # a subset is binned with the training cuts, not re-quantized
  expect_equal(bin_of(c(7, 8), bins), c(4L, 4L))
})





test_that("bin categorical features: K <= T -> one bin per level (bin = code)", {
  codes <- c(1, 2, 3, 2, 1)
  bins <- fit_cat(codes, K = 3, T = 10)

  expect_equal(bins$n_bins, 3L)
  expect_equal(bin_of(codes, bins), c(1L, 2L, 3L, 2L, 1L))
  expect_null(bins$features[[1]]$other_share)
  expect_equal(fit_cat(codes, K = 3, T = 3)$n_bins, 3L)    # boundary K == T
})

test_that("bin categorical features: K > T -> T - 1 most frequent levels + Others", {
  codes <- rep(1:5, times = c(50, 5, 30, 0, 15))
  bins <- fit_cat(codes, K = 5, T = 3)

  expect_equal(bins$n_bins, 3L)
  expect_equal(as.integer(bins$features[[1]]$map), c(1L, 3L, 2L, 3L, 3L))
  expect_equal(bins$features[[1]]$other_share, 0.2)
  expect_equal(bin_of(c(1, 3, 5, 2, 4), bins), c(1L, 2L, 3L, 3L, 3L))
})

test_that("bin categorical features: levels absent from training go to Others", {
  codes <- c(rep(1, 6), rep(2, 4))          # levels 3, 4, 5 never seen
  bins <- fit_cat(codes, K = 5, T = 4)

  expect_equal(bins$n_bins, 3L)             # 2 kept levels + Others
  expect_equal(bin_of(c(1, 2, 4, 5), bins), c(1L, 2L, 3L, 3L))
})









test_that("bin mixed data: shape, type, ranges and no empty bins in training", {
  set.seed(1)
  x <- data.frame(
    a = rnorm(200),                         # numerical, many values
    b = sample(0:1, 200, replace = TRUE),   # numerical, two values
    c = sample(1:5, 200, replace = TRUE)    # categorical, K = 5 > T
  )
  bins <- get_bins(x, meta(2, cat_dims = 5), T = 4)
  out <- apply_bins(x, bins)

  expect_s3_class(bins, "metadata_for_binning")
  expect_equal(bins$n_bins, c(4L, 2L, 4L))  # 4 quantile bins, 2 values, 3 levels + Others
  expect_equal(dim(out), c(200L, 3L))
  expect_true(is.integer(out))
  for (j in 1:3) {
    expect_equal(sort(unique(out[, j])), seq_len(bins$n_bins[j]))
  }
})





test_that("get_bins rejects invalid input", {
  x <- matrix(c(1, 2, 3, 4), ncol = 1)

  expect_error(get_bins(x, meta(1), T = 1), "T must be a number >= 2")
  expect_error(get_bins(x, meta(1), handle_categorical = "none"), "top_frequent")
  expect_error(get_bins(cbind(x, x), meta(1)), "number of col")

  x_na <- x
  x_na[2] <- NA
  expect_error(get_bins(x_na, meta(1)), "NAs")

  # categorical codes must be in 1..K
  expect_error(get_bins(matrix(c(1, 2, 6), ncol = 1), meta(0, 5)), "categorical codes")
  expect_error(get_bins(matrix(c(0, 1, 2), ncol = 1), meta(0, 5)), "categorical codes")
})

test_that("apply_bins handles NULL and rejects invalid input", {
  bins <- fit_num(1:8, T = 4)

  expect_null(apply_bins(NULL, bins))                      # no validation set
  expect_error(apply_bins(cbind(1:3, 1:3), bins), "columns")
  expect_error(apply_bins(matrix(c(1, NA), ncol = 1), bins), "NAs")

  bins_cat <- fit_cat(c(1, 2, 3), K = 3, T = 10)
  expect_error(apply_bins(matrix(4, ncol = 1), bins_cat), "categorical codes")
})





