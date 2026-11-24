# ---------- Helpers ----------

as_float_tensor <- function(m) torch::torch_tensor(m, dtype = torch::torch_float32())

# Output of subtab_wrapper that reconstructs x PERFECTLY. The rows are tiled view by view with rbind(),
# independently of the tiling done inside the loss
make_perfect_prediction <- function(x, n_views, projection_dim = 8) {
  tile <- function(m) do.call(rbind, rep(list(m), n_views))

  list(projection = torch::torch_randn(n_views * nrow(x), projection_dim),
       cont_estimation = as_float_tensor(tile(x)))
}

# Output of subtab_wrapper with random values
make_random_prediction <- function(batch, num_cont, n_views, projection_dim = 8) {
  list(projection = torch::torch_randn(n_views * batch, projection_dim),
       cont_estimation = torch::torch_randn(n_views * batch, num_cont))
}

make_target <- function(x, n_views) list(x = as_float_tensor(x), n_views = n_views)

# Projections used to compare with the OFFICIAL implementation (loss_functions.py). The expected values below were computed
# by running the official formulas (masks of quadrants, positives on the +-B diagonals, cross-entropy with label 0) on these matrices
z_i_reference <- matrix(c(1.0, 0.0,
                          0.0, 1.0,
                          0.5, 0.5), nrow = 3, byrow = TRUE)
z_j_reference <- matrix(c(0.9, 0.1,
                          0.2, 0.8,
                          -0.5, 0.5), nrow = 3, byrow = TRUE)

# ---------- Contrastive loss: same numbers as the official implementation ----------

test_that("contrastive loss gives the same values as the official implementation (dot product)", {
  skip_if_not(torch::torch_is_installed())

  zi <- as_float_tensor(z_i_reference)
  zj <- as_float_tensor(z_j_reference)

  expect_equal(subtab_contrastive_loss(zi, zj, temperature = 0.1)$item(), 1.9529816461, tolerance = 1e-5)
  expect_equal(subtab_contrastive_loss(zi, zj, temperature = 0.5)$item(), 1.2576367229, tolerance = 1e-5)
})

test_that("contrastive loss is symmetric in the two views", {
  skip_if_not(torch::torch_is_installed())

  zi <- torch::torch_randn(8, 6)
  zj <- torch::torch_randn(8, 6)

  expect_equal(subtab_contrastive_loss(zi, zj)$item(), subtab_contrastive_loss(zj, zi)$item(), tolerance = 1e-5)
})

test_that("contrastive loss is much lower when the two views of each sample are aligned (normalized projections)", {
  skip_if_not(torch::torch_is_installed())

  normalized <- function(n) torch::nnf_normalize(torch::torch_randn(n, 8), p = 2, dim = 2)

  z <- normalized(16)
  aligned <- subtab_contrastive_loss(z, z$clone(), temperature = 0.1)$item()
  random <- subtab_contrastive_loss(normalized(16), normalized(16), temperature = 0.1)$item()

  expect_lt(aligned + 2, random)
})

test_that("contrastive loss is differentiable (no NaN from the -Inf of the diagonal)", {
  skip_if_not(torch::torch_is_installed())

  zi <- torch::torch_randn(8, 6, requires_grad = TRUE)
  zj <- torch::torch_randn(8, 6)

  subtab_contrastive_loss(zi, zj)$backward()
  gradient <- torch::as_array(zi$grad)

  expect_false(anyNA(gradient))
  expect_gt(sum(abs(gradient)), 0)
})

# ---------- Reconstruction term (official convention: sum over columns, mean over rows) ----------

test_that("reconstruction uses the SUM over columns divided by the rows, as the official getMSEloss", {
  skip_if_not(torch::torch_is_installed())

  recon <- matrix(c(1, 2, 3,
                    0, 0, 1), nrow = 2, byrow = TRUE)
  target <- matrix(c(0, 2, 5,
                     1, 1, 1), nrow = 2, byrow = TRUE)

  # Squared errors: row 1 = 1 + 0 + 4 = 5, row 2 = 1 + 1 + 0 = 2. Sum 7 / 2 rows = 3.5 (the mean over elements would be 7 / 6)
  loss <- subtab_feature_reconstruction_loss(as_float_tensor(recon), as_float_tensor(target), num_cont = 3)
  expect_equal(loss$item(), 3.5, tolerance = 1e-6)

  # The same number through the whole loss with a single view and no contrastive/distance terms
  pred <- list(projection = torch::torch_randn(2, 4), cont_estimation = as_float_tensor(recon))
  loss_fn <- subtab_loss(num_cont = 3, use_contrastive_loss = FALSE, use_distance_loss = FALSE)
  expect_equal(loss_fn(pred, make_target(target, 1))$item(), 3.5, tolerance = 1e-6)
})

test_that("reconstruction is ~0 when every view reconstructs the whole original sample", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(16 * 12), nrow = 16, ncol = 12)
  pred <- make_perfect_prediction(x, n_views = 4)

  loss_fn <- subtab_loss(num_cont = 12, use_contrastive_loss = FALSE, use_distance_loss = FALSE)
  expect_lt(loss_fn(pred, make_target(x, 4))$item(), 1e-4)
})

test_that("reconstruction is clearly positive for random predictions", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(16 * 12), nrow = 16, ncol = 12)
  pred <- make_random_prediction(16, 12, n_views = 4)

  loss_fn <- subtab_loss(num_cont = 12, use_contrastive_loss = FALSE, use_distance_loss = FALSE)
  expect_gt(loss_fn(pred, make_target(x, 4))$item(), 1)
})

test_that("the rows of the target follow the view-by-view order (an interleaved order is detected)", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(16 * 6), nrow = 16, ncol = 6)
  n_views <- 3
  loss_fn <- subtab_loss(num_cont = 6, use_contrastive_loss = FALSE, use_distance_loss = FALSE)

  # Correct order: [x; x; x]
  correct <- make_perfect_prediction(x, n_views = n_views)
  expect_lt(loss_fn(correct, make_target(x, n_views))$item(), 1e-4)

  # Wrong order: each sample repeated n_views times in a row
  interleaved <- correct
  interleaved$cont_estimation <- as_float_tensor(x[rep(seq_len(nrow(x)), each = n_views), ])
  expect_gt(loss_fn(interleaved, make_target(x, n_views))$item(), 0.1)
})

test_that("reconstruction works with a single view", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(10 * 4), nrow = 10, ncol = 4)
  loss_fn <- subtab_loss(num_cont = 4, use_contrastive_loss = FALSE, use_distance_loss = FALSE)

  expect_lt(loss_fn(make_perfect_prediction(x, n_views = 1), make_target(x, 1))$item(), 1e-4)
})

# ---------- Distance term ----------

test_that("distance loss is the official getMSEloss(zi, zj): sum of squares / samples", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(3 * 2), nrow = 3, ncol = 2)
  pred <- make_perfect_prediction(x, n_views = 2, projection_dim = 2)
  pred$projection <- torch::torch_cat(list(as_float_tensor(z_i_reference), as_float_tensor(z_j_reference)), dim = 1)

  loss_fn <- subtab_loss(num_cont = 2, use_contrastive_loss = FALSE, use_distance_loss = TRUE)
  total <- loss_fn(pred, make_target(x, 2))$item()

  # z_i - z_j: (0.1, -0.1), (-0.2, 0.2), (1.0, 0.0) -> 0.02 + 0.08 + 1.0 = 1.1, divided by 3 samples = 0.3666667. Reconstruction is ~0
  expect_equal(total, 1.1 / 3, tolerance = 1e-5)
  expect_equal(loss_fn$components$distance$item(), 1.1 / 3, tolerance = 1e-5)
})

test_that("distance loss is 0 when all the views have the same projection and positive otherwise", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(16 * 5), nrow = 16, ncol = 5)
  target <- make_target(x, 3)
  loss_fn <- subtab_loss(num_cont = 5, use_contrastive_loss = FALSE, use_distance_loss = TRUE)

  z <- torch::torch_randn(16, 8)
  same <- make_perfect_prediction(x, n_views = 3)
  same$projection <- torch::torch_cat(list(z, z, z), dim = 1)
  loss_fn(same, target)
  expect_lt(loss_fn$components$distance$item(), 1e-6)

  different <- make_perfect_prediction(x, n_views = 3)
  loss_fn(different, target)
  expect_gt(loss_fn$components$distance$item(), 0.1)
})

# ---------- Total loss ----------

test_that("official defaults: contrastive and distance on, tau = 0.1", {
  skip_if_not(torch::torch_is_installed())

  loss_fn <- subtab_loss(num_cont = 5)

  expect_true(loss_fn$use_contrastive_loss)
  expect_true(loss_fn$use_distance_loss)
  expect_equal(loss_fn$temperature, 0.1)
})

test_that("the total is reconstruction + mean contrastive over pairs + mean distance over pairs", {
  skip_if_not(torch::torch_is_installed())

  batch <- 8
  n_views <- 3
  x <- matrix(rnorm(batch * 5), nrow = batch, ncol = 5)
  pred <- make_random_prediction(batch, 5, n_views = n_views, projection_dim = 6)

  loss_fn <- subtab_loss(num_cont = 5)
  total <- loss_fn(pred, make_target(x, n_views))$item()
  components <- loss_fn$components

  expect_equal(total, components$reconstruction$item() + components$contrastive$item() + components$distance$item(), tolerance = 1e-5)

  # Averages over the 3 pairs (1, 2), (1, 3), (2, 3), computed by hand
  z <- pred$projection$reshape(c(n_views, batch, -1))
  pairs <- list(c(1, 2), c(1, 3), c(2, 3))

  manual_contrastive <- mean(sapply(pairs, function(p) subtab_contrastive_loss(z[p[1], , ], z[p[2], , ], 0.1)$item()))
  manual_distance <- mean(sapply(pairs, function(p) ((z[p[1], , ] - z[p[2], , ])$pow(2)$sum() / batch)$item()))

  expect_equal(components$contrastive$item(), manual_contrastive, tolerance = 1e-5)
  expect_equal(components$distance$item(), manual_distance, tolerance = 1e-5)
})

test_that("a disabled term is not computed and its component is NULL", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(8 * 5), nrow = 8, ncol = 5)
  pred <- make_random_prediction(8, 5, n_views = 3)
  target <- make_target(x, 3)

  only_distance <- subtab_loss(num_cont = 5, use_contrastive_loss = FALSE, use_distance_loss = TRUE)
  total <- only_distance(pred, target)$item()
  expect_null(only_distance$components$contrastive)
  expect_equal(total, only_distance$components$reconstruction$item() + only_distance$components$distance$item(), tolerance = 1e-5)

  only_reconstruction <- subtab_loss(num_cont = 5, use_contrastive_loss = FALSE, use_distance_loss = FALSE)
  only_reconstruction(pred, target)
  expect_null(only_reconstruction$components$contrastive)
  expect_null(only_reconstruction$components$distance)
})

test_that("contrastive and distance losses need at least 2 views, reconstruction alone does not", {
  skip_if_not(torch::torch_is_installed())

  x <- matrix(rnorm(10 * 5), nrow = 10, ncol = 5)
  pred <- make_random_prediction(10, 5, n_views = 1)
  target <- make_target(x, 1)

  expect_error(subtab_loss(num_cont = 5)(pred, target), "at least 2 views")
  expect_error(subtab_loss(num_cont = 5, use_contrastive_loss = FALSE, use_distance_loss = FALSE)(pred, target), NA)
})

# ---------- Integration: crop -> noise -> wrapper -> loss ----------

test_that("the whole pipeline gives a finite scalar and the gradient reaches the shared encoder", {
  skip_if_not(torch::torch_is_installed())

  d <- 8
  batch <- 8
  x <- matrix(rnorm(batch * d), nrow = batch, ncol = d)

  views <- make_subtab_views(d = d, n_views = 2, overlap = 0.5)  # Views 1:6 and 3:8, both of width 6
  idx <- subtab_view_indices(views)
  eligible <- subtab_noise_eligibility(idx, subtab_detect_binary_columns(x), noise_on_binary = FALSE)

  x_tensor <- as_float_tensor(x)
  noisy_views <- subtab_add_noise(subtab_crop_views(x_tensor, idx), n_views = nrow(idx), noise_type = "swap_noise",
                                  masking_ratio = 0.3, eligible = eligible)

  wrapper <- subtab_wrapper(num_cont = d, view_width = ncol(idx), hidden_dims = c(16, 16))
  loss_fn <- subtab_loss(num_cont = d)

  # Input: the NOISY views. Target: the CLEAN original batch and the number of views
  loss <- loss_fn(wrapper(list(x = noisy_views)), list(x = x_tensor, n_views = nrow(idx)))

  expect_equal(loss$numel(), 1)
  expect_true(is.finite(loss$item()))

  loss$backward()
  expect_gt(as.numeric(wrapper$main_encoder$encoder[[1]]$weight$grad$abs()$sum()), 0)
})
