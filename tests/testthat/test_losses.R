
# ---------- Helpers ----------

# Fake preprocessed batch (numerical first, categorical last). Same as in the other subtab tests
make_fake_batch <- function(n, num_cont, cat_dims) {
  cont <- matrix(rnorm(n * num_cont), nrow = n, ncol = num_cont)
  if (length(cat_dims) == 0) return(cont)
  cats <- sapply(cat_dims, function(k) sample.int(k, n, replace = TRUE))
  cbind(cont, cats)
}

as_float_tensor <- function(m) torch::torch_tensor(m, dtype = torch::torch_float32())

# Output of subtab_wrapper that reconstructs x PERFECTLY. The rows are tiled view by view with rbind(),
# independently of the tiling done inside the loss
make_perfect_prediction <- function(x, num_cont, cat_dims, n_views, projection_dim = 8) {
  tile <- function(m) do.call(rbind, rep(list(m), n_views))

  cont <- if (num_cont > 0) as_float_tensor(tile(x[, 1:num_cont, drop = FALSE])) else NULL

  cats <- NULL
  if (length(cat_dims) > 0) {
    cats <- lapply(seq_along(cat_dims), function(i) {
      level_values <- tile(x[, num_cont + i, drop = FALSE])[, 1]
      logits <- matrix(0, nrow = length(level_values), ncol = cat_dims[i] + 2)
      logits[cbind(seq_along(level_values), level_values)] <- 50  # Huge logit on the right level
      as_float_tensor(logits)
    })
  }

  list(projection = torch::torch_randn(n_views * nrow(x), projection_dim), cont_estimation = cont, cat_estimation = cats)
}

# Output of subtab_wrapper with random values
make_random_prediction <- function(batch, num_cont, cat_dims, n_views, projection_dim = 8) {
  list(
    projection = torch::torch_randn(n_views * batch, projection_dim),
    cont_estimation = if (num_cont > 0) torch::torch_randn(n_views * batch, num_cont) else NULL,
    cat_estimation = if (length(cat_dims) > 0) lapply(cat_dims, function(k) torch::torch_randn(n_views * batch, k + 2)) else NULL
  )
}

make_target <- function(x, n_views) list(x = as_float_tensor(x), n_views = n_views)

# ---------- Reconstruction term ----------

test_that("reconstruction loss is ~0 when every view reconstructs the whole original sample", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(16, 3, c(4, 3))
  pred <- make_perfect_prediction(x, 3, c(4, 3), n_views = 4)

  loss_fn <- subtab_loss(num_cont = 3, cat_dims = c(4, 3), contrastive_weight = 0, distance_weight = 0)
  expect_lt(loss_fn(pred, make_target(x, 4))$item(), 1e-3)
})



test_that("reconstruction loss is clearly positive for random predictions", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(16, 3, c(4, 3))
  pred <- make_random_prediction(16, 3, c(4, 3), n_views = 4)

  loss_fn <- subtab_loss(num_cont = 3, cat_dims = c(4, 3), contrastive_weight = 0, distance_weight = 0)
  expect_gt(loss_fn(pred, make_target(x, 4))$item(), 1)
})





test_that("weights scale their term and a weight of 0 removes it", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(16, 3, c(4, 3))
  pred <- make_random_prediction(16, 3, c(4, 3), n_views = 3)
  target <- make_target(x, 3)

  only <- function(r, c, d) subtab_loss(3, c(4, 3), reconstruction_weight = r, contrastive_weight = c, distance_weight = d)(pred, target)$item()

  expect_equal(only(3, 0, 0), 3 * only(1, 0, 0), tolerance = 1e-5)
  expect_equal(only(1, 2, 5), only(1, 0, 0) + only(0, 2, 0) + only(0, 0, 5), tolerance = 1e-5)  # The total is the weighted sum
  expect_equal(only(0, 0, 0), 0)
})




test_that("reconstruction works with only numerical or only categorical columns", {
  skip_if_not(torch::torch_is_installed())

  x_num <- make_fake_batch(10, 4, integer(0))
  pred_num <- make_perfect_prediction(x_num, 4, integer(0), n_views = 2)
  loss_num <- subtab_loss(4, integer(0), contrastive_weight = 0, distance_weight = 0)
  expect_lt(loss_num(pred_num, make_target(x_num, 2))$item(), 1e-3)

  x_cat <- make_fake_batch(10, 0, c(4, 3))
  pred_cat <- make_perfect_prediction(x_cat, 0, c(4, 3), n_views = 2)
  loss_cat <- subtab_loss(0, c(4, 3), contrastive_weight = 0, distance_weight = 0)
  expect_lt(loss_cat(pred_cat, make_target(x_cat, 2))$item(), 1e-3)
})


test_that("reconstruction loss with a single view is allowed", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(10, 3, c(4, 3))
  pred <- make_perfect_prediction(x, 3, c(4, 3), n_views = 1)

  loss_fn <- subtab_loss(3, c(4, 3), contrastive_weight = 0, distance_weight = 0)
  expect_lt(loss_fn(pred, make_target(x, 1))$item(), 1e-3)
})


test_that("distance loss is 0 when all the views have the same projection and positive otherwise", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(16, 3, c(4, 3))
  target <- make_target(x, 3)
  loss_fn <- subtab_loss(3, c(4, 3), reconstruction_weight = 0, contrastive_weight = 0, distance_weight = 1)

  # The same projection repeated view by view
  z <- torch::torch_randn(16, 8)
  pred_same <- make_random_prediction(16, 3, c(4, 3), n_views = 3)
  pred_same$projection <- torch::torch_cat(list(z, z, z), dim = 1)

  pred_different <- make_random_prediction(16, 3, c(4, 3), n_views = 3)

  expect_lt(loss_fn(pred_same, target)$item(), 1e-6)
  expect_gt(loss_fn(pred_different, target)$item(), 0.1)
})


test_that("contrastive loss is lower when the views of the same sample have aligned projections", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(16, 3, c(4, 3))
  target <- make_target(x, 3)
  loss_fn <- subtab_loss(3, c(4, 3), reconstruction_weight = 0, contrastive_weight = 1, distance_weight = 0)

  z <- torch::torch_randn(16, 8)
  pred_aligned <- make_random_prediction(16, 3, c(4, 3), n_views = 3)
  pred_aligned$projection <- torch::torch_cat(list(z, z, z), dim = 1)

  pred_random <- make_random_prediction(16, 3, c(4, 3), n_views = 3)

  expect_lt(loss_fn(pred_aligned, target)$item() + 0.5, loss_fn(pred_random, target)$item())
})


test_that("contrastive and distance losses need at least 2 views, reconstruction alone does not", {
  skip_if_not(torch::torch_is_installed())

  x <- make_fake_batch(10, 3, c(4, 3))
  pred <- make_random_prediction(10, 3, c(4, 3), n_views = 1)
  target <- make_target(x, 1)

  expect_error(subtab_loss(3, c(4, 3))(pred, target), "subtab loss: contrastive and distance losses require at least two views. Use n_views >= 2 or set contrastive and distance weights to 0.")
  expect_error(subtab_loss(3, c(4, 3), contrastive_weight = 0, distance_weight = 0)(pred, target), NA)
})



# ---------- Integration: views -> apply_view_mask -> wrapper -> loss ----------

test_that("the whole pipeline gives a finite scalar and the gradient reaches the shared encoder", {
  skip_if_not(torch::torch_is_installed())

  num_cont <- 3
  cat_dims <- c(4, 3)
  batch <- 8
  n_views <- 3

  x <- make_fake_batch(batch, num_cont, cat_dims)
  views <- make_subtab_views_official(d = 5, n_views = n_views, overlap = 0.5)
  applied <- apply_view_mask(x, views, num_cont = num_cont, cat_dims = cat_dims)

  wrapper <- subtab_wrapper(num_cont = num_cont, cat_dims = cat_dims, hidden_dim = 16, head_hidden_dim = 8)
  loss_fn <- subtab_loss(num_cont = num_cont, cat_dims = cat_dims)

  # Target: the ORIGINAL batch (not the masked one) and the number of views
  loss <- loss_fn(wrapper(applied), list(x = as_float_tensor(x), n_views = applied$n_views))

  expect_equal(loss$numel(), 1)  # scalar
  expect_true(is.finite(loss$item()))

  loss$backward()
  expect_gt(as.numeric(wrapper$main_encoder$encoder[[1]]$weight$grad$abs()$sum()), 0)
})

















test_that("vime_loss includes the reconstruction term", {
  skip_if_not_installed("torch")

  B <- 8
  num_cont <- 2
  cat_dims <- 3

  x <- torch::torch_cat(list(
    torch::torch_randn(B, num_cont),
    torch::torch_randint(1, 4, size = c(B, 1))$to(dtype = torch::torch_float())
  ), dim = 2)
  mask <- (torch::torch_rand(B, num_cont + 1) < 0.5)$to(dtype = torch::torch_float())

  pred <- list(  # Simulate predictions
    cont_estimation = torch::torch_randn(B, num_cont),
    cat_estimation  = list(torch::torch_randn(B, cat_dims + 2)),
    mask_estimation = torch::torch_rand(B, num_cont + 1)
  )
  target <- list(x = x, mask = mask)

  loss_fn <- vime_loss(alpha = 1, num_cont = num_cont, cat_dims = cat_dims)
  full    <- loss_fn(pred, target)$item()

  mask_only <- torch::nnf_binary_cross_entropy(pred$mask, mask)$item()
  expect_gt(full, mask_only)   # The combined loss is greater than the loss produced by the mask

  # If alpha = 0, only mask loss
  loss0 <- vime_loss(alpha = 0, num_cont = num_cont, cat_dims = cat_dims)
  expect_equal(loss0(pred, target)$item(), mask_only, tolerance = 1e-6)
})





