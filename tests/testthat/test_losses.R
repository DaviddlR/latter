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





