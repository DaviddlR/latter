


# TODO: Export las funciones de pérdida ????




subtab_loss <- torch::nn_module(
  name = "subtab_loss",

  initialize = function(num_cont, cat_dims, temperature = 0.5,
                        reconstruction_weight = 1.0, contrastive_weight = 1.0, distance_weight = 1.0) {

    self$num_cont <- num_cont
    self$cat_dims <- cat_dims
    self$num_cat <- length(cat_dims)
    self$reconstruction_weight <- reconstruction_weight
    self$contrastive_weight <- contrastive_weight
    self$distance_weight <- distance_weight

    self$contrastive_loss <- nt_xent_loss(temperature = temperature)

  },

  forward = function(input, target) {

    x_original <- target$x          # Clean batch (before views and noise)  [B, d]
    n_views <- target$n_views
    batch_size <- x_original$size(1)


    total_loss <- torch::torch_tensor(0.0, device = x_original$device)


    # 1. Reconstruction loss

    if (self$reconstruction_weight > 0) {
      # We have to compare the reconstructed version of each view with the whole original sample.

      # The rows of the output are ordered by view [view 1 reconstruction, view 2 reconstruction, ... , view N reconstruction]
      # So we can repeat the original batch n_views times

      x_target <- torch::torch_cat(rep(list(x_original), n_views), dim = 1)

      # All views have B rows
      reconstruction_loss <- feature_reconstruction_loss(
        predicted_num_features = input$cont_estimation,
        predicted_cat_features = input$cat_estimation,
        x_target = x_target,
        num_cont = self$num_cont,
        num_cat = self$num_cat
      )

      # Adjust based on reconstruction weight (TODO: check como lo hace el paper original)
      total_loss <- total_loss + self$reconstruction_weight * reconstruction_loss
    }


    # 2. Contrastive and distance losses
    if (self$contrastive_weight > 0 || self$distance_weight > 0) {

      # If there is only one view, alert and error
      # TODO: the original subtab copy the view to create another one. Decide.
      if (n_views < 2) {
        stop("subtab loss: contrastive and distance losses require at least two views. Use n_views >= 2 or set contrastive and distance weights to 0.")
      }

      projection <- input$projection  # [n_views * batch size, hidden_dim]
      projection <- projection$reshape(c(n_views, batch_size, -1))

      contrastive_loss <- torch::torch_tensor(0.0, device = x_original$device)
      distance_loss <- torch::torch_tensor(0.0, device = x_original$device)

      for (a in 1:(n_views - 1)) {
        for (b in (a + 1):n_views) {  # Every pair (a, b) with a < b: n_views * (n_views - 1) / 2 pairs

          z_a <- projection[a, , ]  # [batch_size, h]
          z_b <- projection[b, , ]

          if (self$contrastive_weight > 0) {
            contrastive_loss <- contrastive_loss + self$contrastive_loss(list(z_a, z_b), NULL)
          }

          if (self$distance_weight > 0) {
            distance_loss <- distance_loss + torch::nnf_mse_loss(z_a, z_b)
          }
        }
      }

      n_pairs <- n_views * (n_views - 1) / 2


      total_loss <- total_loss + self$contrastive_weight * (contrastive_loss / n_pairs) + self$distance_weight * (distance_loss / n_pairs)

    }


    # Return total loss
    return(total_loss)

  }


)






feature_reconstruction_loss <- function(predicted_num_features, predicted_cat_features, x_target, num_cont, num_cat) {

  reconstruction_loss <- torch::torch_tensor(0.0, device = x_target$device)

  # Numerical variables
  if (num_cont > 0) {
    if (is.null(predicted_num_features)) stop("feature_reconstruction_loss: num_cont > 0 but no 'predicted_num_features' output.")
    x_cont_target <- x_target[, 1:num_cont, drop = FALSE]
    num_loss <- torch::nnf_mse_loss(predicted_num_features, x_cont_target)
    reconstruction_loss <- reconstruction_loss + num_loss
  }

  # Categorical variables
  if (num_cat > 0) {
    if (is.null(predicted_cat_features)) stop("feature_reconstruction_loss: num_cat > 0 but no 'predicted_cat_features' output.")
    cat_loss <- torch::torch_tensor(0.0, device = x_target$device)

    for (i in seq_len(num_cat)) {
      cat_target <- x_target[, num_cont + i]$to(dtype = torch::torch_long())  # Level of the column (1-based)
      logits <- predicted_cat_features[[i]]

      cat_loss <- cat_loss + torch::nnf_cross_entropy(logits, cat_target)
    }

    reconstruction_loss <- reconstruction_loss + cat_loss
  }

  return(reconstruction_loss)
}








#' BinXent loss
#'
#' Multi-class cross-entropy of every column, averaged over the d columns and
#' the N samples of the batch.
#'
#' @param logits Output of the wrapper, [batch, d, T].
#' @param target Bin indices, [batch, d], 1-based (output of [apply_bins()]
#'   converted to a tensor).
#'
#' @returns A 'torch::nn_module' representing the Binning loss function.
#' @noRd
binning_loss <- torch::nn_module(
  name = "binning_loss",

  initialize = function() {

  },

  forward = function(input, target) {

    # print("input")
    # print(input$shape)
    # print("target")
    # print(target$shape)
    # print("end target")

    n_classes <- input$size(3)
    loss <- torch::nnf_cross_entropy(
      input = input$reshape(c(-1, n_classes)),
      target = target$to(dtype = torch::torch_long())$reshape(-1)
    )

    return (loss)
  }
)





#' Loss function for the VIME algorithm
#'
#' @param alpha \code{Numeric}. Parameter that controls the balance between the feature loss and mask loss
#' @param num_cont \code{Integer}. Number of continual (or numerical) columns in the dataset
#' @param cat_dims \code{List}. Dimensions of categorical columns.
#'
#' @returns A 'torch::nn_module' representing the VIME loss function.
#' @noRd
vime_loss <- torch::nn_module(
  name = "vime_loss",

  initialize = function(alpha = 1.0, num_cont, cat_dims) {
    # Check si se necesita algún parámetro para ajustar
    self$alpha <- alpha
    self$num_cont <- num_cont
    self$num_cat <- length(cat_dims)

  },

  forward = function(input, target) {

    # Extract target
    mask_original <- target$mask
    x_original <- target$x

    # Extract predictions (they come from neural_network_models > vime wrapper)
    predicted_num_features <- input$cont_estimation
    predicted_cat_features <- input$cat_estimation
    predicted_mask <- input$mask_estimation

    # print(predicted_num_features)
    #
    # print(predicted_cat_features)
    #
    # print(predicted_mask)



    # Mask loss: "Mask vector estimator is trained by minimizing the cross-entropy loss"
    mask_loss <- torch::nnf_binary_cross_entropy(predicted_mask, mask_original)


    # Reconstruction loss
    # "Feature vector estimator is trained by minimizing the reconstruction loss" (mean_square_error)

    reconstruction_loss <- feature_reconstruction_loss(predicted_num_features, predicted_cat_features, x_original, self$num_cont, self$num_cat)



    # reconstruction_loss <- torch::torch_tensor(0.0, device = x_original$device)


    # Numerical variables
    # if (self$num_cont > 0) {
    #   if (is.null(predicted_num_features)) stop("vime_loss: num_cont > 0 but no 'cont' output.")
    #   x_cont_target <- x_original[, 1:self$num_cont, drop = FALSE]
    #   num_loss <- torch::nnf_mse_loss(predicted_num_features, x_cont_target)
    #   reconstruction_loss <- reconstruction_loss + num_loss
    # }
    #
    # # Categorical variables
    # if (self$num_cat > 0) {
    #   if (is.null(predicted_cat_features)) stop("vime_loss: num_cat > 0 but no 'cat' output.")
    #   cat_loss <- torch::torch_tensor(0.0, device = x_original$device)
    #
    #   for (i in seq_len(self$num_cat)) {
    #     cat_target <- x_original[, self$num_cont + i]$to(dtype = torch::torch_long())
    #     logits <- predicted_cat_features[[i]]
    #
    #     cat_loss <- cat_loss + torch::nnf_cross_entropy(logits, cat_target)
    #   }
    #
    #   reconstruction_loss <- reconstruction_loss + cat_loss
    # }


    # "Encoder is trained by minimizing the weighted sum of both losses" (alpha parameter)
    final_loss <- mask_loss + self$alpha * reconstruction_loss
    return(final_loss)

  }
)






#' Contrastive loss nt_xent_loss
#'
#' @param temperature \code{Numerical}. Controls how sharply the model discriminates between hard and easy negative examples
#'
#' @returns A 'torch::nn_module' representing the nt_xent_loss function.
#' @noRd
nt_xent_loss <- torch::nn_module(

  name = "nt_xent_loss",

  initialize = function(temperature = 0.5) {
    self$temperature = temperature
  },

  forward = function(input, target) {

    # Get z_i and z_j
    z_i <- input[[1]]
    z_j <- input[[2]]


    current_batch_size = z_i$size(1)



    # Concatenate
    z <- torch::torch_cat(list(z_i, z_j), dim=1) # When normalized, dot product is equivalent to cosine similarity, so we can use matrix multiplication to compute the similarity between all pairs of embeddings in the batch. The resulting sim_matrix will have shape [2 * batch_size, 2 * batch_size], where sim_matrix[i, j] is the similarity between the i-th and j-th embeddings in the concatenated batch.

    # Normalize
    z <- torch::nnf_normalize(z, dim=2)

    # Similarity matrix
    sim_matrix <- torch::torch_matmul(z, z$t())
    sim_matrix <- sim_matrix / self$temperature



    # Mask to avoid self-comparisons
    mask <- torch::torch_eye(2 * current_batch_size, dtype = torch::torch_bool(), device = z_i$device)  # Ones in the diagonal and zeros elsewhere


    # Positive pairs
    diag_B <- torch::torch_diagonal(sim_matrix, offset = current_batch_size)
    diag_C <- torch::torch_diagonal(sim_matrix, offset = -current_batch_size)


    pos_pairs <- torch::torch_cat(list(diag_B, diag_C))
    pos_pairs <- torch::torch_exp(pos_pairs)


    # Remaining samples
    sim_matrix_exp <- torch::torch_exp(sim_matrix)
    sim_matrix_exp <- sim_matrix_exp * (!mask)
    #sim_matrix_exp <- torch::torch_exp(sim_matrix) * (~mask)
    sum_similarity <- sim_matrix_exp$sum(dim=2)

    # Loss
    loss <- (-1) * torch::torch_log(pos_pairs / sum_similarity)
    loss <- loss$mean()

    return(loss)

  }
)




























