


# TODO: Export las funciones de pérdida ????




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

    # Extract predictions
    predicted_num_features <- input$feature_estimation.cont_estimation
    predicted_cat_features <- input$feature_estimation.cat_estimation
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

    reconstruction_loss <- torch::torch_tensor(0.0, device = x_original$device)


    # Numerical variables
    if (self$num_cont > 0 && !is.null(predicted_num_features)) {
      x_cont_target <- x_original[, 1:self$num_cont, drop = FALSE]
      num_loss <- torch::nnf_mse_loss(predicted_num_features, x_cont_target)
      reconstruction_loss <- reconstruction_loss + loss_num
    }

    # Categorical variables
    if (self$num_cat > 0 && !is.null(predicted_cat_features)) {
      cat_loss <- torch::torch_tensor(0.0, device = x_original$device)

      for (i in seq_len(self$num_cat)) {
        cat_target <- x_original[, self$num_cont + i]$to(dtype = torch::torch_long())
        logits <- predicted_cat_features[[i]]

        cat_loss <- cat_loss + torch::nnf_cross_entropy(logits, cat_target)
      }

      reconstruction_loss <- reconstruction_loss + cat_loss
    }


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




