
vime_loss <- torch::nn_module(
  name = "vime_loss",

  initialize = function(alpha = 1.0) {
    # Check si se necesita algún parámetro para ajustar
    self$alpha = alpha
  },

  forward = function(input, target) {

    print("############    LOSS REACHED")

    print(input)

    print("##########      TARGET")

    print(target)

    print("###############################################")

    # Extract target (custom_vime_step_callback)
    mask_original <- target$mask
    x_original <- target$x  # This is the original sample after prepare_data, so that it already contains the numerical features first and the categorical features at the end.

    # Extract predictions (vime_wrapper > forward)
    #predicted_features <- input[[1]]


    predicted_num_features <- input$feature_estimation.cont_estimation
    predicted_cat_features <- input$feature_estimation.cat_estimation
    predicted_mask <- input$mask_estimation

    print(predicted_num_features)

    print(predicted_cat_features)

    print(predicted_mask)

    print("###############################################")

    # Reorder the original sample so that it matches the same order as the prediction (numerical features first, categorical last)
    all_feature_tensors <- c(list(predicted_num_features), predicted_cat_features)
    predicted_features_ordered <- torch::torch_cat(all_feature_tensors, dim = 2)
    #predicted_features_ordered <- torch::torch_cat(list(predicted_num_features, predicted_cat_features), dim = 1)


    # "Mask vector estimator is trained by minimizing the cross-entropy loss"
    mask_loss <- torch::nnf_binary_cross_entropy_with_logits(predicted_mask, mask_original)

    print("REACHED???????")

    # "Feature vector estimator is trained by minimizing the reconstruction loss" (mean_square_error)
    print(predicted_features_ordered)  # DIMS 189 (verificar cómo se forman esas 189)

    print(x_original)  # DIMS 34 (36 menos las dos de exclude columns)

    loss_reconstruction <- torch::nnf_mse_loss(predicted_features_ordered, x_original)

    print("REACHED 2 (aquí falla David del viernes, losses > vime_loss) ???????")

    # "Encoder is trained by minimizing the weighted sum of both losses" (alpha parameter)
    final_loss <- loss_mask + alpha * loss_reconstruction

    print("############    LOSS ENDED")
    return(final_loss)

  }
)





#' Contrastive loss nt_xent_loss
#'
#' @param temperature Controls how sharply the model discriminates between hard and easy negative examples
#'
#' @returns A 'torch::nn_module' representing the nt_xent_loss function.
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




