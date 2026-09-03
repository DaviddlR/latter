
vime_loss <- torch::nn_module(
  name = "vime_loss",

  initialize = function() {
    # Check si se necesita algún parámetro para ajustar
  },

  forward = function(input, target) {
    # TODO



    # Las variables originales están desordenadas con respecto a lo que viene en target (numerical primero y categorical despues)
    # Habrá que reorganizar las de original antes de hacer la loss (check utils > encode_categorical_data)
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




