

#' Basic encoder module.
#'
#' @param in_dim Number of input features.
#' @param hidden_dim Number of hidden or latent features.
#' @param num_hidden Number of blocks of layers of the encoder network.
#' @param dropout Dropout probability.
#'
#' @return A 'torch::nn_module' representing the encoder.
#' @export
#'
#' @examples
#' new_model <- scarf_encoder(100, 256, 4, 0.5)
scarf_encoder <- torch::nn_module(

  name = "Scarf encoder",

  # Init
  initialize = function(in_dim, hidden_dim = 256, num_hidden = 4, dropout = 0.0) {

    layers <- list()

    index_layer <- 1

    if (num_hidden > 1){
      for (i in 1:(num_hidden - 1)){

        # Linear layer
        layers[[index_layer]] <- torch::nn_linear(in_dim, hidden_dim)
        index_layer <- index_layer + 1

        # Batch norm layer
        layers[[index_layer]] <- torch::nn_batch_norm1d(hidden_dim)
        index_layer <- index_layer + 1

        # RELU
        layers[[index_layer]] <- torch::nn_relu(inplace=TRUE)
        index_layer <- index_layer + 1

        # Dropout
        layers[[index_layer]] <- torch::nn_dropout(dropout)
        index_layer <- index_layer + 1

        # Update in_dim after first layer
        in_dim <- hidden_dim

      }
    }

    layers[[index_layer]] <- torch::nn_linear(in_dim, hidden_dim)

    self$encoder <- do.call(torch::nn_sequential, layers)

  },

  # Forward pass
  forward = function(x) {
    self$encoder(x)
  }

)









