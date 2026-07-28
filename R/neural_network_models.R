



#' Classification Head torch module.
#'
#' @param input_dim Input dimension
#' @param n_classes Number of classes to classify
#' @param dropout Dropout layer parameter. Default: 0.0 (without dropout)
#'
#' @returns A 'torch::nn_module' representing the classification head.
classifier_network <- torch::nn_module(
  name = "Small MLP",

  # Init

  initialize = function(input_dim, n_classes, dropout = 0.0) {
    self$in_dim = input_dim
    self$dropout = dropout


    self$classifier <- torch::nn_sequential(
      torch::nn_linear(input_dim, 256),
      torch::nn_batch_norm1d(256),
      torch::nn_relu(inplace=TRUE),
      torch::nn_dropout(self$dropout),

      torch::nn_linear(256, n_classes)
    )
  },

  # Forward
  forward = function(x) {
    self$classifier(x)
  }
)







#' Basic encoder module.
#'
#' @param num_cont Number of numerical features.
#' @param cat_dims Number of dimensions per categorical column.
#' @param hidden_dim Number of hidden or latent features.
#' @param num_hidden Number of blocks of layers of the encoder network.
#' @param dropout Dropout probability.
#'
#' @return A 'torch::nn_module' representing the encoder.
scarf_encoder <- torch::nn_module(

  name = "Scarf encoder",

  # Init
  initialize = function(num_cont, cat_dims, hidden_dim = 256, num_hidden = 4, dropout = 0.0) {


    self$num_cont <- num_cont
    self$has_cat <- length(cat_dims) > 0

    # Embeddings for categorical columns
    if (self$has_cat) {
      self$emb_dims <- sapply(cat_dims, function(x) min(50, ceiling((x + 1) / 2)))  # How many dimensions each embedding

      # Embedding module
      self$embeddings <- torch::nn_module_list(  # For each categorical, an embedding layer
        lapply(seq_along(cat_dims), function(i) {
          torch::nn_embedding(
            num_embeddings = cat_dims[i] + 2, # + 2 for out of range
            embedding_dim = self$emb_dims[i]
          )
        })
      )

      total_in_dim <- num_cont + sum(self$emb_dims)
    } else {
      total_in_dim <- num_cont
    }


    layers <- list()

    index_layer <- 1

    if (num_hidden > 1){
      for (i in 1:(num_hidden - 1)){

        # Linear layer
        layers[[index_layer]] <- torch::nn_linear(total_in_dim, hidden_dim)
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
        total_in_dim <- hidden_dim

      }
    }

    layers[[index_layer]] <- torch::nn_linear(total_in_dim, hidden_dim)

    # Main encoder module
    self$encoder <- do.call(torch::nn_sequential, layers)

  },

  # Forward pass
  forward = function(x) {

    # Check if it has categorical values so that it needs embeddings
    if (self$has_cat) {
      x_cont <- x[, 1:self$num_cont, drop = FALSE]  # Numerical are the first columns
      x_cat <- x[, (self$num_cont + 1):ncol(x), drop = FALSE]$to(dtype = torch::torch_long())  # Categorical are at the end

      # Get embeddings
      embedded_list <- list()
      for (i in seq_len(ncol(x_cat))) {
        cat_col <- x_cat[, i]
        embedded_list[[i]] <- self$embeddings[[i]](cat_col)
      }

      # Concatenate
      x_emb <- torch::torch_cat(embedded_list, dim = 2)
      x_prepared <- torch::torch_cat(list(x_cont, x_emb), dim = 2)

    } else {
      x_prepared <- x
    }

    # Forward through the main encoder
    self$encoder(x_prepared)
  }

)




scarf_projection_head <- torch::nn_module(

  name = "Scarf projection head",

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




SCARF_wrapper <- torch::nn_module(  # Something like SCARF lightning but we do not define train_step here
  name = "SCARF wrapper",

  initialize = function(num_cont, cat_dims, hidden_dim, num_hidden, head_hidden_dim, head_num_hidden, dropout = 0.0) {
    self$main_encoder <- scarf_encoder(num_cont = num_cont, cat_dims = cat_dims, hidden_dim = hidden_dim, num_hidden = num_hidden, dropout = dropout)
    self$projection_head <- scarf_projection_head(in_dim = hidden_dim, hidden_dim = head_hidden_dim, num_hidden = head_num_hidden, dropout = dropout)
  },

  forward = function(x_input) {  # Here it comes a list with (original sample, corrupted sample). See luz callback


    # Take original and corrupted sample
    x_original <- x_input[[1]]
    x_corrupted <- x_input[[2]]

    # Encode it using encoder and projection head
    x_original_encoded <- self$main_encoder(x_original)
    x_corrupted_encoded <- self$main_encoder(x_corrupted)

    z_original <- self$projection_head(x_original_encoded)
    z_corrupted <- self$projection_head(x_corrupted_encoded)


    # z_original <- self$projection_head(self$main_encoder(x_original))
    # z_corrupted <- self$projection_head(self$main_encoder(x_corrupted))

    result <- c(z_original, z_corrupted)

  }
)








