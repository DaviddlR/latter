



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







#' Basic encoder module. This is the official implementation of VIME
#'
#' @param num_cont Number of numerical features.
#' @param cat_dims Number of dimensions per categorical column.
#'
#' @returns A 'torch::nn_module' representing the encoder.
vime_encoder <- torch::nn_module(
  name = "vime_basic_encoder",

  initialize = function(num_cont, cat_dims) {
    self$num_cont <- num_cont
    self$cat_dims <- cat_dims

    self$num_cat <- length(cat_dims)


    # Embeddings for categorical columns
    if (self$cat_dims > 0) {
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

    self$output_dim <- total_in_dim


    self$encoder <- torch::nn_sequential(
      torch::nn_linear(total_in_dim, total_in_dim),
      torch::nn_relu(inplace=TRUE)
    )


  },


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




# VIME mask estimator
# It measures the probability that each column has been corrupted. If predicted probability = 1, it means that the model believes
# that the variable has been corrupted.
# Input dimensions: output of the encoder
# Output dimensions: number of columns of the original sample
vime_mask_estimator <- torch::nn_module(

  initialize = function(in_dim, num_cont, cat_dims) {

    self$in_dim <- in_dim  # Dimensions of the output of the encoder
    self$num_cont <- num_cont
    self$num_cat <- length(cat_dims)

    self$number_of_original_variables = self$num_cont + self$num_cat

    self_mask_estimator = torch::nn_sequential(
      torch::nn_linear(in_dim, self$number_of_original_variables)
    )
  },

  forward = function(z) {
    mask_pred <- torch::torch_sigmoid(self$mask_estimator(z))
  }
)


# VIME feature estimator
# It reconstructs the original, actual values that the sample had before being corrupted
# Input dimensions: output of the encoder
# Output dimensions: embedding logits of each category.
vime_feature_estimator <- torch::nn_module(
  initialize = function(in_dim, num_cont, cat_dims) {

    self$in_dim <- in_dim  # Dimensions of the output of the encoder
    self$num_cont <- num_cont
    self$num_cat <- length(cat_dims)
    self$cat_dims <- cat_dims

    # Numerical features
    if (self$num_cont > 0) {
      self$numerical_feature_estimator = torch::nn_sequential(
        torch::nn_linear(self$in_dim, self$num_cont)
      )
    }


    # Categorical features
    if (self$num_cat > 0) {
      self$categorical_feature_estimator <- torch::nn_module_list(
        lapply(cat_dims, function(dim_k) {
          torch::nn_linear(in_dim, dim_k)
        })
      )
    }

  },

  forward = function(z) {

    # Continual features prediction
    cont_estimation <- if (self$num_cont > 0) self$numerical_feature_estimator(z) else NULL

    # Categorical features prediction
    cat_estimation <- list()

    if (self$num_cat > 0) {
      for (i in seq_along(self$cat_dims)) {
        cat_estimation[[i]] <- self$categorical_feature_estimator[[i]](z)
      }
    }

    return(list(
      cont_estimation = cont_estimation,
      cat_estimation = cat_estimation
    ))
  }

)



# Wrapper
vime_wrapper <- torch::nn_module(

  initialize = function(num_cont, cat_dims) {

    # Encoder
    self$vime_encoder <- vime_encoder(num_cont, cat_dims)
    output_dimension <- self$vime_encoder$output_dim

    # Prediction heads
    self$feature_estimator <- vime_feature_estimator(output_dimension, num_cont, cat_dims)
    self$mask_estimator <- vime_mask_estimator(output_dimension, num_cont, cat_dims)
  },


  forward = function(x_input) {  # Input comes from the luz callback

    # Take original and corrupted sample (see callback VIME)
    x_corrupted <- x_input

    # Encode it using encoder and projection head
    x_corrupted_encoded <- self$vime_encoder(x_corrupted)

    # Get feature estimation and mask estimation (act like projection heads, so we do not need an additional one)
    feature_estimation <- self$feature_estimator(x_corrupted_encoder)  # Cont estimation and cat estimation
    mask_estimation <- self$mask_estimator(x_corrupted_encoder)  # mask prediction

    result <- c(feature_estimation, mask_estimation)
  }
)













#' Basic encoder module. This is the official implementation of SCARF
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








