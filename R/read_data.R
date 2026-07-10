





#' Prepare data for SCARF feature extraction
#'
#' @param dataframe Dataframe for feature extraction
#' @param trained_recipe Trained recipes::recipe for preprocessing
#' @param exclude_columns A \code{string} of columns that the model should ignore (i.e target or ID columns). Default is \code{NULL}.
#' @param want_labels \code{Boolean}. If \code{TRUE}, the function extracts and returns the target labels alongside features. Default is \code{FALSE}.
#' @param label_column \code{String}. Name of the column containing the labels. Required if \code{want_labels = TRUE}. Default is \code{NULL}.
#' @param preprocess \code{Boolean}. Set if the data need preprocessing steps using 'recipes', such as 'step_normalize' or 'step_dummy'. Default is \code{TRUE}, meaning that this process is automatically done.
#'
#' @returns A torch::matrix representing the dataframe ready for feature extraction
prepare_scarf_data_for_feature_extraction = function(dataframe, trained_recipe, exclude_columns = NULL, want_labels = FALSE, label_column = NULL, preprocess = TRUE) {
  df_extract <- as.data.frame(dataframe)

  # Get label if needed
  label_data = NULL

  if (want_labels) {
    label_data <- df_extract[[label_column]]
  }

  # Remove unnecessary columns
  x_extract <- df_extract[, !(names(df_extract) %in% exclude_columns), drop=FALSE]

  if (preprocess){
    # Apply preprocessing to the dataset using the trained recipe and create matrix
    x_extract_processed <- recipes::bake(trained_recipe, new_data = x_extract)
    x_extract_mat <- as.matrix(x_extract_processed)
  } else {
    # Inside a recipe flow. Preprocessing is already done
    x_extract_mat <- as.matrix(x_extract)
  }

  print("Dataset ready for feature extraction: ")
  print(dim(x_extract_mat))
  print(length(label_data))

  return (list(
    x = x_extract_mat,
    y = label_data
    ))
}








#' Prepare data for SCARF pretraining
#'
#' @param dataframe_train Train dataframe
#' @param exclude_columns Columns that the pretraining model should avoid (i.e target or ID columns)
#' @param create_validation Indicate whether a validation set should be created
#' @param validation_proportion Proportion of the training samples that will be used to create the validation set, if required.
#' @param preprocess \code{Boolean}. Set if the data need preprocessing steps using 'recipes', such as 'step_normalize' or 'step_dummy'. Default is \code{TRUE}, meaning that this process is automatically done.
#'
#' @returns Preprocessed train dataset (and validation set if required) and the recipes::recipe used for preprocessing
#
# @examples
# data(iris)
#
# data_ready <- prepare_scarf_data(dataframe_train = iris, exclude_columns = "Species", create_validation = TRUE)
#
# dim(data_ready$train_set)
prepare_scarf_data = function(dataframe_train, exclude_columns = NULL, create_validation = FALSE, validation_proportion = 0.1, preprocess = TRUE) {

  df_train_data <- as.data.frame(dataframe_train)

  # Remove unneeded columns
  x_train_orig <- df_train_data[, !(names(df_train_data) %in% exclude_columns), drop=FALSE]

  # Validation set
  if(create_validation){
    after_validation <- create_validation_set(x_train_orig, validation_proportion)

    x_train <- after_validation$x_tr
    x_val <- after_validation$x_val

  } else {
    x_train <- x_train_orig
    x_val <- NULL
  }

  optimized_recipe = NULL

  # Preprocessing with recipes
  if (preprocess) {
    # We are not inside a recipe workflow, so we prep and bake required preprocessing steps
    # One hot encoding + standard scaler
    rec <- recipes::recipe(~ ., data=x_train)

    rec <- recipes::step_novel(rec, recipes::all_nominal_predictors(), new_level = "unknown") |>  # New categorical levels (should not be used)
      recipes::step_normalize(recipes::all_numeric_predictors()) |>  # Standard normalization
      recipes::step_dummy(recipes::all_nominal_predictors(), one_hot = TRUE)  # One-hot encoding


    # Fit recipe to training set
    trained_recipe <- recipes::prep(rec, training = x_train)

    # Apply preprocessing to train and create matrix
    x_train_processed <- recipes::bake(trained_recipe, new_data = x_train)
    x_train <- as.matrix(x_train_processed)



    # Bake validation set (if exists)
    if(create_validation){
      x_val_processed <- recipes::bake(trained_recipe, new_data = x_val)
      x_val <- as.matrix(x_val_processed)
    }

    print("Train set: ")
    print(dim(x_train))
    if(create_validation){
      print("Validation set: ")
      print(dim(x_val))
    }

    # Optimize recipe by removing unnecessary data (butcher package)
    optimized_recipe <- butcher::butcher(trained_recipe)
  } else {
    x_train <- as.matrix(x_train)

    if (create_validation){
      x_val <- as.matrix(x_val)
    }
  }

  return (list("train_set" = x_train,
               "val_set" = x_val,
               "recipe" = optimized_recipe))

}




