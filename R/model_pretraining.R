

scarf_pretraining = function(path_train_data, path_test_data) {

  # Load data
  preprocessed_datasets = read_parquet_data(path_train_data, path_test_data)

  x_train <- preprocessed_datasets$train_set
  x_val <- preprocessed_datasets$val_set
  x_test <- preprocessed_datasets$test_set

  y_train <- preprocessed_datasets$train_label
  y_val <- preprocessed_datasets$val_label
  y_test <- preprocessed_datasets$test_label

  print(dim(x_train))
  print(dim(x_val))
  print(dim(x_test))
  print(dim(y_train))
  print(dim(y_val))
  print(dim(y_test))

  # Create tensor datasets
  train_ds <- create_tensor_dataset(x_train, y_train)
  val_ds <- create_tensor_dataset(x_val, y_val)
  test_ds <- create_tensor_dataset(x_test, y_test)

  print(train_ds[1]$x)

  # Create dataloader
  train_dl <- dataloader(train_ds,
                         batch_size = 32,
                         shuffle = TRUE)

  val_dl <- dataloader(val_ds,
                       batch_size = 32,
                       shuffle=FALSE)

  test_dl <- dataloader(test_ds,
                        batch_size = 32,
                        shuffle = FALSE)



}

# TODO: corregir read_data. Falla algo
preprocessed_datasets = read_parquet_data("inst/extdata/UNSW_NB15_training-set.parquet", "inst/extdata/UNSW_NB15_testing-set.parquet")

x_train <- preprocessed_datasets$train_set
x_val <- preprocessed_datasets$val_set
x_test <- preprocessed_datasets$test_set

print(x_train[1])


y_train <- preprocessed_datasets$train_label
y_val <- preprocessed_datasets$val_label
y_test <- preprocessed_datasets$test_label

print(dim(x_train))
print(dim(x_val))
print(dim(x_test))
print(dim(y_train))
print(dim(y_val))
print(dim(y_test))

# Create tensor datasets
train_ds <- create_tensor_dataset(x_train, y_train)
val_ds <- create_tensor_dataset(x_val, y_val)
test_ds <- create_tensor_dataset(x_test, y_test)

print(train_ds[1]$x)

# Create dataloader
train_dl <- dataloader(train_ds,
                       batch_size = 32,
                       shuffle = TRUE)

val_dl <- dataloader(val_ds,
                     batch_size = 32,
                     shuffle=FALSE)

test_dl <- dataloader(test_ds,
                      batch_size = 32,
                      shuffle = FALSE)



scarf_pretraining("inst/extdata/UNSW_NB15_training-set.parquet", "inst/extdata/UNSW_NB15_testing-set.parquet")











