
# Create a tensor dataset given the data
create_tensor_dataset <- torch::dataset(

  name = "create_tensor_dataset",


  initialize = function(data) {
    self$data <- as.matrix(data)
  },


  .getitem = function(i) {
    data <- self$data[i, ]  # All columns of a row

    data_tensor <- torch::torch_tensor(data, dtype = torch::torch_float32())

    list(x = data_tensor)

  },


  .length = function(){
    nrow(self$data)
  }

)







# Create a tensor dataset given the data. It also returns the label of each row
create_tensor_dataset_with_label <- torch::dataset(
  name = "tensor_dataset_no_label",

  initialize = function(data, target) {
    self$data <- as.matrix(data)
    self$target <- as.vector(target)
  },


  .getitem = function(i) {
    data <- self$data[i, ]  # All columns of a row
    label <- self$target[i]

    data_tensor <- torch::torch_tensor(data, dtype = torch::torch_float32())
    label_tensor <- torch::torch_tensor(label, dtype = torch::torch_long())

    list(x = data_tensor, y = label_tensor)

  },


  .length = function(){
    nrow(self$data)
  }


)






