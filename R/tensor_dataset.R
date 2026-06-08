
#' Creation of a tensor dataset.
#'
#' @param data Original data samples
#' @param target Original label samples
#'
#' @return A 'torch::dataset' that contains the data.
#' @export
#'
#' @examples
#' X_fake <- matrix(runif(50*4), nrow=50, ncol=4)
#' y_fake <- sample(0:1, 50, replace=TRUE)
#'
#' my_dataset <- tensor_dataset(X_fake, y_fake)
#'
#' first_item <- my_dataset$.getitem(1)
#' print(first_item$x)
#' print(first_item$y)
tensor_dataset <- torch::dataset(
  name = "tensor_dataset",

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






