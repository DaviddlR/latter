
#' Check if GPU is available
#'
#' @returns Command line message indicating whether GPU is available or not.
#' @export
#' @importFrom torch cuda_is_available
#'
#' @examples
#' check_if_gpu_available()
check_if_gpu_available <- function() {

  if(torch::cuda_is_available()){
    print("Graphic card found :D")
  } else {
    print("Graphic card not found D:")
  }

}





