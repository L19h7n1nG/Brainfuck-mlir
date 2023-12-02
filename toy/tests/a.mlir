module {
  toy.func private @transpose_transpose(%arg0: tensor<*xf32>) -> tensor<*xf32> {
    %0 = toy.transpose(%arg0 : tensor<*xf32>) to tensor<*xf32>
    %1 = toy.transpose(%0 : tensor<*xf32>) to tensor<*xf32>
    toy.return %1 : tensor<*xf64>
  }
  toy.func @main() {
    %0 = toy.constant dense<[[1.000000e+00, 2.000000e+00, 3.000000e+00], [4.000000e+00, 5.000000e+00, 6.000000e+00]]> : tensor<2x3xf64>
    %1 = toy.reshape(%0 : tensor<2x3xf64>) to tensor<2x3xf64>
    %2 = toy.constant dense<[[1.000000e+00, 2.000000e+00], [3.000000e+00, 4.000000e+00], [5.000000e+00, 6.000000e+00]]> : tensor<3x2xf32>
    %3 = toy.reshape(%2 : tensor<3x2xf32>) to tensor<3x2xf32>
    %4 = toy.generic_call @transpose_transpose(%3) : (tensor<3x2xf32>) -> tensor<*xf32>
    %5 = "toy.mm"(%1, %4) : (tensor<2x3xf64>, tensor<*xf64>) -> tensor<*xf64>
    toy.print %5 : tensor<*xf64>
    toy.return
  }
}