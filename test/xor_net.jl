using Starswirl
using Base.Test

data = Float64[
  0 1 1 0  # x1
  0 1 0 1  # x2
  0 0 1 1  # y1
]

X = data[1:end-1,:]
Y = data[end,:]

net = NeuralNet([2,2,1])

rand_init = false

if rand_init
  init_nn(net)
else
  # Initialize to constant for testing purposes
  net.weights[1][:] = [
    0.17108293642467984 -0.22450805971884732 0.0
    0.5093710567943869   0.327144463391315   0.0
    0.7699662010264641  -1.0917971463670086  0.0
  ]'

  net.weights[2][:] = [
    -0.8365736061618315  0.0
    -0.2668042503453018  0.0
    -0.26001541959786156 0.0
  ]'
end

allocate_nn(net, size(data,2))

println("Training Neural Net...")
println("(iterations, cost, delta_cost):")
tic()
println(train(net, X, Y;
  lambda = 0.0,
  alpha = 1.0,
  momentum = 0.8,
  max_iter = 2000
))
toc()
println("\nWeights:")
for i = 1:length(net.weights)
  println(net.weights[i]')
end
println("Weights norm:")
for i = 1:length(net.weights)
  println(norm(net.weights[i][:]))
end
println("\nResult: ([label result])")
result = int(forwardprop(net, X)[1,:]);
println([int(Y); result]')
err = norm(Y - result)
println("\nError = $err\n")
if !rand_init
  @test err == 0
end
