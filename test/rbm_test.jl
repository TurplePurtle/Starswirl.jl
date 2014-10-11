using Starswirl

sigmoid(x::Real) = 1.0 ./ (1.0 + exp(-x))
sigmoid!{T<:Real}(x::Array{T}) = map!(sigmoid, x)

net = NeuralNet([6, 2])

X = [
  1 1 1 0 0 0
  1 0 1 0 0 0 
  1 1 1 0 0 0 
  0 0 1 1 1 0 
  0 0 1 1 0 0 
  0 0 1 1 1 0
  0 0 1 1 1 1.
]'

function init_rbm(net::NeuralNet)
  for i = 1:length(net.weights)
    weight = net.weights[i]
    map!(w -> 0.01 * randn(), weight)
    weight[end,:] = 0.0
    weight[:,end] = 0.0
  end
end

function condiv(net::NeuralNet,
                X::Matrix{Float64},
                learning_rate::Float64 = 0.1,
                max_epochs::Int = 1000)

  m = size(X,2)
  X = [X; ones(1,m)]
  alpha = learning_rate / m
  i = 1

  for n = 1:max_epochs
    # Update hidden units
    # Calculate activation energy
    v = X
    pos_hidden_prob = sigmoid!(net.weights[i] * v)
    pos_hidden_prob[end,:] = 1.0 # fix biases
    # Turn unit on with probablility a
    h = map!(x -> x > rand() ? 1.0 : 0.0, copy(pos_hidden_prob))

    pos_a = pos_hidden_prob * X'

    # Reconstruct visible units
    neg_visible_prob = sigmoid!(net.weights[i]' * h)
    neg_visible_prob[end,:] = 1.0 # fix biases

    # Update hidden units again
    neg_hidden_prob = sigmoid!(net.weights[i] * neg_visible_prob)

    neg_a = neg_hidden_prob * neg_visible_prob'

    # Update weights
    net.weights[i] += alpha * (pos_a - neg_a)
    net.weights[i][end,end] = 0.0

    error
  end
end

init_rbm(net)

condiv(net, X, 0.1)
println(net.weights)

x = net.weights[1]' * ([1,0,1]')'
sigmoid!(x)
println(x)

x = net.weights[1]' * ([0,1,1]')'
sigmoid!(x)
println(x)
