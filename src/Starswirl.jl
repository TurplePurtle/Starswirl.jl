module Starswirl

export NeuralNet, init_nn, allocate_nn, forwardprop, backprop, train


sigmoid(x::Real) = 1.0 ./ (1.0 + exp(-x))
sigmoid!{T<:Real}(x::Array{T}) = map!(sigmoid, x)

type NeuralNet
  layers::Vector{Int}
  weights::Vector{Matrix{Float64}}
  grad::Vector{Matrix{Float64}}

  # store arrays so they don't have to be re-allocated
  activ::Vector{Matrix{Float64}}
  del::Vector{Vector{Float64}}
end


function NeuralNet(layers::Vector{Int})
  if length(layers) < 2 || any(layers .< 1)
    error("Bad layer architecture")
  end

  n = length(layers)

  weights = Array(Matrix{Float64}, n-1)
  grad = Array(Matrix{Float64}, n-1)

  for i = 1:n-1
    weights[i] = zeros(layers[i+1] + 1, layers[i] + 1)
    grad[i] = zeros(layers[i+1] + 1, layers[i] + 1)
  end

  a = Array(Matrix{Float64}, n)
  d = Array(Vector{Float64}, n)
  # for i = 1:n
  #   a[i] = zeros(0,0)
  # end
  # d[1] = Float64[]  # d[1] is not used

  NeuralNet(layers, weights, grad, a, d)
end


function init_nn(net::NeuralNet)
  for i = 1:length(net.weights)
    weight = net.weights[i]
    k = 2.0 * sqrt(6.0 / sum(size(weight)))
    map!(w -> k * (rand() - 0.5), weight)
    weight[end,:] = 0.0
  end
end


function allocate_nn(net::NeuralNet, m::Int)
  a = net.activ
  d = net.del

  for i = 1:length(net.layers)
    a[i] = zeros(net.layers[i]+1, m)
    d[i] = zeros(net.layers[i]+1)
  end
end


function forwardprop{T<:Real}(net::NeuralNet,
                     X::Matrix{T})

  n = size(X, 1)
  m = size(X, 2)
  A = ones(n+1, m)
  A[1:n, :] = X

  for i = 1:length(net.weights)
    A = sigmoid!(net.weights[i] * A)
    A[end,:] = 1.0
  end
  sub(A, 1:size(A,1)-1, :)
end


function backprop{T<:Real}(net::NeuralNet,
                  X::Matrix{T},
                  Y::Matrix{T},
                  lambda::Float64 = 0.0)

  n = size(X, 1)
  m = size(X, 2)
  weights = net.weights

  # Forward propagation
  a = net.activ
  # if size(a[1]) != (n+1, m)
  #   a[1] = ones(n+1, m)
  # else
    a[1][end,:] = 1
  # end
  a[1][1:n, :] = X

  for i = 1:length(weights)
    # a[i+1] = weights[i] * a[i]
    A_mul_B!(a[i+1], weights[i], a[i])
    sigmoid!(a[i+1])
    a[i+1][end,:] = 1.0
  end

  Yp = sub(a[end], 1:size(a[end],1)-1, :)

  # Get cost
  cost = 0
  @simd for i = 1:length(Y)
    @inbounds cost += (Y[i] - 1) * log(1 - Yp[i]) - Y[i] * log(Yp[i])
  end
  cost /= m

  # L2 Regularization
  if lambda > 0.0
    reg_cost = reduce((sum,w) -> sum + reduce((s,w) -> s + w*w, 0.0, w)
                      , 0.0, weights)
    cost += reg_cost * lambda / (2*m)
  end

  # perform backprop to obtain gradient
  d = net.del
  grad = net.grad

  for i = 1:length(grad)
    fill!(grad[i], 0.0)
  end

  for i = 1:m
    @simd for j = 1:size(Y,1)
      @inbounds d[end][j] = Yp[j,i] - Y[j,i]
    end
    d[end][end] = 0.0
    # d[end] = [Yp[:,i] - Y[:,i]; 0.0]
    for j = length(d)-1:-1:2
      # sig = a[j][:,i]
      mul = weights[j]' * d[j+1]
      @simd for k = 1:length(mul)
        @inbounds d[j][k] = mul[k] * a[j][k,i] * (1.0 - a[j][k,i])
      end
      d[j][end,:] = 0.0
    end

    for L = length(d)-1:-1:1
      # grad[L] += d[L+1] * a[L][:,i]'
      # devectorized for performance
      for j = 1:size(grad[L],1)
        @simd for k = 1:size(grad[L],2)
          @inbounds grad[L][j,k] += a[L][k,i] * d[L+1][j]
        end
      end
    end
  end

  for i = 1:length(grad)
    grad[i] /= m
    if lambda > 0.0 # Regularize gradient
      grad_del = (lambda / m) * weights[i]
      grad_del[end,:] = 0.0
      grad_del[:,end] = 0.0
      grad[i] += grad_del
    end
  end

  cost
end


function train(net::NeuralNet,
               X::Matrix{Float64},
               Y::Matrix{Float64};
               lambda::Float64 = 0.0,
               alpha::Float64 = 1.0,
               momentum::Float64 = 0.0,
               max_iter::Int = 2000,
               rel_tol::Float64 = 1e-6,
               log_iter::Bool = false,
               weight_max_norm::Float64 = Inf)

  n::Int = length(net.weights)

  # keep track of the last step to use it for momentum
  last_step = Array(Matrix{Float64}, n)
  for j = 1:n
    last_step[j] = zeros(size(net.weights[j]))
  end

  cost::Float64 = 0.0
  last_cost::Float64 = 0.0

  num_iter::Int = max_iter

  for i = 1:max_iter
    last_cost = cost
    cost = backprop(net, X, Y, lambda)

    for j = 1:n
      if momentum > 0.0
        for k = 1:length(net.weights[j])
          step = -alpha * net.grad[j][k]
          net.weights[j][k] += step + momentum * last_step[j][k]
          last_step[j][k] = step
        end
      else
        @simd for k = 1:length(net.weights[j])
          @inbounds net.weights[j][k] += -alpha * net.grad[j][k]
        end
      end

      # weight normalization
      weight_norm = norm(net.weights[j][:])
      if (weight_norm > weight_max_norm)
        net.weights .*= weight_max_norm / weight_norm
      end
    end

    if log_iter
      println("Iteration $(i)\tcost: $(cost)")
    end

    if abs(cost - last_cost) < rel_tol
      num_iter = i
      break
    end
  end

  return (num_iter, cost, cost - last_cost)

end


end # module
