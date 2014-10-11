
using Starswirl
using Base.Test
using MNIST

println("Loading MNIST data...")
X_train, labels_train = traindata()
X_test, labels_test = testdata()

X_train ./= 255.0
X_test ./= 255.0
labels_train = int(labels_train)
labels_test = int(labels_test)

m_train = size(X_train, 2)
m_test = size(X_test, 2)

num_in = size(X_train, 1)
num_out = 10

# Turn labels into vectors
Y_train = zeros(10, m_train)
Y_test = zeros(10, m_test)

for i = 1:m_train
  Y_train[labels_train[i]+1, i] = 1.0
end
for i = 1:m_test
  Y_test[labels_test[i]+1, i] = 1.0
end


println("Preparing neural net...")
tic()
# net = NeuralNet([num_in, int(sqrt(num_in+num_out)), num_out]) # took 24.2 min
net = NeuralNet([num_in, num_out])  # took 3.5 min
allocate_nn(net, size(X_train,2))
init_nn(net)

println("Training neural net...")
println(train(net, X_train, Y_train;
  lambda = 0.0,
  alpha = 0.8,
  momentum = 0.0,
  max_iter = 70,
  log_iter = true,
  weight_max_norm = 10.0
))
toc()

println("Weights L1-norm:")
for i = 1:length(net.weights)
  println("$i: $(norm(net.weights[i][:], 1))")
end
println("Weights L2-norm:")
for i = 1:length(net.weights)
  println("$i: $(norm(net.weights[i][:], 2))")
end

println("Processing test dataset...")
tic()
result = forwardprop(net, X_test)
labels_result = zeros(Int, m_test)
for i = 1:m_test
  labels_result[i] = indmax(result[:,i]) - 1
end
toc()

println("\nResults:")

misclassifications = labels_test .!= labels_result
wrong = sum(misclassifications)
percent_wrong = wrong / m_test
misclassified_labels = zip(
  labels_test[misclassifications],
  labels_result[misclassifications])

println(unique(misclassified_labels))
println("$wrong wrong out of $m_test examples. ($(100*percent_wrong)%)")

@test percent_wrong < 0.25
