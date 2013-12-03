using Vega
using LDA
using DataFrames
using RDatasets

# Try with Iris
iris = data("datasets", "iris")

# Create data matrix
data = matrix(iris[:, 2:5])'

# Create labels vector
labels = Array(Int, size(data, 2))
for i = 1:size(data, 2)
    x = iris[:, 6][i]
    if  x == "setosa"
        labels[i] = 1
    elseif x == "virginica"
        labels[i] = 2
    else
        labels[i] = 3
    end
end

# lineaire
rbf_res = lda(data, labels, 3, linear)
println("Printing results")
print_2Ddecision(rbf_res, data, "linear.png", labels)

# Poly 2
rbf_res = lda(data, labels, 3, polynomial, 2)
println("Printing results")
print_2Ddecision(rbf_res, data, "poly2.png", labels)

# Poly 3
rbf_res = lda(data, labels, 3, polynomial, 3)
println("Printing results")
print_2Ddecision(rbf_res, data, "poly3.png", labels)

# Poly 4
rbf_res = lda(data, labels, 3, polynomial, 4)
println("Printing results")
print_2Ddecision(rbf_res, data, "poly4.png", labels)

# RBF 1.0
rbf_res = lda(data, labels, 3, rbf, 1.0)
println("Printing results")
print_2Ddecision(rbf_res, data, "rbf1_0.png", labels)

# RBF 0.4
rbf_res = lda(data, labels, 3, rbf, 0.4)
println("Printing results")
print_2Ddecision(rbf_res, data, "rbf0_4.png", labels)

# RBF 0.1
t0 = time()
rbf_res = lda(data, labels, 3, rbf, 0.1)
t1 = time()
println("Printing results")
print_2Ddecision(rbf_res, data, "rbf0_1.png", labels)
println(t1 - t0)
