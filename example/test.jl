using Vega
using LDA
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
print_2Ddecision(rbf_res, data, labels)

# Poly 2
rbf_res = lda(data, labels, 3, polynomial, 2)
println("Printing results")
print_2Ddecision(rbf_res, data, labels)

# Poly 3
rbf_res = lda(data, labels, 3, polynomial, 3)
println("Printing results")
print_2Ddecision(rbf_res, data, labels)

# Poly 4
rbf_res = lda(data, labels, 3, polynomial, 4)
println("Printing results")
print_2Ddecision(rbf_res, data, labels)

# RBF 1.0
rbf_res = lda(data, labels, 3, rbf, 1.0)
println("Printing results")
print_2Ddecision(rbf_res, data, labels)

# RBF 0.4
rbf_res = lda(data, labels, 3, rbf, 0.4)
println("Printing results")
print_2Ddecision(rbf_res, data, labels)

# RBF 0.1
rbf_res = lda(data, labels, 3, rbf, 0.1)
println("Printing results")
print_2Ddecision(rbf_res, data, labels)
