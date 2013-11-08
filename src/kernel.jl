
linear(x::Vector, y::Vector) = (x' * y)[1]
polynomial(x::Vector, y::Vector, d) = (x' * y + 1)[1] ^ d
laplacian(x::Vector, y::Vector, sigma) = exp(-norm(x - y) / sigma ^ 2)
rbf(x::Vector, y::Vector, sigma) = exp(-norm(x - y, 1) / sigma ^ 2)
