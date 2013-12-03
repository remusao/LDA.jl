
immutable Lda
    c::Integer
    V::Matrix
    data::Matrix
    kernel::Function
end


function project(lda::Lda, x::Vector, dim::Integer)
    n = size(lda.data, 2)
    Kt = Array(Float64, n)
    for i = 1:n
        Kt[i] = lda.kernel(lda.data[:, i], x)
    end
    return lda.V[:, 1:dim]' * Kt
end


function process(lda::Lda, data::Matrix, dim::Integer)
    n = size(data, 2)
    res = Array(Float64, dim, n)
    for i = 1:n
        res[:, i] = project(lda, data[:, i], dim)
    end
    return res
end


function lda(data::Matrix, labels::Vector, c::Integer = 2, kfun::Function = linear, karg = nothing)

    # Number of training samples
    n = size(data, 2)

    # Count number of element in each class
    nk = Array(Int, c)
    for k = 1:c
        nk[k] = nnz(labels .== k)
    end

    # Defines Kernel function
    if karg == nothing
        kernel(x, y) = kfun(x, y)
    else
        kernel(x, y) = kfun(x, y, karg)
    end

    # Compute gram matrix
    G = Array(Float64, n, n)
    for i = 1:n
        for j = 1:n
            G[i, j] = kernel(data[:, i], data[:, j])
        end
    end

    # Compute intraclass variance Sw
    Sw = Array(Float64, n, n)
    for k = 1:c
        Gk = G[:, labels .== k]
        Sw = Sw + Gk * (eye(Float64, nk[k]) - ones(Float64, nk[k], nk[k]) / nk[k]) * Gk'
    end

    # Compute interclass variance Sb
    indic = zeros(Float64, n, c)
    for i = 1:n
        indic[i, labels[i]] = 1 / nk[labels[i]]
    end
    M1 = G * (indic - ones(Float64, n, c) / n)
    Sb = M1 * M1'

    S, V = eig(inv(Sw + 0.1 * eye(n)) * Sb)

    return Lda(c, V, data, kernel)
end
