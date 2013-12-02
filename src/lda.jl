
immutable Lda
    c::Integer
    Astar
    data::Matrix
    kernel::Function
end


function project(lda::Lda, x::Vector, dim::Integer)
    n = size(lda.data, 2)
    Kt = Array(Float64, n)
    for i = 1:n
        Kt[i] = lda.kernel(lda.data[:, i], x)
    end
    return lda.Astar[:, 1:dim]' * Kt
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

    # Defines Kernel function
    if karg == nothing
        kernel(x, y) = kfun(x, y)
    else
        kernel(x, y) = kfun(x, y, karg)
    end

    # Separates c classes
    class = {}
    for i = 1:c
        push!(class, data[:, labels .== i])
    end


    # Compute Kj[m, n] = kernel(Xn, Xmj)
    K = {}
    for j = 1:c
        Kj = Array(Float64, n, size(class[j], 2))
        for i = 1:n
            for m = 1:size(class[j], 2)
                Kj[i, m] = kernel(data[:, i], class[j][:, m])
            end
        end
        push!(K, Kj)
    end


    # Compute Mi
    # Mij = 1/li * sum(k: 1 -> li) kernel(Xj, Xki)
    M = {}
    for i = 1:c
        Mi = Array(Float64, n)
        for j = 1:n
            for k = 1:size(class[i], 2)
                Mi[j] += kernel(data[:, j], class[i][:, k])
            end
        end
        Mi /= size(class[i], 2)
        push!(M, Mi)
    end


    # Compute N
    # sum(j = 1,2) Kj (I - 1lj) Kj'
    N = 0
    for j = 1:c
        ksize = size(K[j], 2)
        dsize = size(class[j], 2)
        N += K[j] * (eye(ksize) - ones(ksize, ksize) / dsize) * K[j]'
    end

    # Compute M*
    Mstar = Array(Float64, n)
    for j = 1:n
        for k = 1:n
            Mstar[k] = Mstar[k] + kernel(data[:, j], data[:, k])
        end
        Mstar[j] /= n
    end

    # M
    Mtot = 0
    for j = 1:c
        Mtot = Mtot + size(class[j], 2) * (M[j] - Mstar) * (M[j] - Mstar)'
    end

    println(Mstar)
    println(Mtot)
    Astar = eigvecs(inv(N + 0.1 *  eye(size(N, 1))) * Mtot)

    return Lda(c, Astar, data, kernel)
end
