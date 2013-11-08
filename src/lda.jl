
immutable Lda
    alpha
    data::Matrix
    kernel::Function
end


function project(lda::Lda, x::Vector)
    res = 0
    for i = 1:size(lda.data, 2)
        res += lda.alpha[i] * lda.kernel(lda.data[:, i], x)
    end
    return res
end


function process(lda::Lda, data::Matrix)
    n = size(data, 2)
    res = Array(Float64, n)
    for i = 1:n
        res[i] = project(lda, data[:, i])
    end
    return res
end


function lda(data::Matrix, labels::Vector, kfun::Function = linear, karg = nothing)

    # Number of training samples
    n = size(data, 2)

    # Defines Kernel function
    if karg == nothing
        kernel(x, y) = kfun(x, y)
    else
        kernel(x, y) = kfun(x, y, karg)
    end

    # Separates two classes
    class = { data[:, labels .== 1], data[:, labels .== -1] }


    # Compute Kj[m, n] = kernel(Xn, Xmj)
    K = {}
    for j = 1:2
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
    for i = 1:2
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
    for j = 1:2
        ksize = size(K[j], 2)
        dsize = size(class[j], 2)
        N += K[j] * (eye(ksize) - ones(ksize, ksize) / dsize) * K[j]'
    end

    alpha = inv(N + 0.1 *  eye(size(N, 1))) * (M[2] - M[1])

    return Lda(alpha, data, kernel)
end
