
function print_2Ddecision(lda::Lda, data::Matrix, labels = nothing)

    #
    # Output result
    #
    n = size(data, 2)

    projection = process(lda, data, 2)

    y = reshape(projection[2, :], n)
    x = reshape(projection[1, :], n)

    if labels == nothing
        println(plot(x = x, y = y, kind = :scatter))
    else
        println(plot(x = x, y = y, group = labels, kind = :scatter))
    end
end

