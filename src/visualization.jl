
function print_2Ddecision(lda::Lda, data::Matrix)

    #
    # Output result
    #
    n = size(data, 2)

    color = process(lda, data)
    map!(color) do x
        if x < 0
            return 0
        else
            return 1
        end
    end

    y = reshape(data[2, :], n)
    x = reshape(data[1, :], n)

    println(plot(x = x, y = y, group = color, kind = :scatter))
end

