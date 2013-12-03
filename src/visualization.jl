
function print_2Ddecision(lda::Lda, data::Matrix, name::String, labels = nothing)

    #
    # Output result
    #
    n = size(data, 2)

    projection = process(lda, data, 2)

    x = reshape(projection[2, :], n)
    y = reshape(projection[1, :], n)

    if labels == nothing
        p = plot(x = x, y = y, Geom.point)
    else
        p = plot(x = x, y = y, color = labels, Geom.point)
    end
    draw(PNG("test.png", 10inch, 10inch), p)
end
