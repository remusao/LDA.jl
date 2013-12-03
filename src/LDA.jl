module LDA

using Vega
using Gadfly

include("kernel.jl")
include("lda.jl")
include("visualization.jl")

export lda, process, project
export linear, polynomial, laplacian, rbf
export print_2Ddecision

end # module
