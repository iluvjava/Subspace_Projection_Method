include("../demo_utilities.jl")
include("../../src/cg_modified.jl")
using BenchmarkTools, Profile

n = 1024
A = Diagonal(collect(LinRange(1e-3, 1, n)).^2)
b = randn(n)
bNorm = norm(b)
function trial()
    cgm = ConjGradModified(A, b)
    ChangeStorageLimit(cgm, 1024)
    TurnOffReorthgonalize(cgm)
    ResNorm = norm(cgm.r)
    iterationCount = 0
    while ResNorm > 1e-10
        ResNorm = cgm()/bNorm
        iterationCount += 1
        if ResNorm == Inf || ResNorm == NaN
            error("Resnorm is inf or nan.")
        end
    end
    println("iterationCount: $iterationCount")
end
@time trial()
display(@benchmark trial())


# function Trial2()
#     cgm = ConjGradModified(A, b)
#     SetStorageLimit(cgm, 1024)
#     # TurnOffReorthgonalize(cgm)
#     ResNorm = norm(cgm.r)
#     iterationCount = 0
#     while ResNorm > 1e-10
#         ResNorm = cgm()/bNorm
#         iterationCount += 1
#         if ResNorm == Inf || ResNorm == NaN
#             error("Resnorm is inf or nan.")
#         end
#     end
#     println("iterationCount: $iterationCount")

# end
# @time Trial2()

