include("../demo_utilities.jl")
include("../../scratch_papers/cg_modified_redesigned.jl")
using BenchmarkTools


n = 512
A = Diagonal(rand(n).^6)
b = randn(n)
bNorm = norm(b)
function trial()
    cgm = ConjGradModified(A, b)
    SetStorageLimit(cgm, 256)
    # TurnOffReorthgonalize(cgm)
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



