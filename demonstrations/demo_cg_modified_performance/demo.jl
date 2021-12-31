include("../demo_utilities.jl")
include("cg_modified_for_speed.jl")
using BenchmarkTools


n = 128^2

function trial(store = n)
    A = Diagonal(rand(n))
    b = randn(n)
    bNorm = norm(b)
    cgm = ConjGrad(A, b)
    # ChangeStorageLimit(cgm, store)
    ResNorm = norm(cgm.r)
    iterationCount = 0
    res = @benchmark while ResNorm > 1e-2
        ResNorm = cgm()/bNorm
        iterationCount += 1
        if ResNorm == Inf || ResNorm == NaN
            error("Resnorm is inf or nan.")
        end
    end
    return res
end

@benchmark trial()



